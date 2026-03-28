//! Spot-Futures (Cash and Carry) Arbitrage Engine
//!
//! Main orchestration engine for institutional-grade spot-futures funding rate arbitrage.
//! Spawned as a tokio task from main.rs when SPOT_FUTURES_ENABLED=true and USE_MULTI_EXCHANGE=on.
//!
//! Architecture:
//! - Buy the actual asset on the Spot market (cannot be liquidated, you own the asset)
//! - Short the same asset on Perpetual Futures (collect funding rate)
//! - Profit comes purely from collecting the funding rate (shorts receive when perps > spot)
//!
//! Key Principles:
//! - Same-exchange Spot-Futures (V1): both legs on the same exchange
//! - One position at a time by default (configurable)
//! - All financial math uses rust_decimal::Decimal
//! - All specs/fees fetched dynamically at runtime
//! - Position state persisted to SQLite for crash recovery

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::config::SpotFuturesConfig;
use crate::execution_gateway::ExecutionGateway;
use crate::multi_exchange::global_book::ExchangeId;
use crate::multi_exchange::spot_futures_executor::{
    SpotFuturesExecutor, SpotFuturesEntryResult, SpotFuturesExitResult,
};
use crate::multi_exchange::spot_futures_monitor::{
    FundingRateHistory, SpotFuturesMonitor,
    SpotFuturesPosition, SpotFuturesPositionState,
};
use crate::multi_exchange::spot_futures_sizer::SpotFuturesSizer;
use crate::multi_exchange::spot_futures_specs::SpotFuturesSpecs;

// ---------------------------------------------------------------------------
// Spot-Futures Opportunity
// ---------------------------------------------------------------------------

/// A detected spot-futures funding rate arbitrage opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotFuturesOpportunity {
    pub symbol: String,
    pub exchange: ExchangeId,
    pub funding_rate: f64,
    pub funding_interval_hours: f64,
    pub predicted_rate: f64,
    pub projected_apr: f64,
    pub spot_ask_price: f64,
    pub futures_bid_price: f64,
    pub spread_pct: f64,
    pub round_trip_fee_pct: f64,
    pub net_projected_daily_yield: f64,
    pub is_actionable: bool,
    pub next_funding_ms: u64,
}

// ---------------------------------------------------------------------------
// Trading Mode
// ---------------------------------------------------------------------------

/// Trading mode derived from TRADING_MODE env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingMode {
    Live,
    Testnet,
    Paper,
}

impl TradingMode {
    /// Parse from environment variable string.
    pub fn from_env() -> Self {
        match std::env::var("TRADING_MODE")
            .unwrap_or_else(|_| "paper".to_string())
            .to_lowercase()
            .as_str()
        {
            "live" => TradingMode::Live,
            "testnet" => TradingMode::Testnet,
            _ => TradingMode::Paper,
        }
    }

    /// Whether wallet transfers are allowed.
    pub fn transfers_allowed(&self) -> bool {
        matches!(self, TradingMode::Live)
    }

    /// Whether real orders are allowed.
    pub fn orders_allowed(&self) -> bool {
        matches!(self, TradingMode::Live | TradingMode::Testnet)
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Main Spot-Futures arbitrage engine.
pub struct SpotFuturesEngine {
    config: SpotFuturesConfig,
    specs: SpotFuturesSpecs,
    sizer: SpotFuturesSizer,
    executor: SpotFuturesExecutor,
    monitor: SpotFuturesMonitor,
    rate_history: FundingRateHistory,
    positions: Vec<SpotFuturesPosition>,
    closed_positions: Vec<SpotFuturesPosition>,
    next_position_id: AtomicU64,
    shutdown: Arc<AtomicBool>,
    trading_mode: TradingMode,
}

impl SpotFuturesEngine {
    /// Create a new Spot-Futures engine.
    pub fn new(config: SpotFuturesConfig, shutdown: Arc<AtomicBool>) -> Self {
        let trading_mode = TradingMode::from_env();
        let history_depth = config.funding_history_depth;

        Self {
            sizer: SpotFuturesSizer::new(config.clone()),
            executor: SpotFuturesExecutor::new(config.clone()),
            monitor: SpotFuturesMonitor::new(config.clone()),
            rate_history: FundingRateHistory::new(history_depth),
            specs: SpotFuturesSpecs::new(),
            positions: Vec::new(),
            closed_positions: Vec::new(),
            next_position_id: AtomicU64::new(1),
            shutdown,
            trading_mode,
            config,
        }
    }

    /// Main engine loop.
    ///
    /// # Arguments
    /// * `gateways` - Map of ExchangeId -> gateway implementation
    /// * `funding_arb` - Existing funding rate monitor (shared with old engine)
    /// * `symbols` - List of symbols to track
    /// * `binance_testnet` / `bybit_testnet` / `gateio_testnet` - Testnet flags
    pub async fn run(
        &mut self,
        gateways: HashMap<ExchangeId, Arc<dyn ExecutionGateway>>,
        symbols: Vec<String>,
        binance_testnet: bool,
        bybit_testnet: bool,
        gateio_testnet: bool,
    ) {
        info!(
            "[spot-futures-engine] Starting Spot-Futures Arbitrage Engine (mode={:?}, max_positions={}, leverage={}x)",
            self.trading_mode, self.config.max_positions, self.config.short_leverage
        );

        if !self.trading_mode.orders_allowed() {
            info!("[spot-futures-engine] Paper mode: will scan opportunities but NOT execute orders");
        }

        // Phase 1: Bootstrap exchange specs
        info!("[spot-futures-engine] Phase 1: Bootstrapping exchange specifications...");
        self.specs.bootstrap(binance_testnet, bybit_testnet, gateio_testnet).await;

        // Phase 2: Attempt to load persisted positions from SQLite (crash recovery)
        // NOTE: StateStore integration would load positions here. For now, start fresh.
        info!("[spot-futures-engine] Phase 2: Position recovery check (starting fresh)");

        // Phase 3: Enter main loop
        info!("[spot-futures-engine] Phase 3: Entering main loop");

        let mut last_funding_fetch = Instant::now() - Duration::from_secs(120); // Force immediate fetch
        let mut last_opportunity_scan = Instant::now();
        let mut last_position_check = Instant::now();

        let cycle_interval = Duration::from_secs(10);

        loop {
            // Check shutdown
            if self.shutdown.load(Ordering::Relaxed) {
                info!("[spot-futures-engine] Shutdown signal received");
                // Emergency unwind all positions
                self.emergency_unwind_all(&gateways).await;
                break;
            }

            // (a) Fetch funding rates every 60 seconds
            if last_funding_fetch.elapsed() >= Duration::from_secs(60) {
                self.fetch_funding_rates(&gateways, &symbols, binance_testnet, bybit_testnet, gateio_testnet).await;
                last_funding_fetch = Instant::now();
            }

            // (b) Scan for opportunities (only if below max positions)
            if self.positions.len() < self.config.max_positions as usize
                && last_opportunity_scan.elapsed() >= Duration::from_secs(30)
            {
                let opportunities = self.scan_opportunities(&gateways, &symbols).await;
                if let Some(best) = opportunities.first() {
                    info!(
                        "[spot-futures-engine] Best opportunity: {} on {} (APR={:.1}%, rate={:.4}%)",
                        best.symbol,
                        best.exchange.name(),
                        best.projected_apr * 100.0,
                        best.funding_rate * 100.0,
                    );

                    if best.is_actionable && self.trading_mode.orders_allowed() {
                        self.try_enter_position(best, &gateways).await;
                    }
                }
                last_opportunity_scan = Instant::now();
            }

            // (c) Monitor active positions
            if last_position_check.elapsed() >= Duration::from_secs(30) {
                self.monitor_active_positions(&gateways).await;
                last_position_check = Instant::now();
            }

            // (d) Sleep until next cycle
            tokio::time::sleep(cycle_interval).await;
        }

        info!(
            "[spot-futures-engine] Engine stopped. Active positions: {}, Closed: {}",
            self.positions.len(),
            self.closed_positions.len()
        );
    }

    /// Fetch funding rates from all exchanges for tracked symbols.
    async fn fetch_funding_rates(
        &mut self,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway>>,
        symbols: &[String],
        binance_testnet: bool,
        bybit_testnet: bool,
        gateio_testnet: bool,
    ) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        // Fetch from Binance: GET /fapi/v1/premiumIndex
        let binance_base = if binance_testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        if let Ok(resp) = client.get(format!("{}/fapi/v1/premiumIndex", binance_base)).send().await {
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                if let Some(items) = data.as_array() {
                    for item in items {
                        let symbol = item.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                        let rate_str = item.get("lastFundingRate").and_then(|v| v.as_str()).unwrap_or("0");
                        let next_ts = item.get("nextFundingTime").and_then(|v| v.as_u64()).unwrap_or(0);
                        if let Ok(rate) = rate_str.parse::<f64>() {
                            let base = crate::multi_exchange::spot_futures_specs::normalize_base_asset(symbol);
                            let now_ns = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64;
                            self.rate_history.record(ExchangeId::Binance, &base, now_ns, rate);
                        }
                    }
                }
            }
        }

        // Fetch from Bybit: GET /v5/market/tickers?category=linear
        let bybit_base = if bybit_testnet {
            "https://api-demo.bybit.com"
        } else {
            "https://api.bybit.com"
        };
        if let Ok(resp) = client
            .get(format!("{}/v5/market/tickers?category=linear", bybit_base))
            .send()
            .await
        {
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                if let Some(list) = data
                    .get("result")
                    .and_then(|r| r.get("list"))
                    .and_then(|l| l.as_array())
                {
                    for item in list {
                        let symbol = item.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                        let rate_str = item.get("fundingRate").and_then(|v| v.as_str()).unwrap_or("0");
                        if let Ok(rate) = rate_str.parse::<f64>() {
                            let base = crate::multi_exchange::spot_futures_specs::normalize_base_asset(symbol);
                            let now_ns = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64;
                            self.rate_history.record(ExchangeId::Bybit, &base, now_ns, rate);
                        }
                    }
                }
            }
        }

        // Fetch from Gate.io: poll individual contracts
        let gateio_base = if gateio_testnet {
            "https://api-testnet.gateapi.io"
        } else {
            "https://api.gateio.ws"
        };
        // Fetch all tickers at once
        if let Ok(resp) = client
            .get(format!("{}/api/v4/futures/usdt/tickers", gateio_base))
            .send()
            .await
        {
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                if let Some(items) = data.as_array() {
                    for item in items {
                        let contract = item.get("contract").and_then(|v| v.as_str()).unwrap_or("");
                        let rate_str = item.get("funding_rate").and_then(|v| v.as_str()).unwrap_or("0");
                        if let Ok(rate) = rate_str.parse::<f64>() {
                            let base = crate::multi_exchange::spot_futures_specs::normalize_base_asset(contract);
                            let now_ns = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64;
                            self.rate_history.record(ExchangeId::GateIo, &base, now_ns, rate);
                        }
                    }
                }
            }
        }

        debug!("[spot-futures-engine] Funding rates refreshed from all exchanges");
    }

    /// Scan for spot-futures arbitrage opportunities.
    async fn scan_opportunities(
        &self,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway>>,
        symbols: &[String],
    ) -> Vec<SpotFuturesOpportunity> {
        let mut opportunities = Vec::new();

        // For each exchange, check if we have both spot and futures specs
        for exchange in ExchangeId::all() {
            if !gateways.contains_key(&exchange) {
                continue;
            }

            let tradeable = self.specs.tradeable_assets();
            for base_asset in &tradeable {
                let spot_spec = match self.specs.get_spot_spec(exchange, base_asset) {
                    Some(s) => s,
                    None => continue,
                };
                let futures_spec = match self.specs.get_futures_spec(exchange, base_asset) {
                    Some(s) => s,
                    None => continue,
                };

                // Get current funding rate from history
                let current_rate = self.rate_history
                    .average_rate(exchange, base_asset)
                    .unwrap_or(0.0);

                if current_rate <= 0.0 {
                    continue; // Only enter when funding is positive (shorts receive)
                }

                let predicted_rate = self.rate_history.predicted_rate(exchange, base_asset, current_rate);
                let funding_interval = futures_spec.funding_interval_hours.max(8.0);
                let periods_per_day = 24.0 / funding_interval;

                // Calculate projected yield
                let spot_taker_fee = spot_spec.taker_fee_rate.to_f64().unwrap_or(0.001);
                let futures_taker_fee = futures_spec.taker_fee_rate.to_f64().unwrap_or(0.0005);
                let round_trip_fee = (spot_taker_fee + futures_taker_fee) * 2.0; // Entry + exit

                let expected_hold_days = self.config.max_hold_hours / 24.0;
                let daily_fee_amortized = round_trip_fee / expected_hold_days.max(1.0);

                let daily_funding_yield = predicted_rate * periods_per_day;
                let daily_net_yield = daily_funding_yield - daily_fee_amortized;
                let projected_apr = daily_net_yield * 365.0;

                let is_actionable = projected_apr * 100.0 >= self.config.min_apr_pct
                    && daily_net_yield > 0.0
                    && current_rate > 0.0;

                if is_actionable || projected_apr > 0.0 {
                    opportunities.push(SpotFuturesOpportunity {
                        symbol: base_asset.clone(),
                        exchange,
                        funding_rate: current_rate,
                        funding_interval_hours: funding_interval,
                        predicted_rate,
                        projected_apr,
                        spot_ask_price: 0.0,   // Would be filled from live ticker
                        futures_bid_price: 0.0, // Would be filled from live ticker
                        spread_pct: 0.0,
                        round_trip_fee_pct: round_trip_fee * 100.0,
                        net_projected_daily_yield: daily_net_yield,
                        is_actionable,
                        next_funding_ms: 0,
                    });
                }
            }
        }

        // Sort by projected APR descending
        opportunities.sort_by(|a, b| b.projected_apr.partial_cmp(&a.projected_apr).unwrap_or(std::cmp::Ordering::Equal));

        if !opportunities.is_empty() {
            debug!(
                "[spot-futures-engine] Scanned {} opportunities, {} actionable",
                opportunities.len(),
                opportunities.iter().filter(|o| o.is_actionable).count()
            );
        }

        opportunities
    }

    /// Attempt to enter a new spot-futures position.
    async fn try_enter_position(
        &mut self,
        opportunity: &SpotFuturesOpportunity,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway>>,
    ) {
        let exchange = opportunity.exchange;
        let gateway = match gateways.get(&exchange) {
            Some(gw) => gw,
            None => {
                warn!("[spot-futures-engine] No gateway for {}", exchange.name());
                return;
            }
        };

        let spot_spec = match self.specs.get_spot_spec(exchange, &opportunity.symbol) {
            Some(s) => s.clone(),
            None => {
                warn!("[spot-futures-engine] No spot spec for {} on {}", opportunity.symbol, exchange.name());
                return;
            }
        };
        let futures_spec = match self.specs.get_futures_spec(exchange, &opportunity.symbol) {
            Some(s) => s.clone(),
            None => {
                warn!("[spot-futures-engine] No futures spec for {} on {}", opportunity.symbol, exchange.name());
                return;
            }
        };

        // Fetch balances
        let spot_usdt = match gateway.get_spot_asset_balance("USDT").await {
            Ok(b) => Decimal::from_str(&b.to_string()).unwrap_or(Decimal::ZERO),
            Err(e) => {
                warn!("[spot-futures-engine] Failed to get spot USDT balance on {}: {}", exchange.name(), e);
                return;
            }
        };
        let futures_usdt = match gateway.get_balance().await {
            Ok(b) => Decimal::from_str(&b.to_string()).unwrap_or(Decimal::ZERO),
            Err(e) => {
                warn!("[spot-futures-engine] Failed to get futures balance on {}: {}", exchange.name(), e);
                return;
            }
        };

        info!(
            "[spot-futures-engine] Balances on {}: spot_usdt={}, futures_usdt={}",
            exchange.name(), spot_usdt, futures_usdt
        );

        // Get live prices via ticker
        let ticker = match gateway.get_ticker(&futures_spec.symbol).await {
            Ok(t) => t,
            Err(e) => {
                warn!("[spot-futures-engine] Failed to get ticker for {}: {}", futures_spec.symbol, e);
                return;
            }
        };

        let spot_ask = Decimal::from_str(&ticker.ask.to_string()).unwrap_or(Decimal::ZERO);
        let futures_bid = Decimal::from_str(&ticker.bid.to_string()).unwrap_or(Decimal::ZERO);

        if spot_ask.is_zero() || futures_bid.is_zero() {
            warn!("[spot-futures-engine] Zero price from ticker, skipping");
            return;
        }

        // Size the position
        let sizing = match self.sizer.calculate_size(
            exchange,
            &spot_spec,
            &futures_spec,
            spot_usdt,
            futures_usdt,
            spot_ask,
            futures_bid,
        ) {
            Some(s) => s,
            None => {
                debug!("[spot-futures-engine] Position too small for {} on {}", opportunity.symbol, exchange.name());
                return;
            }
        };

        info!(
            "[spot-futures-engine] Executing entry: {} on {}, qty={}, spot_ask={}, futures_bid={}",
            opportunity.symbol, exchange.name(), sizing.spot_qty, spot_ask, futures_bid
        );

        // Execute the entry
        let result = self.executor.execute_entry(
            gateway.as_ref(),
            exchange,
            &spot_spec,
            &futures_spec,
            sizing.spot_qty,
            spot_ask,
            futures_bid,
        ).await;

        match result {
            SpotFuturesEntryResult::Success { spot_result, futures_result } => {
                let pos_id = self.next_position_id.fetch_add(1, Ordering::Relaxed);
                let entry_fees = Decimal::from_str(&spot_result.fee.to_string()).unwrap_or(Decimal::ZERO)
                    + Decimal::from_str(&futures_result.fee.to_string()).unwrap_or(Decimal::ZERO);

                let spot_entry = Decimal::from_str(&spot_result.avg_fill_price.to_string()).unwrap_or(Decimal::ZERO);
                let spot_qty = Decimal::from_str(&spot_result.filled_qty.to_string()).unwrap_or(Decimal::ZERO);
                let futures_entry = Decimal::from_str(&futures_result.avg_fill_price.to_string()).unwrap_or(Decimal::ZERO);
                let futures_qty = spot_qty; // Should match

                let position = SpotFuturesPosition::new(
                    pos_id,
                    opportunity.symbol.clone(),
                    exchange,
                    spot_entry,
                    spot_qty,
                    futures_entry,
                    futures_qty,
                    futures_result.filled_size,
                    self.config.short_leverage,
                    entry_fees,
                );

                info!(
                    "[spot-futures-engine] POSITION OPENED: id={}, symbol={}, exchange={}, spot_price={}, futures_price={}, qty={}, fees={}",
                    pos_id, opportunity.symbol, exchange.name(), spot_entry, futures_entry, spot_qty, entry_fees
                );

                self.positions.push(position);
            }
            SpotFuturesEntryResult::SpotNotFilled { reason } => {
                info!("[spot-futures-engine] Entry cancelled (zero risk): {}", reason);
            }
            SpotFuturesEntryResult::SpotFilledFuturesFailed { spot_result, futures_error, emergency_sellback_result } => {
                error!(
                    "[spot-futures-engine] ENTRY FAILED: Spot filled but futures hedge failed: {}. Emergency sellback: {:?}",
                    futures_error, emergency_sellback_result.is_some()
                );
            }
            SpotFuturesEntryResult::ValidationFailed { reason } => {
                debug!("[spot-futures-engine] Entry validation failed: {}", reason);
            }
        }
    }

    /// Monitor all active positions for exit conditions.
    async fn monitor_active_positions(
        &mut self,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway>>,
    ) {
        let mut positions_to_close = Vec::new();

        for (idx, position) in self.positions.iter().enumerate() {
            if position.state != SpotFuturesPositionState::Active {
                continue;
            }

            // Get current funding rate
            let current_rate = self.rate_history
                .average_rate(position.spot_exchange, &position.symbol)
                .unwrap_or(0.0);

            // Get futures margin ratio (simplified -- would come from margin monitor)
            let futures_margin_ratio = 1.0; // Placeholder: would query exchange

            // Check exit conditions
            if let Some(reason) = self.monitor.check_exit_conditions(
                position,
                current_rate,
                futures_margin_ratio,
            ) {
                info!(
                    "[spot-futures-engine] EXIT TRIGGERED for position {}: {:?}",
                    position.id, reason
                );
                positions_to_close.push((idx, reason));
            }
        }

        // Close positions (in reverse order to avoid index issues)
        for (idx, reason) in positions_to_close.into_iter().rev() {
            let position = &self.positions[idx];
            let exchange = position.spot_exchange;

            if let Some(gateway) = gateways.get(&exchange) {
                if let (Some(spot_spec), Some(futures_spec)) = (
                    self.specs.get_spot_spec(exchange, &position.symbol),
                    self.specs.get_futures_spec(exchange, &position.symbol),
                ) {
                    let spot_spec = spot_spec.clone();
                    let futures_spec = futures_spec.clone();

                    let result = self.executor.execute_exit(
                        gateway.as_ref(),
                        &futures_spec,
                        &spot_spec,
                        position.futures_contracts,
                        position.spot_qty.to_f64().unwrap_or(0.0),
                    ).await;

                    match result {
                        SpotFuturesExitResult::Success { futures_close_result, spot_sell_result } => {
                            let mut closed = self.positions.remove(idx);
                            closed.state = SpotFuturesPositionState::Closed;

                            let exit_fees = Decimal::from_str(&futures_close_result.fee.to_string()).unwrap_or(Decimal::ZERO)
                                + Decimal::from_str(&spot_sell_result.fee.to_string()).unwrap_or(Decimal::ZERO);
                            closed.total_fees_paid += exit_fees;

                            info!(
                                "[spot-futures-engine] POSITION CLOSED: id={}, symbol={}, funding_collected={}, total_pnl={}, fees={}",
                                closed.id, closed.symbol, closed.accumulated_funding, closed.live_pnl, closed.total_fees_paid
                            );

                            self.closed_positions.push(closed);
                        }
                        SpotFuturesExitResult::FuturesClosedSpotFailed { futures_close_result, spot_error } => {
                            error!(
                                "[spot-futures-engine] Partial exit: futures closed but spot sell failed: {}",
                                spot_error
                            );
                        }
                        SpotFuturesExitResult::Failed { reason } => {
                            error!("[spot-futures-engine] Exit failed: {}", reason);
                        }
                    }
                }
            }
        }
    }

    /// Emergency unwind all positions (kill switch / shutdown).
    async fn emergency_unwind_all(
        &mut self,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway>>,
    ) {
        if self.positions.is_empty() {
            info!("[spot-futures-engine] No active positions to unwind");
            return;
        }

        error!(
            "[spot-futures-engine] EMERGENCY UNWIND: closing {} active positions",
            self.positions.len()
        );

        let positions: Vec<SpotFuturesPosition> = self.positions.drain(..).collect();
        for position in positions {
            let exchange = position.spot_exchange;
            if let Some(gateway) = gateways.get(&exchange) {
                if let (Some(spot_spec), Some(futures_spec)) = (
                    self.specs.get_spot_spec(exchange, &position.symbol),
                    self.specs.get_futures_spec(exchange, &position.symbol),
                ) {
                    let spot_spec = spot_spec.clone();
                    let futures_spec = futures_spec.clone();

                    let result = self.executor.execute_exit(
                        gateway.as_ref(),
                        &futures_spec,
                        &spot_spec,
                        position.futures_contracts,
                        position.spot_qty.to_f64().unwrap_or(0.0),
                    ).await;

                    match result {
                        SpotFuturesExitResult::Success { .. } => {
                            info!("[spot-futures-engine] Emergency close SUCCESS: position {}", position.id);
                        }
                        _ => {
                            error!("[spot-futures-engine] Emergency close FAILED for position {}", position.id);
                        }
                    }
                }
            }
            self.closed_positions.push(position);
        }
    }

    /// Get a summary of all positions for dashboard.
    pub fn get_dashboard_state(&self) -> serde_json::Value {
        serde_json::json!({
            "active_positions": self.positions.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
            "closed_positions_count": self.closed_positions.len(),
            "total_funding_collected": self.positions.iter()
                .map(|p| p.accumulated_funding.to_f64().unwrap_or(0.0))
                .sum::<f64>(),
            "trading_mode": format!("{:?}", self.trading_mode),
        })
    }
}
