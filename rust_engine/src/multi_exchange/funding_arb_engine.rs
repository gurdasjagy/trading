//! Institutional-Grade Cross-Exchange Funding Rate Arbitrage Engine
//!
//! Full lifecycle management for delta-neutral funding rate arbitrage:
//!   Phase 1: Pre-Trade Checks (profitability gate, slippage, basis, margin)
//!   Phase 2: Execution (simultaneous dual-leg with legging protection)
//!   Phase 3: Active Monitoring (rate fluctuation, margin, delta neutrality)
//!   Phase 4: Exit (take profit, spread reversal, margin danger, time stop)

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::dashboard_server::DashboardState;

use crate::execution_gateway::{
    ExecutionGateway, OrderIntent, OrderResult, OrderSide, OrderType,
};
use crate::execution_state::PlacementType;
use crate::multi_exchange::global_book::{ExchangeId, GlobalBookRegistry};
use crate::multi_exchange::funding_arb::{CrossExchangeFundingArb, FundingArbOpportunity};
use crate::multi_exchange::margin_monitor::CrossVenueMarginMonitor;
use crate::multi_exchange::funding_arb_executor::{DualLegExecutor, DualLegResult, LegStatus};
use crate::multi_exchange::funding_arb_risk::{PreTradeValidator, PreTradeResult, ExitReason};

// ---------------------------------------------------------------------------
// Position State Machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a funding arbitrage position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FundingArbState {
    /// Opportunity detected, pre-trade checks in progress.
    Validating,
    /// Pre-trade checks passed, executing dual-leg entry.
    Entering,
    /// Both legs filled, position is active and collecting funding.
    Active,
    /// Exit triggered, closing both legs.
    Exiting,
    /// Both legs closed, final PnL calculated.
    Closed,
    /// Entry failed or was aborted (legging risk, insufficient margin, etc.)
    Failed,
}

// ---------------------------------------------------------------------------
// Funding Arbitrage Position
// ---------------------------------------------------------------------------

/// A complete funding arbitrage position with full lifecycle tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbPosition {
    /// Unique position ID.
    pub id: u64,
    /// Trading symbol (e.g., "BTC_USDT").
    pub symbol: String,
    /// Current lifecycle state.
    pub state: FundingArbState,

    // --- Leg Details ---
    /// Exchange where we are SHORT (receiving funding when rate is positive).
    pub short_exchange: ExchangeId,
    /// Exchange where we are LONG (hedge leg).
    pub long_exchange: ExchangeId,
    /// Short leg entry price (actual fill, not mark price).
    pub short_entry_price: f64,
    /// Long leg entry price (actual fill).
    pub long_entry_price: f64,
    /// Position size in contracts.
    pub size: i64,
    /// Leverage used on each leg.
    pub leverage: i32,

    // --- Entry Metrics ---
    /// Entry basis: short_entry_price - long_entry_price.
    /// Positive = we entered with a favorable basis (short higher than long).
    pub entry_basis: f64,
    /// Round-trip fee cost in USDT (entry + exit, both legs).
    pub total_fee_cost_usdt: f64,
    /// Estimated VWAP slippage at entry (both legs combined, USDT).
    pub entry_slippage_usdt: f64,
    /// Net funding rate spread at entry (short_rate - long_rate).
    pub entry_net_rate: f64,
    /// Annualized APR at entry.
    pub entry_annualized_apr: f64,
    /// Minimum funding periods needed to break even on fees.
    pub breakeven_periods: f64,

    // --- Funding Accumulation ---
    /// Total funding received on short leg (positive = received).
    pub funding_received_short: f64,
    /// Total funding paid on long leg (negative = paid).
    pub funding_paid_long: f64,
    /// Net funding accumulated (received - paid).
    pub net_funding_accumulated: f64,
    /// Number of funding periods collected.
    pub funding_periods_collected: u32,
    /// History of funding rates observed per period.
    pub funding_rate_history: Vec<FundingPeriodRecord>,

    // --- Timestamps ---
    /// When the opportunity was first detected (nanoseconds).
    pub detected_ns: u64,
    /// When both legs were filled (nanoseconds).
    pub entry_ns: u64,
    /// When exit was triggered (nanoseconds).
    pub exit_trigger_ns: u64,
    /// When both legs were closed (nanoseconds).
    pub closed_ns: u64,

    // --- Exit Details ---
    /// Short leg exit price.
    pub short_exit_price: f64,
    /// Long leg exit price.
    pub long_exit_price: f64,
    /// Exit basis: short_exit_price - long_exit_price.
    pub exit_basis: f64,
    /// Reason for exit.
    pub exit_reason: Option<ExitReason>,

    // --- PnL Breakdown ---
    /// Price PnL = (short_entry - short_exit) * size + (long_exit - long_entry) * size.
    pub price_pnl: f64,
    /// Net funding PnL = net_funding_accumulated.
    pub funding_pnl: f64,
    /// Total fees paid (entry + exit, both legs).
    pub fees_paid: f64,
    /// Net PnL = price_pnl + funding_pnl - fees_paid.
    pub net_pnl: f64,
    /// Net PnL as percentage of capital deployed.
    pub net_pnl_pct: f64,

    // --- Order IDs for reconciliation ---
    pub short_entry_order_id: String,
    pub long_entry_order_id: String,
    pub short_exit_order_id: String,
    pub long_exit_order_id: String,
}

/// Record of a single funding period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingPeriodRecord {
    pub timestamp: u64,
    pub short_rate: f64,
    pub long_rate: f64,
    pub net_rate: f64,
    pub funding_received: f64,
    pub cumulative_funding: f64,
}

// ---------------------------------------------------------------------------
// Engine Configuration
// ---------------------------------------------------------------------------

/// Configuration for the funding arbitrage engine.
#[derive(Debug, Clone)]
pub struct FundingArbEngineConfig {
    // --- Pre-Trade Thresholds ---
    /// Minimum net funding rate spread to consider (e.g., 0.0001 = 0.01%).
    pub min_net_rate: f64,
    /// Minimum annualized APR to consider (e.g., 0.10 = 10%).
    pub min_annualized_apr: f64,
    /// Maximum acceptable entry slippage in basis points.
    pub max_entry_slippage_bps: f64,
    /// Maximum acceptable basis risk as percentage of position value.
    pub max_basis_risk_pct: f64,
    /// Minimum margin ratio required on both exchanges before entry.
    pub min_entry_margin_ratio: f64,
    /// Minimum number of funding periods the spread must persist to break even.
    pub max_breakeven_periods: f64,

    // --- Position Sizing ---
    /// Maximum position size as percentage of smallest exchange balance.
    pub max_position_pct: f64,
    /// Maximum notional value per position in USDT.
    pub max_notional_usdt: f64,
    /// Leverage to use on each leg.
    pub leverage: i32,
    /// Maximum number of concurrent funding arb positions.
    pub max_concurrent_positions: usize,

    // --- Active Monitoring ---
    /// How often to re-check funding rates (seconds).
    pub rate_check_interval_secs: u64,
    /// How often to check margin health (seconds).
    pub margin_check_interval_secs: u64,
    /// Delta neutrality tolerance (e.g., 0.02 = 2% imbalance allowed).
    pub delta_neutral_tolerance: f64,

    // --- Exit Thresholds ---
    /// Take profit: close when net funding accumulated exceeds this (USDT).
    pub take_profit_usdt: f64,
    /// Take profit: close when net funding accumulated exceeds this (% of capital).
    pub take_profit_pct: f64,
    /// Stop loss: close when net PnL (price + funding) drops below this (USDT).
    pub stop_loss_usdt: f64,
    /// Stop loss: close when net PnL drops below this (% of capital).
    pub stop_loss_pct: f64,
    /// Spread reversal: close when net rate inverts by this amount.
    pub spread_reversal_threshold: f64,
    /// Maximum hold time in hours.
    pub max_hold_hours: f64,
    /// Margin danger: close when either leg's margin ratio drops below this.
    pub margin_danger_ratio: f64,
    /// Number of consecutive funding periods with negative net rate before exit.
    pub max_negative_periods: u32,

    // --- Execution ---
    /// Maximum slippage allowed on entry orders (fraction, e.g., 0.001 = 0.1%).
    pub max_order_slippage: f64,
    /// Timeout for dual-leg execution (milliseconds).
    pub execution_timeout_ms: u64,
    /// Whether to use market orders (true) or limit orders (false) for entry.
    pub use_market_orders: bool,
}

impl Default for FundingArbEngineConfig {
    fn default() -> Self {
        Self {
            min_net_rate: 0.0001,           // 0.01% minimum spread
            min_annualized_apr: 0.10,       // 10% minimum APR
            max_entry_slippage_bps: 5.0,    // 5 bps max slippage
            max_basis_risk_pct: 0.002,      // 0.2% max basis risk
            min_entry_margin_ratio: 0.40,   // 40% margin required
            max_breakeven_periods: 3.0,     // Must break even within 3 funding periods

            max_position_pct: 0.15,         // 15% of smallest balance
            max_notional_usdt: 50_000.0,    // $50k max per position
            leverage: 3,                     // 3x leverage (conservative)
            max_concurrent_positions: 3,     // Max 3 concurrent arb positions

            rate_check_interval_secs: 60,   // Check rates every 60s
            margin_check_interval_secs: 30, // Check margin every 30s
            delta_neutral_tolerance: 0.02,  // 2% delta tolerance

            take_profit_usdt: 100.0,        // $100 TP
            take_profit_pct: 0.005,         // 0.5% TP
            stop_loss_usdt: -50.0,          // -$50 SL
            stop_loss_pct: -0.003,          // -0.3% SL
            spread_reversal_threshold: -0.00005, // Spread inverts by 0.005%
            max_hold_hours: 72.0,           // 72 hour max hold
            margin_danger_ratio: 0.20,      // 20% margin = danger
            max_negative_periods: 3,        // 3 consecutive negative periods

            max_order_slippage: 0.001,      // 0.1% max order slippage
            execution_timeout_ms: 5000,     // 5 second execution timeout
            use_market_orders: true,        // Market orders for guaranteed fill
        }
    }
}

impl FundingArbEngineConfig {
    /// BUG 9 FIX: Testnet-aware configuration with relaxed thresholds.
    /// Testnet order books are thin, funding rates are often random/zero,
    /// and balances are small. The mainnet defaults reject every opportunity
    /// on testnet, making the funding arb engine effectively dead.
    pub fn testnet() -> Self {
        Self {
            min_net_rate: 0.00001,          // 0.001% minimum spread (relaxed 10x)
            min_annualized_apr: 0.01,       // 1% minimum APR (relaxed from 10%)
            max_entry_slippage_bps: 50.0,   // 50 bps max slippage (relaxed from 5)
            max_basis_risk_pct: 0.01,       // 1% max basis risk (relaxed from 0.2%)
            min_entry_margin_ratio: 0.10,   // 10% margin required (relaxed from 40%)
            max_breakeven_periods: 10.0,    // 10 funding periods (relaxed from 3)

            max_position_pct: 0.30,         // 30% of smallest balance (relaxed)
            max_notional_usdt: 10_000.0,    // $10k max (smaller for testnet)
            leverage: 2,                     // 2x leverage (conservative for testnet)
            max_concurrent_positions: 2,     // Max 2 concurrent arb positions

            rate_check_interval_secs: 30,   // Check rates every 30s (faster on testnet)
            margin_check_interval_secs: 15, // Check margin every 15s
            delta_neutral_tolerance: 0.05,  // 5% delta tolerance (relaxed)

            take_profit_usdt: 50.0,         // $50 TP (lower for testnet)
            take_profit_pct: 0.01,          // 1% TP
            stop_loss_usdt: -25.0,          // -$25 SL
            stop_loss_pct: -0.005,          // -0.5% SL
            spread_reversal_threshold: -0.0001, // Spread inverts by 0.01%
            max_hold_hours: 48.0,           // 48 hour max hold
            margin_danger_ratio: 0.10,      // 10% margin = danger
            max_negative_periods: 5,        // 5 consecutive negative periods

            max_order_slippage: 0.005,      // 0.5% max order slippage (relaxed)
            execution_timeout_ms: 10000,    // 10 second execution timeout (longer)
            use_market_orders: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Funding Arbitrage Engine
// ---------------------------------------------------------------------------

/// The main funding arbitrage engine that orchestrates the full lifecycle.
pub struct FundingArbEngine {
    config: FundingArbEngineConfig,
    /// Rate monitoring layer (existing module).
    rate_monitor: CrossExchangeFundingArb,
    /// Active positions.
    positions: Vec<FundingArbPosition>,
    /// Closed positions (for PnL tracking).
    closed_positions: Vec<FundingArbPosition>,
    /// Next position ID.
    next_id: u64,
    /// HTTP client for REST API calls.
    http_client: Client,
    /// Is the engine paused.
    pub paused: bool,
    /// Pause reason.
    pause_reason: Option<String>,
    /// Total realized PnL across all closed positions.
    total_realized_pnl: f64,
    /// Total funding collected across all positions.
    total_funding_collected: f64,
    /// Shutdown signal — engine exits its run() loop when set to true.
    shutdown: Arc<AtomicBool>,
}

impl FundingArbEngine {
    pub fn new(config: FundingArbEngineConfig, shutdown: Arc<AtomicBool>) -> Self {
        let rate_monitor = CrossExchangeFundingArb::new(
            config.min_net_rate,
            config.min_annualized_apr,
        );

        Self {
            config,
            rate_monitor,
            positions: Vec::new(),
            closed_positions: Vec::new(),
            next_id: 1,
            http_client: Client::builder()
                .timeout(Duration::from_secs(10))
                .pool_max_idle_per_host(10)
                .build()
                .expect("Failed to build HTTP client"),
            paused: false,
            pause_reason: None,
            total_realized_pnl: 0.0,
            total_funding_collected: 0.0,
            shutdown,
        }
    }

    /// Main engine loop. Spawned as a tokio task from main.rs.
    ///
    /// This is the core orchestration loop that:
    /// 1. Periodically fetches funding rates from all exchanges
    /// 2. Scans for new opportunities
    /// 3. Validates opportunities through pre-trade checks
    /// 4. Executes dual-leg entries
    /// 5. Monitors active positions
    /// 6. Triggers exits when conditions are met
    pub async fn run(
        &mut self,
        gateways: HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
        global_book_registry: Arc<GlobalBookRegistry>,
        margin_monitor: Arc<RwLock<CrossVenueMarginMonitor>>,
        dashboard_state: Option<Arc<DashboardState>>,
        symbols: Vec<String>,
        gateio_testnet: bool,
        binance_testnet: bool,
        bybit_testnet: bool,
    ) {
        info!("[funding-arb-engine] Starting institutional funding rate arbitrage engine");
        info!("[funding-arb-engine] Config: min_net_rate={:.4}%, min_apr={:.1}%, max_positions={}",
            self.config.min_net_rate * 100.0,
            self.config.min_annualized_apr * 100.0,
            self.config.max_concurrent_positions);

        let mut rate_interval = tokio::time::interval(
            Duration::from_secs(self.config.rate_check_interval_secs)
        );
        let mut margin_interval = tokio::time::interval(
            Duration::from_secs(self.config.margin_check_interval_secs)
        );
        let mut funding_collection_interval = tokio::time::interval(
            Duration::from_secs(8 * 3600) // Every 8 hours (funding period)
        );

        loop {
            // Check shutdown signal
            if self.shutdown.load(Ordering::Relaxed) {
                info!("[funding-arb-engine] Shutdown signal received, stopping engine");
                break;
            }

            tokio::select! {
                // --- Rate Check Tick ---
                _ = rate_interval.tick() => {
                    if self.paused { continue; }

                    // 1. Fetch latest funding rates from all exchanges
                    for symbol in &symbols {
                        self.rate_monitor.fetch_all_rates(
                            &self.http_client,
                            symbol,
                            gateio_testnet,
                            binance_testnet,
                            bybit_testnet,
                        ).await;
                    }

                    // 2. Scan for opportunities
                    let opportunities = self.rate_monitor.scan_opportunities();
                    let actionable: Vec<_> = opportunities.iter()
                        .filter(|o| o.is_actionable)
                        .collect();

                    if !actionable.is_empty() {
                        info!("[funding-arb-engine] Found {} actionable opportunities", actionable.len());
                    }

                    // 3. Check if we can open new positions
                    if self.positions.len() >= self.config.max_concurrent_positions {
                        debug!("[funding-arb-engine] Max concurrent positions reached ({})",
                            self.config.max_concurrent_positions);
                        continue;
                    }

                    // 4. Validate and execute best opportunity
                    for opp in actionable {
                        // Skip if we already have a position for this symbol
                        if self.positions.iter().any(|p| p.symbol == opp.symbol && p.state == FundingArbState::Active) {
                            continue;
                        }

                        // Pre-trade validation
                        let validation = PreTradeValidator::validate(
                            opp,
                            &global_book_registry,
                            &margin_monitor.read(),
                            &self.config,
                        );

                        match validation {
                            PreTradeResult::Approved { estimated_slippage_bps, basis_risk, breakeven_periods, recommended_size } => {
                                info!("[funding-arb-engine] PRE-TRADE APPROVED: {} short@{} long@{} net_rate={:.4}% apr={:.1}% slippage={:.1}bps basis={:.4} breakeven={:.1} periods size={}",
                                    opp.symbol, opp.short_exchange.name(), opp.long_exchange.name(),
                                    opp.net_rate * 100.0, opp.annualized_apr * 100.0,
                                    estimated_slippage_bps, basis_risk, breakeven_periods, recommended_size);

                                // Execute dual-leg entry
                                let result = DualLegExecutor::execute_entry(
                                    &opp.symbol,
                                    opp.short_exchange,
                                    opp.long_exchange,
                                    recommended_size,
                                    self.config.leverage,
                                    self.config.use_market_orders,
                                    self.config.max_order_slippage,
                                    self.config.execution_timeout_ms,
                                    &gateways,
                                ).await;

                                match result {
                                    DualLegResult::BothFilled { short_result, long_result } => {
                                        let position = self.create_position(
                                            opp, &short_result, &long_result,
                                            recommended_size, estimated_slippage_bps,
                                            basis_risk, breakeven_periods,
                                        );
                                        info!("[funding-arb-engine] POSITION OPENED: id={} {} short@{} ({:.4}) long@{} ({:.4}) size={} basis={:.4}",
                                            position.id, position.symbol,
                                            position.short_exchange.name(), position.short_entry_price,
                                            position.long_exchange.name(), position.long_entry_price,
                                            position.size, position.entry_basis);
                                        self.positions.push(position);
                                    }
                                    DualLegResult::PartialFill { filled_leg, unfilled_exchange, filled_size } => {
                                        // LEGGING RISK: One leg filled, other didn't
                                        // Retry emergency close up to 3 times with exponential backoff
                                        warn!("[funding-arb-engine] LEGGING RISK: {} filled on {} but failed on {}. Emergency closing filled leg.",
                                            opp.symbol, filled_leg.exchange().name(), unfilled_exchange.name());

                                        let mut closed = false;
                                        for attempt in 1..=3u32 {
                                            match DualLegExecutor::emergency_close_leg(
                                                &opp.symbol,
                                                &filled_leg,
                                                filled_size,
                                                &gateways,
                                            ).await {
                                                Ok(_) => {
                                                    info!("[funding-arb-engine] Legging risk resolved: closed filled leg (attempt {})", attempt);
                                                    closed = true;
                                                    break;
                                                }
                                                Err(e) => {
                                                    error!("[funding-arb-engine] Emergency close attempt {}/3 failed: {:?}", attempt, e);
                                                    if attempt < 3 {
                                                        tokio::time::sleep(Duration::from_millis(500 * 2u64.pow(attempt - 1))).await;
                                                    }
                                                }
                                            }
                                        }
                                        if !closed {
                                            error!("[funding-arb-engine] CRITICAL: All 3 emergency close attempts failed for {}! Pausing engine.", opp.symbol);
                                            self.pause(&format!("Unresolved legging risk on {}", opp.symbol));
                                        }
                                    }
                                    DualLegResult::BothFailed { short_error, long_error } => {
                                        warn!("[funding-arb-engine] Entry failed on both legs: short={:?} long={:?}",
                                            short_error, long_error);
                                    }
                                }

                                // Only open one position per cycle
                                break;
                            }
                            PreTradeResult::Rejected { reason } => {
                                debug!("[funding-arb-engine] Pre-trade rejected for {}: {}", opp.symbol, reason);
                            }
                        }
                    }

                    // 5. Check exits for active positions
                    self.check_exits(&gateways, &margin_monitor.read()).await;

                    // 6. Update dashboard state
                    if let Some(ref dash) = dashboard_state {
                        let json = serde_json::to_string(&self.to_json()).unwrap_or_else(|_| "[]".to_string());
                        dash.set_funding_arb_json(json);
                    }
                }

                // --- Margin Check Tick ---
                _ = margin_interval.tick() => {
                    if self.paused { continue; }

                    // Monitor margin health for all active positions
                    for pos in &self.positions {
                        if pos.state != FundingArbState::Active { continue; }

                        let margin_read = margin_monitor.read();
                        let short_health = margin_read.get_health(pos.short_exchange).cloned();
                        let long_health = margin_read.get_health(pos.long_exchange).cloned();
                        drop(margin_read);

                        // Check for margin danger
                        if let Some(ref health) = short_health {
                            if health.margin_ratio < self.config.margin_danger_ratio {
                                warn!("[funding-arb-engine] MARGIN DANGER: {} short leg on {} margin={:.1}%",
                                    pos.symbol, pos.short_exchange.name(), health.margin_ratio * 100.0);
                            }
                        }
                        if let Some(ref health) = long_health {
                            if health.margin_ratio < self.config.margin_danger_ratio {
                                warn!("[funding-arb-engine] MARGIN DANGER: {} long leg on {} margin={:.1}%",
                                    pos.symbol, pos.long_exchange.name(), health.margin_ratio * 100.0);
                            }
                        }
                    }
                }

                // --- Funding Collection Tick (every 8 hours) ---
                _ = funding_collection_interval.tick() => {
                    if self.paused { continue; }

                    // Record funding for all active positions
                    for pos in self.positions.iter_mut() {
                        if pos.state != FundingArbState::Active { continue; }

                        // Get current rates
                        let short_rate = self.rate_monitor
                            .get_rate(pos.short_exchange, &pos.symbol)
                            .map(|r| r.rate)
                            .unwrap_or(0.0);
                        let long_rate = self.rate_monitor
                            .get_rate(pos.long_exchange, &pos.symbol)
                            .map(|r| r.rate)
                            .unwrap_or(0.0);

                        let net_rate = short_rate - long_rate;
                        let notional = pos.size as f64 * pos.short_entry_price;
                        let funding_this_period = notional * net_rate;

                        pos.funding_received_short += notional * short_rate;
                        pos.funding_paid_long += notional * long_rate;
                        pos.net_funding_accumulated += funding_this_period;
                        pos.funding_periods_collected += 1;

                        pos.funding_rate_history.push(FundingPeriodRecord {
                            timestamp: now_ns(),
                            short_rate,
                            long_rate,
                            net_rate,
                            funding_received: funding_this_period,
                            cumulative_funding: pos.net_funding_accumulated,
                        });

                        info!("[funding-arb-engine] FUNDING COLLECTED: {} period={} net_rate={:.4}% this_period=${:.4} cumulative=${:.4}",
                            pos.symbol, pos.funding_periods_collected,
                            net_rate * 100.0, funding_this_period, pos.net_funding_accumulated);
                    }

                    // Update dashboard after funding collection
                    if let Some(ref dash) = dashboard_state {
                        let json = serde_json::to_string(&self.to_json()).unwrap_or_else(|_| "[]".to_string());
                        dash.set_funding_arb_json(json);
                    }
                }
            }
        }

        info!("[funding-arb-engine] Engine stopped. Realized PnL: ${:.4}, Funding collected: ${:.4}",
            self.total_realized_pnl, self.total_funding_collected);
    }

    /// Check all active positions for exit conditions.
    async fn check_exits(
        &mut self,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
        margin_monitor: &CrossVenueMarginMonitor,
    ) {
        let now = now_ns();
        let mut positions_to_close: Vec<(usize, ExitReason)> = Vec::new();

        for (idx, pos) in self.positions.iter().enumerate() {
            if pos.state != FundingArbState::Active { continue; }

            // Check all exit conditions
            if let Some(exit_reason) = self.evaluate_exit(pos, margin_monitor, now) {
                positions_to_close.push((idx, exit_reason));
            }
        }

        // Execute exits (reverse order to maintain indices)
        for (idx, exit_reason) in positions_to_close.into_iter().rev() {
            let pos = &mut self.positions[idx];
            pos.state = FundingArbState::Exiting;
            pos.exit_trigger_ns = now;
            pos.exit_reason = Some(exit_reason.clone());

            info!("[funding-arb-engine] EXIT TRIGGERED: {} reason={:?} net_funding=${:.4} periods={}",
                pos.symbol, exit_reason, pos.net_funding_accumulated, pos.funding_periods_collected);

            // Execute dual-leg exit
            let result = DualLegExecutor::execute_exit(
                &pos.symbol,
                pos.short_exchange,
                pos.long_exchange,
                pos.size,
                self.config.execution_timeout_ms,
                gateways,
            ).await;

            match result {
                DualLegResult::BothFilled { short_result, long_result } => {
                    pos.short_exit_price = short_result.avg_fill_price;
                    pos.long_exit_price = long_result.avg_fill_price;
                    pos.exit_basis = pos.short_exit_price - pos.long_exit_price;
                    pos.short_exit_order_id = short_result.order_id.clone();
                    pos.long_exit_order_id = long_result.order_id.clone();

                    // Calculate final PnL
                    let size_f64 = pos.size as f64;
                    pos.price_pnl = (pos.short_entry_price - pos.short_exit_price) * size_f64
                                  + (pos.long_exit_price - pos.long_entry_price) * size_f64;
                    pos.funding_pnl = pos.net_funding_accumulated;
                    pos.fees_paid = pos.total_fee_cost_usdt
                                  + short_result.fee + long_result.fee;
                    pos.net_pnl = pos.price_pnl + pos.funding_pnl - pos.fees_paid;

                    let capital_deployed = size_f64 * pos.short_entry_price / pos.leverage as f64 * 2.0;
                    pos.net_pnl_pct = if capital_deployed > 0.0 {
                        pos.net_pnl / capital_deployed
                    } else { 0.0 };

                    pos.state = FundingArbState::Closed;
                    pos.closed_ns = now_ns();

                    self.total_realized_pnl += pos.net_pnl;
                    self.total_funding_collected += pos.funding_pnl;

                    info!("[funding-arb-engine] POSITION CLOSED: {} | price_pnl=${:.4} funding_pnl=${:.4} fees=${:.4} NET=${:.4} ({:.2}%) | periods={} hold={:.1}h",
                        pos.symbol, pos.price_pnl, pos.funding_pnl, pos.fees_paid,
                        pos.net_pnl, pos.net_pnl_pct * 100.0,
                        pos.funding_periods_collected,
                        (pos.closed_ns - pos.entry_ns) as f64 / 3_600_000_000_000.0);

                    // Move to closed positions
                    let closed = self.positions.remove(idx);
                    self.closed_positions.push(closed);
                }
                _ => {
                    error!("[funding-arb-engine] Exit execution failed for {} - position remains open!", pos.symbol);
                    pos.state = FundingArbState::Active; // Revert state, will retry next cycle
                }
            }
        }
    }

    /// Evaluate whether a position should be exited.
    fn evaluate_exit(
        &self,
        pos: &FundingArbPosition,
        margin_monitor: &CrossVenueMarginMonitor,
        now_ns: u64,
    ) -> Option<ExitReason> {
        let hours_open = (now_ns.saturating_sub(pos.entry_ns)) as f64 / 3_600_000_000_000.0;
        let capital_deployed = pos.size as f64 * pos.short_entry_price / pos.leverage as f64 * 2.0;

        // 1. Take Profit (absolute)
        if pos.net_funding_accumulated >= self.config.take_profit_usdt {
            return Some(ExitReason::TakeProfit {
                accumulated_funding: pos.net_funding_accumulated,
            });
        }

        // 2. Take Profit (percentage)
        if capital_deployed > 0.0 {
            let pnl_pct = (pos.net_funding_accumulated + pos.price_pnl_estimate()) / capital_deployed;
            if pnl_pct >= self.config.take_profit_pct {
                return Some(ExitReason::TakeProfitPct { pnl_pct });
            }
        }

        // 3. Stop Loss
        let estimated_net_pnl = pos.net_funding_accumulated + pos.price_pnl_estimate() - pos.total_fee_cost_usdt;
        if estimated_net_pnl <= self.config.stop_loss_usdt {
            return Some(ExitReason::StopLoss { net_pnl: estimated_net_pnl });
        }

        // 4. Spread Reversal
        let current_net_rate = self.rate_monitor
            .get_rate(pos.short_exchange, &pos.symbol)
            .zip(self.rate_monitor.get_rate(pos.long_exchange, &pos.symbol))
            .map(|(s, l)| s.rate - l.rate)
            .unwrap_or(0.0);

        if current_net_rate < self.config.spread_reversal_threshold {
            return Some(ExitReason::SpreadReversal {
                entry_net_rate: pos.entry_net_rate,
                current_net_rate,
            });
        }

        // 5. Consecutive negative funding periods
        let recent_negative = pos.funding_rate_history.iter().rev()
            .take(self.config.max_negative_periods as usize)
            .filter(|r| r.net_rate < 0.0)
            .count();
        if recent_negative >= self.config.max_negative_periods as usize
            && pos.funding_periods_collected >= self.config.max_negative_periods
        {
            return Some(ExitReason::ConsecutiveNegativePeriods {
                count: recent_negative as u32,
            });
        }

        // 6. Time Stop
        if hours_open >= self.config.max_hold_hours {
            return Some(ExitReason::TimeStop { hours_open });
        }

        // 7. Margin Danger
        for exchange in [pos.short_exchange, pos.long_exchange] {
            if let Some(health) = margin_monitor.get_health(exchange) {
                if health.is_critical {
                    return Some(ExitReason::MarginDanger {
                        exchange,
                        margin_ratio: health.margin_ratio,
                    });
                }
            }
        }

        None
    }

    fn create_position(
        &mut self,
        opp: &FundingArbOpportunity,
        short_result: &OrderResult,
        long_result: &OrderResult,
        size: i64,
        slippage_bps: f64,
        basis_risk: f64,
        breakeven_periods: f64,
    ) -> FundingArbPosition {
        let id = self.next_id;
        self.next_id += 1;
        let now = now_ns();

        let short_fee_bps = opp.short_exchange.taker_fee_bps() as f64;
        let long_fee_bps = opp.long_exchange.taker_fee_bps() as f64;
        let notional = size as f64 * short_result.avg_fill_price;
        let total_fee_cost = notional * (short_fee_bps + long_fee_bps) * 2.0 / 10000.0;

        FundingArbPosition {
            id,
            symbol: opp.symbol.clone(),
            state: FundingArbState::Active,
            short_exchange: opp.short_exchange,
            long_exchange: opp.long_exchange,
            short_entry_price: short_result.avg_fill_price,
            long_entry_price: long_result.avg_fill_price,
            size,
            leverage: self.config.leverage,
            entry_basis: short_result.avg_fill_price - long_result.avg_fill_price,
            total_fee_cost_usdt: total_fee_cost,
            entry_slippage_usdt: notional * slippage_bps / 10000.0,
            entry_net_rate: opp.net_rate,
            entry_annualized_apr: opp.annualized_apr,
            breakeven_periods,
            funding_received_short: 0.0,
            funding_paid_long: 0.0,
            net_funding_accumulated: 0.0,
            funding_periods_collected: 0,
            funding_rate_history: Vec::new(),
            detected_ns: now,
            entry_ns: now,
            exit_trigger_ns: 0,
            closed_ns: 0,
            short_exit_price: 0.0,
            long_exit_price: 0.0,
            exit_basis: 0.0,
            exit_reason: None,
            price_pnl: 0.0,
            funding_pnl: 0.0,
            fees_paid: total_fee_cost,
            net_pnl: 0.0,
            net_pnl_pct: 0.0,
            short_entry_order_id: short_result.order_id.clone(),
            long_entry_order_id: long_result.order_id.clone(),
            short_exit_order_id: String::new(),
            long_exit_order_id: String::new(),
        }
    }

    /// Get the current number of active positions.
    pub fn active_position_count(&self) -> usize {
        self.positions.iter()
            .filter(|p| p.state == FundingArbState::Active)
            .count()
    }

    /// Get the total realized PnL.
    pub fn total_realized_pnl(&self) -> f64 {
        self.total_realized_pnl
    }

    /// Get the total funding collected.
    pub fn total_funding_collected(&self) -> f64 {
        self.total_funding_collected
    }

    /// Pause the engine.
    pub fn pause(&mut self, reason: &str) {
        self.paused = true;
        self.pause_reason = Some(reason.to_string());
        warn!("[funding-arb-engine] Engine PAUSED: {}", reason);
    }

    /// Resume the engine.
    pub fn resume(&mut self) {
        self.paused = false;
        self.pause_reason = None;
        info!("[funding-arb-engine] Engine RESUMED");
    }

    /// Serialize engine state to JSON for dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "active_positions": self.positions.iter()
                .filter(|p| p.state == FundingArbState::Active)
                .count(),
            "total_positions": self.positions.len(),
            "total_realized_pnl": self.total_realized_pnl,
            "total_funding_collected": self.total_funding_collected,
            "closed_count": self.closed_positions.len(),
            "positions": self.positions.iter().map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "symbol": p.symbol,
                    "state": format!("{:?}", p.state),
                    "short_exchange": p.short_exchange.name(),
                    "long_exchange": p.long_exchange.name(),
                    "size": p.size,
                    "entry_net_rate_pct": format!("{:.4}%", p.entry_net_rate * 100.0),
                    "entry_apr_pct": format!("{:.1}%", p.entry_annualized_apr * 100.0),
                    "net_funding": p.net_funding_accumulated,
                    "funding_periods": p.funding_periods_collected,
                    "net_pnl": p.net_pnl,
                })
            }).collect::<Vec<_>>(),
            "opportunities": self.rate_monitor.to_json(),
        })
    }
}

impl FundingArbPosition {
    /// Estimate current price PnL without actually querying exchange prices.
    /// Uses entry prices as proxy (delta-neutral should be ~0).
    ///
    /// TODO: In production, query live prices from gateways for more accurate
    /// real-time PnL estimation. For a truly delta-neutral position, price PnL
    /// should be near zero — the actual PnL comes from funding collection.
    fn price_pnl_estimate(&self) -> f64 {
        0.0
    }

    /// Calculate the capital deployed for this position (both legs combined).
    pub fn capital_deployed(&self) -> f64 {
        self.size as f64 * self.short_entry_price / self.leverage as f64 * 2.0
    }

    /// Calculate the current ROI (net PnL / capital deployed).
    pub fn roi(&self) -> f64 {
        let capital = self.capital_deployed();
        if capital > 0.0 {
            self.net_pnl / capital
        } else {
            0.0
        }
    }

    /// Calculate how long this position has been open in hours.
    pub fn hours_open(&self) -> f64 {
        let now = now_ns();
        let end = if self.closed_ns > 0 { self.closed_ns } else { now };
        (end.saturating_sub(self.entry_ns)) as f64 / 3_600_000_000_000.0
    }

    /// Calculate the annualized return based on actual performance.
    pub fn annualized_return(&self) -> f64 {
        let hours = self.hours_open();
        if hours <= 0.0 {
            return 0.0;
        }
        let roi = self.roi();
        // Annualize: roi * (8760 hours/year / hours_open)
        roi * (8760.0 / hours)
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FundingArbEngineConfig::default();
        assert_eq!(config.min_net_rate, 0.0001);
        assert_eq!(config.min_annualized_apr, 0.10);
        assert_eq!(config.max_entry_slippage_bps, 5.0);
        assert_eq!(config.max_basis_risk_pct, 0.002);
        assert_eq!(config.min_entry_margin_ratio, 0.40);
        assert_eq!(config.max_breakeven_periods, 3.0);
        assert_eq!(config.max_position_pct, 0.15);
        assert_eq!(config.max_notional_usdt, 50_000.0);
        assert_eq!(config.leverage, 3);
        assert_eq!(config.max_concurrent_positions, 3);
        assert_eq!(config.rate_check_interval_secs, 60);
        assert_eq!(config.margin_check_interval_secs, 30);
        assert_eq!(config.delta_neutral_tolerance, 0.02);
        assert_eq!(config.take_profit_usdt, 100.0);
        assert_eq!(config.take_profit_pct, 0.005);
        assert_eq!(config.stop_loss_usdt, -50.0);
        assert_eq!(config.stop_loss_pct, -0.003);
        assert_eq!(config.max_hold_hours, 72.0);
        assert_eq!(config.margin_danger_ratio, 0.20);
        assert_eq!(config.max_negative_periods, 3);
        assert_eq!(config.max_order_slippage, 0.001);
        assert_eq!(config.execution_timeout_ms, 5000);
        assert!(config.use_market_orders);
    }

    #[test]
    fn test_engine_creation() {
        let config = FundingArbEngineConfig::default();
        let shutdown = Arc::new(AtomicBool::new(false));
        let engine = FundingArbEngine::new(config, shutdown);
        assert_eq!(engine.active_position_count(), 0);
        assert_eq!(engine.total_realized_pnl(), 0.0);
        assert_eq!(engine.total_funding_collected(), 0.0);
        assert!(!engine.paused);
    }

    #[test]
    fn test_engine_pause_resume() {
        let config = FundingArbEngineConfig::default();
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut engine = FundingArbEngine::new(config, shutdown);

        engine.pause("test pause");
        assert!(engine.paused);
        assert_eq!(engine.pause_reason.as_deref(), Some("test pause"));

        engine.resume();
        assert!(!engine.paused);
        assert!(engine.pause_reason.is_none());
    }

    #[test]
    fn test_engine_to_json() {
        let config = FundingArbEngineConfig::default();
        let shutdown = Arc::new(AtomicBool::new(false));
        let engine = FundingArbEngine::new(config, shutdown);
        let json = engine.to_json();

        assert_eq!(json["paused"], false);
        assert_eq!(json["active_positions"], 0);
        assert_eq!(json["total_realized_pnl"], 0.0);
        assert_eq!(json["closed_count"], 0);
    }

    #[test]
    fn test_position_state_serialization() {
        let state = FundingArbState::Active;
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("Active"));

        let deserialized: FundingArbState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, FundingArbState::Active);
    }

    #[test]
    fn test_position_pnl_calculations() {
        let pos = FundingArbPosition {
            id: 1,
            symbol: "BTC_USDT".to_string(),
            state: FundingArbState::Closed,
            short_exchange: ExchangeId::GateIo,
            long_exchange: ExchangeId::Binance,
            short_entry_price: 50100.0,
            long_entry_price: 50000.0,
            size: 1,
            leverage: 3,
            entry_basis: 100.0,
            total_fee_cost_usdt: 18.0, // ~18 bps round trip
            entry_slippage_usdt: 5.0,
            entry_net_rate: 0.001,
            entry_annualized_apr: 1.095,
            breakeven_periods: 1.5,
            funding_received_short: 50.0,
            funding_paid_long: 5.0,
            net_funding_accumulated: 45.0,
            funding_periods_collected: 3,
            funding_rate_history: Vec::new(),
            detected_ns: 0,
            entry_ns: 1000000000000000000, // some ts
            exit_trigger_ns: 0,
            closed_ns: 1000000000000000000 + 24 * 3_600_000_000_000, // 24h later
            short_exit_price: 50050.0,
            long_exit_price: 49950.0,
            exit_basis: 100.0,
            exit_reason: Some(ExitReason::TakeProfit { accumulated_funding: 45.0 }),
            // PnL: short (50100-50050)*1=50, long (49950-50000)*1=-50, price_pnl=0
            price_pnl: 0.0,
            funding_pnl: 45.0,
            fees_paid: 20.0,
            net_pnl: 25.0,
            net_pnl_pct: 0.00075, // 25 / (1 * 50100 / 3 * 2)
            short_entry_order_id: "s1".to_string(),
            long_entry_order_id: "l1".to_string(),
            short_exit_order_id: "s2".to_string(),
            long_exit_order_id: "l2".to_string(),
        };

        assert_eq!(pos.price_pnl_estimate(), 0.0);
        assert!(pos.capital_deployed() > 0.0);

        // capital_deployed = 1 * 50100 / 3 * 2 = 33400
        let expected_capital = 1.0 * 50100.0 / 3.0 * 2.0;
        assert!((pos.capital_deployed() - expected_capital).abs() < 0.01);

        // ROI = 25 / 33400
        let expected_roi = 25.0 / expected_capital;
        assert!((pos.roi() - expected_roi).abs() < 0.0001);
    }

    #[test]
    fn test_funding_period_record() {
        let record = FundingPeriodRecord {
            timestamp: 1000,
            short_rate: 0.001,
            long_rate: 0.0001,
            net_rate: 0.0009,
            funding_received: 45.0,
            cumulative_funding: 45.0,
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("0.001"));
        assert!(json.contains("45"));
    }
}
