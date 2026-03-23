//! Multi-Exchange Feature Module (USE_MULTI_EXCHANGE=on)
//!
//! This module provides the complete multi-exchange capability including:
//! - Consolidated Global Order Book with fee and latency adjustments
//! - Smart Order Router (SOR) for optimal cross-exchange execution
//! - Cross-exchange funding rate arbitrage detection
//! - Cross-venue margin health monitoring
//! - Multi-exchange WebSocket ingestion
//!
//! All functionality is gated by the `multi_exchange_enabled` config flag.
//! When disabled, this module compiles but has zero runtime impact.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::config::{ExchangeConfig, SymbolRegistry};
use crate::fixed_point::FixedPrice;

// ═══════════════════════════════════════════════════════════════════════════
// Exchange Identifier
// ═══════════════════════════════════════════════════════════════════════════

/// Exchange identifier enum - zero-cost, no heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExchangeId {
    GateIo = 0,
    Binance = 1,
    Bybit = 2,
}

impl ExchangeId {
    /// Taker fee in basis points for each exchange.
    /// Gate.io: 5 bps, Binance: 4 bps, Bybit: 5.5 bps
    pub fn taker_fee_bps(&self) -> i64 {
        match self {
            ExchangeId::GateIo => 5,
            ExchangeId::Binance => 4,
            ExchangeId::Bybit => 6, // 5.5 rounded up
        }
    }

    /// Maker fee in basis points (negative = rebate).
    /// Gate.io: 2 bps, Binance: 2 bps, Bybit: 1 bps
    pub fn maker_fee_bps(&self) -> i64 {
        match self {
            ExchangeId::GateIo => 2,
            ExchangeId::Binance => 2,
            ExchangeId::Bybit => 1,
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            ExchangeId::GateIo => "Gate.io",
            ExchangeId::Binance => "Binance",
            ExchangeId::Bybit => "Bybit",
        }
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gateio" | "gate.io" | "gate" => Some(ExchangeId::GateIo),
            "binance" => Some(ExchangeId::Binance),
            "bybit" => Some(ExchangeId::Bybit),
            _ => None,
        }
    }

    /// Get exchange index (0, 1, 2)
    pub fn index(&self) -> usize {
        *self as usize
    }
}

impl Default for ExchangeId {
    fn default() -> Self {
        ExchangeId::GateIo
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Global Order Book Structures
// ═══════════════════════════════════════════════════════════════════════════

/// A single price level in the global book, fee-and-latency adjusted.
#[derive(Debug, Clone)]
pub struct GlobalLevel {
    /// Raw price from the exchange (fixed-point, same precision as FlatOrderBook).
    pub raw_price_fp: i64,
    /// Fee-adjusted effective price (raw_price + taker_fee_bps adjustment).
    pub effective_price_fp: i64,
    /// Available quantity at this level (contracts).
    pub qty: i64,
    /// Which exchange this level came from.
    pub exchange: ExchangeId,
    /// Latency penalty in basis points (added to effective_price for asks,
    /// subtracted for bids, based on measured round-trip latency).
    pub latency_penalty_bps: i64,
    /// Timestamp of last update (nanoseconds since epoch).
    pub updated_ns: u64,
}

/// Per-exchange L2 snapshot for a single symbol.
#[derive(Debug, Clone)]
pub struct ExchangeBookSnapshot {
    pub exchange: ExchangeId,
    pub symbol_id: u16,
    pub best_bid_fp: i64,
    pub best_ask_fp: i64,
    pub bid_levels: Vec<(i64, i64)>, // (price_fp, qty)
    pub ask_levels: Vec<(i64, i64)>, // (price_fp, qty)
    pub sequence: u64,
    pub timestamp_ns: u64,
}

impl Default for ExchangeBookSnapshot {
    fn default() -> Self {
        Self {
            exchange: ExchangeId::GateIo,
            symbol_id: 0,
            best_bid_fp: 0,
            best_ask_fp: 0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            sequence: 0,
            timestamp_ns: 0,
        }
    }
}

/// The Consolidated Global Order Book for a single symbol.
/// Merges L2 books from all active exchanges, applying fee and latency
/// adjustments to produce a unified, sorted view of global liquidity.
pub struct GlobalOrderBook {
    pub symbol_id: u16,
    /// Per-exchange snapshots (indexed by ExchangeId as u8).
    exchange_snapshots: [Option<ExchangeBookSnapshot>; 3],
    /// Merged and sorted global bids (best bid first = highest price).
    pub global_bids: Vec<GlobalLevel>,
    /// Merged and sorted global asks (best ask first = lowest price).
    pub global_asks: Vec<GlobalLevel>,
    /// Measured round-trip latency per exchange in microseconds.
    pub latency_us: [u64; 3],
    /// Last full merge timestamp.
    pub last_merge_ns: u64,
}

impl GlobalOrderBook {
    pub fn new(symbol_id: u16) -> Self {
        Self {
            symbol_id,
            exchange_snapshots: [None, None, None],
            global_bids: Vec::with_capacity(60),
            global_asks: Vec::with_capacity(60),
            latency_us: [0; 3],
            last_merge_ns: 0,
        }
    }

    /// Update the snapshot for a specific exchange and trigger a re-merge.
    /// Called by the WS ingestion threads whenever a new book update arrives.
    pub fn update_exchange_snapshot(&mut self, snapshot: ExchangeBookSnapshot) {
        let idx = snapshot.exchange.index();
        self.exchange_snapshots[idx] = Some(snapshot);
        self.merge();
    }

    /// Update measured latency for an exchange (called by WS ingestion on pong).
    pub fn update_latency(&mut self, exchange: ExchangeId, latency_us: u64) {
        self.latency_us[exchange.index()] = latency_us;
    }

    /// Merge all exchange snapshots into the global_bids and global_asks vectors.
    /// Applies fee adjustment: effective_ask = raw_ask * (1 + taker_fee_bps/10000)
    ///                         effective_bid = raw_bid * (1 - taker_fee_bps/10000)
    /// Applies latency penalty: latency_penalty_bps = latency_us / 100 (1us = 0.01bps)
    /// Sorts bids descending by effective_price_fp.
    /// Sorts asks ascending by effective_price_fp.
    pub fn merge(&mut self) {
        self.global_bids.clear();
        self.global_asks.clear();

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_merge_ns = now_ns;

        for (idx, maybe_snap) in self.exchange_snapshots.iter().enumerate() {
            let Some(snap) = maybe_snap else { continue };
            let exchange = snap.exchange;
            let taker_fee_bps = exchange.taker_fee_bps();
            let latency_us = self.latency_us[idx];
            let latency_penalty_bps = (latency_us / 100) as i64; // 1us = 0.01bps

            // Process bid levels
            for &(price_fp, qty) in &snap.bid_levels {
                if qty <= 0 {
                    continue;
                }
                // Effective bid = raw_price * (1 - taker_fee_bps/10000) - latency penalty
                let fee_adj = price_fp * taker_fee_bps / 10000;
                let lat_adj = price_fp * latency_penalty_bps / 10000;
                let effective = price_fp - fee_adj - lat_adj;

                self.global_bids.push(GlobalLevel {
                    raw_price_fp: price_fp,
                    effective_price_fp: effective,
                    qty,
                    exchange,
                    latency_penalty_bps,
                    updated_ns: snap.timestamp_ns,
                });
            }

            // Process ask levels
            for &(price_fp, qty) in &snap.ask_levels {
                if qty <= 0 {
                    continue;
                }
                // Effective ask = raw_price * (1 + taker_fee_bps/10000) + latency penalty
                let fee_adj = price_fp * taker_fee_bps / 10000;
                let lat_adj = price_fp * latency_penalty_bps / 10000;
                let effective = price_fp + fee_adj + lat_adj;

                self.global_asks.push(GlobalLevel {
                    raw_price_fp: price_fp,
                    effective_price_fp: effective,
                    qty,
                    exchange,
                    latency_penalty_bps,
                    updated_ns: snap.timestamp_ns,
                });
            }
        }

        // Sort bids descending by effective price (highest first)
        self.global_bids
            .sort_by(|a, b| b.effective_price_fp.cmp(&a.effective_price_fp));

        // Sort asks ascending by effective price (lowest first)
        self.global_asks
            .sort_by(|a, b| a.effective_price_fp.cmp(&b.effective_price_fp));
    }

    /// Get the global best bid (highest effective bid across all exchanges).
    pub fn best_bid(&self) -> Option<&GlobalLevel> {
        self.global_bids.first()
    }

    /// Get the global best ask (lowest effective ask across all exchanges).
    pub fn best_ask(&self) -> Option<&GlobalLevel> {
        self.global_asks.first()
    }

    /// Calculate global mid price (average of best bid and best ask effective prices).
    pub fn global_mid_fp(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some((bid.effective_price_fp + ask.effective_price_fp) / 2)
            }
            _ => None,
        }
    }

    /// Calculate global spread in basis points.
    pub fn global_spread_bps(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                if bid.effective_price_fp > 0 {
                    let spread = ask.effective_price_fp - bid.effective_price_fp;
                    Some(spread * 10000 / bid.effective_price_fp)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get total available liquidity at the best N levels across all exchanges.
    pub fn total_bid_depth(&self, levels: usize) -> i64 {
        self.global_bids.iter().take(levels).map(|l| l.qty).sum()
    }

    pub fn total_ask_depth(&self, levels: usize) -> i64 {
        self.global_asks.iter().take(levels).map(|l| l.qty).sum()
    }

    /// Check if a specific exchange's book is stale (last update > 5 seconds ago).
    pub fn is_exchange_stale(&self, exchange: ExchangeId, now_ns: u64) -> bool {
        const STALE_THRESHOLD_NS: u64 = 5_000_000_000; // 5 seconds
        match &self.exchange_snapshots[exchange.index()] {
            Some(snap) => now_ns.saturating_sub(snap.timestamp_ns) > STALE_THRESHOLD_NS,
            None => true,
        }
    }

    /// Get the best bid/ask for a specific exchange.
    pub fn exchange_bbo(&self, exchange: ExchangeId) -> Option<(i64, i64)> {
        self.exchange_snapshots[exchange.index()]
            .as_ref()
            .map(|s| (s.best_bid_fp, s.best_ask_fp))
    }
}

/// Thread-safe wrapper for GlobalOrderBook, shared between WS ingestion
/// threads and the strategy evaluator.
pub type SharedGlobalBook = Arc<RwLock<GlobalOrderBook>>;

/// Registry of global books for all symbols.
pub struct GlobalBookRegistry {
    books: DashMap<u16, SharedGlobalBook>,
}

impl GlobalBookRegistry {
    pub fn new() -> Self {
        Self {
            books: DashMap::new(),
        }
    }

    pub fn get_or_create(&self, symbol_id: u16) -> SharedGlobalBook {
        self.books
            .entry(symbol_id)
            .or_insert_with(|| Arc::new(RwLock::new(GlobalOrderBook::new(symbol_id))))
            .clone()
    }

    pub fn get(&self, symbol_id: u16) -> Option<SharedGlobalBook> {
        self.books.get(&symbol_id).map(|entry| entry.clone())
    }

    pub fn all_symbol_ids(&self) -> Vec<u16> {
        self.books.iter().map(|entry| *entry.key()).collect()
    }
}

impl Default for GlobalBookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Smart Order Router (SOR)
// ═══════════════════════════════════════════════════════════════════════════

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// A single slice of a split order destined for one exchange.
#[derive(Debug, Clone, Serialize)]
pub struct OrderSlice {
    pub exchange: ExchangeId,
    pub symbol: String,
    pub side: String,
    pub size: i64,       // contracts
    pub price_fp: i64,   // limit price (0 = market)
    pub expected_cost_bps: f64,
    pub is_maker: bool,
}

/// Result of the SOR calculation.
#[derive(Debug, Clone, Serialize)]
pub struct SorResult {
    pub slices: Vec<OrderSlice>,
    pub total_size: i64,
    pub estimated_slippage_bps: f64,
    pub estimated_savings_bps: f64, // vs. single-exchange execution
    pub routing_reason: String,
}

/// Smart Order Router configuration.
#[derive(Debug, Clone)]
pub struct SorConfig {
    /// Minimum order size (USDT) to trigger splitting across exchanges.
    pub min_split_size_usdt: f64,
    /// Maximum number of exchanges to split across (1-3).
    pub max_venues: usize,
    /// Maximum slippage tolerance in basis points.
    pub max_slippage_bps: f64,
    /// Prefer maker orders when spread allows.
    pub prefer_maker: bool,
}

impl Default for SorConfig {
    fn default() -> Self {
        Self {
            min_split_size_usdt: 5000.0,
            max_venues: 3,
            max_slippage_bps: 30.0,
            prefer_maker: true,
        }
    }
}

pub struct SmartOrderRouter {
    config: SorConfig,
}

impl SmartOrderRouter {
    pub fn new(config: SorConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal order routing given the current global book state.
    ///
    /// Algorithm:
    /// 1. If total_size_usdt < min_split_size_usdt -> route entirely to best
    ///    single exchange (lowest effective ask for buys, highest effective bid
    ///    for sells).
    /// 2. Otherwise, sweep the global book level by level, allocating size to
    ///    each exchange until the order is fully filled or max_venues is reached.
    /// 3. Apply fee adjustment: prefer exchanges with maker rebates when the
    ///    order can rest on the book (is_maker=true).
    /// 4. Return SorResult with per-exchange slices.
    pub fn route(
        &self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        mid_price_fp: i64,
        symbol: &str,
    ) -> SorResult {
        // Estimate USDT value of the order
        let price_f64 = mid_price_fp as f64 / FixedPrice::PRECISION as f64;
        let size_usdt = (total_size as f64 / 1e8) * price_f64;

        // If order is small, route to best single exchange
        if size_usdt < self.config.min_split_size_usdt {
            return self.route_single_best(book, side, total_size, symbol);
        }

        // Multi-venue routing: sweep the global book
        let levels = match side {
            OrderSide::Buy => &book.global_asks,
            OrderSide::Sell => &book.global_bids,
        };

        let mut remaining = total_size;
        let mut slices_map: HashMap<ExchangeId, (i64, i64, f64)> = HashMap::new(); // (size, price_fp, cost_bps)
        let mut total_cost_bps = 0.0;
        let mut exchanges_used = 0;

        for level in levels {
            if remaining <= 0 || exchanges_used >= self.config.max_venues {
                break;
            }

            let fill_size = remaining.min(level.qty);
            if fill_size <= 0 {
                continue;
            }

            let cost_bps = level.exchange.taker_fee_bps() as f64
                + level.latency_penalty_bps as f64 * 0.01;

            let entry = slices_map
                .entry(level.exchange)
                .or_insert((0, level.raw_price_fp, cost_bps));
            entry.0 += fill_size;

            remaining -= fill_size;
            total_cost_bps += cost_bps * (fill_size as f64 / total_size as f64);

            if !slices_map.contains_key(&level.exchange) {
                exchanges_used += 1;
            }
        }

        // Build slices
        let slices: Vec<OrderSlice> = slices_map
            .into_iter()
            .map(|(exchange, (size, price_fp, cost_bps))| OrderSlice {
                exchange,
                symbol: symbol.to_string(),
                side: match side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                },
                size,
                price_fp,
                expected_cost_bps: cost_bps,
                is_maker: false,
            })
            .collect();

        // Estimate savings vs single-exchange (Gate.io default)
        let single_cost = ExchangeId::GateIo.taker_fee_bps() as f64;
        let savings = single_cost - total_cost_bps;

        SorResult {
            slices,
            total_size: total_size - remaining,
            estimated_slippage_bps: total_cost_bps,
            estimated_savings_bps: savings.max(0.0),
            routing_reason: format!(
                "Multi-venue sweep: {} USDT across {} exchanges",
                size_usdt as i64,
                exchanges_used
            ),
        }
    }

    /// Route to the best single exchange.
    fn route_single_best(
        &self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        symbol: &str,
    ) -> SorResult {
        let best_level = match side {
            OrderSide::Buy => book.best_ask(),
            OrderSide::Sell => book.best_bid(),
        };

        let (exchange, price_fp, cost_bps) = match best_level {
            Some(level) => (
                level.exchange,
                level.raw_price_fp,
                level.exchange.taker_fee_bps() as f64,
            ),
            None => (ExchangeId::GateIo, 0, ExchangeId::GateIo.taker_fee_bps() as f64),
        };

        SorResult {
            slices: vec![OrderSlice {
                exchange,
                symbol: symbol.to_string(),
                side: match side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                },
                size: total_size,
                price_fp,
                expected_cost_bps: cost_bps,
                is_maker: false,
            }],
            total_size,
            estimated_slippage_bps: cost_bps,
            estimated_savings_bps: 0.0,
            routing_reason: format!(
                "Single-venue: {} (best {})",
                exchange.name(),
                if matches!(side, OrderSide::Buy) { "ask" } else { "bid" }
            ),
        }
    }

    /// Route a single-exchange order (fallback when multi-exchange is off
    /// or when size is below the split threshold).
    pub fn route_single(
        &self,
        exchange: ExchangeId,
        side: OrderSide,
        total_size: i64,
        price_fp: i64,
        symbol: &str,
    ) -> SorResult {
        let cost_bps = exchange.taker_fee_bps() as f64;

        SorResult {
            slices: vec![OrderSlice {
                exchange,
                symbol: symbol.to_string(),
                side: match side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                },
                size: total_size,
                price_fp,
                expected_cost_bps: cost_bps,
                is_maker: false,
            }],
            total_size,
            estimated_slippage_bps: cost_bps,
            estimated_savings_bps: 0.0,
            routing_reason: format!("Single-venue fallback: {}", exchange.name()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Funding Rate Arbitrage
// ═══════════════════════════════════════════════════════════════════════════

/// Funding rate data for a single exchange + symbol.
#[derive(Debug, Clone, Serialize)]
pub struct FundingRateData {
    pub exchange: ExchangeId,
    pub symbol: String,
    pub rate: f64,            // e.g. 0.0001 = 0.01%
    pub next_funding_ts: u64, // unix timestamp of next funding
    pub updated_ns: u64,
}

/// A detected funding rate arbitrage opportunity.
#[derive(Debug, Clone, Serialize)]
pub struct FundingArbOpportunity {
    /// Exchange to go SHORT on (high positive funding = shorts receive payment).
    pub short_exchange: ExchangeId,
    /// Exchange to go LONG on (low or negative funding = longs receive payment).
    pub long_exchange: ExchangeId,
    pub symbol: String,
    /// Net funding rate spread (short_rate - long_rate).
    pub net_rate: f64,
    /// Annualized APR of the spread.
    pub annualized_apr: f64,
    /// Estimated net profit after fees (basis points per funding period).
    pub net_profit_bps: f64,
    /// Whether this opportunity is currently actionable.
    pub is_actionable: bool,
}

/// Cross-exchange funding rate arbitrage monitor.
pub struct CrossExchangeFundingArb {
    /// Latest funding rates per exchange per symbol.
    rates: HashMap<(ExchangeId, String), FundingRateData>,
    /// Minimum net rate spread to consider actionable (default: 0.005% = 0.5bps).
    min_net_rate: f64,
    /// Minimum annualized APR to consider actionable (default: 10%).
    min_annualized_apr: f64,
}

impl CrossExchangeFundingArb {
    pub fn new(min_net_rate: f64, min_annualized_apr: f64) -> Self {
        Self {
            rates: HashMap::new(),
            min_net_rate,
            min_annualized_apr,
        }
    }

    /// Update funding rate data for a specific exchange and symbol.
    pub fn update_rate(&mut self, data: FundingRateData) {
        self.rates
            .insert((data.exchange, data.symbol.clone()), data);
    }

    /// Scan all known symbols for cross-exchange funding arbitrage opportunities.
    /// Returns a list of actionable opportunities sorted by net_profit_bps descending.
    pub fn scan_opportunities(&self) -> Vec<FundingArbOpportunity> {
        // Get all unique symbols
        let symbols: std::collections::HashSet<&String> =
            self.rates.values().map(|r| &r.symbol).collect();

        let mut opportunities = Vec::new();

        for symbol in symbols {
            if let Some(opp) = self.check_symbol(symbol) {
                if opp.is_actionable {
                    opportunities.push(opp);
                }
            }
        }

        // Sort by net profit descending
        opportunities.sort_by(|a, b| {
            b.net_profit_bps
                .partial_cmp(&a.net_profit_bps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        opportunities
    }

    /// Check if a specific symbol has an actionable opportunity.
    pub fn check_symbol(&self, symbol: &str) -> Option<FundingArbOpportunity> {
        let exchanges = [ExchangeId::GateIo, ExchangeId::Binance, ExchangeId::Bybit];

        let mut rates_for_symbol: Vec<(ExchangeId, f64)> = Vec::new();

        for ex in exchanges {
            if let Some(data) = self.rates.get(&(ex, symbol.to_string())) {
                rates_for_symbol.push((ex, data.rate));
            }
        }

        if rates_for_symbol.len() < 2 {
            return None;
        }

        // Find max and min rates
        let (max_ex, max_rate) = rates_for_symbol
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?
            .clone();
        let (min_ex, min_rate) = rates_for_symbol
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?
            .clone();

        let net_rate = max_rate - min_rate;
        // Funding happens every 8 hours, so annualized = net_rate * 3 * 365
        let annualized_apr = net_rate * 3.0 * 365.0 * 100.0; // as percentage

        // Net profit = net_rate * 10000 (convert to bps) - round-trip fees
        let round_trip_fees = (max_ex.taker_fee_bps() + min_ex.taker_fee_bps()) as f64 * 2.0;
        let net_profit_bps = net_rate * 10000.0 - round_trip_fees;

        let is_actionable =
            net_rate >= self.min_net_rate && annualized_apr >= self.min_annualized_apr;

        Some(FundingArbOpportunity {
            short_exchange: max_ex, // Short on high funding (receive payment)
            long_exchange: min_ex,  // Long on low funding (pay less)
            symbol: symbol.to_string(),
            net_rate,
            annualized_apr,
            net_profit_bps,
            is_actionable,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-Venue Margin Monitor
// ═══════════════════════════════════════════════════════════════════════════

/// Margin health status for a single exchange.
#[derive(Debug, Clone, Serialize)]
pub struct ExchangeMarginHealth {
    pub exchange: ExchangeId,
    pub available_balance: f64,
    pub total_equity: f64,
    pub unrealized_pnl: f64,
    pub margin_ratio: f64, // available / total_equity
    pub is_healthy: bool,  // margin_ratio > 0.3 (30% threshold)
    pub updated_ns: u64,
}

/// Cross-venue margin imbalance alert.
#[derive(Debug, Clone, Serialize)]
pub struct MarginImbalanceAlert {
    pub critical_exchange: ExchangeId,
    pub margin_ratio: f64,
    pub recommended_transfer_usdt: f64,
    pub source_exchange: ExchangeId, // exchange with excess margin
}

/// Cross-venue margin monitor. Runs on the cold path (every 30 seconds).
pub struct CrossVenueMarginMonitor {
    /// Latest margin health per exchange.
    health: HashMap<ExchangeId, ExchangeMarginHealth>,
    /// Minimum acceptable margin ratio before alert (default: 0.30 = 30%).
    min_margin_ratio: f64,
    /// Critical margin ratio threshold (default: 0.15 = 15%).
    critical_margin_ratio: f64,
}

impl CrossVenueMarginMonitor {
    pub fn new(min_margin_ratio: f64, critical_margin_ratio: f64) -> Self {
        Self {
            health: HashMap::new(),
            min_margin_ratio,
            critical_margin_ratio,
        }
    }

    /// Update margin health for a specific exchange.
    pub fn update_health(&mut self, health: ExchangeMarginHealth) {
        self.health.insert(health.exchange, health);
    }

    /// Check for margin imbalances across all exchanges.
    /// Returns alerts for exchanges below the minimum margin ratio.
    pub fn check_imbalances(&self) -> Vec<MarginImbalanceAlert> {
        let mut alerts = Vec::new();

        // Find exchange with lowest margin ratio
        let critical = self
            .health
            .values()
            .filter(|h| h.margin_ratio < self.min_margin_ratio)
            .min_by(|a, b| {
                a.margin_ratio
                    .partial_cmp(&b.margin_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        // Find exchange with highest margin ratio (potential source for transfer)
        let source = self.health.values().max_by(|a, b| {
            a.margin_ratio
                .partial_cmp(&b.margin_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let (Some(crit), Some(src)) = (critical, source) {
            if crit.exchange != src.exchange {
                // Calculate recommended transfer to bring critical exchange to min_margin_ratio
                let target_balance = crit.total_equity * self.min_margin_ratio;
                let transfer_needed = target_balance - crit.available_balance;

                if transfer_needed > 0.0 {
                    alerts.push(MarginImbalanceAlert {
                        critical_exchange: crit.exchange,
                        margin_ratio: crit.margin_ratio,
                        recommended_transfer_usdt: transfer_needed,
                        source_exchange: src.exchange,
                    });
                }
            }
        }

        alerts
    }

    /// Calculate global delta neutrality.
    /// Returns (total_long_usdt, total_short_usdt, net_delta_usdt).
    pub fn calculate_global_delta(
        &self,
        positions: &HashMap<ExchangeId, Vec<(String, f64, f64)>>, // (symbol, size, price)
    ) -> (f64, f64, f64) {
        let mut total_long = 0.0;
        let mut total_short = 0.0;

        for (_exchange, pos_list) in positions {
            for (_, size, price) in pos_list {
                let notional = size.abs() * price;
                if *size > 0.0 {
                    total_long += notional;
                } else {
                    total_short += notional;
                }
            }
        }

        let net_delta = total_long - total_short;
        (total_long, total_short, net_delta)
    }

    /// Get all exchange health data.
    pub fn all_health(&self) -> Vec<&ExchangeMarginHealth> {
        self.health.values().collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-Exchange WebSocket URLs
// ═══════════════════════════════════════════════════════════════════════════

pub const BINANCE_WS_LIVE: &str = "wss://fstream.binance.com/stream";
pub const BINANCE_WS_TESTNET: &str = "wss://stream.binancefuture.com/stream";

pub const BYBIT_WS_LIVE: &str = "wss://stream.bybit.com/v5/public/linear";
pub const BYBIT_WS_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/public/linear";

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_id_fees() {
        assert_eq!(ExchangeId::GateIo.taker_fee_bps(), 5);
        assert_eq!(ExchangeId::Binance.taker_fee_bps(), 4);
        assert_eq!(ExchangeId::Bybit.taker_fee_bps(), 6);
    }

    #[test]
    fn test_exchange_id_from_str() {
        assert_eq!(ExchangeId::from_str("gateio"), Some(ExchangeId::GateIo));
        assert_eq!(ExchangeId::from_str("BINANCE"), Some(ExchangeId::Binance));
        assert_eq!(ExchangeId::from_str("bybit"), Some(ExchangeId::Bybit));
        assert_eq!(ExchangeId::from_str("unknown"), None);
    }

    #[test]
    fn test_global_order_book_merge() {
        let mut book = GlobalOrderBook::new(1);

        // Add Gate.io snapshot
        let snap1 = ExchangeBookSnapshot {
            exchange: ExchangeId::GateIo,
            symbol_id: 1,
            best_bid_fp: 50000_00000000, // 50000.0
            best_ask_fp: 50010_00000000, // 50010.0
            bid_levels: vec![(50000_00000000, 100)],
            ask_levels: vec![(50010_00000000, 100)],
            sequence: 1,
            timestamp_ns: 1,
        };
        book.update_exchange_snapshot(snap1);

        assert!(book.best_bid().is_some());
        assert!(book.best_ask().is_some());
        assert_eq!(book.global_bids.len(), 1);
        assert_eq!(book.global_asks.len(), 1);
    }

    #[test]
    fn test_funding_arb_detection() {
        let mut arb = CrossExchangeFundingArb::new(0.00005, 10.0);

        // Gate.io: 0.01% funding rate (longs pay shorts)
        arb.update_rate(FundingRateData {
            exchange: ExchangeId::GateIo,
            symbol: "BTC_USDT".to_string(),
            rate: 0.0001,
            next_funding_ts: 0,
            updated_ns: 0,
        });

        // Binance: -0.01% funding rate (shorts pay longs)
        arb.update_rate(FundingRateData {
            exchange: ExchangeId::Binance,
            symbol: "BTC_USDT".to_string(),
            rate: -0.0001,
            next_funding_ts: 0,
            updated_ns: 0,
        });

        let opps = arb.scan_opportunities();
        assert!(!opps.is_empty());
        assert_eq!(opps[0].short_exchange, ExchangeId::GateIo); // Short on high funding
        assert_eq!(opps[0].long_exchange, ExchangeId::Binance); // Long on low funding
    }
}
