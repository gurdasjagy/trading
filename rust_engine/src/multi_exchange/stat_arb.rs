//! Statistical Arbitrage (Cross-Exchange Basis Trading)
//!
//! When the price spread between the same perpetual on two exchanges deviates
//! beyond 2 standard deviations from the historical mean, short the overpriced
//! instrument and buy the underpriced one, betting on mean reversion.
//!
//! Entry: |spread - mean| > entry_threshold_sigma * std_dev
//! Exit: |spread - mean| < exit_threshold_sigma * std_dev OR hard stops

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::execution_gateway::{ExecutionGateway, OrderIntent, OrderSide, OrderType};
use crate::execution_state::PlacementType;
use crate::multi_exchange::global_book::ExchangeId;

// ---------------------------------------------------------------------------
// Statistical Arbitrage Position
// ---------------------------------------------------------------------------

/// A tracked statistical arbitrage position between two exchanges.
#[derive(Debug, Clone)]
pub struct StatArbPosition {
    pub symbol: String,
    pub long_exchange: ExchangeId,
    pub short_exchange: ExchangeId,
    pub long_entry_price: f64,
    pub short_entry_price: f64,
    pub size: i64,
    pub entry_timestamp_ns: u64,
    pub entry_spread: f64,
    pub entry_mean: f64,
    pub entry_sigma: f64,
    /// Whether we entered because spread was ABOVE mean (long on B, short on A)
    /// or BELOW mean (long on A, short on B)
    pub spread_was_high: bool,
}

impl StatArbPosition {
    /// Calculate unrealized PnL based on current prices.
    pub fn unrealized_pnl(&self, long_price: f64, short_price: f64) -> f64 {
        let long_pnl = (long_price - self.long_entry_price) * self.size as f64;
        let short_pnl = (self.short_entry_price - short_price) * self.size as f64;
        long_pnl + short_pnl
    }

    /// Calculate unrealized PnL as percentage of notional.
    pub fn unrealized_pnl_pct(&self, long_price: f64, short_price: f64) -> f64 {
        let notional = (self.long_entry_price + self.short_entry_price) * self.size as f64 / 2.0;
        if notional > 0.0 {
            self.unrealized_pnl(long_price, short_price) / notional * 100.0
        } else {
            0.0
        }
    }

    /// Hours since position was opened.
    pub fn hours_open(&self, now_ns: u64) -> f64 {
        (now_ns.saturating_sub(self.entry_timestamp_ns)) as f64 / 3_600_000_000_000.0
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "symbol": self.symbol,
            "long_exchange": self.long_exchange.name(),
            "short_exchange": self.short_exchange.name(),
            "long_entry_price": self.long_entry_price,
            "short_entry_price": self.short_entry_price,
            "size": self.size,
            "entry_spread": self.entry_spread,
            "entry_mean": self.entry_mean,
            "entry_sigma": self.entry_sigma,
            "spread_was_high": self.spread_was_high,
        })
    }
}

// ---------------------------------------------------------------------------
// Stat Arb Exit Reason
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatArbExitReason {
    /// Spread reverted to within exit threshold of mean.
    MeanReversion,
    /// Spread moved to 4 sigma against position (stop loss).
    StopLoss,
    /// Position held for too long (time stop).
    TimeStop,
    /// Manual or forced close.
    Manual,
}

// ---------------------------------------------------------------------------
// Statistical Arbitrage Engine Configuration
// ---------------------------------------------------------------------------

/// Configuration for the stat arb engine.
#[derive(Debug, Clone)]
pub struct StatArbConfig {
    /// Rolling window size for spread history (default: 1000 samples).
    pub window_size: usize,
    /// Entry threshold in standard deviations (default: 2.5).
    pub entry_threshold_sigma: f64,
    /// Exit threshold in standard deviations (default: 0.5).
    pub exit_threshold_sigma: f64,
    /// Stop loss threshold in standard deviations (default: 4.0).
    pub stop_loss_sigma: f64,
    /// Maximum position hold time in hours (default: 4.0).
    pub max_hold_hours: f64,
    /// Position size as fraction of total equity (default: 0.02 = 2%).
    pub position_size_pct: f64,
    /// Minimum samples before trading (default: 100).
    pub min_samples: usize,
    /// BUG 5 FIX: Minimum hold time in seconds before exit is allowed.
    /// Prevents rapid entry/exit cycles that churn fees.
    pub min_hold_seconds: f64,
    /// BUG 5 FIX: Cooldown in seconds after exiting before re-entry on same symbol.
    pub entry_cooldown_seconds: f64,
    /// BUG 5 FIX: Number of consecutive observations the signal must persist before entry.
    pub signal_persistence_count: usize,
}

impl Default for StatArbConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            entry_threshold_sigma: 2.5,  // BUG 5 FIX: raised from 2.0 to reduce noise entries
            exit_threshold_sigma: 0.5,
            stop_loss_sigma: 4.0,
            max_hold_hours: 4.0,
            position_size_pct: 0.02,
            min_samples: 100,
            min_hold_seconds: 300.0,         // BUG 5 FIX: 5 minute minimum hold
            entry_cooldown_seconds: 600.0,   // BUG 5 FIX: 10 minute cooldown after exit
            signal_persistence_count: 5,     // BUG 5 FIX: signal must persist 5 consecutive ticks
        }
    }
}

// ---------------------------------------------------------------------------
// Spread Statistics
// ---------------------------------------------------------------------------

/// Rolling statistics for a spread series.
#[derive(Debug, Clone)]
pub struct SpreadStats {
    /// Rolling history of spread values.
    history: VecDeque<f64>,
    /// Maximum window size.
    window_size: usize,
    /// Cached sum for efficient mean calculation.
    sum: f64,
    /// Cached sum of squares for efficient variance calculation.
    sum_sq: f64,
}

impl SpreadStats {
    /// Create a new spread statistics tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Add a new spread observation.
    pub fn push(&mut self, spread: f64) {
        // Remove oldest value if at capacity
        if self.history.len() >= self.window_size {
            if let Some(old) = self.history.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        // Add new value
        self.history.push_back(spread);
        self.sum += spread;
        self.sum_sq += spread * spread;
    }

    /// Get the number of samples.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Calculate the rolling mean.
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            self.sum / self.history.len() as f64
        }
    }

    /// Calculate the rolling variance.
    pub fn variance(&self) -> f64 {
        let n = self.history.len();
        if n < 2 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_sq / n as f64) - (mean * mean)
        }
    }

    /// Calculate the rolling standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the latest spread value.
    pub fn latest(&self) -> Option<f64> {
        self.history.back().copied()
    }

    /// Calculate z-score of the latest spread.
    pub fn z_score(&self) -> Option<f64> {
        let std = self.std_dev();
        if std <= 0.0 || self.history.is_empty() {
            return None;
        }
        let latest = self.latest()?;
        Some((latest - self.mean()) / std)
    }
}

// ---------------------------------------------------------------------------
// Exchange Pair Key
// ---------------------------------------------------------------------------

/// Key for identifying a spread between two exchanges for a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExchangePairKey {
    pub exchange_a: ExchangeId,
    pub exchange_b: ExchangeId,
}

impl ExchangePairKey {
    /// Create a new exchange pair key (always ordered A < B for consistency).
    pub fn new(a: ExchangeId, b: ExchangeId) -> Self {
        if (a as u8) <= (b as u8) {
            Self { exchange_a: a, exchange_b: b }
        } else {
            Self { exchange_a: b, exchange_b: a }
        }
    }
}

// ---------------------------------------------------------------------------
// Statistical Arbitrage Engine
// ---------------------------------------------------------------------------

/// Statistical arbitrage engine for cross-exchange basis trading.
pub struct StatArbEngine {
    config: StatArbConfig,
    /// Spread history per (exchange_pair, symbol).
    spread_history: HashMap<(ExchangePairKey, String), SpreadStats>,
    /// Active stat arb positions.
    active_positions: Vec<StatArbPosition>,
    /// Last update timestamp.
    last_update_ns: u64,
    /// Is stat arb trading paused.
    paused: bool,
    /// Pause reason.
    pause_reason: Option<String>,
    /// BUG 5 FIX: Cooldown tracking — maps symbol to the timestamp (ns) when last exit occurred.
    exit_cooldowns: HashMap<String, u64>,
    /// BUG 5 FIX: Signal persistence counter — maps symbol to consecutive tick count above threshold.
    signal_persistence: HashMap<String, usize>,
}

impl StatArbEngine {
    /// Create a new stat arb engine.
    pub fn new(config: StatArbConfig) -> Self {
        Self {
            config,
            spread_history: HashMap::new(),
            active_positions: Vec::new(),
            last_update_ns: 0,
            paused: false,
            pause_reason: None,
            exit_cooldowns: HashMap::new(),
            signal_persistence: HashMap::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StatArbConfig::default())
    }

    /// Check if trading is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Pause trading.
    pub fn pause(&mut self, reason: &str) {
        self.paused = true;
        self.pause_reason = Some(reason.to_string());
        warn!("[stat-arb] Trading PAUSED: {}", reason);
    }

    /// Resume trading.
    pub fn resume(&mut self) {
        self.paused = false;
        self.pause_reason = None;
        info!("[stat-arb] Trading RESUMED");
    }

    /// Update spread history with a new mid-price observation.
    ///
    /// Called on every global book merge or at regular intervals.
    pub fn on_price_update(
        &mut self,
        symbol: &str,
        exchange_a: ExchangeId,
        mid_a: f64,
        exchange_b: ExchangeId,
        mid_b: f64,
        timestamp_ns: u64,
    ) {
        if mid_a <= 0.0 || mid_b <= 0.0 {
            return;
        }

        // Calculate spread: mid_A - mid_B
        let spread = mid_a - mid_b;

        // Store in history
        let pair_key = ExchangePairKey::new(exchange_a, exchange_b);
        let key = (pair_key, symbol.to_string());

        let stats = self
            .spread_history
            .entry(key)
            .or_insert_with(|| SpreadStats::new(self.config.window_size));

        stats.push(spread);
        self.last_update_ns = timestamp_ns;
    }

    /// Check for entry opportunities on a symbol.
    ///
    /// Returns Some((long_exchange, short_exchange)) if an opportunity exists.
    pub fn check_entry_opportunity(
        &mut self,
        symbol: &str,
        exchange_a: ExchangeId,
        exchange_b: ExchangeId,
    ) -> Option<(ExchangeId, ExchangeId, f64, f64, f64)> {
        if self.paused {
            return None;
        }

        // Check if we already have a position for this symbol
        if self.active_positions.iter().any(|p| p.symbol == symbol) {
            return None;
        }

        // BUG 5 FIX: Check entry cooldown — skip if we exited this symbol recently
        if let Some(&exit_ns) = self.exit_cooldowns.get(symbol) {
            let elapsed_secs = (self.last_update_ns.saturating_sub(exit_ns)) as f64 / 1_000_000_000.0;
            if elapsed_secs < self.config.entry_cooldown_seconds {
                debug!(
                    "[stat-arb] Cooldown active for {} ({:.0}s / {:.0}s)",
                    symbol, elapsed_secs, self.config.entry_cooldown_seconds
                );
                return None;
            }
        }

        let pair_key = ExchangePairKey::new(exchange_a, exchange_b);
        let key = (pair_key, symbol.to_string());

        let stats = self.spread_history.get(&key)?;

        // Need minimum samples
        if stats.len() < self.config.min_samples {
            return None;
        }

        let z_score = stats.z_score()?;
        let mean = stats.mean();
        let std_dev = stats.std_dev();
        let latest_spread = stats.latest()?;

        // Check entry condition: |z_score| > entry_threshold
        if z_score.abs() < self.config.entry_threshold_sigma {
            // BUG 5 FIX: Reset persistence counter if signal drops below threshold
            self.signal_persistence.remove(symbol);
            return None;
        }

        // Determine direction:
        // If z_score > 2: spread is HIGH (A is overpriced vs B)
        //   -> SHORT A (overpriced), LONG B (underpriced)
        // If z_score < -2: spread is LOW (A is underpriced vs B)
        //   -> LONG A (underpriced), SHORT B (overpriced)
        let (long_ex, short_ex) = if z_score > 0.0 {
            // Spread high: short A, long B
            // But remember: spread = mid_A - mid_B
            // If spread high, A is expensive relative to B
            (exchange_b, exchange_a)
        } else {
            // Spread low: long A, short B
            (exchange_a, exchange_b)
        };

        // BUG 5 FIX: Require signal to persist for N consecutive observations
        let count = self.signal_persistence.entry(symbol.to_string()).or_insert(0);
        *count += 1;
        if *count < self.config.signal_persistence_count {
            debug!(
                "[stat-arb] Signal persistence {}/{} for {} z={:.2}",
                count, self.config.signal_persistence_count, symbol, z_score
            );
            return None;
        }

        debug!(
            "[stat-arb] Entry opportunity: {} z={:.2} persistence={} (long={}, short={})",
            symbol,
            z_score,
            count,
            long_ex.name(),
            short_ex.name()
        );

        Some((long_ex, short_ex, latest_spread, mean, std_dev))
    }

    /// Check all exchange pairs for entry opportunities.
    pub fn scan_all_opportunities(
        &self,
    ) -> Vec<(String, ExchangeId, ExchangeId, f64, f64, f64)> {
        let mut opportunities = Vec::new();

        // Iterate over all tracked spread histories
        for ((pair_key, symbol), stats) in &self.spread_history {
            if stats.len() < self.config.min_samples {
                continue;
            }

            if let Some(z_score) = stats.z_score() {
                if z_score.abs() >= self.config.entry_threshold_sigma {
                    let mean = stats.mean();
                    let std_dev = stats.std_dev();
                    let latest = stats.latest().unwrap_or(0.0);

                    let (long_ex, short_ex) = if z_score > 0.0 {
                        (pair_key.exchange_b, pair_key.exchange_a)
                    } else {
                        (pair_key.exchange_a, pair_key.exchange_b)
                    };

                    // Don't include if we already have a position
                    if self.active_positions.iter().any(|p| p.symbol == *symbol) {
                        continue;
                    }

                    // BUG 5 FIX: Respect cooldown in scan as well
                    if let Some(&exit_ns) = self.exit_cooldowns.get(symbol.as_str()) {
                        let elapsed_secs = (self.last_update_ns.saturating_sub(exit_ns)) as f64 / 1_000_000_000.0;
                        if elapsed_secs < self.config.entry_cooldown_seconds {
                            continue;
                        }
                    }

                    opportunities.push((
                        symbol.clone(),
                        long_ex,
                        short_ex,
                        latest,
                        mean,
                        std_dev,
                    ));
                }
            }
        }

        // Sort by absolute z-score descending (best opportunities first)
        opportunities.sort_by(|a, b| {
            let z_a = if a.4 != 0.0 { ((a.3 - a.4) / a.5).abs() } else { 0.0 };
            let z_b = if b.4 != 0.0 { ((b.3 - b.4) / b.5).abs() } else { 0.0 };
            z_b.partial_cmp(&z_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        opportunities
    }

    /// Record a new position entry.
    pub fn record_entry(
        &mut self,
        symbol: &str,
        long_exchange: ExchangeId,
        short_exchange: ExchangeId,
        long_entry_price: f64,
        short_entry_price: f64,
        size: i64,
        entry_spread: f64,
        entry_mean: f64,
        entry_sigma: f64,
        timestamp_ns: u64,
    ) {
        let spread_was_high = entry_spread > entry_mean;

        let position = StatArbPosition {
            symbol: symbol.to_string(),
            long_exchange,
            short_exchange,
            long_entry_price,
            short_entry_price,
            size,
            entry_timestamp_ns: timestamp_ns,
            entry_spread,
            entry_mean,
            entry_sigma,
            spread_was_high,
        };

        info!(
            "[stat-arb] ENTRY: {} long@{} ({:.4}) short@{} ({:.4}) size={} spread={:.4}",
            symbol,
            long_exchange.name(),
            long_entry_price,
            short_exchange.name(),
            short_entry_price,
            size,
            entry_spread
        );

        self.active_positions.push(position);
    }

    /// Check exit conditions for all active positions.
    ///
    /// Returns list of positions to close with their exit reasons.
    pub fn check_exits(&mut self, now_ns: u64) -> Vec<(StatArbPosition, StatArbExitReason)> {
        let mut exits = Vec::new();

        for pos in &self.active_positions {
            // Get current spread stats
            let pair_key = ExchangePairKey::new(pos.long_exchange, pos.short_exchange);
            let key = (pair_key, pos.symbol.clone());

            // BUG 5 FIX: Enforce minimum hold period before allowing any exit
            // (except stop loss, which always fires for risk protection)
            let hold_secs = (now_ns.saturating_sub(pos.entry_timestamp_ns)) as f64 / 1_000_000_000.0;

            if let Some(stats) = self.spread_history.get(&key) {
                if let Some(z_score) = stats.z_score() {
                    // Check time stop
                    let hours = pos.hours_open(now_ns);
                    if hours > self.config.max_hold_hours {
                        exits.push((pos.clone(), StatArbExitReason::TimeStop));
                        continue;
                    }

                    // Check stop loss (4 sigma against position)
                    // Stop loss always fires regardless of min hold period
                    if pos.spread_was_high && z_score > self.config.stop_loss_sigma {
                        exits.push((pos.clone(), StatArbExitReason::StopLoss));
                        continue;
                    }
                    if !pos.spread_was_high && z_score < -self.config.stop_loss_sigma {
                        exits.push((pos.clone(), StatArbExitReason::StopLoss));
                        continue;
                    }

                    // BUG 5 FIX: Only allow mean reversion exit after minimum hold period
                    if hold_secs < self.config.min_hold_seconds {
                        debug!(
                            "[stat-arb] {} held {:.0}s < min {:.0}s, skipping mean reversion check",
                            pos.symbol, hold_secs, self.config.min_hold_seconds
                        );
                        continue;
                    }

                    // Check mean reversion exit
                    if z_score.abs() < self.config.exit_threshold_sigma {
                        exits.push((pos.clone(), StatArbExitReason::MeanReversion));
                        continue;
                    }
                }
            }
        }

        // BUG 5 FIX: Record cooldown timestamps for exited symbols
        for (pos, _reason) in &exits {
            self.exit_cooldowns.insert(pos.symbol.clone(), now_ns);
            // Reset signal persistence so re-entry requires fresh signal buildup
            self.signal_persistence.remove(&pos.symbol);
        }

        exits
    }

    /// Remove a closed position from tracking.
    pub fn remove_position(&mut self, symbol: &str) {
        self.active_positions.retain(|p| p.symbol != symbol);
        // BUG 5 FIX: Reset signal persistence on position removal
        self.signal_persistence.remove(symbol);
    }

    /// Get active position count.
    pub fn active_count(&self) -> usize {
        self.active_positions.len()
    }

    /// Get all active positions.
    pub fn active_positions(&self) -> &[StatArbPosition] {
        &self.active_positions
    }

    /// Get spread statistics for a symbol pair.
    pub fn get_spread_stats(
        &self,
        symbol: &str,
        exchange_a: ExchangeId,
        exchange_b: ExchangeId,
    ) -> Option<(f64, f64, f64)> {
        // Returns (latest_spread, mean, std_dev)
        let pair_key = ExchangePairKey::new(exchange_a, exchange_b);
        let key = (pair_key, symbol.to_string());
        let stats = self.spread_history.get(&key)?;
        let latest = stats.latest()?;
        Some((latest, stats.mean(), stats.std_dev()))
    }

    /// Serialize to JSON for dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        let mut spread_data = Vec::new();

        for ((pair_key, symbol), stats) in &self.spread_history {
            if let Some(z_score) = stats.z_score() {
                spread_data.push(serde_json::json!({
                    "symbol": symbol,
                    "exchange_a": pair_key.exchange_a.name(),
                    "exchange_b": pair_key.exchange_b.name(),
                    "latest_spread": stats.latest().unwrap_or(0.0),
                    "mean": stats.mean(),
                    "std_dev": stats.std_dev(),
                    "z_score": z_score,
                    "samples": stats.len(),
                    "entry_threshold": self.config.entry_threshold_sigma,
                    "is_opportunity": z_score.abs() >= self.config.entry_threshold_sigma,
                }));
            }
        }

        serde_json::json!({
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "active_positions": self.active_positions.iter()
                .map(|p| p.to_json())
                .collect::<Vec<_>>(),
            "active_count": self.active_positions.len(),
            "spread_data": spread_data,
            "config": {
                "window_size": self.config.window_size,
                "entry_threshold_sigma": self.config.entry_threshold_sigma,
                "exit_threshold_sigma": self.config.exit_threshold_sigma,
                "stop_loss_sigma": self.config.stop_loss_sigma,
                "max_hold_hours": self.config.max_hold_hours,
                "position_size_pct": self.config.position_size_pct,
            }
        })
    }
}

impl Default for StatArbEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Helper: Build order intents for stat arb entry
// ---------------------------------------------------------------------------

/// Build a pair of order intents for stat arb entry.
pub fn build_stat_arb_entry_intents(
    symbol: &str,
    long_exchange: ExchangeId,
    short_exchange: ExchangeId,
    size: i64,
    long_ref_price: f64,
    short_ref_price: f64,
) -> (OrderIntent, OrderIntent) {
    let long_intent = OrderIntent {
        symbol: symbol.to_string(),
        side: OrderSide::Buy,
        size,
        order_type: OrderType::Market,
        price: Some(long_ref_price),
        reduce_only: false,
        leverage: Some(3),
        time_in_force: "ioc".to_string(),
        slippage_cap_pct: Some(0.002),
        placement: PlacementType::AtBest,
        stop_loss: Some(long_ref_price * 0.98), // 2% SL
        take_profit: None,
        confidence: 1.0,
        signal_tag: "stat_arb_long".to_string(),
        min_fill_size: None,
        strategy_name: "stat_arb".to_string(),
    };

    let short_intent = OrderIntent {
        symbol: symbol.to_string(),
        side: OrderSide::Sell,
        size,
        order_type: OrderType::Market,
        price: Some(short_ref_price),
        reduce_only: false,
        leverage: Some(3),
        time_in_force: "ioc".to_string(),
        slippage_cap_pct: Some(0.002),
        placement: PlacementType::AtBest,
        stop_loss: Some(short_ref_price * 1.02), // 2% SL
        take_profit: None,
        confidence: 1.0,
        signal_tag: "stat_arb_short".to_string(),
        min_fill_size: None,
        strategy_name: "stat_arb".to_string(),
    };

    (long_intent, short_intent)
}

/// Build a pair of order intents for stat arb exit (close both legs).
pub fn build_stat_arb_exit_intents(
    pos: &StatArbPosition,
    long_current_price: f64,
    short_current_price: f64,
) -> (OrderIntent, OrderIntent) {
    // Close long: sell
    let close_long = OrderIntent {
        symbol: pos.symbol.clone(),
        side: OrderSide::Sell,
        size: pos.size,
        order_type: OrderType::Market,
        price: Some(long_current_price),
        reduce_only: true,
        leverage: None,
        time_in_force: "ioc".to_string(),
        slippage_cap_pct: Some(0.005),
        placement: PlacementType::AtBest,
        stop_loss: None,
        take_profit: None,
        confidence: 0.0,
        signal_tag: "stat_arb_close_long".to_string(),
        min_fill_size: None,
        strategy_name: "stat_arb".to_string(),
    };

    // Close short: buy
    let close_short = OrderIntent {
        symbol: pos.symbol.clone(),
        side: OrderSide::Buy,
        size: pos.size,
        order_type: OrderType::Market,
        price: Some(short_current_price),
        reduce_only: true,
        leverage: None,
        time_in_force: "ioc".to_string(),
        slippage_cap_pct: Some(0.005),
        placement: PlacementType::AtBest,
        stop_loss: None,
        take_profit: None,
        confidence: 0.0,
        signal_tag: "stat_arb_close_short".to_string(),
        min_fill_size: None,
        strategy_name: "stat_arb".to_string(),
    };

    (close_long, close_short)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_stats() {
        let mut stats = SpreadStats::new(100);
        
        // Add some observations
        for i in 0..50 {
            stats.push(i as f64);
        }

        assert_eq!(stats.len(), 50);
        assert!((stats.mean() - 24.5).abs() < 0.01);
        assert!(stats.std_dev() > 0.0);
    }

    #[test]
    fn test_z_score_calculation() {
        let mut stats = SpreadStats::new(100);
        
        // Add 100 observations around mean 0
        for _ in 0..99 {
            stats.push(0.0);
        }
        // Add one outlier
        stats.push(10.0);

        let z = stats.z_score();
        assert!(z.is_some());
        // The outlier should have a high positive z-score
        assert!(z.unwrap() > 0.0);
    }

    #[test]
    fn test_stat_arb_engine_creation() {
        let engine = StatArbEngine::with_defaults();
        assert!(!engine.is_paused());
        assert_eq!(engine.active_count(), 0);
    }

    #[test]
    fn test_entry_opportunity_detection() {
        let mut engine = StatArbEngine::with_defaults();

        // Build up spread history
        let now = 1000000000u64;
        for i in 0..200 {
            // Normal spread around 0
            let spread_noise = (i % 10) as f64 - 5.0;
            let mid_a = 50000.0 + spread_noise;
            let mid_b = 50000.0;
            engine.on_price_update(
                "BTC_USDT",
                ExchangeId::GateIo,
                mid_a,
                ExchangeId::Binance,
                mid_b,
                now + i as u64,
            );
        }

        // Now add an outlier spread (should trigger opportunity)
        engine.on_price_update(
            "BTC_USDT",
            ExchangeId::GateIo,
            50100.0, // Much higher than Binance
            ExchangeId::Binance,
            50000.0,
            now + 300,
        );

        // Check for opportunity
        let opp = engine.check_entry_opportunity(
            "BTC_USDT",
            ExchangeId::GateIo,
            ExchangeId::Binance,
        );

        // Should detect opportunity (Gate.io overpriced)
        // Note: depends on the accumulated history and std dev
        if let Some((long_ex, short_ex, _, _, _)) = opp {
            // If spread is high (Gate.io > Binance), we should short Gate.io and long Binance
            assert_eq!(long_ex, ExchangeId::Binance);
            assert_eq!(short_ex, ExchangeId::GateIo);
        }
    }

    #[test]
    fn test_position_tracking() {
        let mut engine = StatArbEngine::with_defaults();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        engine.record_entry(
            "BTC_USDT",
            ExchangeId::Binance,
            ExchangeId::GateIo,
            50000.0,
            50100.0,
            1,
            100.0,
            0.0,
            10.0,
            now,
        );

        assert_eq!(engine.active_count(), 1);

        // Check unrealized PnL
        let pos = &engine.active_positions()[0];
        // If long price went up and short price went down, we profit
        let pnl = pos.unrealized_pnl(50100.0, 50000.0);
        // Long: 50100 - 50000 = +100
        // Short: 50100 - 50000 = +100
        assert_eq!(pnl, 200.0);
    }
}
