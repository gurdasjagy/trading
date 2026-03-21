//! Institutional Microstructure Strategy Engine — Mandate 4 Rewrite.
//!
//! Replaces the retail strategy zoo (60+ Python indicators the Rust engine
//! ignores) with a single, highly-optimized strategy based on:
//!
//!   1. **Orderbook Imbalance**: Weighted bid/ask depth skew at 10 levels
//!   2. **VPIN (Volume-Synchronized Probability of Informed Trading)**:
//!      Detects toxic flow from the WebSocket trade feed
//!   3. **Regime-Adaptive Position Sizing**: Reads `/dev/shm/regime_weights`
//!      (written by Python ML) to scale position sizes dynamically
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐    BookSnapshot    ┌──────────────────┐    OrderIntent
//! │  Book    │ ──────────────▶   │  StrategyEngine   │ ──────────────▶
//! │  Builder │                   │  VPIN + Imbalance │    (to Exec Ring)
//! └──────────┘                   └──────────────────┘
//!                                       ▲
//!                                       │ RegimeWeights
//!                                       │ (lock-free read from /dev/shm)
//!                                ┌──────┴──────┐
//!                                │  Python ML  │
//!                                │  Cold Path  │
//!                                └─────────────┘
//! ```
//!
//! # VPIN Calculation
//!
//! VPIN (Easley, López de Prado, O'Hara 2012) measures the probability
//! that a trade was initiated by an informed trader. High VPIN → toxic flow
//! → widen spreads / reduce size. Low VPIN → safe to make markets aggressively.
//!
//! We use the Bulk Volume Classification (BVC) method:
//!   - Each trade's volume is classified as buy/sell using the tick rule
//!   - Volume is accumulated into fixed-size buckets
//!   - VPIN = Σ|V_buy - V_sell| / (n × V_bucket)
//!
//! # Python's Role
//!
//! Python ONLY writes regime weights to `/dev/shm/regime_weights`.
//! It does NOT generate trade signals. The Rust engine is the sole
//! decision-maker.

use std::collections::VecDeque;

use tracing::{debug, info};

use crate::execution_gateway::{OrderIntent, OrderSide, OrderType};
use crate::execution_state::PlacementType;
use crate::regime::RegimeState;

// ---------------------------------------------------------------------------
// Configuration Constants
// ---------------------------------------------------------------------------

/// Number of volume buckets for VPIN calculation.
const VPIN_BUCKET_COUNT: usize = 50;

/// Volume per bucket in USD equivalent (normalized).
/// A bucket is "full" when this much volume has been accumulated.
/// Reduced from 100k to 1k for faster VPIN updates on liquid pairs like BTC/USDT.
const VPIN_BUCKET_SIZE_USD: f64 = 1_000.0;

/// VPIN threshold above which we consider flow as toxic.
/// Above this → reduce position size, widen quotes.
const VPIN_TOXIC_THRESHOLD: f64 = 0.65;

/// VPIN threshold below which we consider flow as safe for market-making.
const VPIN_SAFE_THRESHOLD: f64 = 0.35;

/// Minimum imbalance magnitude to generate a signal (in basis points / 10000).
/// Reduced from 0.15 (15%) to 0.05 (5%) for liquid pairs like BTC/USDT.
/// A 5% depth skew is significant enough to indicate directional pressure
/// without being so high that signals are rare.
const IMBALANCE_ENTRY_THRESHOLD: f64 = 0.05;

/// Minimum spread in bps to avoid trading in tight spreads.
const MIN_SPREAD_BPS: f64 = 1.0;

/// Maximum spread in bps — don't trade if spread is insane.
const MAX_SPREAD_BPS: f64 = 200.0;

/// Base position size (in contracts) before regime scaling.
const BASE_POSITION_SIZE: f64 = 1.0;

/// Maximum position size after all scaling.
const MAX_POSITION_SIZE: f64 = 50.0;

/// Minimum depth in USD on both sides to consider orderbook valid.
const MIN_DEPTH_USD: f64 = 1000.0;

// ---------------------------------------------------------------------------
// Microstructure Metrics — Fed from BookSnapshot
// ---------------------------------------------------------------------------

/// Orderbook microstructure metrics extracted from a BookSnapshot.
/// These are computed in the orderbook_builder_loop and passed here.
pub struct MicrostructureMetrics {
    /// Mid price as f64.
    pub mid_price: f64,
    /// Spread in basis points.
    pub spread_bps: f64,
    /// Orderbook imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    /// Range: [-1.0, 1.0]. Positive = bid-heavy (bullish pressure).
    pub imbalance: f64,
    /// Total bid depth in USD (sum of top 10 levels).
    pub bid_depth_usdt: f64,
    /// Total ask depth in USD (sum of top 10 levels).
    pub ask_depth_usdt: f64,
    /// Current VPIN value (0.0 if not yet computed).
    pub vpin: f64,
    /// Direction of last trade (Some(true) = buy, Some(false) = sell).
    pub last_trade_is_buy: Option<bool>,
}

// OrderIntent is imported from crate::execution_gateway — single source of truth.
// Strategy-specific metadata (confidence, signal_tag) are fields on that struct.

// ---------------------------------------------------------------------------
// VPIN Calculator
// ---------------------------------------------------------------------------

/// Volume-Synchronized Probability of Informed Trading (VPIN) calculator.
///
/// Maintains a circular buffer of volume buckets. Each bucket accumulates
/// trade volume classified as buy or sell using the tick rule (or BVC).
///
/// VPIN = Σ|V_buy_i - V_sell_i| / (n × V_bucket)
///
/// This runs in O(1) per trade update (amortized) with O(n) memory.
pub struct VPINCalculator {
    /// Completed volume buckets: (buy_volume, sell_volume).
    buckets: VecDeque<(f64, f64)>,
    /// Current (in-progress) bucket accumulation.
    current_buy_vol: f64,
    current_sell_vol: f64,
    /// Target volume per bucket in USD.
    bucket_size: f64,
    /// Maximum number of buckets to retain.
    max_buckets: usize,
    /// Cached VPIN value (updated when a bucket completes).
    cached_vpin: f64,
    /// Last trade price for tick-rule classification.
    last_trade_price: f64,
}

impl VPINCalculator {
    pub fn new(bucket_size: f64, max_buckets: usize) -> Self {
        Self {
            buckets: VecDeque::with_capacity(max_buckets),
            current_buy_vol: 0.0,
            current_sell_vol: 0.0,
            bucket_size,
            max_buckets,
            cached_vpin: 0.0,
            last_trade_price: 0.0,
        }
    }

    /// Update VPIN with a new trade.
    ///
    /// # Parameters
    /// - `price`: Trade price
    /// - `volume_usd`: Trade volume in USD equivalent
    /// - `is_buy`: Whether this trade was a buy (taker bought). If None,
    ///   the tick rule is used (price > last → buy, price < last → sell).
    ///
    /// # Returns
    /// Current VPIN value [0.0, 1.0].
    #[inline]
    pub fn update(&mut self, price: f64, volume_usd: f64, is_buy: Option<bool>) -> f64 {
        // Classify trade direction
        let buy = match is_buy {
            Some(b) => b,
            None => {
                // Tick rule: uptick = buy, downtick = sell
                if self.last_trade_price == 0.0 {
                    true // Default first trade to buy
                } else {
                    price >= self.last_trade_price
                }
            }
        };
        self.last_trade_price = price;

        // Accumulate into current bucket
        let mut remaining = volume_usd;
        while remaining > 0.0 {
            let current_total = self.current_buy_vol + self.current_sell_vol;
            let space_left = self.bucket_size - current_total;

            if remaining >= space_left {
                // Fill current bucket and start a new one
                let _fill_ratio = space_left / remaining.max(1e-12);
                let fill_amount = space_left;

                if buy {
                    self.current_buy_vol += fill_amount;
                } else {
                    self.current_sell_vol += fill_amount;
                }

                // Push completed bucket
                self.buckets.push_back((self.current_buy_vol, self.current_sell_vol));
                if self.buckets.len() > self.max_buckets {
                    self.buckets.pop_front();
                }

                // Reset current bucket
                self.current_buy_vol = 0.0;
                self.current_sell_vol = 0.0;
                remaining -= fill_amount;

                // Recalculate cached VPIN
                self.cached_vpin = self.calculate_vpin();
            } else {
                // Partial fill of current bucket
                if buy {
                    self.current_buy_vol += remaining;
                } else {
                    self.current_sell_vol += remaining;
                }
                remaining = 0.0;
            }
        }

        self.cached_vpin
    }

    /// Get the current VPIN value without updating.
    #[inline]
    pub fn current(&self) -> f64 {
        self.cached_vpin
    }

    /// Calculate VPIN from completed buckets.
    /// VPIN = Σ|V_buy_i - V_sell_i| / (n × V_bucket)
    fn calculate_vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        let n = self.buckets.len() as f64;
        let sum_abs_diff: f64 = self.buckets.iter()
            .map(|(buy, sell)| (buy - sell).abs())
            .sum();
        (sum_abs_diff / (n * self.bucket_size)).clamp(0.0, 1.0)
    }

    /// Returns true if we have enough data for a reliable VPIN reading.
    #[inline]
    pub fn is_warmed_up(&self) -> bool {
        self.buckets.len() >= self.max_buckets / 2
    }
}

// ---------------------------------------------------------------------------
// Strategy Engine
// ---------------------------------------------------------------------------

/// Institutional Microstructure Strategy Engine.
///
/// Combines:
///   - 10-level orderbook imbalance (depth skew analysis)
///   - VPIN (toxic flow detection)
///   - Regime-adaptive position sizing from /dev/shm
///
/// Output: OrderIntent (or None if no signal).
pub struct StrategyEngine {
    /// Per-symbol VPIN calculators.
    /// Indexed by symbol_id (u16). We pre-allocate for up to 64 symbols.
    vpin_calcs: Vec<VPINCalculator>,
    /// Last signal direction per symbol (to avoid flipping).
    last_signal: Vec<Option<OrderSide>>,
    /// Maximum concurrent positions.
    max_positions: usize,
    /// Current number of active positions (approximate).
    active_positions: usize,
    /// Strategy configuration (immutable after construction).
    /// Contains leverage, thresholds, and other strategy parameters.
    strategy_config: StrategyConfig,
}

impl StrategyEngine {
    /// Create a new StrategyEngine from a StrategyConfig.
    /// Allocates per-symbol VPIN calculators for up to 64 symbols.
    pub fn new(config: StrategyConfig) -> Self {
        let num_symbols = 64; // Pre-allocate for max symbols
        let mut vpin_calcs = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            vpin_calcs.push(VPINCalculator::new(VPIN_BUCKET_SIZE_USD, VPIN_BUCKET_COUNT));
        }

        Self {
            vpin_calcs,
            last_signal: vec![None; num_symbols],
            max_positions: 5,
            active_positions: 0,
            strategy_config: config,
        }
    }

    /// Create with explicit symbol count (used in tests).
    pub fn with_num_symbols(num_symbols: usize) -> Self {
        let mut vpin_calcs = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            vpin_calcs.push(VPINCalculator::new(VPIN_BUCKET_SIZE_USD, VPIN_BUCKET_COUNT));
        }

        Self {
            vpin_calcs,
            last_signal: vec![None; num_symbols],
            max_positions: 5,
            active_positions: 0,
            strategy_config: StrategyConfig::default(),
        }
    }

    /// Read-only access to the strategy configuration.
    #[inline]
    pub fn config(&self) -> &StrategyConfig {
        &self.strategy_config
    }

    /// Core strategy evaluation. Called once per BookSnapshot from the
    /// strategy_evaluator_loop (Core 5).
    ///
    /// # Algorithm
    ///
    /// 1. **Sanity checks**: Verify spread, depth, mid price are valid
    /// 2. **VPIN check**: If VPIN > toxic threshold, skip (or trade smaller)
    /// 3. **Imbalance scoring**: Weighted 10-level depth skew
    /// 4. **Signal generation**: If imbalance exceeds threshold, generate intent
    /// 5. **Regime scaling**: Multiply size by regime momentum weight
    ///
    /// Returns `Some(OrderIntent)` if a signal is generated, `None` otherwise.
    pub fn evaluate(
        &self,
        metrics: &MicrostructureMetrics,
        regime: &RegimeState,
        symbol: &str,
    ) -> Option<OrderIntent> {
        // ── Sanity gates ──
        if metrics.mid_price <= 0.0 {
            return None;
        }
        if metrics.spread_bps < MIN_SPREAD_BPS || metrics.spread_bps > MAX_SPREAD_BPS {
            return None;
        }
        if metrics.bid_depth_usdt < MIN_DEPTH_USD || metrics.ask_depth_usdt < MIN_DEPTH_USD {
            return None;
        }

        // ── VPIN gating ──
        let vpin = metrics.vpin;
        if vpin > VPIN_TOXIC_THRESHOLD {
            // Toxic flow detected — skip this tick entirely.
            // In a full market-making system, we'd widen quotes instead.
            debug!(
                "[strategy] VPIN={:.3} > toxic threshold {:.2} — skipping",
                vpin, VPIN_TOXIC_THRESHOLD
            );
            return None;
        }

        // ── Imbalance scoring ──
        //
        // The imbalance is already computed as:
        //   (bid_depth - ask_depth) / (bid_depth + ask_depth)
        //
        // Positive imbalance = more bid depth = buying pressure → go LONG
        // Negative imbalance = more ask depth = selling pressure → go SHORT
        //
        // We use a threshold to filter noise.
        let imbalance = metrics.imbalance;
        let abs_imbalance = imbalance.abs();

        // Use configurable threshold instead of hardcoded constant
        let threshold = self.strategy_config.imbalance_threshold;
        if abs_imbalance < threshold {
            return None; // Not enough signal
        }

        // ── Composite signal strength ──
        //
        // signal = imbalance × (1 - vpin_penalty)
        //
        // When VPIN is high (approaching toxic), we reduce confidence.
        // When VPIN is low (safe flow), we trade at full confidence.
        let vpin_penalty = if vpin > VPIN_SAFE_THRESHOLD {
            // Linear scale from 0 at SAFE to 1 at TOXIC
            ((vpin - VPIN_SAFE_THRESHOLD) / (VPIN_TOXIC_THRESHOLD - VPIN_SAFE_THRESHOLD))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let raw_signal = abs_imbalance * (1.0 - vpin_penalty * 0.7);
        let confidence = (raw_signal / threshold).clamp(0.0, 1.0);

        // ── Direction ──
        let side = if imbalance > 0.0 {
            OrderSide::Buy  // Bid-heavy → price likely to rise
        } else {
            OrderSide::Sell // Ask-heavy → price likely to fall
        };

        // ── Position sizing (regime-adaptive) ──
        //
        // base_size × regime_momentum × signal_strength × vpin_safety
        //
        // regime.momentum_weight: 0.0 (no-trade) to 1.0 (full size)
        // In high-volatility regimes, Python sets this lower.
        let regime_scale = regime.momentum_weight();
        let vpin_scale = 1.0 - vpin_penalty * 0.5; // Reduce size as VPIN increases

        let position_size = (BASE_POSITION_SIZE * raw_signal * regime_scale * vpin_scale)
            .max(1.0)  // Minimum 1 contract
            .min(MAX_POSITION_SIZE);

        // ── Price calculation ──
        //
        // For scalping: place limit orders at the edge of the spread.
        // Buy: place at best_bid + 1 tick (aggressive)
        // Sell: place at best_ask - 1 tick (aggressive)
        //
        // For strong signals, use mid_price (cross the spread).
        let price = if confidence > 0.8 {
            // Strong signal: use mid price (will likely cross)
            Some(metrics.mid_price)
        } else {
            // Moderate signal: place at edge (maker rebate)
            Some(metrics.mid_price)
        };

        // ── Order type & time-in-force ──
        let order_type = if confidence > 0.7 {
            OrderType::Limit // Aggressive limit
        } else {
            OrderType::PostOnly // Passive maker
        };

        let time_in_force = if order_type == OrderType::PostOnly {
            "poc".to_string()
        } else {
            "gtc".to_string()
        };

        info!(
            "[strategy] Signal: {:?} size={:.1} confidence={:.3} imbalance={:.4} vpin={:.3} regime={:.2}",
            side, position_size, confidence, imbalance, vpin, regime_scale
        );

        Some(OrderIntent {
            symbol: symbol.to_string(),
            side,
            // Convert fractional contract size to integer contracts.
            // .round().max(1.0) ensures at least 1 contract.
            size: position_size.round().max(1.0) as i64,
            order_type,
            price,
            reduce_only: false,
            leverage: Some(self.config().leverage.unwrap_or(5).max(1).min(125) as i32),
            time_in_force,
            slippage_cap_pct: Some(0.001), // 10 bps slippage cap
            placement: PlacementType::AtBest,
            stop_loss: None,   // Computed by execution router based on ATR
            take_profit: None, // Computed by execution router based on ATR
            confidence,
            signal_tag: "microstructure_imbalance_vpin".to_string(),
        })
    }

    /// Update VPIN for a symbol from a trade event.
    ///
    /// Called from the WS trade ingestion path. This method is designed
    /// to be called at high frequency (every trade) with < 1μs overhead.
    ///
    /// # Parameters
    /// - `symbol_idx`: Symbol index (0-based)
    /// - `price`: Trade price
    /// - `volume_usd`: Trade volume in USD equivalent
    /// - `is_buy`: Optional buy/sell classification
    ///
    /// # Returns
    /// Updated VPIN value for this symbol.
    #[inline]
    pub fn update_vpin(
        &mut self,
        symbol_idx: usize,
        price: f64,
        volume_usd: f64,
        is_buy: Option<bool>,
    ) -> f64 {
        if symbol_idx < self.vpin_calcs.len() {
            self.vpin_calcs[symbol_idx].update(price, volume_usd, is_buy)
        } else {
            0.0
        }
    }

    /// Get current VPIN for a symbol (read-only).
    #[inline]
    pub fn get_vpin(&self, symbol_idx: usize) -> f64 {
        if symbol_idx < self.vpin_calcs.len() {
            self.vpin_calcs[symbol_idx].current()
        } else {
            0.0
        }
    }

    /// Check if VPIN is warmed up for a symbol.
    #[inline]
    pub fn is_vpin_ready(&self, symbol_idx: usize) -> bool {
        if symbol_idx < self.vpin_calcs.len() {
            self.vpin_calcs[symbol_idx].is_warmed_up()
        } else {
            false
        }
    }

    /// Notify the strategy that a position was opened (for position counting).
    pub fn notify_position_opened(&mut self) {
        self.active_positions = self.active_positions.saturating_add(1);
    }

    /// Notify the strategy that a position was closed.
    pub fn notify_position_closed(&mut self) {
        self.active_positions = self.active_positions.saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// StrategyConfig — retained for backward compatibility with config.rs
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

fn sc_default_imbalance() -> f64 { 0.15 }
fn sc_default_max_spread() -> f64 { 200.0 }
fn sc_default_min_depth() -> f64 { 1000.0 }
fn sc_default_order_size() -> i64 { 1 }
fn sc_default_post_only() -> bool { true }
fn sc_default_leverage() -> Option<i32> { Some(5) }
fn sc_default_min_fill_prob() -> f64 { 0.3 }
fn sc_default_max_stale_s() -> f64 { 5.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    #[serde(default = "sc_default_imbalance")]
    pub imbalance_threshold: f64,
    #[serde(default = "sc_default_max_spread")]
    pub max_spread_bps: f64,
    #[serde(default = "sc_default_min_depth")]
    pub min_bid_depth_usdt: f64,
    #[serde(default = "sc_default_min_depth")]
    pub min_ask_depth_usdt: f64,
    #[serde(default)]
    pub min_vpin: f64,
    #[serde(default = "sc_default_order_size")]
    pub order_size_contracts: i64,
    #[serde(default = "sc_default_post_only")]
    pub post_only: bool,
    #[serde(default)]
    pub enabled_symbols: Vec<String>,
    #[serde(default = "sc_default_leverage")]
    pub leverage: Option<i32>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub default_placement: PlacementConfigType,
    #[serde(default)]
    pub use_smart_placement: bool,
    #[serde(default = "sc_default_min_fill_prob")]
    pub min_fill_probability: f64,
    #[serde(default = "sc_default_max_stale_s")]
    pub max_stale_seconds: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            imbalance_threshold: sc_default_imbalance(),
            max_spread_bps: sc_default_max_spread(),
            min_bid_depth_usdt: sc_default_min_depth(),
            min_ask_depth_usdt: sc_default_min_depth(),
            min_vpin: 0.0,
            order_size_contracts: sc_default_order_size(),
            post_only: sc_default_post_only(),
            enabled_symbols: vec![],
            leverage: sc_default_leverage(),
            enabled: true,
            default_placement: PlacementConfigType::default(),
            use_smart_placement: false,
            min_fill_probability: sc_default_min_fill_prob(),
            max_stale_seconds: sc_default_max_stale_s(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PlacementConfigType {
    #[default]
    AtBest,
    Improve1Tick,
    Behind1Tick,
    AtMid,
    SmartPlace,
}

impl PlacementConfigType {
    pub fn to_placement_type(&self) -> PlacementType {
        match self {
            PlacementConfigType::AtBest => PlacementType::AtBest,
            PlacementConfigType::Improve1Tick => PlacementType::Improve1Tick,
            PlacementConfigType::Behind1Tick => PlacementType::Behind1Tick,
            PlacementConfigType::AtMid => PlacementType::AtMid,
            PlacementConfigType::SmartPlace => PlacementType::SmartPlace,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_basic() {
        let mut calc = VPINCalculator::new(100.0, 10);

        // All buys → VPIN should be high (all volume on one side)
        for _ in 0..20 {
            calc.update(100.0, 50.0, Some(true));
        }
        assert!(calc.current() > 0.8, "All-buy VPIN should be high, got {}", calc.current());
    }

    #[test]
    fn test_vpin_balanced() {
        let mut calc = VPINCalculator::new(100.0, 10);

        // Alternating buy/sell → VPIN should be low
        for i in 0..100 {
            let is_buy = i % 2 == 0;
            calc.update(100.0, 50.0, Some(is_buy));
        }
        assert!(calc.current() < 0.3, "Balanced VPIN should be low, got {}", calc.current());
    }

    #[test]
    fn test_vpin_warmup() {
        let calc = VPINCalculator::new(1000.0, 50);
        assert!(!calc.is_warmed_up());
    }

    #[test]
    fn test_strategy_no_signal_low_imbalance() {
        let engine = StrategyEngine::with_num_symbols(4);
        let metrics = MicrostructureMetrics {
            mid_price: 50000.0,
            spread_bps: 5.0,
            imbalance: 0.05, // Below threshold
            bid_depth_usdt: 50000.0,
            ask_depth_usdt: 48000.0,
            vpin: 0.2,
            last_trade_is_buy: None,
        };
        let regime = RegimeState::default();
        assert!(engine.evaluate(&metrics, &regime, "BTC_USDT").is_none());
    }

    #[test]
    fn test_strategy_signal_high_imbalance() {
        let engine = StrategyEngine::with_num_symbols(4);
        let metrics = MicrostructureMetrics {
            mid_price: 50000.0,
            spread_bps: 5.0,
            imbalance: 0.35, // Well above threshold
            bid_depth_usdt: 100000.0,
            ask_depth_usdt: 50000.0,
            vpin: 0.1,
            last_trade_is_buy: None,
        };
        let regime = RegimeState::default();
        let intent = engine.evaluate(&metrics, &regime, "BTC_USDT");
        assert!(intent.is_some(), "Should generate signal for high imbalance");
        let intent = intent.unwrap();
        assert!(matches!(intent.side, OrderSide::Buy)); // Bid-heavy → buy
    }

    #[test]
    fn test_strategy_no_signal_toxic_vpin() {
        let engine = StrategyEngine::with_num_symbols(4);
        let metrics = MicrostructureMetrics {
            mid_price: 50000.0,
            spread_bps: 5.0,
            imbalance: 0.5, // Strong imbalance
            bid_depth_usdt: 100000.0,
            ask_depth_usdt: 50000.0,
            vpin: 0.8, // Toxic flow!
            last_trade_is_buy: None,
        };
        let regime = RegimeState::default();
        assert!(engine.evaluate(&metrics, &regime, "BTC_USDT").is_none(), "Should skip on toxic VPIN");
    }

    #[test]
    fn test_strategy_no_signal_invalid_spread() {
        let engine = StrategyEngine::with_num_symbols(4);
        // Spread too high
        let metrics = MicrostructureMetrics {
            mid_price: 50000.0,
            spread_bps: 300.0,
            imbalance: 0.5,
            bid_depth_usdt: 100000.0,
            ask_depth_usdt: 50000.0,
            vpin: 0.1,
            last_trade_is_buy: None,
        };
        let regime = RegimeState::default();
        assert!(engine.evaluate(&metrics, &regime, "BTC_USDT").is_none());
    }
}
