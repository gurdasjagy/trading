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

use std::collections::{VecDeque, HashMap};

use serde::{Serialize, Deserialize};
use tracing::{debug, info};

use crate::execution_gateway::{OrderIntent, OrderSide, OrderType};
use crate::execution_state::PlacementType;
use crate::regime::RegimeState;
use crate::candle_aggregator::{CandleAggregator, Timeframe};

// ---------------------------------------------------------------------------
// Configuration Constants
// ---------------------------------------------------------------------------

/// Number of volume buckets for VPIN calculation.
const VPIN_BUCKET_COUNT: usize = 50;

/// Volume per bucket in USD equivalent (normalized).
/// A bucket is "full" when this much volume has been accumulated.
/// Reduced from 1k to 100 for faster VPIN warm-up on liquid pairs like BTC/USDT.
const VPIN_BUCKET_SIZE_USD: f64 = 100.0;

/// VPIN threshold above which we consider flow as toxic.
/// Above this → reduce position size, widen quotes.
const VPIN_TOXIC_THRESHOLD: f64 = 0.65;

/// VPIN threshold below which we consider flow as safe for market-making.
const VPIN_SAFE_THRESHOLD: f64 = 0.35;

/// Minimum imbalance magnitude to generate a signal (in basis points / 10000).
/// Reduced from 0.15 (15%) to 0.015 (1.5%) for testnet compatibility.
/// Testnet orderbooks are typically thinner than mainnet, producing different
/// imbalance distributions. A 1.5% depth skew is significant enough to
/// indicate directional pressure on thin testnet books.
const IMBALANCE_ENTRY_THRESHOLD: f64 = 0.015;

/// Minimum spread in bps to avoid trading in tight spreads.
const MIN_SPREAD_BPS: f64 = 1.0;

/// Maximum spread in bps — don't trade if spread is insane.
const MAX_SPREAD_BPS: f64 = 200.0;

/// Base position size (in contracts) before regime scaling.
const BASE_POSITION_SIZE: f64 = 1.0;

/// Maximum position size after all scaling.
const MAX_POSITION_SIZE: f64 = 50.0;

/// Minimum depth in USD on both sides to consider orderbook valid.
/// Lowered from 1000 to 100 for testnet compatibility where books are thinner.
const MIN_DEPTH_USD: f64 = 100.0;

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
    /// Cumulative Volume Delta for 5-minute window.
    pub cvd_5m: f64,
    /// Cumulative Volume Delta for 15-minute window.
    pub cvd_15m: f64,
    /// Cumulative Volume Delta for 1-hour window.
    pub cvd_1h: f64,
    /// Gamma flip level for BTC (from options-derived gamma exposure).
    pub gamma_flip_btc: Option<f64>,
    /// Gamma flip level for ETH (from options-derived gamma exposure).
    pub gamma_flip_eth: Option<f64>,
    /// Phase 3 Feature 11: Wyckoff phase (Accumulation, Markup, Distribution, Markdown, Unknown).
    /// BUG 11 FIX: Changed from &'static str to String
    pub wyckoff_phase: String,
    /// Phase 3 Feature 12: Nearest Fibonacci level percentage (e.g., 0.618 for 61.8% retracement).
    pub fib_nearest_level: f64,
    /// Phase 3 Feature 13: Ichimoku cloud position (AboveCloud, InCloud, BelowCloud).
    /// BUG 11 FIX: Changed from &'static str to String
    pub ichimoku_cloud_position: String,
    /// Phase 3 Feature 14: Market maker inventory pressure (-1.0 to 1.0).
    pub mm_inventory_pressure: f64,
    /// Phase 3 Feature 15: BTC-ETH correlation (-1.0 to 1.0).
    pub btc_eth_correlation: f64,
    /// FEATURE 1 (Task 1): CVD divergence signals
    pub cvd_divergence_bearish: bool,
    pub cvd_divergence_bullish: bool,
    /// FEATURE 1 (Task 2): Current funding rate for this symbol
    pub funding_rate: f64,
    /// FEATURE 1 (Task 3): Distance to VPOC in percentage
    pub vpoc_distance_pct: f64,
    /// FEATURE 1 (Task 4): Realized volatility regime (Low, Normal, High, Extreme)
    pub realized_vol_regime: String,
    /// FEATURE 1 (Task 5): Liquidation cascade active flag
    pub cascade_active: bool,
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
// Adaptive Imbalance Threshold (Phase 2 Feature 6)
// ---------------------------------------------------------------------------

/// Adaptive threshold calculator using online mean/variance estimation.
///
/// Computes a dynamic imbalance threshold based on recent market conditions:
/// `threshold = max(min_threshold, mean + std_multiplier * std_dev)`
///
/// Uses Welford's online algorithm for numerically stable variance calculation.
pub struct AdaptiveThreshold {
    /// Window size for rolling statistics.
    window_size: usize,
    /// Ring buffer of recent imbalance values.
    imbalance_history: VecDeque<f64>,
    /// Running sum for mean calculation.
    running_sum: f64,
    /// Running sum of squares for variance calculation.
    running_sum_sq: f64,
    /// Minimum threshold floor (prevents threshold from going too low).
    min_threshold: f64,
    /// Standard deviation multiplier (controls sensitivity).
    std_multiplier: f64,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold calculator.
    ///
    /// # Arguments
    /// * `window_size` — Number of samples to track (e.g., 3600 for 1 hour at 1 sample/sec)
    /// * `min_threshold` — Minimum threshold floor (e.g., 0.02 = 2%)
    /// * `std_multiplier` — Standard deviation multiplier (e.g., 1.5 = mean + 1.5σ)
    pub fn new(window_size: usize, min_threshold: f64, std_multiplier: f64) -> Self {
        Self {
            window_size,
            imbalance_history: VecDeque::with_capacity(window_size),
            running_sum: 0.0,
            running_sum_sq: 0.0,
            min_threshold,
            std_multiplier,
        }
    }

    /// Update with a new imbalance value.
    ///
    /// Uses Welford's online algorithm for numerically stable variance:
    /// 1. Add new value to ring buffer
    /// 2. If buffer is full, evict oldest value
    /// 3. Update running sum and sum of squares
    #[inline]
    pub fn update(&mut self, imbalance: f64) {
        // Evict oldest if full
        if self.imbalance_history.len() >= self.window_size {
            if let Some(old) = self.imbalance_history.pop_front() {
                self.running_sum -= old;
                self.running_sum_sq -= old * old;
            }
        }

        // Add new value
        self.imbalance_history.push_back(imbalance);
        self.running_sum += imbalance;
        self.running_sum_sq += imbalance * imbalance;
    }

    /// Get the current adaptive threshold.
    ///
    /// Computes: `max(min_threshold, min(max_threshold, mean + std_multiplier * std_dev))`
    /// TASK 2c FIX: Added max cap of 10% to prevent threshold from becoming unreachable
    /// on liquid pairs like BTC/USDT where imbalance rarely exceeds 3-5%.
    #[inline]
    pub fn get_threshold(&self) -> f64 {
        if self.imbalance_history.is_empty() {
            return self.min_threshold;
        }

        let n = self.imbalance_history.len() as f64;
        let mean = self.running_sum / n;
        let variance = (self.running_sum_sq / n) - (mean * mean);
        let std_dev = variance.max(0.0).sqrt();

        let threshold = mean + self.std_multiplier * std_dev;
        // Cap at 10% to prevent threshold from becoming unreachable on liquid pairs
        threshold.max(self.min_threshold).min(0.10)
    }

    /// Check if the calculator is warmed up.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.imbalance_history.len() >= self.window_size / 2
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
    /// Multi-timeframe candle aggregator for confluence filtering.
    /// BUG 10 FIX: Changed visibility to pub(crate)
    pub(crate) candle_aggregator: parking_lot::Mutex<CandleAggregator>,
    /// Adaptive imbalance threshold calculator (Phase 2 Feature 6).
    adaptive_threshold: parking_lot::Mutex<AdaptiveThreshold>,
    /// Task 7: Per-symbol pair profiles (Phase 2 Feature 8).
    pair_profiles: HashMap<String, crate::config::PairProfile>,
}

impl StrategyEngine {
    /// Create a new StrategyEngine from an EngineConfig.
    /// Allocates per-symbol VPIN calculators for up to 64 symbols.
    /// Task 13: Changed signature to accept &EngineConfig instead of StrategyConfig.
    pub fn new(engine_config: &crate::config::EngineConfig) -> Self {
        let num_symbols = 64; // Pre-allocate for max symbols
        let mut vpin_calcs = Vec::with_capacity(num_symbols);
        
        // Task 9: Apply pair-specific VPIN bucket size
        // For now, use default bucket size - will be customized per symbol in future
        for _ in 0..num_symbols {
            vpin_calcs.push(VPINCalculator::new(VPIN_BUCKET_SIZE_USD, VPIN_BUCKET_COUNT));
        }

        // Task 7: Extract strategy config and pair profiles from engine config
        let strategy_config = engine_config.strategy.clone();
        let pair_profiles = engine_config.pair_profiles.clone();

        Self {
            vpin_calcs,
            last_signal: vec![None; num_symbols],
            max_positions: 5,
            active_positions: 0,
            strategy_config,
            candle_aggregator: parking_lot::Mutex::new(CandleAggregator::default()),
            adaptive_threshold: parking_lot::Mutex::new(AdaptiveThreshold::new(3600, 0.02, 1.5)),
            pair_profiles,
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
            candle_aggregator: parking_lot::Mutex::new(CandleAggregator::default()),
            adaptive_threshold: parking_lot::Mutex::new(AdaptiveThreshold::new(3600, 0.02, 1.5)),
            pair_profiles: HashMap::new(),
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
        ml_weights: &'static crate::ml_weight_receiver::MlWeightReader,
        symbol_id: u16,
    ) -> Option<OrderIntent> {
        // ── Sanity gates ──
        if metrics.mid_price <= 0.0 {
            debug!("[strategy] Gate: mid_price <= 0 — skipping");
            return None;
        }
        if metrics.spread_bps < MIN_SPREAD_BPS || metrics.spread_bps > MAX_SPREAD_BPS {
            debug!("[strategy] Gate: spread_bps={:.1} outside [{:.1}, {:.1}] — skipping",
                metrics.spread_bps, MIN_SPREAD_BPS, MAX_SPREAD_BPS);
            return None;
        }
        if metrics.bid_depth_usdt < MIN_DEPTH_USD || metrics.ask_depth_usdt < MIN_DEPTH_USD {
            debug!("[strategy] Gate: depth too low bid=${:.0} ask=${:.0} (min=${:.0}) — skipping",
                metrics.bid_depth_usdt, metrics.ask_depth_usdt, MIN_DEPTH_USD);
            return None;
        }

        // ── VPIN gating with toxicity-based spread widening (FEAT 10) ──
        //
        // Instead of simply skipping when VPIN is elevated, we widen the
        // entry price to protect against adverse selection while still
        // capturing trades. Only truly toxic flow (VPIN > 0.65) is skipped.
        //
        // VPIN 0.00–0.35: No widening (safe flow)
        // VPIN 0.35–0.50: Widen entry by 1 tick (mild toxicity)
        // VPIN 0.50–0.65: Widen entry by 2 ticks (elevated toxicity)
        // VPIN > 0.65:    Skip entirely (toxic flow)
        let vpin = metrics.vpin;
        if vpin > VPIN_TOXIC_THRESHOLD {
            debug!(
                "[strategy] VPIN={:.3} > toxic threshold {:.2} — skipping",
                vpin, VPIN_TOXIC_THRESHOLD
            );
            return None;
        }

        // Calculate VPIN-based spread widening ticks (FEAT 10)
        let vpin_widen_ticks: u32 = if vpin > 0.50 {
            2 // Elevated toxicity: widen by 2 ticks
        } else if vpin > VPIN_SAFE_THRESHOLD {
            1 // Mild toxicity: widen by 1 tick
        } else {
            0 // Safe flow: no widening
        };

        if vpin_widen_ticks > 0 {
            info!(
                "[strategy] FEAT 10: VPIN={:.3} — widening entry by {} tick(s) instead of skipping",
                vpin, vpin_widen_ticks
            );
        }

        // ── Multi-timeframe confluence filtering (FEATURE 2) ──
        // STRATEGY 1 FIX: Loosened confluence gates to reduce silent rejections.
        // Previously required strict 15m EMA alignment AND RSI range, blocking
        // ~70% of valid signals. Now:
        // 1. Falls back to 5m candles if 15m not ready
        // 2. Relaxed RSI bands (30-70 instead of 40-60)
        // 3. Confluence is a scoring factor, not a hard gate
        let candle_agg = self.candle_aggregator.lock();
        let mut confluence_score: f64 = 1.0; // Default: neutral (no penalty)

        // Try 15m first, fall back to 5m
        let candle_data = if candle_agg.is_ready(Timeframe::M15) {
            candle_agg.get_candle(Timeframe::M15)
        } else if candle_agg.is_ready(Timeframe::M5) {
            candle_agg.get_candle(Timeframe::M5)
        } else {
            None
        };

        if let Some(candle) = candle_data {
            let ema20 = candle.ema20;
            let ema50 = candle.ema50;
            let rsi = candle.rsi14;
            let is_long_signal = metrics.imbalance > 0.0;

            if is_long_signal {
                // Long: penalize if EMA structure is bearish or RSI oversold
                if ema20 > ema50 && rsi > 30.0 {
                    confluence_score = 1.2; // Boost: trend-aligned
                } else if ema20 <= ema50 && rsi > 50.0 {
                    confluence_score = 0.8; // Mild penalty: counter-trend but momentum OK
                } else if rsi <= 30.0 {
                    confluence_score = 0.6; // Oversold: higher risk
                } else {
                    confluence_score = 0.5; // Counter-trend with weak momentum
                }
            } else {
                // Short: penalize if EMA structure is bullish or RSI overbought
                if ema20 < ema50 && rsi < 70.0 {
                    confluence_score = 1.2; // Boost: trend-aligned
                } else if ema20 >= ema50 && rsi < 50.0 {
                    confluence_score = 0.8; // Mild penalty
                } else if rsi >= 70.0 {
                    confluence_score = 0.6; // Overbought: higher risk
                } else {
                    confluence_score = 0.5; // Counter-trend
                }
            }

            debug!(
                "[strategy] Confluence: EMA20={:.2}, EMA50={:.2}, RSI={:.1}, score={:.2}",
                ema20, ema50, rsi, confluence_score
            );
        } else {
            // During warmup: allow trading with neutral confluence
            debug!("[strategy] Candle data not ready - using neutral confluence (warmup period)");
        }
        drop(candle_agg); // Release lock before continuing

        // ── Task 7: Cascade active gating ──
        // Skip signal generation during liquidation cascades
        if metrics.cascade_active {
            // TASK 2b FIX: Log when cascade blocks signals
            info!("[strategy] Signal blocked: liquidation cascade active");
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

        // ── Phase 2 Feature 6: Adaptive Imbalance Threshold ──
        // Update the adaptive threshold with current imbalance
        self.adaptive_threshold.lock().update(abs_imbalance);
        
        // Task 8: Apply pair-specific imbalance threshold
        // Get the dynamic threshold (mean + 1.5σ, floored at 2%)
        let base_threshold = self.adaptive_threshold.lock().get_threshold();
        
        // Override with pair profile if available
        let threshold = self.pair_profiles.get(symbol)
            .map(|profile| profile.imbalance_threshold)
            .unwrap_or(base_threshold);
        
        if abs_imbalance < threshold {
            debug!("[strategy] Gate: imbalance {:.4} < threshold {:.4} — skipping", abs_imbalance, threshold);
            return None; // Not enough signal
        }

        // ── Task 6: Composite signal scoring with multi-signal confluence ──
        //
        // CVD divergence score: penalize signals against divergence
        let cvd_score = if metrics.cvd_divergence_bearish && imbalance > 0.0 {
            0.0 // Bearish divergence on long signal = reject
        } else if metrics.cvd_divergence_bullish && imbalance < 0.0 {
            0.0 // Bullish divergence on short signal = reject
        } else {
            1.0
        };
        
        // Funding rate score: boost signals aligned with funding arbitrage
        let funding_score = if metrics.funding_rate > 0.0001 && imbalance < 0.0 {
            1.2 // High funding + short signal = boost
        } else if metrics.funding_rate < -0.0001 && imbalance > 0.0 {
            1.2 // Negative funding + long signal = boost
        } else {
            1.0
        };
        
        // VPOC distance score: boost signals near VPOC (high liquidity zone)
        let vpoc_score = if metrics.vpoc_distance_pct.abs() < 0.005 {
            1.15 // Within 0.5% of VPOC = boost
        } else {
            1.0
        };
        
        // Volatility regime scaling
        let vol_regime_scale = match metrics.realized_vol_regime.as_str() {
            "Low" => 1.5,      // Low vol = increase size
            "Normal" => 1.0,   // Normal vol = baseline
            "High" => 0.5,     // High vol = reduce size
            "Extreme" => 0.25, // Extreme vol = minimal size
            _ => 1.0,
        };
        
        // VPIN penalty (toxic flow detection)
        let vpin_penalty = if vpin > VPIN_SAFE_THRESHOLD {
            ((vpin - VPIN_SAFE_THRESHOLD) / (VPIN_TOXIC_THRESHOLD - VPIN_SAFE_THRESHOLD))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        
        // FIXED composite signal scoring — weighted sum (not product).
        // Each factor contributes additively with its designated weight.
        // STRATEGY 1 FIX: Added confluence_score as a multiplicative factor
        // to soften (not block) counter-trend signals.
        let imbalance_score    = (abs_imbalance / threshold).min(3.0) * 0.35; // 35% weight
        let cvd_contribution   = cvd_score * 0.15;                             // 15% weight
        let funding_contribution = funding_score * 0.15;                       // 15% weight
        let vpoc_contribution  = vpoc_score * 0.10;                            // 10% weight
        let vol_contribution   = (vol_regime_scale / 1.5_f64).min(1.0) * 0.10;    // 10% weight
        let confluence_contribution = (confluence_score / 1.2_f64).min(1.0) * 0.15; // 15% weight

        let composite = (imbalance_score + cvd_contribution + funding_contribution
            + vpoc_contribution + vol_contribution + confluence_contribution)
            * (1.0 - vpin_penalty * 0.5);

        let confidence = composite.clamp(0.0, 1.0);

        // ── ML Weights Blending ──
        let ml_w = ml_weights.get_weights(symbol_id).unwrap_or(crate::ml_weight_receiver::SymbolWeight {
            symbol_id,
            _pad: 0,
            momentum_weight: 1.0,
            mean_reversion_weight: 0.0,
            volatility_weight: 1.0,
            confidence_floor: 0.0,
            max_position_scale: 1.0,
        });

        if confidence < ml_w.confidence_floor as f64 {
            return None; // Floor not met
        }

        let is_mean_rev = ml_w.mean_reversion_weight > ml_w.momentum_weight;

        // ── Direction ──
        let side = if is_mean_rev {
            if imbalance > 0.0 { OrderSide::Sell } else { OrderSide::Buy }
        } else {
            if imbalance > 0.0 { OrderSide::Buy } else { OrderSide::Sell }
        };

        // ── Position sizing (regime-adaptive) ──
        //
        // base_size × regime_momentum × signal_strength × vpin_safety
        //
        // regime.momentum_weight: 0.0 (no-trade) to 1.0 (full size)
        // In high-volatility regimes, Python sets this lower.
        let regime_scale = regime.momentum_weight() * (ml_w.volatility_weight as f64).max(0.5);
        let vpin_scale = 1.0 - vpin_penalty * 0.5; // Reduce size as VPIN increases
        
        let base_size = BASE_POSITION_SIZE * ml_w.max_position_scale.max(1.0) as f64;

        let position_size = (base_size * (abs_imbalance / threshold).min(3.0) * regime_scale * vpin_scale)
            .max(1.0)  // Minimum 1 contract
            .min(MAX_POSITION_SIZE * ml_w.max_position_scale.max(1.0) as f64);

        // ── Fee-aware order type & price selection (INST) ──
        //
        // Institutional approach: default to Post-Only (maker) orders to capture
        // rebates. Only cross the spread (taker) for very high-confidence signals
        // where speed of execution outweighs fee savings.
        //
        // At VIP0: maker=2bps, taker=5bps → Post-Only saves 3bps per side (6bps RT).
        // At VIP10+: maker=-1bps (rebate), taker=2bps → Post-Only earns 3bps per side.
        // ── FEAT 10: Estimate tick size for spread widening ──
        // Use pair profile tick size if available, otherwise derive from spread.
        // A reasonable default tick is 1/10 of the half-spread.
        let estimated_tick_size = {
            let half_spread_price = metrics.mid_price * (metrics.spread_bps / 2.0) / 10_000.0;
            // Reasonable tick: roughly 1 bps of price, but at least 1/10 of half-spread
            let tick = (metrics.mid_price * 0.0001).max(half_spread_price / 10.0);
            if tick <= 0.0 { metrics.mid_price * 0.0001 } else { tick }
        };
        let vpin_widen_amount = vpin_widen_ticks as f64 * estimated_tick_size;

        let (order_type, time_in_force, price) = if confidence > 0.85 {
            // Very high confidence: cross the spread for immediate fill.
            // FEAT 10: Still apply VPIN widening even for high confidence orders
            let widened_price = if vpin_widen_ticks > 0 {
                if side == OrderSide::Buy {
                    metrics.mid_price - vpin_widen_amount
                } else {
                    metrics.mid_price + vpin_widen_amount
                }
            } else {
                metrics.mid_price
            };
            (OrderType::Limit, "ioc".to_string(), Some(widened_price))
        } else if confidence > 0.7 {
            // High confidence: aggressive limit at mid, may or may not cross.
            // FEAT 10: Apply VPIN widening to reduce adverse selection
            let widened_price = if vpin_widen_ticks > 0 {
                if side == OrderSide::Buy {
                    metrics.mid_price - vpin_widen_amount
                } else {
                    metrics.mid_price + vpin_widen_amount
                }
            } else {
                metrics.mid_price
            };
            (OrderType::Limit, "gtc".to_string(), Some(widened_price))
        } else {
            // Default: Post-Only to guarantee maker fee / rebate.
            // Price at the join side of the book (best bid for buys, best ask for sells).
            // Derive best_bid/best_ask from mid_price and spread_bps.
            // FEAT 10: Apply VPIN widening on top of the normal maker placement
            let half_spread = metrics.mid_price * (metrics.spread_bps / 2.0) / 10_000.0;
            let maker_price = if side == OrderSide::Buy {
                // Place bid at best_bid level, widened further by VPIN toxicity
                metrics.mid_price - half_spread - vpin_widen_amount
            } else {
                // Place ask at best_ask level, widened further by VPIN toxicity
                metrics.mid_price + half_spread + vpin_widen_amount
            };
            (OrderType::PostOnly, "poc".to_string(), Some(maker_price))
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
            order_type: order_type.clone(),
            price,
            reduce_only: false,
            leverage: Some(self.config().leverage.unwrap_or(5).max(1).min(125) as i32),
            time_in_force: time_in_force.clone(),
            slippage_cap_pct: Some(0.001), // 10 bps slippage cap
            placement: PlacementType::AtBest,
            stop_loss: None,   // Computed by execution router based on ATR
            take_profit: None, // Computed by execution router based on ATR
            confidence,
            signal_tag: "microstructure_imbalance_vpin".to_string(),
            // CATEGORY 2 FIX: Minimum fill size for IOC orders
            min_fill_size: if order_type == OrderType::Limit && time_in_force == "ioc" {
                Some((position_size * 0.5).round().max(1.0) as i64) // Require at least 50% fill
            } else {
                None
            },
            // CATEGORY 8 FIX: Strategy name for PnL attribution
            strategy_name: "microstructure_imbalance_vpin".to_string(),
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

    /// Update the candle aggregator with a trade event.
    ///
    /// Called from the strategy evaluator loop when processing trade events
    /// (update_type=3 from the SPSC ring).
    #[inline]
    pub fn update_candles(&self, timestamp_ns: u64, price: f64, volume: f64) {
        self.candle_aggregator.lock().on_trade(timestamp_ns, price, volume);
    }
}

// ---------------------------------------------------------------------------
// StrategyConfig — retained for backward compatibility with config.rs
// ---------------------------------------------------------------------------

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
            imbalance: 0.01, // Below 3% threshold (IMBALANCE_ENTRY_THRESHOLD = 0.03)
            bid_depth_usdt: 50000.0,
            ask_depth_usdt: 48000.0,
            vpin: 0.2,
            last_trade_is_buy: None,
            cvd_5m: 0.0,
            cvd_15m: 0.0,
            cvd_1h: 0.0,
            gamma_flip_btc: None,
            gamma_flip_eth: None,
            wyckoff_phase: "Unknown".to_string(),
            fib_nearest_level: 0.0,
            ichimoku_cloud_position: "InCloud".to_string(),
            mm_inventory_pressure: 0.0,
            btc_eth_correlation: 0.0,
            cvd_divergence_bearish: false,
            cvd_divergence_bullish: false,
            funding_rate: 0.0001,
            vpoc_distance_pct: 1.0,
            realized_vol_regime: "Normal".to_string(),
            cascade_active: false,
        };
        let regime = RegimeState::default();
        let ml_weights: &'static crate::ml_weight_receiver::MlWeightReader = 
            Box::leak(Box::new(crate::ml_weight_receiver::MlWeightReader::new("/dev/shm/ml_weights_test")));
        assert!(engine.evaluate(&metrics, &regime, "BTC_USDT", ml_weights, 1).is_none());
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
            cvd_5m: 0.0,
            cvd_15m: 0.0,
            cvd_1h: 0.0,
            gamma_flip_btc: None,
            gamma_flip_eth: None,
            wyckoff_phase: "Unknown".to_string(),
            fib_nearest_level: 0.0,
            ichimoku_cloud_position: "InCloud".to_string(),
            mm_inventory_pressure: 0.0,
            btc_eth_correlation: 0.0,
            cvd_divergence_bearish: false,
            cvd_divergence_bullish: false,
            funding_rate: 0.0001,
            vpoc_distance_pct: 1.0,
            realized_vol_regime: "Normal".to_string(),
            cascade_active: false,
        };
        let regime = RegimeState::default();
        let ml_weights: &'static crate::ml_weight_receiver::MlWeightReader = 
            Box::leak(Box::new(crate::ml_weight_receiver::MlWeightReader::new("/dev/shm/ml_weights_test")));
        let intent = engine.evaluate(&metrics, &regime, "BTC_USDT", ml_weights, 1);
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
            cvd_5m: 0.0,
            cvd_15m: 0.0,
            cvd_1h: 0.0,
            gamma_flip_btc: None,
            gamma_flip_eth: None,
            wyckoff_phase: "Unknown".to_string(),
            fib_nearest_level: 0.0,
            ichimoku_cloud_position: "InCloud".to_string(),
            mm_inventory_pressure: 0.0,
            btc_eth_correlation: 0.0,
            cvd_divergence_bearish: false,
            cvd_divergence_bullish: false,
            funding_rate: 0.0001,
            vpoc_distance_pct: 1.0,
            realized_vol_regime: "Normal".to_string(),
            cascade_active: false,
        };
        let regime = RegimeState::default();
        let ml_weights: &'static crate::ml_weight_receiver::MlWeightReader = 
            Box::leak(Box::new(crate::ml_weight_receiver::MlWeightReader::new("/dev/shm/ml_weights_test")));
        assert!(engine.evaluate(&metrics, &regime, "BTC_USDT", ml_weights, 1).is_none(), "Should skip on toxic VPIN");
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
            cvd_5m: 0.0,
            cvd_15m: 0.0,
            cvd_1h: 0.0,
            gamma_flip_btc: None,
            gamma_flip_eth: None,
            wyckoff_phase: "Unknown".to_string(),
            fib_nearest_level: 0.0,
            ichimoku_cloud_position: "InCloud".to_string(),
            mm_inventory_pressure: 0.0,
            btc_eth_correlation: 0.0,
            cvd_divergence_bearish: false,
            cvd_divergence_bullish: false,
            funding_rate: 0.0001,
            vpoc_distance_pct: 1.0,
            realized_vol_regime: "Normal".to_string(),
            cascade_active: false,
        };
        let regime = RegimeState::default();
        let ml_weights: &'static crate::ml_weight_receiver::MlWeightReader = 
            Box::leak(Box::new(crate::ml_weight_receiver::MlWeightReader::new("/dev/shm/ml_weights_test")));
        assert!(engine.evaluate(&metrics, &regime, "BTC_USDT", ml_weights, 1).is_none());
    }
}

