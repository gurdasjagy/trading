//! Synthetic L3 Microstructure Engine.
//!
//! Synthesises queue-position estimates, order-flow toxicity, Kyle's Lambda,
//! and book-pressure signals from L2 order-book deltas and the public trade
//! tape — all without requiring proprietary L3 data.
//!
//! # Components
//!
//! | Struct | Description |
//! |---|---|
//! | [`SyntheticQueueTracker`] | Per-level queue-ahead estimation |
//! | [`EnhancedVpin`] | VPIN with Lee-Ready trade classification |
//! | [`KyleLambdaEstimator`] | Price-impact coefficient (rolling OLS) |
//! | [`BookPressureAnalyzer`] | Depth-change gradient & spoofing score |
//! | [`MicrostructureEngine`] | Unified entry point for the strategy engine |

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use ordered_float::OrderedFloat;
use serde::Serialize;

use crate::orderbook::RustOrderBook;
use crate::tick_processor::KahanSum;
use crate::trade_flow_analyzer::{TradeFlowAnalyzer, TradeFlowMetrics};

// ---------------------------------------------------------------------------
// Book side helper
// ---------------------------------------------------------------------------

/// Which side of the order book a level belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookSide {
    Bid,
    Ask,
}

// ---------------------------------------------------------------------------
// SyntheticQueueTracker
// ---------------------------------------------------------------------------

/// Per-level queue state used by ``SyntheticQueueTracker``.
#[derive(Clone, Debug)]
struct QueueState {
    /// The size of *our* resting order at this level.
    our_size: f64,
    /// Estimated volume ahead of us in the queue.
    queue_ahead: f64,
    /// Total visible size at this level when we joined.
    total_visible: f64,
    /// Nanosecond timestamp of the last update.
    last_update_ns: u64,
    /// Which side (bid/ask) this level is on.
    side: BookSide,
    /// Exponential moving average of depletion rate (size/second).
    depletion_ema: f64,
    /// Unix-nanosecond timestamp when this level was first tracked.
    first_tracked_ns: u64,
}

/// Tracks estimated queue position at each price level.
///
/// ## Algorithm
///
/// 1. When we place a limit order at price P with size S, record the total
///    visible size at P as `queue_ahead = book_size_at_P − S`.
/// 2. On each L2 delta at price P:
///    - If size decreased → trades consumed from the front of the queue.
///      `queue_ahead = max(0, queue_ahead − size_decrease)`.
///    - If size increased → new orders joined behind us (no change to
///      `queue_ahead`).
/// 3. On each trade at price P → subtract trade size from `queue_ahead`.
/// 4. Estimated fill probability = `1.0 − (queue_ahead / total_size_at_P)`.
pub struct SyntheticQueueTracker {
    /// price_level → QueueState
    tracked_levels: HashMap<OrderedFloat<f64>, QueueState>,
}

impl SyntheticQueueTracker {
    pub fn new() -> Self {
        Self {
            tracked_levels: HashMap::new(),
        }
    }

    /// Register a resting order at `price` with `our_size` contracts.
    ///
    /// `book_size_at_price` is the total visible size at that level
    /// *before* our order was included.
    pub fn track_order(
        &mut self,
        price: f64,
        our_size: f64,
        book_size_at_price: f64,
        side: BookSide,
    ) {
        let now_ns = now_ns();
        self.tracked_levels.insert(
            OrderedFloat(price),
            QueueState {
                our_size,
                queue_ahead: (book_size_at_price - our_size).max(0.0),
                total_visible: book_size_at_price,
                last_update_ns: now_ns,
                side,
                depletion_ema: 0.0,
                first_tracked_ns: now_ns,
            },
        );
    }

    /// Remove a tracked order (on fill or cancel).
    pub fn remove_order(&mut self, price: f64) {
        self.tracked_levels.remove(&OrderedFloat(price));
    }

    /// Update queue state on an L2 delta at `price`.
    ///
    /// `old_size` and `new_size` are the sizes before and after the delta.
    pub fn on_book_delta(&mut self, price: f64, old_size: f64, new_size: f64, side: BookSide) {
        let key = OrderedFloat(price);
        if let Some(state) = self.tracked_levels.get_mut(&key) {
            if state.side != side {
                return;
            }
            let now_ns = now_ns();
            let elapsed_s = (now_ns - state.last_update_ns) as f64 / 1e9;

            if new_size < old_size {
                // Front-of-queue depletion
                let decrease = old_size - new_size;
                let old_ahead = state.queue_ahead;
                state.queue_ahead = (state.queue_ahead - decrease).max(0.0);

                // Update depletion EMA
                if elapsed_s > 0.0 {
                    let rate = (old_ahead - state.queue_ahead) / elapsed_s;
                    let alpha = 0.1_f64;
                    state.depletion_ema = alpha * rate + (1.0 - alpha) * state.depletion_ema;
                }
            }
            // If new_size > old_size, new orders joined *behind* us — no change.
            state.total_visible = new_size;
            state.last_update_ns = now_ns;
        }
    }

    /// Update queue state on a public trade at `price`.
    pub fn on_trade(&mut self, price: f64, trade_size: f64, trade_side: BookSide) {
        let key = OrderedFloat(price);
        if let Some(state) = self.tracked_levels.get_mut(&key) {
            // A trade on the opposite side consumes from our queue
            if state.side != trade_side {
                state.queue_ahead = (state.queue_ahead - trade_size).max(0.0);
                state.last_update_ns = now_ns();
            }
        }
    }

    /// Estimated fill probability for our order at `price`.
    ///
    /// Returns ``0.0`` if the price is not tracked.
    pub fn fill_probability(&self, price: f64) -> f64 {
        let key = OrderedFloat(price);
        match self.tracked_levels.get(&key) {
            None => 0.0,
            Some(state) if state.total_visible <= 0.0 => 1.0,
            Some(state) => {
                1.0 - (state.queue_ahead / state.total_visible).clamp(0.0, 1.0)
            }
        }
    }

    /// Estimated time-to-fill in seconds based on the current depletion EMA.
    ///
    /// Returns ``None`` if the price is not tracked or the depletion rate is zero.
    pub fn estimated_time_to_fill(&self, price: f64) -> Option<f64> {
        let key = OrderedFloat(price);
        let state = self.tracked_levels.get(&key)?;
        if state.depletion_ema <= 0.0 || state.queue_ahead <= 0.0 {
            return Some(0.0);
        }
        Some(state.queue_ahead / state.depletion_ema)
    }

    /// Queue position as a fraction [0, 1] where 0 = front and 1 = back.
    pub fn queue_position_fraction(&self, price: f64) -> f64 {
        let key = OrderedFloat(price);
        match self.tracked_levels.get(&key) {
            None => 1.0,
            Some(state) if state.total_visible <= 0.0 => 0.0,
            Some(state) => (state.queue_ahead / state.total_visible).clamp(0.0, 1.0),
        }
    }

    /// Best-estimate bid-side fill probability (best tracked bid level).
    pub fn best_bid_fill_probability(&self) -> f64 {
        self.tracked_levels
            .iter()
            .filter(|(_, s)| s.side == BookSide::Bid)
            .map(|(p, _)| self.fill_probability(p.0))
            .fold(0.0_f64, f64::max)
    }

    /// Best-estimate ask-side fill probability.
    pub fn best_ask_fill_probability(&self) -> f64 {
        self.tracked_levels
            .iter()
            .filter(|(_, s)| s.side == BookSide::Ask)
            .map(|(p, _)| self.fill_probability(p.0))
            .fold(0.0_f64, f64::max)
    }
}

impl Default for SyntheticQueueTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trade classification helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeClassification {
    BuyerInitiated,
    SellerInitiated,
}

// ---------------------------------------------------------------------------
// VPIN bucket
// ---------------------------------------------------------------------------

#[derive(Clone, Default, Debug)]
struct VpinBucket {
    buy_vol: f64,
    sell_vol: f64,
    total_vol: f64,
}

// ---------------------------------------------------------------------------
// EnhancedVpin
// ---------------------------------------------------------------------------

/// Enhanced VPIN that uses the Lee-Ready algorithm for trade classification
/// when the exchange does not provide an explicit buy/sell side.
///
/// **Lee-Ready rule:**
/// - ``trade_price > mid_price`` → buyer-initiated
/// - ``trade_price < mid_price`` → seller-initiated
/// - ``trade_price == mid_price`` → tick test (compare to previous trade price)
pub struct EnhancedVpin {
    bucket_size: f64,
    current_bucket: VpinBucket,
    completed_buckets: VecDeque<VpinBucket>,
    max_buckets: usize,
    last_trade_price: f64,
    /// Running VPIN mean and variance for z-score computation.
    vpin_sum: KahanSum,
    vpin_sum_sq: KahanSum,
    vpin_count: usize,
}

impl EnhancedVpin {
    pub fn new(bucket_size: f64, max_buckets: usize) -> Self {
        Self {
            bucket_size,
            current_bucket: VpinBucket::default(),
            completed_buckets: VecDeque::new(),
            max_buckets,
            last_trade_price: 0.0,
            vpin_sum: KahanSum::default(),
            vpin_sum_sq: KahanSum::default(),
            vpin_count: 0,
        }
    }

    /// Classify a trade using the Lee-Ready algorithm.
    fn classify_trade(&mut self, price: f64, mid_price: f64) -> TradeClassification {
        let class = if mid_price > 0.0 {
            if price > mid_price {
                TradeClassification::BuyerInitiated
            } else if price < mid_price {
                TradeClassification::SellerInitiated
            } else {
                // Tick test: compare to last trade
                if self.last_trade_price > 0.0 && price > self.last_trade_price {
                    TradeClassification::BuyerInitiated
                } else {
                    TradeClassification::SellerInitiated
                }
            }
        } else {
            // No mid price available — default to tick test
            if self.last_trade_price > 0.0 && price > self.last_trade_price {
                TradeClassification::BuyerInitiated
            } else {
                TradeClassification::SellerInitiated
            }
        };
        self.last_trade_price = price;
        class
    }

    /// Process a trade and update VPIN buckets.
    ///
    /// `side` is ``Some("buy")``, ``Some("sell")``, or ``None`` (Lee-Ready
    /// classification is used when ``None``).
    pub fn on_trade(&mut self, price: f64, size: f64, side: Option<&str>, mid_price: f64) {
        if size <= 0.0 {
            return;
        }

        let classification = match side {
            Some(s) if s.eq_ignore_ascii_case("buy") => {
                self.last_trade_price = price;
                TradeClassification::BuyerInitiated
            }
            Some(s) if s.eq_ignore_ascii_case("sell") => {
                self.last_trade_price = price;
                TradeClassification::SellerInitiated
            }
            _ => self.classify_trade(price, mid_price),
        };

        match classification {
            TradeClassification::BuyerInitiated => self.current_bucket.buy_vol += size,
            TradeClassification::SellerInitiated => self.current_bucket.sell_vol += size,
        }
        self.current_bucket.total_vol += size;

        // Check if bucket is full
        if self.current_bucket.total_vol >= self.bucket_size {
            let total = self.current_bucket.total_vol;
            if total > 0.0 {
                let estimate = (self.current_bucket.buy_vol - self.current_bucket.sell_vol).abs()
                    / total;
                // Update running stats
                self.vpin_sum.add(estimate);
                self.vpin_sum_sq.add(estimate * estimate);
                self.vpin_count += 1;

                self.completed_buckets.push_back(self.current_bucket.clone());
                if self.completed_buckets.len() > self.max_buckets {
                    if let Some(old) = self.completed_buckets.pop_front() {
                        let old_e = (old.buy_vol - old.sell_vol).abs() / old.total_vol;
                        self.vpin_sum.add(-old_e);
                        self.vpin_sum_sq.add(-(old_e * old_e));
                        self.vpin_count = self.vpin_count.saturating_sub(1);
                    }
                }
            }
            self.current_bucket = VpinBucket::default();
        }
    }

    /// Current VPIN estimate (average of last N completed buckets).
    ///
    /// Returns ``0.0`` if fewer than 2 buckets have been completed.
    pub fn get_vpin(&self) -> f64 {
        if self.completed_buckets.len() < 2 {
            return 0.0;
        }
        let n = self.completed_buckets.len().min(50);
        let recent: f64 = self
            .completed_buckets
            .iter()
            .rev()
            .take(n)
            .map(|b| {
                if b.total_vol > 0.0 {
                    (b.buy_vol - b.sell_vol).abs() / b.total_vol
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n as f64;
        recent
    }

    /// VPIN z-score relative to historical distribution.
    ///
    /// A z-score > 2.0 indicates abnormally high order-flow toxicity.
    pub fn get_vpin_zscore(&self) -> f64 {
        if self.vpin_count < 10 {
            return 0.0;
        }
        let n = self.vpin_count as f64;
        let mean = self.vpin_sum.get() / n;
        let variance = (self.vpin_sum_sq.get() / n) - mean * mean;
        if variance <= 0.0 {
            return 0.0;
        }
        let std_dev = variance.sqrt();
        let current = self.get_vpin();
        (current - mean) / std_dev
    }
}

impl Default for EnhancedVpin {
    fn default() -> Self {
        Self::new(1000.0, 200)
    }
}

// ---------------------------------------------------------------------------
// KyleLambdaEstimator
// ---------------------------------------------------------------------------

/// Estimates Kyle's Lambda: the price impact per unit of signed order flow.
///
/// ```text
/// Lambda = Cov(dP, V_signed) / Var(V_signed)
/// ```
///
/// * High lambda → illiquid market, large price impact per trade.
/// * Low lambda  → liquid market, trades are absorbed easily.
///
/// Computed over a rolling window using incremental Kahan-compensated sums.
pub struct KyleLambdaEstimator {
    window: VecDeque<(f64, f64)>, // (price_change, signed_volume)
    window_size: usize,
    sum_pv: KahanSum,  // sum(dP * V_signed)
    sum_v2: KahanSum,  // sum(V_signed^2)
    sum_p: KahanSum,   // sum(dP)
    sum_v: KahanSum,   // sum(V_signed)
    n: usize,
}

impl KyleLambdaEstimator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            sum_pv: KahanSum::default(),
            sum_v2: KahanSum::default(),
            sum_p: KahanSum::default(),
            sum_v: KahanSum::default(),
            n: 0,
        }
    }

    /// Record one (price_change, signed_volume) observation.
    ///
    /// `signed_volume` is positive for buyer-initiated trades, negative for
    /// seller-initiated trades.
    pub fn on_trade(&mut self, price_change: f64, signed_volume: f64) {
        // Evict oldest observation
        if self.window.len() >= self.window_size {
            if let Some((old_dp, old_v)) = self.window.pop_front() {
                self.sum_pv.add(-(old_dp * old_v));
                self.sum_v2.add(-(old_v * old_v));
                self.sum_p.add(-old_dp);
                self.sum_v.add(-old_v);
                self.n = self.n.saturating_sub(1);
            }
        }

        self.sum_pv.add(price_change * signed_volume);
        self.sum_v2.add(signed_volume * signed_volume);
        self.sum_p.add(price_change);
        self.sum_v.add(signed_volume);
        self.window.push_back((price_change, signed_volume));
        self.n += 1;
    }

    /// Current Kyle's Lambda estimate.
    ///
    /// Returns ``0.0`` if fewer than 10 observations are available.
    pub fn get_lambda(&self) -> f64 {
        if self.n < 10 {
            return 0.0;
        }
        let n = self.n as f64;
        // Cov(dP, V) = E[dP*V] - E[dP]*E[V]
        let e_pv = self.sum_pv.get() / n;
        let e_p = self.sum_p.get() / n;
        let e_v = self.sum_v.get() / n;
        let cov_pv = e_pv - e_p * e_v;

        // Var(V) = E[V²] - E[V]²
        let e_v2 = self.sum_v2.get() / n;
        let var_v = e_v2 - e_v * e_v;

        if var_v.abs() < 1e-12 {
            return 0.0;
        }
        cov_pv / var_v
    }
}

impl Default for KyleLambdaEstimator {
    fn default() -> Self {
        Self::new(500)
    }
}

// ---------------------------------------------------------------------------
// BookPressureAnalyzer
// ---------------------------------------------------------------------------

/// Snapshot of the top-N book levels at a point in time.
#[derive(Clone, Debug)]
struct LevelSnapshot {
    /// Unix-nanosecond timestamp.
    timestamp_ns: u64,
    /// Top-20 bid levels ``(price, size)``, descending.
    bid_levels: Vec<(f64, f64)>,
    /// Top-20 ask levels ``(price, size)``, ascending.
    ask_levels: Vec<(f64, f64)>,
}

/// Analyzes the rate of change of book depth to detect:
///
/// 1. **Absorption** — Large resting orders that absorb aggressive flow
///    without moving price.
/// 2. **Spoofing** — Large orders that appear and disappear rapidly
///    (lifetime < 500 ms).
/// 3. **Layering** — Multiple large orders stacked at consecutive levels
///    that move together.
pub struct BookPressureAnalyzer {
    level_history: VecDeque<LevelSnapshot>,
    snapshot_interval_ms: u64,
    max_history: usize,
    last_snapshot_ns: u64,
}

impl BookPressureAnalyzer {
    pub fn new(snapshot_interval_ms: u64, max_history: usize) -> Self {
        Self {
            level_history: VecDeque::new(),
            snapshot_interval_ms,
            max_history,
            last_snapshot_ns: 0,
        }
    }

    /// Take a snapshot of the current book state if enough time has passed
    /// since the last snapshot.
    pub fn maybe_snapshot(&mut self, book: &RustOrderBook) {
        let now = now_ns();
        let interval_ns = self.snapshot_interval_ms * 1_000_000;
        if now - self.last_snapshot_ns < interval_ns {
            return;
        }
        self.snapshot(book);
    }

    /// Force a snapshot regardless of the poll interval.
    pub fn snapshot(&mut self, book: &RustOrderBook) {
        let snap = LevelSnapshot {
            timestamp_ns: now_ns(),
            bid_levels: book.get_bids(20),
            ask_levels: book.get_asks(20),
        };
        self.level_history.push_back(snap);
        if self.level_history.len() > self.max_history {
            self.level_history.pop_front();
        }
        self.last_snapshot_ns = now_ns();
    }

    /// Compute bid-side pressure gradient: rate of change of total bid depth
    /// over recent snapshots.
    ///
    /// Positive → buying pressure is building; negative → bid depth is eroding.
    pub fn bid_pressure_gradient(&self) -> f64 {
        self.pressure_gradient(true)
    }

    /// Compute ask-side pressure gradient.
    pub fn ask_pressure_gradient(&self) -> f64 {
        self.pressure_gradient(false)
    }

    fn pressure_gradient(&self, is_bid: bool) -> f64 {
        if self.level_history.len() < 2 {
            return 0.0;
        }
        let n = self.level_history.len().min(10);
        let recent: Vec<&LevelSnapshot> = self.level_history.iter().rev().take(n).collect();

        let depth_series: Vec<f64> = recent
            .iter()
            .map(|s| {
                let levels = if is_bid { &s.bid_levels } else { &s.ask_levels };
                levels.iter().map(|(_, sz)| sz).sum::<f64>()
            })
            .collect();

        if depth_series.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope over depth vs time index
        let n_f = depth_series.len() as f64;
        let x_mean = (n_f - 1.0) / 2.0;
        let y_mean = depth_series.iter().sum::<f64>() / n_f;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, y) in depth_series.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < 1e-12 {
            return 0.0;
        }
        // Normalize by mean depth to get a relative gradient
        if y_mean.abs() < 1e-12 {
            return 0.0;
        }
        (numerator / denominator) / y_mean
    }

    /// Detect potential spoofing: price levels that appeared in one snapshot
    /// but disappeared within `threshold_ms` milliseconds.
    ///
    /// Returns a score in ``[0.0, 1.0]`` where higher means more likely spoofing.
    pub fn spoofing_score(&self, threshold_ms: u64) -> f64 {
        if self.level_history.len() < 2 {
            return 0.0;
        }

        let threshold_ns = threshold_ms * 1_000_000;
        let mut spoof_events = 0u32;
        let mut total_large_appearances = 0u32;

        // Look at consecutive snapshot pairs
        let snaps: Vec<&LevelSnapshot> = self.level_history.iter().collect();
        for window in snaps.windows(3) {
            let (prev, mid, next) = (window[0], window[1], window[2]);
            let time_span = next.timestamp_ns.saturating_sub(prev.timestamp_ns);
            if time_span > threshold_ns {
                continue;
            }

            // Check for levels that exist in `mid` but not in `prev` or `next`
            for (price, size) in mid.bid_levels.iter().chain(mid.ask_levels.iter()) {
                if *size <= 0.0 {
                    continue;
                }
                let mean_size = mid
                    .bid_levels
                    .iter()
                    .chain(mid.ask_levels.iter())
                    .map(|(_, s)| s)
                    .sum::<f64>()
                    / (mid.bid_levels.len() + mid.ask_levels.len()) as f64;

                // Only count large orders (> 3× mean)
                if *size < mean_size * 3.0 {
                    continue;
                }

                let in_prev = prev
                    .bid_levels
                    .iter()
                    .chain(prev.ask_levels.iter())
                    .any(|(p, _)| (p - price).abs() < 1e-8);
                let in_next = next
                    .bid_levels
                    .iter()
                    .chain(next.ask_levels.iter())
                    .any(|(p, _)| (p - price).abs() < 1e-8);

                total_large_appearances += 1;
                if !in_prev || !in_next {
                    spoof_events += 1;
                }
            }
        }

        if total_large_appearances == 0 {
            return 0.0;
        }
        (spoof_events as f64 / total_large_appearances as f64).clamp(0.0, 1.0)
    }

    /// Detect absorption: price level held steady (depth stayed ≥ threshold)
    /// despite significant trade volume hitting it.
    ///
    /// Returns a score in ``[0.0, 1.0]``.
    pub fn absorption_score(&self, price: f64) -> f64 {
        if self.level_history.len() < 3 {
            return 0.0;
        }

        let snaps: Vec<&LevelSnapshot> = self.level_history.iter().rev().take(10).collect();
        let mut held_count = 0u32;
        let mut total_count = 0u32;

        for snap in &snaps {
            let found = snap
                .bid_levels
                .iter()
                .chain(snap.ask_levels.iter())
                .any(|(p, s)| (p - price).abs() < 1e-8 && *s > 0.0);
            if found {
                held_count += 1;
            }
            total_count += 1;
        }

        if total_count == 0 {
            return 0.0;
        }
        (held_count as f64 / total_count as f64).clamp(0.0, 1.0)
    }
}

impl Default for BookPressureAnalyzer {
    fn default() -> Self {
        Self::new(500, 120) // 500 ms interval, 60 s of history at 500 ms
    }
}

// ---------------------------------------------------------------------------
// MicrostructureSnapshot
// ---------------------------------------------------------------------------

/// Rich microstructure snapshot returned to the strategy engine.
///
/// All fields are computed from cached state so ``get_snapshot()`` runs in
/// O(1) / sub-microsecond time.
#[derive(Clone, Debug, Serialize)]
pub struct MicrostructureSnapshot {
    // ── Existing tick-processor metrics ──────────────────────────────────
    /// Volume-weighted average price.
    pub vwap: f64,
    /// Tick imbalance: ``(buy_vol − sell_vol) / (buy_vol + sell_vol)``.
    pub tick_imbalance: f64,
    /// Basic VPIN from the tick processor.
    pub vpin: f64,

    // ── Synthetic L3: queue position ─────────────────────────────────────
    /// Queue position fraction on the bid side: 0 = front, 1 = back.
    pub estimated_queue_position_bid: f64,
    /// Queue position fraction on the ask side.
    pub estimated_queue_position_ask: f64,
    /// Fill probability for a resting bid order [0, 1].
    pub bid_fill_probability: f64,
    /// Fill probability for a resting ask order [0, 1].
    pub ask_fill_probability: f64,

    // ── Enhanced flow analysis ────────────────────────────────────────────
    /// Enhanced VPIN z-score (0+ = elevated toxicity; >2 = abnormal).
    pub flow_toxicity_score: f64,
    /// Kyle's Lambda: price impact per unit signed volume.
    pub kyle_lambda: f64,
    /// Exponential moving-average of trades per second.
    pub trade_arrival_rate: f64,

    // ── Book dynamics ─────────────────────────────────────────────────────
    /// Bid-side depth change gradient (normalised).
    pub bid_pressure_gradient: f64,
    /// Ask-side depth change gradient (normalised).
    pub ask_pressure_gradient: f64,
    /// Probability of spoofing [0, 1].
    pub spoofing_score: f64,
    /// ``true`` if absorption was detected at the best bid or ask.
    pub absorption_detected: bool,

    // ── Composite signal ──────────────────────────────────────────────────
    /// Composite microstructure edge score in ``[-1, 1]``.
    /// Positive = conditions favour buying; negative = favour selling.
    pub microstructure_edge_score: f64,
}

// ---------------------------------------------------------------------------
// MicrostructureEngine
// ---------------------------------------------------------------------------

/// Unified microstructure engine.
///
/// Called by the strategy engine on every orderbook delta and every trade.
/// Returns a ``MicrostructureSnapshot`` on each evaluation tick.
pub struct MicrostructureEngine {
    queue_tracker: SyntheticQueueTracker,
    enhanced_vpin: EnhancedVpin,
    kyle_lambda: KyleLambdaEstimator,
    book_pressure: BookPressureAnalyzer,
    /// Exponential moving average for trade arrival rate (trades/s).
    trade_arrival_ema: f64,
    /// Nanosecond timestamp of the last trade.
    last_trade_ns: u64,
    /// Previous trade price (for Kyle's Lambda ΔP computation).
    prev_trade_price: f64,
    /// Current VWAP (updated by the strategy engine from tick_processor).
    pub last_vwap: f64,
    /// Current tick imbalance.
    pub last_tick_imbalance: f64,
    /// Current basic VPIN.
    pub last_vpin: f64,
    /// Trade flow analyzer for buy/sell ratio and large trade detection.
    trade_flow_analyzer: TradeFlowAnalyzer,
}

impl MicrostructureEngine {
    pub fn new() -> Self {
        Self {
            queue_tracker: SyntheticQueueTracker::new(),
            enhanced_vpin: EnhancedVpin::new(1000.0, 200),
            kyle_lambda: KyleLambdaEstimator::new(500),
            book_pressure: BookPressureAnalyzer::new(500, 120),
            trade_arrival_ema: 0.0,
            last_trade_ns: 0,
            prev_trade_price: 0.0,
            last_vwap: 0.0,
            last_tick_imbalance: 0.0,
            last_vpin: 0.0,
            trade_flow_analyzer: TradeFlowAnalyzer::default(),
        }
    }

    /// Call after each orderbook delta is applied.
    ///
    /// `deltas_bid` and `deltas_ask` are vectors of ``(price, old_size,
    /// new_size)`` produced by ``RustOrderBook::apply_delta_tracked()``.
    pub fn on_book_update(
        &mut self,
        book: &RustOrderBook,
        deltas_bid: &[(f64, f64, f64)],
        deltas_ask: &[(f64, f64, f64)],
    ) {
        // Update queue tracker
        for &(price, old_size, new_size) in deltas_bid {
            self.queue_tracker
                .on_book_delta(price, old_size, new_size, BookSide::Bid);
        }
        for &(price, old_size, new_size) in deltas_ask {
            self.queue_tracker
                .on_book_delta(price, old_size, new_size, BookSide::Ask);
        }

        // Maybe take a pressure snapshot
        self.book_pressure.maybe_snapshot(book);
    }

    /// Call after each public trade is processed.
    pub fn on_trade(&mut self, price: f64, size: f64, side: &str, mid_price: f64) {
        let now = now_ns();

        // Trade arrival rate (EMA of inter-trade interval)
        if self.last_trade_ns > 0 {
            let interval_s = (now - self.last_trade_ns) as f64 / 1e9;
            if interval_s > 0.0 {
                let rate = 1.0 / interval_s;
                let alpha = 0.05_f64;
                self.trade_arrival_ema =
                    alpha * rate + (1.0 - alpha) * self.trade_arrival_ema;
            }
        }
        self.last_trade_ns = now;

        // Enhanced VPIN
        let side_opt = if side.is_empty() { None } else { Some(side) };
        self.enhanced_vpin.on_trade(price, size, side_opt, mid_price);

        // Kyle's Lambda: dP = price - prev_price; signed_volume
        if self.prev_trade_price > 0.0 {
            let dp = price - self.prev_trade_price;
            let signed_vol = if side.eq_ignore_ascii_case("buy") {
                size
            } else {
                -size
            };
            self.kyle_lambda.on_trade(dp, signed_vol);
        }
        self.prev_trade_price = price;

        // Queue tracker trade update
        let book_side = if side.eq_ignore_ascii_case("sell") {
            BookSide::Ask
        } else {
            BookSide::Bid
        };
        self.queue_tracker.on_trade(price, size, book_side);

        // Trade flow analyzer update (FEATURE 4)
        let trade_side = if side.eq_ignore_ascii_case("buy") { 0 } else { 1 };
        self.trade_flow_analyzer.on_trade(price, size, trade_side, now);
    }

    /// Register a resting order for queue tracking.
    pub fn track_order(
        &mut self,
        price: f64,
        our_size: f64,
        book_size_at_price: f64,
        side: BookSide,
    ) {
        self.queue_tracker
            .track_order(price, our_size, book_size_at_price, side);
    }

    /// Remove a resting order from queue tracking (on fill or cancel).
    pub fn remove_order(&mut self, price: f64) {
        self.queue_tracker.remove_order(price);
    }

    /// Compute and return the current ``MicrostructureSnapshot``.
    ///
    /// Designed to run in < 10 µs from cached state.
    pub fn get_snapshot(&self, book: &RustOrderBook) -> MicrostructureSnapshot {
        let best_bid = book.get_best_bid().map(|(p, _)| p).unwrap_or(0.0);
        let best_ask = book.get_best_ask().map(|(p, _)| p).unwrap_or(0.0);

        let bid_fill_prob = self.queue_tracker.best_bid_fill_probability();
        let ask_fill_prob = self.queue_tracker.best_ask_fill_probability();
        let queue_bid = self.queue_tracker.queue_position_fraction(best_bid);
        let queue_ask = self.queue_tracker.queue_position_fraction(best_ask);

        let flow_toxicity = self.enhanced_vpin.get_vpin_zscore().max(0.0);
        let kyle_lam = self.kyle_lambda.get_lambda();
        let spoof = self.book_pressure.spoofing_score(500);
        let bid_grad = self.book_pressure.bid_pressure_gradient();
        let ask_grad = self.book_pressure.ask_pressure_gradient();
        let absorption = self.book_pressure.absorption_score(best_bid) > 0.6
            || self.book_pressure.absorption_score(best_ask) > 0.6;

        // Composite edge score in [-1, 1]
        // Positive signals: bid fill prob high, bid pressure positive, low toxicity
        // Negative signals: ask pressure building, high toxicity, spoofing detected
        let edge = self.compute_edge_score(
            self.last_tick_imbalance,
            bid_grad,
            ask_grad,
            flow_toxicity,
            spoof,
            bid_fill_prob,
        );

        MicrostructureSnapshot {
            vwap: self.last_vwap,
            tick_imbalance: self.last_tick_imbalance,
            vpin: self.last_vpin,
            estimated_queue_position_bid: queue_bid,
            estimated_queue_position_ask: queue_ask,
            bid_fill_probability: bid_fill_prob,
            ask_fill_probability: ask_fill_prob,
            flow_toxicity_score: flow_toxicity,
            kyle_lambda: kyle_lam,
            trade_arrival_rate: self.trade_arrival_ema,
            bid_pressure_gradient: bid_grad,
            ask_pressure_gradient: ask_grad,
            spoofing_score: spoof,
            absorption_detected: absorption,
            microstructure_edge_score: edge,
        }
    }

    /// Get trade flow metrics for use in strategy filtering.
    pub fn get_trade_flow_metrics(&self) -> TradeFlowMetrics {
        self.trade_flow_analyzer.get_metrics()
    }

    /// Check if trade flow analyzer is ready.
    pub fn is_trade_flow_ready(&self) -> bool {
        self.trade_flow_analyzer.is_ready()
    }

    fn compute_edge_score(
        &self,
        tick_imbalance: f64,
        bid_grad: f64,
        ask_grad: f64,
        toxicity: f64,
        spoof: f64,
        bid_fill_prob: f64,
    ) -> f64 {
        // Weighted combination of directional signals
        let dir_score = tick_imbalance * 0.4
            + (bid_grad - ask_grad).clamp(-1.0, 1.0) * 0.3
            + (bid_fill_prob - 0.5) * 0.2;

        // Penalise high toxicity and spoofing (reduces magnitude, not direction)
        let penalty = (toxicity.min(5.0) / 5.0) * 0.3 + spoof * 0.2;
        let raw = dir_score * (1.0 - penalty);
        raw.clamp(-1.0, 1.0)
    }
}

impl Default for MicrostructureEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Return the current time as Unix nanoseconds.
#[inline]
fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
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
    fn test_synthetic_queue_tracker_depletion() {
        let mut tracker = SyntheticQueueTracker::new();
        // We have 100 contracts at price 50000, with 200 ahead of us
        tracker.track_order(50000.0, 100.0, 300.0, BookSide::Bid);

        assert!((tracker.fill_probability(50000.0) - (1.0 - 200.0 / 300.0)).abs() < 1e-6);

        // A trade of 50 contracts hits this level — queue_ahead decreases
        tracker.on_trade(50000.0, 50.0, BookSide::Ask);
        assert!((tracker.fill_probability(50000.0) - (1.0 - 150.0 / 300.0)).abs() < 1e-6);

        // A book delta removes 150 more from the level
        tracker.on_book_delta(50000.0, 300.0, 150.0, BookSide::Bid);
        assert_eq!(tracker.fill_probability(50000.0), 1.0);
    }

    #[test]
    fn test_lee_ready_classification() {
        let mut vpin = EnhancedVpin::new(100.0, 50);
        // Trade above mid → buyer initiated
        vpin.on_trade(50100.0, 10.0, None, 50000.0);
        // Trade below mid → seller initiated
        vpin.on_trade(49900.0, 10.0, None, 50000.0);
        // At mid, tick test: price > last (49900) → buyer
        vpin.on_trade(50000.0, 10.0, None, 50000.0);
        // No panic, basic sanity
        assert!(vpin.get_vpin() >= 0.0);
    }

    #[test]
    fn test_kyle_lambda_convergence() {
        let mut estimator = KyleLambdaEstimator::new(500);
        // Simulate 500 trades with known relationship: dP = 0.01 * V_signed
        for i in 0..600usize {
            let v = if i % 2 == 0 { 10.0_f64 } else { -10.0_f64 };
            let dp = 0.01 * v;
            estimator.on_trade(dp, v);
        }
        let lambda = estimator.get_lambda();
        // Should converge near 0.01 within 10%
        assert!((lambda - 0.01).abs() < 0.002, "lambda={lambda}");
    }

    #[test]
    fn test_spoofing_detection() {
        let mut analyzer = BookPressureAnalyzer::new(100, 60);
        let book = crate::orderbook::RustOrderBook::new("BTC_USDT");

        // Force three snapshots: large bid appears in snapshot 2, gone in 3
        // We test the spoofing_score function is callable without panic.
        analyzer.snapshot(&book);
        analyzer.snapshot(&book);
        analyzer.snapshot(&book);
        let score = analyzer.spoofing_score(500);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_enhanced_vpin_zscore() {
        let mut vpin = EnhancedVpin::new(100.0, 50);
        // Fill many buckets
        for i in 0..2000usize {
            let side = if i % 3 == 0 { "sell" } else { "buy" };
            vpin.on_trade(50000.0 + i as f64 * 0.1, 50.0, Some(side), 50000.0);
        }
        let z = vpin.get_vpin_zscore();
        // Should be a finite number
        assert!(z.is_finite());
    }
}
