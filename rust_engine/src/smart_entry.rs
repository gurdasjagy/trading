//! Directive 4: Institutional Execution — Smart Entry, Volatility Trailing,
//! Adverse Selection Protection.
//!
//! # Features
//!
//! 1. **Maker-Rebate Optimization**: Posts Post-Only limit orders at best bid/ask
//!    to capture maker rebates. Chases the book if unfilled. Falls back to taker
//!    only if microstructure signals a violent breakout.
//!
//! 2. **Volatility-Adjusted Trailing Stops**: Uses real-time ATR (Average True Range)
//!    to dynamically size trailing stops. Tightens in low-volatility, widens in
//!    high-momentum conditions.
//!
//! 3. **Adverse Selection Protection**: Monitors L2 bid-side cancellations (spoofing
//!    removal). Pauses long entries for 500ms if sudden liquidity voids are detected.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

// ═══════════════════════════════════════════════════════════════════════════
// 1. Smart Entry Router — Maker-Rebate Optimization
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the smart entry router.
#[derive(Debug, Clone)]
pub struct SmartEntryConfig {
    /// Maximum time to wait for maker fill before crossing spread (ms).
    pub maker_timeout_ms: u64,
    /// Chase interval: how often to re-post at new best bid/ask (ms).
    pub chase_interval_ms: u64,
    /// VPIN threshold above which we skip maker and go taker immediately.
    pub vpin_taker_threshold: f64,
    /// Orderflow imbalance threshold for taker fallback.
    pub imbalance_taker_threshold: f64,
    /// Maximum number of chase attempts before going taker.
    pub max_chase_attempts: u32,
}

impl Default for SmartEntryConfig {
    fn default() -> Self {
        Self {
            maker_timeout_ms: 500,
            chase_interval_ms: 100,
            vpin_taker_threshold: 0.7,
            imbalance_taker_threshold: 0.6,
            max_chase_attempts: 5,
        }
    }
}

/// The result of the smart entry router's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryDecision {
    /// Post a Post-Only limit order at the specified price.
    PostMaker,
    /// Cross the spread with a market/IOC order (taker).
    CrossSpread,
    /// Cancel existing maker order and re-post at new price.
    ChaseBook,
    /// Do not enter — adverse conditions detected.
    PauseEntry,
}

/// Tracks the state of a smart entry attempt.
#[derive(Debug, Clone)]
pub struct EntryAttempt {
    pub symbol_id: u16,
    pub is_buy: bool,
    pub target_price: f64,
    pub posted_price: f64,
    pub posted_at: Instant,
    pub chase_count: u32,
    pub order_id: Option<String>,
}

/// Smart entry router that decides between maker and taker orders.
pub struct SmartEntryRouter {
    config: SmartEntryConfig,
    /// Active entry attempts.
    active_entries: Vec<EntryAttempt>,
}

impl SmartEntryRouter {
    pub fn new(config: SmartEntryConfig) -> Self {
        Self {
            config,
            active_entries: Vec::with_capacity(8),
        }
    }

    /// Decide how to enter a position based on current microstructure.
    ///
    /// # Arguments
    /// * `is_buy` — Whether this is a long entry
    /// * `best_bid` — Current best bid price
    /// * `best_ask` — Current best ask price
    /// * `vpin` — Current VPIN (0.0–1.0)
    /// * `imbalance` — Current orderflow imbalance (-1.0 to 1.0)
    /// * `adverse_paused` — Whether the adverse selection detector has paused entries
    /// * `tick_size` — Minimum price increment for the symbol (optional)
    pub fn decide(
        &self,
        is_buy: bool,
        best_bid: f64,
        best_ask: f64,
        vpin: f64,
        imbalance: f64,
        adverse_paused: bool,
        tick_size: Option<f64>,
    ) -> (EntryDecision, f64) {
        // Check adverse selection pause first
        if adverse_paused {
            let price = if is_buy { best_bid } else { best_ask };
            return (EntryDecision::PauseEntry, price);
        }

        // Check if microstructure indicates violent breakout → taker immediately
        if vpin > self.config.vpin_taker_threshold {
            let price = if is_buy { best_ask } else { best_bid };
            info!(
                "[smart-entry] VPIN={:.3} > {:.3}: crossing spread",
                vpin, self.config.vpin_taker_threshold
            );
            return (EntryDecision::CrossSpread, price);
        }

        // Check strong directional imbalance
        if is_buy && imbalance > self.config.imbalance_taker_threshold {
            // Strong buy pressure — cross spread to not miss the move
            return (EntryDecision::CrossSpread, best_ask);
        }
        if !is_buy && imbalance < -self.config.imbalance_taker_threshold {
            // Strong sell pressure — cross spread
            return (EntryDecision::CrossSpread, best_bid);
        }

        // FEATURE 12: Spread-adjusted entry pricing
        // For buy signals: best_bid + tick_size (join the bid queue ahead)
        // For sell signals: best_ask - tick_size (join the ask queue ahead)
        let tick = tick_size.unwrap_or(0.01); // Default to 0.01 if not provided
        let price = if is_buy {
            best_bid + tick
        } else {
            best_ask - tick
        };
        (EntryDecision::PostMaker, price)
    }

    /// Check if an active entry attempt should be chased or converted to taker.
    pub fn check_active_entries(
        &mut self,
        best_bid: f64,
        best_ask: f64,
    ) -> Vec<(usize, EntryDecision, f64)> {
        let mut actions = Vec::new();

        for (idx, entry) in self.active_entries.iter_mut().enumerate() {
            let elapsed = entry.posted_at.elapsed();

            // Check timeout → convert to taker
            if elapsed >= Duration::from_millis(self.config.maker_timeout_ms) {
                let price = if entry.is_buy { best_ask } else { best_bid };
                actions.push((idx, EntryDecision::CrossSpread, price));
                continue;
            }

            // Check if book has moved away → chase
            if elapsed >= Duration::from_millis(self.config.chase_interval_ms) {
                let current_best = if entry.is_buy { best_bid } else { best_ask };
                if (current_best - entry.posted_price).abs() > f64::EPSILON {
                    if entry.chase_count < self.config.max_chase_attempts {
                        entry.chase_count += 1;
                        entry.posted_price = current_best;
                        entry.posted_at = Instant::now();
                        actions.push((idx, EntryDecision::ChaseBook, current_best));
                    } else {
                        // Max chases exceeded → cross spread
                        let price = if entry.is_buy { best_ask } else { best_bid };
                        actions.push((idx, EntryDecision::CrossSpread, price));
                    }
                }
            }
        }

        actions
    }

    /// Register a new active entry attempt.
    pub fn register_entry(
        &mut self,
        symbol_id: u16,
        is_buy: bool,
        price: f64,
        order_id: Option<String>,
    ) {
        self.active_entries.push(EntryAttempt {
            symbol_id,
            is_buy,
            target_price: price,
            posted_price: price,
            posted_at: Instant::now(),
            chase_count: 0,
            order_id,
        });
    }

    /// Remove an entry attempt (filled or cancelled).
    pub fn remove_entry(&mut self, idx: usize) {
        if idx < self.active_entries.len() {
            self.active_entries.swap_remove(idx);
        }
    }

    /// Clear all entries for a symbol.
    pub fn clear_entries_for(&mut self, symbol_id: u16) {
        self.active_entries.retain(|e| e.symbol_id != symbol_id);
    }
}

impl Default for SmartEntryRouter {
    fn default() -> Self {
        Self::new(SmartEntryConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Volatility-Adjusted Trailing Stop
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for volatility-adjusted trailing stops.
#[derive(Debug, Clone)]
pub struct VolTrailingConfig {
    /// ATR period (number of candles/ticks for ATR calculation).
    pub atr_period: usize,
    /// Base ATR multiplier for trailing distance (e.g., 2.0 = 2x ATR).
    pub base_atr_multiplier: f64,
    /// Minimum trailing distance as % of entry price.
    pub min_trail_pct: f64,
    /// Maximum trailing distance as % of entry price.
    pub max_trail_pct: f64,
    /// Volume dryup threshold: tighten trail when recent volume < this * avg volume.
    pub volume_dryup_ratio: f64,
    /// Momentum expansion threshold: widen trail when recent volatility > this * avg.
    pub momentum_expansion_ratio: f64,
}

impl Default for VolTrailingConfig {
    fn default() -> Self {
        Self {
            atr_period: 14,
            base_atr_multiplier: 2.0,
            min_trail_pct: 0.3,
            max_trail_pct: 5.0,
            volume_dryup_ratio: 0.5,
            momentum_expansion_ratio: 1.5,
        }
    }
}

/// Real-time ATR calculator using tick-by-tick high/low data.
pub struct AtrCalculator {
    /// Rolling window of true ranges.
    true_ranges: VecDeque<f64>,
    /// Period for ATR calculation.
    period: usize,
    /// Previous close price.
    prev_close: f64,
    /// Current ATR value.
    current_atr: f64,
    /// Long-term average ATR (for momentum detection).
    long_term_atr: f64,
    /// Rolling volume window.
    volumes: VecDeque<f64>,
    /// Average volume.
    avg_volume: f64,
}

impl AtrCalculator {
    pub fn new(period: usize) -> Self {
        Self {
            true_ranges: VecDeque::with_capacity(period),
            period,
            prev_close: 0.0,
            current_atr: 0.0,
            long_term_atr: 0.0,
            volumes: VecDeque::with_capacity(period * 3),
            avg_volume: 0.0,
        }
    }

    /// Update ATR with a new candle/tick.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) {
        if self.prev_close > 0.0 {
            let tr = (high - low)
                .max((high - self.prev_close).abs())
                .max((low - self.prev_close).abs());

            self.true_ranges.push_back(tr);
            if self.true_ranges.len() > self.period {
                self.true_ranges.pop_front();
            }

            // Compute ATR
            if !self.true_ranges.is_empty() {
                self.current_atr =
                    self.true_ranges.iter().sum::<f64>() / self.true_ranges.len() as f64;
            }
        }
        self.prev_close = close;

        // Track volume
        self.volumes.push_back(volume);
        if self.volumes.len() > self.period * 3 {
            self.volumes.pop_front();
        }
        if !self.volumes.is_empty() {
            self.avg_volume = self.volumes.iter().sum::<f64>() / self.volumes.len() as f64;
        }

        // Update long-term ATR (smoothed)
        if self.long_term_atr == 0.0 {
            self.long_term_atr = self.current_atr;
        } else {
            self.long_term_atr = self.long_term_atr * 0.99 + self.current_atr * 0.01;
        }
    }

    pub fn atr(&self) -> f64 {
        self.current_atr
    }

    pub fn long_term_atr(&self) -> f64 {
        self.long_term_atr
    }

    pub fn avg_volume(&self) -> f64 {
        self.avg_volume
    }

    pub fn is_ready(&self) -> bool {
        self.true_ranges.len() >= self.period / 2
    }
}

/// Volatility-adjusted trailing stop calculator.
pub struct VolatilityTrailingStop {
    config: VolTrailingConfig,
    atr_calc: AtrCalculator,
}

impl VolatilityTrailingStop {
    pub fn new(config: VolTrailingConfig) -> Self {
        let period = config.atr_period;
        Self {
            config,
            atr_calc: AtrCalculator::new(period),
        }
    }

    /// Update with new price data.
    pub fn update_tick(&mut self, high: f64, low: f64, close: f64, volume: f64) {
        self.atr_calc.update(high, low, close, volume);
    }

    /// Calculate the trailing stop distance for a position.
    ///
    /// Returns the trailing stop distance in absolute price terms.
    pub fn calculate_trail_distance(
        &self,
        entry_price: f64,
        recent_volume: f64,
    ) -> f64 {
        if !self.atr_calc.is_ready() || entry_price <= 0.0 {
            // Fallback: use 1% trailing distance
            return entry_price * 0.01;
        }

        let atr = self.atr_calc.atr();
        let long_atr = self.atr_calc.long_term_atr();
        let avg_vol = self.atr_calc.avg_volume();

        // Base trailing distance = ATR * multiplier
        let mut trail_distance = atr * self.config.base_atr_multiplier;

        // Adjust for volume conditions
        if avg_vol > 0.0 {
            let vol_ratio = recent_volume / avg_vol;

            if vol_ratio < self.config.volume_dryup_ratio {
                // Volume drying up — tighten the trail (reduce by 30%)
                trail_distance *= 0.7;
                debug!("[vol-trail] Volume dryup ({:.2}x avg): tightening trail", vol_ratio);
            } else if vol_ratio > self.config.momentum_expansion_ratio * 2.0 {
                // High volume surge — widen the trail (increase by 50%)
                trail_distance *= 1.5;
                debug!("[vol-trail] High volume ({:.2}x avg): widening trail", vol_ratio);
            }
        }

        // Adjust for volatility regime
        if long_atr > 0.0 {
            let vol_ratio = atr / long_atr;
            if vol_ratio > self.config.momentum_expansion_ratio {
                // High volatility — widen trail to let profits run
                trail_distance *= 1.3;
            } else if vol_ratio < 0.5 {
                // Low volatility — tighten trail
                trail_distance *= 0.8;
            }
        }

        // Clamp to min/max percentage of entry price
        let min_trail = entry_price * self.config.min_trail_pct / 100.0;
        let max_trail = entry_price * self.config.max_trail_pct / 100.0;
        trail_distance = trail_distance.max(min_trail).min(max_trail);

        trail_distance
    }

    pub fn atr(&self) -> f64 {
        self.atr_calc.atr()
    }

    pub fn is_ready(&self) -> bool {
        self.atr_calc.is_ready()
    }
}

impl Default for VolatilityTrailingStop {
    fn default() -> Self {
        Self::new(VolTrailingConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Adverse Selection Protection — Spoofing Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for adverse selection protection.
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Minimum cancellation volume % to trigger a pause.
    pub cancel_volume_threshold_pct: f64,
    /// Pause duration after detecting adverse conditions (ms).
    pub pause_duration_ms: u64,
    /// Number of recent snapshots to analyze.
    pub lookback_ticks: usize,
    /// Minimum total bid depth change (negative = cancellations) to trigger.
    pub bid_depth_drop_threshold_pct: f64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            cancel_volume_threshold_pct: 30.0,
            pause_duration_ms: 500,
            lookback_ticks: 5,
            bid_depth_drop_threshold_pct: 20.0,
        }
    }
}

/// Tracks bid-side depth for spoofing detection.
pub struct AdverseSelectionGuard {
    config: AdverseSelectionConfig,
    /// Recent bid depth snapshots.
    bid_depth_history: VecDeque<f64>,
    /// Recent ask depth snapshots.
    ask_depth_history: VecDeque<f64>,
    /// When the current pause expires (None = not paused).
    pause_until: Option<Instant>,
    /// Total pauses triggered.
    pub total_pauses: u64,
}

impl AdverseSelectionGuard {
    pub fn new(config: AdverseSelectionConfig) -> Self {
        let cap = config.lookback_ticks + 1;
        Self {
            config,
            bid_depth_history: VecDeque::with_capacity(cap),
            ask_depth_history: VecDeque::with_capacity(cap),
            pause_until: None,
            total_pauses: 0,
        }
    }

    /// Update with latest L2 book depth.
    ///
    /// # Arguments
    /// * `bid_depth_usdt` — Total bid-side depth in USDT
    /// * `ask_depth_usdt` — Total ask-side depth in USDT
    pub fn update(&mut self, bid_depth_usdt: f64, ask_depth_usdt: f64) {
        self.bid_depth_history.push_back(bid_depth_usdt);
        self.ask_depth_history.push_back(ask_depth_usdt);

        if self.bid_depth_history.len() > self.config.lookback_ticks + 1 {
            self.bid_depth_history.pop_front();
        }
        if self.ask_depth_history.len() > self.config.lookback_ticks + 1 {
            self.ask_depth_history.pop_front();
        }

        // Check for sudden bid-side depth drop (spoofing removal)
        if self.bid_depth_history.len() >= 2 {
            let oldest = *self.bid_depth_history.front().unwrap();
            let newest = *self.bid_depth_history.back().unwrap();

            if oldest > 0.0 {
                let drop_pct = ((oldest - newest) / oldest) * 100.0;
                if drop_pct > self.config.bid_depth_drop_threshold_pct {
                    self.trigger_pause();
                    warn!(
                        "[adverse] Bid depth dropped {:.1}% in {} ticks — pausing entries for {}ms",
                        drop_pct,
                        self.bid_depth_history.len(),
                        self.config.pause_duration_ms,
                    );
                }
            }
        }
    }

    /// Check if long entries should be paused.
    pub fn is_long_paused(&self) -> bool {
        if let Some(until) = self.pause_until {
            Instant::now() < until
        } else {
            false
        }
    }

    /// Check if short entries should be paused (ask-side spoofing).
    pub fn is_short_paused(&self) -> bool {
        // Check ask depth for similar pattern
        if self.ask_depth_history.len() >= 2 {
            let oldest = *self.ask_depth_history.front().unwrap();
            let newest = *self.ask_depth_history.back().unwrap();
            if oldest > 0.0 {
                let drop_pct = ((oldest - newest) / oldest) * 100.0;
                if drop_pct > self.config.bid_depth_drop_threshold_pct {
                    return true;
                }
            }
        }
        false
    }

    fn trigger_pause(&mut self) {
        self.pause_until = Some(
            Instant::now() + Duration::from_millis(self.config.pause_duration_ms),
        );
        self.total_pauses += 1;
    }
}

impl Default for AdverseSelectionGuard {
    fn default() -> Self {
        Self::new(AdverseSelectionConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_entry_maker_default() {
        let router = SmartEntryRouter::default();
        let (decision, price) = router.decide(true, 50000.0, 50010.0, 0.3, 0.1, false, Some(0.01));
        assert_eq!(decision, EntryDecision::PostMaker);
        // FEATURE 12: Price should be best_bid + tick_size = 50000.0 + 0.01 = 50000.01
        assert!((price - 50000.01).abs() < 0.001);
    }

    #[test]
    fn test_smart_entry_taker_on_high_vpin() {
        let router = SmartEntryRouter::default();
        let (decision, price) = router.decide(true, 50000.0, 50010.0, 0.8, 0.1, false, Some(0.01));
        assert_eq!(decision, EntryDecision::CrossSpread);
        assert_eq!(price, 50010.0); // best ask for buy (crossing spread)
    }

    #[test]
    fn test_smart_entry_pause_on_adverse() {
        let router = SmartEntryRouter::default();
        let (decision, _) = router.decide(true, 50000.0, 50010.0, 0.3, 0.1, true, Some(0.01));
        assert_eq!(decision, EntryDecision::PauseEntry);
    }

    #[test]
    fn test_volatility_trailing() {
        let mut vts = VolatilityTrailingStop::default();
        // Feed some price data
        for i in 0..20 {
            let base = 100.0 + (i as f64 * 0.1);
            vts.update_tick(base + 0.5, base - 0.5, base, 1000.0);
        }
        assert!(vts.is_ready());
        let trail = vts.calculate_trail_distance(100.0, 1000.0);
        assert!(trail > 0.0);
        assert!(trail < 100.0); // Sanity check
    }

    #[test]
    fn test_adverse_selection_bid_drop() {
        let mut guard = AdverseSelectionGuard::new(AdverseSelectionConfig {
            bid_depth_drop_threshold_pct: 20.0,
            pause_duration_ms: 500,
            lookback_ticks: 3,
            ..Default::default()
        });

        // Normal depth
        guard.update(100000.0, 100000.0);
        guard.update(95000.0, 100000.0);
        assert!(!guard.is_long_paused());

        // Sudden massive bid drop (50% in 3 ticks — spoof removal)
        guard.update(50000.0, 100000.0);
        assert!(guard.is_long_paused());
        assert_eq!(guard.total_pauses, 1);
    }
}
