//! CATEGORY 5: Tick-by-Tick Spread Analytics & Imbalance Decay Tracking.
//!
//! Provides three critical market microstructure analytics that were missing:
//!
//! 1. **Spread Autocorrelation**: Measures how predictable spread movements are.
//!    High autocorrelation = spread is mean-reverting (good for market-making).
//!
//! 2. **Spread Mean-Reversion Speed**: Half-life of spread deviations from
//!    the equilibrium. Faster reversion = tighter quotes are safe.
//!
//! 3. **Order Book Imbalance Decay**: How quickly imbalance reverts to zero
//!    after a spike. Fast decay = imbalance is noise. Slow decay = real signal.
//!
//! 4. **Trade Arrival Rate (Poisson Intensity)**: Average trades per second
//!    for regime detection. High intensity = trending/volatile regime.
//!    Low intensity = range-bound/quiet regime.

use std::collections::VecDeque;
use tracing::debug;

/// Window size for spread autocorrelation calculation.
const SPREAD_WINDOW: usize = 500;
/// Window size for imbalance decay tracking.
const IMBALANCE_DECAY_WINDOW: usize = 200;
/// Window size for trade arrival rate.
const ARRIVAL_RATE_WINDOW_SECS: f64 = 300.0; // 5 minutes

/// Tick-by-tick spread analytics engine.
pub struct SpreadAnalytics {
    /// Recent spread observations (in basis points).
    spreads: VecDeque<f64>,
    /// Timestamps of spread observations (nanoseconds).
    spread_times_ns: VecDeque<u64>,
    /// Running sum for mean calculation.
    spread_sum: f64,
    /// Running sum of squares for variance.
    spread_sum_sq: f64,
    /// Lag-1 product sum for autocorrelation.
    lag1_product_sum: f64,
    /// Previous spread value for autocorrelation.
    prev_spread: f64,
    /// Cached autocorrelation coefficient.
    cached_autocorrelation: f64,
    /// Cached mean-reversion half-life (seconds).
    cached_halflife_secs: f64,
}

impl SpreadAnalytics {
    /// Create a new spread analytics engine.
    pub fn new() -> Self {
        Self {
            spreads: VecDeque::with_capacity(SPREAD_WINDOW),
            spread_times_ns: VecDeque::with_capacity(SPREAD_WINDOW),
            spread_sum: 0.0,
            spread_sum_sq: 0.0,
            lag1_product_sum: 0.0,
            prev_spread: 0.0,
            cached_autocorrelation: 0.0,
            cached_halflife_secs: f64::INFINITY,
        }
    }

    /// Update with a new spread observation.
    ///
    /// # Arguments
    /// * `spread_bps` - Current bid-ask spread in basis points
    /// * `timestamp_ns` - Observation timestamp in nanoseconds
    pub fn update(&mut self, spread_bps: f64, timestamp_ns: u64) {
        // Evict oldest if window is full
        if self.spreads.len() >= SPREAD_WINDOW {
            if let Some(old) = self.spreads.pop_front() {
                self.spread_sum -= old;
                self.spread_sum_sq -= old * old;
            }
            self.spread_times_ns.pop_front();
        }

        // Update lag-1 autocorrelation accumulators
        if !self.spreads.is_empty() {
            let mean = self.spread_sum / self.spreads.len() as f64;
            let centered_prev = self.prev_spread - mean;
            let centered_curr = spread_bps - mean;
            self.lag1_product_sum += centered_prev * centered_curr;
        }

        // Add new observation
        self.spreads.push_back(spread_bps);
        self.spread_times_ns.push_back(timestamp_ns);
        self.spread_sum += spread_bps;
        self.spread_sum_sq += spread_bps * spread_bps;
        self.prev_spread = spread_bps;

        // Recompute autocorrelation periodically (every 50 observations)
        if self.spreads.len() >= 50 && self.spreads.len() % 50 == 0 {
            self.compute_autocorrelation();
            self.compute_halflife();
        }
    }

    /// Compute lag-1 autocorrelation of spread series.
    fn compute_autocorrelation(&mut self) {
        let n = self.spreads.len() as f64;
        if n < 10.0 {
            self.cached_autocorrelation = 0.0;
            return;
        }

        let mean = self.spread_sum / n;
        let variance = (self.spread_sum_sq / n) - (mean * mean);

        if variance <= 0.0 {
            self.cached_autocorrelation = 0.0;
            return;
        }

        // Compute lag-1 autocorrelation properly
        let mut covar = 0.0;
        let spreads: Vec<f64> = self.spreads.iter().copied().collect();
        for i in 1..spreads.len() {
            covar += (spreads[i] - mean) * (spreads[i - 1] - mean);
        }
        covar /= (n - 1.0);

        self.cached_autocorrelation = (covar / variance).clamp(-1.0, 1.0);
    }

    /// Compute mean-reversion half-life using Ornstein-Uhlenbeck model.
    ///
    /// Half-life = -ln(2) / ln(autocorrelation)
    /// This gives the time (in observation periods) for a spread deviation
    /// to revert halfway back to the mean.
    fn compute_halflife(&mut self) {
        if self.cached_autocorrelation <= 0.0 || self.cached_autocorrelation >= 1.0 {
            self.cached_halflife_secs = f64::INFINITY;
            return;
        }

        let halflife_periods = -0.693_147_2 / self.cached_autocorrelation.ln();

        // Convert periods to seconds using average inter-observation time
        let avg_interval_ns = if self.spread_times_ns.len() >= 2 {
            let first = *self.spread_times_ns.front().unwrap();
            let last = *self.spread_times_ns.back().unwrap();
            if last > first {
                (last - first) as f64 / (self.spread_times_ns.len() - 1) as f64
            } else {
                1_000_000_000.0 // Default 1 second
            }
        } else {
            1_000_000_000.0
        };

        self.cached_halflife_secs = halflife_periods * avg_interval_ns / 1_000_000_000.0;
    }

    /// Get the current spread autocorrelation coefficient [-1.0, 1.0].
    /// Positive = persistent spreads (trending), Negative = mean-reverting.
    pub fn autocorrelation(&self) -> f64 {
        self.cached_autocorrelation
    }

    /// Get the spread mean-reversion half-life in seconds.
    /// Lower = faster mean-reversion = safer for tight market-making quotes.
    pub fn halflife_secs(&self) -> f64 {
        self.cached_halflife_secs
    }

    /// Get the current mean spread in basis points.
    pub fn mean_spread_bps(&self) -> f64 {
        if self.spreads.is_empty() {
            return 0.0;
        }
        self.spread_sum / self.spreads.len() as f64
    }

    /// Get the current spread standard deviation.
    pub fn spread_std_bps(&self) -> f64 {
        let n = self.spreads.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.spread_sum / n;
        let variance = (self.spread_sum_sq / n) - (mean * mean);
        variance.max(0.0).sqrt()
    }

    /// Check if analytics are warmed up (enough data).
    pub fn is_warmed_up(&self) -> bool {
        self.spreads.len() >= 50
    }
}

/// Order book imbalance decay tracker.
///
/// Tracks how quickly imbalance reverts after a spike, which indicates
/// whether an imbalance signal is real (slow decay) or noise (fast decay).
pub struct ImbalanceDecayTracker {
    /// Recent imbalance observations.
    imbalances: VecDeque<(f64, u64)>, // (imbalance, timestamp_ns)
    /// Recent imbalance spikes detected.
    spikes: VecDeque<ImbalanceSpike>,
    /// Decay threshold: imbalance must exceed this to be considered a spike.
    spike_threshold: f64,
    /// Average decay rate (seconds to revert to 50% of spike magnitude).
    avg_decay_halflife_secs: f64,
    /// Number of spikes analyzed.
    spike_count: u64,
}

/// A detected imbalance spike with its decay characteristics.
#[derive(Debug, Clone)]
struct ImbalanceSpike {
    /// Peak imbalance magnitude.
    peak_imbalance: f64,
    /// Timestamp of the peak.
    peak_time_ns: u64,
    /// Time to revert to 50% of peak (nanoseconds). 0 if still active.
    halflife_ns: u64,
    /// Whether this spike has been resolved (reverted below threshold).
    resolved: bool,
}

impl ImbalanceDecayTracker {
    /// Create a new imbalance decay tracker.
    pub fn new(spike_threshold: f64) -> Self {
        Self {
            imbalances: VecDeque::with_capacity(IMBALANCE_DECAY_WINDOW),
            spikes: VecDeque::with_capacity(100),
            spike_threshold,
            avg_decay_halflife_secs: 0.0,
            spike_count: 0,
        }
    }

    /// Update with a new imbalance observation.
    pub fn update(&mut self, imbalance: f64, timestamp_ns: u64) {
        let abs_imb = imbalance.abs();

        // Track observation
        if self.imbalances.len() >= IMBALANCE_DECAY_WINDOW {
            self.imbalances.pop_front();
        }
        self.imbalances.push_back((imbalance, timestamp_ns));

        // Check for new spike
        let is_spike = abs_imb > self.spike_threshold;
        let last_was_spike = self.imbalances.len() >= 2 && {
            let prev = self.imbalances[self.imbalances.len() - 2].0.abs();
            prev <= self.spike_threshold
        };

        if is_spike && last_was_spike {
            // New spike detected
            self.spikes.push_back(ImbalanceSpike {
                peak_imbalance: abs_imb,
                peak_time_ns: timestamp_ns,
                halflife_ns: 0,
                resolved: false,
            });
            if self.spikes.len() > 100 {
                self.spikes.pop_front();
            }
        }

        // Check if any active spikes have decayed to 50%
        for spike in self.spikes.iter_mut() {
            if spike.resolved {
                continue;
            }
            if abs_imb <= spike.peak_imbalance * 0.5 && timestamp_ns > spike.peak_time_ns {
                spike.halflife_ns = timestamp_ns - spike.peak_time_ns;
                spike.resolved = true;
                self.spike_count += 1;

                // Update running average
                let hl_secs = spike.halflife_ns as f64 / 1_000_000_000.0;
                let n = self.spike_count as f64;
                self.avg_decay_halflife_secs =
                    (self.avg_decay_halflife_secs * (n - 1.0) + hl_secs) / n;
            }
        }
    }

    /// Get the average imbalance decay half-life in seconds.
    /// Lower = faster decay = imbalance is noise.
    /// Higher = slower decay = imbalance is a real signal.
    pub fn avg_decay_halflife(&self) -> f64 {
        self.avg_decay_halflife_secs
    }

    /// Whether the current imbalance decay rate suggests real signal.
    /// Returns true if decay is slow enough to be worth trading on.
    pub fn is_signal_quality(&self) -> bool {
        // If decay half-life > 5 seconds, the imbalance is likely real
        self.spike_count >= 3 && self.avg_decay_halflife_secs > 5.0
    }

    /// Get the number of spikes analyzed.
    pub fn spike_count(&self) -> u64 {
        self.spike_count
    }
}

/// Trade arrival rate tracker (Poisson intensity estimator).
///
/// Tracks the rate of trade arrivals to detect regime changes:
/// - High arrival rate -> trending/volatile regime
/// - Low arrival rate -> range-bound/quiet regime
pub struct TradeArrivalRate {
    /// Recent trade timestamps (nanoseconds).
    trade_times: VecDeque<u64>,
    /// Window size in nanoseconds.
    window_ns: u64,
    /// Cached arrival rate (trades per second).
    cached_rate: f64,
    /// Historical rates for regime detection.
    rate_history: VecDeque<f64>,
    /// Long-term average rate.
    long_term_avg: f64,
    /// Long-term rate count.
    long_term_count: u64,
}

impl TradeArrivalRate {
    /// Create a new trade arrival rate tracker.
    pub fn new() -> Self {
        Self {
            trade_times: VecDeque::with_capacity(10000),
            window_ns: (ARRIVAL_RATE_WINDOW_SECS * 1_000_000_000.0) as u64,
            cached_rate: 0.0,
            rate_history: VecDeque::with_capacity(1000),
            long_term_avg: 0.0,
            long_term_count: 0,
        }
    }

    /// Record a new trade arrival.
    pub fn record_trade(&mut self, timestamp_ns: u64) {
        // Prune expired trades
        let cutoff = timestamp_ns.saturating_sub(self.window_ns);
        while let Some(&front) = self.trade_times.front() {
            if front < cutoff {
                self.trade_times.pop_front();
            } else {
                break;
            }
        }

        self.trade_times.push_back(timestamp_ns);

        // Compute rate (trades per second in the window)
        if self.trade_times.len() >= 2 {
            let first = *self.trade_times.front().unwrap();
            let last = *self.trade_times.back().unwrap();
            let duration_secs = (last - first) as f64 / 1_000_000_000.0;
            if duration_secs > 0.0 {
                self.cached_rate = (self.trade_times.len() - 1) as f64 / duration_secs;
            }
        }

        // Update rate history every 100 trades
        if self.trade_times.len() % 100 == 0 {
            self.rate_history.push_back(self.cached_rate);
            if self.rate_history.len() > 1000 {
                self.rate_history.pop_front();
            }

            // Update long-term average
            self.long_term_count += 1;
            let n = self.long_term_count as f64;
            self.long_term_avg = (self.long_term_avg * (n - 1.0) + self.cached_rate) / n;
        }
    }

    /// Get the current trade arrival rate (trades per second).
    pub fn current_rate(&self) -> f64 {
        self.cached_rate
    }

    /// Get the long-term average arrival rate.
    pub fn long_term_rate(&self) -> f64 {
        self.long_term_avg
    }

    /// Get the arrival rate z-score (how many standard deviations from average).
    /// High z-score = unusually high activity = potential regime change.
    pub fn rate_zscore(&self) -> f64 {
        if self.rate_history.len() < 10 {
            return 0.0;
        }

        let n = self.rate_history.len() as f64;
        let mean: f64 = self.rate_history.iter().sum::<f64>() / n;
        let variance: f64 = self.rate_history.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev <= 0.0 {
            return 0.0;
        }

        (self.cached_rate - mean) / std_dev
    }

    /// Detect regime from arrival rate.
    /// Returns: "quiet", "normal", "active", "extreme"
    pub fn detect_regime(&self) -> &'static str {
        let z = self.rate_zscore();
        if z < -1.5 {
            "quiet"
        } else if z < 0.5 {
            "normal"
        } else if z < 2.0 {
            "active"
        } else {
            "extreme"
        }
    }

    /// Check if warm-up period is complete.
    pub fn is_warmed_up(&self) -> bool {
        self.trade_times.len() >= 100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_analytics_warmup() {
        let sa = SpreadAnalytics::new();
        assert!(!sa.is_warmed_up());
        assert_eq!(sa.mean_spread_bps(), 0.0);
    }

    #[test]
    fn test_spread_analytics_mean() {
        let mut sa = SpreadAnalytics::new();
        for i in 0..100 {
            sa.update(5.0, i * 1_000_000_000);
        }
        assert!((sa.mean_spread_bps() - 5.0).abs() < 0.01);
        assert!(sa.is_warmed_up());
    }

    #[test]
    fn test_imbalance_decay_tracker() {
        let mut tracker = ImbalanceDecayTracker::new(0.1);
        // Simulate a spike and decay
        tracker.update(0.05, 1_000_000_000); // Below threshold
        tracker.update(0.20, 2_000_000_000); // Spike!
        tracker.update(0.15, 3_000_000_000); // Decaying
        tracker.update(0.09, 4_000_000_000); // Below 50% of peak (0.10)
        assert!(tracker.spike_count() >= 1);
    }

    #[test]
    fn test_trade_arrival_rate() {
        let mut tar = TradeArrivalRate::new();
        let base = 1_000_000_000u64; // 1 second
        // 10 trades per second
        for i in 0..200 {
            tar.record_trade(base + i * 100_000_000); // every 100ms
        }
        assert!(tar.is_warmed_up());
        let rate = tar.current_rate();
        assert!(rate > 5.0, "Rate should be ~10 trades/sec, got {}", rate);
        assert!(rate < 15.0, "Rate should be ~10 trades/sec, got {}", rate);
    }

    #[test]
    fn test_arrival_rate_regime() {
        let tar = TradeArrivalRate::new();
        assert_eq!(tar.detect_regime(), "normal"); // Default with no data
    }
}
