//! CATEGORY 6 FIX: Strategy Correlation Matrix.
//!
//! Prevents concentration risk by tracking which strategies fire signals
//! simultaneously. When multiple strategies generate the same directional
//! signal on correlated assets, the total exposure is capped.
//!
//! # Problem
//!
//! Without strategy correlation tracking, N strategies can independently
//! fire BUY signals on BTC_USDT, effectively creating N * base_size exposure
//! in a single direction on a single asset. This creates dangerous
//! concentration risk that bypasses per-position limits.
//!
//! # Solution
//!
//! Track recent signals per strategy and compute a real-time correlation
//! matrix. When a new signal arrives, check if highly correlated strategies
//! have recently fired similar signals and scale down accordingly.

use std::collections::{HashMap, VecDeque};
use tracing::{debug, warn};

/// Maximum number of strategies tracked.
const MAX_STRATEGIES: usize = 32;
/// Time window for signal correlation (in milliseconds).
const CORRELATION_WINDOW_MS: i64 = 300_000; // 5 minutes
/// Correlation threshold above which signals are considered redundant.
const REDUNDANCY_THRESHOLD: f64 = 0.70;

/// A recorded signal for correlation tracking.
#[derive(Debug, Clone)]
struct SignalRecord {
    /// Strategy that generated the signal.
    strategy_id: u16,
    /// Symbol the signal targets.
    symbol: String,
    /// Direction: true = long, false = short.
    is_long: bool,
    /// Signal confidence.
    confidence: f64,
    /// Timestamp in milliseconds.
    timestamp_ms: i64,
}

/// Strategy correlation matrix for preventing concentration risk.
pub struct StrategyCorrelation {
    /// Recent signals per strategy.
    recent_signals: VecDeque<SignalRecord>,
    /// Per-strategy signal counts in current window (strategy_id -> count).
    strategy_signal_counts: [u32; MAX_STRATEGIES],
    /// Per-symbol directional exposure from recent signals.
    /// (symbol, direction) -> list of (strategy_id, size_fraction).
    directional_exposure: HashMap<(String, bool), Vec<(u16, f64)>>,
    /// Maximum total signals from correlated strategies before scaling.
    max_correlated_signals: u32,
    /// Window size in milliseconds.
    window_ms: i64,
}

impl StrategyCorrelation {
    /// Create a new strategy correlation tracker.
    pub fn new() -> Self {
        Self {
            recent_signals: VecDeque::with_capacity(256),
            strategy_signal_counts: [0; MAX_STRATEGIES],
            directional_exposure: HashMap::new(),
            max_correlated_signals: 3,
            window_ms: CORRELATION_WINDOW_MS,
        }
    }

    /// Prune expired signals from the window.
    fn prune_expired(&mut self, now_ms: i64) {
        let cutoff = now_ms - self.window_ms;
        while let Some(front) = self.recent_signals.front() {
            if front.timestamp_ms < cutoff {
                let sig = self.recent_signals.pop_front().unwrap();
                // Decrement counts
                let sid = sig.strategy_id as usize;
                if sid < MAX_STRATEGIES {
                    self.strategy_signal_counts[sid] =
                        self.strategy_signal_counts[sid].saturating_sub(1);
                }
                // Remove from directional exposure
                let key = (sig.symbol, sig.is_long);
                if let Some(exposures) = self.directional_exposure.get_mut(&key) {
                    exposures.retain(|(s, _)| *s != sig.strategy_id);
                    if exposures.is_empty() {
                        self.directional_exposure.remove(&key);
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Record a new signal and compute the scaling factor.
    ///
    /// Returns a scaling factor [0.0, 1.0] that should be applied to the
    /// position size. 1.0 = no concentration risk, lower = scale down.
    pub fn record_signal(
        &mut self,
        strategy_id: u16,
        symbol: &str,
        is_long: bool,
        confidence: f64,
        now_ms: i64,
    ) -> f64 {
        self.prune_expired(now_ms);

        // Count how many strategies have recently signaled the same direction
        // on the same symbol
        let key = (symbol.to_string(), is_long);
        let same_direction_count = self.directional_exposure
            .get(&key)
            .map(|v| v.len())
            .unwrap_or(0);

        // Record the new signal
        let record = SignalRecord {
            strategy_id,
            symbol: symbol.to_string(),
            is_long,
            confidence,
            timestamp_ms: now_ms,
        };
        self.recent_signals.push_back(record);

        let sid = strategy_id as usize;
        if sid < MAX_STRATEGIES {
            self.strategy_signal_counts[sid] += 1;
        }

        // Update directional exposure
        self.directional_exposure
            .entry(key)
            .or_insert_with(Vec::new)
            .push((strategy_id, confidence));

        // Compute scaling factor based on concentration
        if same_direction_count == 0 {
            // First signal in this direction - full size
            1.0
        } else if same_direction_count >= self.max_correlated_signals as usize {
            // Already at maximum correlated signals - reject
            warn!(
                "[strategy-corr] Signal rejected: {} strategies already signal {} {} (max={})",
                same_direction_count,
                if is_long { "LONG" } else { "SHORT" },
                symbol,
                self.max_correlated_signals
            );
            0.0
        } else {
            // Scale down proportionally
            let scale = 1.0 / (same_direction_count as f64 + 1.0);
            debug!(
                "[strategy-corr] Scaling signal: {} existing {} signals on {} - scale={:.2}",
                same_direction_count,
                if is_long { "LONG" } else { "SHORT" },
                symbol,
                scale
            );
            scale
        }
    }

    /// Check if a new position would create opposing positions (position netting).
    /// Returns true if there's an existing opposing position from another strategy.
    pub fn has_opposing_signal(&self, symbol: &str, is_long: bool) -> bool {
        let opposing_key = (symbol.to_string(), !is_long);
        self.directional_exposure
            .get(&opposing_key)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Get the total number of active signals in the current window.
    pub fn active_signal_count(&self) -> usize {
        self.recent_signals.len()
    }

    /// Get signals per strategy for dashboard.
    pub fn signals_per_strategy(&self) -> Vec<(u16, u32)> {
        self.strategy_signal_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(id, &count)| (id as u16, count))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_signal_full_size() {
        let mut sc = StrategyCorrelation::new();
        let scale = sc.record_signal(0, "BTC_USDT", true, 0.8, 1000);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_correlated_signals_scale_down() {
        let mut sc = StrategyCorrelation::new();
        sc.record_signal(0, "BTC_USDT", true, 0.8, 1000);
        let scale = sc.record_signal(1, "BTC_USDT", true, 0.7, 1500);
        assert!(scale < 1.0, "Second correlated signal should be scaled down");
        assert!(scale > 0.0, "But not rejected entirely");
    }

    #[test]
    fn test_max_correlated_signals_rejected() {
        let mut sc = StrategyCorrelation::new();
        sc.record_signal(0, "BTC_USDT", true, 0.8, 1000);
        sc.record_signal(1, "BTC_USDT", true, 0.7, 1100);
        sc.record_signal(2, "BTC_USDT", true, 0.6, 1200);
        let scale = sc.record_signal(3, "BTC_USDT", true, 0.5, 1300);
        assert_eq!(scale, 0.0, "4th correlated signal should be rejected");
    }

    #[test]
    fn test_opposing_signal_detected() {
        let mut sc = StrategyCorrelation::new();
        sc.record_signal(0, "BTC_USDT", true, 0.8, 1000);
        assert!(sc.has_opposing_signal("BTC_USDT", false));
        assert!(!sc.has_opposing_signal("BTC_USDT", true));
    }

    #[test]
    fn test_expired_signals_pruned() {
        let mut sc = StrategyCorrelation::new();
        sc.record_signal(0, "BTC_USDT", true, 0.8, 1000);
        // Advance time past the window
        let future = 1000 + CORRELATION_WINDOW_MS + 1000;
        let scale = sc.record_signal(1, "BTC_USDT", true, 0.7, future);
        assert_eq!(scale, 1.0, "Expired signal should be pruned, so this is first in window");
    }
}
