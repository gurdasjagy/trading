//! Regime Adapter — Typed wrapper around `regime_shm.rs` for the bridge.
//!
//! Provides a clean, typed interface for reading market regime data from
//! Python's shared memory. Integrates with the circuit breaker for
//! automatic regime-based risk adjustment.
//!
//! # Regime Types
//!
//! Python ML models classify the market into regimes:
//!   - **RiskOn**: Normal trading, full position sizes
//!   - **RiskOff**: Reduced activity, smaller positions
//!   - **Crisis**: Extreme volatility, minimal trading
//!   - **Trending**: Strong directional movement, trend-following only
//!   - **MeanReverting**: Range-bound, mean-reversion strategies favored
//!   - **Unknown**: Insufficient data, conservative defaults

use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;

/// Market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Normal market conditions — trade at full size.
    RiskOn,
    /// Elevated uncertainty — reduce position sizes.
    RiskOff,
    /// Extreme volatility / crash — minimal or no trading.
    Crisis,
    /// Strong directional trend — trend-following strategies.
    Trending,
    /// Range-bound market — mean-reversion strategies.
    MeanReverting,
    /// Insufficient data — use conservative defaults.
    Unknown,
}

impl MarketRegime {
    /// Convert a regime string from Python to the enum.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "risk_on" | "riskon" | "normal" => MarketRegime::RiskOn,
            "risk_off" | "riskoff" | "cautious" => MarketRegime::RiskOff,
            "crisis" | "crash" | "extreme" => MarketRegime::Crisis,
            "trending" | "trend" | "momentum" => MarketRegime::Trending,
            "mean_reverting" | "meanreverting" | "range" => MarketRegime::MeanReverting,
            _ => MarketRegime::Unknown,
        }
    }

    /// Get the default risk scaling factor for this regime.
    ///
    /// Returns a multiplier in [0.0, 1.0] applied to position sizes.
    pub fn default_risk_scale(&self) -> f64 {
        match self {
            MarketRegime::RiskOn => 1.0,
            MarketRegime::RiskOff => 0.5,
            MarketRegime::Crisis => 0.1,
            MarketRegime::Trending => 0.8,
            MarketRegime::MeanReverting => 0.6,
            MarketRegime::Unknown => 0.3,
        }
    }

    /// Should the circuit breaker be extra sensitive in this regime?
    pub fn tighten_risk_limits(&self) -> bool {
        matches!(self, MarketRegime::Crisis | MarketRegime::Unknown)
    }
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::RiskOn => write!(f, "risk_on"),
            MarketRegime::RiskOff => write!(f, "risk_off"),
            MarketRegime::Crisis => write!(f, "crisis"),
            MarketRegime::Trending => write!(f, "trending"),
            MarketRegime::MeanReverting => write!(f, "mean_reverting"),
            MarketRegime::Unknown => write!(f, "unknown"),
        }
    }
}

/// Complete regime state snapshot from Python.
#[derive(Debug, Clone)]
pub struct RegimeSnapshot {
    /// Current regime classification.
    pub regime: MarketRegime,
    /// Momentum weight (0.0 to 1.0). Python's confidence in trend signals.
    pub momentum_weight: f64,
    /// Volatility weight (0.0 to 1.0). Higher = more volatile.
    pub volatility_weight: f64,
    /// Mean-reversion weight (0.0 to 1.0).
    pub mean_reversion_weight: f64,
    /// Overall risk scale factor (0.0 to 1.0).
    pub risk_scale: f64,
    /// Timestamp of the last regime update (nanoseconds).
    pub timestamp_ns: u64,
    /// Sequence number (for change detection).
    pub sequence: u64,
}

impl Default for RegimeSnapshot {
    fn default() -> Self {
        Self {
            regime: MarketRegime::Unknown,
            momentum_weight: 0.3,
            volatility_weight: 0.5,
            mean_reversion_weight: 0.3,
            risk_scale: 0.3,
            timestamp_ns: 0,
            sequence: 0,
        }
    }
}

impl RegimeSnapshot {
    /// Check if the regime data is stale.
    pub fn is_stale(&self, max_age_ns: u64) -> bool {
        if self.timestamp_ns == 0 {
            return true;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now.saturating_sub(self.timestamp_ns) > max_age_ns
    }

    /// Get the effective risk multiplier (combining regime + explicit scale).
    pub fn effective_risk_multiplier(&self) -> f64 {
        let regime_scale = self.regime.default_risk_scale();
        (regime_scale * self.risk_scale).clamp(0.0, 1.0)
    }
}

/// Regime adapter that wraps the existing `regime_shm` module.
///
/// Provides typed access to the market regime with change detection
/// and staleness checking.
pub struct RegimeAdapter {
    /// Last known regime snapshot.
    last_snapshot: RegimeSnapshot,
    /// Last sequence number read.
    last_sequence: AtomicU64,
    /// Total regime changes observed.
    total_changes: u64,
    /// Maximum staleness before falling back to Unknown.
    max_stale_ns: u64,
}

impl RegimeAdapter {
    /// Create a new regime adapter.
    ///
    /// `max_stale_secs` is the maximum age of regime data before we
    /// fall back to `Unknown` regime (conservative).
    pub fn new(max_stale_secs: f64) -> Self {
        Self {
            last_snapshot: RegimeSnapshot::default(),
            last_sequence: AtomicU64::new(0),
            total_changes: 0,
            max_stale_ns: (max_stale_secs * 1e9) as u64,
        }
    }

    /// Create with default staleness threshold (30 seconds).
    pub fn with_defaults() -> Self {
        Self::new(30.0)
    }

    /// Update the regime from SHM data.
    ///
    /// Reads from the existing `RegimeSharedState` and converts to
    /// a typed `RegimeSnapshot`.
    pub fn update_from_shm(
        &mut self,
        momentum_weight: f64,
        volatility_weight: f64,
        mean_reversion_weight: f64,
        regime_label: &str,
        sequence: u64,
        timestamp_ns: u64,
    ) -> bool {
        let old_seq = self.last_sequence.load(Ordering::Relaxed);
        if sequence == old_seq {
            return false; // No change
        }

        let regime = MarketRegime::from_str(regime_label);
        let old_regime = self.last_snapshot.regime;

        self.last_snapshot = RegimeSnapshot {
            regime,
            momentum_weight: momentum_weight.clamp(0.0, 1.0),
            volatility_weight: volatility_weight.clamp(0.0, 1.0),
            mean_reversion_weight: mean_reversion_weight.clamp(0.0, 1.0),
            risk_scale: regime.default_risk_scale(),
            timestamp_ns,
            sequence,
        };

        self.last_sequence.store(sequence, Ordering::Relaxed);

        if regime != old_regime {
            self.total_changes += 1;
            debug!(
                "[regime-adapter] Regime changed: {} -> {} (seq={})",
                old_regime, regime, sequence
            );
        }

        true
    }

    /// Get the current regime snapshot.
    ///
    /// If the data is stale (exceeds max_stale_ns), returns a conservative
    /// `Unknown` regime with low risk scaling.
    pub fn current_regime(&self) -> RegimeSnapshot {
        let snapshot = self.last_snapshot.clone();
        if snapshot.is_stale(self.max_stale_ns) {
            RegimeSnapshot::default() // Fall back to Unknown
        } else {
            snapshot
        }
    }

    /// Get the effective risk multiplier for the current regime.
    pub fn risk_multiplier(&self) -> f64 {
        self.current_regime().effective_risk_multiplier()
    }

    /// Get total regime changes.
    #[inline]
    pub fn total_changes(&self) -> u64 {
        self.total_changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_from_str() {
        assert_eq!(MarketRegime::from_str("risk_on"), MarketRegime::RiskOn);
        assert_eq!(MarketRegime::from_str("crisis"), MarketRegime::Crisis);
        assert_eq!(MarketRegime::from_str("unknown_regime"), MarketRegime::Unknown);
    }

    #[test]
    fn test_regime_risk_scale() {
        assert!((MarketRegime::RiskOn.default_risk_scale() - 1.0).abs() < f64::EPSILON);
        assert!(MarketRegime::Crisis.default_risk_scale() < 0.2);
    }

    #[test]
    fn test_regime_adapter_update() {
        let mut adapter = RegimeAdapter::with_defaults();

        let changed = adapter.update_from_shm(
            0.8, 0.3, 0.2, "risk_on", 1, super::super::now_ns(),
        );
        assert!(changed);

        let snapshot = adapter.current_regime();
        assert_eq!(snapshot.regime, MarketRegime::RiskOn);
        assert!((snapshot.momentum_weight - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_regime_adapter_no_change() {
        let mut adapter = RegimeAdapter::with_defaults();

        adapter.update_from_shm(0.8, 0.3, 0.2, "risk_on", 1, super::super::now_ns());
        let changed = adapter.update_from_shm(0.8, 0.3, 0.2, "risk_on", 1, super::super::now_ns());
        assert!(!changed); // Same sequence
    }

    #[test]
    fn test_stale_regime_fallback() {
        let mut adapter = RegimeAdapter::new(0.001); // 1ms staleness
        adapter.update_from_shm(0.8, 0.3, 0.2, "risk_on", 1, 1); // Very old timestamp

        let snapshot = adapter.current_regime();
        assert_eq!(snapshot.regime, MarketRegime::Unknown); // Stale -> fallback
    }
}
