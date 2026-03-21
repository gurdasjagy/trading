//! Realized Volatility Calculator — Phase 2 Feature 9.
//!
//! Implements Parkinson's high-low range estimator for realized volatility
//! with regime classification and position scaling.
//!
//! # Parkinson Estimator
//!
//! σ² = (1 / (4 ln 2)) × (1/n) × Σ[ln(H_i / L_i)]²
//!
//! Where:
//! - H_i = high price in period i
//! - L_i = low price in period i
//! - n = number of periods
//!
//! Annualization: multiply by sqrt(252 × periods_per_day)
//! For 5-minute bars: sqrt(252 × 288) = sqrt(72576) ≈ 269.4
//!
//! # Volatility Regimes
//!
//! - Low: < 15% annualized → 2x position size
//! - Normal: 15-40% → 1x position size
//! - High: 40-80% → 0.5x position size
//! - Extreme: > 80% → 0.25x position size

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Volatility Regime
// ═══════════════════════════════════════════════════════════════════════════

/// Volatility regime classification based on annualized realized volatility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegime {
    /// Low volatility (< 15% annualized).
    Low,
    /// Normal volatility (15-40% annualized).
    Normal,
    /// High volatility (40-80% annualized).
    High,
    /// Extreme volatility (> 80% annualized).
    Extreme,
}

impl VolatilityRegime {
    /// Convert regime to string representation.
    pub fn to_string(&self) -> &'static str {
        match self {
            VolatilityRegime::Low => "low",
            VolatilityRegime::Normal => "normal",
            VolatilityRegime::High => "high",
            VolatilityRegime::Extreme => "extreme",
        }
    }

    /// Get position size scaling factor for this regime.
    ///
    /// Returns inverse volatility scaling:
    /// - Low vol → 2.0x (increase size)
    /// - Normal → 1.0x (baseline)
    /// - High → 0.5x (reduce size)
    /// - Extreme → 0.25x (aggressive reduction)
    pub fn get_scale_factor(&self) -> f64 {
        match self {
            VolatilityRegime::Low => 2.0,
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::High => 0.5,
            VolatilityRegime::Extreme => 0.25,
        }
    }

    /// Classify volatility into a regime.
    ///
    /// # Arguments
    /// * `vol_pct` — Annualized volatility in percentage (e.g., 25.0 = 25%)
    pub fn from_volatility(vol_pct: f64) -> Self {
        if vol_pct < 15.0 {
            VolatilityRegime::Low
        } else if vol_pct < 40.0 {
            VolatilityRegime::Normal
        } else if vol_pct < 80.0 {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Realized Volatility Calculator
// ═══════════════════════════════════════════════════════════════════════════

/// Realized volatility calculator using Parkinson's high-low range estimator.
///
/// Maintains a rolling window of (high, low, timestamp) tuples and computes
/// annualized volatility using the Parkinson formula.
pub struct RealizedVolatilityCalculator {
    /// Rolling window of (high, low, timestamp_ns) tuples.
    window: VecDeque<(f64, f64, u64)>,
    /// Window size in seconds (e.g., 300 for 5 minutes).
    window_size_secs: u64,
    /// Cached volatility value (updated on each tick).
    cached_volatility: f64,
    /// Cached regime (updated on each tick).
    cached_regime: VolatilityRegime,
}

impl RealizedVolatilityCalculator {
    /// Create a new realized volatility calculator.
    ///
    /// # Arguments
    /// * `window_size_secs` — Window size in seconds (e.g., 300 for 5 minutes)
    pub fn new(window_size_secs: u64) -> Self {
        Self {
            window: VecDeque::with_capacity(1000),
            window_size_secs,
            cached_volatility: 0.0,
            cached_regime: VolatilityRegime::Normal,
        }
    }

    /// Update with a new tick.
    ///
    /// # Arguments
    /// * `timestamp_ns` — Tick timestamp in nanoseconds
    /// * `high` — High price in this tick
    /// * `low` — Low price in this tick
    /// * `close` — Close price in this tick (unused in Parkinson, but kept for API consistency)
    pub fn on_tick(&mut self, timestamp_ns: u64, high: f64, low: f64, _close: f64) {
        // Add new tick
        self.window.push_back((high, low, timestamp_ns));

        // Evict old ticks outside the window
        let cutoff_ns = timestamp_ns.saturating_sub(self.window_size_secs * 1_000_000_000);
        while let Some(&(_, _, ts)) = self.window.front() {
            if ts < cutoff_ns {
                self.window.pop_front();
            } else {
                break;
            }
        }

        // Recalculate volatility
        self.cached_volatility = self.calculate_parkinson_volatility();
        self.cached_regime = VolatilityRegime::from_volatility(self.cached_volatility);
    }

    /// Calculate Parkinson volatility from the current window.
    ///
    /// Returns annualized volatility in percentage (e.g., 25.0 = 25%).
    fn calculate_parkinson_volatility(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }

        let n = self.window.len() as f64;
        let sum_sq: f64 = self.window.iter()
            .filter_map(|(high, low, _)| {
                if *high > 0.0 && *low > 0.0 && high >= low {
                    let ratio = high / low;
                    Some(ratio.ln().powi(2))
                } else {
                    None
                }
            })
            .sum();

        if sum_sq == 0.0 {
            return 0.0;
        }

        // Parkinson formula: σ² = (1 / (4 ln 2)) × (1/n) × Σ[ln(H/L)]²
        let ln2 = std::f64::consts::LN_2;
        let variance = (1.0 / (4.0 * ln2)) * (sum_sq / n);
        let std_dev = variance.sqrt();

        // Annualize: assume 5-minute bars, 288 bars per day, 252 trading days
        // Annualization factor = sqrt(252 × 288) ≈ 269.4
        let annualization_factor = (252.0 * 288.0_f64).sqrt();
        let annualized_vol = std_dev * annualization_factor;

        // Convert to percentage
        annualized_vol * 100.0
    }

    /// Get the current volatility regime.
    #[inline]
    pub fn get_regime(&self) -> VolatilityRegime {
        self.cached_regime
    }

    /// Get the current annualized volatility in percentage.
    #[inline]
    pub fn get_volatility(&self) -> f64 {
        self.cached_volatility
    }

    /// Get position size scaling factor based on current regime.
    ///
    /// Returns inverse volatility scaling (higher vol → smaller size).
    #[inline]
    pub fn get_position_scale(&self) -> f64 {
        self.cached_regime.get_scale_factor()
    }

    /// Check if the calculator is warmed up (has enough data).
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.window.len() >= 10
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_regime_classification() {
        assert_eq!(VolatilityRegime::from_volatility(10.0), VolatilityRegime::Low);
        assert_eq!(VolatilityRegime::from_volatility(25.0), VolatilityRegime::Normal);
        assert_eq!(VolatilityRegime::from_volatility(50.0), VolatilityRegime::High);
        assert_eq!(VolatilityRegime::from_volatility(100.0), VolatilityRegime::Extreme);
    }

    #[test]
    fn test_regime_scale_factors() {
        assert_eq!(VolatilityRegime::Low.get_scale_factor(), 2.0);
        assert_eq!(VolatilityRegime::Normal.get_scale_factor(), 1.0);
        assert_eq!(VolatilityRegime::High.get_scale_factor(), 0.5);
        assert_eq!(VolatilityRegime::Extreme.get_scale_factor(), 0.25);
    }

    #[test]
    fn test_parkinson_basic() {
        let mut calc = RealizedVolatilityCalculator::new(300);
        let base_ts = 1_000_000_000_000u64;

        // Add 20 ticks with 1% high-low range
        for i in 0..20 {
            let price = 50000.0;
            let high = price * 1.005;
            let low = price * 0.995;
            calc.on_tick(base_ts + i * 1_000_000_000, high, low, price);
        }

        assert!(calc.is_ready());
        let vol = calc.get_volatility();
        // Should be non-zero and reasonable (< 100%)
        assert!(vol > 0.0 && vol < 100.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut calc = RealizedVolatilityCalculator::new(10); // 10-second window
        let base_ts = 1_000_000_000_000u64;

        // Add 20 ticks over 20 seconds
        for i in 0..20 {
            calc.on_tick(base_ts + i * 1_000_000_000, 50000.0, 49000.0, 49500.0);
        }

        // Should only keep last 10 seconds worth
        assert!(calc.window.len() <= 11); // Allow for boundary conditions
    }

    #[test]
    fn test_zero_volatility() {
        let mut calc = RealizedVolatilityCalculator::new(300);
        let base_ts = 1_000_000_000_000u64;

        // Add ticks with no price movement
        for i in 0..20 {
            calc.on_tick(base_ts + i * 1_000_000_000, 50000.0, 50000.0, 50000.0);
        }

        // Volatility should be zero or very close
        assert!(calc.get_volatility() < 1.0);
    }
}
