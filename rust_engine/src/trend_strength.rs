//! Multi-Timeframe Trend Strength Index (TSI) — Phase 2 Feature 5.
//!
//! Combines EMA crossover state, RSI zones, and ADX trend strength across
//! multiple timeframes (1m, 5m, 15m, 1h) to produce a composite trend score.
//!
//! TSI Score Range: [0.0, 1.0]
//!   - 0.0-0.3: Weak/no trend (reduce position size)
//!   - 0.3-0.7: Moderate trend (normal sizing)
//!   - 0.7-1.0: Strong trend (increase position size)
//!
//! Follows the pattern from cumulative_delta.rs for struct layout and
//! gamma_shm.rs for calculation logic.

use crate::candle_aggregator::{CandleAggregator, Timeframe};

// ═══════════════════════════════════════════════════════════════════════════
// Trend Strength Index
// ═══════════════════════════════════════════════════════════════════════════

/// Multi-timeframe trend strength calculator.
pub struct TrendStrengthIndex {
    /// Weight for 1-minute timeframe.
    m1_weight: f64,
    /// Weight for 5-minute timeframe.
    m5_weight: f64,
    /// Weight for 15-minute timeframe.
    m15_weight: f64,
    /// Weight for 1-hour timeframe.
    h1_weight: f64,
}

impl TrendStrengthIndex {
    /// Create a new TSI calculator with default weights.
    ///
    /// Weights are calibrated to favor longer timeframes:
    ///   - M1: 10% (noise filter)
    ///   - M5: 20% (short-term momentum)
    ///   - M15: 30% (medium-term trend)
    ///   - H1: 40% (primary trend)
    pub fn new() -> Self {
        Self {
            m1_weight: 0.1,
            m5_weight: 0.2,
            m15_weight: 0.3,
            h1_weight: 0.4,
        }
    }

    /// Calculate the composite TSI score from candle aggregator.
    ///
    /// # Algorithm
    /// For each timeframe:
    ///   1. EMA Crossover Score: +1 if EMA9 > EMA21 (bullish), -1 if bearish, 0 if neutral
    ///   2. RSI Zone Score: +1 if RSI > 50 (bullish), -1 if < 50 (bearish)
    ///   3. ADX Strength: Scale by ADX/100 (0.0-1.0)
    ///   4. Timeframe Score = (EMA + RSI) / 2 * ADX
    ///   5. Weighted TSI = Σ(timeframe_score * weight)
    ///
    /// Returns: TSI score in range [0.0, 1.0]
    pub fn calculate_tsi(&self, candles: &parking_lot::MutexGuard<CandleAggregator>) -> f64 {
        let mut weighted_sum = 0.0;

        // M1 timeframe
        if candles.is_ready(Timeframe::M1) {
            if let Some(candle) = candles.get_candle(Timeframe::M1) {
                let score = self.calculate_timeframe_score(candle.ema20, candle.ema50, candle.rsi14, candle.adx14);
                weighted_sum += score * self.m1_weight;
            }
        }

        // M5 timeframe
        if candles.is_ready(Timeframe::M5) {
            if let Some(candle) = candles.get_candle(Timeframe::M5) {
                let score = self.calculate_timeframe_score(candle.ema20, candle.ema50, candle.rsi14, candle.adx14);
                weighted_sum += score * self.m5_weight;
            }
        }

        // M15 timeframe
        if candles.is_ready(Timeframe::M15) {
            if let Some(candle) = candles.get_candle(Timeframe::M15) {
                let score = self.calculate_timeframe_score(candle.ema20, candle.ema50, candle.rsi14, candle.adx14);
                weighted_sum += score * self.m15_weight;
            }
        }

        // H1 timeframe
        if candles.is_ready(Timeframe::H1) {
            if let Some(candle) = candles.get_candle(Timeframe::H1) {
                let score = self.calculate_timeframe_score(candle.ema20, candle.ema50, candle.rsi14, candle.adx14);
                weighted_sum += score * self.h1_weight;
            }
        }

        // Normalize to [0.0, 1.0] range
        // Raw score is in [-1.0, 1.0], so we shift and scale
        (weighted_sum + 1.0) / 2.0
    }

    /// Calculate trend score for a single timeframe.
    ///
    /// # Arguments
    /// * `ema20` — 20-period EMA
    /// * `ema50` — 50-period EMA
    /// * `rsi` — RSI value (0-100)
    /// * `adx` — ADX value (0-100)
    ///
    /// # Returns
    /// Score in range [-1.0, 1.0] where:
    ///   - +1.0 = strong bullish trend
    ///   - 0.0 = no trend
    ///   - -1.0 = strong bearish trend
    fn calculate_timeframe_score(&self, ema20: f64, ema50: f64, rsi: f64, adx: f64) -> f64 {
        // EMA crossover score
        let ema_score = if ema20 > ema50 {
            1.0
        } else if ema20 < ema50 {
            -1.0
        } else {
            0.0
        };

        // RSI zone score
        let rsi_score = if rsi > 50.0 {
            1.0
        } else if rsi < 50.0 {
            -1.0
        } else {
            0.0
        };

        // Combine EMA and RSI
        let directional_score = (ema_score + rsi_score) / 2.0;

        // Scale by ADX strength (0-100 → 0.0-1.0)
        let adx_strength = (adx / 100.0).clamp(0.0, 1.0);

        // Final score: direction * strength
        directional_score * adx_strength
    }
}

impl Default for TrendStrengthIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_score_bullish() {
        let tsi = TrendStrengthIndex::new();
        // Strong bullish: EMA20 > EMA50, RSI > 50, high ADX
        let score = tsi.calculate_timeframe_score(100.0, 95.0, 70.0, 80.0);
        assert!(score > 0.5, "Expected bullish score > 0.5, got {}", score);
    }

    #[test]
    fn test_timeframe_score_bearish() {
        let tsi = TrendStrengthIndex::new();
        // Strong bearish: EMA20 < EMA50, RSI < 50, high ADX
        let score = tsi.calculate_timeframe_score(95.0, 100.0, 30.0, 80.0);
        assert!(score < -0.5, "Expected bearish score < -0.5, got {}", score);
    }

    #[test]
    fn test_timeframe_score_weak_trend() {
        let tsi = TrendStrengthIndex::new();
        // Weak trend: low ADX
        let score = tsi.calculate_timeframe_score(100.0, 95.0, 70.0, 10.0);
        assert!(score.abs() < 0.2, "Expected weak trend score near 0, got {}", score);
    }

    #[test]
    fn test_tsi_weights_sum_to_one() {
        let tsi = TrendStrengthIndex::new();
        let sum = tsi.m1_weight + tsi.m5_weight + tsi.m15_weight + tsi.h1_weight;
        assert!((sum - 1.0).abs() < 0.001, "Weights should sum to 1.0, got {}", sum);
    }
}
