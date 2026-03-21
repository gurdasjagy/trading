//! Correlation-Based Position Limiter — FEATURE 6.
//!
//! Implements correlation-based exposure limits to prevent over-concentration
//! in correlated assets. Uses a hardcoded correlation matrix for BTC/ETH/SOL.
//!
//! # Correlation Matrix
//!
//! ```text
//! BTC-ETH: 0.85
//! BTC-SOL: 0.75
//! ETH-SOL: 0.70
//! ```
//!
//! # Risk Limit
//!
//! Total correlated exposure must not exceed 150% of max single position size.
//! If a new position would breach this limit, it is rejected or scaled down.
//!
//! Follows the pattern from pre_trade_risk.rs correlation check (lines 365-389).

use std::collections::HashMap;
use tracing::{info, warn};

// ═══════════════════════════════════════════════════════════════════════════
// Correlation Matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Hardcoded correlation coefficients between major crypto assets.
const CORRELATION_MATRIX: &[(&str, &str, f64)] = &[
    ("BTC_USDT", "ETH_USDT", 0.85),
    ("BTC_USDT", "SOL_USDT", 0.75),
    ("ETH_USDT", "SOL_USDT", 0.70),
];

// ═══════════════════════════════════════════════════════════════════════════
// Correlation Limiter
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks position sizes and enforces correlation-based exposure limits.
pub struct CorrelationLimiter {
    /// Current position sizes by symbol (in USDT notional).
    positions: HashMap<String, f64>,
    /// Correlation matrix: (symbol1, symbol2) -> correlation coefficient.
    correlations: HashMap<(String, String), f64>,
    /// Maximum single position size (USDT).
    max_single_position: f64,
    /// Maximum correlated exposure as a multiple of max_single_position.
    max_correlated_exposure_multiplier: f64,
}

impl CorrelationLimiter {
    /// Create a new correlation limiter.
    ///
    /// # Arguments
    /// * `max_single_position` — Maximum size for a single position (USDT)
    /// * `max_correlated_exposure_multiplier` — Max correlated exposure as multiple of single position (e.g., 1.5 = 150%)
    pub fn new(max_single_position: f64, max_correlated_exposure_multiplier: f64) -> Self {
        let mut correlations = HashMap::new();

        // Load hardcoded correlation matrix
        for &(sym1, sym2, corr) in CORRELATION_MATRIX {
            correlations.insert((sym1.to_string(), sym2.to_string()), corr);
            correlations.insert((sym2.to_string(), sym1.to_string()), corr); // Symmetric
        }

        info!(
            "[correlation-limiter] Initialized: max_single={:.2}, max_correlated_mult={:.2}",
            max_single_position, max_correlated_exposure_multiplier
        );

        Self {
            positions: HashMap::new(),
            correlations,
            max_single_position,
            max_correlated_exposure_multiplier,
        }
    }

    /// Compute the total correlated exposure for a given symbol.
    ///
    /// # Formula
    ///
    /// `correlated_exposure = Σ (position_size[other] * correlation[symbol, other])`
    ///
    /// # Arguments
    /// * `symbol` — The symbol to compute exposure for
    ///
    /// # Returns
    /// Total correlated exposure in USDT
    pub fn compute_correlated_exposure(&self, symbol: &str) -> f64 {
        let mut total = 0.0;

        for (other_symbol, &other_size) in &self.positions {
            if other_symbol == symbol {
                // Self-correlation is 1.0
                total += other_size;
            } else {
                // Look up correlation coefficient
                let key = (symbol.to_string(), other_symbol.clone());
                if let Some(&corr) = self.correlations.get(&key) {
                    total += other_size * corr;
                }
            }
        }

        total
    }

    /// Check if a new position would exceed the correlation limit.
    ///
    /// # Arguments
    /// * `symbol` — Symbol for the new position
    /// * `new_position_size` — Size of the new position (USDT notional)
    ///
    /// # Returns
    /// `Ok(())` if the position is allowed, `Err(reason)` if rejected
    pub fn check_position_limit(
        &self,
        symbol: &str,
        new_position_size: f64,
    ) -> Result<(), String> {
        // Compute current correlated exposure
        let current_exposure = self.compute_correlated_exposure(symbol);

        // Compute exposure after adding the new position
        let new_exposure = current_exposure + new_position_size;

        // Check against limit
        let max_exposure = self.max_single_position * self.max_correlated_exposure_multiplier;

        if new_exposure > max_exposure {
            let msg = format!(
                "Correlated exposure limit exceeded: {} + {:.2} = {:.2} > {:.2} (max)",
                symbol, new_position_size, new_exposure, max_exposure
            );
            warn!("[correlation-limiter] {}", msg);
            Err(msg)
        } else {
            Ok(())
        }
    }

    /// Compute the maximum allowed position size for a symbol given current exposure.
    ///
    /// # Arguments
    /// * `symbol` — Symbol to compute max size for
    ///
    /// # Returns
    /// Maximum allowed position size in USDT
    pub fn max_allowed_position_size(&self, symbol: &str) -> f64 {
        let current_exposure = self.compute_correlated_exposure(symbol);
        let max_exposure = self.max_single_position * self.max_correlated_exposure_multiplier;
        (max_exposure - current_exposure).max(0.0)
    }

    /// Update the position size for a symbol.
    ///
    /// # Arguments
    /// * `symbol` — Symbol to update
    /// * `position_size` — New position size (USDT notional). Use 0.0 to remove.
    pub fn update_position(&mut self, symbol: &str, position_size: f64) {
        if position_size <= 0.0 {
            self.positions.remove(symbol);
        } else {
            self.positions.insert(symbol.to_string(), position_size);
        }
    }

    /// Get the current position size for a symbol.
    pub fn get_position(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).copied().unwrap_or(0.0)
    }

    /// Get all current positions.
    pub fn get_all_positions(&self) -> &HashMap<String, f64> {
        &self.positions
    }

    /// Clear all positions (useful for testing or end-of-day reset).
    pub fn clear_all_positions(&mut self) {
        self.positions.clear();
    }
}

impl Default for CorrelationLimiter {
    fn default() -> Self {
        Self::new(2000.0, 1.5) // $2000 max single, 150% correlated exposure
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlated_exposure_calculation() {
        let mut limiter = CorrelationLimiter::new(1000.0, 1.5);

        // Add BTC position
        limiter.update_position("BTC_USDT", 500.0);

        // Compute correlated exposure for ETH (corr=0.85 with BTC)
        let eth_exposure = limiter.compute_correlated_exposure("ETH_USDT");
        // Expected: 500 * 0.85 = 425
        assert!((eth_exposure - 425.0).abs() < 0.01);
    }

    #[test]
    fn test_position_limit_check() {
        let mut limiter = CorrelationLimiter::new(1000.0, 1.5);

        // Add BTC position
        limiter.update_position("BTC_USDT", 800.0);

        // Try to add ETH position
        // Current ETH exposure: 800 * 0.85 = 680
        // New position: 600
        // Total: 680 + 600 = 1280
        // Limit: 1000 * 1.5 = 1500
        // Should be allowed
        assert!(limiter.check_position_limit("ETH_USDT", 600.0).is_ok());

        // Try to add a larger ETH position
        // Total: 680 + 900 = 1580 > 1500
        // Should be rejected
        assert!(limiter.check_position_limit("ETH_USDT", 900.0).is_err());
    }

    #[test]
    fn test_max_allowed_position_size() {
        let mut limiter = CorrelationLimiter::new(1000.0, 1.5);

        // Add BTC position
        limiter.update_position("BTC_USDT", 800.0);

        // Max allowed ETH position
        // Current ETH exposure: 800 * 0.85 = 680
        // Max exposure: 1500
        // Max allowed: 1500 - 680 = 820
        let max_eth = limiter.max_allowed_position_size("ETH_USDT");
        assert!((max_eth - 820.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_correlated_positions() {
        let mut limiter = CorrelationLimiter::new(1000.0, 1.5);

        // Add BTC and ETH positions
        limiter.update_position("BTC_USDT", 500.0);
        limiter.update_position("ETH_USDT", 400.0);

        // Compute correlated exposure for SOL
        // SOL-BTC: 0.75, SOL-ETH: 0.70
        // Exposure: 500 * 0.75 + 400 * 0.70 = 375 + 280 = 655
        let sol_exposure = limiter.compute_correlated_exposure("SOL_USDT");
        assert!((sol_exposure - 655.0).abs() < 0.01);
    }

    #[test]
    fn test_position_update_and_removal() {
        let mut limiter = CorrelationLimiter::new(1000.0, 1.5);

        limiter.update_position("BTC_USDT", 500.0);
        assert_eq!(limiter.get_position("BTC_USDT"), 500.0);

        limiter.update_position("BTC_USDT", 0.0);
        assert_eq!(limiter.get_position("BTC_USDT"), 0.0);
    }

    #[test]
    fn test_uncorrelated_symbol() {
        let mut limiter = CorrelationLimiter::new(1000.0, 1.5);

        // Add BTC position
        limiter.update_position("BTC_USDT", 800.0);

        // Check exposure for an uncorrelated symbol (not in matrix)
        let xrp_exposure = limiter.compute_correlated_exposure("XRP_USDT");
        // Should be 0 (no correlation data)
        assert_eq!(xrp_exposure, 0.0);

        // Should be able to add full position
        assert!(limiter.check_position_limit("XRP_USDT", 1000.0).is_ok());
    }
}
