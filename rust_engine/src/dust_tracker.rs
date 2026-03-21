//! Dust Tracker — Fractional Contract Remainder Carry-Over.
//!
//! Gate.io futures ONLY accepts integer contract sizes. When the strategy
//! calculates a fractional position size (e.g., 2.7 contracts), the naive
//! `trunc()` approach loses 0.7 contracts every trade. Over hundreds of
//! trades this creates significant positional drift.
//!
//! The DustTracker accumulates fractional remainders per-symbol and carries
//! them forward to subsequent trades. When accumulated dust reaches >= 1.0,
//! it's added to the next order for that symbol.
//!
//! # Example
//!
//! ```text
//! Trade 1: strategy wants 2.7 → send 2, dust = 0.7
//! Trade 2: strategy wants 3.4 → 3.4 + 0.7 = 4.1 → send 4, dust = 0.1
//! Trade 3: strategy wants 1.2 → 1.2 + 0.1 = 1.3 → send 1, dust = 0.3
//! ```
//!
//! # Thread Safety
//!
//! NOT thread-safe. Owned by the strategy evaluator thread (single writer).

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// DustTracker
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks fractional contract remainders per symbol to prevent positional drift.
pub struct DustTracker {
    /// Accumulated dust per symbol_id. Positive for buy-side dust, negative for sell-side.
    dust: HashMap<u16, f64>,
    /// Maximum absolute dust to carry (safety clamp).
    max_dust: f64,
    /// Total dust accumulated across all symbols (for telemetry).
    pub total_dust_accumulated: f64,
    /// Total extra contracts recovered from dust.
    pub total_dust_recovered: u64,
}

impl DustTracker {
    /// Create a new dust tracker.
    ///
    /// `max_dust`: Maximum absolute fractional remainder to accumulate
    /// before discarding. Default 5.0 (prevents unbounded accumulation
    /// if the strategy keeps generating tiny orders).
    pub fn new(max_dust: f64) -> Self {
        Self {
            dust: HashMap::with_capacity(32),
            max_dust,
            total_dust_accumulated: 0.0,
            total_dust_recovered: 0,
        }
    }

    /// Create with default settings (max_dust = 5.0).
    pub fn with_defaults() -> Self {
        Self::new(5.0)
    }

    /// Convert a fractional contract size to an integer, carrying forward
    /// the remainder for the next trade on this symbol.
    ///
    /// Returns the integer contract count to submit to the exchange.
    /// The sign is preserved (positive for buy, negative for sell).
    ///
    /// # Arguments
    /// * `symbol_id` — Symbol identifier for dust tracking.
    /// * `size_f` — The fractional contract size from the strategy.
    /// * `is_buy` — Direction of the trade (affects dust sign).
    ///
    /// # Returns
    /// Integer contract count (always >= 1 for non-zero input, 0 if
    /// the adjusted size rounds down to zero).
    pub fn float_to_contracts(&mut self, symbol_id: u16, size_f: f64, is_buy: bool) -> i64 {
        if size_f.is_nan() || size_f.is_infinite() || size_f == 0.0 {
            return 0;
        }

        let abs_size = size_f.abs();
        let sign = if is_buy { 1.0 } else { -1.0 };

        // Retrieve accumulated dust for this symbol
        let accumulated = self.dust.get(&symbol_id).copied().unwrap_or(0.0);

        // Add dust to the current order (same direction only)
        let effective_size = if (accumulated > 0.0 && is_buy) || (accumulated < 0.0 && !is_buy) {
            abs_size + accumulated.abs()
        } else if accumulated.abs() > 0.0 {
            // Opposite direction: dust partially cancels
            let net = abs_size - accumulated.abs();
            if net < 0.0 {
                // Dust exceeds current order — reduce dust, return 0
                let remaining_dust = accumulated.abs() - abs_size;
                self.dust.insert(symbol_id, remaining_dust * if !is_buy { -1.0 } else { 1.0 });
                return 0;
            }
            net
        } else {
            abs_size
        };

        // Truncate to integer (floor toward zero)
        let integer_contracts = effective_size.floor() as i64;
        let remainder = effective_size - integer_contracts as f64;

        // Store the new dust (with direction)
        let new_dust = remainder * sign;
        let clamped_dust = new_dust.clamp(-self.max_dust, self.max_dust);
        self.dust.insert(symbol_id, clamped_dust);

        // Telemetry
        self.total_dust_accumulated += remainder;
        if effective_size > abs_size {
            self.total_dust_recovered += (effective_size - abs_size).floor() as u64;
        }

        // Ensure minimum of 1 contract if the strategy wanted at least 0.5
        if integer_contracts == 0 && abs_size >= 0.5 {
            1
        } else {
            integer_contracts
        }
    }

    /// Get the accumulated dust for a symbol.
    pub fn get_dust(&self, symbol_id: u16) -> f64 {
        self.dust.get(&symbol_id).copied().unwrap_or(0.0)
    }

    /// Clear dust for a symbol (e.g., when position is fully closed).
    pub fn clear_dust(&mut self, symbol_id: u16) {
        self.dust.remove(&symbol_id);
    }

    /// Clear all dust (e.g., on daily reset).
    pub fn clear_all(&mut self) {
        self.dust.clear();
    }

    /// Get the number of symbols with accumulated dust.
    pub fn tracked_symbols(&self) -> usize {
        self.dust.len()
    }
}

impl Default for DustTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_truncation() {
        let mut dt = DustTracker::with_defaults();

        // 2.7 contracts → send 2, dust = 0.7
        let contracts = dt.float_to_contracts(1, 2.7, true);
        assert_eq!(contracts, 2);
        assert!((dt.get_dust(1) - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_dust_carryover() {
        let mut dt = DustTracker::with_defaults();

        // Trade 1: 2.7 → send 2, dust = 0.7
        let c1 = dt.float_to_contracts(1, 2.7, true);
        assert_eq!(c1, 2);

        // Trade 2: 3.4 + 0.7 dust = 4.1 → send 4, dust = 0.1
        let c2 = dt.float_to_contracts(1, 3.4, true);
        assert_eq!(c2, 4);
        assert!((dt.get_dust(1) - 0.1).abs() < 0.001);

        // Trade 3: 1.2 + 0.1 = 1.3 → send 1, dust = 0.3
        let c3 = dt.float_to_contracts(1, 1.2, true);
        assert_eq!(c3, 1);
        assert!((dt.get_dust(1) - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_per_symbol_isolation() {
        let mut dt = DustTracker::with_defaults();

        dt.float_to_contracts(1, 2.7, true);
        dt.float_to_contracts(2, 3.9, true);

        assert!((dt.get_dust(1) - 0.7).abs() < 0.001);
        assert!((dt.get_dust(2) - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_minimum_one_contract() {
        let mut dt = DustTracker::with_defaults();

        // 0.7 contracts → should get 1 (since >= 0.5)
        let c = dt.float_to_contracts(1, 0.7, true);
        assert_eq!(c, 1);
    }

    #[test]
    fn test_zero_input() {
        let mut dt = DustTracker::with_defaults();
        assert_eq!(dt.float_to_contracts(1, 0.0, true), 0);
    }

    #[test]
    fn test_nan_input() {
        let mut dt = DustTracker::with_defaults();
        assert_eq!(dt.float_to_contracts(1, f64::NAN, true), 0);
    }

    #[test]
    fn test_clear_dust() {
        let mut dt = DustTracker::with_defaults();
        dt.float_to_contracts(1, 2.7, true);
        assert!(dt.get_dust(1) > 0.0);

        dt.clear_dust(1);
        assert_eq!(dt.get_dust(1), 0.0);
    }

    #[test]
    fn test_dust_clamping() {
        let mut dt = DustTracker::new(2.0); // max 2.0 dust

        // Generate a lot of dust by always requesting 0.99
        for _ in 0..100 {
            dt.float_to_contracts(1, 0.99, true);
        }

        // Dust should never exceed max_dust
        assert!(dt.get_dust(1).abs() <= 2.0);
    }

    #[test]
    fn test_exact_integer_no_dust() {
        let mut dt = DustTracker::with_defaults();

        let c = dt.float_to_contracts(1, 5.0, true);
        assert_eq!(c, 5);
        assert!(dt.get_dust(1).abs() < 0.001);
    }
}

