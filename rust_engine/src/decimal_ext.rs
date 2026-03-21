//! Decimal Extension — Conversion traits between FixedPrice/FixedQty and rust_decimal.
//!
//! # Architecture
//!
//! The codebase uses TWO numeric representations:
//!   - `FixedPrice`/`FixedQty` (i64-backed): Used on the HOT PATH for orderbook
//!     and SPSC operations. Zero-alloc, O(1) arithmetic, nanosecond latency.
//!   - `rust_decimal::Decimal`: Used for RISK CALCULATIONS where precision matters
//!     more than speed (margin checks, PnL, portfolio exposure, partial fills).
//!
//! This module provides zero-cost conversion traits between the two worlds.
//! The boundary rule: convert at the edge, compute in the appropriate type.
//!
//! # Precision
//!
//! - `FixedPrice` has 1e8 precision (8 decimal places)
//! - `FixedQty` has 1e4 precision (4 decimal places)
//! - `rust_decimal` has 28-digit precision
//!
//! Conversions are exact for values that fit in both representations.

use rust_decimal::Decimal;
use rust_decimal::prelude::*;

use crate::fixed_point::{FixedPrice, FixedQty};

// ═══════════════════════════════════════════════════════════════════════════
// FixedPrice <-> Decimal
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a FixedPrice (i64 with 1e8 scale) to rust_decimal::Decimal.
///
/// This is exact: FixedPrice stores price * 1e8 as i64, so we divide by 1e8.
/// rust_decimal can represent this without loss.
#[inline]
pub fn fixed_price_to_decimal(fp: FixedPrice) -> Decimal {
    Decimal::new(fp.raw(), 8)
}

/// Convert a rust_decimal::Decimal to FixedPrice.
///
/// Rounds to 8 decimal places, then converts to i64.
/// Returns None if the value overflows i64 after scaling.
#[inline]
pub fn decimal_to_fixed_price(d: Decimal) -> Option<FixedPrice> {
    let scaled = d.checked_mul(Decimal::new(100_000_000, 0))?;
    let rounded = scaled.round();
    rounded.to_i64().map(FixedPrice)
}

/// Infallible conversion — panics on overflow (use in non-hot-path code only).
#[inline]
pub fn decimal_to_fixed_price_unchecked(d: Decimal) -> FixedPrice {
    decimal_to_fixed_price(d).expect("Decimal overflow converting to FixedPrice")
}

// ═══════════════════════════════════════════════════════════════════════════
// FixedQty <-> Decimal
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a FixedQty (i64 with 1e4 scale) to rust_decimal::Decimal.
#[inline]
pub fn fixed_qty_to_decimal(fq: FixedQty) -> Decimal {
    Decimal::new(fq.raw(), 4)
}

/// Convert a rust_decimal::Decimal to FixedQty.
///
/// Rounds to 4 decimal places, then converts to i64.
/// Returns None if the value overflows i64 after scaling.
#[inline]
pub fn decimal_to_fixed_qty(d: Decimal) -> Option<FixedQty> {
    let scaled = d.checked_mul(Decimal::new(10_000, 0))?;
    let rounded = scaled.round();
    rounded.to_i64().map(FixedQty)
}

/// Infallible conversion — panics on overflow.
#[inline]
pub fn decimal_to_fixed_qty_unchecked(d: Decimal) -> FixedQty {
    decimal_to_fixed_qty(d).expect("Decimal overflow converting to FixedQty")
}

// ═══════════════════════════════════════════════════════════════════════════
// f64 <-> Decimal convenience (for risk module boundaries)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert f64 to Decimal. Uses `from_f64_retain` for maximum precision.
/// Returns Decimal::ZERO if the f64 is NaN or Infinity.
#[inline]
pub fn f64_to_decimal(v: f64) -> Decimal {
    if v.is_finite() {
        Decimal::from_f64_retain(v).unwrap_or(Decimal::ZERO)
    } else {
        Decimal::ZERO
    }
}

/// Convert Decimal to f64. Lossy but acceptable for logging/telemetry.
#[inline]
pub fn decimal_to_f64(d: Decimal) -> f64 {
    d.to_f64().unwrap_or(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Decimal Risk Math Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute notional value: price * quantity.
/// Both inputs are FixedPrice-scale i64 values.
/// Returns Decimal for precise risk calculations.
#[inline]
pub fn compute_notional_decimal(price_fp: i64, qty_fp: i64) -> Decimal {
    let price = Decimal::new(price_fp, 8);
    let qty = Decimal::new(qty_fp.abs(), 4);
    price * qty
}

/// Compute margin requirement: notional / leverage.
/// Returns Decimal for precise risk checks.
#[inline]
pub fn compute_margin_decimal(price_fp: i64, qty_fp: i64, leverage: u32) -> Decimal {
    let notional = compute_notional_decimal(price_fp, qty_fp);
    if leverage == 0 {
        notional / Decimal::new(5, 0) // Default 5x
    } else {
        notional / Decimal::new(leverage as i64, 0)
    }
}

/// Compute PnL for a position.
///
/// For longs:  pnl = (exit_price - entry_price) * qty
/// For shorts: pnl = (entry_price - exit_price) * qty
///
/// All prices in FixedPrice (1e8 scale), qty in FixedQty (1e4 scale).
#[inline]
pub fn compute_pnl_decimal(
    entry_price_fp: i64,
    exit_price_fp: i64,
    qty_fp: i64,
    is_long: bool,
) -> Decimal {
    let entry = Decimal::new(entry_price_fp, 8);
    let exit = Decimal::new(exit_price_fp, 8);
    let qty = Decimal::new(qty_fp.abs(), 4);
    if is_long {
        (exit - entry) * qty
    } else {
        (entry - exit) * qty
    }
}

/// Compute percentage change between two prices.
/// Returns as a Decimal fraction (e.g., 0.05 = 5%).
#[inline]
pub fn compute_pct_change_decimal(from_fp: i64, to_fp: i64) -> Decimal {
    let from = Decimal::new(from_fp, 8);
    let to = Decimal::new(to_fp, 8);
    if from.is_zero() {
        Decimal::ZERO
    } else {
        (to - from) / from
    }
}

/// Check if a drawdown percentage exceeds a threshold.
/// Both values are Decimal fractions (e.g., 0.05 = 5%).
#[inline]
pub fn exceeds_drawdown_limit(current_drawdown: Decimal, max_drawdown: Decimal) -> bool {
    current_drawdown.abs() > max_drawdown
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_price_roundtrip() {
        let original = FixedPrice::from_f64(50123.456789);
        let decimal = fixed_price_to_decimal(original);
        let back = decimal_to_fixed_price(decimal).unwrap();
        assert_eq!(original.raw(), back.raw());
    }

    #[test]
    fn test_fixed_qty_roundtrip() {
        let original = FixedQty::from_f64(42.5);
        let decimal = fixed_qty_to_decimal(original);
        let back = decimal_to_fixed_qty(decimal).unwrap();
        assert_eq!(original.raw(), back.raw());
    }

    #[test]
    fn test_notional_calculation() {
        let price_fp = FixedPrice::from_f64(50000.0).raw();  // $50,000
        let qty_fp = FixedQty::from_f64(10.0).raw();          // 10 contracts
        let notional = compute_notional_decimal(price_fp, qty_fp);
        let expected = Decimal::new(500_000, 0); // $500,000
        assert_eq!(notional, expected);
    }

    #[test]
    fn test_margin_calculation() {
        let price_fp = FixedPrice::from_f64(50000.0).raw();
        let qty_fp = FixedQty::from_f64(10.0).raw();
        let margin = compute_margin_decimal(price_fp, qty_fp, 10);
        let expected = Decimal::new(50_000, 0); // $500,000 / 10x = $50,000
        assert_eq!(margin, expected);
    }

    #[test]
    fn test_pnl_long_profit() {
        let entry = FixedPrice::from_f64(50000.0).raw();
        let exit = FixedPrice::from_f64(51000.0).raw();
        let qty = FixedQty::from_f64(1.0).raw();
        let pnl = compute_pnl_decimal(entry, exit, qty, true);
        let expected = Decimal::new(1000, 0); // $1,000 profit
        assert_eq!(pnl, expected);
    }

    #[test]
    fn test_pnl_short_profit() {
        let entry = FixedPrice::from_f64(50000.0).raw();
        let exit = FixedPrice::from_f64(49000.0).raw();
        let qty = FixedQty::from_f64(1.0).raw();
        let pnl = compute_pnl_decimal(entry, exit, qty, false);
        let expected = Decimal::new(1000, 0); // $1,000 profit
        assert_eq!(pnl, expected);
    }

    #[test]
    fn test_pct_change() {
        let from = FixedPrice::from_f64(100.0).raw();
        let to = FixedPrice::from_f64(105.0).raw();
        let pct = compute_pct_change_decimal(from, to);
        // 5% = 0.05
        assert!(pct > Decimal::new(4, 2) && pct < Decimal::new(6, 2));
    }

    #[test]
    fn test_f64_to_decimal_nan() {
        let d = f64_to_decimal(f64::NAN);
        assert_eq!(d, Decimal::ZERO);
    }

    #[test]
    fn test_f64_to_decimal_normal() {
        let d = f64_to_decimal(123.456);
        assert!(decimal_to_f64(d) > 123.0 && decimal_to_f64(d) < 124.0);
    }
}
