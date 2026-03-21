//! Fixed-point arithmetic types for deterministic, zero-drift price and quantity representation.
//!
//! All prices and quantities are stored as `i64` integers with a configurable precision factor.
//! This eliminates IEEE 754 floating-point non-determinism where `0.1 + 0.2 != 0.3`.
//!
//! - **`FixedPrice`**: 1e8 precision (8 decimal places, satoshi-level for crypto).
//!   Price 50000.12345678 → i64 = 5_000_012_345_678.
//! - **`FixedQty`**: 1e4 precision (4 decimal places for contract quantities).
//!   Quantity 1.5 → i64 = 15_000.
//!
//! All arithmetic uses saturating operations to prevent overflow panics.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ---------------------------------------------------------------------------
// FixedPrice — 1e8 precision
// ---------------------------------------------------------------------------

/// Fixed-point price with 1e8 precision (8 decimal places).
///
/// All arithmetic uses `saturating_add` / `saturating_sub` to prevent overflow.
/// Two prices computed via different paths but representing the same decimal value
/// will **always** compare equal — unlike IEEE 754 `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct FixedPrice(pub i64);

impl FixedPrice {
    /// Precision factor: 1e8 (100,000,000).
    pub const PRECISION: i64 = 100_000_000;

    /// Zero price constant.
    pub const ZERO: Self = Self(0);

    /// Create a `FixedPrice` from an `f64`.
    ///
    /// The value is rounded to the nearest integer representation.
    #[inline(always)]
    pub fn from_f64(price: f64) -> Self {
        Self((price * Self::PRECISION as f64).round() as i64)
    }

    /// Convert back to `f64` for display or FFI boundary.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::PRECISION as f64
    }

    /// Compute the mid-price between bid and ask.
    /// Uses integer division — perfectly deterministic.
    #[inline(always)]
    pub fn mid(bid: Self, ask: Self) -> Self {
        Self((bid.0.saturating_add(ask.0)) / 2)
    }

    /// Compute the spread in basis points (integer result).
    ///
    /// Formula: `((ask - bid) * 10_000) / mid`
    /// Returns 0 if mid is zero.
    #[inline(always)]
    pub fn spread_bps(bid: Self, ask: Self) -> i64 {
        let mid = (bid.0.saturating_add(ask.0)) / 2;
        if mid == 0 {
            return 0;
        }
        ((ask.0.saturating_sub(bid.0)).saturating_mul(10_000)) / mid
    }

    /// Compute spread in basis points as f64 for higher precision.
    #[inline(always)]
    pub fn spread_bps_f64(bid: Self, ask: Self) -> f64 {
        let mid = (bid.0 as f64 + ask.0 as f64) / 2.0;
        if mid == 0.0 {
            return 0.0;
        }
        ((ask.0 - bid.0) as f64 * 10_000.0) / mid
    }

    /// Raw i64 value.
    #[inline(always)]
    pub fn raw(self) -> i64 {
        self.0
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }
}

impl Add for FixedPrice {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub for FixedPrice {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Mul for FixedPrice {
    type Output = Self;
    /// Multiply two fixed-point prices. Result is scaled back by PRECISION.
    /// Note: This is primarily for price × scalar operations.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Use i128 intermediate to avoid overflow during multiplication
        let result = (self.0 as i128 * rhs.0 as i128) / Self::PRECISION as i128;
        Self(result.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
    }
}

impl Div for FixedPrice {
    type Output = Self;
    /// Divide two fixed-point prices. Result is scaled by PRECISION.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Self(0); // Safe division by zero
        }
        let result = (self.0 as i128 * Self::PRECISION as i128) / rhs.0 as i128;
        Self(result.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
    }
}

impl fmt::Display for FixedPrice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / Self::PRECISION;
        let frac = (self.0 % Self::PRECISION).unsigned_abs();
        write!(f, "{}.{:08}", whole, frac)
    }
}

impl From<f64> for FixedPrice {
    #[inline(always)]
    fn from(v: f64) -> Self {
        Self::from_f64(v)
    }
}

impl From<FixedPrice> for f64 {
    #[inline(always)]
    fn from(v: FixedPrice) -> f64 {
        v.to_f64()
    }
}

impl Serialize for FixedPrice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize as f64 for JSON compatibility
        serializer.serialize_f64(self.to_f64())
    }
}

impl<'de> Deserialize<'de> for FixedPrice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = f64::deserialize(deserializer)?;
        Ok(Self::from_f64(v))
    }
}

// ---------------------------------------------------------------------------
// FixedQty — 1e4 precision
// ---------------------------------------------------------------------------

/// Fixed-point quantity with 1e4 precision (4 decimal places).
///
/// Used for contract quantities where sub-cent precision is sufficient.
/// Quantity 1.5 contracts → i64 = 15_000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct FixedQty(pub i64);

impl FixedQty {
    /// Precision factor: 1e4 (10,000).
    pub const PRECISION: i64 = 10_000;

    /// Zero quantity constant.
    pub const ZERO: Self = Self(0);

    /// Create a `FixedQty` from an `f64`.
    #[inline(always)]
    pub fn from_f64(qty: f64) -> Self {
        Self((qty * Self::PRECISION as f64).round() as i64)
    }

    /// Convert back to `f64` for display or FFI boundary.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::PRECISION as f64
    }

    /// Raw i64 value.
    #[inline(always)]
    pub fn raw(self) -> i64 {
        self.0
    }

    /// Check if quantity is zero.
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }
}

impl Add for FixedQty {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub for FixedQty {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Mul for FixedQty {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let result = (self.0 as i128 * rhs.0 as i128) / Self::PRECISION as i128;
        Self(result.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
    }
}

impl Div for FixedQty {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Self(0);
        }
        let result = (self.0 as i128 * Self::PRECISION as i128) / rhs.0 as i128;
        Self(result.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
    }
}

impl fmt::Display for FixedQty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / Self::PRECISION;
        let frac = (self.0 % Self::PRECISION).unsigned_abs();
        write!(f, "{}.{:04}", whole, frac)
    }
}

impl From<f64> for FixedQty {
    #[inline(always)]
    fn from(v: f64) -> Self {
        Self::from_f64(v)
    }
}

impl From<FixedQty> for f64 {
    #[inline(always)]
    fn from(v: FixedQty) -> f64 {
        v.to_f64()
    }
}

impl Serialize for FixedQty {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.to_f64())
    }
}

impl<'de> Deserialize<'de> for FixedQty {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = f64::deserialize(deserializer)?;
        Ok(Self::from_f64(v))
    }
}

/// Compute notional value: price × quantity.
/// Returns the result as an i64 in FixedPrice precision (1e8).
///
/// Since FixedPrice has 1e8 precision and FixedQty has 1e4 precision,
/// the raw multiplication gives 1e12 — we divide by FixedQty::PRECISION
/// to get back to 1e8 (FixedPrice scale).
#[inline(always)]
pub fn notional_fp(price: FixedPrice, qty: FixedQty) -> i64 {
    // price.0 is in 1e8 scale, qty.0 is in 1e4 scale
    // product is in 1e12 scale, divide by 1e4 to get 1e8 scale
    let product = price.0 as i128 * qty.0 as i128;
    (product / FixedQty::PRECISION as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Compute notional value as f64 USDT.
#[inline(always)]
pub fn notional_usdt(price: FixedPrice, qty: FixedQty) -> f64 {
    price.to_f64() * qty.to_f64()
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_price_roundtrip() {
        let price = 50000.12345678_f64;
        let fp = FixedPrice::from_f64(price);
        let back = fp.to_f64();
        assert!((price - back).abs() < 1e-8, "Roundtrip failed: {} vs {}", price, back);
    }

    #[test]
    fn test_fixed_price_determinism() {
        // The classic IEEE 754 failure case: 0.1 + 0.2
        // With fixed-point, adding the fixed representations must be deterministic.
        let a = FixedPrice::from_f64(0.1);
        let b = FixedPrice::from_f64(0.2);
        let c = FixedPrice::from_f64(0.3);
        let sum = a + b;
        assert_eq!(sum, c, "0.1 + 0.2 must equal 0.3 in fixed-point: {:?} vs {:?}", sum, c);
    }

    #[test]
    fn test_fixed_price_determinism_100_runs() {
        for _ in 0..100 {
            let bid = FixedPrice::from_f64(50000.50);
            let ask = FixedPrice::from_f64(50001.00);
            let mid = FixedPrice::mid(bid, ask);
            assert_eq!(mid.0, 5_000_075_000_000, "Mid must be bitwise identical across runs");
            let spread = FixedPrice::spread_bps(bid, ask);
            assert_eq!(spread, 0, "Spread BPS must be bitwise identical across runs (very tight spread)");
        }
    }

    #[test]
    fn test_spread_bps() {
        let bid = FixedPrice::from_f64(100.0);
        let ask = FixedPrice::from_f64(101.0);
        let bps = FixedPrice::spread_bps(bid, ask);
        // Expected: (1 / 100.5) * 10000 ≈ 99.5 BPS
        // Integer math: (1 * 1e8 * 10000) / (100.5 * 1e8) = 99
        assert!(bps >= 99 && bps <= 100, "Spread BPS should be ~99-100, got {}", bps);

        let bps_f64 = FixedPrice::spread_bps_f64(bid, ask);
        assert!((bps_f64 - 99.5025).abs() < 0.1, "Spread BPS f64 should be ~99.5, got {}", bps_f64);
    }

    #[test]
    fn test_fixed_price_overflow_protection() {
        let max_price = FixedPrice(i64::MAX);
        let one = FixedPrice(1);
        let result = max_price + one;
        assert_eq!(result.0, i64::MAX, "Saturating add must not overflow");

        let min_price = FixedPrice(i64::MIN);
        let result = min_price - one;
        assert_eq!(result.0, i64::MIN, "Saturating sub must not underflow");
    }

    #[test]
    fn test_fixed_qty_roundtrip() {
        let qty = 1.5_f64;
        let fq = FixedQty::from_f64(qty);
        assert_eq!(fq.0, 15_000, "1.5 should be 15000 at 1e4 precision");
        let back = fq.to_f64();
        assert!((qty - back).abs() < 1e-4, "Roundtrip failed: {} vs {}", qty, back);
    }

    #[test]
    fn test_fixed_qty_zero() {
        let zero = FixedQty::ZERO;
        assert!(zero.is_zero());
        assert_eq!(zero.0, 0);
    }

    #[test]
    fn test_mid_price() {
        let bid = FixedPrice::from_f64(50000.0);
        let ask = FixedPrice::from_f64(50002.0);
        let mid = FixedPrice::mid(bid, ask);
        assert_eq!(mid, FixedPrice::from_f64(50001.0));
    }

    #[test]
    fn test_notional() {
        let price = FixedPrice::from_f64(50000.0);
        let qty = FixedQty::from_f64(2.0);
        let usdt = notional_usdt(price, qty);
        assert!((usdt - 100000.0).abs() < 0.01, "Notional should be 100000, got {}", usdt);
    }

    #[test]
    fn test_display() {
        let p = FixedPrice::from_f64(50000.12345678);
        let s = format!("{}", p);
        assert!(s.starts_with("50000.1234"), "Display should show price: {}", s);

        let q = FixedQty::from_f64(1.5);
        let s = format!("{}", q);
        assert_eq!(s, "1.5000", "Display should show qty: {}", s);
    }

    #[test]
    fn test_serde_roundtrip() {
        let price = FixedPrice::from_f64(50000.5);
        let json = serde_json::to_string(&price).unwrap();
        let back: FixedPrice = serde_json::from_str(&json).unwrap();
        assert_eq!(price, back, "Serde roundtrip failed");
    }

    #[test]
    fn test_ordering() {
        let a = FixedPrice::from_f64(100.0);
        let b = FixedPrice::from_f64(200.0);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, FixedPrice::from_f64(100.0));
    }

    #[test]
    fn test_division_by_zero() {
        let a = FixedPrice::from_f64(100.0);
        let zero = FixedPrice::ZERO;
        let result = a / zero;
        assert_eq!(result, FixedPrice::ZERO, "Division by zero should return zero");
    }
}

