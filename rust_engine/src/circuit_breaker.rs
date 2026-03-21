//! Global circuit breaker & kill switch — Phase 3.
//!
//! Provides institutional-grade trading halt mechanisms for the Rust hot path.
//! The circuit breaker monitors:
//!   - Consecutive losses and daily drawdown
//!   - Spread/volatility anomalies (> 5 sigma)
//!   - Order rate anomalies (burst detection)
//!   - Explicit kill switch (manual or programmatic)
//!
//! When tripped, ALL trading is halted immediately:
//!   1. Kill switch AtomicBool set to `true` (read by all threads in <1ns)
//!   2. All resting maker orders are canceled
//!   3. Optionally, all positions are flattened (market close)
//!   4. Recovery requires manual reset or cooldown expiry
//!
//! # Thread Safety
//!
//! The `CircuitBreaker` is designed to be readable from any thread via
//! `AtomicBool::load(Relaxed)` and writable only from the execution router
//! (Core 6) or the telemetry thread (Core 7) for metrics updates.
//!
//! The `is_trading_halted()` check costs ~1 ns (single atomic load).

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use tracing::{error, warn, info};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Circuit breaker configuration. Loaded once at startup.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Maximum consecutive losing trades before halt.
    pub max_consecutive_losses: u32,
    /// Maximum daily drawdown as a fraction (e.g., 0.05 = 5%).
    pub max_daily_drawdown_pct: f64,
    /// Maximum spread in basis points before halt (absolute ceiling).
    pub max_spread_bps: i64,
    /// Spread z-score threshold for anomaly detection (default: 5.0 sigma).
    pub spread_zscore_threshold: f64,
    /// Maximum orders per second before throttling.
    pub max_orders_per_second: u32,
    /// Cooldown period in seconds after a trip before auto-recovery.
    /// 0 = manual reset required.
    pub cooldown_seconds: u64,
    /// Whether to flatten all positions when tripped.
    pub flatten_on_trip: bool,
    /// Maximum single-trade loss in USDT (FixedPrice) before halt.
    pub max_single_loss_usdt_fp: i64,
    /// Maximum total exposure in USDT (FixedPrice) across all positions.
    pub max_total_exposure_usdt_fp: i64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            max_consecutive_losses: 5,
            max_daily_drawdown_pct: 0.05, // 5% daily max loss
            max_spread_bps: 500,          // 5% spread = something is very wrong
            spread_zscore_threshold: 5.0,
            max_orders_per_second: 50,
            cooldown_seconds: 0,          // manual reset by default
            flatten_on_trip: false,       // conservative: don't auto-flatten
            max_single_loss_usdt_fp: 500_0000_0000, // $500 max single loss
            max_total_exposure_usdt_fp: 10_000_0000_0000, // $10,000 max exposure
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trip Reason
// ═══════════════════════════════════════════════════════════════════════════

/// Why the circuit breaker tripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripReason {
    /// Too many consecutive losses.
    ConsecutiveLosses,
    /// Daily drawdown exceeded threshold.
    DailyDrawdown,
    /// Spread anomaly (> configured sigma).
    SpreadAnomaly,
    /// Order rate too high (potential runaway loop).
    OrderRateAnomaly,
    /// Manual kill switch activated.
    ManualKill,
    /// Single trade loss exceeded threshold.
    SingleTradeLoss,
    /// Total exposure exceeded threshold.
    ExposureLimitBreached,
    /// Exchange connectivity lost.
    ConnectivityLost,
}

impl std::fmt::Display for TripReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TripReason::ConsecutiveLosses => write!(f, "consecutive_losses"),
            TripReason::DailyDrawdown => write!(f, "daily_drawdown"),
            TripReason::SpreadAnomaly => write!(f, "spread_anomaly"),
            TripReason::OrderRateAnomaly => write!(f, "order_rate_anomaly"),
            TripReason::ManualKill => write!(f, "manual_kill"),
            TripReason::SingleTradeLoss => write!(f, "single_trade_loss"),
            TripReason::ExposureLimitBreached => write!(f, "exposure_limit"),
            TripReason::ConnectivityLost => write!(f, "connectivity_lost"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CircuitBreaker
// ═══════════════════════════════════════════════════════════════════════════

/// Global circuit breaker state.
///
/// All fields use atomics for lock-free cross-thread visibility.
/// The `halted` flag is the single source of truth — every thread checks
/// this before submitting orders.
pub struct CircuitBreaker {
    /// Master kill switch. `true` = ALL trading halted.
    halted: AtomicBool,
    /// Reason for the last trip (encoded as u8).
    trip_reason: AtomicU32,
    /// Timestamp (nanoseconds) when the circuit breaker was tripped.
    trip_timestamp_ns: AtomicU64,

    // ── Tracking Counters ──
    /// Consecutive losing trades.
    consecutive_losses: AtomicU32,
    /// BUG 7 FIX: Add current_equity and peak_equity fields
    pub current_equity: AtomicI64,
    pub peak_equity: AtomicI64,
    /// Daily realized PnL in FixedPrice (can be negative).
    daily_pnl_fp: AtomicI64,
    /// Starting balance for daily drawdown calculation (FixedPrice).
    daily_start_balance_fp: AtomicI64,
    /// Orders submitted in the current second.
    orders_this_second: AtomicU32,
    /// Timestamp of the current tracking second.
    current_second_ns: AtomicU64,
    /// Total open exposure in USDT (FixedPrice).
    total_exposure_fp: AtomicI64,

    // ── Spread Statistics (running mean/variance for z-score) ──
    /// Exponential moving average of spread (FixedPrice BPS * 100).
    spread_ema: AtomicI64,
    /// Exponential moving variance of spread (scaled).
    spread_emvar: AtomicI64,
    /// Number of spread observations.
    spread_count: AtomicU64,

    /// Configuration (immutable after construction).
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            halted: AtomicBool::new(false),
            trip_reason: AtomicU32::new(0),
            trip_timestamp_ns: AtomicU64::new(0),
            consecutive_losses: AtomicU32::new(0),
            daily_pnl_fp: AtomicI64::new(0),
            daily_start_balance_fp: AtomicI64::new(0),
            orders_this_second: AtomicU32::new(0),
            current_second_ns: AtomicU64::new(0),
            total_exposure_fp: AtomicI64::new(0),
            spread_ema: AtomicI64::new(0),
            spread_emvar: AtomicI64::new(0),
            spread_count: AtomicU64::new(0),
            current_equity: AtomicI64::new(0),
            peak_equity: AtomicI64::new(0),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    // ── Fast Path Checks (called by every thread) ──────────────────────

    /// Check if trading is halted. This is the HOT PATH check.
    /// Cost: ~1 ns (single atomic load with Relaxed ordering).
    ///
    /// Every thread calls this before submitting orders or processing signals.
    #[inline(always)]
    pub fn is_trading_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// Get the trip reason (0 = not tripped).
    #[inline]
    pub fn trip_reason_code(&self) -> u32 {
        self.trip_reason.load(Ordering::Relaxed)
    }

    // ── Trip & Reset ───────────────────────────────────────────────────

    /// Trip the circuit breaker. Called by the execution router or risk monitor.
    ///
    /// Once tripped, `is_trading_halted()` returns `true` for all threads
    /// within nanoseconds (cache-coherence propagation).
    pub fn trip(&self, reason: TripReason) {
        let was_halted = self.halted.swap(true, Ordering::SeqCst);
        if !was_halted {
            let ts = now_ns();
            self.trip_reason.store(reason as u32, Ordering::Relaxed);
            self.trip_timestamp_ns.store(ts, Ordering::Relaxed);
            error!(
                "🚨 CIRCUIT BREAKER TRIPPED: reason={}, ts={}",
                reason, ts
            );
        }
    }

    /// Manually reset the circuit breaker. Trading resumes.
    ///
    /// Only call this after confirming the issue has been resolved.
    pub fn reset(&self) {
        self.halted.store(false, Ordering::SeqCst);
        self.trip_reason.store(0, Ordering::Relaxed);
        self.consecutive_losses.store(0, Ordering::Relaxed);
        self.orders_this_second.store(0, Ordering::Relaxed);
        info!("✅ Circuit breaker RESET — trading resumed");
    }

    /// Check if cooldown has expired and auto-reset if configured.
    pub fn check_cooldown(&self) {
        if !self.is_trading_halted() {
            return;
        }
        if self.config.cooldown_seconds == 0 {
            return; // manual reset required
        }
        let trip_ts = self.trip_timestamp_ns.load(Ordering::Relaxed);
        let elapsed_ns = now_ns().saturating_sub(trip_ts);
        let cooldown_ns = self.config.cooldown_seconds * 1_000_000_000;
        if elapsed_ns >= cooldown_ns {
            info!(
                "Circuit breaker cooldown expired ({}s). Auto-resetting.",
                self.config.cooldown_seconds
            );
            self.reset();
        }
    }

    // ── Event Handlers (called by execution router) ────────────────────

    /// Record a trade result. Updates consecutive loss counter and daily PnL.
    ///
    /// `pnl_fp` is the realized PnL in FixedPrice (positive = profit).
    pub fn on_trade_result(&self, pnl_fp: i64) {
        if pnl_fp < 0 {
            let losses = self.consecutive_losses.fetch_add(1, Ordering::Relaxed) + 1;
            if losses >= self.config.max_consecutive_losses {
                self.trip(TripReason::ConsecutiveLosses);
            }

            // Check single-trade loss limit
            if pnl_fp.abs() > self.config.max_single_loss_usdt_fp {
                self.trip(TripReason::SingleTradeLoss);
            }
        } else {
            // Reset consecutive loss counter on a winning trade
            self.consecutive_losses.store(0, Ordering::Relaxed);
        }

        // Update daily PnL
        let new_daily = self.daily_pnl_fp.fetch_add(pnl_fp, Ordering::Relaxed) + pnl_fp;
        let start_balance = self.daily_start_balance_fp.load(Ordering::Relaxed);
        if start_balance > 0 {
            let drawdown_pct = -(new_daily as f64) / (start_balance as f64);
            if drawdown_pct > self.config.max_daily_drawdown_pct {
                self.trip(TripReason::DailyDrawdown);
            }
        }
    }

    /// Record an order submission. Checks rate limiting.
    pub fn on_order_submitted(&self) {
        let now = now_ns();
        let current_sec = now / 1_000_000_000;
        let tracked_sec = self.current_second_ns.load(Ordering::Relaxed) / 1_000_000_000;

        if current_sec != tracked_sec {
            // New second — reset counter
            self.orders_this_second.store(1, Ordering::Relaxed);
            self.current_second_ns.store(now, Ordering::Relaxed);
        } else {
            let count = self.orders_this_second.fetch_add(1, Ordering::Relaxed) + 1;
            if count > self.config.max_orders_per_second {
                warn!(
                    "Order rate anomaly: {} orders/sec exceeds limit of {}",
                    count, self.config.max_orders_per_second
                );
                self.trip(TripReason::OrderRateAnomaly);
            }
        }
    }

    /// Update spread observation for z-score anomaly detection.
    ///
    /// `spread_bps` is the current bid-ask spread in basis points.
    pub fn on_spread_update(&self, spread_bps: i64) {
        // Absolute ceiling check
        if spread_bps > self.config.max_spread_bps {
            self.trip(TripReason::SpreadAnomaly);
            return;
        }

        let count = self.spread_count.fetch_add(1, Ordering::Relaxed);

        // EMA parameters (alpha = 0.01 for ~100-sample window)
        // We use integer arithmetic scaled by 10000 to avoid floating point.
        let alpha_scaled = 100i64; // 0.01 * 10000
        let one_minus_alpha_scaled = 9900i64; // (1 - 0.01) * 10000

        if count < 100 {
            // Warmup: simple accumulation
            let old_ema = self.spread_ema.load(Ordering::Relaxed);
            let new_ema = if count == 0 {
                spread_bps * 10000
            } else {
                (old_ema * count as i64 + spread_bps * 10000) / (count as i64 + 1)
            };
            self.spread_ema.store(new_ema, Ordering::Relaxed);
        } else {
            // EMA update
            let old_ema = self.spread_ema.load(Ordering::Relaxed);
            let new_ema = (alpha_scaled * spread_bps * 10000
                + one_minus_alpha_scaled * old_ema)
                / 10000;
            self.spread_ema.store(new_ema, Ordering::Relaxed);

            // EMA variance update (Welford-like)
            let diff = spread_bps * 10000 - new_ema;
            let old_var = self.spread_emvar.load(Ordering::Relaxed);
            let new_var =
                (one_minus_alpha_scaled * old_var + alpha_scaled * diff * diff / 10000) / 10000;
            self.spread_emvar.store(new_var, Ordering::Relaxed);

            // Z-score check
            if new_var > 0 {
                // z = (x - mean) / std = diff / sqrt(var)
                // We compare diff^2 > threshold^2 * var to avoid sqrt
                let threshold_sq = (self.config.spread_zscore_threshold
                    * self.config.spread_zscore_threshold
                    * 10000.0) as i64;
                if diff * diff > threshold_sq * new_var / 10000 {
                    warn!(
                        "Spread z-score anomaly: spread={}bps, ema={}, var={}",
                        spread_bps,
                        new_ema / 10000,
                        new_var / 10000
                    );
                    self.trip(TripReason::SpreadAnomaly);
                }
            }
        }
    }

    /// Update total exposure tracking.
    pub fn update_exposure(&self, total_exposure_fp: i64) {
        self.total_exposure_fp.store(total_exposure_fp, Ordering::Relaxed);
        if total_exposure_fp > self.config.max_total_exposure_usdt_fp {
            self.trip(TripReason::ExposureLimitBreached);
        }
    }

    /// Set the daily starting balance (call at start of day or engine start).
    pub fn set_daily_start_balance(&self, balance_fp: i64) {
        self.daily_start_balance_fp.store(balance_fp, Ordering::Relaxed);
        self.daily_pnl_fp.store(0, Ordering::Relaxed);
    }

    /// Get current state for telemetry/dashboard.
    pub fn get_state(&self) -> CircuitBreakerState {
        CircuitBreakerState {
            halted: self.is_trading_halted(),
            trip_reason: self.trip_reason.load(Ordering::Relaxed),
            consecutive_losses: self.consecutive_losses.load(Ordering::Relaxed),
            daily_pnl_fp: self.daily_pnl_fp.load(Ordering::Relaxed),
            total_exposure_fp: self.total_exposure_fp.load(Ordering::Relaxed),
            orders_this_second: self.orders_this_second.load(Ordering::Relaxed),
            current_equity: self.current_equity.load(Ordering::Relaxed),
            peak_equity: self.peak_equity.load(Ordering::Relaxed),
        }
    }

    /// Whether positions should be flattened when tripped.
    pub fn should_flatten(&self) -> bool {
        self.is_trading_halted() && self.config.flatten_on_trip
    }
}

/// Snapshot of circuit breaker state for telemetry.
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub halted: bool,
    pub trip_reason: u32,
    pub consecutive_losses: u32,
    pub daily_pnl_fp: i64,
    pub total_exposure_fp: i64,
    pub orders_this_second: u32,
    pub current_equity: i64,
    pub peak_equity: i64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let cb = CircuitBreaker::with_defaults();
        assert!(!cb.is_trading_halted());
        assert_eq!(cb.trip_reason_code(), 0);
    }

    #[test]
    fn test_manual_trip_and_reset() {
        let cb = CircuitBreaker::with_defaults();
        cb.trip(TripReason::ManualKill);
        assert!(cb.is_trading_halted());
        assert_eq!(cb.trip_reason_code(), TripReason::ManualKill as u32);

        cb.reset();
        assert!(!cb.is_trading_halted());
    }

    #[test]
    fn test_consecutive_losses_trip() {
        let config = CircuitBreakerConfig {
            max_consecutive_losses: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.on_trade_result(-100); // loss 1
        assert!(!cb.is_trading_halted());
        cb.on_trade_result(-200); // loss 2
        assert!(!cb.is_trading_halted());
        cb.on_trade_result(-300); // loss 3 — should trip
        assert!(cb.is_trading_halted());
    }

    #[test]
    fn test_consecutive_losses_reset_on_win() {
        let config = CircuitBreakerConfig {
            max_consecutive_losses: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.on_trade_result(-100); // loss 1
        cb.on_trade_result(-200); // loss 2
        cb.on_trade_result(500);  // WIN — resets counter
        cb.on_trade_result(-100); // loss 1 again
        cb.on_trade_result(-200); // loss 2 again
        assert!(!cb.is_trading_halted()); // should NOT trip
    }

    #[test]
    fn test_daily_drawdown_trip() {
        let config = CircuitBreakerConfig {
            max_daily_drawdown_pct: 0.05, // 5%
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);
        cb.set_daily_start_balance(10_000_0000_0000); // $10,000

        // Lose 4% — should NOT trip
        cb.on_trade_result(-400_0000_0000);
        assert!(!cb.is_trading_halted());

        // Lose another 2% (total 6%) — should trip
        cb.on_trade_result(-200_0000_0000);
        assert!(cb.is_trading_halted());
    }

    #[test]
    fn test_spread_anomaly_absolute() {
        let config = CircuitBreakerConfig {
            max_spread_bps: 100, // 1% max
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.on_spread_update(50); // 0.5% — fine
        assert!(!cb.is_trading_halted());

        cb.on_spread_update(150); // 1.5% — exceeds absolute limit
        assert!(cb.is_trading_halted());
    }

    #[test]
    fn test_order_rate_limiting() {
        let config = CircuitBreakerConfig {
            max_orders_per_second: 5,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        for _ in 0..5 {
            cb.on_order_submitted();
        }
        assert!(!cb.is_trading_halted());

        cb.on_order_submitted(); // 6th order — should trip
        assert!(cb.is_trading_halted());
    }

    #[test]
    fn test_exposure_limit() {
        let config = CircuitBreakerConfig {
            max_total_exposure_usdt_fp: 10_000_0000_0000, // $10,000
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.update_exposure(5_000_0000_0000); // $5,000 — fine
        assert!(!cb.is_trading_halted());

        cb.update_exposure(15_000_0000_0000); // $15,000 — exceeds limit
        assert!(cb.is_trading_halted());
    }

    #[test]
    fn test_state_snapshot() {
        let cb = CircuitBreaker::with_defaults();
        cb.on_trade_result(-100);
        let state = cb.get_state();
        assert!(!state.halted);
        assert_eq!(state.consecutive_losses, 1);
        assert_eq!(state.daily_pnl_fp, -100);
    }
}

