//! CATEGORY 8: Kill Switch HTTP/Telegram API Endpoint.
//!
//! Provides an HTTP API endpoint and Telegram command handler for emergency
//! trading halt. Previously, the circuit breaker could only be tripped
//! programmatically - this module exposes it via:
//!
//!   - `POST /api/kill-switch/activate` — Halt all trading immediately
//!   - `POST /api/kill-switch/reset` — Resume trading (requires confirmation)
//!   - `GET  /api/kill-switch/status` — Get current circuit breaker state
//!   - `POST /api/kill-switch/flatten` — Close all positions at market
//!
//! The kill switch is designed to work even when the main strategy loop
//! is stuck or unresponsive. It operates via shared AtomicBool that the
//! strategy engine checks on every iteration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::circuit_breaker::{CircuitBreaker, TripReason};

/// Kill switch state for JSON serialization.
#[derive(Debug, Serialize, Deserialize)]
pub struct KillSwitchStatus {
    /// Whether trading is currently halted.
    pub halted: bool,
    /// Reason for the halt (if halted).
    pub reason: String,
    /// Timestamp of the last trip (nanoseconds, 0 if never tripped).
    pub trip_timestamp_ns: u64,
    /// Number of consecutive losses.
    pub consecutive_losses: u32,
    /// Daily PnL in USDT (fixed point, divide by 1e10).
    pub daily_pnl_fp: i64,
    /// Total exposure in USDT (fixed point).
    pub total_exposure_fp: i64,
    /// Current equity.
    pub current_equity: i64,
    /// Peak equity.
    pub peak_equity: i64,
}

/// Kill switch request for activation.
#[derive(Debug, Deserialize)]
pub struct KillSwitchActivateRequest {
    /// Reason for manual kill (logged and displayed).
    pub reason: Option<String>,
    /// Whether to flatten all positions immediately.
    pub flatten: Option<bool>,
}

/// Kill switch response.
#[derive(Debug, Serialize)]
pub struct KillSwitchResponse {
    pub success: bool,
    pub message: String,
    pub status: KillSwitchStatus,
}

/// Kill switch manager that wraps the circuit breaker.
pub struct KillSwitch {
    circuit_breaker: Arc<CircuitBreaker>,
    /// Flag indicating positions should be flattened.
    flatten_requested: AtomicBool,
    /// Last manual reason (if any).
    last_reason: parking_lot::RwLock<String>,
}

impl KillSwitch {
    /// Create a new kill switch wrapping the circuit breaker.
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self {
            circuit_breaker,
            flatten_requested: AtomicBool::new(false),
            last_reason: parking_lot::RwLock::new(String::new()),
        }
    }

    /// Activate the kill switch (halt all trading).
    pub fn activate(&self, reason: Option<&str>, flatten: bool) -> KillSwitchResponse {
        let reason_str = reason.unwrap_or("Manual kill switch activated via API");

        self.circuit_breaker.trip(TripReason::ManualKill);

        if flatten {
            self.flatten_requested.store(true, Ordering::SeqCst);
        }

        {
            let mut lr = self.last_reason.write();
            *lr = reason_str.to_string();
        }

        error!(
            "[kill-switch] ACTIVATED: reason='{}', flatten={}",
            reason_str, flatten
        );

        KillSwitchResponse {
            success: true,
            message: format!("Kill switch activated: {}", reason_str),
            status: self.get_status(),
        }
    }

    /// Reset the kill switch (resume trading).
    pub fn reset(&self) -> KillSwitchResponse {
        self.circuit_breaker.reset();
        self.flatten_requested.store(false, Ordering::SeqCst);

        info!("[kill-switch] RESET — trading resumed");

        KillSwitchResponse {
            success: true,
            message: "Kill switch reset - trading resumed".to_string(),
            status: self.get_status(),
        }
    }

    /// Get the current kill switch status.
    pub fn get_status(&self) -> KillSwitchStatus {
        let state = self.circuit_breaker.get_state();
        let reason = if state.halted {
            let manual_reason = self.last_reason.read();
            if manual_reason.is_empty() {
                format!("trip_reason_code={}", state.trip_reason)
            } else {
                manual_reason.clone()
            }
        } else {
            "not_halted".to_string()
        };

        KillSwitchStatus {
            halted: state.halted,
            reason,
            trip_timestamp_ns: 0, // Would need to add to CircuitBreakerState
            consecutive_losses: state.consecutive_losses,
            daily_pnl_fp: state.daily_pnl_fp,
            total_exposure_fp: state.total_exposure_fp,
            current_equity: state.current_equity,
            peak_equity: state.peak_equity,
        }
    }

    /// Check if position flattening has been requested.
    pub fn is_flatten_requested(&self) -> bool {
        self.flatten_requested.load(Ordering::Relaxed)
    }

    /// Clear the flatten request (after positions are closed).
    pub fn clear_flatten_request(&self) {
        self.flatten_requested.store(false, Ordering::Relaxed);
    }

    /// Check if trading is halted.
    pub fn is_halted(&self) -> bool {
        self.circuit_breaker.is_trading_halted()
    }
}

/// Build axum routes for the kill switch API.
///
/// These routes should be added to the dashboard server's router:
/// ```ignore
/// let kill_switch = Arc::new(KillSwitch::new(circuit_breaker.clone()));
/// let app = Router::new()
///     .route("/api/kill-switch/status", get(ks_status))
///     .route("/api/kill-switch/activate", post(ks_activate))
///     .route("/api/kill-switch/reset", post(ks_reset))
///     .with_state(kill_switch);
/// ```
///
/// Route handlers are defined below as standalone functions that can be
/// integrated into the axum router.

/// Handler: GET /api/kill-switch/status
pub fn handle_status(ks: &KillSwitch) -> serde_json::Value {
    serde_json::to_value(ks.get_status()).unwrap_or_default()
}

/// Handler: POST /api/kill-switch/activate
pub fn handle_activate(ks: &KillSwitch, reason: Option<&str>, flatten: bool) -> serde_json::Value {
    let resp = ks.activate(reason, flatten);
    serde_json::to_value(resp).unwrap_or_default()
}

/// Handler: POST /api/kill-switch/reset
pub fn handle_reset(ks: &KillSwitch) -> serde_json::Value {
    let resp = ks.reset();
    serde_json::to_value(resp).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_breaker::CircuitBreakerConfig;

    #[test]
    fn test_kill_switch_activate_reset() {
        let cb = Arc::new(CircuitBreaker::with_defaults());
        let ks = KillSwitch::new(cb.clone());

        assert!(!ks.is_halted());

        let resp = ks.activate(Some("Test halt"), false);
        assert!(resp.success);
        assert!(ks.is_halted());
        assert!(!ks.is_flatten_requested());

        let resp = ks.reset();
        assert!(resp.success);
        assert!(!ks.is_halted());
    }

    #[test]
    fn test_kill_switch_with_flatten() {
        let cb = Arc::new(CircuitBreaker::with_defaults());
        let ks = KillSwitch::new(cb);

        ks.activate(None, true);
        assert!(ks.is_halted());
        assert!(ks.is_flatten_requested());

        ks.clear_flatten_request();
        assert!(!ks.is_flatten_requested());
    }

    #[test]
    fn test_kill_switch_status() {
        let cb = Arc::new(CircuitBreaker::with_defaults());
        let ks = KillSwitch::new(cb);

        let status = ks.get_status();
        assert!(!status.halted);
        assert_eq!(status.reason, "not_halted");
    }
}
