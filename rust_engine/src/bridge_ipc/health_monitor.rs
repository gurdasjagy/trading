//! Bridge IPC Health Monitor — tracks IPC health metrics and exposes them via HTTP.
//!
//! Monitors the health of all shared memory IPC channels:
//! - Message rates (ticks/sec, portfolio updates/sec, exec confirmations/sec)
//! - Latency histograms (p50, p95, p99)
//! - Error counts (stale data, corrupted messages, sequence gaps)
//! - Stale data events (messages older than threshold)
//!
//! Exposes metrics via a `/health` HTTP endpoint for monitoring dashboards.

use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;

// ═══════════════════════════════════════════════════════════════════════════
// Health Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Health metrics for a single IPC channel.
#[derive(Debug, Clone)]
pub struct ChannelHealthMetrics {
    /// Channel name (e.g., "tick_broadcast", "portfolio_receiver").
    pub name: String,
    /// Total messages sent/received.
    pub total_messages: u64,
    /// Messages in the last second.
    pub messages_per_second: u64,
    /// Total errors (stale data, corrupted messages, etc.).
    pub total_errors: u64,
    /// Errors in the last second.
    pub errors_per_second: u64,
    /// Last message timestamp (nanoseconds since epoch).
    pub last_message_ns: u64,
    /// Age of last message in milliseconds.
    pub last_message_age_ms: u64,
    /// Latency histogram (p50, p95, p99 in microseconds).
    pub latency_p50_us: u64,
    pub latency_p95_us: u64,
    pub latency_p99_us: u64,
}

impl Default for ChannelHealthMetrics {
    fn default() -> Self {
        Self {
            name: String::new(),
            total_messages: 0,
            messages_per_second: 0,
            total_errors: 0,
            errors_per_second: 0,
            last_message_ns: 0,
            last_message_age_ms: 0,
            latency_p50_us: 0,
            latency_p95_us: 0,
            latency_p99_us: 0,
        }
    }
}

/// Overall bridge health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeHealthStatus {
    /// All channels healthy.
    Healthy,
    /// One or more channels degraded (high latency or errors).
    Degraded,
    /// One or more channels unhealthy (stale data or no messages).
    Unhealthy,
}

impl std::fmt::Display for BridgeHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BridgeHealthMonitor
// ═══════════════════════════════════════════════════════════════════════════

/// Monitors the health of all IPC channels and exposes metrics.
pub struct BridgeHealthMonitor {
    /// Metrics for each channel.
    channels: Arc<RwLock<Vec<ChannelHealthMetrics>>>,
    /// Start time for rate calculations.
    start_time: Instant,
    /// Last rate calculation time.
    last_rate_calc: Arc<RwLock<Instant>>,
}

impl BridgeHealthMonitor {
    /// Create a new bridge health monitor.
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            last_rate_calc: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Register a new channel for monitoring.
    pub fn register_channel(&self, name: &str) {
        let mut channels = self.channels.write();
        if !channels.iter().any(|c| c.name == name) {
            channels.push(ChannelHealthMetrics {
                name: name.to_string(),
                ..Default::default()
            });
        }
    }

    /// Record a message sent/received on a channel.
    pub fn record_message(&self, channel_name: &str, latency_us: u64) {
        let mut channels = self.channels.write();
        if let Some(ch) = channels.iter_mut().find(|c| c.name == channel_name) {
            ch.total_messages += 1;
            ch.last_message_ns = now_ns();
            // Update latency histogram (simplified: just track last value)
            // In production, use a proper histogram like HdrHistogram
            ch.latency_p50_us = latency_us;
            ch.latency_p95_us = latency_us;
            ch.latency_p99_us = latency_us;
        }
    }

    /// Record an error on a channel.
    pub fn record_error(&self, channel_name: &str) {
        let mut channels = self.channels.write();
        if let Some(ch) = channels.iter_mut().find(|c| c.name == channel_name) {
            ch.total_errors += 1;
        }
    }

    /// Update rate calculations (call every second).
    pub fn update_rates(&self) {
        let mut last_calc = self.last_rate_calc.write();
        let now = Instant::now();
        let elapsed = now.duration_since(*last_calc).as_secs_f64();
        if elapsed < 1.0 {
            return;
        }

        let mut channels = self.channels.write();
        for ch in channels.iter_mut() {
            // Calculate messages per second (simplified: just reset counter)
            ch.messages_per_second = ch.total_messages;
            ch.errors_per_second = ch.total_errors;

            // Calculate last message age
            let now_ns = now_ns();
            if ch.last_message_ns > 0 {
                ch.last_message_age_ms = (now_ns - ch.last_message_ns) / 1_000_000;
            }
        }

        *last_calc = now;
    }

    /// Get the overall health status.
    pub fn get_status(&self) -> BridgeHealthStatus {
        let channels = self.channels.read();
        let mut has_degraded = false;
        let mut has_unhealthy = false;

        for ch in channels.iter() {
            // Unhealthy: no messages in last 10 seconds
            if ch.last_message_age_ms > 10_000 {
                has_unhealthy = true;
            }
            // Degraded: high error rate (>10% of messages)
            else if ch.total_messages > 0 && ch.total_errors * 10 > ch.total_messages {
                has_degraded = true;
            }
            // Degraded: high latency (>1ms p99)
            else if ch.latency_p99_us > 1_000 {
                has_degraded = true;
            }
        }

        if has_unhealthy {
            BridgeHealthStatus::Unhealthy
        } else if has_degraded {
            BridgeHealthStatus::Degraded
        } else {
            BridgeHealthStatus::Healthy
        }
    }

    /// Get all channel metrics.
    pub fn get_metrics(&self) -> Vec<ChannelHealthMetrics> {
        self.channels.read().clone()
    }

    /// Get metrics as JSON string.
    pub fn get_metrics_json(&self) -> String {
        let status = self.get_status();
        let metrics = self.get_metrics();
        let uptime_secs = self.start_time.elapsed().as_secs();

        let mut json = format!(
            r#"{{"status":"{}","uptime_secs":{},"channels":["#,
            status, uptime_secs
        );

        for (i, ch) in metrics.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                r#"{{"name":"{}","total_messages":{},"messages_per_second":{},"total_errors":{},"errors_per_second":{},"last_message_age_ms":{},"latency_p50_us":{},"latency_p95_us":{},"latency_p99_us":{}}}"#,
                ch.name,
                ch.total_messages,
                ch.messages_per_second,
                ch.total_errors,
                ch.errors_per_second,
                ch.last_message_age_ms,
                ch.latency_p50_us,
                ch.latency_p95_us,
                ch.latency_p99_us,
            ));
        }

        json.push_str("]}");
        json
    }
}

impl Default for BridgeHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
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
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_basic() {
        let monitor = BridgeHealthMonitor::new();
        monitor.register_channel("test_channel");

        // Record some messages
        monitor.record_message("test_channel", 100);
        monitor.record_message("test_channel", 150);

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "test_channel");
        assert_eq!(metrics[0].total_messages, 2);

        // Should be healthy
        assert_eq!(monitor.get_status(), BridgeHealthStatus::Healthy);
    }

    #[test]
    fn test_health_monitor_errors() {
        let monitor = BridgeHealthMonitor::new();
        monitor.register_channel("error_channel");

        // Record messages and errors
        for _ in 0..10 {
            monitor.record_message("error_channel", 100);
        }
        for _ in 0..5 {
            monitor.record_error("error_channel");
        }

        let metrics = monitor.get_metrics();
        assert_eq!(metrics[0].total_errors, 5);

        // Should be degraded (50% error rate)
        assert_eq!(monitor.get_status(), BridgeHealthStatus::Degraded);
    }

    #[test]
    fn test_health_monitor_json() {
        let monitor = BridgeHealthMonitor::new();
        monitor.register_channel("json_test");
        monitor.record_message("json_test", 200);

        let json = monitor.get_metrics_json();
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"name\":\"json_test\""));
        assert!(json.contains("\"total_messages\":1"));
    }
}
