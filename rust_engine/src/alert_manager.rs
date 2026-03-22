//! Centralized Alert Manager with Rate Limiting and Priority.
//!
//! Aggregates alerts from all components and dispatches via configured channels
//! (Telegram, Discord, PagerDuty) with rate limiting and priority handling.
//!
//! # Features
//!
//! - Priority-based alert routing (Info, Warning, Critical, Emergency)
//! - Rate limiting per category to prevent alert fatigue
//! - Deduplication to suppress repeated alerts
//! - Multi-channel dispatch (Telegram, Discord, PagerDuty, Email)
//! - Alert queue with async dispatch
//!
//! # Usage
//!
//! ```ignore
//! let mut manager = AlertManager::new();
//! manager.set_rate_limit("risk", 5); // Max 5 risk alerts per minute
//!
//! manager.alert(Alert {
//!     priority: AlertPriority::Critical,
//!     category: "risk".to_string(),
//!     title: "Circuit Breaker Tripped".to_string(),
//!     message: "Daily loss limit exceeded".to_string(),
//!     ..Default::default()
//! });
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Alert priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertPriority {
    /// Informational messages, no action required.
    Info = 0,
    /// Warnings that may require attention.
    Warning = 1,
    /// Critical issues requiring immediate attention.
    Critical = 2,
    /// Emergency situations requiring immediate response.
    Emergency = 3,
}

impl Default for AlertPriority {
    fn default() -> Self {
        Self::Info
    }
}

impl AlertPriority {
    /// Get emoji prefix for the priority level.
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Info => "ℹ️",
            Self::Warning => "⚠️",
            Self::Critical => "🔴",
            Self::Emergency => "🚨",
        }
    }
    
    /// Get text label for the priority level.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Critical => "CRITICAL",
            Self::Emergency => "EMERGENCY",
        }
    }
}

/// Alert dispatch channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertChannel {
    Telegram,
    Discord,
    PagerDuty,
    Email,
    Console,
}

/// An alert to be dispatched.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Priority level.
    pub priority: AlertPriority,
    /// Category for rate limiting (e.g., "risk", "execution", "system").
    pub category: String,
    /// Short alert title.
    pub title: String,
    /// Detailed alert message.
    pub message: String,
    /// Timestamp in nanoseconds since epoch.
    pub timestamp_ns: u64,
    /// Deduplication key (optional, for suppressing repeated alerts).
    pub dedup_key: Option<String>,
    /// Symbol (optional, for trade-related alerts).
    pub symbol: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl Default for Alert {
    fn default() -> Self {
        Self {
            priority: AlertPriority::Info,
            category: "general".to_string(),
            title: String::new(),
            message: String::new(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            dedup_key: None,
            symbol: None,
            metadata: HashMap::new(),
        }
    }
}

impl Alert {
    /// Create a new alert with the given priority and message.
    pub fn new(priority: AlertPriority, category: &str, title: &str, message: &str) -> Self {
        Self {
            priority,
            category: category.to_string(),
            title: title.to_string(),
            message: message.to_string(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            dedup_key: Some(format!("{}:{}", category, title)),
            symbol: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create an info alert.
    pub fn info(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertPriority::Info, category, title, message)
    }
    
    /// Create a warning alert.
    pub fn warning(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertPriority::Warning, category, title, message)
    }
    
    /// Create a critical alert.
    pub fn critical(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertPriority::Critical, category, title, message)
    }
    
    /// Create an emergency alert.
    pub fn emergency(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertPriority::Emergency, category, title, message)
    }
    
    /// Add a symbol to the alert.
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }
    
    /// Add metadata to the alert.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Format the alert for Telegram.
    pub fn format_telegram(&self) -> String {
        let mut msg = format!(
            "{} *{}*\n\n*{}*\n{}",
            self.priority.emoji(),
            self.priority.label(),
            self.title,
            self.message
        );
        
        if let Some(sym) = &self.symbol {
            msg.push_str(&format!("\n\n📈 Symbol: {}", sym));
        }
        
        if !self.metadata.is_empty() {
            msg.push_str("\n\n📋 Details:");
            for (k, v) in &self.metadata {
                msg.push_str(&format!("\n• {}: {}", k, v));
            }
        }
        
        msg
    }
    
    /// Format the alert for Discord.
    pub fn format_discord(&self) -> String {
        format!(
            "**{}** | {} | {}\n{}",
            self.priority.label(),
            self.category,
            self.title,
            self.message
        )
    }
}

/// Rate limiter state for a category.
struct RateLimitState {
    limit: u32,
    window_start: Instant,
    count: u32,
}

/// Centralized alert manager.
pub struct AlertManager {
    /// Rate limit per category (max alerts per minute).
    rate_limits: HashMap<String, RateLimitState>,
    /// Last alert time per dedup key (for suppression).
    dedup_cache: HashMap<String, Instant>,
    /// Dedup window duration.
    dedup_window: Duration,
    /// Enabled channels.
    enabled_channels: Vec<AlertChannel>,
    /// Alert queue for async dispatch.
    queue: Vec<Alert>,
    /// Minimum priority to dispatch.
    min_priority: AlertPriority,
    /// Total alerts queued.
    total_queued: u64,
    /// Total alerts suppressed (rate limit or dedup).
    total_suppressed: u64,
    /// Total alerts dispatched.
    total_dispatched: u64,
}

impl AlertManager {
    /// Create a new alert manager.
    pub fn new() -> Self {
        Self {
            rate_limits: HashMap::new(),
            dedup_cache: HashMap::new(),
            dedup_window: Duration::from_secs(300), // 5 minute dedup
            enabled_channels: vec![AlertChannel::Telegram, AlertChannel::Console],
            queue: Vec::new(),
            min_priority: AlertPriority::Info,
            total_queued: 0,
            total_suppressed: 0,
            total_dispatched: 0,
        }
    }
    
    /// Configure rate limit for a category.
    pub fn set_rate_limit(&mut self, category: &str, max_per_minute: u32) {
        self.rate_limits.insert(
            category.to_string(),
            RateLimitState {
                limit: max_per_minute,
                window_start: Instant::now(),
                count: 0,
            }
        );
    }
    
    /// Set the deduplication window.
    pub fn set_dedup_window(&mut self, duration: Duration) {
        self.dedup_window = duration;
    }
    
    /// Set the minimum priority to dispatch.
    pub fn set_min_priority(&mut self, priority: AlertPriority) {
        self.min_priority = priority;
    }
    
    /// Enable an alert channel.
    pub fn enable_channel(&mut self, channel: AlertChannel) {
        if !self.enabled_channels.contains(&channel) {
            self.enabled_channels.push(channel);
        }
    }
    
    /// Disable an alert channel.
    pub fn disable_channel(&mut self, channel: AlertChannel) {
        self.enabled_channels.retain(|c| *c != channel);
    }
    
    /// Queue an alert for dispatch.
    /// Returns true if the alert was queued, false if suppressed.
    pub fn alert(&mut self, alert: Alert) -> bool {
        // Check minimum priority
        if alert.priority < self.min_priority {
            self.total_suppressed += 1;
            return false;
        }
        
        // Check dedup
        if let Some(key) = &alert.dedup_key {
            if let Some(last) = self.dedup_cache.get(key) {
                if last.elapsed() < self.dedup_window {
                    self.total_suppressed += 1;
                    debug!("Alert suppressed (dedup): {}", alert.title);
                    return false;
                }
            }
            self.dedup_cache.insert(key.clone(), Instant::now());
        }
        
        // Check rate limit
        if let Some(state) = self.rate_limits.get_mut(&alert.category) {
            if state.window_start.elapsed() > Duration::from_secs(60) {
                state.window_start = Instant::now();
                state.count = 0;
            }
            if state.count >= state.limit {
                self.total_suppressed += 1;
                debug!("Alert rate limited: {}", alert.title);
                return false;
            }
            state.count += 1;
        }
        
        self.queue.push(alert);
        self.total_queued += 1;
        true
    }
    
    /// Get pending alerts for dispatch.
    pub fn drain_queue(&mut self) -> Vec<Alert> {
        let alerts = std::mem::take(&mut self.queue);
        self.total_dispatched += alerts.len() as u64;
        alerts
    }
    
    /// Get the number of pending alerts.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
    
    /// Clean up old dedup entries.
    pub fn cleanup_dedup_cache(&mut self) {
        let now = Instant::now();
        self.dedup_cache.retain(|_, last| now.duration_since(*last) < self.dedup_window * 2);
    }
    
    /// Get alert statistics.
    pub fn stats(&self) -> AlertStats {
        AlertStats {
            total_queued: self.total_queued,
            total_suppressed: self.total_suppressed,
            total_dispatched: self.total_dispatched,
            pending: self.queue.len(),
            enabled_channels: self.enabled_channels.clone(),
            rate_limits: self.rate_limits.keys().cloned().collect(),
        }
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert manager statistics.
#[derive(Debug, Clone)]
pub struct AlertStats {
    pub total_queued: u64,
    pub total_suppressed: u64,
    pub total_dispatched: u64,
    pub pending: usize,
    pub enabled_channels: Vec<AlertChannel>,
    pub rate_limits: Vec<String>,
}

// ── Alert Helper Functions ──

/// Quick alert function for critical issues.
pub fn alert_critical(manager: &mut AlertManager, category: &str, title: &str, message: &str) -> bool {
    manager.alert(Alert::critical(category, title, message))
}

/// Quick alert function for warnings.
pub fn alert_warning(manager: &mut AlertManager, category: &str, title: &str, message: &str) -> bool {
    manager.alert(Alert::warning(category, title, message))
}

/// Quick alert function for info messages.
pub fn alert_info(manager: &mut AlertManager, category: &str, title: &str, message: &str) -> bool {
    manager.alert(Alert::info(category, title, message))
}

// ── Alert Macros ──

/// Macro for creating and queuing a critical alert.
#[macro_export]
macro_rules! alert_critical {
    ($mgr:expr, $cat:expr, $title:expr, $($arg:tt)*) => {
        $mgr.alert($crate::alert_manager::Alert {
            priority: $crate::alert_manager::AlertPriority::Critical,
            category: $cat.to_string(),
            title: $title.to_string(),
            message: format!($($arg)*),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            dedup_key: Some(format!("{}:{}", $cat, $title)),
            symbol: None,
            metadata: std::collections::HashMap::new(),
        })
    };
}

/// Macro for creating and queuing a warning alert.
#[macro_export]
macro_rules! alert_warning {
    ($mgr:expr, $cat:expr, $title:expr, $($arg:tt)*) => {
        $mgr.alert($crate::alert_manager::Alert {
            priority: $crate::alert_manager::AlertPriority::Warning,
            category: $cat.to_string(),
            title: $title.to_string(),
            message: format!($($arg)*),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            dedup_key: Some(format!("{}:{}", $cat, $title)),
            symbol: None,
            metadata: std::collections::HashMap::new(),
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alert_creation() {
        let alert = Alert::critical("risk", "Circuit Breaker", "Tripped due to loss limit");
        
        assert_eq!(alert.priority, AlertPriority::Critical);
        assert_eq!(alert.category, "risk");
        assert_eq!(alert.title, "Circuit Breaker");
        assert!(alert.dedup_key.is_some());
    }
    
    #[test]
    fn test_rate_limiting() {
        let mut manager = AlertManager::new();
        manager.set_rate_limit("test", 2); // Max 2 per minute
        
        // First two should succeed
        assert!(manager.alert(Alert::info("test", "Test 1", "Message")));
        assert!(manager.alert(Alert::info("test", "Test 2", "Message")));
        
        // Third should be rate limited
        assert!(!manager.alert(Alert::info("test", "Test 3", "Message")));
        
        assert_eq!(manager.stats().total_suppressed, 1);
    }
    
    #[test]
    fn test_deduplication() {
        let mut manager = AlertManager::new();
        manager.set_dedup_window(Duration::from_secs(1));
        
        // First alert should succeed
        assert!(manager.alert(Alert::info("test", "Same Title", "Message 1")));
        
        // Same dedup key should be suppressed
        assert!(!manager.alert(Alert::info("test", "Same Title", "Message 2")));
        
        assert_eq!(manager.stats().total_suppressed, 1);
    }
    
    #[test]
    fn test_priority_filtering() {
        let mut manager = AlertManager::new();
        manager.set_min_priority(AlertPriority::Warning);
        
        // Info should be suppressed
        assert!(!manager.alert(Alert::info("test", "Info", "Message")));
        
        // Warning should pass
        assert!(manager.alert(Alert::warning("test", "Warning", "Message")));
        
        assert_eq!(manager.queue_len(), 1);
    }
    
    #[test]
    fn test_telegram_format() {
        let alert = Alert::critical("risk", "Circuit Breaker", "Daily loss exceeded")
            .with_symbol("BTC_USDT")
            .with_metadata("loss", "-$500");
        
        let formatted = alert.format_telegram();
        assert!(formatted.contains("CRITICAL"));
        assert!(formatted.contains("Circuit Breaker"));
        assert!(formatted.contains("BTC_USDT"));
        assert!(formatted.contains("loss"));
    }
}
