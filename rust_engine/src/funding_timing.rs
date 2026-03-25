//! FEATURE 1: Funding Rate Timestamp-Aware Entry/Exit
//!
//! Institutional bots enter positions seconds before the funding snapshot
//! and exit immediately after to minimize directional exposure time.
//!
//! Funding snapshots occur every 8 hours on most exchanges:
//! - Binance: 00:00, 08:00, 16:00 UTC
//! - Bybit:   00:00, 08:00, 16:00 UTC
//! - Gate.io: 00:00, 08:00, 16:00 UTC

use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// Funding timing engine for optimal entry/exit around funding snapshots.
pub struct FundingTimingEngine {
    /// Seconds before funding snapshot to enter position (default: 60).
    entry_window_secs: u64,
    /// Seconds after funding snapshot to exit position (default: 30).
    exit_window_secs: u64,
    /// Funding intervals in hours from midnight UTC (default: [0, 8, 16]).
    funding_hours: Vec<u64>,
}

impl FundingTimingEngine {
    /// Create a new FundingTimingEngine with default settings.
    pub fn new() -> Self {
        Self {
            entry_window_secs: 60,
            exit_window_secs: 30,
            funding_hours: vec![0, 8, 16],
        }
    }

    /// Create with custom entry/exit windows.
    pub fn with_windows(entry_secs: u64, exit_secs: u64) -> Self {
        Self {
            entry_window_secs: entry_secs,
            exit_window_secs: exit_secs,
            funding_hours: vec![0, 8, 16],
        }
    }

    /// Create with custom funding schedule.
    pub fn with_schedule(entry_secs: u64, exit_secs: u64, hours: Vec<u64>) -> Self {
        Self {
            entry_window_secs: entry_secs,
            exit_window_secs: exit_secs,
            funding_hours: hours,
        }
    }

    /// Returns seconds until the next funding snapshot.
    pub fn seconds_until_next_funding(&self) -> u64 {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let secs_into_day = now_secs % 86400;
        let current_hour_secs = secs_into_day;

        for &funding_hour in &self.funding_hours {
            let funding_secs = funding_hour * 3600;
            if funding_secs > current_hour_secs {
                return funding_secs - current_hour_secs;
            }
        }

        // Wrap to next day's first funding hour
        let first_funding_secs = self.funding_hours.first().copied().unwrap_or(0) * 3600;
        86400 - current_hour_secs + first_funding_secs
    }

    /// Returns seconds since the last funding snapshot.
    pub fn seconds_since_last_funding(&self) -> u64 {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let secs_into_day = now_secs % 86400;

        // Find the most recent funding hour that has passed
        let mut last_funding_secs = 0u64;
        for &funding_hour in &self.funding_hours {
            let fs = funding_hour * 3600;
            if fs <= secs_into_day {
                last_funding_secs = fs;
            }
        }

        // If no funding hour has passed today, use last hour from previous day
        if last_funding_secs == 0 && secs_into_day < self.funding_hours.first().copied().unwrap_or(0) * 3600 {
            let last_hour = self.funding_hours.last().copied().unwrap_or(16);
            last_funding_secs = 86400 - (last_hour * 3600);
            return secs_into_day + last_funding_secs;
        }

        secs_into_day - last_funding_secs
    }

    /// Returns true if we're within the entry window before funding.
    ///
    /// The entry window is the last N seconds before the funding snapshot,
    /// allowing the bot to enter just before funding is credited.
    pub fn is_entry_window(&self) -> bool {
        let secs = self.seconds_until_next_funding();
        let in_window = secs <= self.entry_window_secs;
        if in_window {
            debug!(
                "[funding-timing] ENTRY WINDOW: {}s until next funding snapshot",
                secs
            );
        }
        in_window
    }

    /// Returns true if we should exit (funding was just credited).
    ///
    /// The exit window is the first N seconds after the funding snapshot,
    /// allowing the bot to exit immediately after receiving funding.
    pub fn is_exit_window(&self) -> bool {
        let since = self.seconds_since_last_funding();
        let in_window = since <= self.exit_window_secs;
        if in_window {
            debug!(
                "[funding-timing] EXIT WINDOW: {}s since last funding snapshot",
                since
            );
        }
        in_window
    }

    /// Get the entry window size in seconds.
    pub fn entry_window(&self) -> u64 {
        self.entry_window_secs
    }

    /// Get the exit window size in seconds.
    pub fn exit_window(&self) -> u64 {
        self.exit_window_secs
    }

    /// Log current timing state for telemetry.
    pub fn log_timing_state(&self) {
        let until = self.seconds_until_next_funding();
        let since = self.seconds_since_last_funding();
        info!(
            "[funding-timing] Next funding in {}s ({}m), last funding {}s ago, entry={} exit={}",
            until,
            until / 60,
            since,
            self.is_entry_window(),
            self.is_exit_window()
        );
    }
}

impl Default for FundingTimingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let engine = FundingTimingEngine::new();
        assert_eq!(engine.entry_window_secs, 60);
        assert_eq!(engine.exit_window_secs, 30);
        assert_eq!(engine.funding_hours, vec![0, 8, 16]);
    }

    #[test]
    fn test_seconds_until_next_funding_is_bounded() {
        let engine = FundingTimingEngine::new();
        let secs = engine.seconds_until_next_funding();
        // Should always be <= 8 hours (28800 seconds)
        assert!(secs <= 28800, "secs_until={} exceeds 8h", secs);
        assert!(secs > 0, "secs_until should be > 0");
    }

    #[test]
    fn test_entry_exit_mutually_exclusive_mostly() {
        // Entry and exit windows are very short (60s and 30s) out of 28800s,
        // so in normal operation they should rarely both be true.
        let engine = FundingTimingEngine::new();
        let _entry = engine.is_entry_window();
        let _exit = engine.is_exit_window();
        // Just verify no panics
    }

    #[test]
    fn test_custom_windows() {
        let engine = FundingTimingEngine::with_windows(120, 60);
        assert_eq!(engine.entry_window_secs, 120);
        assert_eq!(engine.exit_window_secs, 60);
    }

    #[test]
    fn test_custom_schedule() {
        let engine = FundingTimingEngine::with_schedule(60, 30, vec![0, 4, 8, 12, 16, 20]);
        assert_eq!(engine.funding_hours.len(), 6);
        let secs = engine.seconds_until_next_funding();
        // 4-hour intervals = max 14400 seconds
        assert!(secs <= 14400);
    }
}
