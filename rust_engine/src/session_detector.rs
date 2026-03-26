//! CATEGORY 8: Session-Aware Trading Detection.
//!
//! Detects Asian/London/New York trading sessions for crypto markets.
//! While crypto trades 24/7, volatility and volume patterns follow
//! traditional market hours because institutional desks operate during
//! business hours.
//!
//! # Session Definitions (UTC)
//!
//! - **Asian Session**: 00:00-08:00 UTC (Tokyo/Singapore/HK open)
//! - **London Session**: 07:00-16:00 UTC (overlaps with Asian close)
//! - **New York Session**: 12:00-21:00 UTC (overlaps with London close)
//! - **Asian/London Overlap**: 07:00-08:00 UTC (high volatility)
//! - **London/NY Overlap**: 12:00-16:00 UTC (highest volatility)
//!
//! # Usage
//!
//! The strategy engine uses session detection to:
//! - Adjust position sizes (larger during high-vol overlaps)
//! - Modify VPIN thresholds (tighter during low-vol Asian session)
//! - Select appropriate strategies (mean-reversion during Asian, momentum during NY)

use tracing::debug;

/// Trading session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingSession {
    /// Asian session: 00:00-08:00 UTC (lower volatility, range-bound).
    Asian,
    /// London session: 07:00-16:00 UTC (increasing volatility, breakouts).
    London,
    /// New York session: 12:00-21:00 UTC (highest volume for crypto).
    NewYork,
    /// Asian/London overlap: 07:00-08:00 UTC (moderate volatility spike).
    AsianLondonOverlap,
    /// London/NY overlap: 12:00-16:00 UTC (highest volatility and volume).
    LondonNyOverlap,
    /// Late NY / pre-Asian: 21:00-00:00 UTC (low volatility, thin books).
    LateSession,
}

impl TradingSession {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Asian => "Asian",
            Self::London => "London",
            Self::NewYork => "NewYork",
            Self::AsianLondonOverlap => "Asian/London Overlap",
            Self::LondonNyOverlap => "London/NY Overlap",
            Self::LateSession => "Late Session",
        }
    }

    /// Get the volatility multiplier for this session.
    /// Based on historical crypto volatility patterns:
    /// - Overlaps have 1.5-2x normal volatility
    /// - Asian session has 0.6-0.8x normal volatility
    /// - NY session has 1.0-1.3x normal volatility
    pub fn volatility_multiplier(&self) -> f64 {
        match self {
            Self::Asian => 0.7,
            Self::London => 1.0,
            Self::NewYork => 1.2,
            Self::AsianLondonOverlap => 1.3,
            Self::LondonNyOverlap => 1.5,
            Self::LateSession => 0.5,
        }
    }

    /// Get the recommended position size multiplier.
    /// Conservative approach: reduce size during low-vol, increase during high-vol.
    pub fn size_multiplier(&self) -> f64 {
        match self {
            Self::Asian => 0.8,          // Smaller in low-vol
            Self::London => 1.0,         // Normal
            Self::NewYork => 1.0,        // Normal
            Self::AsianLondonOverlap => 1.1, // Slightly larger for breakouts
            Self::LondonNyOverlap => 1.2,    // Larger for highest liquidity
            Self::LateSession => 0.6,    // Minimal in thin markets
        }
    }

    /// Whether mean-reversion strategies are preferred in this session.
    pub fn prefer_mean_reversion(&self) -> bool {
        matches!(self, Self::Asian | Self::LateSession)
    }

    /// Whether momentum strategies are preferred in this session.
    pub fn prefer_momentum(&self) -> bool {
        matches!(
            self,
            Self::London | Self::NewYork | Self::AsianLondonOverlap | Self::LondonNyOverlap
        )
    }

    /// Get the recommended VPIN toxic threshold for this session.
    /// Tighter during low-vol sessions where toxic flow is more impactful.
    pub fn vpin_toxic_threshold(&self) -> f64 {
        match self {
            Self::Asian => 0.55,         // Tighter: toxic flow more dangerous in thin books
            Self::London => 0.65,        // Standard
            Self::NewYork => 0.65,       // Standard
            Self::AsianLondonOverlap => 0.60, // Moderate
            Self::LondonNyOverlap => 0.70,    // Relaxed: more liquidity absorbs toxic flow
            Self::LateSession => 0.50,   // Very tight: thin books
        }
    }
}

/// Session detector that determines current trading session from UTC time.
pub struct SessionDetector {
    /// Current detected session (cached).
    current_session: TradingSession,
    /// Last hour checked (to avoid recomputing every tick).
    last_hour: u32,
}

impl SessionDetector {
    /// Create a new session detector.
    pub fn new() -> Self {
        Self {
            current_session: TradingSession::Asian,
            last_hour: 255, // Force initial detection
        }
    }

    /// Detect the current trading session from a UTC timestamp.
    ///
    /// # Arguments
    /// * `timestamp_secs` - Unix timestamp in seconds (UTC)
    ///
    /// # Returns
    /// The current trading session.
    pub fn detect(&mut self, timestamp_secs: u64) -> TradingSession {
        // Extract hour of day in UTC
        let secs_in_day = timestamp_secs % 86400;
        let hour = (secs_in_day / 3600) as u32;

        // Only recompute if hour changed
        if hour == self.last_hour {
            return self.current_session;
        }
        self.last_hour = hour;

        self.current_session = Self::session_for_hour(hour);
        debug!(
            "[session] Hour {} UTC -> {} (vol_mult={:.1}, size_mult={:.1})",
            hour,
            self.current_session.name(),
            self.current_session.volatility_multiplier(),
            self.current_session.size_multiplier(),
        );

        self.current_session
    }

    /// Pure function: determine session from UTC hour.
    fn session_for_hour(hour: u32) -> TradingSession {
        match hour {
            0..=6 => TradingSession::Asian,
            7 => TradingSession::AsianLondonOverlap,
            8..=11 => TradingSession::London,
            12..=15 => TradingSession::LondonNyOverlap,
            16..=20 => TradingSession::NewYork,
            21..=23 => TradingSession::LateSession,
            _ => TradingSession::Asian, // Unreachable but safe
        }
    }

    /// Get the current session without updating.
    pub fn current(&self) -> TradingSession {
        self.current_session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asian_session() {
        let mut detector = SessionDetector::new();
        // 03:00 UTC
        let session = detector.detect(3 * 3600);
        assert_eq!(session, TradingSession::Asian);
        assert!(session.prefer_mean_reversion());
        assert!(!session.prefer_momentum());
    }

    #[test]
    fn test_london_ny_overlap() {
        let mut detector = SessionDetector::new();
        // 14:00 UTC
        let session = detector.detect(14 * 3600);
        assert_eq!(session, TradingSession::LondonNyOverlap);
        assert!(session.volatility_multiplier() > 1.0);
        assert!(session.prefer_momentum());
    }

    #[test]
    fn test_late_session() {
        let mut detector = SessionDetector::new();
        // 22:00 UTC
        let session = detector.detect(22 * 3600);
        assert_eq!(session, TradingSession::LateSession);
        assert!(session.size_multiplier() < 1.0);
    }

    #[test]
    fn test_session_caching() {
        let mut detector = SessionDetector::new();
        // Same hour should return cached result
        let s1 = detector.detect(3 * 3600);
        let s2 = detector.detect(3 * 3600 + 1800); // 30 min later, same hour
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_vpin_thresholds() {
        assert!(TradingSession::LateSession.vpin_toxic_threshold() < TradingSession::LondonNyOverlap.vpin_toxic_threshold());
    }
}
