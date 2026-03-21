//! Signal Adapter — Typed wrapper around `signal_queue.rs` for the bridge.
//!
//! Provides a clean, typed interface for consuming trade signals from
//! Python's brain container. Wraps the existing lock-free SPSC signal queue
//! with type-safe enums and validation.
//!
//! # Signal Flow
//!
//! ```text
//! Python Brain                    Rust Execution
//! ┌──────────┐   /dev/shm/       ┌──────────────┐
//! │ Strategy │──▶signal_queue──▶│SignalAdapter  │──▶ Risk Check ──▶ Execute
//! │ Engine   │                   │(typed wrapper)│
//! └──────────┘                   └──────────────┘
//! ```
//!
//! Python writes raw `TradeIntent` structs to the SHM signal queue.
//! This adapter reads them, validates, and converts to typed `BrainSignal` structs.

use tracing::{debug, warn};

/// A validated trade signal from Python's brain container.
#[derive(Debug, Clone)]
pub struct BrainSignal {
    /// Symbol identifier.
    pub symbol_id: u16,
    /// Signal direction.
    pub direction: SignalDirection,
    /// Signal strength (0.0 to 1.0).
    pub strength: f64,
    /// Suggested position size as fraction of max (0.0 to 1.0).
    pub size_fraction: f64,
    /// Signal source/strategy name.
    pub source: SignalSource,
    /// Maximum acceptable entry price (FixedPrice, 0 = any).
    pub max_entry_price_fp: i64,
    /// Minimum acceptable entry price (FixedPrice, 0 = any).
    pub min_entry_price_fp: i64,
    /// Time-to-live in milliseconds (0 = no expiry).
    pub ttl_ms: u32,
    /// Whether this signal is reduce-only.
    pub reduce_only: bool,
    /// Nanosecond timestamp when the signal was generated.
    pub timestamp_ns: u64,
    /// Sequence number from the signal queue.
    pub sequence: u64,
}

/// Signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    /// Open or add to a long position.
    Long,
    /// Open or add to a short position.
    Short,
    /// Close/flatten the position.
    Flat,
}

impl SignalDirection {
    /// Convert from raw side byte (0 = buy/long, 1 = sell/short, 2 = flat).
    pub fn from_raw(side: u8) -> Self {
        match side {
            0 => SignalDirection::Long,
            1 => SignalDirection::Short,
            _ => SignalDirection::Flat,
        }
    }
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDirection::Long => write!(f, "LONG"),
            SignalDirection::Short => write!(f, "SHORT"),
            SignalDirection::Flat => write!(f, "FLAT"),
        }
    }
}

/// Source strategy that generated the signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSource {
    /// Reinforcement learning model.
    ReinforcementLearning,
    /// Sentiment analysis (NLP).
    Sentiment,
    /// Random forest classifier.
    RandomForest,
    /// Regime-based strategy.
    RegimeBased,
    /// Microstructure strategy (orderbook imbalance + VPIN).
    Microstructure,
    /// Composite/ensemble signal.
    Ensemble,
    /// Unknown source.
    Unknown,
}

impl SignalSource {
    /// Convert from raw source identifier.
    pub fn from_raw(source_id: u8) -> Self {
        match source_id {
            1 => SignalSource::ReinforcementLearning,
            2 => SignalSource::Sentiment,
            3 => SignalSource::RandomForest,
            4 => SignalSource::RegimeBased,
            5 => SignalSource::Microstructure,
            6 => SignalSource::Ensemble,
            _ => SignalSource::Unknown,
        }
    }
}

impl std::fmt::Display for SignalSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalSource::ReinforcementLearning => write!(f, "RL"),
            SignalSource::Sentiment => write!(f, "SENT"),
            SignalSource::RandomForest => write!(f, "RF"),
            SignalSource::RegimeBased => write!(f, "REGIME"),
            SignalSource::Microstructure => write!(f, "MICRO"),
            SignalSource::Ensemble => write!(f, "ENSEMBLE"),
            SignalSource::Unknown => write!(f, "UNK"),
        }
    }
}

/// Signal adapter that wraps the existing signal queue.
///
/// Reads raw signals from the SPSC SHM queue, validates them,
/// and provides typed BrainSignal structs.
pub struct SignalAdapter {
    /// Total signals consumed.
    total_consumed: u64,
    /// Total signals rejected (validation failure).
    total_rejected: u64,
    /// Last signal timestamp for ordering verification.
    last_signal_ns: u64,
    /// Minimum signal strength to accept (filters noise).
    min_strength: f64,
}

impl SignalAdapter {
    /// Create a new signal adapter.
    ///
    /// `min_strength` filters out weak signals (0.0 accepts all).
    pub fn new(min_strength: f64) -> Self {
        Self {
            total_consumed: 0,
            total_rejected: 0,
            last_signal_ns: 0,
            min_strength: min_strength.clamp(0.0, 1.0),
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(0.1) // Minimum 10% signal strength
    }

    /// Convert a raw TradeIntent from the signal queue to a typed BrainSignal.
    ///
    /// Returns `Some(BrainSignal)` if the signal is valid, `None` if it fails
    /// validation checks.
    pub fn process_raw_signal(
        &mut self,
        symbol_id: u16,
        side: u8,
        strength: f64,
        size_fraction: f64,
        source_id: u8,
        max_entry_price_fp: i64,
        min_entry_price_fp: i64,
        ttl_ms: u32,
        reduce_only: bool,
        timestamp_ns: u64,
        sequence: u64,
    ) -> Option<BrainSignal> {
        // Validation: strength must be above threshold
        if strength < self.min_strength {
            self.total_rejected += 1;
            return None;
        }

        // Validation: strength must be in valid range
        if !strength.is_finite() || strength > 1.0 {
            warn!("[signal-adapter] Invalid strength: {}", strength);
            self.total_rejected += 1;
            return None;
        }

        // Validation: size_fraction must be positive
        if size_fraction <= 0.0 || !size_fraction.is_finite() || size_fraction > 1.0 {
            warn!("[signal-adapter] Invalid size_fraction: {}", size_fraction);
            self.total_rejected += 1;
            return None;
        }

        // Validation: price constraints must be consistent
        if max_entry_price_fp > 0 && min_entry_price_fp > 0
            && max_entry_price_fp < min_entry_price_fp
        {
            warn!("[signal-adapter] Price constraint violation: max < min");
            self.total_rejected += 1;
            return None;
        }

        let signal = BrainSignal {
            symbol_id,
            direction: SignalDirection::from_raw(side),
            strength: strength.clamp(0.0, 1.0),
            size_fraction: size_fraction.clamp(0.0, 1.0),
            source: SignalSource::from_raw(source_id),
            max_entry_price_fp,
            min_entry_price_fp,
            ttl_ms,
            reduce_only,
            timestamp_ns,
            sequence,
        };

        self.total_consumed += 1;
        self.last_signal_ns = timestamp_ns;

        debug!(
            "[signal-adapter] Signal: {} {} strength={:.3} size={:.3} src={}",
            signal.direction, symbol_id, signal.strength, signal.size_fraction, signal.source
        );

        Some(signal)
    }

    /// Get the total signals consumed.
    #[inline]
    pub fn total_consumed(&self) -> u64 {
        self.total_consumed
    }

    /// Get the total signals rejected.
    #[inline]
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    /// Get the acceptance rate (0.0 to 1.0).
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.total_consumed + self.total_rejected;
        if total == 0 {
            1.0
        } else {
            self.total_consumed as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_direction() {
        assert_eq!(SignalDirection::from_raw(0), SignalDirection::Long);
        assert_eq!(SignalDirection::from_raw(1), SignalDirection::Short);
        assert_eq!(SignalDirection::from_raw(2), SignalDirection::Flat);
        assert_eq!(SignalDirection::from_raw(255), SignalDirection::Flat);
    }

    #[test]
    fn test_signal_source() {
        assert_eq!(SignalSource::from_raw(1), SignalSource::ReinforcementLearning);
        assert_eq!(SignalSource::from_raw(6), SignalSource::Ensemble);
        assert_eq!(SignalSource::from_raw(99), SignalSource::Unknown);
    }

    #[test]
    fn test_valid_signal() {
        let mut adapter = SignalAdapter::with_defaults();
        let signal = adapter.process_raw_signal(
            1,     // symbol_id
            0,     // side (long)
            0.85,  // strength
            0.5,   // size_fraction
            5,     // source (microstructure)
            0,     // max_entry
            0,     // min_entry
            5000,  // ttl_ms
            false, // reduce_only
            123456789, // timestamp
            1,     // sequence
        );
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert_eq!(s.direction, SignalDirection::Long);
        assert!((s.strength - 0.85).abs() < f64::EPSILON);
        assert_eq!(s.source, SignalSource::Microstructure);
    }

    #[test]
    fn test_weak_signal_rejected() {
        let mut adapter = SignalAdapter::new(0.5); // Min 50% strength
        let signal = adapter.process_raw_signal(
            1, 0, 0.3, 0.5, 1, 0, 0, 5000, false, 0, 1,
        );
        assert!(signal.is_none());
        assert_eq!(adapter.total_rejected(), 1);
    }

    #[test]
    fn test_invalid_price_constraint() {
        let mut adapter = SignalAdapter::with_defaults();
        let signal = adapter.process_raw_signal(
            1, 0, 0.85, 0.5, 1,
            100,  // max < min
            200,  // min > max
            5000, false, 0, 1,
        );
        assert!(signal.is_none());
    }

    #[test]
    fn test_acceptance_rate() {
        let mut adapter = SignalAdapter::new(0.5);
        adapter.process_raw_signal(1, 0, 0.8, 0.5, 1, 0, 0, 0, false, 0, 1); // Accept
        adapter.process_raw_signal(1, 0, 0.3, 0.5, 1, 0, 0, 0, false, 0, 2); // Reject
        assert!((adapter.acceptance_rate() - 0.5).abs() < f64::EPSILON);
    }
}
