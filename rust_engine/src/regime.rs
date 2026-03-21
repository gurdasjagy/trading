//! Regime state interface for the deterministic hot path.
//!
//! **Issue 2 Rewrite**: Replaced JSON file polling + `parking_lot::RwLock` with
//! shared-memory seqlock reader via `regime_shm::SharedMemRegimeReader`.
//!
//! The Python `RegimeService` writes regime weights to `/dev/shm/regime_weights`
//! using a seqlock pattern. The Rust hot path reads them lock-free via this module.
//!
//! ## Migration Notes
//!
//! - `RegimeState` (old, heap-allocated with `Vec<String>` fields) is replaced by
//!   `RegimeWeights` (fixed-size `#[repr(C, packed)]`, zero allocation).
//! - `RegimeReader` (old, file-polling + RwLock) is replaced by
//!   `SharedMemRegimeReader` (seqlock, mmap, lock-free).
//! - The public API is preserved where possible for backward compatibility.

use crate::regime_shm::{RegimeWeights, SharedMemRegimeReader};

// Re-export core types for backward compatibility
pub use crate::regime_shm::{regime_type, volatility_type};

// ═══════════════════════════════════════════════════════════════════════════
// RegimeState — backward-compatible wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Backward-compatible regime state.
///
/// Wraps `RegimeWeights` and provides the same API as the old `RegimeState`
/// struct (with `String` fields) for code that hasn't been migrated yet.
///
/// New code should use `RegimeWeights` directly.
#[derive(Debug, Clone)]
pub struct RegimeState {
    /// The underlying fixed-size regime weights.
    pub weights: RegimeWeights,
}

impl RegimeState {
    /// Create from raw `RegimeWeights`.
    pub fn from_weights(weights: RegimeWeights) -> Self {
        Self { weights }
    }

    /// Returns `true` if this state is expired.
    pub fn is_expired(&self) -> bool {
        self.weights.is_expired()
    }

    /// Returns the overall regime as a string (for backward compatibility).
    pub fn overall_regime(&self) -> &'static str {
        match self.weights.overall_regime {
            regime_type::TRENDING_BULLISH => "trending_bullish",
            regime_type::TRENDING_BEARISH => "trending_bearish",
            regime_type::RANGING => "ranging",
            regime_type::HIGH_VOLATILITY => "high_volatility",
            regime_type::CHOPPY => "choppy",
            _ => "unknown",
        }
    }

    /// Returns the volatility regime as a string.
    pub fn volatility_regime(&self) -> &'static str {
        match self.weights.volatility_regime {
            volatility_type::LOW => "low",
            volatility_type::MODERATE => "moderate",
            volatility_type::HIGH => "high",
            volatility_type::EXTREME => "extreme",
            _ => "high",
        }
    }

    /// Aggregate sentiment score in [-1.0, 1.0].
    pub fn sentiment_score(&self) -> f64 {
        self.weights.sentiment_score()
    }

    /// Position size multiplier.
    pub fn recommended_position_scale(&self) -> f64 {
        self.weights.position_scale()
    }

    /// Returns `true` if the given strategy is blocked by the current regime.
    pub fn is_strategy_blocked(&self, strategy_id: u8) -> bool {
        self.weights.is_strategy_blocked(strategy_id)
    }

    /// Returns the effective leverage cap.
    pub fn effective_leverage_cap(&self, default: i32) -> i32 {
        self.weights.effective_leverage_cap(default)
    }

    /// Momentum weight for position sizing [0.0, 1.0].
    /// Used by the Microstructure Imbalance Strategy to scale positions.
    /// High volatility / bearish regime → lower weight → smaller positions.
    pub fn momentum_weight(&self) -> f64 {
        self.weights.position_scale().clamp(0.0, 1.0)
    }

    /// Returns a conservative default regime state.
    pub fn safe_default() -> Self {
        Self {
            weights: RegimeWeights::safe_default(),
        }
    }
}

impl Default for RegimeState {
    fn default() -> Self {
        Self::safe_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RegimeReader — backward-compatible wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Non-blocking regime state reader.
///
/// Wraps `SharedMemRegimeReader` and provides the same API as the old
/// `RegimeReader` (file-polling + RwLock) for backward compatibility.
pub struct RegimeReader {
    inner: SharedMemRegimeReader,
}

impl RegimeReader {
    /// Create a new `RegimeReader`.
    ///
    /// Args:
    /// * `file_path` — path to the shared memory file (e.g., `/dev/shm/regime_weights`).
    /// * `_poll_interval_ms` — ignored (kept for API compatibility; seqlock doesn't poll).
    pub fn new(file_path: &str, _poll_interval_ms: u64) -> Self {
        Self {
            inner: SharedMemRegimeReader::new(file_path),
        }
    }

    /// Return the current regime state.
    ///
    /// Reads from shared memory via seqlock. Returns the cached value if
    /// the read fails, or safe defaults if everything is expired.
    pub fn get_current(&mut self) -> RegimeState {
        let weights = self.inner.get_current();
        RegimeState::from_weights(weights)
    }

    /// Try to read fresh regime weights.
    pub fn try_read(&mut self) -> Option<RegimeWeights> {
        self.inner.try_read()
    }

    /// Get the raw `RegimeWeights` (preferred for new code).
    pub fn get_weights(&mut self) -> RegimeWeights {
        self.inner.get_current()
    }

    /// Background poll loop — **NO-OP** in the new architecture.
    ///
    /// The seqlock reader doesn't need a background poll; it reads
    /// on-demand. This method is kept for API compatibility but does nothing.
    pub async fn poll_loop(&self) {
        // In the new architecture, reads happen on-demand via seqlock.
        // This loop exists only for backward compatibility with code that
        // spawns `tokio::spawn(reader.poll_loop())`.
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        }
    }
}
