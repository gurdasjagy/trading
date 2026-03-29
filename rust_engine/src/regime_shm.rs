//! Shared-memory regime reader for Python → Rust regime weight updates.
//!
//! The Python `RegimeService` computes macro-regime weights every N minutes
//! and writes them to `/dev/shm/regime_weights` using a seqlock pattern.
//! This module provides a lock-free reader for the Rust hot path.
//!
//! **Issue 2**: Replaces `regime.rs` JSON file + `parking_lot::RwLock` approach.
//!
//! # Seqlock Read Protocol
//!
//! ```text
//! loop {
//!     seq1 = read_sequence()       // atomic load
//!     if seq1 is odd → retry       // writer is active
//!     data = read_payload()        // memcpy
//!     seq2 = read_sequence()       // atomic load
//!     if seq1 == seq2 → return data  // consistent read
//!     // else: writer was active during read, retry
//! }
//! ```

use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Default shared memory path for regime weights.
pub const REGIME_SHM_PATH: &str = "/dev/shm/regime_weights";

/// Magic bytes for regime shared memory validation.
pub const REGIME_MAGIC: u64 = 0x5245_4749_4D45_5754; // "REGIMEWT"

/// Version of the regime weights layout.
pub const REGIME_VERSION: u32 = 1;

/// Maximum retries for seqlock read before giving up.
const MAX_READ_RETRIES: u32 = 100;

// ═══════════════════════════════════════════════════════════════════════════
// RegimeWeights — fixed-size, no heap allocation
// ═══════════════════════════════════════════════════════════════════════════

/// Regime enum values for the `overall_regime` field.
pub mod regime_type {
    pub const UNKNOWN: u8 = 0;
    pub const TRENDING_BULLISH: u8 = 1;
    pub const TRENDING_BEARISH: u8 = 2;
    pub const RANGING: u8 = 3;
    pub const HIGH_VOLATILITY: u8 = 4;
    pub const CHOPPY: u8 = 5;
}

/// Volatility regime enum values.
pub mod volatility_type {
    pub const LOW: u8 = 0;
    pub const MODERATE: u8 = 1;
    pub const HIGH: u8 = 2;
    pub const EXTREME: u8 = 3;
}

/// Fixed-size regime weights. Written by Python, read by Rust.
///
/// **No `Vec`, no `String`, no heap allocation.** Every field is fixed-size.
///
/// Layout must match Python `struct.pack` format exactly.
/// Total size: 112 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct RegimeWeights {
    /// Magic bytes for validation.
    pub magic: u64,                        // 8 bytes
    /// Layout version.
    pub version: u32,                      // 4 bytes
    /// Padding.
    pub _pad0: u32,                        // 4 bytes
    /// Seqlock sequence — odd = writing, even = consistent.
    pub sequence: u64,                     // 8 bytes
    /// Unix-millisecond timestamp when this state was computed.
    pub timestamp_ms: i64,                 // 8 bytes
    /// Overall market regime (see `regime_type` module).
    pub overall_regime: u8,                // 1 byte
    /// Volatility sub-regime (see `volatility_type` module).
    pub volatility_regime: u8,             // 1 byte
    /// Padding.
    pub _pad1: [u8; 2],                    // 2 bytes
    /// Aggregate sentiment score [-10000, 10000] (fixed-point 1e4, maps to [-1.0, 1.0]).
    pub sentiment_score_fp: i32,           // 4 bytes
    /// Sentiment confidence [0, 10000] (fixed-point 1e4, maps to [0.0, 1.0]).
    pub sentiment_confidence_fp: i32,      // 4 bytes
    /// Fear & Greed index [0, 100].
    pub fear_greed_index: i32,             // 4 bytes
    /// BTC dominance trend: 0=flat, 1=rising, 2=falling.
    pub btc_dominance_trend: u8,           // 1 byte
    /// Funding rate bias: 0=neutral, 1=long_crowded, 2=short_crowded.
    pub funding_rate_bias: u8,             // 1 byte
    /// Padding.
    pub _pad2: [u8; 2],                    // 2 bytes
    /// Cross-asset correlation [0, 10000] (fixed-point 1e4).
    pub cross_asset_correlation_fp: i32,   // 4 bytes
    /// News impact score [0, 10000] (fixed-point 1e4).
    pub news_impact_score_fp: i32,         // 4 bytes
    /// Position size multiplier [0, 40000] (fixed-point 1e4, maps to [0.0, 4.0]).
    pub position_scale_fp: i32,            // 4 bytes
    /// Maximum leverage override (0 = no override).
    pub max_leverage_override: i32,        // 4 bytes
    /// TTL in seconds. After expiry, safe defaults are used.
    pub ttl_seconds: i32,                  // 4 bytes
    /// Bitmask of allowed strategy IDs (bit N = strategy N is allowed).
    pub allowed_strategies_mask: u64,      // 8 bytes
    /// Bitmask of blocked strategy IDs (bit N = strategy N is blocked).
    pub blocked_strategies_mask: u64,      // 8 bytes
    /// Reserved for future expansion.
    pub _reserved: [u8; 24],               // 24 bytes
}
// Total: 8+4+4+8+8+1+1+2+4+4+4+1+1+2+4+4+4+4+4+8+8+24 = 112 bytes

impl Default for RegimeWeights {
    fn default() -> Self {
        Self {
            magic: REGIME_MAGIC,
            version: REGIME_VERSION,
            _pad0: 0,
            sequence: 0,
            timestamp_ms: 0,
            overall_regime: regime_type::UNKNOWN,
            volatility_regime: volatility_type::HIGH,
            _pad1: [0; 2],
            sentiment_score_fp: 0,
            sentiment_confidence_fp: 0,
            fear_greed_index: 50,
            btc_dominance_trend: 0,
            funding_rate_bias: 0,
            _pad2: [0; 2],
            cross_asset_correlation_fp: 0,
            news_impact_score_fp: 0,
            position_scale_fp: 5000, // 0.5 = conservative default
            max_leverage_override: 0,
            ttl_seconds: 600,
            allowed_strategies_mask: u64::MAX, // all allowed by default
            blocked_strategies_mask: 0,
            _reserved: [0; 24],
        }
    }
}

impl RegimeWeights {
    /// Returns `true` if this regime state is expired.
    pub fn is_expired(&self) -> bool {
        if self.timestamp_ms == 0 {
            return true;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        (now_ms - self.timestamp_ms) > (self.ttl_seconds as i64 * 1000)
    }

    /// Get the position scale as f64 (from fixed-point 1e4).
    pub fn position_scale(&self) -> f64 {
        self.position_scale_fp as f64 / 10_000.0
    }

    /// Get the sentiment score as f64 (from fixed-point 1e4).
    pub fn sentiment_score(&self) -> f64 {
        self.sentiment_score_fp as f64 / 10_000.0
    }

    /// Returns `true` if strategy with the given ID is blocked.
    pub fn is_strategy_blocked(&self, strategy_id: u8) -> bool {
        if strategy_id >= 64 {
            return false;
        }
        (self.blocked_strategies_mask >> strategy_id) & 1 == 1
    }

    /// Returns the effective leverage cap.
    pub fn effective_leverage_cap(&self, default: i32) -> i32 {
        if self.max_leverage_override > 0 {
            default.min(self.max_leverage_override)
        } else {
            default
        }
    }

    /// Returns a conservative safe default.
    /// BUG #10 FIX: Use current timestamp so the safe default is treated as
    /// fresh (not expired). An expired safe default would also return safe_default()
    /// but incurs unnecessary is_expired() overhead on every read.
    /// Also changed position_scale_fp from 5000 (0.5x) to 10000 (1.0x) -- when
    /// there's no regime signal, use full sizing and let other risk controls do their job.
    pub fn safe_default() -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        Self {
            magic: REGIME_MAGIC,
            sequence: 0,
            timestamp_ms: now_ms, // FIX: was 0, which makes is_expired() always true
            overall_regime: regime_type::UNKNOWN,
            volatility_regime: volatility_type::HIGH,
            position_scale_fp: 10_000, // FIX: was 5000 (0.5x). Use 1.0x when no regime data.
            ttl_seconds: 600,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SharedMemRegimeReader
// ═══════════════════════════════════════════════════════════════════════════

/// Lock-free regime reader from shared memory.
///
/// Uses the seqlock pattern to ensure consistent reads without any locks.
/// Designed for the hot path — `try_read()` completes in < 1µs.
/// BUG 14 FIX: Uses `UnsafeCell` for interior mutability so that `get_current()`
/// and `try_read()` take `&self` instead of `&mut self`. This eliminates the
/// undefined-behavior const-to-mut cast in the strategy thread.
///
/// Safety: `SharedMemRegimeReader` is only accessed from one thread (the strategy
/// evaluator). `Send` is derived automatically. It is NOT `Sync`.
pub struct SharedMemRegimeReader {
    mmap: std::cell::UnsafeCell<Option<Mmap>>,
    path: String,
    /// Cached last good read (used when SHM is unavailable).
    cached: std::cell::UnsafeCell<RegimeWeights>,
}

// Safety: SharedMemRegimeReader is confined to a single thread (strategy evaluator).
// It is never shared across threads concurrently.
unsafe impl Send for SharedMemRegimeReader {}

impl SharedMemRegimeReader {
    /// Create a new `SharedMemRegimeReader`.
    ///
    /// If the shared memory file doesn't exist yet (Python hasn't started),
    /// returns a reader with the safe default. It will attempt to open the
    /// file on subsequent `try_read()` calls.
    pub fn new(path: &str) -> Self {
        let mmap = Self::try_open(path);
        Self {
            mmap: std::cell::UnsafeCell::new(mmap),
            path: path.to_string(),
            cached: std::cell::UnsafeCell::new(RegimeWeights::safe_default()),
        }
    }

    /// Attempt to open the shared memory file.
    fn try_open(path: &str) -> Option<Mmap> {
        let file = OpenOptions::new().read(true).open(path).ok()?;
        let mmap = unsafe { Mmap::map(&file).ok()? };
        if mmap.len() < std::mem::size_of::<RegimeWeights>() {
            return None;
        }
        Some(mmap)
    }

    /// Read the sequence number atomically.
    #[inline]
    fn read_sequence(mmap: &Mmap) -> u64 {
        // Sequence is at offset: magic(8) + version(4) + _pad0(4) = 16
        let ptr = unsafe { mmap.as_ptr().add(16) as *const AtomicU64 };
        let atomic = unsafe { &*ptr };
        atomic.load(Ordering::Acquire)
    }

    /// Try to read a consistent `RegimeWeights` from shared memory.
    ///
    /// Returns `Some(weights)` if a consistent read was obtained.
    /// Returns `None` if the writer is busy or the file is unavailable.
    ///
    /// BUG 14 FIX: Takes `&self` with interior mutability instead of `&mut self`.
    pub fn try_read(&self) -> Option<RegimeWeights> {
        // Safety: single-threaded access (strategy evaluator thread only)
        let mmap_ref = unsafe { &mut *self.mmap.get() };
        let cached_ref = unsafe { &mut *self.cached.get() };

        // Try to open the file if not yet mapped
        if mmap_ref.is_none() {
            *mmap_ref = Self::try_open(&self.path);
        }

        let mmap = mmap_ref.as_ref()?;
        let size = std::mem::size_of::<RegimeWeights>();

        if mmap.len() < size {
            return None;
        }

        for _ in 0..MAX_READ_RETRIES {
            let seq1 = Self::read_sequence(mmap);

            // Odd sequence means writer is active — spin
            if seq1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }

            // Read the full struct
            let weights = unsafe {
                std::ptr::read_unaligned(mmap.as_ptr() as *const RegimeWeights)
            };

            std::sync::atomic::fence(Ordering::Acquire);
            let seq2 = Self::read_sequence(mmap);

            if seq1 == seq2 {
                // Validate magic
                if weights.magic != REGIME_MAGIC {
                    return None;
                }
                *cached_ref = weights;
                return Some(weights);
            }

            std::hint::spin_loop();
        }

        None // Writer was busy for too long
    }

    /// Get the current regime weights.
    ///
    /// Attempts a fresh read; falls back to the last cached value.
    /// If the cached value is expired, returns safe defaults.
    ///
    /// BUG 14 FIX: Takes `&self` with interior mutability instead of `&mut self`.
    pub fn get_current(&self) -> RegimeWeights {
        if let Some(weights) = self.try_read() {
            if !weights.is_expired() {
                return weights;
            }
        }

        // Safety: single-threaded access
        let cached = unsafe { &*self.cached.get() };

        // Fall back to cached value if not expired
        if !cached.is_expired() {
            return *cached;
        }

        // Everything expired — return safe default
        RegimeWeights::safe_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_regime_weights_size() {
        assert_eq!(mem::size_of::<RegimeWeights>(), 112);
    }

    #[test]
    fn test_safe_defaults() {
        let w = RegimeWeights::safe_default();
        assert_eq!(w.overall_regime, regime_type::UNKNOWN);
        assert_eq!(w.volatility_regime, volatility_type::HIGH);
        // BUG #10 FIX: Changed from 0.5 to 1.0 (full sizing when no regime data)
        assert!((w.position_scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_strategy_blocked() {
        let mut w = RegimeWeights::default();
        w.blocked_strategies_mask = 0b101; // strategies 0 and 2 blocked
        assert!(w.is_strategy_blocked(0));
        assert!(!w.is_strategy_blocked(1));
        assert!(w.is_strategy_blocked(2));
    }

    #[test]
    fn test_leverage_cap() {
        let mut w = RegimeWeights::default();
        w.max_leverage_override = 10;
        assert_eq!(w.effective_leverage_cap(20), 10);
        assert_eq!(w.effective_leverage_cap(5), 5);

        w.max_leverage_override = 0;
        assert_eq!(w.effective_leverage_cap(20), 20);
    }

    #[test]
    fn test_reader_missing_file() {
        let mut reader = SharedMemRegimeReader::new("/tmp/nonexistent_regime_test");
        let result = reader.try_read();
        assert!(result.is_none());

        // get_current should return safe default (now takes &self)
        let w = reader.get_current();
        assert_eq!(w.overall_regime, regime_type::UNKNOWN);
    }

    #[test]
    fn test_write_and_read_regime() {
        let path = "/tmp/test_regime_shm";
        let _ = std::fs::remove_file(path);

        // Write a regime weights struct to a file
        let weights = RegimeWeights {
            magic: REGIME_MAGIC,
            version: REGIME_VERSION,
            _pad0: 0,
            sequence: 2, // even = consistent
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64,
            overall_regime: regime_type::TRENDING_BULLISH,
            volatility_regime: volatility_type::MODERATE,
            _pad1: [0; 2],
            sentiment_score_fp: 5000,
            sentiment_confidence_fp: 8000,
            fear_greed_index: 70,
            btc_dominance_trend: 1,
            funding_rate_bias: 0,
            _pad2: [0; 2],
            cross_asset_correlation_fp: 3000,
            news_impact_score_fp: 2000,
            position_scale_fp: 10000, // 1.0
            max_leverage_override: 10,
            ttl_seconds: 600,
            allowed_strategies_mask: u64::MAX,
            blocked_strategies_mask: 0,
            _reserved: [0; 24],
        };

        // Write to file
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &weights as *const RegimeWeights as *const u8,
                mem::size_of::<RegimeWeights>(),
            )
        };
        std::fs::write(path, bytes).unwrap();

        // Read it back via the reader
        let mut reader = SharedMemRegimeReader::new(path);
        let read_weights = reader.try_read().unwrap();
        assert_eq!(read_weights.overall_regime, regime_type::TRENDING_BULLISH);
        assert_eq!(read_weights.volatility_regime, volatility_type::MODERATE);
        let sentiment = { read_weights.sentiment_score_fp };
        assert_eq!(sentiment, 5000);
        assert!((read_weights.position_scale() - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }
}

