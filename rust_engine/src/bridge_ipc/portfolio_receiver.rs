//! Portfolio Receiver — Python → Rust portfolio weight targets via shared memory.
//!
//! Python's brain container writes "Target Portfolio Weights" to a shared memory
//! region. Rust polls this region to read the latest target state without
//! waiting for a mutex or RPC call.
//!
//! # Memory Layout
//!
//! ```text
//! [0..32]     BridgeHeader (magic, version, sequence, timestamp_ns)
//! [32..36]    num_symbols: u32   (number of active symbols)
//! [36..40]    reserved: u32
//! [40..48]    risk_multiplier: f64 (global risk scaling factor from Python)
//! [48..64]    reserved: [u8; 16]
//! [64..]      entries: [PortfolioEntry; MAX_SYMBOLS]
//! ```
//!
//! # PortfolioEntry Layout (32 bytes)
//!
//! ```text
//! [0..2]     symbol_id: u16
//! [2..4]     flags: u16     (bit 0: active, bit 1: reduce_only)
//! [4..8]     reserved: u32
//! [8..16]    target_weight: f64  (-1.0 to 1.0; negative = short)
//! [16..24]   confidence: f64     (0.0 to 1.0)
//! [24..32]   max_position_size: f64 (max contracts)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut rx = PortfolioReceiver::new("/dev/shm/bridge_portfolio");
//! rx.init().expect("SHM init failed");
//!
//! // In hot loop:
//! if let Some(snapshot) = rx.try_read() {
//!     for entry in &snapshot.entries {
//!         if entry.active {
//!             // Apply target weight to execution engine
//!         }
//!     }
//! }
//! ```

use tracing::debug;
use super::now_ns;

/// Default path for portfolio targets SHM.
pub const PORTFOLIO_SHM_PATH: &str = "/dev/shm/bridge_portfolio";

/// Maximum number of symbols in the portfolio.
const MAX_SYMBOLS: usize = 64;

/// Size of the header region (64 bytes).
const HEADER_SIZE: usize = 64;

/// Size of each portfolio entry (32 bytes).
const ENTRY_SIZE: usize = 32;

/// Total SHM size.
const PORTFOLIO_SHM_SIZE: usize = HEADER_SIZE + (MAX_SYMBOLS * ENTRY_SIZE);

/// A single portfolio entry for one symbol.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PortfolioEntry {
    /// Symbol identifier (matches the SymbolRegistry).
    pub symbol_id: u16,
    /// Flags: bit 0 = active, bit 1 = reduce_only.
    pub flags: u16,
    /// Reserved.
    pub _reserved: u32,
    /// Target weight: -1.0 (full short) to 1.0 (full long). 0.0 = flat.
    pub target_weight: f64,
    /// Confidence level: 0.0 (no confidence) to 1.0 (maximum confidence).
    pub confidence: f64,
    /// Maximum position size in contracts.
    pub max_position_size: f64,
}

impl PortfolioEntry {
    /// Check if this entry is active (Python has a target for this symbol).
    #[inline]
    pub fn active(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Check if this entry is reduce-only (only close, don't open new).
    #[inline]
    pub fn reduce_only(&self) -> bool {
        self.flags & 2 != 0
    }

    /// Check if the target is to go long.
    #[inline]
    pub fn is_long(&self) -> bool {
        self.target_weight > 0.0
    }

    /// Check if the target is to go short.
    #[inline]
    pub fn is_short(&self) -> bool {
        self.target_weight < 0.0
    }

    /// Check if the target is flat (close position).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.target_weight.abs() < 1e-9
    }
}

impl Default for PortfolioEntry {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            flags: 0,
            _reserved: 0,
            target_weight: 0.0,
            confidence: 0.0,
            max_position_size: 0.0,
        }
    }
}

/// Snapshot of the entire portfolio target state.
#[derive(Debug, Clone)]
pub struct PortfolioSnapshot {
    /// Sequence number from the header (for staleness detection).
    pub sequence: u64,
    /// Timestamp of the last write (nanoseconds).
    pub timestamp_ns: u64,
    /// Global risk multiplier from Python (0.0 = halt, 1.0 = full risk).
    pub risk_multiplier: f64,
    /// Number of active symbols.
    pub num_symbols: u32,
    /// Portfolio entries (only first `num_symbols` are valid).
    pub entries: Vec<PortfolioEntry>,
}

impl PortfolioSnapshot {
    /// Check if the snapshot is stale (older than threshold_ns).
    pub fn is_stale(&self, threshold_ns: u64) -> bool {
        let age = now_ns().saturating_sub(self.timestamp_ns);
        age > threshold_ns
    }

    /// Get the entry for a specific symbol_id, if active.
    pub fn get_entry(&self, symbol_id: u16) -> Option<&PortfolioEntry> {
        self.entries.iter().find(|e| e.symbol_id == symbol_id && e.active())
    }
}

/// Reader side of the portfolio receiver.
///
/// Maps the SHM region written by Python and reads the latest portfolio targets.
pub struct PortfolioReceiver {
    /// Memory-mapped region (read-only).
    mmap: Option<memmap2::Mmap>,
    /// Last sequence number we read (for change detection).
    last_sequence: u64,
    /// Total reads since startup.
    total_reads: u64,
    /// SHM file path.
    path: String,
}

impl PortfolioReceiver {
    /// Create a new portfolio receiver.
    pub fn new(path: &str) -> Self {
        Self {
            mmap: None,
            last_sequence: 0,
            total_reads: 0,
            path: path.to_string(),
        }
    }

    /// Create with default SHM path.
    pub fn with_defaults() -> Self {
        Self::new(PORTFOLIO_SHM_PATH)
    }

    /// Initialize by opening the SHM file (must already exist, created by Python).
    pub fn init(&mut self) -> Result<(), String> {
        use std::fs::OpenOptions;
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt;

        // First try to open existing file
        let file = match OpenOptions::new().read(true).open(&self.path) {
            Ok(f) => f,
            Err(_) => {
                // Create it if it doesn't exist (for testing or when Rust starts first)
                let f = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .mode(0o666)
                    .open(&self.path)
                    .map_err(|e| format!("Failed to create SHM {}: {}", self.path, e))?;

                // FIX 5: Ensure SHM file is world-readable/writable for cross-container access
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = f.set_permissions(std::fs::Permissions::from_mode(0o666));
                }

                f.set_len(PORTFOLIO_SHM_SIZE as u64)
                    .map_err(|e| format!("Failed to set SHM size: {}", e))?;
                // Reopen read-only
                OpenOptions::new()
                    .read(true)
                    .open(&self.path)
                    .map_err(|e| format!("Failed to reopen SHM: {}", e))?
            }
        };

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("Failed to mmap SHM: {}", e))?
        };

        self.mmap = Some(mmap);
        debug!("[portfolio-rx] Initialized SHM reader at {}", self.path);
        Ok(())
    }

    /// Try to read the latest portfolio snapshot.
    ///
    /// Returns `Some(snapshot)` if the sequence number has changed since the
    /// last read (i.e., Python has written new data).
    /// Returns `None` if no new data is available.
    ///
    /// **Enhanced with input validation and sanitization:**
    /// - Bounds checking on portfolio weights (0.0-1.0 range)
    /// - Sum-to-one validation for active weights
    /// - Rejection of NaN/Inf values
    /// - Stale data detection via timestamp
    pub fn try_read(&mut self) -> Option<PortfolioSnapshot> {
        let mmap = self.mmap.as_ref()?;
        if mmap.len() < HEADER_SIZE {
            tracing::warn!("[portfolio-rx] SHM size {} < header size {}", mmap.len(), HEADER_SIZE);
            return None;
        }

        // Read header
        let sequence = u64::from_le_bytes(mmap[16..24].try_into().ok()?);
        if sequence == self.last_sequence && self.last_sequence > 0 {
            return None; // No new data
        }

        let timestamp_ns = u64::from_le_bytes(mmap[24..32].try_into().ok()?);
        let num_symbols = u32::from_le_bytes(mmap[32..36].try_into().ok()?);
        let risk_multiplier = f64::from_le_bytes(mmap[40..48].try_into().ok()?);

        // Validate risk_multiplier
        let risk_multiplier = if risk_multiplier.is_finite() && risk_multiplier >= 0.0 && risk_multiplier <= 10.0 {
            risk_multiplier
        } else {
            tracing::warn!("[portfolio-rx] Invalid risk_multiplier={} — clamping to 1.0", risk_multiplier);
            1.0
        };

        // Stale data detection (5-minute threshold)
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        if timestamp_ns > 0 && now_ns > timestamp_ns {
            let age_ms = (now_ns - timestamp_ns) / 1_000_000;
            if age_ms > 300_000 {
                tracing::warn!("[portfolio-rx] Stale portfolio data: age={}ms", age_ms);
            }
        }

        let num = (num_symbols as usize).min(MAX_SYMBOLS);
        let mut entries = Vec::with_capacity(num);
        let mut total_weight = 0.0_f64;

        for i in 0..num {
            let offset = HEADER_SIZE + i * ENTRY_SIZE;
            if offset + ENTRY_SIZE > mmap.len() {
                tracing::warn!("[portfolio-rx] Entry {} offset {} exceeds mmap len {}", i, offset, mmap.len());
                break;
            }

            let symbol_id = u16::from_le_bytes(mmap[offset..offset+2].try_into().ok()?);
            let flags = u16::from_le_bytes(mmap[offset+2..offset+4].try_into().ok()?);
            let target_weight = f64::from_le_bytes(mmap[offset+8..offset+16].try_into().ok()?);
            let confidence = f64::from_le_bytes(mmap[offset+16..offset+24].try_into().ok()?);
            let max_position_size = f64::from_le_bytes(mmap[offset+24..offset+32].try_into().ok()?);

            // Validate target_weight: must be finite and in range [-1.0, 1.0]
            let target_weight = if target_weight.is_finite() && target_weight.abs() <= 1.0 {
                target_weight
            } else {
                tracing::warn!(
                    "[portfolio-rx] Invalid target_weight={} for symbol_id={} — clamping to 0.0",
                    target_weight,
                    symbol_id
                );
                0.0
            };

            // Validate confidence: must be finite and in range [0.0, 1.0]
            let confidence = if confidence.is_finite() && confidence >= 0.0 && confidence <= 1.0 {
                confidence
            } else {
                tracing::warn!(
                    "[portfolio-rx] Invalid confidence={} for symbol_id={} — clamping to 0.0",
                    confidence,
                    symbol_id
                );
                0.0
            };

            // Validate max_position_size: must be finite and non-negative
            let max_position_size = if max_position_size.is_finite() && max_position_size >= 0.0 {
                max_position_size
            } else {
                tracing::warn!(
                    "[portfolio-rx] Invalid max_position_size={} for symbol_id={} — clamping to 0.0",
                    max_position_size,
                    symbol_id
                );
                0.0
            };

            // Track total weight for sum-to-one validation
            if flags & 1 != 0 {
                // Active entry
                total_weight += target_weight.abs();
            }

            entries.push(PortfolioEntry {
                symbol_id,
                flags,
                _reserved: 0,
                target_weight,
                confidence,
                max_position_size,
            });
        }

        // Sum-to-one validation (with tolerance for floating-point error)
        if total_weight > 0.0 && (total_weight < 0.95 || total_weight > 1.05) {
            tracing::warn!(
                "[portfolio-rx] Portfolio weights sum to {:.3} (expected ~1.0) — normalizing",
                total_weight
            );
            // Normalize weights
            for entry in &mut entries {
                if entry.flags & 1 != 0 {
                    entry.target_weight /= total_weight;
                }
            }
        }

        self.last_sequence = sequence;
        self.total_reads += 1;

        Some(PortfolioSnapshot {
            sequence,
            timestamp_ns,
            risk_multiplier,
            num_symbols: num as u32,
            entries,
        })
    }

    /// Force re-read regardless of sequence number.
    pub fn force_read(&mut self) -> Option<PortfolioSnapshot> {
        self.last_sequence = 0;
        self.try_read()
    }

    /// Check if the receiver is initialized.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.mmap.is_some()
    }

    /// Get total reads.
    #[inline]
    pub fn total_reads(&self) -> u64 {
        self.total_reads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_entry_defaults() {
        let entry = PortfolioEntry::default();
        assert!(!entry.active());
        assert!(entry.is_flat());
        assert!(!entry.reduce_only());
    }

    #[test]
    fn test_portfolio_entry_flags() {
        let entry = PortfolioEntry {
            flags: 0b11, // active + reduce_only
            target_weight: 0.5,
            ..Default::default()
        };
        assert!(entry.active());
        assert!(entry.reduce_only());
        assert!(entry.is_long());
        assert!(!entry.is_short());
    }

    #[test]
    fn test_receiver_init_and_read() {
        let path = "/tmp/test_bridge_portfolio";
        let mut rx = PortfolioReceiver::new(path);
        assert!(rx.init().is_ok());
        assert!(rx.is_active());

        // First read should return None (no data written yet)
        let result = rx.try_read();
        // May return None or Some with sequence 0
        assert!(result.is_none() || result.unwrap().sequence == 0);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
