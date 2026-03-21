//! Lock-Free SHM Signal Queue — Alpha Oracle Architecture.
//!
//! Implements a single-producer (Python) / single-consumer (Rust) ring buffer
//! in `/dev/shm/alpha_signal_queue` for zero-copy, zero-lock trade intent
//! communication between the Python Alpha Oracle and the Rust Execution Engine.
//!
//! # Memory Layout (mmap)
//!
//! ```text
//! Offset  Size   Field            Description
//! ──────  ────   ─────            ───────────
//! 0       8      magic            0x414C504841_5349474E ("ALPHASIGN")
//! 8       4      version          Protocol version (1)
//! 12      4      capacity         Number of slots in the ring
//! 16      8      write_cursor     Producer write position (atomic u64)
//! 24      8      read_cursor      Consumer read position (atomic u64)
//! 32      N*256  slots[N]         Ring buffer entries (TradeIntent)
//! ```
//!
//! # TradeIntent Layout (256 bytes per slot)
//!
//! ```text
//! Offset  Size  Field               Type/Notes
//! ──────  ────  ─────               ──────────
//! 0       32    symbol              UTF-8, zero-padded (e.g. "BTC_USDT")
//! 32      1     side                0=buy/long, 1=sell/short
//! 33      1     intent_type         0=open, 1=close, 2=reduce
//! 34      2     _pad1               alignment
//! 36      4     leverage            i32
//! 40      8     size_contracts      i64 (integer contracts for Gate.io)
//! 48      8     entry_price_fp      i64 (price × 1e8)
//! 56      8     stop_loss_fp        i64 (price × 1e8, 0 = none)
//! 64      8     take_profit_fp      i64 (price × 1e8, 0 = none)
//! 72      8     confidence_fp       i64 (confidence × 1e8, range [0, 1e8])
//! 80      8     risk_reward_fp      i64 (R:R ratio × 1e4)
//! 88      8     timestamp_ns        u64 (monotonic nanoseconds)
//! 96      4     confluence_count    u32 (number of strategies that agree)
//! 100     4     total_strategies    u32 (total strategies evaluated)
//! 104     64    signal_tag          UTF-8, zero-padded (strategy confluence ID)
//! 168     8     max_slippage_fp     i64 (max slippage × 1e8)
//! 176     80    _reserved           future use
//! ```
//!
//! # Guarantees
//!
//! - **Zero-copy**: Python writes directly to mmap, Rust reads directly.
//! - **Zero-lock**: Uses atomic cursors with Release/Acquire ordering.
//! - **Zero-latency**: No syscalls, no IPC overhead, just memory reads.
//! - **Bounded**: Fixed-size ring prevents unbounded memory growth.
//! - **Torn-read safe**: Each slot is written atomically (Python writes
//!   payload first, then advances write_cursor with Release).

use std::sync::atomic::Ordering;
use std::path::Path;
use tracing::info;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

pub const SIGNAL_QUEUE_PATH: &str = "/dev/shm/alpha_signal_queue";
pub const SIGNAL_QUEUE_MAGIC: u64 = 0x414C_5048_4153_4947; // "ALPHASIG"
pub const SIGNAL_QUEUE_VERSION: u32 = 1;
pub const SIGNAL_QUEUE_CAPACITY: u32 = 256; // Power of 2 for fast modulo
pub const HEADER_SIZE: usize = 32;
pub const SLOT_SIZE: usize = 256;
pub const TOTAL_SHM_SIZE: usize = HEADER_SIZE + (SIGNAL_QUEUE_CAPACITY as usize * SLOT_SIZE);

/// Fixed-point precision for prices and confidence.
pub const FP_PRECISION: f64 = 1e8;
/// Fixed-point precision for risk/reward ratio.
pub const RR_PRECISION: f64 = 1e4;

// ═══════════════════════════════════════════════════════════════════════════
// TradeIntent — parsed from SHM slot
// ═══════════════════════════════════════════════════════════════════════════

/// A trade intent signal from the Python Alpha Oracle.
#[derive(Debug, Clone)]
pub struct TradeIntent {
    pub symbol: String,
    pub side: u8,         // 0=long, 1=short
    pub intent_type: u8,  // 0=open, 1=close, 2=reduce
    pub leverage: i32,
    pub size_contracts: i64,
    pub entry_price: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub confidence: f64,
    pub risk_reward: f64,
    pub timestamp_ns: u64,
    pub confluence_count: u32,
    pub total_strategies: u32,
    pub signal_tag: String,
    pub max_slippage: f64,
}

impl TradeIntent {
    /// Parse a TradeIntent from a 256-byte slot buffer.
    pub fn from_slot(slot: &[u8; SLOT_SIZE]) -> Option<Self> {
        if slot.iter().all(|&b| b == 0) {
            return None; // Empty slot
        }

        // Parse symbol (bytes 0..32, UTF-8 zero-padded)
        let symbol_end = slot[0..32].iter().position(|&b| b == 0).unwrap_or(32);
        let symbol = std::str::from_utf8(&slot[0..symbol_end])
            .ok()?
            .to_string();

        if symbol.is_empty() {
            return None;
        }

        let side = slot[32];
        let intent_type = slot[33];
        let leverage = i32::from_le_bytes([slot[36], slot[37], slot[38], slot[39]]);
        let size_contracts = i64::from_le_bytes(slot[40..48].try_into().ok()?);
        let entry_price_fp = i64::from_le_bytes(slot[48..56].try_into().ok()?);
        let stop_loss_fp = i64::from_le_bytes(slot[56..64].try_into().ok()?);
        let take_profit_fp = i64::from_le_bytes(slot[64..72].try_into().ok()?);
        let confidence_fp = i64::from_le_bytes(slot[72..80].try_into().ok()?);
        let risk_reward_fp = i64::from_le_bytes(slot[80..88].try_into().ok()?);
        let timestamp_ns = u64::from_le_bytes(slot[88..96].try_into().ok()?);
        let confluence_count = u32::from_le_bytes(slot[96..100].try_into().ok()?);
        let total_strategies = u32::from_le_bytes(slot[100..104].try_into().ok()?);

        // Parse signal_tag (bytes 104..168, UTF-8 zero-padded)
        let tag_end = slot[104..168]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(64);
        let signal_tag = std::str::from_utf8(&slot[104..104 + tag_end])
            .unwrap_or("unknown")
            .to_string();

        let max_slippage_fp = i64::from_le_bytes(slot[168..176].try_into().ok()?);

        Some(TradeIntent {
            symbol,
            side,
            intent_type,
            leverage,
            size_contracts,
            entry_price: entry_price_fp as f64 / FP_PRECISION,
            stop_loss: if stop_loss_fp != 0 {
                Some(stop_loss_fp as f64 / FP_PRECISION)
            } else {
                None
            },
            take_profit: if take_profit_fp != 0 {
                Some(take_profit_fp as f64 / FP_PRECISION)
            } else {
                None
            },
            confidence: confidence_fp as f64 / FP_PRECISION,
            risk_reward: risk_reward_fp as f64 / RR_PRECISION,
            timestamp_ns,
            confluence_count,
            total_strategies,
            signal_tag,
            max_slippage: max_slippage_fp as f64 / FP_PRECISION,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Signal Queue Consumer (Rust side)
// ═══════════════════════════════════════════════════════════════════════════

/// Reads TradeIntent signals from the SHM ring buffer written by Python.
pub struct SignalQueueConsumer {
    /// Memory-mapped shared memory region.
    mmap: memmap2::MmapMut,
    /// Cached read cursor (local copy — the authoritative one is in SHM).
    local_read_cursor: u64,
}

impl SignalQueueConsumer {
    /// Open (or create) the signal queue SHM file and return a consumer.
    pub fn open() -> std::io::Result<Self> {
        let path = Path::new(SIGNAL_QUEUE_PATH);

        // Create the SHM file if it doesn't exist (Rust can also initialize it)
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        // BUG 5 FIX: Ensure SHM file is world-readable/writable so both
        // the Rust engine and Python cold-path can access it regardless
        // of which container creates it first.
        #[cfg(unix)]
        {
            let perms = std::fs::Permissions::from_mode(0o666);
            let _ = file.set_permissions(perms);
        }

        let metadata = file.metadata()?;
        if metadata.len() < TOTAL_SHM_SIZE as u64 {
            file.set_len(TOTAL_SHM_SIZE as u64)?;
        }

        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        // Validate or initialize header
        let magic = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if magic != SIGNAL_QUEUE_MAGIC {
            // Initialize the header
            let mut mmap = mmap;
            mmap[0..8].copy_from_slice(&SIGNAL_QUEUE_MAGIC.to_le_bytes());
            mmap[8..12].copy_from_slice(&SIGNAL_QUEUE_VERSION.to_le_bytes());
            mmap[12..16].copy_from_slice(&SIGNAL_QUEUE_CAPACITY.to_le_bytes());
            // write_cursor and read_cursor start at 0 (already zeroed)
            info!(
                "[signal_queue] Initialized SHM at {} ({} bytes, {} slots)",
                SIGNAL_QUEUE_PATH, TOTAL_SHM_SIZE, SIGNAL_QUEUE_CAPACITY
            );

            // Read the current read_cursor from SHM
            let read_cursor = u64::from_le_bytes(mmap[24..32].try_into().unwrap());

            Ok(Self {
                mmap,
                local_read_cursor: read_cursor,
            })
        } else {
            // Read the current read_cursor from SHM
            let read_cursor = u64::from_le_bytes(mmap[24..32].try_into().unwrap());
            info!(
                "[signal_queue] Opened existing SHM at {} (read_cursor={})",
                SIGNAL_QUEUE_PATH, read_cursor
            );

            Ok(Self {
                mmap,
                local_read_cursor: read_cursor,
            })
        }
    }

    /// Try to read the next TradeIntent from the queue (non-blocking).
    ///
    /// Returns `None` if the queue is empty (read_cursor == write_cursor).
    /// This is designed to be called in a tight polling loop with near-zero
    /// overhead when the queue is empty (just two atomic loads).
    pub fn try_pop(&mut self) -> Option<TradeIntent> {
        // Read write_cursor with Acquire ordering to see Python's latest write.
        // We read from the mmap directly using from_le_bytes (safe on LE archs).
        let write_cursor = {
            let bytes: [u8; 8] = self.mmap[16..24].try_into().unwrap();
            u64::from_le_bytes(bytes)
        };
        // Acquire fence: ensures we see the slot data AFTER write_cursor was updated
        std::sync::atomic::fence(Ordering::Acquire);

        if self.local_read_cursor >= write_cursor {
            return None; // Queue empty
        }

        // Calculate slot offset
        let slot_idx = (self.local_read_cursor % SIGNAL_QUEUE_CAPACITY as u64) as usize;
        let slot_offset = HEADER_SIZE + slot_idx * SLOT_SIZE;

        // Read the slot data
        let mut slot_buf = [0u8; SLOT_SIZE];
        slot_buf.copy_from_slice(&self.mmap[slot_offset..slot_offset + SLOT_SIZE]);

        // Parse the TradeIntent
        let intent = TradeIntent::from_slot(&slot_buf);

        // Advance the read cursor (with Release ordering for Python to see)
        self.local_read_cursor += 1;
        let cursor_bytes = self.local_read_cursor.to_le_bytes();
        self.mmap[24..32].copy_from_slice(&cursor_bytes);
        // Release fence: ensures read_cursor update is visible
        std::sync::atomic::fence(Ordering::Release);

        intent
    }

    /// Drain all available signals from the queue.
    pub fn drain(&mut self) -> Vec<TradeIntent> {
        let mut signals = Vec::new();
        while let Some(intent) = self.try_pop() {
            signals.push(intent);
        }
        signals
    }

    /// Return the number of pending (unread) signals.
    pub fn pending_count(&self) -> u64 {
        let write_cursor = u64::from_le_bytes(
            self.mmap[16..24].try_into().unwrap(),
        );
        write_cursor.saturating_sub(self.local_read_cursor)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_intent_parse_empty_slot() {
        let slot = [0u8; SLOT_SIZE];
        assert!(TradeIntent::from_slot(&slot).is_none());
    }

    #[test]
    fn test_trade_intent_parse() {
        let mut slot = [0u8; SLOT_SIZE];

        // Write symbol "BTC_USDT"
        let sym = b"BTC_USDT";
        slot[0..sym.len()].copy_from_slice(sym);

        // side = 0 (long)
        slot[32] = 0;
        // intent_type = 0 (open)
        slot[33] = 0;
        // leverage = 10
        slot[36..40].copy_from_slice(&10i32.to_le_bytes());
        // size_contracts = 5
        slot[40..48].copy_from_slice(&5i64.to_le_bytes());
        // entry_price = 50000.0 (× 1e8)
        slot[48..56].copy_from_slice(&(5_000_000_000_000i64).to_le_bytes());
        // stop_loss = 49000.0 (× 1e8)
        slot[56..64].copy_from_slice(&(4_900_000_000_000i64).to_le_bytes());
        // confidence = 0.85 (× 1e8)
        slot[72..80].copy_from_slice(&(85_000_000i64).to_le_bytes());
        // risk_reward = 2.5 (× 1e4)
        slot[80..88].copy_from_slice(&(25_000i64).to_le_bytes());
        // timestamp_ns
        slot[88..96].copy_from_slice(&1000u64.to_le_bytes());
        // confluence_count = 45
        slot[96..100].copy_from_slice(&45u32.to_le_bytes());
        // total_strategies = 60
        slot[100..104].copy_from_slice(&60u32.to_le_bytes());

        let intent = TradeIntent::from_slot(&slot).expect("Should parse");
        assert_eq!(intent.symbol, "BTC_USDT");
        assert_eq!(intent.side, 0);
        assert_eq!(intent.leverage, 10);
        assert_eq!(intent.size_contracts, 5);
        assert!((intent.entry_price - 50000.0).abs() < 0.01);
        assert!((intent.confidence - 0.85).abs() < 0.01);
        assert_eq!(intent.confluence_count, 45);
        assert_eq!(intent.total_strategies, 60);
    }
}
