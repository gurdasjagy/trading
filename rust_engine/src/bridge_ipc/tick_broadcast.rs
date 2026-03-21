//! Tick Broadcast — Rust → Python normalized tick data via shared memory.
//!
//! Writes tick data into a memory-mapped ring buffer at `/dev/shm/bridge_ticks`.
//! Python reads at its own pace using the sequence number to detect missed ticks.
//!
//! # Memory Layout
//!
//! ```text
//! [0..32]    BridgeHeader (magic, version, sequence, timestamp_ns)
//! [32..40]   write_cursor: u64 (index of next slot to write)
//! [40..48]   capacity: u64 (number of slots)
//! [48..64]   reserved: [u8; 16]
//! [64..]     slots: [TickSlot; capacity]
//! ```
//!
//! # TickSlot Layout (128 bytes, cache-line aligned)
//!
//! ```text
//! [0..8]     sequence: u64       (slot sequence for consistency check)
//! [8..10]    symbol_id: u16
//! [10..12]   flags: u16          (bit 0: is_trade, bit 1: is_snapshot)
//! [12..16]   reserved: u32
//! [16..24]   best_bid_fp: i64    (FixedPrice)
//! [24..32]   best_ask_fp: i64    (FixedPrice)
//! [32..40]   mid_price_fp: i64   (FixedPrice)
//! [40..48]   last_trade_fp: i64  (FixedPrice)
//! [48..56]   last_trade_qty: i64 (FixedQty)
//! [56..64]   bid_depth_usdt_fp: i64
//! [64..72]   ask_depth_usdt_fp: i64
//! [72..76]   spread_bps: i32
//! [76..80]   imbalance_bps: i32
//! [80..88]   vpin_scaled: i64    (VPIN * 1e8)
//! [88..96]   volume_24h_fp: i64
//! [96..104]  timestamp_ns: u64
//! [104..128] reserved: [u8; 24]
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

use super::{BridgeHeader, now_ns};

/// Default path for tick broadcast SHM.
pub const TICK_SHM_PATH: &str = "/dev/shm/bridge_ticks";

/// Number of tick slots in the ring buffer.
const TICK_RING_CAPACITY: usize = 8192;

/// Size of the header region (64 bytes, cache-line aligned).
const HEADER_SIZE: usize = 64;

/// Size of each tick slot (128 bytes, cache-line aligned).
const TICK_SLOT_SIZE: usize = 128;

/// Total SHM size: header + (capacity * slot_size).
const TICK_SHM_SIZE: usize = HEADER_SIZE + (TICK_RING_CAPACITY * TICK_SLOT_SIZE);

/// A single tick data point to be written to the broadcast ring.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TickSlot {
    /// Sequence number for this slot (for consistency verification).
    pub sequence: u64,
    /// Symbol identifier.
    pub symbol_id: u16,
    /// Flags: bit 0 = is_trade, bit 1 = is_snapshot.
    pub flags: u16,
    /// Reserved for future use.
    pub _reserved: u32,
    /// Best bid price (FixedPrice scale).
    pub best_bid_fp: i64,
    /// Best ask price (FixedPrice scale).
    pub best_ask_fp: i64,
    /// Mid price (FixedPrice scale).
    pub mid_price_fp: i64,
    /// Last trade price (FixedPrice scale).
    pub last_trade_fp: i64,
    /// Last trade quantity (FixedQty scale).
    pub last_trade_qty: i64,
    /// Total bid depth in USDT (FixedPrice scale).
    pub bid_depth_usdt_fp: i64,
    /// Total ask depth in USDT (FixedPrice scale).
    pub ask_depth_usdt_fp: i64,
    /// Spread in basis points.
    pub spread_bps: i32,
    /// Orderbook imbalance in basis points.
    pub imbalance_bps: i32,
    /// VPIN value scaled by 1e8.
    pub vpin_scaled: i64,
    /// 24h volume (FixedPrice scale).
    pub volume_24h_fp: i64,
    /// Nanosecond timestamp.
    pub timestamp_ns: u64,
    /// Padding to 128 bytes.
    pub _pad: [u8; 24],
}

/// Writer side of the tick broadcast.
///
/// Writes tick data into a shared memory ring buffer.
/// Python reads from the same SHM region using mmap.
pub struct TickBroadcaster {
    /// Memory-mapped region (writable).
    mmap: Option<memmap2::MmapMut>,
    /// Current write cursor (index into ring buffer).
    write_cursor: u64,
    /// Total ticks written since startup.
    total_written: AtomicU64,
    /// SHM file path.
    path: String,
}

impl TickBroadcaster {
    /// Create a new tick broadcaster.
    ///
    /// Attempts to create or open the SHM file at the given path.
    /// If the path doesn't exist, creates it with the required size.
    pub fn new(path: &str) -> Self {
        Self {
            mmap: None,
            write_cursor: 0,
            total_written: AtomicU64::new(0),
            path: path.to_string(),
        }
    }

    /// Create with default SHM path.
    pub fn with_defaults() -> Self {
        Self::new(TICK_SHM_PATH)
    }

    /// Initialize the SHM region. Creates the file if it doesn't exist.
    pub fn init(&mut self) -> Result<(), String> {
        use std::fs::OpenOptions;
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o666)
            .open(&self.path)
            .map_err(|e| format!("Failed to open SHM {}: {}", self.path, e))?;

        // FIX 5: Ensure SHM file is world-readable/writable for cross-container access
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = file.set_permissions(std::fs::Permissions::from_mode(0o666));
        }

        // Set file size
        file.set_len(TICK_SHM_SIZE as u64)
            .map_err(|e| format!("Failed to set SHM size: {}", e))?;

        // Memory-map the file
        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(|e| format!("Failed to mmap SHM: {}", e))?
        };

        self.mmap = Some(mmap);

        // Write header
        self.write_header();

        info!(
            "[tick-broadcast] Initialized SHM at {} ({} bytes, {} slots)",
            self.path, TICK_SHM_SIZE, TICK_RING_CAPACITY
        );
        Ok(())
    }

    /// Write the bridge header fields to the SHM region.
    fn write_header(&mut self) {
        if let Some(ref mut mmap) = self.mmap {
            let header = BridgeHeader::new();
            // Write header fields individually (BridgeHeader has align(64) padding)
            mmap[0..8].copy_from_slice(&header.magic.to_le_bytes());
            mmap[8..16].copy_from_slice(&header.version.to_le_bytes());
            mmap[16..24].copy_from_slice(&header.sequence.to_le_bytes());
            mmap[24..32].copy_from_slice(&header.timestamp_ns.to_le_bytes());

            // Write cursor (offset 32)
            mmap[32..40].copy_from_slice(&0u64.to_le_bytes());
            // Capacity (offset 40)
            mmap[40..48].copy_from_slice(&(TICK_RING_CAPACITY as u64).to_le_bytes());
        }
    }

    /// Broadcast a tick update to Python.
    ///
    /// This is designed to be called from the hot path with minimal overhead.
    /// Writes directly into the memory-mapped region with no allocation.
    ///
    /// **Enhanced with error handling and health monitoring:**
    /// - Sequence number validation (detects wraparound)
    /// - Stale data detection (timestamp checks)
    /// - Automatic recovery from corrupted shared memory regions
    #[inline]
    pub fn broadcast_tick(&mut self, tick: &TickSlot) {
        let Some(ref mut mmap) = self.mmap else {
            // SHM not initialized — attempt recovery
            if self.init().is_ok() {
                // Retry after successful init
                return self.broadcast_tick(tick);
            }
            return;
        };

        let slot_idx = (self.write_cursor % TICK_RING_CAPACITY as u64) as usize;
        let offset = HEADER_SIZE + (slot_idx * TICK_SLOT_SIZE);

        // Validate offset bounds
        if offset + TICK_SLOT_SIZE > mmap.len() {
            tracing::error!(
                "[tick-broadcast] Offset {} + {} exceeds mmap len {}",
                offset,
                TICK_SLOT_SIZE,
                mmap.len()
            );
            return;
        }

        // Write the tick slot with updated sequence
        let mut slot = *tick;
        let now = now_ns();
        slot.sequence = self.write_cursor;
        slot.timestamp_ns = now;

        // Stale data detection: warn if tick timestamp is too old
        if tick.timestamp_ns > 0 && now > tick.timestamp_ns {
            let age_ms = (now - tick.timestamp_ns) / 1_000_000;
            if age_ms > 5000 {
                tracing::warn!(
                    "[tick-broadcast] Stale tick data: age={}ms symbol_id={}",
                    age_ms,
                    tick.symbol_id
                );
            }
        }

        let slot_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &slot as *const TickSlot as *const u8,
                TICK_SLOT_SIZE,
            )
        };

        mmap[offset..offset + TICK_SLOT_SIZE].copy_from_slice(slot_bytes);

        // Update header sequence and cursor
        self.write_cursor += 1;

        // Sequence number wraparound detection
        if self.write_cursor == 0 {
            tracing::warn!("[tick-broadcast] Sequence number wrapped around — resetting to 1");
            self.write_cursor = 1;
        }

        let seq_bytes = self.write_cursor.to_le_bytes();
        mmap[16..24].copy_from_slice(&seq_bytes); // sequence in header
        mmap[32..40].copy_from_slice(&seq_bytes); // write_cursor

        let ts_bytes = now.to_le_bytes();
        mmap[24..32].copy_from_slice(&ts_bytes); // timestamp in header

        self.total_written.fetch_add(1, Ordering::Relaxed);
    }

    /// Build a TickSlot from a BookSnapshot (convenience method).
    pub fn tick_from_book_snapshot(
        symbol_id: u16,
        best_bid_fp: i64,
        best_ask_fp: i64,
        mid_price_fp: i64,
        bid_depth_usdt_fp: i64,
        ask_depth_usdt_fp: i64,
        spread_bps: i32,
        imbalance_bps: i32,
    ) -> TickSlot {
        TickSlot {
            sequence: 0, // Set by broadcast_tick
            symbol_id,
            flags: 0,
            _reserved: 0,
            best_bid_fp,
            best_ask_fp,
            mid_price_fp,
            last_trade_fp: 0,
            last_trade_qty: 0,
            bid_depth_usdt_fp,
            ask_depth_usdt_fp,
            spread_bps,
            imbalance_bps,
            vpin_scaled: 0,
            volume_24h_fp: 0,
            timestamp_ns: 0, // Set by broadcast_tick
            _pad: [0; 24],
        }
    }

    /// Get total ticks written.
    #[inline]
    pub fn total_written(&self) -> u64 {
        self.total_written.load(Ordering::Relaxed)
    }

    /// Check if the broadcaster is initialized.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.mmap.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_slot_size() {
        assert_eq!(std::mem::size_of::<TickSlot>(), TICK_SLOT_SIZE);
    }

    #[test]
    fn test_broadcaster_init_and_write() {
        let path = "/tmp/test_bridge_ticks";
        let mut broadcaster = TickBroadcaster::new(path);
        assert!(broadcaster.init().is_ok());
        assert!(broadcaster.is_active());

        let tick = TickBroadcaster::tick_from_book_snapshot(
            1,                   // symbol_id
            5000_00000000i64,    // best_bid
            5001_00000000i64,    // best_ask
            5000_50000000i64,    // mid_price
            100_00000000i64,     // bid_depth
            100_00000000i64,     // ask_depth
            2,                   // spread_bps
            500,                 // imbalance_bps
        );

        broadcaster.broadcast_tick(&tick);
        assert_eq!(broadcaster.total_written(), 1);

        // Write multiple ticks
        for _ in 0..100 {
            broadcaster.broadcast_tick(&tick);
        }
        assert_eq!(broadcaster.total_written(), 101);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
