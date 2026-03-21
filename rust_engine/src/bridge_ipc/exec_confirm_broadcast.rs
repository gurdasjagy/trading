//! Execution Confirmation Broadcast — Rust → Python execution results via SHM.
//!
//! After the Rust execution container places/fills/cancels an order, it writes
//! the confirmation to this SHM ring buffer. Python reads these confirmations
//! for analytics, PnL tracking, and strategy feedback.
//!
//! # Memory Layout
//!
//! ```text
//! [0..32]     BridgeHeader
//! [32..40]    write_cursor: u64
//! [40..48]    capacity: u64
//! [48..64]    reserved
//! [64..]      slots: [ExecConfirmSlot; capacity]
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

use super::{BridgeHeader, now_ns};

/// Default path for execution confirmation SHM.
pub const EXEC_SHM_PATH: &str = "/dev/shm/bridge_exec";

/// Number of execution confirmation slots.
const EXEC_RING_CAPACITY: usize = 2048;

/// Header size (64 bytes, cache-line aligned).
const HEADER_SIZE: usize = 64;

/// Each execution confirmation slot (128 bytes).
const EXEC_SLOT_SIZE: usize = 128;

/// Total SHM size.
const EXEC_SHM_SIZE: usize = HEADER_SIZE + (EXEC_RING_CAPACITY * EXEC_SLOT_SIZE);

/// Execution event type.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecEventType {
    /// Order submitted to exchange.
    OrderSubmitted = 1,
    /// Order acknowledged by exchange.
    OrderAcked = 2,
    /// Order partially filled.
    PartialFill = 3,
    /// Order fully filled.
    FullFill = 4,
    /// Order cancelled.
    OrderCancelled = 5,
    /// Order rejected by exchange.
    OrderRejected = 6,
    /// Order rejected by risk engine.
    RiskRejected = 7,
    /// Position opened.
    PositionOpened = 8,
    /// Position closed.
    PositionClosed = 9,
}

/// A single execution confirmation slot.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExecConfirmSlot {
    /// Sequence number for this slot.
    pub sequence: u64,
    /// Symbol identifier.
    pub symbol_id: u16,
    /// Event type (see ExecEventType).
    pub event_type: u8,
    /// Side: 0 = buy, 1 = sell.
    pub side: u8,
    /// Reserved.
    pub _reserved: u32,
    /// Client order ID (internal).
    pub client_order_id: u64,
    /// Exchange order ID.
    pub exchange_order_id: u64,
    /// Order price (FixedPrice).
    pub price_fp: i64,
    /// Filled quantity (FixedQty, 0 if not a fill).
    pub filled_qty_fp: i64,
    /// Average fill price (FixedPrice, 0 if not a fill).
    pub avg_fill_price_fp: i64,
    /// Realized PnL (FixedPrice, 0 if not a close).
    pub realized_pnl_fp: i64,
    /// Fee charged (FixedPrice).
    pub fee_fp: i64,
    /// Is this a maker fill?
    pub is_maker: u8,
    /// Leverage used.
    pub leverage: u8,
    /// Padding.
    pub _pad: [u8; 6],
    /// Nanosecond timestamp.
    pub timestamp_ns: u64,
    /// Padding to 128 bytes.
    pub _pad2: [u8; 40],
}

impl Default for ExecConfirmSlot {
    fn default() -> Self {
        Self {
            sequence: 0,
            symbol_id: 0,
            event_type: 0,
            side: 0,
            _reserved: 0,
            client_order_id: 0,
            exchange_order_id: 0,
            price_fp: 0,
            filled_qty_fp: 0,
            avg_fill_price_fp: 0,
            realized_pnl_fp: 0,
            fee_fp: 0,
            is_maker: 0,
            leverage: 0,
            _pad: [0; 6],
            timestamp_ns: 0,
            _pad2: [0; 40],
        }
    }
}

/// Writer side of the execution confirmation broadcast.
pub struct ExecConfirmBroadcaster {
    /// Memory-mapped region (writable).
    mmap: Option<memmap2::MmapMut>,
    /// Current write cursor.
    write_cursor: u64,
    /// Total confirmations written.
    total_written: AtomicU64,
    /// SHM file path.
    path: String,
}

impl ExecConfirmBroadcaster {
    /// Create a new execution confirmation broadcaster.
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
        Self::new(EXEC_SHM_PATH)
    }

    /// Initialize the SHM region.
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

        file.set_len(EXEC_SHM_SIZE as u64)
            .map_err(|e| format!("Failed to set SHM size: {}", e))?;

        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(|e| format!("Failed to mmap SHM: {}", e))?
        };

        self.mmap = Some(mmap);
        self.write_header();

        info!(
            "[exec-broadcast] Initialized SHM at {} ({} bytes, {} slots)",
            self.path, EXEC_SHM_SIZE, EXEC_RING_CAPACITY
        );
        Ok(())
    }

    /// Write the bridge header fields individually.
    fn write_header(&mut self) {
        if let Some(ref mut mmap) = self.mmap {
            let header = BridgeHeader::new();
            // Write header fields individually (BridgeHeader has align(64) padding)
            mmap[0..8].copy_from_slice(&header.magic.to_le_bytes());
            mmap[8..16].copy_from_slice(&header.version.to_le_bytes());
            mmap[16..24].copy_from_slice(&header.sequence.to_le_bytes());
            mmap[24..32].copy_from_slice(&header.timestamp_ns.to_le_bytes());
            mmap[32..40].copy_from_slice(&0u64.to_le_bytes());
            mmap[40..48].copy_from_slice(&(EXEC_RING_CAPACITY as u64).to_le_bytes());
        }
    }

    /// Broadcast an execution confirmation.
    #[inline]
    pub fn broadcast(&mut self, confirm: &ExecConfirmSlot) {
        let Some(ref mut mmap) = self.mmap else { return };

        let slot_idx = (self.write_cursor % EXEC_RING_CAPACITY as u64) as usize;
        let offset = HEADER_SIZE + (slot_idx * EXEC_SLOT_SIZE);

        let mut slot = *confirm;
        slot.sequence = self.write_cursor;
        slot.timestamp_ns = now_ns();

        let slot_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &slot as *const ExecConfirmSlot as *const u8,
                EXEC_SLOT_SIZE,
            )
        };

        if offset + EXEC_SLOT_SIZE <= mmap.len() {
            mmap[offset..offset + EXEC_SLOT_SIZE].copy_from_slice(slot_bytes);

            self.write_cursor += 1;
            let seq_bytes = self.write_cursor.to_le_bytes();
            mmap[16..24].copy_from_slice(&seq_bytes);
            mmap[32..40].copy_from_slice(&seq_bytes);

            let ts_bytes = now_ns().to_le_bytes();
            mmap[24..32].copy_from_slice(&ts_bytes);
        }

        self.total_written.fetch_add(1, Ordering::Relaxed);
    }

    /// Convenience: broadcast an order fill event.
    pub fn broadcast_fill(
        &mut self,
        symbol_id: u16,
        side: u8,
        client_order_id: u64,
        exchange_order_id: u64,
        filled_qty_fp: i64,
        fill_price_fp: i64,
        fee_fp: i64,
        is_maker: bool,
        is_partial: bool,
    ) {
        self.broadcast(&ExecConfirmSlot {
            symbol_id,
            event_type: if is_partial {
                ExecEventType::PartialFill as u8
            } else {
                ExecEventType::FullFill as u8
            },
            side,
            client_order_id,
            exchange_order_id,
            price_fp: fill_price_fp,
            filled_qty_fp,
            avg_fill_price_fp: fill_price_fp,
            fee_fp,
            is_maker: if is_maker { 1 } else { 0 },
            ..Default::default()
        });
    }

    /// Convenience: broadcast a position close event.
    pub fn broadcast_position_closed(
        &mut self,
        symbol_id: u16,
        exit_price_fp: i64,
        realized_pnl_fp: i64,
    ) {
        self.broadcast(&ExecConfirmSlot {
            symbol_id,
            event_type: ExecEventType::PositionClosed as u8,
            price_fp: exit_price_fp,
            realized_pnl_fp,
            ..Default::default()
        });
    }

    /// Get total confirmations written.
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
    fn test_exec_slot_size() {
        assert_eq!(std::mem::size_of::<ExecConfirmSlot>(), EXEC_SLOT_SIZE);
    }

    #[test]
    fn test_broadcaster_init() {
        let path = "/tmp/test_bridge_exec";
        let mut broadcaster = ExecConfirmBroadcaster::new(path);
        assert!(broadcaster.init().is_ok());
        assert!(broadcaster.is_active());

        broadcaster.broadcast_fill(
            1,      // symbol_id
            0,      // side (buy)
            12345,  // client_order_id
            67890,  // exchange_order_id
            10_0000,// filled_qty
            5000_00000000, // fill_price
            5_00000000,    // fee
            true,   // is_maker
            false,  // is_partial
        );

        assert_eq!(broadcaster.total_written(), 1);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
