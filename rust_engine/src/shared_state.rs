//! Shared-memory state writer for Rust → Python real-time data sharing.
//!
//! Uses a **seqlock** pattern for lock-free concurrent access:
//!   - Writer (Rust): increments sequence to odd (writing), writes data,
//!     increments sequence to even (consistent).
//!   - Reader (Python): reads sequence, reads data, reads sequence again.
//!     If both sequences match and are even, the read is valid.
//!
//! The state is stored in `/dev/shm/trading_state` as a memory-mapped file.
//! Python reads it via `mmap` + `struct.unpack` (see `shared_state_reader.py`).
//!
//! **Issue 2**: Replaces ZeroMQ telemetry subscription for dashboard data.

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Default shared memory path for engine state.
pub const STATE_SHM_PATH: &str = "/dev/shm/trading_state";

/// Maximum number of symbols tracked in shared state.
pub const MAX_SYMBOLS: usize = 64;

/// Magic bytes at the start of the shared memory file for validation.
pub const STATE_MAGIC: u64 = 0x5452_4144_4553_5441; // "TRADESTA"

/// Version number for the shared state layout.
/// BUMP THIS when changing SymbolState or EngineStateHeader layout.
pub const STATE_VERSION: u32 = 2;

/// Expected size of SymbolState in bytes. Must match Python's struct.unpack.
pub const SYMBOL_STATE_SIZE: usize = 176;
/// Expected size of EngineStateHeader in bytes.
pub const ENGINE_HEADER_SIZE: usize = 128;

// ═══════════════════════════════════════════════════════════════════════════
// SymbolState — per-symbol real-time data
// ═══════════════════════════════════════════════════════════════════════════

/// Per-symbol real-time state. Written by Rust, read by Python.
///
/// Layout must match Python `struct.unpack` format EXACTLY.
/// Uses `repr(C, packed)` to guarantee no compiler-inserted padding.
/// Total size: 176 bytes per symbol.
///
/// **Directive 3**: Compile-time size assertion ensures Rust and Python agree.
/// **Directive 1**: Added position lifecycle fields (entry_price, pnl, peak_pnl, etc.)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct SymbolState {
    // ── Market Data (existing fields) ──
    /// Symbol ID (from SymbolRegistry).
    pub symbol_id: u16,
    /// Exchange ID (0=gateio).
    pub exchange_id: u8,
    /// Status flags (bit 0: has_data, bit 1: is_stale, bit 2: has_position).
    pub flags: u8,
    /// Explicit alignment padding.
    pub _pad: [u8; 4],
    /// Best bid price (FixedPrice i64).
    pub best_bid_fp: i64,
    /// Best ask price (FixedPrice i64).
    pub best_ask_fp: i64,
    /// Best bid quantity (FixedQty i64).
    pub best_bid_qty_fp: i64,
    /// Best ask quantity (FixedQty i64).
    pub best_ask_qty_fp: i64,
    /// Mid price (FixedPrice i64).
    pub mid_price_fp: i64,
    /// Spread in basis points (FixedPrice i64, scaled 1e4).
    pub spread_bps_fp: i64,
    /// 1-minute VWAP (FixedPrice i64).
    pub vwap_1m_fp: i64,
    /// Order book imbalance ratio [-1.0, 1.0] (fixed-point i32, scaled 1e4).
    pub imbalance_fp: i32,
    /// VPIN toxicity [0.0, 1.0] (fixed-point i32, scaled 1e4).
    pub vpin_fp: i32,
    /// Kyle's lambda (fixed-point i32, scaled 1e8).
    pub kyle_lambda_fp: i32,
    /// Padding for 8-byte alignment.
    pub _pad2: [u8; 4],

    // ── Position Lifecycle (Directive 1 — new fields) ──
    /// Position side: 0=none, 1=long, 2=short.
    pub position_side: u8,
    /// Position state (from PositionState enum).
    pub position_state: u8,
    /// Padding for alignment.
    pub _pad3: [u8; 6],
    /// Position entry price (FixedPrice i64).
    pub entry_price_fp: i64,
    /// Position size in contracts.
    pub position_size: i64,
    /// Current unrealized PnL (USDT × 1e8).
    pub unrealized_pnl_fp: i64,
    /// PnL percentage (scaled × 1e4, so 1.5% = 15000).
    pub pnl_pct_fp: i32,
    /// Padding.
    pub _pad4: [u8; 4],
    /// Peak PnL since entry (USDT × 1e8).
    pub peak_pnl_fp: i64,
    /// PnL from peak percentage (scaled × 1e4).
    pub pnl_from_peak_pct_fp: i32,
    /// Consecutive declining ticks.
    pub consecutive_declining: u32,

    // ── Timestamps & Counters ──
    /// Last update timestamp (nanoseconds since epoch).
    pub last_update_ns: u64,
    /// Total book updates received for this symbol.
    pub book_updates_count: u64,

    // ── Reserved for future expansion ──
    /// Reserved bytes to bring total to 176 for forward-compatible IPC.
    /// Current fields sum to 152 bytes; this adds 24 to reach 176.
    pub _reserved: [u8; 24],
}
// Compile-time assertion: SymbolState MUST be exactly SYMBOL_STATE_SIZE bytes.
// If this fails, the struct layout has changed and Python must be updated too.
const _: () = assert!(
    std::mem::size_of::<SymbolState>() == SYMBOL_STATE_SIZE,
    "SymbolState size mismatch! Update Python shared_state_reader.py"
);

impl Default for SymbolState {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            exchange_id: 0,
            flags: 0,
            _pad: [0; 4],
            best_bid_fp: 0,
            best_ask_fp: 0,
            best_bid_qty_fp: 0,
            best_ask_qty_fp: 0,
            mid_price_fp: 0,
            spread_bps_fp: 0,
            vwap_1m_fp: 0,
            imbalance_fp: 0,
            vpin_fp: 0,
            kyle_lambda_fp: 0,
            _pad2: [0; 4],
            position_side: 0,
            position_state: 0,
            _pad3: [0; 6],
            entry_price_fp: 0,
            position_size: 0,
            unrealized_pnl_fp: 0,
            pnl_pct_fp: 0,
            _pad4: [0; 4],
            peak_pnl_fp: 0,
            pnl_from_peak_pct_fp: 0,
            consecutive_declining: 0,
            last_update_ns: 0,
            book_updates_count: 0,
            _reserved: [0; 24],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EngineState — aggregate engine metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregate engine state header. Written once at the start of the SHM file.
///
/// Layout:
/// ```text
/// [0..8]    magic (u64)
/// [8..12]   version (u32)
/// [12..16]  num_symbols (u32)
/// [16..24]  sequence (u64) — seqlock
/// [24..32]  uptime_seconds (u64)
/// [32..40]  total_book_updates (u64)
/// [40..48]  total_orders_sent (u64)
/// [48..56]  total_fills (u64)
/// [56..64]  total_pnl_fp (i64) — FixedPrice
/// [64..72]  engine_start_ns (u64)
/// [72..80]  last_heartbeat_ns (u64)
/// [80..128] reserved
/// [128..]   SymbolState[MAX_SYMBOLS] array
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct EngineStateHeader {
    /// Magic bytes for validation.
    pub magic: u64,
    /// Layout version.
    pub version: u32,
    /// Number of active symbols.
    pub num_symbols: u32,
    /// Seqlock sequence — odd means writer is active, even means consistent.
    pub sequence: u64,
    /// Engine uptime in seconds.
    pub uptime_seconds: u64,
    /// Total book updates processed across all symbols.
    pub total_book_updates: u64,
    /// Total orders sent to exchanges.
    pub total_orders_sent: u64,
    /// Total fills received.
    pub total_fills: u64,
    /// Total PnL in FixedPrice (i64).
    pub total_pnl_fp: i64,
    /// Engine start timestamp (nanoseconds since epoch).
    pub engine_start_ns: u64,
    /// Last heartbeat timestamp (nanoseconds since epoch).
    pub last_heartbeat_ns: u64,
    /// Account available balance (USDT × 1e8).
    pub balance_fp: i64,
    /// Account equity (USDT × 1e8).
    pub equity_fp: i64,
    /// Total unrealized PnL (USDT × 1e8).
    pub total_unrealized_pnl_fp: i64,
    /// Active position count.
    pub active_positions: u32,
    /// SymbolState struct size (for Python to validate).
    pub symbol_state_size: u32,
    /// Reserved for future expansion.
    pub _reserved: [u8; 16],
}
// Compile-time assertion: EngineStateHeader MUST be exactly ENGINE_HEADER_SIZE bytes.
const _: () = assert!(
    std::mem::size_of::<EngineStateHeader>() == ENGINE_HEADER_SIZE,
    "EngineStateHeader size mismatch! Update Python shared_state_reader.py"
);

impl Default for EngineStateHeader {
    fn default() -> Self {
        Self {
            magic: STATE_MAGIC,
            version: STATE_VERSION,
            num_symbols: 0,
            sequence: 0,
            uptime_seconds: 0,
            total_book_updates: 0,
            total_orders_sent: 0,
            total_fills: 0,
            total_pnl_fp: 0,
            engine_start_ns: 0,
            last_heartbeat_ns: 0,
            balance_fp: 0,
            equity_fp: 0,
            total_unrealized_pnl_fp: 0,
            active_positions: 0,
            symbol_state_size: SYMBOL_STATE_SIZE as u32,
            _reserved: [0; 16],
        }
    }
}

/// Total size of the shared state file.
const STATE_FILE_SIZE: usize =
    std::mem::size_of::<EngineStateHeader>() + MAX_SYMBOLS * std::mem::size_of::<SymbolState>();

// ═══════════════════════════════════════════════════════════════════════════
// SharedStateWriter
// ═══════════════════════════════════════════════════════════════════════════

/// Writes engine state to shared memory using the seqlock pattern.
///
/// The writer runs on the telemetry thread (Core 7). It updates the
/// shared memory region atomically so that Python readers never see
/// torn/inconsistent data.
pub struct SharedStateWriter {
    mmap: MmapMut,
    #[allow(dead_code)]
    path: String,
}

impl SharedStateWriter {
    /// Create a new `SharedStateWriter`.
    ///
    /// Creates or opens the shared memory file and initialises the header.
    pub fn new(path: &str) -> io::Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o666)
            .open(path)?;

        // BUG 5 FIX: Ensure SHM file is world-readable/writable so both
        // the Rust engine and Python cold-path can access it regardless
        // of which container creates it first.
        #[cfg(unix)]
        {
            let perms = std::fs::Permissions::from_mode(0o666);
            let _ = file.set_permissions(perms);
        }

        file.set_len(STATE_FILE_SIZE as u64)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let mut writer = Self {
            mmap,
            path: path.to_string(),
        };

        // Write initial header with magic and version
        let header = EngineStateHeader {
            engine_start_ns: crate::journal::now_ns(),
            ..Default::default()
        };
        writer.write_header(&header);

        Ok(writer)
    }

    /// Get a mutable pointer to the sequence field in the header.
    /// The sequence field is at offset 16 in the EngineStateHeader.
    #[inline]
    fn seq_ptr(&self) -> *const AtomicU64 {
        // sequence field offset: magic(8) + version(4) + num_symbols(4) = 16
        unsafe { self.mmap.as_ptr().add(16) as *const AtomicU64 }
    }

    /// Begin a write transaction — increments sequence to odd.
    #[inline]
    fn begin_write(&self) {
        let seq = unsafe { &*self.seq_ptr() };
        let current = seq.load(Ordering::Relaxed);
        seq.store(current + 1, Ordering::Release); // Now odd → writing
        std::sync::atomic::fence(Ordering::Release);
    }

    /// End a write transaction — increments sequence to even.
    #[inline]
    fn end_write(&self) {
        std::sync::atomic::fence(Ordering::Release);
        let seq = unsafe { &*self.seq_ptr() };
        let current = seq.load(Ordering::Relaxed);
        seq.store(current + 1, Ordering::Release); // Now even → consistent
    }

    /// Write the engine state header (excluding sequence, which is managed by seqlock).
    fn write_header(&mut self, header: &EngineStateHeader) {
        let size = std::mem::size_of::<EngineStateHeader>();
        let src = header as *const EngineStateHeader as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.mmap.as_mut_ptr(), size);
        }
    }

    /// Update the full engine state (header + all symbol states) atomically.
    ///
    /// Uses the seqlock pattern:
    ///   1. Increment sequence to odd (writing)
    ///   2. Write header fields + symbol array
    ///   3. Increment sequence to even (consistent)
    pub fn update(&mut self, header: &EngineStateHeader, symbols: &[SymbolState]) {
        self.begin_write();

        // Write header (skip the sequence field which is at offset 16)
        let header_size = std::mem::size_of::<EngineStateHeader>();
        let src = header as *const EngineStateHeader as *const u8;
        // Write everything before sequence (offset 0..16)
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.mmap.as_mut_ptr(), 16);
        }
        // Write everything after sequence (offset 24..header_size)
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.add(24),
                self.mmap.as_mut_ptr().add(24),
                header_size - 24,
            );
        }

        // Write symbol states
        let sym_offset = header_size;
        let sym_size = std::mem::size_of::<SymbolState>();
        for (i, sym) in symbols.iter().enumerate() {
            if i >= MAX_SYMBOLS {
                break;
            }
            let offset = sym_offset + i * sym_size;
            let src = sym as *const SymbolState as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(src, self.mmap.as_mut_ptr().add(offset), sym_size);
            }
        }

        self.end_write();
    }

    /// Update a single symbol's state in the shared memory.
    pub fn update_symbol(&mut self, index: usize, symbol: &SymbolState) {
        if index >= MAX_SYMBOLS {
            return;
        }

        self.begin_write();

        let header_size = std::mem::size_of::<EngineStateHeader>();
        let sym_size = std::mem::size_of::<SymbolState>();
        let offset = header_size + index * sym_size;
        let src = symbol as *const SymbolState as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.mmap.as_mut_ptr().add(offset), sym_size);
        }

        self.end_write();
    }

    /// Update just the heartbeat timestamp and aggregate counters.
    pub fn update_heartbeat(
        &mut self,
        uptime_seconds: u64,
        total_book_updates: u64,
        total_orders_sent: u64,
        total_fills: u64,
        total_pnl_fp: i64,
    ) {
        self.begin_write();

        let now = crate::journal::now_ns();

        // Write uptime_seconds at offset 24
        unsafe {
            let ptr = self.mmap.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                &uptime_seconds as *const u64 as *const u8,
                ptr.add(24),
                8,
            );
            std::ptr::copy_nonoverlapping(
                &total_book_updates as *const u64 as *const u8,
                ptr.add(32),
                8,
            );
            std::ptr::copy_nonoverlapping(
                &total_orders_sent as *const u64 as *const u8,
                ptr.add(40),
                8,
            );
            std::ptr::copy_nonoverlapping(
                &total_fills as *const u64 as *const u8,
                ptr.add(48),
                8,
            );
            std::ptr::copy_nonoverlapping(
                &total_pnl_fp as *const i64 as *const u8,
                ptr.add(56),
                8,
            );
            // engine_start_ns stays at offset 64 (written once at init)
            std::ptr::copy_nonoverlapping(
                &now as *const u64 as *const u8,
                ptr.add(72), // last_heartbeat_ns
                8,
            );
        }

        self.end_write();
    }

    /// Flush the mmap to ensure visibility.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
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
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<SymbolState>(), SYMBOL_STATE_SIZE);
        assert_eq!(mem::size_of::<EngineStateHeader>(), ENGINE_HEADER_SIZE);
    }

    #[test]
    fn test_shared_state_write_read() {
        let path = "/tmp/test_shared_state";
        let _ = std::fs::remove_file(path);

        let mut writer = SharedStateWriter::new(path).unwrap();

        let header = EngineStateHeader {
            num_symbols: 2,
            uptime_seconds: 100,
            total_book_updates: 50000,
            total_orders_sent: 100,
            total_fills: 80,
            total_pnl_fp: 1_000_000,
            ..Default::default()
        };

        let symbols = vec![
            SymbolState {
                symbol_id: 1,
                exchange_id: 0,
                flags: 1,
                best_bid_fp: 50000_00000000,
                best_ask_fp: 50001_00000000,
                ..Default::default()
            },
            SymbolState {
                symbol_id: 2,
                exchange_id: 0,
                flags: 1,
                best_bid_fp: 3000_00000000,
                best_ask_fp: 3001_00000000,
                ..Default::default()
            },
        ];

        writer.update(&header, &symbols);
        writer.flush().unwrap();

        // Read back the raw file and verify magic + version
        let data = std::fs::read(path).unwrap();
        assert!(data.len() >= mem::size_of::<EngineStateHeader>());

        let read_header = unsafe {
            std::ptr::read_unaligned(data.as_ptr() as *const EngineStateHeader)
        };
        assert_eq!(read_header.magic, STATE_MAGIC);
        assert_eq!(read_header.version, STATE_VERSION);

        let _ = std::fs::remove_file(path);
    }
}
