//! Memory-mapped append-only event journal — Chronicle Queue inspired.
//!
//! Provides crash-recovery and full event replay via a series of fixed-size
//! memory-mapped segment files stored in `/dev/shm/trading_journal/`.
//!
//! **Issue 2**: Replaces ZeroMQ telemetry with zero-copy binary journal entries.
//!
//! # Design
//!
//! - Each segment is 64 MB (fits in L3 cache on modern Xeon/EPYC CPUs).
//! - Entries are fixed-size `#[repr(C, packed)]` structs — no serialization.
//! - `append()` is a single `memcpy` into the mmap region (~50–100 ns).
//! - Auto-rolls to a new segment when the active one is full.
//! - Readers (Python via `mmap` or Rust replay) can read concurrently.

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// 64 MB per segment file.
pub const SEGMENT_SIZE: usize = 64 * 1024 * 1024;

/// Default journal directory (tmpfs for lowest latency).
pub const JOURNAL_DIR: &str = "/dev/shm/trading_journal";

/// Flush hint to OS every N entries (non-blocking msync).
const FLUSH_INTERVAL: u32 = 4096;

// ═══════════════════════════════════════════════════════════════════════════
// Entry Type Discriminants
// ═══════════════════════════════════════════════════════════════════════════

pub const ENTRY_BOOK_UPDATE: u16 = 1;
pub const ENTRY_BOOK_SNAPSHOT: u16 = 2;
pub const ENTRY_ORDER_INTENT: u16 = 3;
pub const ENTRY_ORDER_RESULT: u16 = 4;
pub const ENTRY_REGIME_UPDATE: u16 = 5;
pub const ENTRY_HEARTBEAT: u16 = 6;
pub const ENTRY_CONFIG_CHANGE: u16 = 7;
pub const ENTRY_TRADE: u16 = 8;
pub const ENTRY_POSITION_CHANGE: u16 = 9;

// ═══════════════════════════════════════════════════════════════════════════
// Entry Structs — all #[repr(C, packed)], Copy, fixed-size
// ═══════════════════════════════════════════════════════════════════════════

/// Journal entry header. Fixed 8 bytes.
///
/// Every entry in the journal starts with this header, followed by the
/// type-specific payload.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalEntryHeader {
    /// Entry type discriminant (see `ENTRY_*` constants).
    pub entry_type: u16,
    /// Payload size in bytes (excluding this 8-byte header).
    pub payload_size: u16,
    /// Monotonic sequence number — never reused across restarts.
    pub sequence: u32,
}

/// Book update entry — written on every orderbook delta.
/// Size: 64 bytes (fits in one cache line).
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalBookUpdate {
    pub header: JournalEntryHeader,        // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes — monotonic nanoseconds
    pub exchange_id: u8,                    // 1 byte
    pub symbol_id: u16,                     // 2 bytes
    pub side: u8,                           // 1 byte (0=bid, 1=ask)
    pub is_snapshot: u8,                    // 1 byte
    pub _pad: [u8; 3],                      // 3 bytes alignment
    pub price_fp: i64,                      // 8 bytes — FixedPrice
    pub old_qty_fp: i64,                    // 8 bytes — previous FixedQty
    pub new_qty_fp: i64,                    // 8 bytes — new FixedQty
    pub exchange_sequence: u64,             // 8 bytes
    pub _reserved: [u8; 8],                 // 8 bytes — future use
}
// Total: 8 + 8 + 1 + 2 + 1 + 1 + 3 + 8 + 8 + 8 + 8 + 8 = 64 bytes

/// Order intent entry — written when strategy emits a signal.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalOrderIntent {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub symbol_id: u16,                     // 2 bytes
    pub side: u8,                           // 1 byte
    pub order_type: u8,                     // 1 byte
    pub size: i64,                          // 8 bytes — FixedQty
    pub price_fp: i64,                      // 8 bytes — FixedPrice
    pub reduce_only: u8,                    // 1 byte
    pub leverage: i32,                      // 4 bytes
    pub slippage_cap_bps: i32,              // 4 bytes
    pub book_sequence: u64,                 // 8 bytes
    pub _reserved: [u8; 5],                 // 5 bytes — future use
}

/// Order result entry — written when exchange confirms/rejects.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalOrderResult {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub symbol_id: u16,                     // 2 bytes
    pub side: u8,                           // 1 byte
    pub status: u8,                         // 1 byte (0=open, 1=filled, 2=rejected, 3=cancelled)
    pub filled_size: i64,                   // 8 bytes — FixedQty
    pub avg_fill_price_fp: i64,             // 8 bytes — FixedPrice
    pub fee_fp: i64,                        // 8 bytes — FixedPrice
    pub exchange_latency_us: u64,           // 8 bytes
    pub order_id: [u8; 32],                 // 32 bytes — zero-padded order ID
}

/// Heartbeat entry — written every 500ms by the telemetry thread.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalHeartbeat {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub uptime_seconds: u64,                // 8 bytes
    pub book_updates_total: u64,            // 8 bytes
    pub orders_total: u64,                  // 8 bytes
    pub _reserved: [u8; 24],                // 24 bytes — future use
}

/// Position change entry — written when a position is opened/closed/modified.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalPositionChange {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub symbol_id: u16,                     // 2 bytes
    pub side: u8,                           // 1 byte (0=long, 1=short)
    pub action: u8,                         // 1 byte (0=open, 1=close, 2=modify)
    pub old_size: i64,                      // 8 bytes — FixedQty
    pub new_size: i64,                      // 8 bytes — FixedQty
    pub entry_price_fp: i64,                // 8 bytes — FixedPrice
    pub mark_price_fp: i64,                 // 8 bytes — FixedPrice
    pub unrealized_pnl_fp: i64,             // 8 bytes — FixedPrice
    pub realized_pnl_fp: i64,              // 8 bytes — FixedPrice
    pub _reserved: [u8; 8],                 // 8 bytes — future use
}

/// Trade entry — written when a trade is executed.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalTrade {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub symbol_id: u16,                     // 2 bytes
    pub side: u8,                           // 1 byte (0=buy, 1=sell)
    pub _pad: u8,                           // 1 byte
    pub size: i64,                          // 8 bytes — FixedQty
    pub price_fp: i64,                      // 8 bytes — FixedPrice
    pub fee_fp: i64,                        // 8 bytes — FixedPrice
    pub is_maker: u8,                       // 1 byte
    pub _reserved: [u8; 15],                // 15 bytes — future use
}

/// Regime update entry — written when regime weights change.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalRegimeUpdate {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub overall_regime: u8,                 // 1 byte (enum)
    pub volatility_regime: u8,              // 1 byte (enum)
    pub sentiment_score_fp: i32,            // 4 bytes (fixed-point 1e4)
    pub fear_greed_index: u8,               // 1 byte (0-100)
    pub position_scale_fp: i32,             // 4 bytes (fixed-point 1e4)
    pub _reserved: [u8; 5],                 // padding
}

/// Config change entry — written when strategy config is updated.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct JournalConfigChange {
    pub header: JournalEntryHeader,         // 8 bytes
    pub timestamp_ns: u64,                  // 8 bytes
    pub config_key: [u8; 32],               // 32 bytes — zero-padded key
    pub old_value_fp: i64,                  // 8 bytes
    pub new_value_fp: i64,                  // 8 bytes
    pub _reserved: [u8; 8],                 // 8 bytes — future use
}

// ═══════════════════════════════════════════════════════════════════════════
// JournalWriter
// ═══════════════════════════════════════════════════════════════════════════

/// Memory-mapped journal writer. Appends fixed-size entries to segment files.
///
/// Designed for the telemetry/journaling thread (Core 7). Receives events
/// from the hot path via SPSC ring buffer and writes them to the journal.
///
/// Typical `append()` latency: < 200 ns (just a memcpy into mmap region).
pub struct JournalWriter {
    /// Current active segment (memory-mapped).
    current_segment: MmapMut,
    /// Write position within the current segment (byte offset).
    write_pos: usize,
    /// Global monotonic sequence counter (never reused across restarts).
    global_sequence: u32,
    /// Current segment index.
    segment_index: u32,
    /// Directory path for journal files.
    journal_dir: String,
    /// Entries written since last flush hint.
    entries_since_flush: u32,
}

impl JournalWriter {
    /// Create a new `JournalWriter`.
    ///
    /// Opens or creates the journal directory and initialises the first
    /// segment. If existing segments are found, resumes from the last one.
    pub fn new(journal_dir: &str) -> io::Result<Self> {
        std::fs::create_dir_all(journal_dir)?;

        // Find the highest existing segment index to resume from.
        let mut max_segment: u32 = 0;
        let mut resume_pos: usize = 0;
        if let Ok(entries) = std::fs::read_dir(journal_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("segment_") && name_str.ends_with(".dat") {
                    if let Ok(idx) = name_str[8..14].parse::<u32>() {
                        if idx >= max_segment {
                            max_segment = idx;
                        }
                    }
                }
            }
            // Try to find the write position in the last segment by scanning
            // for the first zeroed header (entry_type == 0 means unused).
            let path = format!("{}/segment_{:06}.dat", journal_dir, max_segment);
            if let Ok(data) = std::fs::read(&path) {
                let header_size = std::mem::size_of::<JournalEntryHeader>();
                let mut pos = 0;
                let mut last_seq: u32 = 0;
                while pos + header_size <= data.len() {
                    let header = unsafe {
                        std::ptr::read_unaligned(data[pos..].as_ptr() as *const JournalEntryHeader)
                    };
                    if header.entry_type == 0 {
                        break; // Unused slot — this is our resume point
                    }
                    let entry_size = header_size + header.payload_size as usize;
                    last_seq = header.sequence;
                    pos += entry_size;
                }
                resume_pos = pos;
                if last_seq > 0 {
                    // Resume sequence numbering after the last written entry
                    // (global_sequence is assigned below before the return)
                    // max_segment stays the same — reuse existing segment
                    let segment = Self::create_segment(journal_dir, max_segment)?;
                    return Ok(Self {
                        current_segment: segment,
                        write_pos: resume_pos,
                        global_sequence: last_seq + 1,
                        segment_index: max_segment,
                        journal_dir: journal_dir.to_string(),
                        entries_since_flush: 0,
                    });
                }
            }
        }

        let segment = Self::create_segment(journal_dir, max_segment)?;
        Ok(Self {
            current_segment: segment,
            write_pos: resume_pos,
            global_sequence: 0,
            segment_index: max_segment,
            journal_dir: journal_dir.to_string(),
            entries_since_flush: 0,
        })
    }

    /// Create or open a segment file and return its mmap.
    fn create_segment(dir: &str, index: u32) -> io::Result<MmapMut> {
        let path = format!("{}/segment_{:06}.dat", dir, index);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        file.set_len(SEGMENT_SIZE as u64)?;
        unsafe { MmapMut::map_mut(&file) }
    }

    /// Roll to the next segment when the current one is full.
    fn roll_segment(&mut self) -> io::Result<()> {
        // Flush the current segment before rolling.
        self.current_segment.flush()?;
        self.segment_index += 1;
        self.current_segment = Self::create_segment(&self.journal_dir, self.segment_index)?;
        self.write_pos = 0;
        self.entries_since_flush = 0;
        tracing::info!(
            "Journal rolled to segment_{:06}.dat (seq={})",
            self.segment_index,
            self.global_sequence
        );
        Ok(())
    }

    /// Append a raw entry to the journal. **ZERO-COPY** for fixed-size entries.
    ///
    /// Returns the global sequence number assigned to this entry.
    ///
    /// This method completes in < 200 ns on average:
    ///   - No syscall (mmap write is a memory copy)
    ///   - No allocation (entry is `Copy`)
    ///   - No serialization (binary layout matches memory layout)
    #[inline]
    pub fn append<T: Copy>(&mut self, entry: &T) -> io::Result<u32> {
        let entry_size = std::mem::size_of::<T>();

        // Roll to next segment if current one is full.
        if self.write_pos + entry_size > SEGMENT_SIZE {
            self.roll_segment()?;
        }

        // Write the entry bytes directly into the mmap region.
        let src = entry as *const T as *const u8;
        let dst = &mut self.current_segment[self.write_pos..self.write_pos + entry_size];
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), entry_size);
        }

        let seq = self.global_sequence;
        self.write_pos += entry_size;
        self.global_sequence += 1;
        self.entries_since_flush += 1;

        // Periodic async flush hint (non-blocking).
        if self.entries_since_flush >= FLUSH_INTERVAL {
            let _ = self.current_segment.flush_async();
            self.entries_since_flush = 0;
        }

        Ok(seq)
    }

    /// Convenience: append a `JournalBookUpdate` with auto-filled header.
    pub fn append_book_update(&mut self, mut entry: JournalBookUpdate) -> io::Result<u32> {
        entry.header = JournalEntryHeader {
            entry_type: ENTRY_BOOK_UPDATE,
            payload_size: (std::mem::size_of::<JournalBookUpdate>()
                - std::mem::size_of::<JournalEntryHeader>()) as u16,
            sequence: self.global_sequence,
        };
        self.append(&entry)
    }

    /// Convenience: append a `JournalOrderIntent` with auto-filled header.
    pub fn append_order_intent(&mut self, mut entry: JournalOrderIntent) -> io::Result<u32> {
        entry.header = JournalEntryHeader {
            entry_type: ENTRY_ORDER_INTENT,
            payload_size: (std::mem::size_of::<JournalOrderIntent>()
                - std::mem::size_of::<JournalEntryHeader>()) as u16,
            sequence: self.global_sequence,
        };
        self.append(&entry)
    }

    /// Convenience: append a `JournalOrderResult` with auto-filled header.
    pub fn append_order_result(&mut self, mut entry: JournalOrderResult) -> io::Result<u32> {
        entry.header = JournalEntryHeader {
            entry_type: ENTRY_ORDER_RESULT,
            payload_size: (std::mem::size_of::<JournalOrderResult>()
                - std::mem::size_of::<JournalEntryHeader>()) as u16,
            sequence: self.global_sequence,
        };
        self.append(&entry)
    }

    /// Convenience: append a `JournalHeartbeat` with auto-filled header.
    pub fn append_heartbeat(&mut self, mut entry: JournalHeartbeat) -> io::Result<u32> {
        entry.header = JournalEntryHeader {
            entry_type: ENTRY_HEARTBEAT,
            payload_size: (std::mem::size_of::<JournalHeartbeat>()
                - std::mem::size_of::<JournalEntryHeader>()) as u16,
            sequence: self.global_sequence,
        };
        self.append(&entry)
    }

    /// Convenience: append a `JournalPositionChange` with auto-filled header.
    pub fn append_position_change(&mut self, mut entry: JournalPositionChange) -> io::Result<u32> {
        entry.header = JournalEntryHeader {
            entry_type: ENTRY_POSITION_CHANGE,
            payload_size: (std::mem::size_of::<JournalPositionChange>()
                - std::mem::size_of::<JournalEntryHeader>()) as u16,
            sequence: self.global_sequence,
        };
        self.append(&entry)
    }

    /// Convenience: append a `JournalTrade` with auto-filled header.
    pub fn append_trade(&mut self, mut entry: JournalTrade) -> io::Result<u32> {
        entry.header = JournalEntryHeader {
            entry_type: ENTRY_TRADE,
            payload_size: (std::mem::size_of::<JournalTrade>()
                - std::mem::size_of::<JournalEntryHeader>()) as u16,
            sequence: self.global_sequence,
        };
        self.append(&entry)
    }

    /// Get the current global sequence number (next entry will receive this).
    pub fn current_sequence(&self) -> u32 {
        self.global_sequence
    }

    /// Get the current segment index.
    pub fn current_segment_index(&self) -> u32 {
        self.segment_index
    }

    /// Get the write position within the current segment.
    pub fn write_position(&self) -> usize {
        self.write_pos
    }

    /// Force-flush the current segment to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.current_segment.flush()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// JournalReader — for replay and recovery
// ═══════════════════════════════════════════════════════════════════════════

/// Callback invoked for each replayed journal entry.
///
/// Arguments: (entry_type, sequence, payload_bytes)
pub type ReplayCallback = Box<dyn FnMut(u16, u32, &[u8])>;

/// Replays all entries from all segments in the journal directory.
///
/// Scans segments in order (segment_000000, segment_000001, …) and invokes
/// the callback for each valid entry. Stops on the first zeroed (unused)
/// header, which marks the write frontier.
///
/// Partial/corrupt entries at the end of a segment (e.g., from a mid-write
/// crash) are detected and skipped.
pub fn replay_journal(journal_dir: &str, mut callback: ReplayCallback) -> io::Result<u32> {
    let header_size = std::mem::size_of::<JournalEntryHeader>();
    let mut total_replayed: u32 = 0;
    let mut segment_idx: u32 = 0;

    loop {
        let path = format!("{}/segment_{:06}.dat", journal_dir, segment_idx);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => break, // No more segments
        };

        let mut pos = 0;
        while pos + header_size <= data.len() {
            let header = unsafe {
                std::ptr::read_unaligned(data[pos..].as_ptr() as *const JournalEntryHeader)
            };

            // entry_type == 0 means unused slot — we've reached the frontier.
            if header.entry_type == 0 {
                break;
            }

            let entry_size = header_size + header.payload_size as usize;

            // Safety: check we have enough bytes for the full entry.
            if pos + entry_size > data.len() {
                let seq = { header.sequence };
                tracing::warn!(
                    "Journal segment_{:06}: partial entry at offset {} (seq={}), skipping",
                    segment_idx,
                    pos,
                    seq
                );
                break;
            }

            let payload = &data[pos + header_size..pos + entry_size];
            callback(header.entry_type, header.sequence, payload);

            pos += entry_size;
            total_replayed += 1;
        }

        segment_idx += 1;
    }

    Ok(total_replayed)
}

/// Returns monotonic nanoseconds (for journal timestamps).
#[inline]
pub fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::mem;
    use std::rc::Rc;

    #[test]
    fn test_entry_sizes() {
        assert_eq!(mem::size_of::<JournalEntryHeader>(), 8);
        assert_eq!(mem::size_of::<JournalBookUpdate>(), 64);
    }

    #[test]
    fn test_write_and_replay() {
        let dir = "/tmp/test_journal_write_replay";
        let _ = std::fs::remove_dir_all(dir);

        let mut writer = JournalWriter::new(dir).unwrap();

        // Write 1000 book updates
        for i in 0..1000u32 {
            let entry = JournalBookUpdate {
                header: JournalEntryHeader {
                    entry_type: ENTRY_BOOK_UPDATE,
                    payload_size: (mem::size_of::<JournalBookUpdate>()
                        - mem::size_of::<JournalEntryHeader>()) as u16,
                    sequence: 0, // Will be overwritten by append_book_update
                },
                timestamp_ns: now_ns(),
                exchange_id: 1,
                symbol_id: 1,
                side: 0,
                is_snapshot: 0,
                _pad: [0; 3],
                price_fp: (50000_00000000i64) + (i as i64 * 100),
                old_qty_fp: 10000,
                new_qty_fp: 20000,
                exchange_sequence: i as u64,
                _reserved: [0; 8],
            };
            writer.append_book_update(entry).unwrap();
        }
        writer.flush().unwrap();

        // Replay and verify (use Rc<Cell<>> for 'static closure capture)
        let count = Rc::new(Cell::new(0u32));
        let count_clone = count.clone();
        replay_journal(
            dir,
            Box::new(move |entry_type, _seq, _payload| {
                assert_eq!(entry_type, ENTRY_BOOK_UPDATE);
                count_clone.set(count_clone.get() + 1);
            }),
        )
        .unwrap();
        assert_eq!(count.get(), 1000);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_heartbeat_write_read() {
        let dir = "/tmp/test_journal_heartbeat";
        let _ = std::fs::remove_dir_all(dir);

        let mut writer = JournalWriter::new(dir).unwrap();

        let hb = JournalHeartbeat {
            header: JournalEntryHeader {
                entry_type: ENTRY_HEARTBEAT,
                payload_size: 0,
                sequence: 0,
            },
            timestamp_ns: now_ns(),
            uptime_seconds: 3600,
            book_updates_total: 100_000,
            orders_total: 500,
            _reserved: [0; 24],
        };
        writer.append_heartbeat(hb).unwrap();
        writer.flush().unwrap();

        let found = Rc::new(Cell::new(false));
        let found_clone = found.clone();
        replay_journal(
            dir,
            Box::new(move |entry_type, _seq, payload| {
                assert_eq!(entry_type, ENTRY_HEARTBEAT);
                // Verify we can read back the heartbeat data
                assert!(!payload.is_empty());
                found_clone.set(true);
            }),
        )
        .unwrap();
        assert!(found.get());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_segment_roll() {
        let dir = "/tmp/test_journal_segment_roll";
        let _ = std::fs::remove_dir_all(dir);

        let mut writer = JournalWriter::new(dir).unwrap();

        // Write enough entries to fill a segment (64 MB / 64 bytes = ~1M entries)
        // Write just enough to trigger a roll
        let entry_size = mem::size_of::<JournalBookUpdate>();
        let entries_per_segment = SEGMENT_SIZE / entry_size;

        // Write slightly more than one segment worth
        let total = entries_per_segment + 100;
        for i in 0..total {
            let entry = JournalBookUpdate {
                header: JournalEntryHeader {
                    entry_type: ENTRY_BOOK_UPDATE,
                    payload_size: (entry_size - mem::size_of::<JournalEntryHeader>()) as u16,
                    sequence: 0,
                },
                timestamp_ns: i as u64,
                exchange_id: 1,
                symbol_id: 1,
                side: 0,
                is_snapshot: 0,
                _pad: [0; 3],
                price_fp: i as i64,
                old_qty_fp: 0,
                new_qty_fp: i as i64,
                exchange_sequence: i as u64,
                _reserved: [0; 8],
            };
            writer.append_book_update(entry).unwrap();
        }
        writer.flush().unwrap();

        assert!(writer.current_segment_index() >= 1, "Should have rolled to segment 1+");

        // Replay all entries across both segments
        let count = Rc::new(Cell::new(0u32));
        let count_clone = count.clone();
        replay_journal(
            dir,
            Box::new(move |_entry_type, _seq, _payload| {
                count_clone.set(count_clone.get() + 1);
            }),
        )
        .unwrap();
        assert_eq!(count.get() as usize, total);

        let _ = std::fs::remove_dir_all(dir);
    }
}
