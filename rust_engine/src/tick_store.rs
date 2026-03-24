//! High-performance tick database using memory-mapped append-only logs.
//!
//! Format: Each day creates a new file: /data/ticks/YYYY-MM-DD/{symbol}.tick
//! Each tick is 48 bytes: timestamp(8) + price(8) + qty(8) + side(1) + flags(1) + pad(22)
//!
//! For replay: mmap the file read-only and iterate at desired speed.
//!
//! # Architecture
//! 
//! This module provides persistent tick storage for:
//! - Historical replay for backtesting
//! - Audit trail for regulatory compliance
//! - Strategy validation against real market data
//!
//! Files are preallocated to 480MB (~10M ticks) to minimize fragmentation.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use memmap2::{MmapMut, MmapOptions};
use tracing::{debug, error, info};

/// Size of each tick record in bytes.
const TICK_SIZE: usize = 48;

/// Number of ticks to preallocate per file (~480MB).
const PREALLOC_TICKS: usize = 10_000_000;

/// Tick flags for categorizing tick types.
pub mod tick_flags {
    /// This tick represents a trade execution.
    pub const FLAG_TRADE: u8 = 1;
    /// This tick represents a book update (bid/ask change).
    pub const FLAG_BOOK_UPDATE: u8 = 2;
    /// This tick marks the start of a snapshot.
    pub const FLAG_SNAPSHOT_MARKER: u8 = 4;
    /// This tick is from the ask side.
    pub const FLAG_ASK_SIDE: u8 = 8;
    /// This tick is aggressive (market order).
    pub const FLAG_AGGRESSIVE: u8 = 16;
}

/// A single tick record stored in the tick database.
/// 
/// Layout (48 bytes total):
/// - timestamp_ns: 8 bytes (nanoseconds since epoch)
/// - price_fp: 8 bytes (fixed-point price, scaled by 1e8)
/// - qty_fp: 8 bytes (fixed-point quantity, scaled by 1e8)
/// - side: 1 byte (0=bid, 1=ask)
/// - flags: 1 byte (bitfield: trade/book_update/snapshot)
/// - symbol_id: 2 bytes (symbol identifier)
/// - _pad: 20 bytes (reserved for future use)
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StoredTick {
    /// Timestamp in nanoseconds since Unix epoch.
    pub timestamp_ns: u64,
    /// Price in fixed-point format (actual_price * 1e8).
    pub price_fp: i64,
    /// Quantity in fixed-point format (actual_qty * 1e8).
    pub qty_fp: i64,
    /// Side: 0 = bid, 1 = ask.
    pub side: u8,
    /// Flags bitfield (see tick_flags module).
    pub flags: u8,
    /// Symbol identifier for multi-symbol storage.
    pub symbol_id: u16,
    /// Reserved padding for future extensions.
    pub _pad: [u8; 20],
}

impl StoredTick {
    /// Create a new tick with the given parameters.
    pub fn new(
        timestamp_ns: u64,
        price: f64,
        qty: f64,
        side: u8,
        flags: u8,
        symbol_id: u16,
    ) -> Self {
        Self {
            timestamp_ns,
            price_fp: (price * 1e8) as i64,
            qty_fp: (qty * 1e8) as i64,
            side,
            flags,
            symbol_id,
            _pad: [0u8; 20],
        }
    }
    
    /// Get the price as a floating-point number.
    #[inline]
    pub fn price(&self) -> f64 {
        self.price_fp as f64 / 1e8
    }
    
    /// Get the quantity as a floating-point number.
    #[inline]
    pub fn qty(&self) -> f64 {
        self.qty_fp as f64 / 1e8
    }
    
    /// Check if this tick represents a trade.
    #[inline]
    pub fn is_trade(&self) -> bool {
        self.flags & tick_flags::FLAG_TRADE != 0
    }
    
    /// Check if this tick is a book update.
    #[inline]
    pub fn is_book_update(&self) -> bool {
        self.flags & tick_flags::FLAG_BOOK_UPDATE != 0
    }
}

/// Writer for appending ticks to a memory-mapped file.
pub struct TickWriter {
    mmap: MmapMut,
    offset: usize,
    path: PathBuf,
    ticks_written: u64,
}

impl TickWriter {
    /// Create a new tick writer for the given symbol and date.
    ///
    /// # Arguments
    /// * `base_path` - Base directory for tick storage (e.g., /data/ticks)
    /// * `symbol` - Trading symbol (e.g., "BTC_USDT")
    /// * `date` - Date string in YYYY-MM-DD format
    ///
    /// # Returns
    /// A new TickWriter or an error if the file couldn't be created.
    pub fn new(base_path: &str, symbol: &str, date: &str) -> io::Result<Self> {
        let dir = PathBuf::from(format!("{}/{}", base_path, date));
        std::fs::create_dir_all(&dir)?;
        
        let path = dir.join(format!("{}.tick", symbol));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
        
        // Preallocate file to avoid fragmentation
        file.set_len((TICK_SIZE * PREALLOC_TICKS) as u64)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        info!("Opened tick store: {} ({} ticks preallocated)", path.display(), PREALLOC_TICKS);
        
        Ok(Self { 
            mmap, 
            offset: 0, 
            path,
            ticks_written: 0,
        })
    }
    
    /// Open an existing tick file for appending.
    pub fn open_append(base_path: &str, symbol: &str, date: &str) -> io::Result<Self> {
        let path = PathBuf::from(format!("{}/{}/{}.tick", base_path, date, symbol));
        
        if !path.exists() {
            return Self::new(base_path, symbol, date);
        }
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Find the first empty slot (where timestamp is 0)
        let mut offset = 0;
        while offset + TICK_SIZE <= mmap.len() {
            let ts = u64::from_le_bytes(mmap[offset..offset+8].try_into().unwrap());
            if ts == 0 {
                break;
            }
            offset += TICK_SIZE;
        }
        
        let ticks_written = (offset / TICK_SIZE) as u64;
        info!("Resuming tick store: {} at offset {} ({} existing ticks)", 
              path.display(), offset, ticks_written);
        
        Ok(Self { mmap, offset, path, ticks_written })
    }
    
    /// Write a single tick to the file.
    ///
    /// # Arguments
    /// * `tick` - The tick to write
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if the file is full or write failed
    #[inline]
    pub fn write(&mut self, tick: &StoredTick) -> io::Result<()> {
        if self.offset + TICK_SIZE > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::Other, 
                "Tick file full - rotate to new file"
            ));
        }
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tick as *const StoredTick as *const u8,
                TICK_SIZE
            )
        };
        
        self.mmap[self.offset..self.offset + TICK_SIZE].copy_from_slice(bytes);
        self.offset += TICK_SIZE;
        self.ticks_written += 1;
        Ok(())
    }
    
    /// Write a batch of ticks efficiently.
    pub fn write_batch(&mut self, ticks: &[StoredTick]) -> io::Result<usize> {
        let mut written = 0;
        for tick in ticks {
            if self.write(tick).is_ok() {
                written += 1;
            } else {
                break;
            }
        }
        Ok(written)
    }
    
    /// Flush the memory map to disk.
    pub fn sync(&self) -> io::Result<()> {
        self.mmap.flush()
    }
    
    /// Get the number of ticks written.
    pub fn ticks_written(&self) -> u64 {
        self.ticks_written
    }
    
    /// Get the file path.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
    
    /// Get remaining capacity in ticks.
    pub fn remaining_capacity(&self) -> usize {
        (self.mmap.len() - self.offset) / TICK_SIZE
    }
}

impl Drop for TickWriter {
    fn drop(&mut self) {
        if let Err(e) = self.sync() {
            error!("Failed to sync tick store on drop: {}", e);
        } else {
            debug!("Tick store synced: {} ticks", self.ticks_written);
        }
    }
}

/// Reader for replaying ticks from a memory-mapped file.
pub struct TickReader {
    data: memmap2::Mmap,
    offset: usize,
    len: usize,
}

impl TickReader {
    /// Open a tick file for reading.
    ///
    /// # Arguments
    /// * `base_path` - Base directory for tick storage
    /// * `symbol` - Trading symbol
    /// * `date` - Date string in YYYY-MM-DD format
    pub fn open(base_path: &str, symbol: &str, date: &str) -> io::Result<Self> {
        let path = PathBuf::from(format!("{}/{}/{}.tick", base_path, date, symbol));
        let file = File::open(&path)?;
        let data = unsafe { MmapOptions::new().map(&file)? };
        
        // Count actual ticks (non-zero timestamps)
        let mut len = 0;
        let max_ticks = data.len() / TICK_SIZE;
        for i in 0..max_ticks {
            let ts = u64::from_le_bytes(
                data[i * TICK_SIZE..i * TICK_SIZE + 8].try_into().unwrap()
            );
            if ts == 0 {
                break;
            }
            len += 1;
        }
        
        info!("Opened tick reader: {} ({} ticks)", path.display(), len);
        Ok(Self { data, offset: 0, len })
    }
    
    /// Read the next tick, advancing the cursor.
    pub fn next(&mut self) -> Option<StoredTick> {
        if self.offset >= self.len {
            return None;
        }
        
        let byte_offset = self.offset * TICK_SIZE;
        let tick: StoredTick = unsafe {
            std::ptr::read(self.data[byte_offset..].as_ptr() as *const StoredTick)
        };
        self.offset += 1;
        Some(tick)
    }
    
    /// Peek at the next tick without advancing the cursor.
    pub fn peek(&self) -> Option<StoredTick> {
        if self.offset >= self.len {
            return None;
        }
        
        let byte_offset = self.offset * TICK_SIZE;
        let tick: StoredTick = unsafe {
            std::ptr::read(self.data[byte_offset..].as_ptr() as *const StoredTick)
        };
        Some(tick)
    }
    
    /// Seek to a specific tick index.
    pub fn seek(&mut self, tick_index: usize) {
        self.offset = tick_index.min(self.len);
    }
    
    /// Seek to a specific timestamp (binary search).
    pub fn seek_timestamp(&mut self, target_ns: u64) {
        let mut left = 0;
        let mut right = self.len;
        
        while left < right {
            let mid = left + (right - left) / 2;
            let byte_offset = mid * TICK_SIZE;
            let ts = u64::from_le_bytes(
                self.data[byte_offset..byte_offset + 8].try_into().unwrap()
            );
            
            if ts < target_ns {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        self.offset = left;
    }
    
    /// Get the total number of ticks in the file.
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the reader is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the current position.
    pub fn position(&self) -> usize {
        self.offset
    }
    
    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Iterator for TickReader {
    type Item = StoredTick;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

/// Manager for rotating tick files across days.
pub struct TickStoreManager {
    base_path: String,
    writers: std::collections::HashMap<String, TickWriter>,
    current_date: String,
    enabled: bool,
}

impl TickStoreManager {
    /// Create a new tick store manager.
    pub fn new(base_path: &str, enabled: bool) -> Self {
        Self {
            base_path: base_path.to_string(),
            writers: std::collections::HashMap::new(),
            current_date: Self::today(),
            enabled,
        }
    }
    
    fn today() -> String {
        chrono_lite_date()
    }
    
    /// Record a tick for a symbol.
    pub fn record(&mut self, symbol: &str, tick: StoredTick) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let today = Self::today();
        if today != self.current_date {
            // Day rollover - close old writers
            self.writers.clear();
            self.current_date = today.clone();
            info!("Tick store rolled over to new day: {}", self.current_date);
        }
        
        let writer = self.writers
            .entry(symbol.to_string())
            .or_insert_with(|| {
                TickWriter::new(&self.base_path, symbol, &self.current_date)
                    .expect("Failed to create tick writer")
            });
        
        writer.write(&tick)
    }
    
    /// Sync all writers to disk.
    pub fn sync_all(&self) -> io::Result<()> {
        for writer in self.writers.values() {
            writer.sync()?;
        }
        Ok(())
    }
    
    /// Get statistics for all symbols.
    pub fn stats(&self) -> Vec<(String, u64)> {
        self.writers
            .iter()
            .map(|(sym, w)| (sym.clone(), w.ticks_written()))
            .collect()
    }
}

/// Simple date formatter (avoids chrono dependency).
fn chrono_lite_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    // Convert to days since epoch, then to date components
    let days = (secs / 86400) as i64;
    let (year, month, day) = days_to_ymd(days + 719468); // Days since year 0
    
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert days since year 0 to year/month/day.
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = (days - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    
    (year as i32, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_stored_tick_size() {
        assert_eq!(std::mem::size_of::<StoredTick>(), TICK_SIZE);
    }
    
    #[test]
    fn test_tick_write_read() {
        // Use a unique temp directory to avoid tempfile dependency
        let base_path = std::env::temp_dir()
            .join(format!("tick_store_test_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()));
        let base_path = base_path.to_str().unwrap().to_string();
        fs::create_dir_all(&base_path).unwrap();
        
        // Write some ticks
        {
            let mut writer = TickWriter::new(&base_path, "BTC_USDT", "2024-01-15").unwrap();
            
            for i in 0..100 {
                let tick = StoredTick::new(
                    1705344000_000_000_000 + i as u64 * 1_000_000,
                    43000.0 + i as f64,
                    0.1,
                    if i % 2 == 0 { 0 } else { 1 },
                    tick_flags::FLAG_TRADE,
                    0,
                );
                writer.write(&tick).unwrap();
            }
            writer.sync().unwrap();
        }
        
        // Read them back
        let reader = TickReader::open(&base_path, "BTC_USDT", "2024-01-15").unwrap();
        assert_eq!(reader.len(), 100);
        
        // Cleanup
        let _ = fs::remove_dir_all(&base_path);
    }
}
