//! Gamma Exposure Shared Memory Reader
//!
//! Reads options-derived gamma exposure data from Python cold-path service.
//! Uses seqlock pattern for lock-free, zero-copy reads.
//!
//! Memory Layout:
//! - Header (64 bytes): magic, version, sequence, timestamp_ms, num_symbols
//! - Per-symbol entries (96 bytes each): symbol, gamma_flip_level, total_gamma, max_gamma_strike, min_gamma_strike
//! - Max 16 symbols, total size 1600 bytes

use memmap2::MmapOptions;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

const MAGIC: u64 = 0x47414D4D415F4558; // "GAMMA_EX"
const VERSION: u32 = 1;
const MAX_SYMBOLS: usize = 16;
const HEADER_SIZE: usize = 64;
const ENTRY_SIZE: usize = 96;
const TOTAL_SIZE: usize = HEADER_SIZE + (ENTRY_SIZE * MAX_SYMBOLS);
const MAX_RETRIES: usize = 100;
const DATA_TTL_MS: u64 = 5 * 60 * 1000; // 5 minutes

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Header {
    magic: u64,
    version: u32,
    _padding1: u32,
    sequence: u64,
    timestamp_ms: u64,
    num_symbols: u32,
    _padding2: u32,
    _reserved: [u8; 24],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GammaEntry {
    symbol: [u8; 32],
    gamma_flip_level_fp: u64,  // f64 as u64 bits
    total_gamma_fp: u64,
    max_gamma_strike_fp: u64,
    min_gamma_strike_fp: u64,
    _reserved: [u8; 32],
}

impl GammaEntry {
    fn symbol_str(&self) -> &str {
        let end = self.symbol.iter().position(|&b| b == 0).unwrap_or(32);
        std::str::from_utf8(&self.symbol[..end]).unwrap_or("")
    }

    fn gamma_flip_level(&self) -> f64 {
        f64::from_bits(self.gamma_flip_level_fp)
    }

    fn total_gamma(&self) -> f64 {
        f64::from_bits(self.total_gamma_fp)
    }

    fn max_gamma_strike(&self) -> f64 {
        f64::from_bits(self.max_gamma_strike_fp)
    }

    fn min_gamma_strike(&self) -> f64 {
        f64::from_bits(self.min_gamma_strike_fp)
    }
}

#[derive(Debug, Clone)]
pub struct GammaExposureData {
    pub timestamp_ms: u64,
    pub entries: Vec<(String, f64, f64, f64, f64)>, // (symbol, gamma_flip, total_gamma, max_strike, min_strike)
}

pub struct GammaExposureReader {
    mmap: memmap2::Mmap,
}

impl GammaExposureReader {
    pub fn new(path: &str) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(path)
            .unwrap_or_else(|e| panic!("Failed to open gamma exposure SHM at {}: {}", path, e));

        let mmap = unsafe {
            MmapOptions::new()
                .len(TOTAL_SIZE)
                .map(&file)
                .unwrap_or_else(|e| panic!("Failed to mmap gamma exposure SHM: {}", e))
        };

        debug!("📊 Gamma exposure reader initialized at {}", path);
        Self { mmap }
    }

    /// Try to read gamma exposure data using seqlock protocol
    pub fn try_read(&self) -> Option<GammaExposureData> {
        for attempt in 0..MAX_RETRIES {
            // Read sequence number (start)
            let seq_start = self.read_sequence();
            if seq_start % 2 != 0 {
                // Writer is currently writing, retry
                if attempt > 10 {
                    debug!("Gamma SHM: writer active (seq={}), retry {}", seq_start, attempt);
                }
                std::hint::spin_loop();
                continue;
            }

            // Read header
            let header = self.read_header();

            // Validate magic and version
            if header.magic != MAGIC {
                warn!("Gamma SHM: invalid magic: 0x{:X}", header.magic);
                return None;
            }
            if header.version != VERSION {
                warn!("Gamma SHM: version mismatch: {}", header.version);
                return None;
            }

            // Check if data is expired
            if self.is_expired(header.timestamp_ms) {
                debug!("Gamma SHM: data expired (age > 5 minutes)");
                return None;
            }

            // Read entries
            let num_symbols = header.num_symbols.min(MAX_SYMBOLS as u32) as usize;
            let mut entries = Vec::with_capacity(num_symbols);

            for i in 0..num_symbols {
                let entry = self.read_entry(i);
                let symbol = entry.symbol_str().to_string();
                let gamma_flip = entry.gamma_flip_level();
                let total_gamma = entry.total_gamma();
                let max_strike = entry.max_gamma_strike();
                let min_strike = entry.min_gamma_strike();

                entries.push((symbol, gamma_flip, total_gamma, max_strike, min_strike));
            }

            // Read sequence number (end)
            let seq_end = self.read_sequence();

            // Check if sequence changed during read
            if seq_start == seq_end {
                // Successful read
                return Some(GammaExposureData {
                    timestamp_ms: header.timestamp_ms,
                    entries,
                });
            }

            // Sequence changed, retry
            if attempt > 10 {
                debug!("Gamma SHM: sequence changed during read ({}→{}), retry {}", seq_start, seq_end, attempt);
            }
            std::hint::spin_loop();
        }

        warn!("Gamma SHM: failed to read after {} retries", MAX_RETRIES);
        None
    }

    /// Get gamma flip level for a specific symbol
    pub fn get_gamma_flip_level(&self, symbol: &str) -> Option<f64> {
        let data = self.try_read()?;

        for (sym, gamma_flip, _total, _max, _min) in data.entries {
            if sym == symbol {
                // Validate gamma flip level is in reasonable range
                let valid_range = match symbol {
                    "BTC" => 1000.0..=1000000.0,
                    "ETH" => 100.0..=100000.0,
                    _ => 0.0..=f64::MAX,
                };

                if gamma_flip > 0.0 && valid_range.contains(&gamma_flip) {
                    return Some(gamma_flip);
                } else {
                    debug!("Gamma flip level for {} out of range: {}", symbol, gamma_flip);
                    return None;
                }
            }
        }

        None
    }

    /// Safe default: return None for all symbols
    pub fn safe_default() -> Option<f64> {
        None
    }

    fn read_sequence(&self) -> u64 {
        let ptr = self.mmap.as_ptr() as *const AtomicU64;
        let atomic_seq = unsafe { &*ptr.add(1) }; // sequence is at offset 8 (after magic)
        atomic_seq.load(Ordering::Acquire)
    }

    fn read_header(&self) -> Header {
        let ptr = self.mmap.as_ptr() as *const Header;
        unsafe { *ptr }
    }

    fn read_entry(&self, index: usize) -> GammaEntry {
        let offset = HEADER_SIZE + (index * ENTRY_SIZE);
        let ptr = unsafe { self.mmap.as_ptr().add(offset) as *const GammaEntry };
        unsafe { *ptr }
    }

    fn is_expired(&self, timestamp_ms: u64) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        now_ms.saturating_sub(timestamp_ms) > DATA_TTL_MS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(MAGIC, 0x47414D4D415F4558);
        assert_eq!(VERSION, 1);
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(ENTRY_SIZE, 96);
        assert_eq!(TOTAL_SIZE, 1600);
    }

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<Header>(), HEADER_SIZE);
        assert_eq!(std::mem::size_of::<GammaEntry>(), ENTRY_SIZE);
    }
}
