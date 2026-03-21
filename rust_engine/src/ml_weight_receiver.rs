use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{info, warn};

const MAGIC_BYTES: u64 = 0x4D4C5F5747485453; // "ML_WGHTS"
const MAX_SYMBOLS: usize = 1024;
const SHM_SIZE: usize = 65536;

#[repr(C, align(64))]
pub struct SharedMemWeights {
    pub seqlock: AtomicU32,
    pub _pad1: u32,
    pub magic: u64,
    pub model_version: u64,
    pub num_symbols: u32,
    pub _pad2: u32,
    pub weights: [SymbolWeight; MAX_SYMBOLS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SymbolWeight {
    pub symbol_id: u16,
    pub _pad: u16,
    pub momentum_weight: f32,
    pub mean_reversion_weight: f32,
    pub volatility_weight: f32,
    pub confidence_floor: f32,
    pub max_position_scale: f32,
}

pub struct MlWeightReader {
    shm_ptr: *const SharedMemWeights,
}

unsafe impl Send for MlWeightReader {}
unsafe impl Sync for MlWeightReader {}

impl MlWeightReader {
    pub fn new(shm_path: &str) -> Self {
        info!("Initializing ML Weight Reader at {}", shm_path);
        
        // This relies on unix for shm memory.
        #[cfg(unix)]
        let ptr = {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .custom_flags(libc::O_CLOEXEC)
                .open(shm_path);
                
            match file {
                Ok(f) => {
                    let _ = f.set_len(SHM_SIZE as u64);
                    unsafe {
                        let addr = libc::mmap(
                            ptr::null_mut(),
                            SHM_SIZE,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED,
                            std::os::unix::io::AsRawFd::as_raw_fd(&f),
                            0,
                        );
                        if addr == libc::MAP_FAILED {
                            warn!("Failed to mmap ml_weights SHM, using fallback");
                            ptr::null()
                        } else {
                            addr as *const SharedMemWeights
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to open ml_weights SHM file: {}, using fallback", e);
                    ptr::null()
                }
            }
        };

        #[cfg(not(unix))]
        let ptr = {
            warn!("ML Weights SHM is not supported on non-unix, using fallback");
            ptr::null()
        };

        if !ptr.is_null() {
            unsafe {
                let shm = &*(ptr);
                if shm.magic != MAGIC_BYTES {
                    warn!("ML weights SHM magic mismatch, waiting for writer");
                }
            }
        }

        Self { shm_ptr: ptr }
    }

    pub fn get_weights(&self, symbol_id: u16) -> Option<SymbolWeight> {
        if self.shm_ptr.is_null() {
            return None;
        }

        unsafe {
            let shm = &*self.shm_ptr;
            let mut retries = 0;
            loop {
                let seq1 = shm.seqlock.load(Ordering::Acquire);
                if seq1 & 1 != 0 {
                    std::hint::spin_loop();
                    continue;
                }

                let num_symbols = shm.num_symbols as usize;
                let mut found = None;
                for i in 0..num_symbols.min(MAX_SYMBOLS) {
                    if shm.weights[i].symbol_id == symbol_id {
                        found = Some(shm.weights[i]);
                        break;
                    }
                }

                let seq2 = shm.seqlock.load(Ordering::Acquire);
                if seq1 == seq2 {
                    return found;
                }

                retries += 1;
                if retries > 1000 {
                    return None;
                }
            }
        }
    }
}
