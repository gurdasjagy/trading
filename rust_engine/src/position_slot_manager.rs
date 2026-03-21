//! Position Slot Manager — Strict Global Position Limiter.
//!
//! Implements a strict global position limit of exactly N open trades at
//! any time (default: 3) to prevent API rate limit bans from Gate.io.
//!
//! This is implemented entirely in Rust using atomic counters — NOT Python.
//! The Rust gateway synchronously rejects or queues new signals from Python
//! until a slot is freed by a closed/filled exit order.
//!
//! # Design
//!
//! Uses a counting semaphore pattern with `AtomicU32`:
//! - `acquire()` attempts to decrement the counter (CAS loop)
//! - `release()` increments the counter
//! - If counter == 0, the slot is full and the caller is rejected
//!
//! # Thread Safety
//!
//! All operations are lock-free using `compare_exchange_weak` with
//! `Ordering::AcqRel` for the CAS and `Ordering::Acquire` for loads.
//! Safe to call from any thread.
//!
//! # Integration
//!
//! Called BEFORE the SPSC buffer push in `strategy_evaluator_loop()`:
//! ```ignore
//! if !slot_manager.try_acquire() {
//!     warn!("Position slot full — rejecting signal");
//!     continue;
//! }
//! // ... push to SPSC ...
//! // On position close:
//! slot_manager.release();
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use tracing::warn;

// ═══════════════════════════════════════════════════════════════════════════
// PositionSlotManager
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic counting semaphore for position slot management.
///
/// Guarantees that at most `max_slots` positions are open at any time.
/// Zero-overhead when not contended (single atomic CAS).
pub struct PositionSlotManager {
    /// Number of available slots. Starts at `max_slots`.
    /// Decremented on acquire, incremented on release.
    available: AtomicU32,
    /// Maximum number of concurrent positions allowed.
    max_slots: u32,
    /// Total acquire attempts (for telemetry).
    pub total_attempts: AtomicU64,
    /// Total successful acquires.
    pub total_acquired: AtomicU64,
    /// Total rejections (attempts when no slot available).
    pub total_rejected: AtomicU64,
    /// Total releases.
    pub total_released: AtomicU64,
}

impl PositionSlotManager {
    /// Create a new slot manager with the given maximum concurrent positions.
    pub fn new(max_slots: u32) -> Self {
        Self {
            available: AtomicU32::new(max_slots),
            max_slots,
            total_attempts: AtomicU64::new(0),
            total_acquired: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            total_released: AtomicU64::new(0),
        }
    }

    /// Create with the default limit of 3 concurrent positions.
    pub fn default_3_slots() -> Self {
        Self::new(3)
    }

    /// Try to acquire a position slot.
    ///
    /// Returns `true` if a slot was successfully acquired (caller may open
    /// a new position), `false` if all slots are occupied.
    ///
    /// This is a lock-free CAS loop. On the happy path (slot available)
    /// it completes in a single CAS (~10-20ns on modern x86-64).
    #[inline]
    pub fn try_acquire(&self) -> bool {
        self.total_attempts.fetch_add(1, Ordering::Relaxed);

        loop {
            let current = self.available.load(Ordering::Acquire);
            if current == 0 {
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                return false;
            }

            match self.available.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.total_acquired.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(_) => {
                    // CAS failed — retry (spurious failure or race)
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Release a position slot. Called when a position is closed/filled.
    ///
    /// # Safety
    /// The caller MUST only call release() if they previously successfully
    /// called try_acquire(). Calling release() without a prior acquire()
    /// will make available > max_slots, which is a logic error.
    #[inline]
    pub fn release(&self) {
        let prev = self.available.fetch_add(1, Ordering::Release);
        self.total_released.fetch_add(1, Ordering::Relaxed);

        // Defensive: clamp to max_slots to prevent overflow from double-release
        if prev >= self.max_slots {
            warn!("[slot-mgr] Release without acquire detected — clamping to max");
            self.available.store(self.max_slots, Ordering::Release);
        }
    }

    /// Get the number of currently available slots.
    #[inline]
    pub fn available_slots(&self) -> u32 {
        self.available.load(Ordering::Acquire)
    }

    /// Get the number of currently occupied slots (active positions).
    #[inline]
    pub fn active_positions(&self) -> u32 {
        self.max_slots.saturating_sub(self.available.load(Ordering::Acquire))
    }

    /// Get the maximum number of slots.
    #[inline]
    pub fn max_slots(&self) -> u32 {
        self.max_slots
    }

    /// Force-reset all slots to available (emergency only).
    pub fn force_reset(&self) {
        warn!("[slot-mgr] Force-resetting all {} slots to available", self.max_slots);
        self.available.store(self.max_slots, Ordering::SeqCst);
    }

    /// Get telemetry snapshot.
    pub fn get_metrics(&self) -> SlotMetrics {
        SlotMetrics {
            max_slots: self.max_slots,
            available: self.available_slots(),
            active: self.active_positions(),
            total_attempts: self.total_attempts.load(Ordering::Relaxed),
            total_acquired: self.total_acquired.load(Ordering::Relaxed),
            total_rejected: self.total_rejected.load(Ordering::Relaxed),
            total_released: self.total_released.load(Ordering::Relaxed),
        }
    }
}

/// Telemetry snapshot of the slot manager state.
#[derive(Debug, Clone)]
pub struct SlotMetrics {
    pub max_slots: u32,
    pub available: u32,
    pub active: u32,
    pub total_attempts: u64,
    pub total_acquired: u64,
    pub total_rejected: u64,
    pub total_released: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_acquire_release() {
        let mgr = PositionSlotManager::new(3);
        assert_eq!(mgr.available_slots(), 3);
        assert_eq!(mgr.active_positions(), 0);

        assert!(mgr.try_acquire());
        assert_eq!(mgr.available_slots(), 2);
        assert_eq!(mgr.active_positions(), 1);

        assert!(mgr.try_acquire());
        assert!(mgr.try_acquire());
        assert_eq!(mgr.available_slots(), 0);
        assert_eq!(mgr.active_positions(), 3);

        // 4th acquire should fail
        assert!(!mgr.try_acquire());
        assert_eq!(mgr.active_positions(), 3);

        // Release one
        mgr.release();
        assert_eq!(mgr.available_slots(), 1);

        // Now can acquire again
        assert!(mgr.try_acquire());
        assert_eq!(mgr.available_slots(), 0);
    }

    #[test]
    fn test_concurrent_access() {
        let mgr = Arc::new(PositionSlotManager::new(3));
        let mut handles = Vec::new();

        // Spawn 10 threads all trying to acquire
        for _ in 0..10 {
            let mgr_clone = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                let acquired = mgr_clone.try_acquire();
                if acquired {
                    // Hold for a bit
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    mgr_clone.release();
                }
                acquired
            }));
        }

        let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let acquired_count = results.iter().filter(|&&r| r).count();

        // At most 3 should have succeeded at any given time
        // But since they release, more than 3 total may have acquired
        assert!(acquired_count >= 3, "At least 3 should acquire");

        // After all threads done, all slots should be available
        assert_eq!(mgr.available_slots(), 3);
    }

    #[test]
    fn test_metrics() {
        let mgr = PositionSlotManager::new(2);
        assert!(mgr.try_acquire());
        assert!(mgr.try_acquire());
        assert!(!mgr.try_acquire()); // rejected

        let metrics = mgr.get_metrics();
        assert_eq!(metrics.max_slots, 2);
        assert_eq!(metrics.available, 0);
        assert_eq!(metrics.active, 2);
        assert_eq!(metrics.total_attempts, 3);
        assert_eq!(metrics.total_acquired, 2);
        assert_eq!(metrics.total_rejected, 1);
    }

    #[test]
    fn test_force_reset() {
        let mgr = PositionSlotManager::new(3);
        assert!(mgr.try_acquire());
        assert!(mgr.try_acquire());
        assert_eq!(mgr.available_slots(), 1);

        mgr.force_reset();
        assert_eq!(mgr.available_slots(), 3);
    }
}

