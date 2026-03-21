//! Bridge IPC — Zero-Copy Inter-Process Communication between Rust and Python.
//!
//! # Architecture
//!
//! This module is the boundary layer between:
//!   - **Execution Container** (Rust): Market data, risk, order routing
//!   - **Brain Container** (Python): AI models, strategy, regime detection
//!
//! Communication is strictly asynchronous and non-blocking:
//!   - Rust NEVER waits for Python
//!   - Python reads Rust data at its own pace (may miss updates)
//!   - All shared memory uses seqlock or SPSC ring buffer patterns
//!
//! # Directions
//!
//! ```text
//! ┌─────────────────┐                        ┌─────────────────┐
//! │  Rust Execution  │  ── tick_broadcast ──▶ │  Python Brain   │
//! │  Container       │  ── exec_confirm ───▶ │  Container      │
//! │                  │                        │                 │
//! │                  │  ◀── portfolio_rx ──── │                 │
//! │                  │  ◀── regime_rx ─────── │                 │
//! │                  │  ◀── signal_rx ─────── │                 │
//! └─────────────────┘                        └─────────────────┘
//! ```
//!
//! # Sub-modules
//!
//! - `tick_broadcast`: Rust → Python normalized tick data via SHM ring buffer
//! - `portfolio_receiver`: Python → Rust portfolio weight targets via SHM
//! - `exec_confirm_broadcast`: Rust → Python execution confirmations via SHM
//! - `regime_adapter`: Wraps existing `regime_shm.rs` with typed interface
//! - `signal_adapter`: Wraps existing `signal_queue.rs` with typed interface
//!
//! # Memory Layout
//!
//! All SHM regions use a common header format:
//! ```text
//! [0..8]   magic: u64      = 0x5255_5354_4252_4447 ("RUSTBRDG")
//! [8..16]  version: u64    = 1
//! [16..24] sequence: u64   = monotonic counter (writer increments)
//! [24..32] timestamp_ns: u64 = last write timestamp
//! [32..N]  payload: [u8]   = type-specific data
//! ```

pub mod tick_broadcast;
pub mod portfolio_receiver;
pub mod exec_confirm_broadcast;
pub mod regime_adapter;
pub mod signal_adapter;

/// Magic number for bridge SHM headers: "RUSTBRDG" in little-endian.
pub const BRIDGE_MAGIC: u64 = 0x5255_5354_4252_4447;

/// Current bridge protocol version.
pub const BRIDGE_VERSION: u64 = 1;

/// Default SHM base path for bridge regions.
pub const BRIDGE_SHM_BASE: &str = "/dev/shm/bridge_";

/// Common SHM header (32 bytes) at the start of every bridge region.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct BridgeHeader {
    /// Magic number for validation.
    pub magic: u64,
    /// Protocol version.
    pub version: u64,
    /// Monotonically increasing sequence number (writer increments).
    pub sequence: u64,
    /// Nanosecond timestamp of the last write.
    pub timestamp_ns: u64,
}

impl BridgeHeader {
    /// Create a new header with initial values.
    pub fn new() -> Self {
        Self {
            magic: BRIDGE_MAGIC,
            version: BRIDGE_VERSION,
            sequence: 0,
            timestamp_ns: 0,
        }
    }

    /// Validate that this header has the correct magic and version.
    pub fn is_valid(&self) -> bool {
        self.magic == BRIDGE_MAGIC && self.version == BRIDGE_VERSION
    }
}

impl Default for BridgeHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in nanoseconds.
#[inline]
pub fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Bridge health status.
#[derive(Debug, Clone)]
pub struct BridgeHealth {
    /// Is the tick broadcast SHM mapped and writable?
    pub tick_broadcast_active: bool,
    /// Is the portfolio receiver SHM mapped and readable?
    pub portfolio_receiver_active: bool,
    /// Is the execution confirmation SHM mapped and writable?
    pub exec_confirm_active: bool,
    /// Is the regime reader connected?
    pub regime_reader_active: bool,
    /// Is the signal queue connected?
    pub signal_queue_active: bool,
    /// Total ticks broadcast since startup.
    pub ticks_broadcast: u64,
    /// Total portfolio updates received since startup.
    pub portfolio_updates_received: u64,
    /// Total execution confirmations sent since startup.
    pub exec_confirms_sent: u64,
}

impl Default for BridgeHealth {
    fn default() -> Self {
        Self {
            tick_broadcast_active: false,
            portfolio_receiver_active: false,
            exec_confirm_active: false,
            regime_reader_active: false,
            signal_queue_active: false,
            ticks_broadcast: 0,
            portfolio_updates_received: 0,
            exec_confirms_sent: 0,
        }
    }
}
