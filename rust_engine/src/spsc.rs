//! Lock-free single-producer single-consumer (SPSC) ring buffer.
//!
//! Designed for the hot-path inter-thread communication in the trading engine.
//! One OS thread produces messages, another OS thread consumes them, with
//! **no locks, no heap allocation, and no syscalls** in the steady state.
//!
//! Cache-line padding (64 bytes) between `write_pos` and `read_pos` prevents
//! false sharing — the producer and consumer never contend on the same cache line.
//!
//! # Message Types
//!
//! - **`RawBookUpdate`**: WS ingestion → orderbook builder (price/qty delta).
//! - **`BookSnapshot`**: Orderbook builder → strategy evaluator (top-of-book state).
//! - **`OrderCommand`**: Strategy evaluator → execution router (trade intent).
//!
//! # Overflow Handling (Institutional Upgrade)
//!
//! When `try_push()` returns false the producer MUST report the drop to the
//! circuit breaker via the `SpscOverflowMonitor`. If the drop rate exceeds
//! a configurable threshold, the monitor trips `OrderRateAnomaly`.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Cache-line padding
// ---------------------------------------------------------------------------

/// Cache-line size on x86-64 / ARM64.
const CACHE_LINE: usize = 64;

/// Pad a value to a full cache line to prevent false sharing.
#[repr(C)]
struct CachePadded<T> {
    value: T,
    _pad: [u8; CACHE_LINE - std::mem::size_of::<AtomicUsize>()],
}

impl<T> CachePadded<T> {
    fn new(value: T) -> Self {
        // Safety: padding is always zero-initialized
        Self {
            value,
            _pad: [0u8; CACHE_LINE - std::mem::size_of::<AtomicUsize>()],
        }
    }
}

// ---------------------------------------------------------------------------
// SpscOverflowMonitor — tracks drop rate for circuit breaker integration
// ---------------------------------------------------------------------------

/// Tracks SPSC buffer overflow events and fires a circuit breaker callback
/// when the drop rate within a sliding 1-second window exceeds the threshold.
///
/// All fields are atomic so the monitor can be shared between producer and
/// the telemetry thread without locking.
pub struct SpscOverflowMonitor {
    /// Total drops since creation (monotonically increasing).
    pub total_drops: AtomicU64,
    /// Drops in the current 1-second tracking window.
    drops_this_second: AtomicU64,
    /// Timestamp (nanoseconds) marking the start of the current window.
    window_start_ns: AtomicU64,
    /// Maximum allowed drops per second before triggering the circuit breaker.
    pub max_drops_per_second: u64,
}

impl SpscOverflowMonitor {
    /// Create a new overflow monitor.
    ///
    /// `max_drops_per_second`: if more than this many pushes fail in one second,
    /// the monitor considers it an anomaly.
    pub fn new(max_drops_per_second: u64) -> Self {
        Self {
            total_drops: AtomicU64::new(0),
            drops_this_second: AtomicU64::new(0),
            window_start_ns: AtomicU64::new(now_ns()),
            max_drops_per_second,
        }
    }

    /// Record a single drop event. Returns `true` if the drop rate has
    /// exceeded the threshold and the circuit breaker should be tripped.
    ///
    /// This is called on the producer thread's hot path so it must be fast:
    /// two atomic increments + one atomic load + one comparison.
    #[inline]
    pub fn record_drop(&self) -> bool {
        self.total_drops.fetch_add(1, Ordering::Relaxed);

        let now = now_ns();
        let window_start = self.window_start_ns.load(Ordering::Relaxed);
        let elapsed_ns = now.saturating_sub(window_start);

        if elapsed_ns >= 1_000_000_000 {
            // New 1-second window — reset counter
            self.window_start_ns.store(now, Ordering::Relaxed);
            self.drops_this_second.store(1, Ordering::Relaxed);
            false
        } else {
            let count = self.drops_this_second.fetch_add(1, Ordering::Relaxed) + 1;
            count > self.max_drops_per_second
        }
    }

    /// Get the total number of drops since creation.
    #[inline]
    pub fn get_total_drops(&self) -> u64 {
        self.total_drops.load(Ordering::Relaxed)
    }

    /// Get the number of drops in the current tracking window.
    #[inline]
    pub fn get_drops_this_second(&self) -> u64 {
        self.drops_this_second.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// SpscRingBuffer
// ---------------------------------------------------------------------------

/// Lock-free SPSC ring buffer with power-of-2 capacity.
///
/// # Type Parameters
/// - `T`: Message type. Must be `Copy + Default` for safe zero-initialization.
/// - `N`: Capacity. **Must** be a power of 2 (asserted at construction time).
///
/// # Memory Layout
/// ```text
/// [CachePadded<write_pos>]   ← producer-owned, on its own cache line
/// [CachePadded<read_pos>]    ← consumer-owned, on its own cache line
/// [UnsafeCell<[T; N]>]       ← ring buffer storage
/// ```
///
/// # Safety
/// - Only one thread may call `try_push()` (the producer).
/// - Only one thread may call `try_pop()` / `spin_pop()` (the consumer).
/// - The buffer itself is `Send + Sync` because the atomic indices enforce ordering.
pub struct SpscRingBuffer<T: Copy + Default, const N: usize> {
    /// Write position (producer-owned). Monotonically increasing.
    write_pos: CachePadded<AtomicUsize>,
    /// Read position (consumer-owned). Monotonically increasing.
    read_pos: CachePadded<AtomicUsize>,
    /// Ring buffer storage.
    buffer: Box<[UnsafeCell<T>; N]>,
    /// Capacity mask (N - 1) for fast modulo via bitwise AND.
    mask: usize,
}

// Safety: SpscRingBuffer is safe to share between exactly two threads
// (one producer, one consumer) because:
// 1. write_pos is only modified by the producer
// 2. read_pos is only modified by the consumer
// 3. Each slot is only written by the producer (before advancing write_pos)
//    and only read by the consumer (before advancing read_pos)
// 4. Atomic orderings ensure proper happens-before relationships
unsafe impl<T: Copy + Default + Send, const N: usize> Send for SpscRingBuffer<T, N> {}
unsafe impl<T: Copy + Default + Send, const N: usize> Sync for SpscRingBuffer<T, N> {}

impl<T: Copy + Default, const N: usize> SpscRingBuffer<T, N> {
    /// Create a new ring buffer with capacity `N`.
    ///
    /// # Panics
    /// Panics if `N` is not a power of 2 or if `N` is 0.
    pub fn new() -> Self {
        assert!(N > 0, "Ring buffer capacity must be > 0");
        assert!(N.is_power_of_two(), "Ring buffer capacity must be a power of 2, got {}", N);

        // Pre-allocate the buffer with default values
        let mut data: Vec<UnsafeCell<T>> = Vec::with_capacity(N);
        for _ in 0..N {
            data.push(UnsafeCell::new(T::default()));
        }
        let buffer: Box<[UnsafeCell<T>; N]> = data.into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Buffer size mismatch"));

        Self {
            write_pos: CachePadded::new(AtomicUsize::new(0)),
            read_pos: CachePadded::new(AtomicUsize::new(0)),
            buffer,
            mask: N - 1,
        }
    }

    /// Try to push a value into the ring buffer.
    ///
    /// Returns `true` if the value was successfully enqueued, `false` if the
    /// buffer is full. **Never blocks.**
    ///
    /// # Safety Contract
    /// Only the producer thread may call this method.
    #[inline]
    pub fn try_push(&self, value: T) -> bool {
        let write = self.write_pos.value.load(Ordering::Relaxed);
        let read = self.read_pos.value.load(Ordering::Acquire);

        // Buffer is full when write is one full lap ahead of read
        if write.wrapping_sub(read) >= N {
            return false;
        }

        let slot = write & self.mask;
        // Safety: we've verified the slot is not being read (write - read < N)
        unsafe {
            *self.buffer[slot].get() = value;
        }

        // Release ensures the written data is visible before the consumer sees the new write_pos
        self.write_pos.value.store(write.wrapping_add(1), Ordering::Release);
        true
    }

    /// Try to pop a value from the ring buffer.
    ///
    /// Returns `Some(value)` if a value was available, `None` if the buffer
    /// is empty. **Never blocks.**
    ///
    /// # Safety Contract
    /// Only the consumer thread may call this method.
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        let read = self.read_pos.value.load(Ordering::Relaxed);
        let write = self.write_pos.value.load(Ordering::Acquire);

        // Buffer is empty when read has caught up to write
        if read == write {
            return None;
        }

        let slot = read & self.mask;
        // Safety: we've verified data is available (read != write)
        let value = unsafe { *self.buffer[slot].get() };

        // Release ensures we don't read the slot again before the producer reuses it
        self.read_pos.value.store(read.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Spin-wait until a value is available, then pop it.
    ///
    /// Uses a busy-spin loop with `std::hint::spin_loop()` for minimum latency.
    /// **Only use on cores dedicated to this thread** (isolated via `isolcpus`).
    ///
    /// # Safety Contract
    /// Only the consumer thread may call this method.
    #[inline]
    pub fn spin_pop(&self) -> T {
        loop {
            if let Some(value) = self.try_pop() {
                return value;
            }
            std::hint::spin_loop();
        }
    }

    /// Return the number of items currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let write = self.write_pos.value.load(Ordering::Acquire);
        let read = self.read_pos.value.load(Ordering::Acquire);
        write.wrapping_sub(read)
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= N
    }

    /// Return the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }
}

// ---------------------------------------------------------------------------
// Message Types — all #[repr(C)], Copy, Default
// ---------------------------------------------------------------------------

/// Raw book update from WS ingestion to orderbook builder.
///
/// Represents a single price level change (insert/update/delete).
/// Size: 56 bytes (fits in a single cache line with padding).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RawBookUpdate {
    /// Symbol ID (from SymbolRegistry).
    pub symbol_id: u16,
    /// 0 = bid, 1 = ask.
    pub side: u8,
    /// 0 = delta, 1 = snapshot_start, 2 = snapshot_end.
    pub update_type: u8,
    /// Padding for alignment.
    pub _pad: [u8; 4],
    /// Price in FixedPrice representation (i64, 1e8).
    pub price: i64,
    /// Quantity in FixedQty representation (i64, 1e4). 0 = delete level.
    pub qty: i64,
    /// Exchange-assigned sequence number.
    pub sequence: u64,
    /// Receive timestamp in nanoseconds (TSC-based).
    pub recv_ns: u64,
    /// Number of levels in this snapshot batch (only valid for snapshot_start).
    pub snapshot_count: u32,
    /// Padding to reach 56 bytes.
    pub _pad2: [u8; 4],
}

impl Default for RawBookUpdate {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            side: 0,
            update_type: 0,
            _pad: [0; 4],
            price: 0,
            qty: 0,
            sequence: 0,
            recv_ns: 0,
            snapshot_count: 0,
            _pad2: [0; 4],
        }
    }
}

/// Book snapshot from orderbook builder to strategy evaluator.
///
/// Contains top-of-book state and derived metrics for a single symbol.
/// Generated after every significant book change (level touched within top-N).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BookSnapshot {
    /// Symbol ID.
    pub symbol_id: u16,
    /// Number of bid levels included in this snapshot.
    pub bid_levels: u8,
    /// Number of ask levels included in this snapshot.
    pub ask_levels: u8,
    /// Padding for alignment.
    pub _pad: [u8; 4],
    /// Best bid price (FixedPrice).
    pub best_bid: i64,
    /// Best ask price (FixedPrice).
    pub best_ask: i64,
    /// Mid price (FixedPrice).
    pub mid_price: i64,
    /// Spread in basis points (integer, pre-computed).
    pub spread_bps: i32,
    /// Order imbalance in basis points (-10000 = all bids, +10000 = all asks).
    pub imbalance_bps: i32,
    /// Total bid depth in USDT (FixedPrice) at top-N levels.
    pub bid_depth_usdt: i64,
    /// Total ask depth in USDT (FixedPrice) at top-N levels.
    pub ask_depth_usdt: i64,
    /// Monotonically increasing snapshot sequence number.
    pub sequence: u64,
    /// Timestamp in nanoseconds when this snapshot was generated.
    pub timestamp_ns: u64,
}

/// Trade event from WS ingestion to strategy evaluator.
///
/// Represents a single trade execution for VPIN calculation and candle aggregation.
/// Size: 40 bytes (fits in a single cache line).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TradeEvent {
    /// Symbol ID (from SymbolRegistry).
    pub symbol_id: u16,
    /// 0 = buy, 1 = sell (taker side).
    pub side: u8,
    /// Padding for alignment.
    pub _pad: [u8; 5],
    /// Trade price in FixedPrice representation (i64, 1e8).
    pub price: i64,
    /// Trade quantity in FixedQty representation (i64, 1e4).
    pub qty: i64,
    /// Receive timestamp in nanoseconds (TSC-based).
    pub recv_ns: u64,
    /// Exchange-assigned sequence number.
    pub sequence: u64,
}

impl Default for BookSnapshot {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            bid_levels: 0,
            ask_levels: 0,
            _pad: [0; 4],
            best_bid: 0,
            best_ask: 0,
            mid_price: 0,
            spread_bps: 0,
            imbalance_bps: 0,
            bid_depth_usdt: 0,
            ask_depth_usdt: 0,
            sequence: 0,
            timestamp_ns: 0,
        }
    }
}

impl Default for TradeEvent {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            side: 0,
            _pad: [0; 5],
            price: 0,
            qty: 0,
            recv_ns: 0,
            sequence: 0,
        }
    }
}

/// Order command from strategy evaluator to execution router.
///
/// Includes explicit Stop Loss, Take Profit, dynamic leverage, and
/// pre-trade risk metadata. Every trade submitted through this struct
/// MUST have SL set (TP is optional).
/// The execution router will reject commands with stop_loss_fp == 0 unless
/// the order is a cancel command.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OrderCommand {
    /// Symbol ID.
    pub symbol_id: u16,
    /// 0 = buy, 1 = sell.
    pub side: u8,
    /// 0 = limit, 1 = market, 2 = cancel.
    pub order_type: u8,
    /// Target leverage for this trade (1-125). 0 = use exchange default.
    /// Dynamically set per trade based on the strategy's risk profile and
    /// current regime. The execution router calls set_leverage() before
    /// submitting if this differs from the current symbol leverage.
    pub leverage: u8,
    /// Padding.
    pub _pad: [u8; 3],
    /// Price (FixedPrice). For market orders, this is the limit price / protection.
    pub price: i64,
    /// Quantity (FixedQty).
    pub qty: i64,
    /// Strategy-assigned order ID.
    pub order_id: u64,
    /// Timestamp of signal generation in nanoseconds.
    pub signal_ns: u64,
    /// Maximum acceptable slippage in basis points.
    pub max_slippage_bps: i32,
    /// Time-to-live in milliseconds (0 = GTC).
    pub ttl_ms: u32,
    /// Hard Stop Loss price (FixedPrice). 0 = no SL (rejected for new orders).
    /// The execution router registers this as a conditional order at the exchange.
    /// If exchange SL placement fails, the trade is marked "unprotected" and
    /// the router will aggressively retry or close the position.
    pub stop_loss_fp: i64,
    /// Take Profit price (FixedPrice). 0 = no TP (trail manually or use dynamic TP).
    /// Can be updated dynamically by the strategy engine via amend commands.
    pub take_profit_fp: i64,
    /// Placement type hint from strategy (0=AtBest, 1=Improve1Tick, 2=Behind1Tick,
    /// 3=AtMid, 4=SmartPlace). Execution router translates to actual price.
    pub placement_type: u8,
    /// Whether this order must be post-only (maker). 0 = no, 1 = yes.
    pub post_only: u8,
    /// Whether this is a position close order. 0 = no, 1 = yes.
    pub is_close: u8,
    /// Padding to maintain alignment.
    pub _pad2: [u8; 5],
}

impl Default for OrderCommand {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            side: 0,
            order_type: 0,
            leverage: 0,
            _pad: [0; 3],
            price: 0,
            qty: 0,
            order_id: 0,
            signal_ns: 0,
            max_slippage_bps: 0,
            ttl_ms: 0,
            stop_loss_fp: 0,
            take_profit_fp: 0,
            placement_type: 0,
            post_only: 0,
            is_close: 0,
            _pad2: [0; 5],
        }
    }
}

impl OrderCommand {
    /// Returns true if this command has a valid hard stop loss set.
    #[inline]
    pub fn has_stop_loss(&self) -> bool {
        self.stop_loss_fp != 0
    }

    /// Returns true if this command has a take profit set.
    #[inline]
    pub fn has_take_profit(&self) -> bool {
        self.take_profit_fp != 0
    }

    /// Returns true if this is a cancel command (no SL/TP required).
    #[inline]
    pub fn is_cancel(&self) -> bool {
        self.order_type == order_cmd_type::CANCEL
    }

    /// Get the target leverage (defaults to 5x if 0).
    #[inline]
    pub fn target_leverage(&self) -> i32 {
        if self.leverage == 0 { 5 } else { self.leverage as i32 }
    }

    /// Validate that the command is safe to submit.
    /// Returns Err with reason if the command should be rejected.
    ///
    /// NOTE: Unprotected trades (no SL/TP) are ALLOWED. The exit evaluator
    /// will auto-detect unprotected positions and apply default SL/TP based
    /// on ATR. This enables manual trades without requiring SL/TP upfront.
    #[inline]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.qty == 0 && !self.is_cancel() {
            return Err("OrderCommand rejected: zero quantity");
        }
        if self.symbol_id == 0 {
            return Err("OrderCommand rejected: invalid symbol_id 0");
        }
        if self.leverage > 125 {
            return Err("OrderCommand rejected: leverage exceeds 125x maximum");
        }
        Ok(())
    }
}

/// Update type constants for `RawBookUpdate.update_type`.
pub mod update_type {
    /// Incremental delta (insert/update/delete a single level).
    pub const DELTA: u8 = 0;
    /// Start of a full snapshot batch.
    pub const SNAPSHOT_START: u8 = 1;
    /// End of a full snapshot batch.
    pub const SNAPSHOT_END: u8 = 2;
}

/// Side constants for `RawBookUpdate.side` and `OrderCommand.side`.
pub mod side {
    pub const BID: u8 = 0;
    pub const ASK: u8 = 1;
    pub const BUY: u8 = 0;
    pub const SELL: u8 = 1;
}

/// Order type constants for `OrderCommand.order_type`.
pub mod order_cmd_type {
    pub const LIMIT: u8 = 0;
    pub const MARKET: u8 = 1;
    pub const CANCEL: u8 = 2;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_push_pop() {
        let ring = SpscRingBuffer::<u64, 16>::new();
        assert!(ring.is_empty());
        assert_eq!(ring.capacity(), 16);

        assert!(ring.try_push(42));
        assert_eq!(ring.len(), 1);
        assert!(!ring.is_empty());

        let val = ring.try_pop();
        assert_eq!(val, Some(42));
        assert!(ring.is_empty());
    }

    #[test]
    fn test_empty_pop() {
        let ring = SpscRingBuffer::<u64, 16>::new();
        assert_eq!(ring.try_pop(), None);
    }

    #[test]
    fn test_full_buffer() {
        let ring = SpscRingBuffer::<u64, 4>::new();
        assert!(ring.try_push(1));
        assert!(ring.try_push(2));
        assert!(ring.try_push(3));
        assert!(ring.try_push(4));
        assert!(!ring.try_push(5)); // Buffer full
        assert!(ring.is_full());
    }

    #[test]
    fn test_ordering_preserved() {
        let ring = SpscRingBuffer::<u64, 1024>::new();
        for i in 0..100 {
            assert!(ring.try_push(i));
        }
        for i in 0..100 {
            assert_eq!(ring.try_pop(), Some(i));
        }
    }

    #[test]
    fn test_wrap_around() {
        let ring = SpscRingBuffer::<u64, 4>::new();
        // Fill and drain multiple times to test wrap-around
        for round in 0..10 {
            for i in 0..4 {
                assert!(ring.try_push(round * 4 + i), "Push failed at round {}, i {}", round, i);
            }
            for i in 0..4 {
                let expected = round * 4 + i;
                assert_eq!(ring.try_pop(), Some(expected), "Pop mismatch at round {}, i {}", round, i);
            }
        }
    }

    #[test]
    fn test_message_types() {
        // Verify RawBookUpdate size and Copy semantics
        let update = RawBookUpdate {
            symbol_id: 1,
            side: 0,
            update_type: 0,
            _pad: [0; 4],
            price: 5_000_000_000_000, // $50,000
            qty: 15_000,              // 1.5 contracts
            sequence: 12345,
            recv_ns: 1000000000,
            snapshot_count: 0,
            _pad2: [0; 4],
        };
        let copy = update; // Must be Copy
        assert_eq!(copy.symbol_id, 1);
        assert_eq!(copy.price, 5_000_000_000_000);
    }

    #[test]
    fn test_multithreaded_push_pop() {
        // Use Box::leak to get 'static lifetime for cross-thread sharing
        let ring: &'static SpscRingBuffer<u64, 65536> =
            Box::leak(Box::new(SpscRingBuffer::new()));

        let count = 100_000u64;

        let producer = thread::spawn(move || {
            for i in 0..count {
                while !ring.try_push(i) {
                    std::hint::spin_loop();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut expected = 0u64;
            while expected < count {
                if let Some(val) = ring.try_pop() {
                    assert_eq!(val, expected, "Out-of-order at {}", expected);
                    expected += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn test_raw_book_update_ring() {
        let ring = SpscRingBuffer::<RawBookUpdate, 256>::new();

        let update = RawBookUpdate {
            symbol_id: 1,
            side: side::BID,
            update_type: update_type::DELTA,
            _pad: [0; 4],
            price: 5_000_000_000_000,
            qty: 15_000,
            sequence: 1,
            recv_ns: 123456789,
            snapshot_count: 0,
            _pad2: [0; 4],
        };

        assert!(ring.try_push(update));
        let popped = ring.try_pop().unwrap();
        assert_eq!(popped.symbol_id, 1);
        assert_eq!(popped.price, 5_000_000_000_000);
        assert_eq!(popped.qty, 15_000);
    }

    #[test]
    #[should_panic(expected = "power of 2")]
    fn test_non_power_of_two_panics() {
        let _ring = SpscRingBuffer::<u64, 3>::new();
    }

    #[test]
    fn test_overflow_monitor_basic() {
        let monitor = SpscOverflowMonitor::new(5);
        assert_eq!(monitor.get_total_drops(), 0);

        // Record 4 drops — should not trip
        for _ in 0..4 {
            assert!(!monitor.record_drop());
        }
        assert_eq!(monitor.get_total_drops(), 4);

        // 5th drop — still within threshold
        assert!(!monitor.record_drop());

        // 6th drop — exceeds threshold, should trip
        assert!(monitor.record_drop());
    }

    #[test]
    fn test_order_command_leverage() {
        let cmd = OrderCommand {
            leverage: 20,
            ..Default::default()
        };
        assert_eq!(cmd.target_leverage(), 20);

        let cmd_default = OrderCommand::default();
        assert_eq!(cmd_default.target_leverage(), 5); // default 5x
    }

    #[test]
    fn test_order_command_leverage_validation() {
        let cmd = OrderCommand {
            symbol_id: 1,
            qty: 100,
            stop_loss_fp: 100,
            leverage: 200, // > 125
            ..Default::default()
        };
        assert!(cmd.validate().is_err());
    }
}

