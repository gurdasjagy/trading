//! Market-By-Order (MBO) book tracking — DashMap Concurrent Upgrade.
//!
//! Maintains a per-price-level queue of individual orders (FIFO), enabling
//! precise queue position estimation for our resting orders.
//!
//! # DashMap Upgrade (Phase 2)
//!
//! The bid_levels and ask_levels now use `DashMap<i64, MboLevel>` instead of
//! `HashMap`, allowing concurrent reads/writes from multiple threads without
//! lock contention. This is critical for the execution container where:
//!   - The WS ingestion thread writes order events
//!   - The strategy thread reads depth/queue state
//!   - The execution thread reads queue positions for our orders
//!
//! # Data Sources
//!
//! - **Gate.io**: `futures.order_book_update` channel provides individual order
//!   events with order ID, price, size, and action (add/modify/delete).
//! - Synthesised from L2 deltas + trade tape
//!   (approximation via `SyntheticQueueTracker` in microstructure.rs).
//!
//! # Integration
//!
//! `MboBook` owns the ground truth of individual order positions.
//! When tracking our own orders, it computes:
//!   - `queue_ahead`: volume ahead of us at the same price level
//!   - `fill_probability`: estimated probability of getting filled
//!   - `estimated_ttf_s`: estimated time-to-fill in seconds

use std::collections::HashMap;
use dashmap::DashMap;
use parking_lot::RwLock;

// ═══════════════════════════════════════════════════════════════════════════
// MBO Event Types
// ═══════════════════════════════════════════════════════════════════════════

/// Action type for MBO events from the exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MboAction {
    /// New order added to the book.
    Add,
    /// Existing order modified (size change).
    Modify,
    /// Order removed from the book (filled or cancelled).
    Delete,
}

/// A single MBO event received from the exchange feed.
#[derive(Debug, Clone, Copy)]
pub struct MboEvent {
    /// Exchange-assigned order ID.
    pub order_id: u64,
    /// Price in FixedPrice representation.
    pub price_fp: i64,
    /// Quantity (remaining for Add/Modify, filled for Delete).
    pub qty: i64,
    /// Action: add, modify, delete.
    pub action: MboAction,
    /// Side: 0 = bid, 1 = ask.
    pub side: u8,
    /// Nanosecond timestamp when we received this event.
    pub timestamp_ns: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// MboOrder — individual order tracked in the MBO book
// ═══════════════════════════════════════════════════════════════════════════

/// Individual order tracked in MBO.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MboOrder {
    /// Exchange-assigned order ID.
    pub order_id: u64,
    /// Price in FixedPrice representation.
    pub price_fp: i64,
    /// Remaining quantity at this order.
    pub remaining_qty: i64,
    /// Original quantity at placement.
    pub original_qty: i64,
    /// Nanosecond timestamp when we first saw this order.
    pub timestamp_ns: u64,
    /// Side: 0 = bid, 1 = ask.
    pub side: u8,
    /// Padding for alignment.
    pub _pad: [u8; 7],
}

// ═══════════════════════════════════════════════════════════════════════════
// MboLevel — per-price-level order queue
// ═══════════════════════════════════════════════════════════════════════════

/// Per-price-level MBO tracker.
///
/// Maintains the queue of individual orders at each price level in FIFO order.
/// This allows us to know EXACTLY where our order is in the queue.
pub struct MboLevel {
    /// Orders at this level, in FIFO order (front = first to fill).
    orders: Vec<MboOrder>,
    /// Total remaining quantity at this level.
    total_qty: i64,
}

impl MboLevel {
    pub fn new() -> Self {
        Self {
            orders: Vec::with_capacity(64),
            total_qty: 0,
        }
    }

    /// Add an order to the back of the queue.
    pub fn add_order(&mut self, order: MboOrder) {
        self.total_qty += order.remaining_qty;
        self.orders.push(order);
    }

    /// Remove an order by ID. Returns the removed order if found.
    pub fn remove_order(&mut self, order_id: u64) -> Option<MboOrder> {
        if let Some(pos) = self.orders.iter().position(|o| o.order_id == order_id) {
            let removed = self.orders.remove(pos);
            self.total_qty -= removed.remaining_qty;
            Some(removed)
        } else {
            None
        }
    }

    /// Remove an order by ID and return its queue position.
    pub fn remove_order_with_pos(&mut self, order_id: u64) -> Option<(MboOrder, usize)> {
        if let Some(pos) = self.orders.iter().position(|o| o.order_id == order_id) {
            let removed = self.orders.remove(pos);
            self.total_qty -= removed.remaining_qty;
            Some((removed, pos))
        } else {
            None
        }
    }

    /// Modify an order's remaining quantity.
    pub fn modify_order(&mut self, order_id: u64, new_qty: i64) -> bool {
        if let Some(order) = self.orders.iter_mut().find(|o| o.order_id == order_id) {
            let old_qty = order.remaining_qty;
            order.remaining_qty = new_qty;
            self.total_qty += new_qty - old_qty;
            true
        } else {
            false
        }
    }

    /// Get total remaining quantity at this level.
    #[inline]
    pub fn total_qty(&self) -> i64 {
        self.total_qty
    }

    /// Get number of orders at this level.
    #[inline]
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Get the volume ahead of the order at position `pos`.
    pub fn volume_ahead(&self, pos: usize) -> i64 {
        self.orders[..pos].iter().map(|o| o.remaining_qty).sum()
    }

    /// Find the position (index) of an order by ID.
    pub fn find_position(&self, order_id: u64) -> Option<usize> {
        self.orders.iter().position(|o| o.order_id == order_id)
    }

    /// Check if the level is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
}

impl Default for MboLevel {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OurOrderState — tracking state for our own resting orders
// ═══════════════════════════════════════════════════════════════════════════

/// State of one of our own orders being tracked in the MBO book.
pub struct OurOrderState {
    /// Exchange-assigned order ID.
    pub order_id: u64,
    /// Price in FixedPrice representation.
    pub price_fp: i64,
    /// Side: 0 = bid, 1 = ask.
    pub side: u8,
    /// Our order's size.
    pub our_size: i64,
    /// Volume ahead of us in the queue.
    pub queue_ahead: i64,
    /// Estimated fill probability [0, 1].
    pub fill_probability: f64,
    /// Estimated time-to-fill in seconds.
    pub estimated_ttf_s: f64,
    /// Depletion rate EMA (volume/second consumed from the front).
    depletion_ema: f64,
    /// Last update timestamp in nanoseconds.
    last_update_ns: u64,
}

impl OurOrderState {
    pub fn new(order_id: u64, price_fp: i64, side: u8, our_size: i64, queue_ahead: i64) -> Self {
        Self {
            order_id,
            price_fp,
            side,
            our_size,
            queue_ahead,
            fill_probability: 0.0,
            estimated_ttf_s: f64::INFINITY,
            depletion_ema: 0.0,
            last_update_ns: now_ns(),
        }
    }

    /// Recalculate fill probability and TTF based on current queue state.
    fn recalculate(&mut self, total_qty_at_level: i64) {
        if total_qty_at_level <= 0 {
            self.fill_probability = 1.0;
            self.estimated_ttf_s = 0.0;
            return;
        }

        self.fill_probability =
            1.0 - (self.queue_ahead as f64 / total_qty_at_level as f64).clamp(0.0, 1.0);

        if self.depletion_ema > 0.0 && self.queue_ahead > 0 {
            self.estimated_ttf_s = self.queue_ahead as f64 / self.depletion_ema;
        } else if self.queue_ahead <= 0 {
            self.estimated_ttf_s = 0.0;
        } else {
            self.estimated_ttf_s = f64::INFINITY;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MboBook — DashMap-backed concurrent MBO book per symbol
// ═══════════════════════════════════════════════════════════════════════════

/// Full MBO book for a single symbol using DashMap for concurrent access.
///
/// # Concurrency Model
///
/// - `bid_levels` and `ask_levels`: DashMap allows concurrent reads from
///   strategy/execution threads while the WS ingestion thread writes.
///   DashMap uses sharded internal locking — different price levels can
///   be accessed concurrently without contention.
///
/// - `our_orders`: Uses parking_lot::RwLock since it's accessed less
///   frequently and needs consistent reads across multiple entries.
///
/// - `event_count`: Atomic for lock-free telemetry reads.
pub struct MboBook {
    /// Bid levels indexed by price_fp — DashMap for concurrent access.
    bid_levels: DashMap<i64, MboLevel>,
    /// Ask levels indexed by price_fp — DashMap for concurrent access.
    ask_levels: DashMap<i64, MboLevel>,
    /// Our own orders tracked by order_id — RwLock for consistent multi-key reads.
    our_orders: RwLock<HashMap<u64, OurOrderState>>,
    /// Total number of MBO events processed.
    event_count: std::sync::atomic::AtomicU64,
}

impl MboBook {
    pub fn new() -> Self {
        Self {
            bid_levels: DashMap::with_capacity(256),
            ask_levels: DashMap::with_capacity(256),
            our_orders: RwLock::new(HashMap::with_capacity(16)),
            event_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get the levels map for a given side.
    #[inline]
    fn levels(&self, side: u8) -> &DashMap<i64, MboLevel> {
        if side == 0 { &self.bid_levels } else { &self.ask_levels }
    }

    /// Process an MBO event from the exchange.
    pub fn on_mbo_event(&self, event: &MboEvent) {
        self.event_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        match event.action {
            MboAction::Add => {
                let levels = self.levels(event.side);
                let mut level = levels.entry(event.price_fp).or_default();
                level.add_order(MboOrder {
                    order_id: event.order_id,
                    price_fp: event.price_fp,
                    remaining_qty: event.qty,
                    original_qty: event.qty,
                    timestamp_ns: event.timestamp_ns,
                    side: event.side,
                    _pad: [0; 7],
                });
            }

            MboAction::Delete => {
                let levels = self.levels(event.side);
                let mut found_pos: Option<usize> = None;
                let mut level_empty = false;

                // Scope the DashMap guard
                if let Some(mut level) = levels.get_mut(&event.price_fp) {
                    if let Some((_, pos)) = level.remove_order_with_pos(event.order_id) {
                        found_pos = Some(pos);
                    }
                    level_empty = level.is_empty();
                }

                // Update our queue positions if an order was removed
                if found_pos.is_some() {
                    self.update_our_queue_positions(event.price_fp, event.side);
                }

                // Clean up empty levels
                if level_empty {
                    levels.remove(&event.price_fp);
                }
            }

            MboAction::Modify => {
                let levels = self.levels(event.side);
                if let Some(mut level) = levels.get_mut(&event.price_fp) {
                    level.modify_order(event.order_id, event.qty);
                }
                // Recalculate our queue positions at this price
                self.recalculate_our_positions_at_price(event.price_fp, event.side);
            }
        }
    }

    /// Register one of our own orders for queue position tracking.
    pub fn track_our_order(
        &self,
        order_id: u64,
        price_fp: i64,
        side: u8,
        our_size: i64,
    ) {
        let levels = self.levels(side);
        let queue_ahead = levels
            .get(&price_fp)
            .map(|level| level.total_qty())
            .unwrap_or(0);

        let mut orders = self.our_orders.write();
        orders.insert(
            order_id,
            OurOrderState::new(order_id, price_fp, side, our_size, queue_ahead),
        );
    }

    /// Remove a tracked order (filled or cancelled).
    pub fn untrack_our_order(&self, order_id: u64) {
        let mut orders = self.our_orders.write();
        orders.remove(&order_id);
    }

    /// Get queue position info for one of our orders.
    pub fn get_queue_position(&self, order_id: u64) -> Option<(i64, f64, f64)> {
        let orders = self.our_orders.read();
        orders.get(&order_id).map(|state| {
            (state.queue_ahead, state.fill_probability, state.estimated_ttf_s)
        })
    }

    /// Get our order state snapshot by order_id.
    /// Returns a tuple of (queue_ahead, fill_probability, estimated_ttf_s, our_size).
    pub fn get_our_order_snapshot(&self, order_id: u64) -> Option<(i64, f64, f64, i64)> {
        let orders = self.our_orders.read();
        orders.get(&order_id).map(|state| {
            (state.queue_ahead, state.fill_probability, state.estimated_ttf_s, state.our_size)
        })
    }

    /// Get the total depth at a price level.
    pub fn depth_at_price(&self, price_fp: i64, side: u8) -> i64 {
        let levels = self.levels(side);
        levels.get(&price_fp).map(|l| l.total_qty()).unwrap_or(0)
    }

    /// Get total event count (useful for telemetry).
    #[inline]
    pub fn event_count(&self) -> u64 {
        self.event_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the number of tracked order levels (bid side).
    pub fn bid_level_count(&self) -> usize {
        self.bid_levels.len()
    }

    /// Get the number of tracked order levels (ask side).
    pub fn ask_level_count(&self) -> usize {
        self.ask_levels.len()
    }

    /// Get the total number of tracked orders across all levels (bid side).
    pub fn bid_order_count(&self) -> usize {
        self.bid_levels.iter().map(|e| e.value().order_count()).sum()
    }

    /// Get the total number of tracked orders across all levels (ask side).
    pub fn ask_order_count(&self) -> usize {
        self.ask_levels.iter().map(|e| e.value().order_count()).sum()
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    /// Update queue positions for our orders when an order at a price/side is removed.
    fn update_our_queue_positions(&self, price_fp: i64, side: u8) {
        let levels = self.levels(side);
        let total_qty = levels
            .get(&price_fp)
            .map(|l| l.total_qty())
            .unwrap_or(0);

        let mut orders = self.our_orders.write();
        for our_order in orders.values_mut() {
            if our_order.price_fp == price_fp && our_order.side == side {
                let now = now_ns();
                let elapsed_s = (now - our_order.last_update_ns) as f64 / 1e9;

                if our_order.queue_ahead > 0 {
                    let old_ahead = our_order.queue_ahead;
                    our_order.queue_ahead = our_order.queue_ahead.saturating_sub(1);

                    // Update depletion EMA
                    if elapsed_s > 0.0 {
                        let rate = (old_ahead - our_order.queue_ahead) as f64 / elapsed_s;
                        let alpha = 0.1;
                        our_order.depletion_ema =
                            alpha * rate + (1.0 - alpha) * our_order.depletion_ema;
                    }
                }

                our_order.last_update_ns = now;
                our_order.recalculate(total_qty);
            }
        }
    }

    /// Recalculate our order positions at a specific price level.
    fn recalculate_our_positions_at_price(&self, price_fp: i64, side: u8) {
        let levels = self.levels(side);
        let total_qty = levels
            .get(&price_fp)
            .map(|l| l.total_qty())
            .unwrap_or(0);

        let mut orders = self.our_orders.write();
        for our_order in orders.values_mut() {
            if our_order.price_fp == price_fp && our_order.side == side {
                our_order.recalculate(total_qty);
            }
        }
    }
}

impl Default for MboBook {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(order_id: u64, price_fp: i64, qty: i64, action: MboAction, side: u8) -> MboEvent {
        MboEvent {
            order_id,
            price_fp,
            qty,
            action,
            side,
            timestamp_ns: now_ns(),
        }
    }

    #[test]
    fn test_add_orders_to_level() {
        let book = MboBook::new();

        // Add 100 orders to bid level at price 50000
        for i in 0..100 {
            let event = make_event(i, 5000_00000000, 10_0000, MboAction::Add, 0);
            book.on_mbo_event(&event);
        }

        assert_eq!(book.bid_level_count(), 1);
        assert_eq!(book.depth_at_price(5000_00000000, 0), 100 * 10_0000);
        assert_eq!(book.event_count(), 100);
    }

    #[test]
    fn test_queue_position_tracking() {
        let book = MboBook::new();

        // Add 10 orders ahead of ours
        for i in 0..10 {
            let event = make_event(i, 5000_00000000, 10_0000, MboAction::Add, 0);
            book.on_mbo_event(&event);
        }

        // Track our order
        book.track_our_order(999, 5000_00000000, 0, 5_0000);

        // Verify initial queue position
        let (queue_ahead, _fill_prob, _ttf) = book.get_queue_position(999).unwrap();
        assert_eq!(queue_ahead, 100_0000); // 10 orders * 10_0000 qty
    }

    #[test]
    fn test_delete_orders_ahead_improves_position() {
        let book = MboBook::new();

        // Add 5 orders
        for i in 0..5 {
            let event = make_event(i, 5000_00000000, 10_0000, MboAction::Add, 0);
            book.on_mbo_event(&event);
        }

        // Track our order
        book.track_our_order(999, 5000_00000000, 0, 5_0000);

        let initial_ahead = book.get_queue_position(999).unwrap().0;

        // Delete the first order (front of queue)
        let del_event = make_event(0, 5000_00000000, 10_0000, MboAction::Delete, 0);
        book.on_mbo_event(&del_event);

        // Queue ahead should decrease
        let new_ahead = book.get_queue_position(999).unwrap().0;
        assert!(new_ahead < initial_ahead, "Queue ahead should decrease after deletion ahead");
    }

    #[test]
    fn test_modify_order_updates_total_qty() {
        let book = MboBook::new();

        // Add an order
        let event = make_event(1, 5000_00000000, 100_0000, MboAction::Add, 0);
        book.on_mbo_event(&event);

        assert_eq!(book.depth_at_price(5000_00000000, 0), 100_0000);

        // Modify: reduce size
        let mod_event = make_event(1, 5000_00000000, 50_0000, MboAction::Modify, 0);
        book.on_mbo_event(&mod_event);

        assert_eq!(book.depth_at_price(5000_00000000, 0), 50_0000);
    }

    #[test]
    fn test_delete_all_cleans_level() {
        let book = MboBook::new();

        let event = make_event(1, 5000_00000000, 10_0000, MboAction::Add, 1);
        book.on_mbo_event(&event);
        assert_eq!(book.ask_level_count(), 1);

        let del_event = make_event(1, 5000_00000000, 10_0000, MboAction::Delete, 1);
        book.on_mbo_event(&del_event);
        assert_eq!(book.ask_level_count(), 0);
    }

    #[test]
    fn test_untrack_order() {
        let book = MboBook::new();
        book.track_our_order(1, 5000_00000000, 0, 10_0000);
        assert!(book.get_queue_position(1).is_some());

        book.untrack_our_order(1);
        assert!(book.get_queue_position(1).is_none());
    }

    #[test]
    fn test_multiple_price_levels() {
        let book = MboBook::new();

        // Bids at 50000
        for i in 0..5 {
            book.on_mbo_event(&make_event(i, 5000_00000000, 10_0000, MboAction::Add, 0));
        }
        // Bids at 49999
        for i in 5..10 {
            book.on_mbo_event(&make_event(i, 4999_00000000, 20_0000, MboAction::Add, 0));
        }
        // Asks at 50001
        for i in 10..15 {
            book.on_mbo_event(&make_event(i, 5001_00000000, 15_0000, MboAction::Add, 1));
        }

        assert_eq!(book.bid_level_count(), 2);
        assert_eq!(book.ask_level_count(), 1);
        assert_eq!(book.depth_at_price(5000_00000000, 0), 50_0000);
        assert_eq!(book.depth_at_price(4999_00000000, 0), 100_0000);
        assert_eq!(book.depth_at_price(5001_00000000, 1), 75_0000);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let book = Arc::new(MboBook::new());

        // Add some initial orders
        for i in 0..100 {
            book.on_mbo_event(&make_event(i, 5000_00000000, 10_0000, MboAction::Add, 0));
        }

        // Concurrent reads while writing
        let book_reader = book.clone();
        let reader = std::thread::spawn(move || {
            let mut depth_reads = 0;
            for _ in 0..1000 {
                let _ = book_reader.depth_at_price(5000_00000000, 0);
                depth_reads += 1;
            }
            depth_reads
        });

        let book_writer = book.clone();
        let writer = std::thread::spawn(move || {
            for i in 100..200 {
                book_writer.on_mbo_event(&make_event(
                    i, 5000_00000000, 10_0000, MboAction::Add, 0,
                ));
            }
        });

        writer.join().unwrap();
        let reads = reader.join().unwrap();
        assert!(reads > 0);
        assert!(book.bid_order_count() >= 100); // At least original orders
    }
}
