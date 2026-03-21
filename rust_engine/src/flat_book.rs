//! Zero-allocation L2 orderbook using pre-allocated flat arrays.
//!
//! Instead of `BTreeMap` with heap allocation on every insert, this uses a flat
//! pre-allocated array indexed by price tick offset from a reference price.
//! This gives O(1) insert/lookup with **zero heap allocation** after initialization.
//!
//! For 10,000 levels × 8 bytes = 80 KB per side = 160 KB per book.
//! Fits entirely in L2 cache of a modern CPU core.

use crate::fixed_point::{FixedPrice, FixedQty, notional_fp};

// ---------------------------------------------------------------------------
// FlatBookConfig
// ---------------------------------------------------------------------------

/// Configuration for the flat array orderbook.
///
/// For BTC_USDT: tick_size = 0.1 USDT, max_levels = 10_000.
/// This covers ±500 USDT from the reference price (more than sufficient for L2).
#[derive(Debug, Clone, Copy)]
pub struct FlatBookConfig {
    /// Tick size in FixedPrice representation.
    /// For 0.1 USDT at 1e8 precision: 10_000_000.
    pub tick_size_fp: i64,
    /// Number of price levels per side (e.g., 10_000).
    pub max_levels: usize,
    /// Center price in FixedPrice representation — re-centered periodically.
    pub reference_price_fp: i64,
}

impl Default for FlatBookConfig {
    fn default() -> Self {
        Self {
            tick_size_fp: 10_000_000,       // 0.1 USDT at 1e8
            max_levels: 10_000,
            reference_price_fp: 0,          // Will be set on first snapshot
        }
    }
}

// ---------------------------------------------------------------------------
// FlatOrderBook
// ---------------------------------------------------------------------------

/// Zero-allocation L2 orderbook using pre-allocated flat arrays.
///
/// Each slot represents one price tick. Index = (price - reference) / tick_size.
///
/// Memory layout: `[qty_level_0, qty_level_1, ..., qty_level_N]`
/// where `level_i` corresponds to `price = reference + i * tick_size`.
pub struct FlatOrderBook {
    /// Bid quantities indexed by tick offset (descending from reference).
    /// Index 0 = reference_price, Index 1 = reference_price - tick_size, etc.
    bid_levels: Box<[FixedQty]>,

    /// Ask quantities indexed by tick offset (ascending from reference).
    /// Index 0 = reference_price + tick_size, Index 1 = reference_price + 2*tick_size, etc.
    ask_levels: Box<[FixedQty]>,

    /// Configuration (tick size, max levels, reference price).
    config: FlatBookConfig,

    /// Best bid index (maintained on update for O(1) access).
    best_bid_idx: usize,

    /// Best ask index (maintained on update for O(1) access).
    best_ask_idx: usize,

    /// Sequence number for change detection.
    sequence: u64,

    /// Nanosecond timestamp of last update.
    last_update_ns: u64,

    /// Symbol identifier — fixed-size, no String heap allocation.
    symbol: [u8; 32],
}

impl FlatOrderBook {
    /// Create a new flat orderbook. Called ONCE at startup.
    /// All memory is pre-allocated. No further heap allocation occurs.
    pub fn new(config: FlatBookConfig, symbol: &str) -> Self {
        let bid_levels = vec![FixedQty(0); config.max_levels].into_boxed_slice();
        let ask_levels = vec![FixedQty(0); config.max_levels].into_boxed_slice();
        let mut sym = [0u8; 32];
        let bytes = symbol.as_bytes();
        let copy_len = bytes.len().min(32);
        sym[..copy_len].copy_from_slice(&bytes[..copy_len]);
        Self {
            bid_levels,
            ask_levels,
            config,
            best_bid_idx: config.max_levels, // Invalid = no bid
            best_ask_idx: config.max_levels, // Invalid = no ask
            sequence: 0,
            last_update_ns: 0,
            symbol: sym,
        }
    }

    /// Get the symbol as a string slice.
    pub fn symbol_str(&self) -> &str {
        let end = self.symbol.iter().position(|&b| b == 0).unwrap_or(32);
        std::str::from_utf8(&self.symbol[..end]).unwrap_or("???")
    }

    // ------------------------------------------------------------------
    // Price ↔ Index conversion
    // ------------------------------------------------------------------

    /// Convert a FixedPrice to a bid index.
    /// Bid index 0 = reference_price, 1 = reference - tick, etc. (descending).
    #[inline(always)]
    fn bid_index(&self, price: FixedPrice) -> Option<usize> {
        let diff = self.config.reference_price_fp - price.0;
        if diff < 0 {
            return None; // Price above reference — not a valid bid index
        }
        let idx = (diff / self.config.tick_size_fp) as usize;
        if idx >= self.config.max_levels {
            None
        } else {
            Some(idx)
        }
    }

    /// Convert a FixedPrice to an ask index.
    /// Ask index 0 = reference + tick, 1 = reference + 2*tick, etc. (ascending).
    #[inline(always)]
    fn ask_index(&self, price: FixedPrice) -> Option<usize> {
        let diff = price.0 - self.config.reference_price_fp;
        if diff <= 0 {
            return None; // Price at or below reference — not a valid ask index
        }
        let idx = ((diff - 1) / self.config.tick_size_fp) as usize;
        if idx >= self.config.max_levels {
            None
        } else {
            Some(idx)
        }
    }

    /// Convert a bid index back to a FixedPrice.
    #[inline(always)]
    fn bid_price_at(&self, idx: usize) -> FixedPrice {
        FixedPrice(self.config.reference_price_fp - (idx as i64 * self.config.tick_size_fp))
    }

    /// Convert an ask index back to a FixedPrice.
    #[inline(always)]
    fn ask_price_at(&self, idx: usize) -> FixedPrice {
        FixedPrice(self.config.reference_price_fp + ((idx as i64 + 1) * self.config.tick_size_fp))
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Update a single bid level. Returns the old quantity at that price.
    #[inline]
    pub fn update_bid(&mut self, price: FixedPrice, qty: FixedQty) -> FixedQty {
        let idx = match self.bid_index(price) {
            Some(i) => i,
            None => return FixedQty(0), // Out of range
        };
        let old = self.bid_levels[idx];
        self.bid_levels[idx] = qty;

        // Update best_bid_idx
        if qty.0 > 0 && idx < self.best_bid_idx {
            // New best (closer to reference = higher price)
            self.best_bid_idx = idx;
        } else if qty.0 == 0 && idx == self.best_bid_idx {
            // Removed the best bid — scan forward for next best
            self.best_bid_idx = self.scan_best_bid(idx);
        }

        self.sequence += 1;
        old
    }

    /// Update a single ask level. Returns the old quantity at that price.
    #[inline]
    pub fn update_ask(&mut self, price: FixedPrice, qty: FixedQty) -> FixedQty {
        let idx = match self.ask_index(price) {
            Some(i) => i,
            None => return FixedQty(0), // Out of range
        };
        let old = self.ask_levels[idx];
        self.ask_levels[idx] = qty;

        // Update best_ask_idx
        if qty.0 > 0 && idx < self.best_ask_idx {
            // New best (closer to reference = lower price)
            self.best_ask_idx = idx;
        } else if qty.0 == 0 && idx == self.best_ask_idx {
            // Removed the best ask — scan forward for next best
            self.best_ask_idx = self.scan_best_ask(idx);
        }

        self.sequence += 1;
        old
    }

    /// Apply a tracked delta update, returning (old_qty, new_qty) for microstructure analysis.
    #[inline]
    pub fn apply_delta_tracked(
        &mut self,
        price: FixedPrice,
        qty: FixedQty,
        is_bid: bool,
    ) -> (FixedQty, FixedQty) {
        let old = if is_bid {
            self.update_bid(price, qty)
        } else {
            self.update_ask(price, qty)
        };
        (old, qty)
    }

    /// Apply a full snapshot: clear all levels and insert from slice.
    pub fn apply_snapshot(&mut self, bids: &[(FixedPrice, FixedQty)], asks: &[(FixedPrice, FixedQty)]) {
        // Clear all levels
        for level in self.bid_levels.iter_mut() {
            *level = FixedQty(0);
        }
        for level in self.ask_levels.iter_mut() {
            *level = FixedQty(0);
        }

        // Reset best indices
        self.best_bid_idx = self.config.max_levels; // Invalid
        self.best_ask_idx = self.config.max_levels; // Invalid

        // Auto-set reference price from first snapshot if not set
        if self.config.reference_price_fp == 0 && !bids.is_empty() && !asks.is_empty() {
            let mid = (bids[0].0 .0 + asks[0].0 .0) / 2;
            self.config.reference_price_fp = mid;
        }

        // Insert bids
        for &(price, qty) in bids {
            if qty.0 > 0 {
                if let Some(idx) = self.bid_index(price) {
                    self.bid_levels[idx] = qty;
                    if self.best_bid_idx >= self.config.max_levels || idx < self.best_bid_idx {
                        self.best_bid_idx = idx;
                    }
                }
            }
        }

        // Insert asks
        for &(price, qty) in asks {
            if qty.0 > 0 {
                if let Some(idx) = self.ask_index(price) {
                    self.ask_levels[idx] = qty;
                    if self.best_ask_idx >= self.config.max_levels || idx < self.best_ask_idx {
                        self.best_ask_idx = idx;
                    }
                }
            }
        }

        self.sequence += 1;
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Best bid price and quantity, or None if no bids.
    #[inline]
    pub fn best_bid(&self) -> Option<(FixedPrice, FixedQty)> {
        if self.best_bid_idx >= self.config.max_levels {
            return None;
        }
        let qty = self.bid_levels[self.best_bid_idx];
        if qty.0 == 0 {
            return None;
        }
        Some((self.bid_price_at(self.best_bid_idx), qty))
    }

    /// Best ask price and quantity, or None if no asks.
    #[inline]
    pub fn best_ask(&self) -> Option<(FixedPrice, FixedQty)> {
        if self.best_ask_idx >= self.config.max_levels {
            return None;
        }
        let qty = self.ask_levels[self.best_ask_idx];
        if qty.0 == 0 {
            return None;
        }
        Some((self.ask_price_at(self.best_ask_idx), qty))
    }

    /// Mid price as FixedPrice, or ZERO if either side is empty.
    #[inline]
    pub fn mid_price(&self) -> FixedPrice {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => FixedPrice::mid(bid, ask),
            _ => FixedPrice::ZERO,
        }
    }

    /// Spread in basis points (integer).
    #[inline]
    pub fn spread_bps(&self) -> i64 {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => FixedPrice::spread_bps(bid, ask),
            _ => 0,
        }
    }

    /// Compute bid-side depth in USDT for the top `n` levels.
    pub fn bid_depth_usdt(&self, n: usize) -> f64 {
        let mut total = 0i128;
        let mut count = 0;
        let start = self.best_bid_idx;
        if start >= self.config.max_levels {
            return 0.0;
        }
        for idx in start..self.config.max_levels {
            let qty = self.bid_levels[idx];
            if qty.0 > 0 {
                let price = self.bid_price_at(idx);
                total += notional_fp(price, qty) as i128;
                count += 1;
                if count >= n {
                    break;
                }
            }
        }
        total as f64 / FixedPrice::PRECISION as f64
    }

    /// Compute ask-side depth in USDT for the top `n` levels.
    pub fn ask_depth_usdt(&self, n: usize) -> f64 {
        let mut total = 0i128;
        let mut count = 0;
        let start = self.best_ask_idx;
        if start >= self.config.max_levels {
            return 0.0;
        }
        for idx in start..self.config.max_levels {
            let qty = self.ask_levels[idx];
            if qty.0 > 0 {
                let price = self.ask_price_at(idx);
                total += notional_fp(price, qty) as i128;
                count += 1;
                if count >= n {
                    break;
                }
            }
        }
        total as f64 / FixedPrice::PRECISION as f64
    }

    /// Get top N bid levels as (price, qty) pairs, sorted descending by price.
    pub fn get_bids(&self, depth: usize) -> Vec<(FixedPrice, FixedQty)> {
        let mut result = Vec::with_capacity(depth);
        if self.best_bid_idx >= self.config.max_levels {
            return result;
        }
        for idx in self.best_bid_idx..self.config.max_levels {
            let qty = self.bid_levels[idx];
            if qty.0 > 0 {
                result.push((self.bid_price_at(idx), qty));
                if result.len() >= depth {
                    break;
                }
            }
        }
        result
    }

    /// Get top N ask levels as (price, qty) pairs, sorted ascending by price.
    pub fn get_asks(&self, depth: usize) -> Vec<(FixedPrice, FixedQty)> {
        let mut result = Vec::with_capacity(depth);
        if self.best_ask_idx >= self.config.max_levels {
            return result;
        }
        for idx in self.best_ask_idx..self.config.max_levels {
            let qty = self.ask_levels[idx];
            if qty.0 > 0 {
                result.push((self.ask_price_at(idx), qty));
                if result.len() >= depth {
                    break;
                }
            }
        }
        result
    }

    /// Book imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth).
    /// Range: [-1.0, 1.0]. Positive = more bid-side depth.
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_d = self.bid_depth_usdt(depth);
        let ask_d = self.ask_depth_usdt(depth);
        let total = bid_d + ask_d;
        if total <= 0.0 {
            return 0.0;
        }
        (bid_d - ask_d) / total
    }

    /// Recenter the book when the mid-price has drifted too far from reference.
    ///
    /// Should be called when `|current_mid - reference| > 100 * tick_size`.
    /// Recentering preserves all existing levels by copying them to new offsets.
    pub fn recenter(&mut self) {
        let mid = self.mid_price();
        if mid.0 == 0 {
            return;
        }

        let drift = (mid.0 - self.config.reference_price_fp).abs();
        let threshold = self.config.tick_size_fp * 100;

        if drift <= threshold {
            return; // No recentering needed
        }

        // Collect current levels
        let old_bids = self.get_bids(self.config.max_levels);
        let old_asks = self.get_asks(self.config.max_levels);

        // Update reference to current mid
        self.config.reference_price_fp = mid.0;

        // Clear and re-insert
        for level in self.bid_levels.iter_mut() {
            *level = FixedQty(0);
        }
        for level in self.ask_levels.iter_mut() {
            *level = FixedQty(0);
        }
        self.best_bid_idx = self.config.max_levels;
        self.best_ask_idx = self.config.max_levels;

        for (price, qty) in old_bids {
            if let Some(idx) = self.bid_index(price) {
                self.bid_levels[idx] = qty;
                if self.best_bid_idx >= self.config.max_levels || idx < self.best_bid_idx {
                    self.best_bid_idx = idx;
                }
            }
        }

        for (price, qty) in old_asks {
            if let Some(idx) = self.ask_index(price) {
                self.ask_levels[idx] = qty;
                if self.best_ask_idx >= self.config.max_levels || idx < self.best_ask_idx {
                    self.best_ask_idx = idx;
                }
            }
        }
    }

    /// Set the last update timestamp (nanoseconds).
    #[inline]
    pub fn set_timestamp_ns(&mut self, ns: u64) {
        self.last_update_ns = ns;
    }

    /// Get the last update timestamp (nanoseconds).
    #[inline]
    pub fn timestamp_ns(&self) -> u64 {
        self.last_update_ns
    }

    /// Get the sequence number.
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Get the reference price.
    #[inline]
    pub fn reference_price(&self) -> FixedPrice {
        FixedPrice(self.config.reference_price_fp)
    }

    /// Set the reference price (used during initialization).
    #[inline]
    pub fn set_reference_price(&mut self, price: FixedPrice) {
        self.config.reference_price_fp = price.0;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Scan forward from `start` to find the next non-zero bid level.
    fn scan_best_bid(&self, start: usize) -> usize {
        for idx in start..self.config.max_levels {
            if self.bid_levels[idx].0 > 0 {
                return idx;
            }
        }
        self.config.max_levels // No bids found
    }

    /// Scan forward from `start` to find the next non-zero ask level.
    fn scan_best_ask(&self, start: usize) -> usize {
        for idx in start..self.config.max_levels {
            if self.ask_levels[idx].0 > 0 {
                return idx;
            }
        }
        self.config.max_levels // No asks found
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_book() -> FlatOrderBook {
        let config = FlatBookConfig {
            tick_size_fp: 10_000_000, // 0.1 USDT at 1e8
            max_levels: 1000,
            reference_price_fp: 5_000_000_000_000, // $50,000.00
        };
        FlatOrderBook::new(config, "BTC_USDT")
    }

    #[test]
    fn test_new_book_empty() {
        let book = make_book();
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
        assert_eq!(book.mid_price(), FixedPrice::ZERO);
        assert_eq!(book.symbol_str(), "BTC_USDT");
    }

    #[test]
    fn test_update_bid_ask() {
        let mut book = make_book();

        // Insert a bid at $49,999.90 (0.1 below reference)
        let bid_price = FixedPrice::from_f64(49999.9);
        let bid_qty = FixedQty::from_f64(2.5);
        book.update_bid(bid_price, bid_qty);

        // Insert an ask at $50,000.10 (0.1 above reference)
        let ask_price = FixedPrice::from_f64(50000.1);
        let ask_qty = FixedQty::from_f64(1.0);
        book.update_ask(ask_price, ask_qty);

        let (bb_price, bb_qty) = book.best_bid().unwrap();
        assert!((bb_price.to_f64() - 49999.9).abs() < 0.2);
        assert!((bb_qty.to_f64() - 2.5).abs() < 0.01);

        let (ba_price, ba_qty) = book.best_ask().unwrap();
        assert!((ba_price.to_f64() - 50000.1).abs() < 0.2);
        assert!((ba_qty.to_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_remove_level() {
        let mut book = make_book();

        let price = FixedPrice::from_f64(49999.9);
        book.update_bid(price, FixedQty::from_f64(2.0));
        assert!(book.best_bid().is_some());

        // Remove by setting qty to 0
        book.update_bid(price, FixedQty(0));
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_snapshot_and_delta() {
        let mut book = make_book();

        // Apply snapshot
        let bids = vec![
            (FixedPrice::from_f64(49999.9), FixedQty::from_f64(1.0)),
            (FixedPrice::from_f64(49999.8), FixedQty::from_f64(2.0)),
        ];
        let asks = vec![
            (FixedPrice::from_f64(50000.1), FixedQty::from_f64(0.5)),
            (FixedPrice::from_f64(50000.2), FixedQty::from_f64(1.5)),
        ];
        book.apply_snapshot(&bids, &asks);

        assert!(book.best_bid().is_some());
        assert!(book.best_ask().is_some());

        // Apply delta — update best bid qty
        let (old, new) = book.apply_delta_tracked(
            FixedPrice::from_f64(49999.9),
            FixedQty::from_f64(3.0),
            true,
        );
        assert!((old.to_f64() - 1.0).abs() < 0.01);
        assert!((new.to_f64() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_depth_usdt() {
        let mut book = make_book();

        let bids = vec![
            (FixedPrice::from_f64(49999.9), FixedQty::from_f64(1.0)),
            (FixedPrice::from_f64(49999.8), FixedQty::from_f64(1.0)),
        ];
        let asks = vec![
            (FixedPrice::from_f64(50000.1), FixedQty::from_f64(1.0)),
        ];
        book.apply_snapshot(&bids, &asks);

        let bid_depth = book.bid_depth_usdt(10);
        assert!(bid_depth > 99000.0, "Bid depth should be ~$100K, got {}", bid_depth);

        let ask_depth = book.ask_depth_usdt(10);
        assert!(ask_depth > 49000.0, "Ask depth should be ~$50K, got {}", ask_depth);
    }

    #[test]
    fn test_imbalance() {
        let mut book = make_book();

        // Equal depth on both sides
        let bids = vec![(FixedPrice::from_f64(49999.9), FixedQty::from_f64(1.0))];
        let asks = vec![(FixedPrice::from_f64(50000.1), FixedQty::from_f64(1.0))];
        book.apply_snapshot(&bids, &asks);

        let imb = book.imbalance(10);
        assert!(imb.abs() < 0.01, "Equal depth should give ~0 imbalance, got {}", imb);
    }

    #[test]
    fn test_recenter() {
        // Use a wider book so we can place levels far from reference
        let config = FlatBookConfig {
            tick_size_fp: 10_000_000, // 0.1 USDT
            max_levels: 10_000,
            reference_price_fp: 5_000_000_000_000, // $50,000.00
        };
        let mut book = FlatOrderBook::new(config, "BTC_USDT");

        // Place bid far below reference and ask just above to create asymmetric mid
        // Bid at $49,975 → 250 ticks below reference
        // Ask at $50,000.1 → 1 tick above reference
        // Mid = ($49,975 + $50,000.1) / 2 = $49,987.55
        // Drift = |$49,987.55 - $50,000| = $12.45 = ~124 ticks > 100 threshold → triggers recenter
        let bid_price = FixedPrice::from_f64(49975.0);
        let ask_price = FixedPrice::from_f64(50000.1);
        book.update_bid(bid_price, FixedQty::from_f64(1.0));
        book.update_ask(ask_price, FixedQty::from_f64(1.0));

        let mid_before = book.mid_price();
        assert!(mid_before.0 != 0, "Mid should be valid before recenter");

        book.recenter();

        // After recenter, the mid price should be preserved (levels re-indexed around new reference)
        let mid_after = book.mid_price();
        assert!(
            (mid_before.to_f64() - mid_after.to_f64()).abs() < 1.0,
            "Recenter should preserve levels: before={}, after={}",
            mid_before.to_f64(),
            mid_after.to_f64()
        );
    }

    #[test]
    fn test_many_levels() {
        let config = FlatBookConfig {
            tick_size_fp: 10_000_000,
            max_levels: 10_000,
            reference_price_fp: 5_000_000_000_000,
        };
        let mut book = FlatOrderBook::new(config, "BTC_USDT");

        // Insert 1000 bid levels
        for i in 1..=1000 {
            let price = FixedPrice(5_000_000_000_000 - i * 10_000_000); // ref - i * tick
            let qty = FixedQty::from_f64(1.0);
            book.update_bid(price, qty);
        }

        // Best bid should be the highest (closest to reference)
        let (best_price, _) = book.best_bid().unwrap();
        let expected = FixedPrice(5_000_000_000_000 - 10_000_000);
        assert_eq!(best_price, expected, "Best bid should be reference - 1 tick");
    }

    #[test]
    fn test_get_bids_asks() {
        let mut book = make_book();

        let bids = vec![
            (FixedPrice::from_f64(49999.9), FixedQty::from_f64(1.0)),
            (FixedPrice::from_f64(49999.8), FixedQty::from_f64(2.0)),
            (FixedPrice::from_f64(49999.7), FixedQty::from_f64(3.0)),
        ];
        let asks = vec![
            (FixedPrice::from_f64(50000.1), FixedQty::from_f64(0.5)),
            (FixedPrice::from_f64(50000.2), FixedQty::from_f64(1.5)),
        ];
        book.apply_snapshot(&bids, &asks);

        let top_bids = book.get_bids(2);
        assert_eq!(top_bids.len(), 2);

        let top_asks = book.get_asks(2);
        assert_eq!(top_asks.len(), 2);
    }
}
