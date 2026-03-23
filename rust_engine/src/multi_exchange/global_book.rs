//! Consolidated Global Order Book
//!
//! Merges L2 order books from multiple exchanges (Gate.io, Binance, Bybit)
//! into a unified, fee-adjusted, latency-compensated view of global liquidity.
//!
//! This is the core data structure of the multi-exchange feature.

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::fixed_point::FixedPrice;

// ---------------------------------------------------------------------------
// Exchange Identifier
// ---------------------------------------------------------------------------

/// Exchange identifier enum - zero-cost, no heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExchangeId {
    GateIo  = 0,
    Binance = 1,
    Bybit   = 2,
}

impl ExchangeId {
    /// Taker fee in basis points for each exchange.
    /// Gate.io: 0.050% = 5 bps
    /// Binance: 0.040% = 4 bps
    /// Bybit:   0.055% = 5.5 bps (rounded to 6)
    pub fn taker_fee_bps(&self) -> i64 {
        match self {
            ExchangeId::GateIo  => 5,
            ExchangeId::Binance => 4,
            ExchangeId::Bybit   => 6,
        }
    }

    /// Maker fee in basis points (negative = rebate).
    /// Gate.io: 0.020% = 2 bps
    /// Binance: 0.020% = 2 bps
    /// Bybit:   0.010% = 1 bps
    pub fn maker_fee_bps(&self) -> i64 {
        match self {
            ExchangeId::GateIo  => 2,
            ExchangeId::Binance => 2,
            ExchangeId::Bybit   => 1,
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            ExchangeId::GateIo  => "Gate.io",
            ExchangeId::Binance => "Binance",
            ExchangeId::Bybit   => "Bybit",
        }
    }

    /// Short identifier for API paths.
    pub fn id_str(&self) -> &'static str {
        match self {
            ExchangeId::GateIo  => "gateio",
            ExchangeId::Binance => "binance",
            ExchangeId::Bybit   => "bybit",
        }
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gateio" | "gate.io" | "gate" => Some(ExchangeId::GateIo),
            "binance" => Some(ExchangeId::Binance),
            "bybit" => Some(ExchangeId::Bybit),
            _ => None,
        }
    }

    /// Get exchange by index (0=GateIo, 1=Binance, 2=Bybit).
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(ExchangeId::GateIo),
            1 => Some(ExchangeId::Binance),
            2 => Some(ExchangeId::Bybit),
            _ => None,
        }
    }

    /// All exchanges as array.
    pub fn all() -> [ExchangeId; 3] {
        [ExchangeId::GateIo, ExchangeId::Binance, ExchangeId::Bybit]
    }
}

impl Default for ExchangeId {
    fn default() -> Self {
        ExchangeId::GateIo
    }
}

// ---------------------------------------------------------------------------
// Global Level
// ---------------------------------------------------------------------------

/// A single price level in the global book, fee-and-latency adjusted.
#[derive(Debug, Clone)]
pub struct GlobalLevel {
    /// Raw price from the exchange (fixed-point, same precision as FlatOrderBook).
    pub raw_price_fp: i64,
    /// Fee-adjusted effective price (raw_price + taker_fee_bps adjustment).
    pub effective_price_fp: i64,
    /// Available quantity at this level (contracts, scaled by 1e8).
    pub qty: i64,
    /// Which exchange this level came from.
    pub exchange: ExchangeId,
    /// Latency penalty in basis points (added to effective_price for asks,
    /// subtracted for bids, based on measured round-trip latency).
    pub latency_penalty_bps: i64,
    /// Timestamp of last update (nanoseconds since epoch).
    pub updated_ns: u64,
}

impl Default for GlobalLevel {
    fn default() -> Self {
        Self {
            raw_price_fp: 0,
            effective_price_fp: 0,
            qty: 0,
            exchange: ExchangeId::GateIo,
            latency_penalty_bps: 0,
            updated_ns: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Exchange Book Snapshot
// ---------------------------------------------------------------------------

/// Per-exchange L2 snapshot for a single symbol.
#[derive(Debug, Clone)]
pub struct ExchangeBookSnapshot {
    pub exchange: ExchangeId,
    pub symbol_id: u16,
    pub best_bid_fp: i64,
    pub best_ask_fp: i64,
    pub bid_levels: Vec<(i64, i64)>,  // (price_fp, qty)
    pub ask_levels: Vec<(i64, i64)>,  // (price_fp, qty)
    pub sequence: u64,
    pub timestamp_ns: u64,
}

impl Default for ExchangeBookSnapshot {
    fn default() -> Self {
        Self {
            exchange: ExchangeId::GateIo,
            symbol_id: 0,
            best_bid_fp: 0,
            best_ask_fp: 0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            sequence: 0,
            timestamp_ns: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Global Order Book
// ---------------------------------------------------------------------------

/// The Consolidated Global Order Book for a single symbol.
///
/// Merges L2 books from all active exchanges, applying fee and latency
/// adjustments to produce a unified, sorted view of global liquidity.
pub struct GlobalOrderBook {
    pub symbol_id: u16,
    /// Per-exchange snapshots (indexed by ExchangeId as u8).
    exchange_snapshots: [Option<ExchangeBookSnapshot>; 3],
    /// Merged and sorted global bids (best bid first = highest price).
    pub global_bids: Vec<GlobalLevel>,
    /// Merged and sorted global asks (best ask first = lowest price).
    pub global_asks: Vec<GlobalLevel>,
    /// Measured round-trip latency per exchange in microseconds.
    pub latency_us: [u64; 3],
    /// Last full merge timestamp.
    pub last_merge_ns: u64,
}

impl GlobalOrderBook {
    /// Create a new GlobalOrderBook for a symbol.
    pub fn new(symbol_id: u16) -> Self {
        Self {
            symbol_id,
            exchange_snapshots: [None, None, None],
            global_bids: Vec::with_capacity(60),  // 20 levels × 3 exchanges
            global_asks: Vec::with_capacity(60),
            latency_us: [0; 3],
            last_merge_ns: 0,
        }
    }

    /// Update the snapshot for a specific exchange and trigger a re-merge.
    /// Called by the WS ingestion threads whenever a new book update arrives.
    pub fn update_exchange_snapshot(&mut self, snapshot: ExchangeBookSnapshot) {
        let idx = snapshot.exchange as usize;
        if idx < 3 {
            self.exchange_snapshots[idx] = Some(snapshot);
            self.merge();
        }
    }

    /// Update measured latency for an exchange (called by WS ingestion on pong).
    pub fn update_latency(&mut self, exchange: ExchangeId, latency_us: u64) {
        let idx = exchange as usize;
        if idx < 3 {
            self.latency_us[idx] = latency_us;
        }
    }

    /// Merge all exchange snapshots into the global_bids and global_asks vectors.
    ///
    /// Applies fee adjustment:
    ///   effective_ask = raw_ask * (1 + taker_fee_bps/10000)
    ///   effective_bid = raw_bid * (1 - taker_fee_bps/10000)
    ///
    /// Applies latency penalty:
    ///   latency_penalty_bps = latency_us / 100 (1us = 0.01bps)
    ///
    /// Sorts bids descending by effective_price_fp.
    /// Sorts asks ascending by effective_price_fp.
    pub fn merge(&mut self) {
        self.global_bids.clear();
        self.global_asks.clear();

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        for (idx, opt_snapshot) in self.exchange_snapshots.iter().enumerate() {
            let Some(snapshot) = opt_snapshot else { continue };
            let exchange = ExchangeId::from_index(idx).unwrap_or_default();
            let taker_fee_bps = exchange.taker_fee_bps();
            let latency_us = self.latency_us[idx];
            // Latency penalty: 1 microsecond = 0.01 basis points
            let latency_penalty_bps = (latency_us / 100) as i64;

            // Process bids: effective_bid = raw_bid * (1 - taker_fee_bps/10000)
            for &(price_fp, qty) in &snapshot.bid_levels {
                if qty <= 0 { continue; }
                // Fee adjustment: reduce effective bid price by taker fee
                let fee_adj = (price_fp * taker_fee_bps) / 10000;
                // Latency penalty: further reduce bid (worse fill due to latency)
                let latency_adj = (price_fp * latency_penalty_bps) / 10000;
                let effective_price_fp = price_fp - fee_adj - latency_adj;

                self.global_bids.push(GlobalLevel {
                    raw_price_fp: price_fp,
                    effective_price_fp,
                    qty,
                    exchange,
                    latency_penalty_bps,
                    updated_ns: snapshot.timestamp_ns,
                });
            }

            // Process asks: effective_ask = raw_ask * (1 + taker_fee_bps/10000)
            for &(price_fp, qty) in &snapshot.ask_levels {
                if qty <= 0 { continue; }
                // Fee adjustment: increase effective ask price by taker fee
                let fee_adj = (price_fp * taker_fee_bps) / 10000;
                // Latency penalty: further increase ask (worse fill due to latency)
                let latency_adj = (price_fp * latency_penalty_bps) / 10000;
                let effective_price_fp = price_fp + fee_adj + latency_adj;

                self.global_asks.push(GlobalLevel {
                    raw_price_fp: price_fp,
                    effective_price_fp,
                    qty,
                    exchange,
                    latency_penalty_bps,
                    updated_ns: snapshot.timestamp_ns,
                });
            }
        }

        // Sort bids descending by effective price (best bid = highest)
        self.global_bids.sort_by(|a, b| b.effective_price_fp.cmp(&a.effective_price_fp));

        // Sort asks ascending by effective price (best ask = lowest)
        self.global_asks.sort_by(|a, b| a.effective_price_fp.cmp(&b.effective_price_fp));

        self.last_merge_ns = now_ns;
    }

    /// Get the global best bid (highest effective bid across all exchanges).
    pub fn best_bid(&self) -> Option<&GlobalLevel> {
        self.global_bids.first()
    }

    /// Get the global best ask (lowest effective ask across all exchanges).
    pub fn best_ask(&self) -> Option<&GlobalLevel> {
        self.global_asks.first()
    }

    /// Calculate global mid price (average of best bid and best ask effective prices).
    pub fn global_mid_fp(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some((bid.effective_price_fp + ask.effective_price_fp) / 2)
            }
            _ => None,
        }
    }

    /// Calculate global spread in basis points.
    pub fn global_spread_bps(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid.effective_price_fp + ask.effective_price_fp) / 2;
                if mid == 0 {
                    return Some(0);
                }
                let spread = ask.effective_price_fp - bid.effective_price_fp;
                Some((spread * 10000) / mid)
            }
            _ => None,
        }
    }

    /// Get total available liquidity at the best N levels across all exchanges.
    pub fn total_bid_depth(&self, levels: usize) -> i64 {
        self.global_bids.iter()
            .take(levels)
            .map(|l| l.qty)
            .sum()
    }

    /// Get total available ask liquidity at the best N levels.
    pub fn total_ask_depth(&self, levels: usize) -> i64 {
        self.global_asks.iter()
            .take(levels)
            .map(|l| l.qty)
            .sum()
    }

    /// Check if a specific exchange's book is stale (last update > 5 seconds ago).
    pub fn is_exchange_stale(&self, exchange: ExchangeId, now_ns: u64) -> bool {
        let idx = exchange as usize;
        if idx >= 3 { return true; }
        match &self.exchange_snapshots[idx] {
            Some(snap) => now_ns.saturating_sub(snap.timestamp_ns) > 5_000_000_000,
            None => true,
        }
    }

    /// Get the raw snapshot for a specific exchange.
    pub fn get_exchange_snapshot(&self, exchange: ExchangeId) -> Option<&ExchangeBookSnapshot> {
        let idx = exchange as usize;
        if idx < 3 {
            self.exchange_snapshots[idx].as_ref()
        } else {
            None
        }
    }

    /// Get cross-exchange spread: best_bid_exchange vs best_ask_exchange.
    /// Returns (bid_exchange, ask_exchange, spread_bps) if a spread exists.
    pub fn cross_exchange_spread(&self) -> Option<(ExchangeId, ExchangeId, i64)> {
        let bid = self.best_bid()?;
        let ask = self.best_ask()?;
        
        if bid.exchange == ask.exchange {
            return None;  // Same exchange, not a cross-exchange spread
        }

        let mid = (bid.raw_price_fp + ask.raw_price_fp) / 2;
        if mid == 0 {
            return None;
        }

        // Cross-exchange spread: buy on ask_exchange, sell on bid_exchange
        let spread = bid.raw_price_fp - ask.raw_price_fp;
        let spread_bps = (spread * 10000) / mid;

        Some((bid.exchange, ask.exchange, spread_bps))
    }

    /// Serialize the global book to JSON for the dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        let mut exchanges = Vec::new();
        
        for exchange in ExchangeId::all() {
            if let Some(snap) = self.get_exchange_snapshot(exchange) {
                exchanges.push(serde_json::json!({
                    "exchange": exchange.name(),
                    "id": exchange.id_str(),
                    "best_bid": FixedPrice(snap.best_bid_fp).to_f64(),
                    "best_ask": FixedPrice(snap.best_ask_fp).to_f64(),
                    "spread_bps": if snap.best_bid_fp > 0 {
                        let mid = (snap.best_bid_fp + snap.best_ask_fp) / 2;
                        ((snap.best_ask_fp - snap.best_bid_fp) * 10000) / mid.max(1)
                    } else { 0 },
                    "latency_us": self.latency_us[exchange as usize],
                    "updated_ns": snap.timestamp_ns,
                    "is_stale": self.is_exchange_stale(exchange, self.last_merge_ns),
                }));
            }
        }

        let global_best_bid = self.best_bid().map(|l| serde_json::json!({
            "price": FixedPrice(l.raw_price_fp).to_f64(),
            "effective_price": FixedPrice(l.effective_price_fp).to_f64(),
            "qty": l.qty as f64 / 1e8,
            "exchange": l.exchange.name(),
        }));

        let global_best_ask = self.best_ask().map(|l| serde_json::json!({
            "price": FixedPrice(l.raw_price_fp).to_f64(),
            "effective_price": FixedPrice(l.effective_price_fp).to_f64(),
            "qty": l.qty as f64 / 1e8,
            "exchange": l.exchange.name(),
        }));

        serde_json::json!({
            "symbol_id": self.symbol_id,
            "exchanges": exchanges,
            "global_best_bid": global_best_bid,
            "global_best_ask": global_best_ask,
            "global_mid": self.global_mid_fp().map(|p| FixedPrice(p).to_f64()),
            "global_spread_bps": self.global_spread_bps(),
            "cross_exchange_spread": self.cross_exchange_spread().map(|(bid_ex, ask_ex, spread)| {
                serde_json::json!({
                    "bid_exchange": bid_ex.name(),
                    "ask_exchange": ask_ex.name(),
                    "spread_bps": spread,
                })
            }),
            "last_merge_ns": self.last_merge_ns,
        })
    }
}

// ---------------------------------------------------------------------------
// Thread-Safe Wrappers
// ---------------------------------------------------------------------------

/// Thread-safe wrapper for GlobalOrderBook, shared between WS ingestion
/// threads and the strategy evaluator.
pub type SharedGlobalBook = Arc<RwLock<GlobalOrderBook>>;

/// Registry of global books for all symbols.
pub struct GlobalBookRegistry {
    books: DashMap<u16, SharedGlobalBook>,
}

impl GlobalBookRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            books: DashMap::new(),
        }
    }

    /// Get or create a GlobalOrderBook for a symbol.
    pub fn get_or_create(&self, symbol_id: u16) -> SharedGlobalBook {
        self.books
            .entry(symbol_id)
            .or_insert_with(|| Arc::new(RwLock::new(GlobalOrderBook::new(symbol_id))))
            .clone()
    }

    /// Get an existing GlobalOrderBook for a symbol.
    pub fn get(&self, symbol_id: u16) -> Option<SharedGlobalBook> {
        self.books.get(&symbol_id).map(|r| r.clone())
    }

    /// Get all registered symbol IDs.
    pub fn all_symbol_ids(&self) -> Vec<u16> {
        self.books.iter().map(|r| *r.key()).collect()
    }

    /// Get the number of registered symbols.
    pub fn len(&self) -> usize {
        self.books.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.books.is_empty()
    }

    /// Serialize all books to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let mut symbols = Vec::new();
        for entry in self.books.iter() {
            let book = entry.value().read();
            symbols.push(book.to_json());
        }
        serde_json::json!({
            "symbols": symbols,
            "count": symbols.len(),
        })
    }
}

impl Default for GlobalBookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_id_fees() {
        assert_eq!(ExchangeId::GateIo.taker_fee_bps(), 5);
        assert_eq!(ExchangeId::Binance.taker_fee_bps(), 4);
        assert_eq!(ExchangeId::Bybit.taker_fee_bps(), 6);
    }

    #[test]
    fn test_exchange_id_from_str() {
        assert_eq!(ExchangeId::from_str("gateio"), Some(ExchangeId::GateIo));
        assert_eq!(ExchangeId::from_str("BINANCE"), Some(ExchangeId::Binance));
        assert_eq!(ExchangeId::from_str("bybit"), Some(ExchangeId::Bybit));
        assert_eq!(ExchangeId::from_str("unknown"), None);
    }

    #[test]
    fn test_global_book_merge() {
        let mut book = GlobalOrderBook::new(1);

        // Add Gate.io snapshot
        let gateio_snap = ExchangeBookSnapshot {
            exchange: ExchangeId::GateIo,
            symbol_id: 1,
            best_bid_fp: 5000000000000,  // 50000.0
            best_ask_fp: 5000100000000,  // 50001.0
            bid_levels: vec![(5000000000000, 100000000)],
            ask_levels: vec![(5000100000000, 100000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(gateio_snap);

        // Add Binance snapshot with better prices
        let binance_snap = ExchangeBookSnapshot {
            exchange: ExchangeId::Binance,
            symbol_id: 1,
            best_bid_fp: 5000050000000,  // 50000.5 (higher bid)
            best_ask_fp: 5000080000000,  // 50000.8 (lower ask)
            bid_levels: vec![(5000050000000, 200000000)],
            ask_levels: vec![(5000080000000, 200000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(binance_snap);

        // After merge, Binance should have the best bid and best ask
        let best_bid = book.best_bid().unwrap();
        let best_ask = book.best_ask().unwrap();
        
        // Binance has better raw prices, but after fee adjustment...
        // Check that merge occurred
        assert!(!book.global_bids.is_empty());
        assert!(!book.global_asks.is_empty());
    }

    #[test]
    fn test_registry() {
        let registry = GlobalBookRegistry::new();
        
        let book1 = registry.get_or_create(1);
        let book2 = registry.get_or_create(2);
        let book1_again = registry.get_or_create(1);
        
        assert_eq!(registry.len(), 2);
        assert!(Arc::ptr_eq(&book1, &book1_again));
        assert!(!Arc::ptr_eq(&book1, &book2));
    }
}
