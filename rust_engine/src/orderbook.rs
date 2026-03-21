//! Rust-native order book engine backed by ``FlatOrderBook`` with fixed-point arithmetic.
//!
//! **Issue 1 Rewrite**: Internal storage migrated from `BTreeMap<OrderedFloat<f64>, f64>`
//! to `FlatOrderBook` with `FixedPrice` / `FixedQty` types.
//!
//! The PyO3 `#[pymethods]` interface is preserved for backward compatibility with the
//! Python dashboard. All f64 ↔ FixedPrice conversions happen at the FFI boundary.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::fixed_point::{FixedPrice, FixedQty};
use crate::flat_book::{FlatBookConfig, FlatOrderBook};

/// Large-order detection multiplier: a level is "large" when its notional
/// exceeds this multiple of the mean notional across visible book levels.
const LARGE_ORDER_MULTIPLIER: f64 = 5.0;
/// Number of top levels used for depth calculations (matches Python `_DEPTH_LEVELS`).
const DEPTH_LEVELS: usize = 10;
/// Fraction inside the spread used for optimal price calculation.
const OPTIMAL_PRICE_SPREAD_FRACTION: f64 = 0.1;
/// Minimum price tick to avoid division/rounding issues.
const MIN_PRICE_TICK: f64 = 1e-8;

// ---------------------------------------------------------------------------
// RustOrderBook — PyO3 wrapper over FlatOrderBook
// ---------------------------------------------------------------------------

/// In-memory L2 order book backed by `FlatOrderBook` with fixed-point arithmetic.
///
/// The PyO3 interface accepts and returns `f64` values — the same API as before.
/// Internally, all prices and quantities are stored as `FixedPrice` / `FixedQty`.
#[pyclass]
pub struct RustOrderBook {
    /// The underlying flat array orderbook with fixed-point types.
    inner: FlatOrderBook,
    /// Wall-clock time of last mutation (for staleness detection).
    last_update: Option<Instant>,
    /// Unix-millisecond timestamp recorded on last update (for snapshot export).
    last_update_ms: u64,
    /// Symbol name (kept as String for Python compatibility).
    symbol: String,
}

#[pymethods]
impl RustOrderBook {
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    #[new]
    pub fn new(symbol: &str) -> Self {
        let config = FlatBookConfig {
            tick_size_fp: 10_000_000,       // 0.1 USDT default
            max_levels: 10_000,
            reference_price_fp: 0,          // Set on first snapshot
        };
        Self {
            inner: FlatOrderBook::new(config, symbol),
            last_update: None,
            last_update_ms: 0,
            symbol: symbol.to_string(),
        }
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Full snapshot replacement: clear both sides and insert all levels.
    pub fn update_snapshot(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        let fp_bids: Vec<(FixedPrice, FixedQty)> = bids
            .iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(p, s)| (FixedPrice::from_f64(*p), FixedQty::from_f64(*s)))
            .collect();
        let fp_asks: Vec<(FixedPrice, FixedQty)> = asks
            .iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(p, s)| (FixedPrice::from_f64(*p), FixedQty::from_f64(*s)))
            .collect();
        self.inner.apply_snapshot(&fp_bids, &fp_asks);
        self.touch();
    }

    /// Incremental delta update.
    pub fn apply_delta(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        for (price, size) in bids {
            let fp = FixedPrice::from_f64(price);
            let fq = if size == 0.0 { FixedQty(0) } else { FixedQty::from_f64(size) };
            self.inner.update_bid(fp, fq);
        }
        for (price, size) in asks {
            let fp = FixedPrice::from_f64(price);
            let fq = if size == 0.0 { FixedQty(0) } else { FixedQty::from_f64(size) };
            self.inner.update_ask(fp, fq);
        }
        // Periodically recenter
        self.inner.recenter();
        self.touch();
    }

    /// Incremental delta update that returns per-level change vectors.
    pub fn apply_delta_tracked(
        &mut self,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> (Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>) {
        let mut bid_changes: Vec<(f64, f64, f64)> = Vec::new();
        let mut ask_changes: Vec<(f64, f64, f64)> = Vec::new();

        for (price, new_size) in bids {
            let fp = FixedPrice::from_f64(price);
            let fq = if new_size == 0.0 { FixedQty(0) } else { FixedQty::from_f64(new_size) };
            let (old, new) = self.inner.apply_delta_tracked(fp, fq, true);
            if old != new {
                bid_changes.push((price, old.to_f64(), new.to_f64()));
            }
        }

        for (price, new_size) in asks {
            let fp = FixedPrice::from_f64(price);
            let fq = if new_size == 0.0 { FixedQty(0) } else { FixedQty::from_f64(new_size) };
            let (old, new) = self.inner.apply_delta_tracked(fp, fq, false);
            if old != new {
                ask_changes.push((price, old.to_f64(), new.to_f64()));
            }
        }

        self.inner.recenter();
        self.touch();
        (bid_changes, ask_changes)
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// O(1) best bid as `(price, size)` or `None`.
    pub fn get_best_bid(&self) -> Option<(f64, f64)> {
        self.inner.best_bid().map(|(p, q)| (p.to_f64(), q.to_f64()))
    }

    /// O(1) best ask as `(price, size)` or `None`.
    pub fn get_best_ask(&self) -> Option<(f64, f64)> {
        self.inner.best_ask().map(|(p, q)| (p.to_f64(), q.to_f64()))
    }

    /// Return the symbol this order book was created for.
    pub fn get_symbol(&self) -> String {
        self.symbol.clone()
    }

    /// Mid-price; returns 0.0 if either side is empty.
    pub fn get_mid_price(&self) -> f64 {
        self.inner.mid_price().to_f64()
    }

    /// Spread in basis points; returns 0.0 if mid is zero.
    pub fn get_spread_bps(&self) -> f64 {
        match (self.inner.best_bid(), self.inner.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => FixedPrice::spread_bps_f64(bid, ask),
            _ => 0.0,
        }
    }

    /// Top `depth` bid levels as `[(price, size), ...]` sorted descending by price.
    pub fn get_bids(&self, depth: usize) -> Vec<(f64, f64)> {
        self.inner.get_bids(depth)
            .iter()
            .map(|(p, q)| (p.to_f64(), q.to_f64()))
            .collect()
    }

    /// Top `depth` ask levels as `[(price, size), ...]` sorted ascending by price.
    pub fn get_asks(&self, depth: usize) -> Vec<(f64, f64)> {
        self.inner.get_asks(depth)
            .iter()
            .map(|(p, q)| (p.to_f64(), q.to_f64()))
            .collect()
    }

    /// Return a Python dict matching the existing Python strategy format.
    pub fn get_snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);

        let bids_list = PyList::empty_bound(py);
        for (price, size) in self.get_bids(100) {
            let pair = PyList::empty_bound(py);
            pair.append(price)?;
            pair.append(size)?;
            bids_list.append(&pair)?;
        }
        dict.set_item("bids", &bids_list)?;

        let asks_list = PyList::empty_bound(py);
        for (price, size) in self.get_asks(100) {
            let pair = PyList::empty_bound(py);
            pair.append(price)?;
            pair.append(size)?;
            asks_list.append(&pair)?;
        }
        dict.set_item("asks", &asks_list)?;
        dict.set_item("timestamp", self.last_update_ms)?;

        Ok(dict.into_any().unbind())
    }

    /// Compute depth USDT for both sides (top N levels).
    pub fn get_depth_usdt(&self, n: usize) -> (f64, f64) {
        let bid_depth = self.inner.bid_depth_usdt(n);
        let ask_depth = self.inner.ask_depth_usdt(n);
        (bid_depth, ask_depth)
    }
}

impl RustOrderBook {
    /// Record the current time as the last update instant.
    fn touch(&mut self) {
        self.last_update = Some(Instant::now());
        self.last_update_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
    }

    /// Access the inner FlatOrderBook (for binary target use).
    pub fn inner(&self) -> &FlatOrderBook {
        &self.inner
    }

    /// Mutable access to the inner FlatOrderBook.
    pub fn inner_mut(&mut self) -> &mut FlatOrderBook {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// RustBookAnalyzer — stateless analytics
// ---------------------------------------------------------------------------

/// Stateless order-book analytics engine.
///
/// All methods are `#[staticmethod]`s operating on a `RustOrderBook` reference.
#[pyclass]
pub struct RustBookAnalyzer;

#[pymethods]
impl RustBookAnalyzer {
    /// Compute rich execution metrics from a `RustOrderBook`.
    ///
    /// Returns a Python dict with the **exact same keys and value types** as
    /// `GateioBookAnalyzer.analyze_book()`.
    #[staticmethod]
    pub fn analyze(py: Python<'_>, book: &RustOrderBook, depth: usize) -> PyResult<PyObject> {
        let bid_levels = book.get_bids(depth);
        let ask_levels = book.get_asks(depth);

        if bid_levels.is_empty() || ask_levels.is_empty() {
            return Self::empty_result(py);
        }

        let best_bid = bid_levels[0].0;
        let best_ask = ask_levels[0].0;
        let mid_price = (best_bid + best_ask) / 2.0;
        let spread_bps = if mid_price > 0.0 {
            ((best_ask - best_bid) / mid_price) * 10_000.0
        } else {
            0.0
        };

        let top_bids: Vec<(f64, f64)> = bid_levels.iter().copied().take(DEPTH_LEVELS).collect();
        let top_asks: Vec<(f64, f64)> = ask_levels.iter().copied().take(DEPTH_LEVELS).collect();

        let bid_depth_usdt: f64 = top_bids.iter().map(|(p, s)| p * s).sum();
        let ask_depth_usdt: f64 = top_asks.iter().map(|(p, s)| p * s).sum();
        let total_depth = bid_depth_usdt + ask_depth_usdt;
        let imbalance = if total_depth > 0.0 {
            (bid_depth_usdt - ask_depth_usdt) / total_depth
        } else {
            0.0
        };

        let large_bids = find_large_levels(&top_bids, py)?;
        let large_asks = find_large_levels(&top_asks, py)?;

        let tick = (best_ask - best_bid) / 2.0;
        let half_tick_threshold = tick * OPTIMAL_PRICE_SPREAD_FRACTION;
        let price_increment = half_tick_threshold.max(MIN_PRICE_TICK);
        let optimal_buy_price = best_bid + price_increment;
        let optimal_sell_price = best_ask - price_increment;

        let dict = PyDict::new_bound(py);
        dict.set_item("imbalance", round4(imbalance))?;
        dict.set_item("spread_bps", round4(spread_bps))?;
        dict.set_item("bid_depth_usdt", round2(bid_depth_usdt))?;
        dict.set_item("ask_depth_usdt", round2(ask_depth_usdt))?;
        dict.set_item("large_bid_levels", large_bids)?;
        dict.set_item("large_ask_levels", large_asks)?;
        dict.set_item("optimal_buy_price", round8(optimal_buy_price))?;
        dict.set_item("optimal_sell_price", round8(optimal_sell_price))?;
        dict.set_item("best_bid", best_bid)?;
        dict.set_item("best_ask", best_ask)?;
        dict.set_item("mid_price", round8(mid_price))?;
        Ok(dict.into_any().unbind())
    }

    /// Estimate the VWAP fill price for `amount_contracts` contracts.
    #[staticmethod]
    pub fn calculate_market_impact(
        book: &RustOrderBook,
        side: &str,
        amount_contracts: f64,
    ) -> f64 {
        if amount_contracts <= 0.0 {
            return 0.0;
        }

        let levels: Vec<(f64, f64)> = if side.to_lowercase() == "buy" {
            book.get_asks(10_000) // Get all available ask levels
        } else {
            book.get_bids(10_000) // Get all available bid levels
        };

        if levels.is_empty() {
            return 0.0;
        }

        let mut remaining = amount_contracts;
        let mut total_cost = 0.0;
        let mut last_price = levels.last().map(|(p, _)| *p).unwrap_or(0.0);

        for (price, qty) in &levels {
            let fill = remaining.min(*qty);
            total_cost += fill * price;
            remaining -= fill;
            last_price = *price;
            if remaining <= 0.0 {
                break;
            }
        }

        if remaining > 0.0 {
            total_cost += remaining * last_price;
        }

        total_cost / amount_contracts
    }

    /// Return the all-zeros empty-result dict.
    #[staticmethod]
    pub fn empty_result(py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("imbalance", 0.0_f64)?;
        dict.set_item("spread_bps", 0.0_f64)?;
        dict.set_item("bid_depth_usdt", 0.0_f64)?;
        dict.set_item("ask_depth_usdt", 0.0_f64)?;
        dict.set_item("large_bid_levels", PyList::empty_bound(py))?;
        dict.set_item("large_ask_levels", PyList::empty_bound(py))?;
        dict.set_item("optimal_buy_price", 0.0_f64)?;
        dict.set_item("optimal_sell_price", 0.0_f64)?;
        dict.set_item("best_bid", 0.0_f64)?;
        dict.set_item("best_ask", 0.0_f64)?;
        dict.set_item("mid_price", 0.0_f64)?;
        Ok(dict.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn find_large_levels(levels: &[(f64, f64)], py: Python<'_>) -> PyResult<PyObject> {
    let result_list = PyList::empty_bound(py);
    if levels.is_empty() {
        return Ok(result_list.into_any().unbind());
    }

    let notionals: Vec<f64> = levels.iter().map(|(p, s)| p * s).collect();
    let avg = notionals.iter().sum::<f64>() / notionals.len() as f64;
    let threshold = avg * LARGE_ORDER_MULTIPLIER;

    for ((price, size), notional) in levels.iter().zip(notionals.iter()) {
        if *notional >= threshold {
            let entry = PyDict::new_bound(py);
            entry.set_item("price", *price)?;
            entry.set_item("size", *size)?;
            entry.set_item("notional_usdt", round2(*notional))?;
            result_list.append(&entry)?;
        }
    }
    Ok(result_list.into_any().unbind())
}

#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[inline]
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

#[inline]
fn round8(v: f64) -> f64 {
    (v * 100_000_000.0).round() / 100_000_000.0
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_orderbook(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "orderbook")?;
    m.add_class::<RustOrderBook>()?;
    m.add_class::<RustBookAnalyzer>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
