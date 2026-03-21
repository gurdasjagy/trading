//! Phase 3: Rust-native tick processor — VWAP, tick imbalance, and VPIN.
//!
//! Maintains a per-symbol ring buffer of ``ParsedTick`` entries and updates
//! running accumulators incrementally so that ``get_vwap()`` and
//! ``get_tick_imbalance()`` are O(1) lookups rather than O(N) scans.

use std::collections::{HashMap, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Kahan compensated summation to reduce floating-point drift over many ticks.
#[derive(Clone, Default)]
pub(crate) struct KahanSum {
    pub(crate) sum: f64,
    pub(crate) compensation: f64,
}

impl KahanSum {
    pub(crate) fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    pub(crate) fn sub(&mut self, value: f64) {
        self.add(-value);
    }

    pub(crate) fn get(&self) -> f64 {
        self.sum
    }
}

/// Pre-parsed tick — avoids repeated string/float coercion on hot path.
#[derive(Clone)]
struct ParsedTick {
    price: f64,
    amount: f64,
    is_buy: bool,
}

/// Accumulating VPIN volume bucket.
#[derive(Clone, Default)]
struct VpinBucket {
    buy_vol: f64,
    sell_vol: f64,
    total_vol: f64,
}

// ---------------------------------------------------------------------------
// Per-symbol state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SymbolState {
    /// Ring buffer of parsed ticks (bounded by window_size).
    ticks: VecDeque<ParsedTick>,
    /// Incremental price×volume accumulator.
    running_pv: KahanSum,
    /// Incremental volume accumulator.
    running_vol: KahanSum,
    /// Incremental buy-volume accumulator.
    running_buy_vol: KahanSum,
    /// Incremental sell-volume accumulator.
    running_sell_vol: KahanSum,
    /// Currently open VPIN bucket.
    vpin_current: VpinBucket,
    /// Completed VPIN bucket estimates (max 200 entries).
    vpin_history: VecDeque<f64>,
}

// ---------------------------------------------------------------------------
// RustTickProcessor
// ---------------------------------------------------------------------------

/// High-performance tick processor exposed to Python via PyO3.
///
/// Maintains per-symbol ring buffers and incremental accumulators for
/// VWAP, tick imbalance, and VPIN computation.
#[pyclass]
pub struct RustTickProcessor {
    window_size: usize,
    vpin_bucket_size: f64,
    symbols: HashMap<String, SymbolState>,
}

#[pymethods]
impl RustTickProcessor {
    /// Create a new tick processor.
    ///
    /// Args:
    ///     window_size: Number of ticks to retain in the rolling window.
    ///     vpin_bucket_size: Target volume per VPIN bucket.
    #[new]
    #[pyo3(signature = (window_size=1000, vpin_bucket_size=1000.0))]
    pub fn new(window_size: usize, vpin_bucket_size: f64) -> Self {
        Self {
            window_size,
            vpin_bucket_size,
            symbols: HashMap::new(),
        }
    }

    /// Ingest a single tick with pre-extracted numeric values.
    ///
    /// Args:
    ///     symbol: Trading pair symbol.
    ///     price: Trade price (must be > 0).
    ///     amount: Trade volume (must be > 0).
    ///     side: ``"buy"`` or ``"sell"`` (case-insensitive).
    pub fn process_tick(&mut self, symbol: &str, price: f64, amount: f64, side: &str) {
        if price.is_nan() || amount.is_nan() || price <= 0.0 || amount <= 0.0 {
            return;
        }

        let is_buy = side.eq_ignore_ascii_case("buy");
        let tick = ParsedTick { price, amount, is_buy };

        let state = self.symbols.entry(symbol.to_string()).or_default();

        // Evict oldest tick if ring buffer is full.
        if state.ticks.len() >= self.window_size {
            if let Some(old) = state.ticks.pop_front() {
                state.running_pv.sub(old.price * old.amount);
                state.running_vol.sub(old.amount);
                if old.is_buy {
                    state.running_buy_vol.sub(old.amount);
                } else {
                    state.running_sell_vol.sub(old.amount);
                }
            }
        }

        // Accumulate new tick.
        state.running_pv.add(tick.price * tick.amount);
        state.running_vol.add(tick.amount);
        if tick.is_buy {
            state.running_buy_vol.add(tick.amount);
        } else {
            state.running_sell_vol.add(tick.amount);
        }
        state.ticks.push_back(tick);

        // Update VPIN bucket.
        {
            let bucket = &mut state.vpin_current;
            if is_buy {
                bucket.buy_vol += amount;
            } else {
                bucket.sell_vol += amount;
            }
            bucket.total_vol += amount;

            if bucket.total_vol >= self.vpin_bucket_size {
                let total = bucket.total_vol;
                if total > 0.0 {
                    let estimate = (bucket.buy_vol - bucket.sell_vol).abs() / total;
                    state.vpin_history.push_back(estimate);
                    if state.vpin_history.len() > 200 {
                        state.vpin_history.pop_front();
                    }
                }
                *bucket = VpinBucket::default();
            }
        }
    }

    /// Convenience method: extract price/amount/side from a Python dict and
    /// call ``process_tick()``.  Matches the key resolution logic in the
    /// Python ``TickProcessor.process_tick()``.
    pub fn process_tick_dict(
        &mut self,
        symbol: &str,
        tick: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let price = extract_f64_from_dict(tick, &["price", "last"])?;
        let amount = extract_f64_from_dict(tick, &["amount", "size"])?;
        let side = extract_str_from_dict(tick, "side");
        if price > 0.0 && amount > 0.0 {
            self.process_tick(symbol, price, amount, &side);
        }
        Ok(())
    }

    /// Batch ingestion: list of ``(price, amount, side)`` tuples.
    pub fn process_ticks(&mut self, symbol: &str, ticks: Vec<(f64, f64, String)>) {
        for (price, amount, side) in ticks {
            self.process_tick(symbol, price, amount, &side);
        }
    }

    /// Return the volume-weighted average price over the rolling window.
    ///
    /// O(1) — reads from incremental accumulators.
    pub fn get_vwap(&self, symbol: &str) -> f64 {
        let state = match self.symbols.get(symbol) {
            Some(s) if !s.ticks.is_empty() => s,
            _ => return 0.0,
        };
        let vol = state.running_vol.get();
        if vol <= 0.0 {
            return 0.0;
        }
        let pv = state.running_pv.get();
        pv / vol
    }

    /// Return tick imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol).
    ///
    /// Range: [-1, +1].  O(1).
    pub fn get_tick_imbalance(&self, symbol: &str) -> f64 {
        let state = match self.symbols.get(symbol) {
            Some(s) if !s.ticks.is_empty() => s,
            _ => return 0.0,
        };
        let buy = state.running_buy_vol.get().max(0.0);
        let sell = state.running_sell_vol.get().max(0.0);
        let total = buy + sell;
        if total <= 0.0 {
            return 0.0;
        }
        (buy - sell) / total
    }

    /// Return volume-weighted mid-price blending VWAP with order-book mid.
    ///
    /// Formula: ``(vwap + (bid + ask) / 2) / 2``.
    pub fn get_vwap_mid_price(&self, symbol: &str, bid: f64, ask: f64) -> f64 {
        let vwap = self.get_vwap(symbol);
        let book_mid = if bid > 0.0 && ask > 0.0 {
            (bid + ask) / 2.0
        } else {
            0.0
        };
        if vwap <= 0.0 {
            return book_mid;
        }
        if book_mid <= 0.0 {
            return vwap;
        }
        (vwap + book_mid) / 2.0
    }

    /// Return the current VPIN estimate (average of last 50 completed buckets).
    ///
    /// Returns 0.0 if fewer than 2 buckets have been completed.
    pub fn get_vpin(&self, symbol: &str) -> f64 {
        let state = match self.symbols.get(symbol) {
            Some(s) => s,
            None => return 0.0,
        };
        let history = &state.vpin_history;
        if history.len() < 2 {
            return 0.0;
        }
        let last_50: Vec<f64> = history.iter().rev().take(50).cloned().collect();
        let sum: f64 = last_50.iter().sum();
        sum / last_50.len() as f64
    }

    /// Return all microstructure metrics for *symbol* as a Python dict.
    pub fn get_metrics(&self, symbol: &str, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("vwap", self.get_vwap(symbol))?;
        dict.set_item("tick_imbalance", self.get_tick_imbalance(symbol))?;
        dict.set_item("vpin", self.get_vpin(symbol))?;
        dict.set_item("tick_count", self.get_tick_count(symbol))?;
        Ok(dict.into())
    }

    /// Return the number of ticks in the rolling window for *symbol*.
    pub fn get_tick_count(&self, symbol: &str) -> usize {
        self.symbols
            .get(symbol)
            .map(|s| s.ticks.len())
            .unwrap_or(0)
    }

    /// Process a tick **and** return the updated basic metrics.
    ///
    /// This is the preferred method when the caller also has the current book
    /// mid-price available.  The mid-price is passed through so that the caller
    /// can forward it to ``MicrostructureEngine::on_trade()`` for accurate
    /// Lee-Ready trade classification without a second ``get_mid_price()`` call.
    ///
    /// Returns ``(vwap, tick_imbalance, vpin)``.
    pub fn process_tick_with_book(
        &mut self,
        symbol: &str,
        price: f64,
        amount: f64,
        side: &str,
        _mid_price: f64,
    ) -> (f64, f64, f64) {
        self.process_tick(symbol, price, amount, side);
        (
            self.get_vwap(symbol),
            self.get_tick_imbalance(symbol),
            self.get_vpin(symbol),
        )
    }
}

// ---------------------------------------------------------------------------
// Dict extraction helpers (for process_tick_dict)
// ---------------------------------------------------------------------------

fn extract_f64_from_dict(dict: &Bound<'_, PyDict>, keys: &[&str]) -> PyResult<f64> {
    for key in keys {
        if let Some(val) = dict.get_item(key)? {
            if val.is_none() {
                continue;
            }
            if let Ok(f) = val.extract::<f64>() {
                if f.is_finite() {
                    return Ok(f);
                }
            }
            // Try string → float coercion.
            if let Ok(s) = val.extract::<String>() {
                if let Ok(f) = s.parse::<f64>() {
                    return Ok(f);
                }
            }
        }
    }
    Ok(0.0)
}

fn extract_str_from_dict(dict: &Bound<'_, PyDict>, key: &str) -> String {
    dict.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default()
        .to_lowercase()
}
