//! WebSocket message parser for exchange tick/orderbook/trade feeds.
//!
//! Provides zero-copy JSON parsing via simd-json + serde for the highest-frequency
//! hot path in the system (~100+ messages/second per symbol).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Helper: resolve a numeric value from a serde_json::Value field.
// Handles both JSON numbers and JSON strings (Gate.io sends some as strings).
// ---------------------------------------------------------------------------
fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn value_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

/// Resolve a field by trying multiple key names in order.
fn resolve_f64<'a>(payload: &'a Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = payload.get(key) {
            if let Some(f) = value_to_f64(v) {
                return Some(f);
            }
        }
    }
    None
}

fn resolve_i64<'a>(payload: &'a Value, keys: &[&str]) -> Option<i64> {
    for key in keys {
        if let Some(v) = payload.get(key) {
            if let Some(i) = value_to_i64(v) {
                return Some(i);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// RustTicker — PyO3-exposed ticker object
// ---------------------------------------------------------------------------

/// A parsed ticker snapshot from a WebSocket message.
///
/// Field names and semantics match the existing Python ``Ticker`` Pydantic model
/// so strategy code can consume either object interchangeably.
#[pyclass]
pub struct RustTicker {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub bid: f64,
    #[pyo3(get)]
    pub ask: f64,
    #[pyo3(get)]
    pub last: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub funding_rate: Option<f64>,
    #[pyo3(get)]
    pub open_interest: Option<f64>,
}

#[pymethods]
impl RustTicker {
    fn __repr__(&self) -> String {
        format!(
            "RustTicker(symbol={}, last={}, bid={}, ask={})",
            self.symbol, self.last, self.bid, self.ask
        )
    }
}

// ---------------------------------------------------------------------------
// parse_ticker_message
// ---------------------------------------------------------------------------

/// Parse a raw WebSocket bytes payload into a :class:`RustTicker`.
///
/// Implements the exact same field-resolution logic as
/// ``MarketDataFeed._parse_ticker()`` in ``websocket_feeds.py``:
///
/// * Unwraps the ``data`` key when present; if ``data`` is a list the first
///   element is taken.
/// * Resolves ``last`` from: ``"last"``, ``"c"``, ``"close"``, ``"lastPr"``.
/// * Resolves ``bid`` from: ``"bid"``, ``"b"``, ``"bestBid"``, fallback to last.
/// * Resolves ``ask`` from: ``"ask"``, ``"a"``, ``"bestAsk"``, fallback to last.
/// * Resolves ``high`` from: ``"high"``, ``"h"``, fallback to last.
/// * Resolves ``low`` from: ``"low"``, ``"l"``, fallback to last.
/// * Resolves ``volume`` from: ``"volume"``, ``"v"``, ``"baseVolume"``, fallback to 0.0.
/// * Resolves ``timestamp`` from: ``"timestamp"``, ``"t"``, ``"ts"``, fallback to 0.
///
/// Returns ``None`` (Python ``None``) when the payload is a subscription
/// confirmation with no price fields.
#[pyfunction]
pub fn parse_ticker_message(symbol: &str, raw_json: &[u8]) -> PyResult<Option<RustTicker>> {
    // simd-json mutates the buffer; work on a copy so the caller's data is unaffected.
    let mut buf: Vec<u8> = raw_json.to_vec();

    let root: Value = simd_json::serde::from_slice(&mut buf)
        .map_err(|e| PyValueError::new_err(format!("simd-json parse error: {e}")))?;

    // Unwrap the 'data' key when present
    let inner = root.get("data").unwrap_or(&root);

    // If data is a list, take the first element
    let payload: &Value = if let Value::Array(arr) = inner {
        match arr.first() {
            Some(first) => first,
            None => return Ok(None),
        }
    } else {
        inner
    };

    if !payload.is_object() {
        return Ok(None);
    }

    // Resolve 'last' — if absent this is a subscription-confirmation message
    let last = match resolve_f64(payload, &["last", "c", "close", "lastPr"]) {
        Some(v) => v,
        None => return Ok(None),
    };

    let bid = resolve_f64(payload, &["bid", "b", "bestBid"]).unwrap_or(last);
    let ask = resolve_f64(payload, &["ask", "a", "bestAsk"]).unwrap_or(last);
    let high = resolve_f64(payload, &["high", "h"]).unwrap_or(last);
    let low = resolve_f64(payload, &["low", "l"]).unwrap_or(last);
    let volume = resolve_f64(payload, &["volume", "v", "baseVolume"]).unwrap_or(0.0);
    let timestamp = resolve_i64(payload, &["timestamp", "t", "ts"]).unwrap_or(0);

    let funding_rate = resolve_f64(payload, &["fundingRate", "funding_rate"]);
    let open_interest = resolve_f64(payload, &["openInterest", "open_interest", "oi"]);

    Ok(Some(RustTicker {
        symbol: symbol.to_string(),
        bid,
        ask,
        last,
        high,
        low,
        volume,
        timestamp,
        funding_rate,
        open_interest,
    }))
}

// ---------------------------------------------------------------------------
// parse_orderbook_message
// ---------------------------------------------------------------------------

/// Parse a raw WebSocket order-book payload into a Python dict.
///
/// Returns ``{"bids": [[price, size], ...], "asks": [[price, size], ...], "timestamp": int}``
/// matching the format the existing Python code expects.
///
/// Unwraps the ``data`` key when present (same convention as ``_parse_ticker``).
/// Returns ``None`` when the payload cannot be parsed as an order book.
#[pyfunction]
pub fn parse_orderbook_message(py: Python<'_>, raw_json: &[u8]) -> PyResult<Option<PyObject>> {
    let mut buf: Vec<u8> = raw_json.to_vec();

    let root: Value = simd_json::serde::from_slice(&mut buf)
        .map_err(|e| PyValueError::new_err(format!("simd-json parse error: {e}")))?;

    let inner = root.get("data").unwrap_or(&root);
    let payload: &Value = if let Value::Array(arr) = inner {
        match arr.first() {
            Some(first) => first,
            None => return Ok(None),
        }
    } else {
        inner
    };

    if !payload.is_object() {
        return Ok(None);
    }

    let bids_val = payload.get("bids");
    let asks_val = payload.get("asks");

    let dict = PyDict::new_bound(py);

    let bids_list = PyList::empty_bound(py);
    if let Some(Value::Array(bids)) = bids_val {
        for level in bids {
            let level_list = PyList::empty_bound(py);
            if let Value::Array(pair) = level {
                for item in pair.iter().take(2) {
                    level_list.append(value_to_f64(item).unwrap_or(0.0))?;
                }
            }
            bids_list.append(&level_list)?;
        }
    }
    dict.set_item("bids", &bids_list)?;

    let asks_list = PyList::empty_bound(py);
    if let Some(Value::Array(asks)) = asks_val {
        for level in asks {
            let level_list = PyList::empty_bound(py);
            if let Value::Array(pair) = level {
                for item in pair.iter().take(2) {
                    level_list.append(value_to_f64(item).unwrap_or(0.0))?;
                }
            }
            asks_list.append(&level_list)?;
        }
    }
    dict.set_item("asks", &asks_list)?;

    let ts = resolve_i64(payload, &["timestamp", "t", "ts"]).unwrap_or(0);
    dict.set_item("timestamp", ts)?;

    Ok(Some(dict.into_any().unbind()))
}

// ---------------------------------------------------------------------------
// parse_trade_message
// ---------------------------------------------------------------------------

/// Parse a raw WebSocket trade payload into a Python dict.
///
/// Unwraps the ``data`` key when present and returns the inner object.
/// Returns ``None`` when the payload cannot be parsed.
#[pyfunction]
pub fn parse_trade_message(py: Python<'_>, raw_json: &[u8]) -> PyResult<Option<PyObject>> {
    let mut buf: Vec<u8> = raw_json.to_vec();

    let root: Value = simd_json::serde::from_slice(&mut buf)
        .map_err(|e| PyValueError::new_err(format!("simd-json parse error: {e}")))?;

    let inner = root.get("data").unwrap_or(&root);

    let dict = value_to_pydict(py, inner)?;
    Ok(Some(dict.into_any().unbind()))
}

// ---------------------------------------------------------------------------
// parse_ws_message
// ---------------------------------------------------------------------------

/// Parse any raw WebSocket message bytes into a Python dict.
///
/// This is a general-purpose parser used by ``gateio_client.py`` to replace
/// ``json.loads(raw_msg)`` calls in the WS loops.  Unlike the ticker-specific
/// parser it does not do any field resolution — it returns the full decoded
/// message as a Python dict.
#[pyfunction]
pub fn parse_ws_message(py: Python<'_>, raw_json: &[u8]) -> PyResult<PyObject> {
    let mut buf: Vec<u8> = raw_json.to_vec();

    let root: Value = simd_json::serde::from_slice(&mut buf)
        .map_err(|e| PyValueError::new_err(format!("simd-json parse error: {e}")))?;

    let dict = value_to_pydict(py, &root)?;
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// detect_significant_move
// ---------------------------------------------------------------------------

/// Return ``True`` when ``|new_price - old_price| / old_price >= threshold``.
///
/// Pure arithmetic helper — replaces the Python expression in
/// ``WebSocketDataManager._update_ticker``.
#[pyfunction]
pub fn detect_significant_move(old_price: f64, new_price: f64, threshold: f64) -> bool {
    if old_price <= 0.0 {
        return false;
    }
    ((new_price - old_price).abs() / old_price) >= threshold
}

// ---------------------------------------------------------------------------
// Internal: convert serde_json::Value → Python object
// ---------------------------------------------------------------------------

fn value_to_pyobject(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    use pyo3::IntoPy;
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok((*b).into_py(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.clone().into_py(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(value_to_pyobject(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(_) => {
            let dict = value_to_pydict(py, value)?;
            Ok(dict.into_any().unbind())
        }
    }
}

fn value_to_pydict<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    if let Value::Object(map) = value {
        for (k, v) in map {
            dict.set_item(k, value_to_pyobject(py, v)?)?;
        }
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_ws_parser(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "ws_parser")?;
    m.add_class::<RustTicker>()?;
    m.add_function(wrap_pyfunction!(parse_ticker_message, &m)?)?;
    m.add_function(wrap_pyfunction!(parse_orderbook_message, &m)?)?;
    m.add_function(wrap_pyfunction!(parse_trade_message, &m)?)?;
    m.add_function(wrap_pyfunction!(parse_ws_message, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_significant_move, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
