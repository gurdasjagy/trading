//! Phase 4: Pre-submission execution plan computation.
//!
//! Consolidates fee calculation, contract/base amount sizing, precision
//! rounding, and venue scoring into pure-Rust functions that are called
//! synchronously from within the async Python execution pipeline.
//!
//! Expected latency: ~10–50 µs per call (vs. 1–5 ms in Python).

use std::collections::HashMap;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// MarketInfo — cached per-symbol metadata
// ---------------------------------------------------------------------------

/// Cached market metadata for a single symbol.
#[pyclass]
#[derive(Clone)]
pub struct MarketInfo {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub is_contract: bool,
    #[pyo3(get)]
    pub is_inverse: bool,
    #[pyo3(get)]
    pub contract_size: f64,
    #[pyo3(get)]
    pub min_amount: f64,
    #[pyo3(get)]
    pub step_size: f64,
    #[pyo3(get)]
    pub price_precision: f64,
}

#[pymethods]
impl MarketInfo {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: &str,
        is_contract: bool,
        is_inverse: bool,
        contract_size: f64,
        min_amount: f64,
        step_size: f64,
        price_precision: f64,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            is_contract,
            is_inverse,
            contract_size: if contract_size > 0.0 { contract_size } else { 1.0 },
            min_amount,
            step_size,
            price_precision,
        }
    }
}

// ---------------------------------------------------------------------------
// FeeTable — pre-loaded per-exchange fee rates
// ---------------------------------------------------------------------------

/// Pre-loaded exchange fee rates: ``exchange_name → (maker_rate, taker_rate)``.
///
/// Default rates match ``fee_calculator.py``.
#[pyclass]
#[derive(Clone)]
pub struct FeeTable {
    /// ``exchange_name.to_lowercase() → (maker_rate, taker_rate)``
    rates: HashMap<String, (f64, f64)>,
}

#[pymethods]
impl FeeTable {
    /// Create a ``FeeTable`` pre-populated with default exchange rates.
    #[new]
    pub fn new() -> Self {
        let mut rates = HashMap::new();
        rates.insert("mexc".to_string(), (0.0002, 0.0006));
        rates.insert("gateio".to_string(), (-0.00025, 0.00075));
        rates.insert("bingx".to_string(), (0.0002, 0.0005));
        rates.insert("bitget".to_string(), (0.0002, 0.0006));
        Self { rates }
    }

    /// Add or update the fee rate for *exchange*.
    pub fn set_rate(&mut self, exchange: &str, maker: f64, taker: f64) {
        self.rates.insert(exchange.to_lowercase(), (maker, taker));
    }

    /// Return the exchange with the lowest fee for *order_type*.
    ///
    /// ``order_type``: ``"limit"`` (uses maker rate) or ``"market"`` (taker).
    pub fn get_cheapest_exchange(&self, exchanges: Vec<String>, order_type: &str) -> String {
        let use_maker = order_type.eq_ignore_ascii_case("limit");
        let mut best: Option<(String, f64)> = None;
        for ex in &exchanges {
            let key = ex.to_lowercase();
            let rate = self.rates.get(&key).map_or(0.0006, |(maker, taker)| {
                if use_maker { *maker } else { *taker }
            });
            match best {
                None => best = Some((ex.clone(), rate)),
                Some((_, br)) if rate < br => best = Some((ex.clone(), rate)),
                _ => {}
            }
        }
        best.map(|(name, _)| name)
            .unwrap_or_else(|| exchanges.into_iter().next().unwrap_or_default())
    }

    /// Return the taker fee for *exchange* (used internally by compute_execution_plan).
    fn taker_fee(&self, exchange: &str) -> f64 {
        let key = exchange.to_lowercase();
        self.rates.get(&key).map_or(0.0006, |(_, taker)| *taker)
    }
}

// ---------------------------------------------------------------------------
// ExecutionPlan — output of compute_execution_plan
// ---------------------------------------------------------------------------

/// Fully-computed pre-submission execution plan.
///
/// Replaces the Python fee/sizing/precision blocks in
/// ``TradeExecutor.execute_trade()``.
#[pyclass]
#[derive(Clone)]
pub struct ExecutionPlan {
    /// Final contract or base amount after all calculations.
    #[pyo3(get)]
    pub amount_to_order: f64,
    #[pyo3(get)]
    pub is_contract: bool,
    /// Notional position value in USDT.
    #[pyo3(get)]
    pub total_notional_usdt: f64,
    /// Margin required (notional / leverage).
    #[pyo3(get)]
    pub required_margin: f64,
    /// Estimated round-trip fee.
    #[pyo3(get)]
    pub estimated_round_trip_fee: f64,
    /// Position size after fee deduction.
    #[pyo3(get)]
    pub fee_adjusted_size: f64,
    /// Whether the trade passes viability checks.
    #[pyo3(get)]
    pub is_viable: bool,
    /// Empty when viable; descriptive message on rejection.
    #[pyo3(get)]
    pub rejection_reason: String,
    /// Recommended order type: ``"market"``, ``"limit"``, or ``"post_only"``.
    #[pyo3(get)]
    pub optimal_order_type: String,
    /// Slippage cap percentage.
    #[pyo3(get)]
    pub slippage_cap_pct: f64,
}

#[pymethods]
impl ExecutionPlan {}

// ---------------------------------------------------------------------------
// compute_execution_plan
// ---------------------------------------------------------------------------

/// Compute a fully-validated execution plan from a trade signal + market state.
///
/// This single Rust call replaces the Python fee-calculation, contract-sizing,
/// and precision-rounding blocks in ``TradeExecutor.execute_trade()``.
///
/// Returns an ``ExecutionPlan`` with ``is_viable = false`` (never panics) when
/// any input is invalid (NaN, zero price, etc.).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_execution_plan(
    market_info: &MarketInfo,
    fee_table: &FeeTable,
    exchange_name: &str,
    current_price: f64,
    position_size_usdt: f64,
    leverage: i32,
    direction: &str,
    signal_confidence: f64,
    book_imbalance: f64,
    spread_bps: f64,
    expected_profit_pct: f64,
    max_entry_slippage_pct: f64,
) -> PyResult<ExecutionPlan> {
    // ---- NaN / safety guards ------------------------------------------------
    let reject = |reason: &str| -> PyResult<ExecutionPlan> {
        Ok(ExecutionPlan {
            amount_to_order: 0.0,
            is_contract: market_info.is_contract,
            total_notional_usdt: 0.0,
            required_margin: 0.0,
            estimated_round_trip_fee: 0.0,
            fee_adjusted_size: 0.0,
            is_viable: false,
            rejection_reason: reason.to_string(),
            optimal_order_type: "market".to_string(),
            slippage_cap_pct: max_entry_slippage_pct,
        })
    };

    if current_price.is_nan() || current_price.is_infinite() {
        return reject("current_price is NaN or infinite");
    }
    if current_price <= 0.0 {
        return reject("current_price must be > 0");
    }
    if position_size_usdt.is_nan() || position_size_usdt <= 0.0 {
        return reject("position_size_usdt must be > 0");
    }

    let lev = leverage.max(1) as f64;

    // ---- Fee-adjusted sizing ------------------------------------------------
    let entry_fee_rate = fee_table.taker_fee(exchange_name).abs();
    let total_notional_usdt = position_size_usdt * lev;
    let estimated_round_trip_fee = total_notional_usdt * entry_fee_rate * 2.0;
    let fee_buffer = estimated_round_trip_fee * 1.1;

    if fee_buffer >= position_size_usdt {
        return reject(&format!(
            "Position size ({:.2} USDT) is too small to cover estimated fees ({:.2} USDT)",
            position_size_usdt, fee_buffer,
        ));
    }

    let fee_adjusted_size = position_size_usdt - fee_buffer;

    // ---- Contract / base amount calculation ---------------------------------
    let raw_amount: f64 = if market_info.is_contract {
        if market_info.is_inverse {
            // Inverse contracts: amount in USD value divided by contract_size
            position_size_usdt / market_info.contract_size
        } else {
            // Linear contracts: whole integer contracts
            let notional = position_size_usdt * lev;
            (notional / current_price / market_info.contract_size).floor()
        }
    } else {
        // Spot / base currency
        fee_adjusted_size / current_price
    };

    // Round up to minimum 1 contract for contract markets
    let mut amount = if market_info.is_contract && raw_amount < 1.0 {
        1.0
    } else {
        raw_amount
    };

    // ---- Precision rounding (step_size) -------------------------------------
    if market_info.step_size > 0.0 {
        amount = (amount / market_info.step_size).floor() * market_info.step_size;
    }

    // ---- Viability checks ---------------------------------------------------
    if amount <= 0.0 {
        return reject(&format!(
            "Calculated amount ({:.8}) is <= 0 after rounding",
            amount,
        ));
    }
    if market_info.min_amount > 0.0 && amount < market_info.min_amount {
        return reject(&format!(
            "Order size ({:.8}) is below minimum ({:.8})",
            amount, market_info.min_amount,
        ));
    }

    // Fee viability: break-even check
    let profit_threshold = expected_profit_pct * 0.5;
    let break_even_pct = if total_notional_usdt > 0.0 {
        estimated_round_trip_fee / total_notional_usdt * 100.0
    } else {
        0.0
    };
    if break_even_pct >= profit_threshold && profit_threshold > 0.0 {
        return reject(&format!(
            "Fee break-even ({:.4}%) >= 50% of expected profit ({:.4}%)",
            break_even_pct, expected_profit_pct,
        ));
    }

    // ---- Optimal order type -------------------------------------------------
    let optimal_order_type = if signal_confidence > 0.8 && spread_bps < 3.0 {
        "post_only"
    } else if signal_confidence > 0.6 && spread_bps < 5.0 {
        "limit"
    } else {
        "market"
    };

    // ---- Margin requirement -------------------------------------------------
    let required_margin = total_notional_usdt / lev;

    // ---- Suppress unused variable warning -----------------------------------
    let _ = (direction, book_imbalance);

    Ok(ExecutionPlan {
        amount_to_order: amount,
        is_contract: market_info.is_contract,
        total_notional_usdt,
        required_margin,
        estimated_round_trip_fee,
        fee_adjusted_size,
        is_viable: true,
        rejection_reason: String::new(),
        optimal_order_type: optimal_order_type.to_string(),
        slippage_cap_pct: max_entry_slippage_pct,
    })
}

// ---------------------------------------------------------------------------
// score_venues
// ---------------------------------------------------------------------------

/// Score and rank venues, returning ``[(venue_name, allocation_pct, expected_fee)]``.
///
/// Implements the same weighted scoring formula as
/// ``SmartOrderRouter._score_venues()`` + ``_calculate_routing_allocation()``.
///
/// Args:
///     venues: ``[(name, taker_fee, relevant_liquidity, spread_pct, reliability_score, historical_fill_rate)]``
///     amount: Total order amount (for liquidity score normalisation).
///     min_venue_allocation: Minimum allocation fraction to retain a venue.
///     max_venues: Maximum number of venues to include.
///
/// Returns:
///     Sorted list of ``(venue_name, allocation_pct, expected_fee)`` tuples.
#[pyfunction]
pub fn score_venues(
    venues: Vec<(String, f64, f64, f64, f64, f64)>,
    amount: f64,
    min_venue_allocation: f64,
    max_venues: usize,
) -> PyResult<Vec<(String, f64, f64)>> {
    if venues.is_empty() || amount <= 0.0 {
        return Ok(vec![]);
    }

    // Compute weighted scores.
    let mut scored: Vec<(String, f64, f64)> = venues
        .iter()
        .map(|(name, taker_fee, relevant_liq, spread_pct, reliability, fill_rate)| {
            let liquidity_score = (relevant_liq / (amount * 2.0)).min(1.0);
            let raw_fee_score = 1.0 - (taker_fee / 0.002);
            let fee_score = raw_fee_score.clamp(0.0, 1.0);
            let spread_score = 1.0 - (spread_pct / 0.5).min(1.0);
            let total_score = liquidity_score * 0.40
                + fee_score * 0.25
                + spread_score * 0.20
                + reliability * 0.10
                + fill_rate * 0.05;
            (name.clone(), total_score, *taker_fee)
        })
        .collect();

    // Sort descending by score.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top N venues.
    scored.truncate(max_venues);

    if scored.is_empty() {
        return Ok(vec![]);
    }

    // Proportional allocation.
    let total_score: f64 = scored.iter().map(|(_, s, _)| *s).sum();
    let alloc: Vec<(String, f64, f64)> = if total_score <= 0.0 {
        let equal = 1.0 / scored.len() as f64;
        scored
            .iter()
            .map(|(name, _, fee)| (name.clone(), equal, *fee))
            .collect()
    } else {
        scored
            .iter()
            .map(|(name, score, fee)| (name.clone(), score / total_score, *fee))
            .collect()
    };

    // Filter below minimum allocation.
    let mut filtered: Vec<(String, f64, f64)> = alloc
        .into_iter()
        .filter(|(_, pct, _)| *pct >= min_venue_allocation)
        .collect();

    if filtered.is_empty() {
        return Ok(vec![]);
    }

    // Renormalise.
    let total_pct: f64 = filtered.iter().map(|(_, p, _)| *p).sum();
    if total_pct > 0.0 {
        for entry in &mut filtered {
            entry.1 /= total_pct;
        }
    }

    // Attach expected fee: allocation_pct * amount * taker_fee
    let result: Vec<(String, f64, f64)> = filtered
        .into_iter()
        .map(|(name, alloc_pct, taker_fee)| {
            let expected_fee = alloc_pct * amount * taker_fee;
            (name, alloc_pct, expected_fee)
        })
        .collect();

    Ok(result)
}
