//! ``rust_trading_engine`` — PyO3 extension module.
//!
//! Exposes five sub-modules (PyO3 cdylib for Python backward compatibility):
//! * ``rust_trading_engine.ws_parser``      — Phase 1: WebSocket JSON parsing
//! * ``rust_trading_engine.orderbook``      — Phase 2: order-book engine & analytics
//! * ``rust_trading_engine.tick_processor`` — Phase 3: VWAP/imbalance/VPIN ring buffer
//! * ``rust_trading_engine.execution_prep`` — Phase 4: pre-submission execution plan
//! * ``rust_trading_engine.microstructure`` — Phase 5: synthetic L3 microstructure engine
//!
//! The standalone binary (`trading_engine`) is defined in `main.rs`.
//!
//! **Issue 1**: Added fixed_point, flat_book, and spsc modules.
//! The orderbook module now uses FlatOrderBook internally but presents
//! the same PyO3 interface to Python.
//!
//! **Institutional Upgrade**: Added pre_trade_risk, position_slot_manager,
//! exit_evaluator, event_sequencer, and dust_tracker modules.

// Modules shared between binary and library
pub mod config;
pub mod execution_gateway;
pub mod gateio_gateway;
pub mod forex_gateway;   // Mandate 3: Forex hot-path execution
pub mod strategy_engine;
pub mod telemetry;
pub mod ws_ingestion;
pub mod regime;
pub mod microstructure;

// New Issue 1 modules — foundational data structures
pub mod fixed_point;
pub mod flat_book;
pub mod spsc;

// New Issue 2 modules — event sourcing & lock-free IPC
pub mod journal;
pub mod shared_state;
pub mod regime_shm;

// New Issue 3 modules — institutional execution, MBO & QPE
pub mod execution_state;
pub mod mbo_book;
pub mod adverse_selection;
pub mod smart_router;
pub mod ws_order_manager;

// Institutional Upgrade modules
pub mod pre_trade_risk;
pub mod position_slot_manager;
pub mod exit_evaluator;
pub mod candle_aggregator;
pub mod event_sequencer;
pub mod dust_tracker;
pub mod circuit_breaker;
pub mod trade_flow_analyzer;
pub mod correlation_limiter;
pub mod telegram_alert;
pub mod cumulative_delta;
pub mod volume_profile;
pub mod liquidation_detector;
pub mod trend_strength;
pub mod execution_analytics;
pub mod realized_vol;
pub mod wyckoff_detector;
pub mod fibonacci_detector;
pub mod ichimoku_cloud;
pub mod market_maker_inventory;
pub mod cross_asset_correlation;
pub mod gamma_shm;
pub mod funding_rate;
pub mod queue_position_estimator;
pub mod fast_ws_parser;
pub mod term_structure;
pub mod calendar_spread_engine;
pub mod matching_engine;
pub mod synthetic_orders;
pub mod ml_weight_receiver;

// Directive 1: Rust-native position lifecycle (replaces Python TradeTracker)
pub mod position_lifecycle;
// Directive 2: Exchange-aware position sizing with quanto_multiplier
pub mod position_sizer;
// Directive 4: Smart entry, volatility trailing, adverse selection guard
pub mod smart_entry;
// Directive 5: Dual token-bucket rate limiter
pub mod rate_limiter;
// Dashboard HTTP server for real-time data
pub mod dashboard_server;

// Phase 4 Architecture: Decimal extension, event bus, bridge IPC, risk calculator
pub mod decimal_ext;
pub mod event_bus;
pub mod bridge_ipc;
pub mod risk_calculator;

// Phase 4: SHM signal queue for Alpha Oracle architecture
pub mod signal_queue;
// Phase 2: Event-sourced order state machine
pub mod order_state_machine;
// Latency tracking
pub mod latency_tracker;
// Market impact estimation
pub mod market_impact;
// Order lifecycle tracking
pub mod order_lifecycle;

// ── Institutional Features (from update.txt) ──
pub mod hw_timestamp;    // Feature 1: Hardware Timestamp Support
pub mod tick_store;      // Feature 2: Tick Database with Replay
pub mod var_engine;      // Feature 3: Real-Time VaR Engine
pub mod options_greeks;  // Feature 4: Options Greeks (Black-Scholes)
pub mod arbitrage_engine; // Feature 5: Multi-Exchange Arbitrage
pub mod fee_optimizer;   // Feature 6: Maker Rebate Optimization
pub mod alert_manager;   // Feature 8: Enhanced Monitoring & Alerting
pub mod twap_executor;   // Feature 7: Adaptive TWAP (with AdaptiveTwap)

// ── Multi-Exchange Feature (USE_MULTI_EXCHANGE) ──
pub mod multi_exchange;  // Global order book, SOR, funding arb, margin monitor

// ── Multi-Exchange Gateway Support (for arbitrage) ──
pub mod bybit_gateway;   // Bybit v5 unified API gateway
pub mod binance_gateway; // Binance Futures gateway

// ── Dynamic Instrument Manager (real-time contract specs from all exchanges) ──
pub mod instrument_manager;

// ── Comprehensive Upgrade: New modules ──
pub mod size_normalizer;    // BUG 3: USDT-to-contracts conversion
pub mod funding_timing;     // FEAT 1: Funding rate timestamp-aware entry/exit

// PyO3-specific modules (only compiled into the cdylib)
mod execution_prep;
mod orderbook;
mod tick_processor;
mod ws_parser;

use pyo3::prelude::*;

/// Register a submodule and insert it into ``sys.modules`` so that
/// ``from rust_trading_engine.<name> import ...`` works at runtime.
fn add_submodule(
    parent: &Bound<'_, PyModule>,
    child: Bound<'_, PyModule>,
    qualified_name: &str,
) -> PyResult<()> {
    let py = parent.py();
    parent.add_submodule(&child)?;
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item(qualified_name, child)?;
    Ok(())
}

#[pymodule]
fn rust_trading_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build and register ws_parser submodule
    let ws_mod = PyModule::new_bound(m.py(), "ws_parser")?;
    ws_mod.add_class::<ws_parser::RustTicker>()?;
    ws_mod.add_function(wrap_pyfunction!(ws_parser::parse_ticker_message, &ws_mod)?)?;
    ws_mod.add_function(wrap_pyfunction!(ws_parser::parse_orderbook_message, &ws_mod)?)?;
    ws_mod.add_function(wrap_pyfunction!(ws_parser::parse_trade_message, &ws_mod)?)?;
    ws_mod.add_function(wrap_pyfunction!(ws_parser::parse_ws_message, &ws_mod)?)?;
    ws_mod.add_function(wrap_pyfunction!(ws_parser::detect_significant_move, &ws_mod)?)?;
    add_submodule(m, ws_mod, "rust_trading_engine.ws_parser")?;

    // Build and register orderbook submodule
    let ob_mod = PyModule::new_bound(m.py(), "orderbook")?;
    ob_mod.add_class::<orderbook::RustOrderBook>()?;
    ob_mod.add_class::<orderbook::RustBookAnalyzer>()?;
    add_submodule(m, ob_mod, "rust_trading_engine.orderbook")?;

    // Build and register tick_processor submodule
    let tp_mod = PyModule::new_bound(m.py(), "tick_processor")?;
    tp_mod.add_class::<tick_processor::RustTickProcessor>()?;
    add_submodule(m, tp_mod, "rust_trading_engine.tick_processor")?;

    // Build and register execution_prep submodule
    let ep_mod = PyModule::new_bound(m.py(), "execution_prep")?;
    ep_mod.add_class::<execution_prep::MarketInfo>()?;
    ep_mod.add_class::<execution_prep::FeeTable>()?;
    ep_mod.add_class::<execution_prep::ExecutionPlan>()?;
    ep_mod.add_function(wrap_pyfunction!(execution_prep::compute_execution_plan, &ep_mod)?)?;
    ep_mod.add_function(wrap_pyfunction!(execution_prep::score_venues, &ep_mod)?)?;
    add_submodule(m, ep_mod, "rust_trading_engine.execution_prep")?;

    Ok(())
}

