//! Standalone Rust trading engine binary — Gate.io Only Refactor.
//!
//! Architecture: Rust is the SOLE master executable. The entire hot-path lives in Rust:
//!   WS Ingestion → L2/L3 Book Build → Microstructure Calc → Signal Trigger → Execution Routing
//!
//! Python becomes a gRPC/Shared-Memory COLD microservice for:
//!   - Offline ML model training and regime detection
//!   - Daily parameter recalibration
//!   - Dashboard and monitoring
//!
//! Communication: Python writes to /dev/shm shared memory buffers. Rust reads them
//! on its own schedule with ZERO blocking. No ZeroMQ. No JSON serialization on hot path.
//!
//! Exchange: Gate.io is the SOLE exchange for all modes:
//!   - live:        Gate.io mainnet futures (GATEIO_API_KEY / GATEIO_SECRET_KEY)
//!   - testnet:     Gate.io testnet futures (GATEIO_TESTNET_API_KEY / GATEIO_TESTNET_SECRET_KEY)
//!   - forex_live:  Gate.io TradFi MT5      (GATEIO_TRADFI_LOGIN / PASSWORD / SERVER)
//!   - forex_demo:  Gate.io TradFi demo MT5 (GATEIO_TRADFI_DEMO_LOGIN / PASSWORD / SERVER)
//!
//! Thread Topology:
//!   Core 2:     WS Ingestion — Gate.io (busy-spin)
//!   Core 3:     Orderbook Builder
//!   Core 4:     Signal/Strategy Evaluator + Market Impact
//!   Core 5:     Execution Router + Order Lifecycle + Latency Tracking
//!   Core 6:     Telemetry/Journaling (append-only event log)
//!   Core 7-10:  Microstructure Analytics (VPIN, Kyle Lambda, QPE)

mod config;
mod execution_gateway;
mod gateio_gateway;
mod forex_gateway;
mod orderbook;
mod tick_processor;
mod regime;
mod regime_shm;
mod microstructure;
mod strategy_engine;
mod ml_weight_receiver;
mod telemetry;
mod ws_ingestion;
mod fixed_point;
mod flat_book;
mod spsc;
mod journal;
mod shared_state;
mod candle_aggregator;

// Issue 3: Institutional execution modules
mod execution_state;
mod mbo_book;
mod adverse_selection;
mod smart_router;
mod ws_order_manager;
mod queue_position_estimator;
mod fast_ws_parser;
mod term_structure;
mod calendar_spread_engine;
mod matching_engine;
mod synthetic_orders;

// Institutional upgrades: order lifecycle, latency tracking, market impact
mod order_lifecycle;
mod latency_tracker;
mod market_impact;

// Phase 4 Architecture: Decimal extension, event bus, bridge IPC, risk calculator
mod decimal_ext;
mod event_bus;
mod bridge_ipc;
mod risk_calculator;

// Multi-Exchange Feature (USE_MULTI_EXCHANGE=on)
mod multi_exchange;
mod binance_gateway;
mod bybit_gateway;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use futures_util::future;
use tracing::{debug, error, info, warn};

use crate::config::{
    EngineConfig, ExchangeConfig, SymbolRegistry, ThreadTopology,
    build_symbol_registry, build_flat_book_configs,
};
use crate::execution_gateway::ExecutionGateway;
use crate::fixed_point::FixedPrice;

// TASK 2: Funding Arb Position tracking struct
/// Tracks an active cross-exchange funding rate arbitrage position.
#[derive(Debug, Clone)]
struct FundingArbPosition {
    symbol: String,
    short_exchange: multi_exchange::ExchangeId,
    long_exchange: multi_exchange::ExchangeId,
    short_entry_price: f64,
    long_entry_price: f64,
    size: i64,
    entry_timestamp_ns: u64,
    entry_net_rate: f64,
}
use crate::flat_book::FlatOrderBook;
use crate::gateio_gateway::GateIoGateway;
use crate::order_lifecycle::OrderLifecycleTracker;
use crate::latency_tracker::PipelineLatencyTracker;
use crate::market_impact::{MarketImpactModel, ImpactParams};
use crate::circuit_breaker::CircuitBreaker;
use crate::spsc::{SpscRingBuffer, SpscOverflowMonitor, RawBookUpdate, BookSnapshot, OrderCommand};
use crate::strategy_engine::StrategyEngine;
use crate::telemetry::TelemetryPublisher;
use crate::pre_trade_risk::PreTradeRiskEngine;
use crate::position_slot_manager::PositionSlotManager;
use crate::position_lifecycle::PositionLifecycleManager;
use crate::position_sizer::PositionSizer;
use crate::rate_limiter::RateLimiterPool;
use crate::smart_entry::{SmartEntryRouter as SmartEntryRouterV2, VolatilityTrailingStop, AdverseSelectionGuard};
use crate::dashboard_server::DashboardState;
use crate::correlation_limiter::CorrelationLimiter;
use crate::cumulative_delta::CumulativeDeltaTracker;
use crate::volume_profile::VolumeProfile;
use crate::liquidation_detector::{LiquidationCascadeDetector, LiquidationCascadeState};
use crate::trend_strength::TrendStrengthIndex;
use crate::execution_analytics::ExecutionAnalytics;
use crate::realized_vol::RealizedVolatilityCalculator; // Task 16
use crate::candle_aggregator::Timeframe;
// use crate::event_sequencer::{EventSequencer, SequencedEventKind};

// ---------------------------------------------------------------------------
// SPSC Ring Buffer sizes (power of 2)
// ---------------------------------------------------------------------------

/// Ring buffer capacity for WS → Book Builder channel.
const WS_TO_BOOK_CAPACITY: usize = 65536;
/// Ring buffer capacity for Book Builder → Strategy channel.
const BOOK_TO_STRATEGY_CAPACITY: usize = 4096;
/// Ring buffer capacity for Strategy → Execution channel.
const STRATEGY_TO_EXEC_CAPACITY: usize = 1024;
/// Ring buffer capacity for WS → Strategy trades channel.
const WS_TO_STRATEGY_TRADES_CAPACITY: usize = 16384;

// ---------------------------------------------------------------------------
// Shared Memory Regime Reader (Issue 2 — seqlock-based)
// ---------------------------------------------------------------------------
// The SharedMemRegimeReader is now in regime_shm.rs.
// It uses a seqlock pattern to read from /dev/shm/regime_weights
// instead of JSON file parsing. Zero-copy, lock-free.

// ---------------------------------------------------------------------------
// MANDATE 2: .env Credential Loading
// ---------------------------------------------------------------------------

/// Explicitly locate and load the .env file so that Gate.io and
/// Forex API credentials are injected into `std::env` before `load_config()`.
///
/// **FIX**: Added Docker-specific paths (/app/.env, /app/crypto_trading_bot/.env)
/// and support for DOTENV_PATH environment variable override.
///
/// Search order:
///   1. `$DOTENV_PATH`                       (explicit override via env var)
///   2. `/app/crypto_trading_bot/.env`        (Docker container standard path)
///   3. `/app/.env`                           (Docker container root)
///   4. `../crypto_trading_bot/.env`          (running from rust_engine/ dir)
///   5. `./crypto_trading_bot/.env`           (running from repo root)
///   6. `./.env`                              (fallback: current directory)
///   7. Adjacent to the executable binary
///
/// Uses `dotenvy::from_path()` so each candidate is tried explicitly.
/// Non-fatal: if no .env exists the engine still works with Docker-injected vars.
fn load_dotenv() {
    // Check for explicit override first
    if let Ok(explicit_path) = std::env::var("DOTENV_PATH") {
        let path = std::path::PathBuf::from(&explicit_path);
        if path.exists() {
            match dotenvy::from_path(&path) {
                Ok(()) => {
                    eprintln!("[dotenv] Loaded credentials from DOTENV_PATH={}", path.display());
                    return;
                }
                Err(e) => {
                    eprintln!("[dotenv] Warning: DOTENV_PATH={} failed to parse: {}", path.display(), e);
                }
            }
        } else {
            eprintln!("[dotenv] Warning: DOTENV_PATH={} does not exist", explicit_path);
        }
    }

    // Build candidate list including Docker-standard paths
    let mut candidates = vec![
        std::path::PathBuf::from("/app/crypto_trading_bot/.env"),
        std::path::PathBuf::from("/app/.env"),
        std::path::PathBuf::from("../crypto_trading_bot/.env"),
        std::path::PathBuf::from("./crypto_trading_bot/.env"),
        std::path::PathBuf::from("./.env"),
    ];

    // Also check adjacent to the binary itself (handles any Docker WORKDIR)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(".env"));
            candidates.push(dir.join("crypto_trading_bot/.env"));
            // Also check parent of binary dir (common in target/release/)
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join(".env"));
                candidates.push(parent.join("crypto_trading_bot/.env"));
            }
        }
    }

    for path in &candidates {
        if path.exists() {
            match dotenvy::from_path(path) {
                Ok(()) => {
                    eprintln!("[dotenv] Loaded credentials from {}", path.display());
                    return;
                }
                Err(e) => {
                    eprintln!("[dotenv] Warning: found {} but failed to parse: {}", path.display(), e);
                }
            }
        }
    }

    // Last resort: let dotenvy search upward from CWD
    match dotenvy::dotenv() {
        Ok(p) => eprintln!("[dotenv] Loaded credentials from {}", p.display()),
        Err(_) => eprintln!("[dotenv] No .env found — using environment variables directly"),
    }

    // Final diagnostic: check if API keys ended up populated
    let has_live_key = std::env::var("GATEIO_API_KEY").is_ok();
    let has_test_key = std::env::var("GATEIO_TESTNET_API_KEY").is_ok();
    if !has_live_key && !has_test_key {
        eprintln!("[dotenv] WARNING: Neither GATEIO_API_KEY nor GATEIO_TESTNET_API_KEY found in environment!");
        eprintln!("[dotenv] Ensure .env is placed at one of: {:?}", candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>());
    } else {
        eprintln!("[dotenv] API keys detected: live={}, testnet={}", has_live_key, has_test_key);
    }
}

// ---------------------------------------------------------------------------
// CLI & Config Loading
// ---------------------------------------------------------------------------

/// Minimal CLI argument parser. Returns the value of `--config <path>` if provided.
fn parse_config_path() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == "--config" {
            return iter.next().cloned();
        }
        if let Some(val) = arg.strip_prefix("--config=") {
            return Some(val.to_string());
        }
    }
    None
}

/// Expand `${VAR_NAME}` placeholders using environment variables.
fn expand_env_vars(s: &str) -> String {
    let mut start = 0;
    let mut output = String::with_capacity(s.len());
    while start < s.len() {
        if let Some(dollar_pos) = s[start..].find("${") {
            let abs_dollar = start + dollar_pos;
            output.push_str(&s[start..abs_dollar]);
            if let Some(close_pos) = s[abs_dollar..].find('}') {
                let var_name = &s[abs_dollar + 2..abs_dollar + close_pos];
                if let Ok(val) = std::env::var(var_name) {
                    output.push_str(&val);
                } else {
                    output.push_str(&s[abs_dollar..abs_dollar + close_pos + 1]);
                }
                start = abs_dollar + close_pos + 1;
            } else {
                output.push_str(&s[abs_dollar..]);
                return output;
            }
        } else {
            output.push_str(&s[start..]);
            break;
        }
    }
    output
}

/// Apply environment variable overrides to a loaded config.
///
/// This function MUST run on every config path (file-loaded or default) so that
/// environment variables like `TRADING_MODE`, `GATEIO_TESTNET_API_KEY`, etc.
/// always take precedence over values baked into config files.
fn apply_env_overrides(cfg: &mut EngineConfig) {
    // ── Detect trading mode from env (critical for API key selection) ──
    let trading_mode = std::env::var("TRADING_MODE")
        .unwrap_or_else(|_| "paper".to_string())
        .to_lowercase();
    let is_testnet = trading_mode == "testnet";
    let is_forex_demo = trading_mode == "forex_demo";

    // ── Gate.io Crypto credentials (mode-aware) ──
    // testnet: GATEIO_TESTNET_API_KEY / GATEIO_TESTNET_SECRET_KEY
    // live/paper: GATEIO_API_KEY / GATEIO_SECRET_KEY
    //
    // CRITICAL: .trim() all keys to strip \n, \r, whitespace that .env parsers
    // may leave behind.  Untrimmed keys cause HMAC signature mismatches which
    // Gate.io reports as INVALID_KEY (the signature doesn't match, so the key
    // appears invalid even though it exists on the server).
    if is_testnet {
        cfg.gateio_api_key = std::env::var("GATEIO_TESTNET_API_KEY")
            .or_else(|_| std::env::var("EXCHANGE_TESTNET_API_KEY"))
            .ok()
            .map(|s| s.trim().to_string());
        cfg.gateio_api_secret = std::env::var("GATEIO_TESTNET_SECRET_KEY")
            .or_else(|_| std::env::var("GATEIO_TESTNET_API_SECRET"))
            .or_else(|_| std::env::var("EXCHANGE_TESTNET_API_SECRET"))
            .ok()
            .map(|s| s.trim().to_string());
        for ex in cfg.exchanges.iter_mut() {
            if ex.name == "gateio" {
                ex.testnet = true;
                // Gate.io migrated USDT futures testnet WS to ws-testnet.gate.com.
                // The old wss://fx-ws-testnet.gateio.ws/v4/ws/usdt defaults to BTC
                // contracts per the official docs warning.
                ex.ws_url = "wss://ws-testnet.gate.com/v4/ws/futures/usdt".to_string();
                // CRITICAL: The correct testnet REST URL is api-testnet.gateapi.io
                // (same as CCXT's set_sandbox_mode). The old fx-api-testnet.gateio.ws
                // is a DIFFERENT server with a DIFFERENT API key pool — keys created
                // on the Gate.io testnet page only work with gateapi.io domain.
                ex.rest_url = Some("https://api-testnet.gateapi.io/api/v4".to_string());
            }
        }
        eprintln!("[config] TRADING_MODE=testnet -> GATEIO_TESTNET_API_KEY / GATEIO_TESTNET_SECRET_KEY");
    } else {
        cfg.gateio_api_key = std::env::var("GATEIO_API_KEY")
            .ok()
            .map(|s| s.trim().to_string());
        cfg.gateio_api_secret = std::env::var("GATEIO_SECRET_KEY")
            .or_else(|_| std::env::var("GATEIO_API_SECRET"))
            .ok()
            .map(|s| s.trim().to_string());
        eprintln!("[config] TRADING_MODE={} -> GATEIO_API_KEY / GATEIO_SECRET_KEY", trading_mode);
    }

    // Propagate API keys into the exchange config
    for ex in cfg.exchanges.iter_mut() {
        if ex.name == "gateio" {
            ex.api_key = cfg.gateio_api_key.clone();
            ex.secret_key = cfg.gateio_api_secret.clone();
            // FIX 3: Log key validation status to help debug INVALID_KEY errors
            let ak = ex.api_key.as_deref().unwrap_or("");
            let sk = ex.secret_key.as_deref().unwrap_or("");
            if ak.is_empty() || sk.is_empty() {
                eprintln!("[config] WARNING: Gate.io API key or secret is empty — signal-only mode");
            } else {
                eprintln!("[config] Gate.io credentials loaded: key={}...{} ({}chars), secret=***{}chars",
                    &ak[..ak.len().min(4)], &ak[ak.len().saturating_sub(4)..], ak.len(), sk.len());
            }
        }
    }

    // ── Forex credentials ──
    if is_forex_demo {
        cfg.forex_login = std::env::var("GATEIO_TRADFI_DEMO_LOGIN")
            .or_else(|_| std::env::var("GATEIO_TRADFI_LOGIN"))
            .ok();
        cfg.forex_password = std::env::var("GATEIO_TRADFI_DEMO_PASSWORD")
            .or_else(|_| std::env::var("GATEIO_TRADFI_PASSWORD"))
            .ok();
        cfg.forex_server = std::env::var("GATEIO_TRADFI_DEMO_SERVER")
            .or_else(|_| std::env::var("GATEIO_TRADFI_SERVER"))
            .ok();
    } else {
        cfg.forex_login = std::env::var("GATEIO_TRADFI_LOGIN").ok();
        cfg.forex_password = std::env::var("GATEIO_TRADFI_PASSWORD").ok();
        cfg.forex_server = std::env::var("GATEIO_TRADFI_SERVER").ok();
    }

    // ── Strategy configuration ──
    if let Ok(enabled_str) = std::env::var("STRATEGY_ENABLED") {
        cfg.strategy.enabled = enabled_str.eq_ignore_ascii_case("true") || enabled_str == "1";
    }

    // ── FIX 10: Trading pairs override (TRADING_PAIRS env var) ──
    if let Ok(pairs_str) = std::env::var("TRADING_PAIRS") {
        let pairs: Vec<String> = pairs_str.split(',')
            .map(|s| {
                // Convert slashes to underscores: BTC/USDT -> BTC_USDT
                s.trim().replace('/', "_").to_uppercase()
            })
            .filter(|s| !s.is_empty())
            .collect();
        if !pairs.is_empty() {
            eprintln!("[config] TRADING_PAIRS override: {:?}", pairs);
            cfg.symbols = pairs.clone();
            // Also update exchange symbols
            for ex in cfg.exchanges.iter_mut() {
                if ex.name == "gateio" {
                    ex.symbols = pairs.clone();
                }
            }
        }
    }

    // ── FIX 11: Leverage override (DEFAULT_LEVERAGE env var) ──
    if let Ok(lev_str) = std::env::var("DEFAULT_LEVERAGE") {
        if let Ok(lev) = lev_str.parse::<i32>() {
            let clamped = lev.clamp(1, 125);
            cfg.strategy.leverage = Some(clamped);
            eprintln!("[config] DEFAULT_LEVERAGE override: {}x", clamped);
        }
    }

    // ── FIX 12: Max open positions override (MAX_OPEN_POSITIONS env var) ──
    if let Ok(max_str) = std::env::var("MAX_OPEN_POSITIONS") {
        if let Ok(max_pos) = max_str.parse::<usize>() {
            let clamped = max_pos.max(1);
            cfg.risk.max_open_positions = clamped;
            eprintln!("[config] MAX_OPEN_POSITIONS override: {}", clamped);
        }
    }

    // ── Infrastructure ──
    cfg.zmq_telemetry_bind = std::env::var("ZMQ_TELEMETRY_BIND")
        .or_else(|_| std::env::var("TELEMETRY_ADDR"))
        .unwrap_or_else(|_| cfg.zmq_telemetry_bind.clone());
    cfg.zmq_config_bind = std::env::var("ZMQ_CONFIG_BIND")
        .or_else(|_| std::env::var("CONFIG_PULL_ADDR"))
        .unwrap_or_else(|_| cfg.zmq_config_bind.clone());
    if let Ok(path) = std::env::var("REGIME_FILE_PATH") {
        cfg.regime_file_path = path;
    }

    // ── Multi-Exchange Toggle ──────────────────────────────────────────────────
    let use_multi = std::env::var("USE_MULTI_EXCHANGE")
        .map(|v| matches!(v.to_lowercase().as_str(), "on" | "true" | "1" | "yes"))
        .unwrap_or(false);
    cfg.multi_exchange_enabled = use_multi;
    cfg.multi_exchange.enabled = use_multi;

    if use_multi {
        // Binance credentials
        cfg.multi_exchange.binance_api_key = std::env::var("BINANCE_API_KEY")
            .ok().map(|s| s.trim().to_string());
        cfg.multi_exchange.binance_secret_key = std::env::var("BINANCE_SECRET_KEY")
            .ok().map(|s| s.trim().to_string());
        cfg.multi_exchange.binance_testnet = std::env::var("BINANCE_TESTNET")
            .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "on"))
            .unwrap_or(false);

        // Bybit credentials
        cfg.multi_exchange.bybit_api_key = std::env::var("BYBIT_API_KEY")
            .ok().map(|s| s.trim().to_string());
        cfg.multi_exchange.bybit_secret_key = std::env::var("BYBIT_SECRET_KEY")
            .ok().map(|s| s.trim().to_string());
        cfg.multi_exchange.bybit_testnet = std::env::var("BYBIT_TESTNET")
            .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "on"))
            .unwrap_or(false);

        // Override max open positions to 5 when multi-exchange is on
        cfg.risk.max_open_positions = cfg.multi_exchange.max_open_positions as usize;
        
        // SOR configuration
        if let Ok(min_split) = std::env::var("SOR_MIN_SPLIT_SIZE_USDT") {
            if let Ok(val) = min_split.parse::<f64>() {
                cfg.multi_exchange.sor.min_split_size_usdt = val;
            }
        }
        if let Ok(max_venues) = std::env::var("SOR_MAX_VENUES") {
            if let Ok(val) = max_venues.parse::<usize>() {
                cfg.multi_exchange.sor.max_venues = val.clamp(1, 3);
            }
        }
        if let Ok(max_slip) = std::env::var("SOR_MAX_SLIPPAGE_BPS") {
            if let Ok(val) = max_slip.parse::<f64>() {
                cfg.multi_exchange.sor.max_slippage_bps = val;
            }
        }

        // Funding arb configuration
        if let Ok(min_rate) = std::env::var("FUNDING_ARB_MIN_NET_RATE") {
            if let Ok(val) = min_rate.parse::<f64>() {
                cfg.multi_exchange.funding_arb.min_net_rate = val;
            }
        }
        if let Ok(min_apr) = std::env::var("FUNDING_ARB_MIN_APR") {
            if let Ok(val) = min_apr.parse::<f64>() {
                cfg.multi_exchange.funding_arb.min_annualized_apr = val / 100.0; // Convert percentage
            }
        }

        // Margin monitor configuration
        if let Ok(min_ratio) = std::env::var("MARGIN_MONITOR_MIN_RATIO") {
            if let Ok(val) = min_ratio.parse::<f64>() {
                cfg.multi_exchange.margin_monitor.min_margin_ratio = val;
            }
        }
        if let Ok(crit_ratio) = std::env::var("MARGIN_MONITOR_CRITICAL_RATIO") {
            if let Ok(val) = crit_ratio.parse::<f64>() {
                cfg.multi_exchange.margin_monitor.critical_margin_ratio = val;
            }
        }

        eprintln!("[config] USE_MULTI_EXCHANGE=on -> max_open_positions={}", cfg.risk.max_open_positions);
        eprintln!("[config] Binance: testnet={}, endpoint={}, has_key={}",
            cfg.multi_exchange.binance_testnet,
            if cfg.multi_exchange.binance_testnet { "https://testnet.binancefuture.com" } else { "https://fapi.binance.com" },
            cfg.multi_exchange.binance_api_key.is_some());
        eprintln!("[config] Bybit: testnet={}, endpoint={}, has_key={}",
            cfg.multi_exchange.bybit_testnet,
            if cfg.multi_exchange.bybit_testnet { "https://api-demo.bybit.com" } else { "https://api.bybit.com" },
            cfg.multi_exchange.bybit_api_key.is_some());
    } else {
        eprintln!("[config] USE_MULTI_EXCHANGE=off -> single-exchange mode (Gate.io only)");
    }
}

fn load_config() -> EngineConfig {
    if let Some(config_path) = parse_config_path() {
        match std::fs::read_to_string(&config_path) {
            Ok(raw_content) => {
                let content = expand_env_vars(&raw_content);
                let is_toml = config_path.ends_with(".toml");
                let result = if is_toml {
                    toml::from_str::<EngineConfig>(&content)
                        .map_err(|e| format!("TOML parse error: {}", e))
                } else {
                    serde_json::from_str::<EngineConfig>(&content)
                        .map_err(|e| format!("JSON parse error: {}", e))
                };
                match result {
                    Ok(mut cfg) => {
                        info!("Loaded config from {}", config_path);
                        apply_env_overrides(&mut cfg);
                        return cfg;
                    }
                    Err(e) => warn!("Failed to parse config file {}: {}", config_path, e),
                }
            }
            Err(e) => warn!("Failed to read config file {}: {}", config_path, e),
        }
    }

    if let Ok(config_path) = std::env::var("TRADING_CONFIG") {
        match std::fs::read_to_string(&config_path) {
            Ok(raw_content) => {
                let content = expand_env_vars(&raw_content);
                let is_toml = config_path.ends_with(".toml");
                let result = if is_toml {
                    toml::from_str::<EngineConfig>(&content)
                        .map_err(|e| format!("TOML parse error: {}", e))
                } else {
                    serde_json::from_str::<EngineConfig>(&content)
                        .map_err(|e| format!("JSON parse error: {}", e))
                };
                match result {
                    Ok(mut cfg) => {
                        info!("Loaded config from TRADING_CONFIG={}", config_path);
                        apply_env_overrides(&mut cfg);
                        return cfg;
                    }
                    Err(e) => warn!("Failed to parse config file {}: {}", config_path, e),
                }
            }
            Err(e) => warn!("Failed to read config file {}: {}", config_path, e),
        }
    }

    let mut cfg = EngineConfig::default();
    apply_env_overrides(&mut cfg);
    cfg
}

/// Return `true` only when `k` is a non-empty, resolved API key.
/// Rejects empty strings, unresolved env var placeholders, and placeholder sentinel values.
fn is_valid_key(k: &str) -> bool {
    let k = k.trim();
    !k.is_empty()
        && !k.starts_with("${")
        && !k.starts_with("your_")
        && k != "PLACEHOLDER"
        && k != "placeholder"
        && k != "xxx"
        && k.len() >= 8  // Gate.io API keys are at least 8 characters
}

/// Build a live execution gateway for an exchange.
fn build_gateway(cfg: &ExchangeConfig) -> Option<Arc<dyn ExecutionGateway + Send + Sync>> {
    let api_key = cfg.api_key.as_deref().unwrap_or("");
    let secret_key = cfg.secret_key.as_deref().unwrap_or("");
    if !is_valid_key(api_key) || !is_valid_key(secret_key) {
        info!("{}: no valid API credentials — signal-only mode", cfg.name);
        return None;
    }
    match cfg.name.as_str() {
        "gateio" => {
            info!("{}: live execution gateway initialised", cfg.name);
            Some(Arc::new(GateIoGateway::new(
                api_key.to_string(),
                secret_key.to_string(),
                cfg.testnet,
            )))
        }
        other => {
            warn!("{}: no execution gateway implementation — signal-only mode", other);
            None
        }
    }
}

// Issue 3 modules
mod circuit_breaker;
mod execution_prep;
mod ws_parser;

// Institutional Upgrade modules — pre-trade risk, slot limiting, native exits,
// deterministic event sourcing, and dust-tracking for fractional contracts.
mod pre_trade_risk;
mod position_slot_manager;
mod exit_evaluator;
mod event_sequencer;
mod dust_tracker;
mod trade_flow_analyzer;
mod correlation_limiter;
mod telegram_alert;
mod cumulative_delta;
mod volume_profile;
mod liquidation_detector;
mod trend_strength;
mod execution_analytics;
mod realized_vol; // Task 16: Phase 2 Feature 9
mod wyckoff_detector; // Phase 3 Feature 11
mod fibonacci_detector; // Phase 3 Feature 12
mod ichimoku_cloud; // Phase 3 Feature 13
mod market_maker_inventory; // Phase 3 Feature 14
mod cross_asset_correlation; // Phase 3 Feature 15

// Directive 1: Rust-native position lifecycle (replaces Python TradeTracker)
mod position_lifecycle;
// Directive 2: Exchange-aware position sizing with quanto_multiplier
mod position_sizer;
// Directive 4: Smart entry, volatility trailing, adverse selection guard
mod smart_entry;
// Directive 5: Dual token-bucket rate limiter
mod rate_limiter;
// Dashboard HTTP server for real-time data
mod dashboard_server;
// Phase 2: Event-sourced order state machine for institutional-grade reconciliation
mod order_state_machine;
// Phase 4: SHM signal queue for Alpha Oracle architecture
mod signal_queue;
mod funding_rate;    // Upgrade 1: Funding Rate Arbitrage
#[allow(dead_code)]  // TWAP engine fully implemented, wiring pending
mod twap_executor;   // Upgrade 3: TWAP/Iceberg Order Execution
mod gamma_shm;       // Phase 2 Feature 4: Options-Derived Gamma Exposure

// ── Institutional Features (from update.txt) ──
mod hw_timestamp;    // Feature 1: Hardware Timestamp Support
mod tick_store;      // Feature 2: Tick Database with Replay
mod var_engine;      // Feature 3: Real-Time VaR Engine
mod options_greeks;  // Feature 4: Options Greeks (Black-Scholes)
mod arbitrage_engine; // Feature 5: Multi-Exchange Arbitrage
mod fee_optimizer;   // Feature 6: Maker Rebate Optimization
mod alert_manager;   // Feature 8: Enhanced Monitoring & Alerting

// ---------------------------------------------------------------------------
// Thread Loops — Real Implementations
// ---------------------------------------------------------------------------

/// Current time in nanoseconds (for SPSC messages).
#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Current time in seconds (for WS subscription messages).
#[inline]
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parse a numeric value from a JSON field (handles both Number and String).
#[inline]
fn json_to_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

/// WS ingestion loop for Gate.io. Runs on a dedicated core.
///
/// Connects to the Gate.io futures WebSocket, subscribes to orderbook and trade
/// channels, parses incoming messages, and pushes `RawBookUpdate` structs into
/// the SPSC ring buffer for consumption by the orderbook builder (Core 4).
///
/// Implements automatic reconnection with exponential backoff + jitter.
fn ws_ingestion_loop_gateio(
    ring: &'static SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>,
    trades_ring: &'static SpscRingBuffer<spsc::TradeEvent, WS_TO_STRATEGY_TRADES_CAPACITY>,
    config: ExchangeConfig,
    registry: Arc<SymbolRegistry>,
) {
    info!("[ws-gateio] Starting WS ingestion on dedicated core");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime for ws-gateio");

    rt.block_on(async {
        let mut backoff_secs = 1u64;
        let mut drop_count: u64 = 0;
        let mut msg_count: u64 = 0;

        loop {
            info!("[ws-gateio] Connecting to {}", config.ws_url);

            match ws_connect_and_ingest_gateio(ring, trades_ring, &config, &registry, &mut drop_count, &mut msg_count).await {
                Ok(()) => {
                    info!("[ws-gateio] Connection closed normally, reconnecting in 1s");
                    backoff_secs = 1;
                }
                Err(e) => {
                    let jitter = (backoff_secs as f64 * 0.2) as u64;
                    let sleep_secs = backoff_secs.saturating_add(jitter);
                    warn!("[ws-gateio] Error: {}. Reconnecting in {}s (drops={}, msgs={})",
                        e, sleep_secs, drop_count, msg_count);
                    backoff_secs = (backoff_secs * 2).min(60);
                }
            }

            tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
        }
    });
}

/// Connect to Gate.io WS and process messages into the SPSC ring.
///
/// BUG 3 FIX: Uses `tokio::select!` with a 15-second ping interval to keep the
/// connection alive. Gate.io closes idle connections after ~30 seconds.
/// BUG 13 FIX: `write` is wrapped in `Arc<tokio::sync::Mutex<_>>` so both the
/// read loop and ping timer can access it concurrently.
async fn ws_connect_and_ingest_gateio(
    ring: &'static SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>,
    trades_ring: &'static SpscRingBuffer<spsc::TradeEvent, WS_TO_STRATEGY_TRADES_CAPACITY>,
    config: &ExchangeConfig,
    registry: &SymbolRegistry,
    drop_count: &mut u64,
    msg_count: &mut u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    let (ws_stream, _) = connect_async(config.ws_url.as_str()).await?;
    let (write, mut read) = ws_stream.split();

    // Wrap write in Arc<Mutex> for shared access between read loop and ping timer
    let write = Arc::new(tokio::sync::Mutex::new(write));

    info!("[ws-gateio] Connected, subscribing to {} symbols", config.symbols.len());

    // Subscribe to orderbook and trade channels for each symbol
    {
        let mut w = write.lock().await;
        for symbol in &config.symbols {
            let sub_book = serde_json::json!({
                "time": now_secs(),
                "channel": "futures.order_book",
                "event": "subscribe",
                "payload": [symbol, "20", "100ms"]
            });
            w.send(Message::Text(sub_book.to_string())).await?;

            let sub_trades = serde_json::json!({
                "time": now_secs(),
                "channel": "futures.trades",
                "event": "subscribe",
                "payload": [symbol]
            });
            w.send(Message::Text(sub_trades.to_string())).await?;
        }
    }

    // Periodic ping to keep connection alive (Gate.io drops idle connections after ~30s)
    let mut ping_interval = tokio::time::interval(Duration::from_secs(15));
    ping_interval.tick().await; // consume the immediate first tick

    // Process incoming messages with concurrent ping keepalive
    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(msg)) => {
                        match msg {
                            Message::Text(text) => {
                                let recv_ns = now_ns();
                                *msg_count += 1;

                                // Parse JSON — skip subscription confirmations
                                let parsed: serde_json::Value = match serde_json::from_str(&text) {
                                    Ok(v) => v,
                                    Err(_) => continue,
                                };

                                let channel = parsed.get("channel").and_then(|v| v.as_str()).unwrap_or("");
                                let event = parsed.get("event").and_then(|v| v.as_str()).unwrap_or("");

                                if event == "subscribe" || event == "unsubscribe" {
                                    continue;
                                }

                                let result = match parsed.get("result") {
                                    Some(r) => r,
                                    None => continue,
                                };

                                match channel {
                                    "futures.order_book" => {
                                        // Parse orderbook update/snapshot
                                        let contract = result.get("contract")
                                            .or_else(|| result.get("s"))
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let sym_id = registry.get_id(contract);
                                        if sym_id == 0 { continue; }

                                        // Parse bids
                                        if let Some(bids) = result.get("bids").and_then(|v| v.as_array()) {
                                            for level in bids {
                                                if let Some(arr) = level.as_array() {
                                                    if arr.len() >= 2 {
                                                        let price = json_to_f64(&arr[0]).unwrap_or(0.0);
                                                        let qty = json_to_f64(&arr[1]).unwrap_or(0.0);
                                                        let update = RawBookUpdate {
                                                            symbol_id: sym_id,
                                                            side: spsc::side::BID,
                                                            update_type: spsc::update_type::DELTA,
                                                            _pad: [0; 4],
                                                            price: FixedPrice::from_f64(price).raw(),
                                                            qty: fixed_point::FixedQty::from_f64(qty).raw(),
                                                            sequence: *msg_count,
                                                            recv_ns,
                                                            snapshot_count: 0,
                                                            _pad2: [0; 4],
                                                        };
                                                        if !ring.try_push(update) {
                                                            *drop_count += 1;
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Parse asks
                                        if let Some(asks) = result.get("asks").and_then(|v| v.as_array()) {
                                            for level in asks {
                                                if let Some(arr) = level.as_array() {
                                                    if arr.len() >= 2 {
                                                        let price = json_to_f64(&arr[0]).unwrap_or(0.0);
                                                        let qty = json_to_f64(&arr[1]).unwrap_or(0.0);
                                                        let update = RawBookUpdate {
                                                            symbol_id: sym_id,
                                                            side: spsc::side::ASK,
                                                            update_type: spsc::update_type::DELTA,
                                                            _pad: [0; 4],
                                                            price: FixedPrice::from_f64(price).raw(),
                                                            qty: fixed_point::FixedQty::from_f64(qty).raw(),
                                                            sequence: *msg_count,
                                                            recv_ns,
                                                            snapshot_count: 0,
                                                            _pad2: [0; 4],
                                                        };
                                                        if !ring.try_push(update) {
                                                            *drop_count += 1;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    "futures.trades" => {
                                        // Parse trade events (for VPIN / adverse selection)
                                        if let Some(trades) = result.as_array() {
                                            for trade in trades {
                                                let price = trade.get("price")
                                                    .and_then(json_to_f64)
                                                    .unwrap_or(0.0);
                                                let size = trade.get("size")
                                                    .and_then(|v| v.as_i64())
                                                    .unwrap_or(0);
                                                let contract = trade.get("contract")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("");
                                                let sym_id = registry.get_id(contract);
                                                if sym_id == 0 || price == 0.0 { continue; }

                                                // Route trade events to dedicated trades ring for VPIN
                                                let side = if size > 0 { 0u8 } else { 1u8 }; // 0=buy, 1=sell
                                                let trade_event = spsc::TradeEvent {
                                                    symbol_id: sym_id,
                                                    side,
                                                    _pad: [0; 5],
                                                    price: FixedPrice::from_f64(price).raw(),
                                                    qty: fixed_point::FixedQty::from_f64(size.abs() as f64).raw(),
                                                    recv_ns,
                                                    sequence: *msg_count,
                                                };
                                                if !trades_ring.try_push(trade_event) {
                                                    *drop_count += 1;
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Message::Ping(data) => {
                                let mut w = write.lock().await;
                                let _ = w.send(Message::Pong(data)).await;
                            }
                            Message::Close(_) => {
                                info!("[ws-gateio] Received Close frame");
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                    Some(Err(e)) => return Err(Box::new(e)),
                    None => return Ok(()),
                }
            }
            _ = ping_interval.tick() => {
                // Send Gate.io futures ping to keep connection alive
                let ping_msg = serde_json::json!({
                    "time": now_secs(),
                    "channel": "futures.ping"
                });
                let mut w = write.lock().await;
                if let Err(e) = w.send(Message::Text(ping_msg.to_string())).await {
                    warn!("[ws-gateio] Ping send failed: {}", e);
                    return Err(Box::new(e));
                }
            }
        }
    }
}



/// Orderbook builder loop. Reads RawBookUpdate from WS ingestion rings,
/// applies them to FlatOrderBooks, and pushes BookSnapshots to the strategy ring.
///
/// TASK 8: Also updates the GlobalBookRegistry with Gate.io's current BBO and top levels
/// so the global book has a complete view of liquidity across all exchanges.
fn orderbook_builder_loop(
    ws_ring_gateio: &'static SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>,
    strategy_ring: &'static SpscRingBuffer<BookSnapshot, BOOK_TO_STRATEGY_CAPACITY>,
    books: &mut Vec<FlatOrderBook>,
    _registry: Arc<SymbolRegistry>,
    shared_prices: Arc<Vec<AtomicU64>>,
    global_book_registry: Option<Arc<multi_exchange::GlobalBookRegistry>>,
) {
    info!("[book-builder] Starting orderbook builder on dedicated core");
    loop {
        // Round-robin read from both exchange rings
        if let Some(update) = ws_ring_gateio.try_pop() {
            let sym_idx = update.symbol_id as usize;
            if sym_idx > 0 && sym_idx <= books.len() {
                let book = &mut books[sym_idx - 1];
                let price = FixedPrice(update.price);
                let qty = fixed_point::FixedQty(update.qty);
                let is_bid = update.side == spsc::side::BID;
                book.apply_delta_tracked(price, qty, is_bid);
                book.set_timestamp_ns(update.recv_ns);
                
                // Update shared price
                let mid = book.mid_price().to_f64();
                if sym_idx - 1 < shared_prices.len() {
                    shared_prices[sym_idx - 1].store(mid.to_bits(), Ordering::Relaxed);
                }
                
                // Push snapshot to strategy
                if let (Some((bid, bid_qty)), Some((ask, ask_qty))) = (book.best_bid(), book.best_ask()) {
                    let snapshot = BookSnapshot {
                        symbol_id: update.symbol_id,
                        bid_levels: 10,
                        ask_levels: 10,
                        _pad: [0; 4],
                        best_bid: bid.raw(),
                        best_ask: ask.raw(),
                        mid_price: book.mid_price().raw(),
                        spread_bps: book.spread_bps() as i32,
                        imbalance_bps: (book.imbalance(10) * 10000.0) as i32,
                        bid_depth_usdt: (book.bid_depth_usdt(10) * FixedPrice::PRECISION as f64) as i64,
                        ask_depth_usdt: (book.ask_depth_usdt(10) * FixedPrice::PRECISION as f64) as i64,
                        sequence: book.sequence(),
                        timestamp_ns: update.recv_ns,
                    };
                    let _ = strategy_ring.try_push(snapshot);
                    
                    // TASK 8: Update GlobalBookRegistry with Gate.io's book data
                    // This ensures the global book has a complete view of liquidity
                    if let Some(ref gbr) = global_book_registry {
                        // Extract top 20 levels from FlatOrderBook for complete liquidity picture
                        let top_bids = book.get_bids(20);
                        let top_asks = book.get_asks(20);
                        
                        let bid_levels: Vec<(i64, i64)> = top_bids.iter()
                            .map(|(p, q)| (p.raw(), q.raw()))
                            .collect();
                        let ask_levels: Vec<(i64, i64)> = top_asks.iter()
                            .map(|(p, q)| (p.raw(), q.raw()))
                            .collect();
                        
                        let gateio_snapshot = multi_exchange::global_book::ExchangeBookSnapshot {
                            exchange: multi_exchange::ExchangeId::GateIo,
                            symbol_id: update.symbol_id,
                            best_bid_fp: bid.raw(),
                            best_ask_fp: ask.raw(),
                            bid_levels,
                            ask_levels,
                            sequence: book.sequence(),
                            timestamp_ns: update.recv_ns,
                        };
                        
                        let gbook = gbr.get_or_create(update.symbol_id);
                        gbook.write().update_exchange_snapshot(gateio_snapshot);
                    }
                }
            }
        }

        // BUG 8 FIX: Hybrid spin-then-yield pattern to avoid CPU burn
        if ws_ring_gateio.is_empty() {
            for _ in 0..100 {
                std::hint::spin_loop();
            }
            if ws_ring_gateio.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(10));
            }
        }
    }
}

/// Strategy evaluator loop. Reads BookSnapshots and emits OrderCommands.
///
/// Issue 2: Now uses `regime_shm::SharedMemRegimeReader` (seqlock-based)
/// instead of the old JSON file reader.
fn strategy_evaluator_loop(
    book_ring: &'static SpscRingBuffer<BookSnapshot, BOOK_TO_STRATEGY_CAPACITY>,
    exec_ring: &'static SpscRingBuffer<OrderCommand, STRATEGY_TO_EXEC_CAPACITY>,
    trades_ring: &'static SpscRingBuffer<spsc::TradeEvent, WS_TO_STRATEGY_TRADES_CAPACITY>,
    regime_reader: &regime_shm::SharedMemRegimeReader,
    strategy: Arc<StrategyEngine>,
    registry: Arc<SymbolRegistry>,
    latency_tracker: &'static PipelineLatencyTracker,
    impact_model: &'static MarketImpactModel,
    overflow_monitor: &'static SpscOverflowMonitor,
    circuit_breaker: Option<&'static CircuitBreaker>,
    pre_trade_risk: &'static PreTradeRiskEngine,
    position_slots: &'static PositionSlotManager,
    correlation_limiter: &'static CorrelationLimiter,
    position_sizer: &'static PositionSizer,
    funding_rates: Arc<parking_lot::RwLock<HashMap<String, f64>>>,
    exec_analytics: &'static parking_lot::Mutex<ExecutionAnalytics>,
    ml_weights: &'static ml_weight_receiver::MlWeightReader,
    manual_pos_rx: crossbeam_channel::Receiver<dashboard_server::ManualPositionTrack>,
) {
    info!("[strategy] Starting strategy evaluator on dedicated core");
    let mut last_regime_check = std::time::Instant::now();
    let mut current_regime = regime::RegimeState::default();
    // BUG 5 FIX: Move recent_highs/recent_lows outside the loop
    let mut recent_highs: Vec<(u64, f64)> = Vec::new();
    let mut recent_lows: Vec<(u64, f64)> = Vec::new();
    // BUG 14 FIX: No raw pointer cast needed — regime_reader.get_current() now takes &self
    // via UnsafeCell interior mutability in SharedMemRegimeReader.

    // Thread-local institutional modules (owned by strategy thread — no sharing)
    let mut dust_tracker = dust_tracker::DustTracker::with_defaults();
    let mut exit_evaluator = exit_evaluator::ExitEvaluator::new();
    // Directive 1: Position lifecycle manager replaces Python's TradeTracker
    let mut lifecycle_mgr = PositionLifecycleManager::with_defaults();
    // Directive 4: Smart entry, volatility trailing, adverse selection guard
    let smart_entry = SmartEntryRouterV2::default();
    let mut vol_trailing = VolatilityTrailingStop::default();
    let mut adverse_guard = AdverseSelectionGuard::default();
    // FIX 9: Initialize VPIN calculator to replace the always-zero stub
    let mut vpin_calculator = microstructure::EnhancedVpin::new(100_000.0, 50); // 100k USDT bucket size, 50 buckets
    // Phase 2 Feature 1: CVD tracker initialization
    let mut cvd_tracker = CumulativeDeltaTracker::new();
    info!("[strategy] 📊 CVD tracker initialized (5m/15m/1h windows)");
    // Phase 2 Feature 2: Volume Profile initialization
    let mut volume_profile = VolumeProfile::new(0.1);
    let mut last_profile_reset = std::time::Instant::now();
    info!("[strategy] 📊 Volume Profile initialized (VPOC + Value Area)");
    // Phase 2 Feature 3: Liquidation cascade detector initialization
    let mut liq_detector = LiquidationCascadeDetector::new();
    info!("[strategy] 🌊 Liquidation cascade detector initialized");
    // Phase 2 Feature 4: Gamma exposure reader initialization
    let gamma_reader = gamma_shm::GammaExposureReader::new("/dev/shm/gamma_exposure");
    info!("[strategy] 📊 Gamma exposure reader initialized");
    
    // Phase 2 Feature 7: Execution Analytics initialization
    // let exec_analytics = exec_analytics.lock();
    info!("[strategy] 📊 Execution Analytics initialized (slippage + shortfall + impact tracking)");
    // Phase 2 Feature 5: Multi-Timeframe Trend Strength Index initialization
    let mut tsi_calculator = TrendStrengthIndex::new();
    info!("[strategy] 📈 Trend Strength Index initialized (M1=10%, M5=20%, M15=30%, H1=40%)");
    // Task 17: Phase 2 Feature 9 - Realized Volatility Calculator initialization
    let mut realized_vol_calc = RealizedVolatilityCalculator::new(300); // 5-minute window (300 seconds)
    info!("[strategy] 📊 Realized Volatility Calculator initialized (5-minute Parkinson estimator)");
    // Task 25: Phase 2 Feature 10 - Trade Flow Analyzer initialization
    let mut trade_flow_analyzer = trade_flow_analyzer::TradeFlowAnalyzer::new(100, 10.0);
    info!("[strategy] 📊 Trade Flow Analyzer initialized (toxicity scoring: VPIN + CVD + large trades)");
    
    // Phase 3 Feature 11: Wyckoff Accumulation/Distribution Detector initialization
    let mut wyckoff_detector = wyckoff_detector::WyckoffDetector::new(100);
    info!("[strategy] 📊 Wyckoff detector initialized (100-period window)");
    
    // Phase 3 Feature 12: Fibonacci Retracement Auto-Detection initialization
    let mut fibonacci_detector = fibonacci_detector::FibonacciDetector::new();
    info!("[strategy] 📊 Fibonacci detector initialized (1h/4h swing detection)");
    
    // Phase 3 Feature 13: Ichimoku Cloud initialization
    let mut ichimoku_cloud = ichimoku_cloud::IchimokuCloud::new();
    info!("[strategy] 📊 Ichimoku Cloud initialized (9/26/52 periods)");
    // FIX 2: Track last Ichimoku candle timestamp to avoid duplicate updates
    let mut last_ichimoku_candle_ts: u64 = 0;
    
    // Phase 3 Feature 14: Market Maker Inventory Model initialization
    let mut mm_inventory = market_maker_inventory::MarketMakerInventoryModel::new(1000);
    info!("[strategy] 📊 Market Maker Inventory Model initialized (1000-period window)");
    
    // Phase 3 Feature 15: Cross-Asset Correlation Monitor initialization
    let mut correlation_monitor = cross_asset_correlation::CrossAssetCorrelationMonitor::new(60);
    info!("[strategy] 📊 Cross-Asset Correlation Monitor initialized (60-period window)");
    // Upgrade 4: Per-position trailing stop states
    let mut trailing_stops: HashMap<u16, exit_evaluator::TrailingStopState> = HashMap::new();
    // Upgrade 1: Funding rate check counter (check every 1000 snapshots)
    let mut funding_check_counter: u64 = 0;
    // Upgrade 1: Funding rate monitor for arbitrage opportunities
    let mut funding_monitor = funding_rate::FundingRateMonitor::new(
        std::env::var("GATEIO_API_KEY").unwrap_or_default(),
        std::env::var("GATEIO_API_SECRET").unwrap_or_default(),
        std::env::var("GATEIO_TESTNET").unwrap_or_default() == "true",
    );
    // FIX 3: Track funding arb positions for exit logic (symbol_id -> (open_timestamp_ns, entry_price))
    let mut funding_arb_positions: HashMap<u16, (u64, f64)> = HashMap::new();
    info!("[strategy] 🧮 DustTracker initialized (max_dust=5.0)");
    info!("[strategy] 📊 VPIN calculator initialized (bucket=100k, depth=50)");
    info!("[strategy] 📈 Trailing stop tracking initialized (break-even + partial TP)");
    info!("[strategy] 💰 Funding rate checks every ~1000 book snapshots");
    info!("[strategy] 🚪 ExitEvaluator initialized (ParabolicSAR + ATR + Chandelier + HardSLTP)");
    info!("[strategy] 📊 PositionLifecycleManager initialized (reversal=30%, max_loss=2%)");
    info!("[strategy] 🎯 SmartEntryRouter initialized (maker-rebate optimization)");
    info!("[strategy] 📈 VolatilityTrailingStop initialized (ATR-based)");
    info!("[strategy] ��️ AdverseSelectionGuard initialized (spoofing detection)");

    loop {
        // Check for manual position tracking requests
        while let Ok(track) = manual_pos_rx.try_recv() {
            exit_evaluator.track_position(
                track.symbol_id,
                track.is_long,
                track.entry_price,
                track.stop_loss,
                track.take_profit,
                track.size,
                300_000_000_000u64, // 5 minutes TTL in nanoseconds
            );
            lifecycle_mgr.track_position(
                track.symbol_id,
                track.is_long,
                track.entry_price,
                track.size,
                track.leverage,
            );
            info!("[strategy] 🖐 Manual position tracked: sym_id={} is_long={} entry={:.4} SL={:.4} TP={:.4}",
                track.symbol_id, track.is_long, track.entry_price, track.stop_loss, track.take_profit
            );
        }

        // Early bail-out: if circuit breaker is tripped, skip evaluation.
        if let Some(ref cb) = circuit_breaker {
            if cb.is_trading_halted() {
                while book_ring.try_pop().is_some() {}
                std::hint::spin_loop();
                continue;
            }
        }

        if let Some(snapshot) = book_ring.try_pop() {
            let symbol_name = registry.get_name(snapshot.symbol_id);
            let signal_start = std::time::Instant::now();

            // Periodically refresh regime from shared memory (every 1s)
            if last_regime_check.elapsed() > Duration::from_secs(1) {
                // BUG 14 FIX: get_current() now takes &self — no unsafe cast needed
                let weights = regime_reader.get_current();
                current_regime = regime::RegimeState::from_weights(weights);
                last_regime_check = std::time::Instant::now();
            }

            // Phase 2 Feature 2: Daily volume profile reset (every 24 hours)
            if last_profile_reset.elapsed() > Duration::from_secs(24 * 60 * 60) {
                volume_profile.reset_profile();
                last_profile_reset = std::time::Instant::now();
                info!("[strategy] 🔄 Volume profile reset (daily)");
            }

            // FIX 2: Update Ichimoku cloud with proper 1h candle data (before draining trades)
            {
                let candle_agg = strategy.candle_aggregator.lock();
                if let Some(candle_1h) = candle_agg.get_latest_completed(Timeframe::H1) {
                    // Only update if this is a new candle (timestamp changed)
                    if candle_1h.timestamp_ns != last_ichimoku_candle_ts {
                        ichimoku_cloud.update_candle(candle_1h.high, candle_1h.low, candle_1h.close);
                        last_ichimoku_candle_ts = candle_1h.timestamp_ns;
                        debug!("[strategy] 📊 Ichimoku updated with 1h candle: H={:.2} L={:.2} C={:.2}",
                               candle_1h.high, candle_1h.low, candle_1h.close);
                    }
                }
            }

            // Drain trade ring and update VPIN + candles + CVD + Volume Profile + Liquidation Detector + Trade Flow Analyzer + Phase 3 detectors
            while let Some(trade) = trades_ring.try_pop() {
                let price = FixedPrice(trade.price).to_f64();
                let volume = fixed_point::FixedQty(trade.qty).to_f64();
                let side = if trade.side == 0 { Some("buy") } else { Some("sell") };
                let mid = FixedPrice(snapshot.mid_price).to_f64();
                let is_buy = trade.side == 0;
                let volume_usdt = price * volume;
                vpin_calculator.on_trade(price, volume, side, mid);
                cvd_tracker.on_trade(trade.recv_ns, volume, is_buy);
                volume_profile.update_trade(price, volume, 0.1);
                liq_detector.on_trade(trade.recv_ns, volume_usdt);
                strategy.update_candles(trade.recv_ns, price, volume);
                // Task 26: Feed trade events to TradeFlowAnalyzer
                trade_flow_analyzer.on_trade(price, volume, trade.side, trade.recv_ns);
                
                // Phase 3: Update Wyckoff detector with price and volume
                wyckoff_detector.update(price, volume);
                
                // Phase 3: Update Fibonacci detector with price history
                fibonacci_detector.update(trade.recv_ns, price);
                
                // FIX 2: Update Ichimoku cloud with proper 1h candle data
                // Only update when a new 1h candle completes (not on every trade)
                // This is handled separately below after draining the trade ring
                
                // Phase 3: Update Market Maker Inventory with trade flow
                mm_inventory.on_trade(volume, is_buy);
                
                // Phase 3: Update Cross-Asset Correlation with price data
                correlation_monitor.update_price(symbol_name, trade.recv_ns, price);
            }

            // FIX 12: Check funding rate arbitrage opportunities every 1000 snapshots
            funding_check_counter += 1;
            if funding_check_counter % 1000 == 0 {
                let mid_price = FixedPrice(snapshot.mid_price).to_f64();
                
                // Update funding rate monitor with real funding rate from shared storage
                let rate = funding_rates.read().get(symbol_name).copied().unwrap_or(0.0001);
                funding_monitor.update_funding_rate(symbol_name, rate);
                
                // FIX 3: Check existing funding arb positions for exit conditions
                let mut positions_to_close = Vec::new();
                for (&sym_id, &(open_ts, entry_price)) in funding_arb_positions.iter() {
                    let time_open_secs = (snapshot.timestamp_ns - open_ts) / 1_000_000_000;
                    let unrealized_loss_pct = ((mid_price - entry_price) / entry_price).abs();
                    
                    // Exit if position open > 8 hours (28800 seconds) OR unrealized loss > 1.5%
                    if time_open_secs > 28800 || unrealized_loss_pct > 0.015 {
                        positions_to_close.push(sym_id);
                        let reason = if time_open_secs > 28800 {
                            "8h time limit"
                        } else {
                            "1.5% loss limit"
                        };
                        info!(
                            "[strategy] 💰 Funding arb exit: {} after {:.1}h, loss={:.2}% — {}",
                            registry.get_name(sym_id), time_open_secs as f64 / 3600.0,
                            unrealized_loss_pct * 100.0, reason
                        );
                        
                        // Generate market close order
                        let close_cmd = OrderCommand {
                            symbol_id: sym_id,
                            side: spsc::side::BUY, // Close SHORT = BUY
                            order_type: spsc::order_cmd_type::MARKET,
                            leverage: 3,
                            _pad: [0; 3],
                            price: FixedPrice::from_f64(mid_price).raw(),
                            qty: fixed_point::FixedQty::from_f64(100.0).raw(),
                            order_id: snapshot.sequence.wrapping_add(2000),
                            signal_ns: snapshot.timestamp_ns,
                            max_slippage_bps: 50,
                            ttl_ms: 5000,
                            stop_loss_fp: 0,
                            take_profit_fp: 0,
                            placement_type: 0,
                            post_only: 0,
                            is_close: 1,
                            _pad2: [0; 5],
                        };
                        
                        if exec_ring.try_push(close_cmd) {
                            info!("[strategy] 💰 Funding arb close order submitted: {}", registry.get_name(sym_id));
                        } else {
                            warn!("[strategy] Funding arb close dropped — execution ring full");
                        }
                    }
                }
                
                // Remove closed positions from tracking
                for sym_id in positions_to_close {
                    funding_arb_positions.remove(&sym_id);
                }
                
                // Check for arbitrage opportunities (funding rate > 0.01%)
                {
                    let opportunities = funding_monitor.check_all_opportunities();
                    for opp in opportunities {
                        if opp.rate_pct > 0.01 {
                            info!(
                                "[strategy] 💰 Funding rate arbitrage: {} funding={:.4}% APR={:.2}% — opening SHORT position",
                                opp.symbol, opp.rate_pct, opp.annualized_pct
                            );
                            
                            // Generate SHORT order to capture funding rate
                            let arb_cmd = OrderCommand {
                                symbol_id: snapshot.symbol_id,
                                side: spsc::side::SELL, // SHORT to receive funding
                                order_type: spsc::order_cmd_type::MARKET,
                                leverage: 3, // Conservative leverage for funding arb
                                _pad: [0; 3],
                                price: FixedPrice::from_f64(mid_price).raw(),
                                qty: fixed_point::FixedQty::from_f64(100.0).raw(), // Fixed size for funding arb
                                order_id: snapshot.sequence.wrapping_add(1000),
                                signal_ns: snapshot.timestamp_ns,
                                max_slippage_bps: 20,
                                ttl_ms: 10000,
                                stop_loss_fp: FixedPrice::from_f64(mid_price * 1.02).raw(), // 2% SL
                                take_profit_fp: 0, // No TP — hold for funding collection
                                placement_type: 0,
                                post_only: 0,
                                is_close: 0,
                                _pad2: [0; 5],
                            };
                            
                            // Pre-trade risk check
                            if let Err(rejection) = pre_trade_risk.check(&arb_cmd) {
                                warn!("[strategy] Funding arb rejected by risk: {:?}", rejection);
                                continue;
                            }
                            
                            // Acquire position slot
                            if !position_slots.try_acquire() {
                                warn!("[strategy] Funding arb skipped — position slots full");
                                continue;
                            }
                            
                            // Push to execution ring
                            if !exec_ring.try_push(arb_cmd) {
                                warn!("[strategy] Funding arb dropped — execution ring full");
                                position_slots.release();
                            } else {
                                // FIX 3: Track this funding arb position for exit logic
                                funding_arb_positions.insert(snapshot.symbol_id, (snapshot.timestamp_ns, mid_price));
                                info!("[strategy] 💰 Funding arb order submitted: {} @ {:.4}", symbol_name, mid_price);
                            }
                        }
                    }
                }
            }

            // FIX 9: Feed VPIN calculator with actual trade events from WS.
            // Trade events are encoded in the SPSC ring with update_type=3.
            // The orderbook_builder_loop should forward these to a separate trade ring,
            // but for now we use the book snapshot's mid price as a proxy.
            // TODO: Add dedicated trade event SPSC ring for proper VPIN calculation.
            {
                let mid = FixedPrice(snapshot.mid_price).to_f64();
                let bid_depth = snapshot.bid_depth_usdt as f64 / FixedPrice::PRECISION as f64;
                let ask_depth = snapshot.ask_depth_usdt as f64 / FixedPrice::PRECISION as f64;
                let total_depth = bid_depth + ask_depth;
                if mid > 0.0 && total_depth > 0.0 {
                    // Estimate trade volume as a fraction of visible depth per tick
                    let synthetic_vol = total_depth * 0.01; // ~1% of visible depth
                    let side = if snapshot.imbalance_bps > 0 { Some("buy") } else { Some("sell") };
                    vpin_calculator.on_trade(mid, synthetic_vol, side, mid);
                }
            }

            // Phase 3: Detect Wyckoff phase
            let (wyckoff_phase, _wyckoff_confidence) = wyckoff_detector.detect_phase();
            
            // Phase 3: Get nearest Fibonacci level
            let current_price = FixedPrice(snapshot.mid_price).to_f64();
            let (fib_level_pct, fib_distance_bps, fib_is_approaching) = fibonacci_detector
                .get_nearest_level(current_price)
                .unwrap_or((0.0, 9999.0, false));
            
            // Phase 3: Get Ichimoku cloud position
            let ichimoku_position = ichimoku_cloud.get_cloud_position(current_price);
            
            // Phase 3: Get Market Maker inventory pressure
            let mm_inventory_pressure = mm_inventory.get_inventory_pressure();
            
            // Phase 3: Get BTC-ETH correlation
            let btc_eth_corr = correlation_monitor.get_correlation("BTC_USDT", "ETH_USDT").unwrap_or(0.0);
            
            // Build microstructure metrics from the snapshot
            let current_price = FixedPrice(snapshot.mid_price).to_f64();
            
            // Task 8: Calculate CVD divergence signals
            let bearish_divergence = cvd_tracker.detect_bearish_divergence(&recent_highs);
            let bullish_divergence = cvd_tracker.detect_bullish_divergence(&recent_lows);
            
            let metrics = strategy_engine::MicrostructureMetrics {
                mid_price: current_price,
                spread_bps: snapshot.spread_bps as f64,
                imbalance: snapshot.imbalance_bps as f64 / 10000.0,
                bid_depth_usdt: snapshot.bid_depth_usdt as f64 / FixedPrice::PRECISION as f64,
                ask_depth_usdt: snapshot.ask_depth_usdt as f64 / FixedPrice::PRECISION as f64,
                vpin: vpin_calculator.get_vpin(), // FIX 9: compute live VPIN from trade flow
                last_trade_is_buy: None,
                cvd_5m: cvd_tracker.get_cvd_5m(),
                cvd_15m: cvd_tracker.get_cvd_15m(),
                cvd_1h: cvd_tracker.get_cvd_1h(),
                gamma_flip_btc: gamma_reader.get_gamma_flip_level("BTC"),
                gamma_flip_eth: gamma_reader.get_gamma_flip_level("ETH"),
                wyckoff_phase: format!("{:?}", wyckoff_phase),
                fib_nearest_level: fib_level_pct,
                ichimoku_cloud_position: format!("{:?}", ichimoku_position),
                mm_inventory_pressure,
                btc_eth_correlation: btc_eth_corr,
                // Task 8: Add new FEATURE 1 fields
                cvd_divergence_bearish: bearish_divergence,
                cvd_divergence_bullish: bullish_divergence,
                funding_rate: funding_rates.read().get(symbol_name).copied().unwrap_or(0.0001),
                vpoc_distance_pct: volume_profile.get_vpoc().map(|vpoc| ((current_price - vpoc) / vpoc).abs()).unwrap_or(1.0),
                realized_vol_regime: format!("{:?}", realized_vol_calc.get_regime()),
                cascade_active: liq_detector.get_state() == LiquidationCascadeState::Active || liq_detector.get_state() == LiquidationCascadeState::Extreme,
            };

            // Evaluate strategy
            
            // Task 24: Calculate order flow toxicity score before signal generation
            let cvd_divergence_score = 0.0; // Placeholder - would need actual divergence calculation
            let toxicity = trade_flow_analyzer.calculate_toxicity_score(metrics.vpin, cvd_divergence_score);
            
            // TASK 2d FIX: Halt trading if toxicity > 0.7 with info-level logging
            if toxicity > 0.7 {
                info!("[strategy] Order flow toxicity {:.2} > 0.7 - halting signal generation for this tick", toxicity);
                continue;
            }

            // ── Phase 2 Feature 1: CVD Divergence Detection ──
            // Track recent price highs/lows for divergence detection
            // For simplicity, we use a rolling window approach
            
            // Add current price point (simplified - in production would track actual swing highs/lows)
            let current_price = FixedPrice(snapshot.mid_price).to_f64();
            recent_highs.push((snapshot.timestamp_ns, current_price));
            recent_lows.push((snapshot.timestamp_ns, current_price));
            
            // Implement rolling window with max 100 entries
            if recent_highs.len() > 100 {
                recent_highs.remove(0);
            }
            if recent_lows.len() > 100 {
                recent_lows.remove(0);
            }
            
            // Check for bearish divergence (price makes new high, CVD makes lower high)
            let bearish_divergence = cvd_tracker.detect_bearish_divergence(&recent_highs);
            // Check for bullish divergence (price makes new low, CVD makes higher low)
            let bullish_divergence = cvd_tracker.detect_bullish_divergence(&recent_lows);

            // ── Directive 4: Update adverse selection guard with book depth ──
            adverse_guard.update(
                snapshot.bid_depth_usdt as f64 / FixedPrice::PRECISION as f64,
                snapshot.ask_depth_usdt as f64 / FixedPrice::PRECISION as f64,
            );

            // ── Directive 4: Update volatility trailing with tick data ──
            let tick_high = FixedPrice(snapshot.best_ask).to_f64();
            let tick_low = FixedPrice(snapshot.best_bid).to_f64();
            let tick_close = FixedPrice(snapshot.mid_price).to_f64();
            vol_trailing.update_tick(tick_high, tick_low, tick_close, 0.0);
            
            // Task 18: Update realized volatility calculator with tick data
            realized_vol_calc.on_tick(snapshot.timestamp_ns, tick_high, tick_low, tick_close);

            // ── Directive 1: Evaluate lifecycle exits on EVERY tick ──
            // PositionLifecycleManager tracks PnL tick-by-tick and fires
            // reversal/max-loss/sustained-decline closes BEFORE the ExitEvaluator.
            if lifecycle_mgr.active_count() > 0 {
                let mid = FixedPrice(snapshot.mid_price).to_f64();
                // Clone the keys to avoid borrow conflict
                let tracked_symbols: Vec<u16> = lifecycle_mgr
                    .all_positions()
                    .map(|p| p.symbol_id)
                    .collect();
                for sym_id in tracked_symbols {
                    if let Some(close_action) = lifecycle_mgr.on_tick(sym_id, mid) {
                        // Generate market close order
                        let pos = lifecycle_mgr.get_position(sym_id);
                        let exit_side = match pos {
                            Some(p) if p.is_long => spsc::side::SELL,
                            _ => spsc::side::BUY,
                        };
                        let pos_size = lifecycle_mgr.get_position(sym_id).map(|p| p.size.abs()).unwrap_or(1);
                        let exit_cmd = OrderCommand {
                            symbol_id: sym_id,
                            side: exit_side,
                            order_type: spsc::order_cmd_type::MARKET,
                            leverage: 1,
                            _pad: [0; 3],
                            price: FixedPrice::from_f64(mid).raw(),
                            qty: fixed_point::FixedQty::from_f64(pos_size as f64).raw(),
                            order_id: snapshot.sequence.wrapping_add(100),
                            signal_ns: snapshot.timestamp_ns,
                            max_slippage_bps: 100,
                            ttl_ms: 1000,
                            stop_loss_fp: 0,
                            take_profit_fp: 0,
                            placement_type: 0,
                            post_only: 0,
                            is_close: 1,
                            _pad2: [0; 5],
                        };
                        if exec_ring.try_push(exit_cmd) {
                            lifecycle_mgr.untrack_position(sym_id);
                            position_slots.release();
                            info!(
                                "[lifecycle] EXIT: sym={} reason={} pnl={:.2}% peak={:.2}%",
                                registry.get_name(sym_id),
                                close_action.reason,
                                close_action.trigger_pnl_pct,
                                close_action.peak_pnl_pct,
                            );
                            
                            // Task 1: Record fill in ExecutionAnalytics
                            exec_analytics.lock().record_fill(
                                mid,
                                mid,
                                mid,
                                mid,
                                mid * pos_size as f64,
                            );
                        }
                    }
                }
            }

            // ── Institutional: Evaluate Exits on All Tracked Positions ──
            // Run on EVERY tick before new signal generation. The ExitEvaluator
            // checks ParabolicSAR, ATR Trailing, Chandelier, and hard SL/TP
            // against the live best bid/ask.
            if exit_evaluator.tracked_count() > 0 {
                let exit_high = FixedPrice(snapshot.best_ask).to_f64();
                let exit_low = FixedPrice(snapshot.best_bid).to_f64();
                let exit_close = FixedPrice(snapshot.mid_price).to_f64();
                let exit_mid = exit_close;
                let exits = exit_evaluator.evaluate_all(
                    snapshot.symbol_id, exit_high, exit_low, exit_close, exit_mid,
                );
                for exit_signal in &exits {
                    if exit_signal.urgency >= 2 {
                        // Emergency exit: generate MARKET close order
                        let exit_side = if exit_signal.exit_side == spsc::side::BUY {
                            spsc::side::BUY
                        } else {
                            spsc::side::SELL
                        };
                        let pos_size = exit_evaluator.get_position_size(exit_signal.symbol_id).unwrap_or(1);
                        let exit_cmd = OrderCommand {
                            symbol_id: exit_signal.symbol_id,
                            side: exit_side,
                            order_type: spsc::order_cmd_type::MARKET,
                            leverage: 1, // irrelevant for close
                            _pad: [0; 3],
                            price: FixedPrice::from_f64(exit_mid).raw(),
                            qty: fixed_point::FixedQty::from_f64(pos_size as f64).raw(),
                            order_id: snapshot.sequence.wrapping_add(1),
                            signal_ns: snapshot.timestamp_ns,
                            max_slippage_bps: 100, // wider for urgency
                            ttl_ms: 1000,
                            stop_loss_fp: 0,
                            take_profit_fp: 0,
                            placement_type: 0,
                            post_only: 0,
                            is_close: 1,
                            _pad2: [0; 5],
                        };
                        if exec_ring.try_push(exit_cmd) {
                            exit_evaluator.untrack_position(exit_signal.symbol_id);
                            trailing_stops.remove(&exit_signal.symbol_id); // Upgrade 4: cleanup
                            position_slots.release();
                            info!(
                                "[strategy] 🚨 EXIT fired: sym={} reason={:?} urgency={}",
                                registry.get_name(exit_signal.symbol_id),
                                exit_signal.reason, exit_signal.urgency
                            );
                        } else {
                            warn!("[strategy] Exit ring full — critical exit lost for sym={}",
                                registry.get_name(exit_signal.symbol_id));
                        }
                    }
                }
            }
            // ── Upgrade 4: Update trailing stops for all tracked positions ──
            if !trailing_stops.is_empty() {
                let mid = FixedPrice(snapshot.mid_price).to_f64();
                let atr = exit_evaluator.get_position_atr(snapshot.symbol_id);
                if let Some(ts) = trailing_stops.get_mut(&snapshot.symbol_id) {
                    match ts.update_trailing_stop(mid, atr) {
                        exit_evaluator::TrailingStopUpdate::MoveToBreakEven { new_sl } => {
                            exit_evaluator.update_sl_tp(snapshot.symbol_id, new_sl, 0.0);
                            info!(
                                "[strategy] 🔒 Break-even SL set for {}: SL moved to {:.4}",
                                registry.get_name(snapshot.symbol_id), new_sl
                            );
                            // FIX 4: Update exchange-side conditional order
                            // This is a fire-and-forget async update — if it fails, the
                            // Rust-side exit evaluator still protects the position
                            // TODO: Implement exchange-side SL/TP update via REST API
                        }
                        exit_evaluator::TrailingStopUpdate::TrailStop { new_sl } => {
                            exit_evaluator.update_sl_tp(snapshot.symbol_id, new_sl, 0.0);
                            info!(
                                "[strategy] 📈 Trailing SL updated for {}: new SL={:.4}",
                                registry.get_name(snapshot.symbol_id), new_sl
                            );
                            // FIX 4: Update exchange-side conditional order
                            // Gate.io requires canceling the old SL and submitting a new one
                            // TODO: Implement via gateio_gateway.update_conditional_sl()
                        }
                        exit_evaluator::TrailingStopUpdate::PartialClose { fraction, reason } => {
                            // Generate a partial close order
                            if let Some(pos_size) = exit_evaluator.get_position_size(snapshot.symbol_id) {
                                let close_size = (pos_size as f64 * fraction).ceil() as i64;
                                let is_long = exit_evaluator.is_position_long(snapshot.symbol_id).unwrap_or(true);
                                let exit_side = if is_long { spsc::side::SELL } else { spsc::side::BUY };
                                let partial_cmd = OrderCommand {
                                    symbol_id: snapshot.symbol_id,
                                    side: exit_side,
                                    order_type: spsc::order_cmd_type::MARKET,
                                    leverage: 1,
                                    _pad: [0; 3],
                                    price: FixedPrice::from_f64(mid).raw(),
                                    qty: fixed_point::FixedQty::from_f64(close_size as f64).raw(),
                                    order_id: snapshot.sequence.wrapping_add(2),
                                    signal_ns: snapshot.timestamp_ns,
                                    max_slippage_bps: 50,
                                    ttl_ms: 2000,
                                    stop_loss_fp: 0,
                                    take_profit_fp: 0,
                                    placement_type: 0,
                                    post_only: 0,
                                    is_close: 1,
                                    _pad2: [0; 5],
                                };
                                if exec_ring.try_push(partial_cmd) {
                                    info!(
                                        "[strategy] 💰 Partial TP: {} close {:.0}% ({} contracts) — {}",
                                        registry.get_name(snapshot.symbol_id),
                                        fraction * 100.0, close_size, reason
                                    );
                                }
                            }
                        }
                        exit_evaluator::TrailingStopUpdate::NoAction => {}
                    }
                }
            }

            // Task 19: Periodic logging of Phase 2 Feature 7-10 metrics (every 100 snapshots)
            static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);
            {
                let count = SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed);
                if count % 100 == 0 {
                    let exec_analytics_guard = exec_analytics.lock();
                    let vol_regime = realized_vol_calc.get_regime();
                    let vol_pct = realized_vol_calc.get_volatility();
                    
                    // Phase 3: Log all Phase 3 metrics
                    info!(
                        "[strategy] 📊 Phase 2 Metrics: slippage={:.2}bps impact={:.2}bps vol={:.1}%({}) toxicity={:.2}",
                        exec_analytics_guard.get_avg_slippage_bps(),
                        exec_analytics_guard.get_avg_impact_bps(),
                        vol_pct,
                        vol_regime.to_string(),
                        toxicity
                    );
                    
                    info!(
                        "[strategy] 📊 Phase 3 Metrics: wyckoff={} fib_level={:.1}% fib_dist={:.0}bps ichimoku={} mm_inv={:.2} btc_eth_corr={:.2}",
                        wyckoff_phase.to_string(),
                        fib_level_pct * 100.0,
                        fib_distance_bps,
                        ichimoku_position.to_string(),
                        mm_inventory_pressure,
                        btc_eth_corr
                    );
                }
            }
            
            if let Some(mut intent) = strategy.evaluate(&metrics, &current_regime, symbol_name, ml_weights, snapshot.symbol_id) {
                let entry_price = intent.price.unwrap_or(metrics.mid_price);
                let is_buy = matches!(intent.side, execution_gateway::OrderSide::Buy);

                // ── Phase 2 Feature 1: Apply CVD Divergence Adjustments ──
                if bearish_divergence && is_buy {
                    // Bearish divergence detected on a long signal - reduce confidence by 30%
                    intent.confidence *= 0.7;
                    info!("[strategy] 🔻 CVD bearish divergence detected for {} — reducing long confidence to {:.2}",
                        symbol_name, intent.confidence);
                    // Skip signal if confidence drops too low
                    if intent.confidence < 0.3 {
                        info!("[strategy] 🔻 CVD bearish divergence — skipping long signal for {}", symbol_name);
                        continue;
                    }
                } else if bullish_divergence && !is_buy {
                    // Bullish divergence detected on a short signal - reduce confidence by 30%
                    intent.confidence *= 0.7;
                    info!("[strategy] 🔺 CVD bullish divergence detected for {} — reducing short confidence to {:.2}",
                        symbol_name, intent.confidence);
                    // Skip signal if confidence drops too low
                    if intent.confidence < 0.3 {
                        info!("[strategy] 🔺 CVD bullish divergence — skipping short signal for {}", symbol_name);
                        continue;
                    }
                }
                
                // ── Phase 3 Feature 11: Wyckoff Phase Filtering ──
                // Bias long signals during Accumulation phase, short signals during Distribution phase
                match wyckoff_phase {
                    wyckoff_detector::WyckoffPhase::Accumulation if is_buy => {
                        // Accumulation phase + long signal = increase confidence by 20%
                        intent.confidence = (intent.confidence * 1.2).min(1.0);
                        info!("[strategy] 📈 Wyckoff Accumulation phase — boosting long confidence to {:.2}", intent.confidence);
                    }
                    wyckoff_detector::WyckoffPhase::Distribution if !is_buy => {
                        // Distribution phase + short signal = increase confidence by 20%
                        intent.confidence = (intent.confidence * 1.2).min(1.0);
                        info!("[strategy] 📉 Wyckoff Distribution phase — boosting short confidence to {:.2}", intent.confidence);
                    }
                    wyckoff_detector::WyckoffPhase::Accumulation if !is_buy => {
                        // Accumulation phase + short signal = reduce confidence by 30%
                        intent.confidence *= 0.7;
                        info!("[strategy] 📈 Wyckoff Accumulation phase — reducing short confidence to {:.2}", intent.confidence);
                    }
                    wyckoff_detector::WyckoffPhase::Distribution if is_buy => {
                        // Distribution phase + long signal = reduce confidence by 30%
                        intent.confidence *= 0.7;
                        info!("[strategy] 📉 Wyckoff Distribution phase — reducing long confidence to {:.2}", intent.confidence);
                    }
                    _ => {}
                }

                // ── Phase 2 Feature 2: VPOC-Based Bias ──
                if let Some(vpoc) = volume_profile.get_vpoc() {
                    let price_to_vpoc_pct = ((current_price - vpoc) / vpoc).abs();
                    
                    // If price is within 1% of VPOC, apply bias
                    if price_to_vpoc_pct < 0.01 {
                        if current_price < vpoc && is_buy {
                            // Price approaching VPOC from below - increase long confidence by 15%
                            intent.confidence = (intent.confidence * 1.15).min(1.0);
                            info!("[strategy] 🎯 Price approaching VPOC {:.2} from below — long bias (confidence={:.2})",
                                vpoc, intent.confidence);
                        } else if current_price > vpoc && !is_buy {
                            // Price approaching VPOC from above - increase short confidence by 15%
                            intent.confidence = (intent.confidence * 1.15).min(1.0);
                            info!("[strategy] 🎯 Price approaching VPOC {:.2} from above — short bias (confidence={:.2})",
                                vpoc, intent.confidence);
                        }
                    }
                }
                
                // ── Phase 3 Feature 12: Fibonacci Level Confluence Boost ──
                // Increase confidence by 15% when price is within 1% of a Fibonacci level
                if fib_is_approaching && fib_distance_bps < 100.0 {
                    // Check if signal direction aligns with Fibonacci level
                    // Approaching from below = support for long, from above = resistance for short
                    let fib_levels = fibonacci_detector.calculate_fib_levels();
                    if let Some(levels) = fib_levels {
                        let nearest_fib_price = levels.levels.iter()
                            .min_by(|a, b| {
                                let dist_a = (current_price - *a).abs();
                                let dist_b = (current_price - *b).abs();
                                dist_a.partial_cmp(&dist_b).unwrap()
                            })
                            .copied()
                            .unwrap_or(current_price);
                        
                        let approaching_from_below = current_price < nearest_fib_price;
                        
                        if (approaching_from_below && is_buy) || (!approaching_from_below && !is_buy) {
                            intent.confidence = (intent.confidence * 1.15).min(1.0);
                            info!("[strategy] 📐 Fibonacci {:.1}% level confluence — boosting confidence to {:.2}",
                                fib_level_pct * 100.0, intent.confidence);
                        }
                    }
                }

                // ── Phase 2 Feature 5: TSI Calculation ──
                let tsi_score = tsi_calculator.calculate_tsi(&strategy.candle_aggregator.lock());
                
                // ── Phase 3 Feature 13: Ichimoku Cloud Trend Filter ──
                // Skip long signals when price is below cloud, short signals when above cloud
                match ichimoku_position {
                    ichimoku_cloud::CloudPosition::BelowCloud if is_buy => {
                        info!("[strategy] ☁️ Ichimoku: price below cloud — skipping long signal");
                        continue;
                    }
                    ichimoku_cloud::CloudPosition::AboveCloud if !is_buy => {
                        info!("[strategy] ☁️ Ichimoku: price above cloud — skipping short signal");
                        continue;
                    }
                    _ => {}
                }
                
                // ── Phase 2 Feature 3: Liquidation Cascade Response ──
                let liq_state = liq_detector.get_state();
                let (imb_mult, trail_mult) = liq_detector.get_adjusted_thresholds();
                
                if liq_state == LiquidationCascadeState::Active || liq_state == LiquidationCascadeState::Extreme {
                    warn!("[strategy] 🌊 Liquidation cascade {:?} detected — adjusting thresholds (imb_mult={:.1}x, trail_mult={:.1}x)",
                        liq_state, imb_mult, trail_mult);
                    
                    // Reduce position size during cascades
                    let cascade_size_reduction = match liq_state {
                        LiquidationCascadeState::Active => 0.5,  // 50% reduction
                        LiquidationCascadeState::Extreme => 0.25, // 75% reduction
                        _ => 1.0,
                    };
                    
                    // Apply size reduction (will be applied later in position sizing)
                    intent.confidence *= cascade_size_reduction;
                    
                    if intent.confidence < 0.2 {
                        warn!("[strategy] 🌊 Liquidation cascade — skipping signal due to low confidence");
                        continue;
                    }
                }

                // ── Phase 2 Feature 4: Gamma Flip Level Support/Resistance ──
                // Check if current price is within 1% of gamma flip level
                let current_price = FixedPrice(snapshot.mid_price).to_f64();
                let gamma_flip = if symbol_name.contains("BTC") {
                    metrics.gamma_flip_btc
                } else if symbol_name.contains("ETH") {
                    metrics.gamma_flip_eth
                } else {
                    None
                };

                if let Some(gamma_flip_level) = gamma_flip {
                    let distance_pct = ((current_price - gamma_flip_level) / gamma_flip_level).abs();
                    
                    if distance_pct < 0.01 {
                        // Price is within 1% of gamma flip level
                        if current_price < gamma_flip_level * 0.99 && is_buy {
                            // Approaching from below on a long signal → resistance
                            intent.confidence *= 0.8;
                            info!(
                                "[strategy] 🎯 Gamma flip resistance: price {:.2} approaching {:.2} from below — reducing long confidence to {:.2}",
                                current_price, gamma_flip_level, intent.confidence
                            );
                        } else if current_price > gamma_flip_level * 1.01 && !is_buy {
                            // Approaching from above on a short signal → support
                            intent.confidence *= 0.8;
                            info!(
                                "[strategy] 🎯 Gamma flip support: price {:.2} approaching {:.2} from above — reducing short confidence to {:.2}",
                                current_price, gamma_flip_level, intent.confidence
                            );
                        }
                    }
                }
                
                // ── Phase 3 Feature 14: Market Maker Inventory Contrarian Signal ──
                // Reduce confidence when MM inventory pressure is extreme and signal aligns with MM position
                let (mm_direction, mm_pressure_score) = mm_inventory.get_inventory_signal();
                if mm_pressure_score > 0.8 {
                    // MM is heavily positioned
                    if (mm_direction == -1 && !is_buy) || (mm_direction == 1 && is_buy) {
                        // Signal aligns with MM position (MM short + short signal, or MM long + long signal)
                        // This is contrarian - reduce confidence by 20%
                        intent.confidence *= 0.8;
                        info!(
                            "[strategy] 🏦 MM inventory extreme ({:.2}) — reducing confidence to {:.2}",
                            mm_pressure_score, intent.confidence
                        );
                    }
                }

                // FEATURE 13: Drawdown-based position scaling
                // Track peak equity and reduce position size during drawdowns
                let (drawdown_scalar, should_halt) = if let Some(ref cb) = circuit_breaker {
                    let cb_state = cb.get_state();
                    let current_equity = cb_state.current_equity as f64 / 1e8;
                    let peak_equity = cb_state.peak_equity as f64 / 1e8;
                    
                    if peak_equity > 0.0 {
                        let drawdown_pct = (peak_equity - current_equity) / peak_equity;
                        
                        if drawdown_pct > 0.05 {
                            // Halt new trades when drawdown > 5%
                            warn!(
                                "[strategy] 🛑 Drawdown {:.2}% exceeds 5% threshold — halting new trades",
                                drawdown_pct * 100.0
                            );
                            (0.0, true)
                        } else if drawdown_pct > 0.02 {
                            // Reduce position size by drawdown_pct * 10 when drawdown > 2%
                            let reduction = drawdown_pct * 10.0;
                            let scalar = (1.0 - reduction).max(0.1);
                            info!(
                                "[strategy] 📉 Drawdown {:.2}% — reducing position size by {:.1}%",
                                drawdown_pct * 100.0, reduction * 100.0
                            );
                            (scalar, false)
                        } else {
                            (1.0, false)
                        }
                    } else {
                        (1.0, false)
                    }
                } else {
                    (1.0, false)
                };

                // Skip trade if drawdown halt is active
                if should_halt {
                    continue;
                }

                // ── Directive 4: Smart Entry Decision ──
                // Decide whether to post maker or cross spread based on microstructure.
                // FEATURE 12: Fetch tick_size from position_sizer ContractSpec
                let tick_size = position_sizer.get_spec(symbol_name)
                    .map(|spec| {
                        // Calculate tick size from price precision
                        // e.g., precision=2 → tick_size=0.01
                        10.0_f64.powi(-(spec.order_price_precision as i32))
                    });
                
                // Task 2: Check ExecutionAnalytics for order type override
                let should_use_limit = exec_analytics.lock().should_use_limit_orders();
                
                let (entry_decision, smart_price) = smart_entry.decide(
                    is_buy,
                    FixedPrice(snapshot.best_bid).to_f64(),
                    FixedPrice(snapshot.best_ask).to_f64(),
                    metrics.vpin,
                    metrics.imbalance,
                    adverse_guard.is_long_paused() && is_buy,
                    tick_size,
                );

                // Skip entry if adverse selection guard paused it
                if entry_decision == smart_entry::EntryDecision::PauseEntry {
                    info!("[strategy] ⏸️ Entry paused by adverse selection guard for {}",
                        registry.get_name(snapshot.symbol_id));
                    continue;
                }

                // ── Directive 4: Volatility-adjusted SL/TP ──
                // Use real-time ATR to size stops instead of fixed percentages.
                let mut trail_distance = vol_trailing.calculate_trail_distance(entry_price, 0.0);
                
                // Task 20: Apply realized volatility regime scaling to trailing stop distance
                let vol_regime_scale = realized_vol_calc.get_regime().get_scale_factor();
                trail_distance *= vol_regime_scale;
                
                let sl_pct = (trail_distance / entry_price).max(0.005).min(0.05);
                let tp_pct = (sl_pct * 2.0).min(0.10); // 2:1 reward-risk minimum
                let stop_loss_price = if is_buy {
                    entry_price * (1.0 - sl_pct)
                } else {
                    entry_price * (1.0 + sl_pct)
                };
                let take_profit_price = if is_buy {
                    entry_price * (1.0 + tp_pct)
                } else {
                    entry_price * (1.0 - tp_pct)
                };

                // Directive 4: Set order type based on smart entry decision
                // Task 2: Override to LIMIT if ExecutionAnalytics detects high slippage
                let (cmd_order_type, cmd_post_only, effective_price) = if should_use_limit {
                    info!("[strategy] 📊 Execution analytics: forcing LIMIT order (avg slippage > 5bps)");
                    (spsc::order_cmd_type::LIMIT, 1u8, smart_price)
                } else {
                    match entry_decision {
                        smart_entry::EntryDecision::PostMaker => {
                            (spsc::order_cmd_type::LIMIT, 1u8, smart_price)
                        }
                        smart_entry::EntryDecision::CrossSpread => {
                            (spsc::order_cmd_type::MARKET, 0u8, smart_price)
                        }
                        smart_entry::EntryDecision::ChaseBook => {
                            (spsc::order_cmd_type::LIMIT, 1u8, smart_price)
                        }
                        _ => {
                            (spsc::order_cmd_type::LIMIT, 0u8, entry_price)
                        }
                    }
                };

                // Task 3: Reduce size if market impact is high (ExecutionAnalytics)
                let should_reduce_size = exec_analytics.lock().should_reduce_size();
                let mut impact_scalar = if should_reduce_size {
                    info!("[strategy] 📊 Execution analytics: reducing size by 50% (avg impact > 10bps)");
                    0.5
                } else {
                    1.0
                };
                
                // Task 24: Reduce size by 50% if toxicity > 0.5
                if toxicity > 0.5 {
                    impact_scalar *= 0.5;
                    info!("[strategy] 🚫 Order flow toxicity {:.2} > 0.5 — reducing size by 50%", toxicity);
                }
                
                // ── Phase 3 Feature 15: Cross-Asset Correlation Check ──
                // Reduce position size when BTC-ETH correlation drops below 0.7 for ETH trades
                if symbol_name.contains("ETH") && btc_eth_corr < 0.7 {
                    impact_scalar *= 0.5;
                    info!("[strategy] 🔗 BTC-ETH correlation {:.2} < 0.7 — reducing ETH position size by 50%", btc_eth_corr);
                }
                
                // Apply equity market risk-off rules when BTC-SPX correlation spikes above 0.8
                let btc_spx_corr = correlation_monitor.get_correlation("BTC_USDT", "SPX").unwrap_or(0.0);
                if btc_spx_corr > 0.8 {
                    impact_scalar *= 0.7;
                    info!("[strategy] 📉 BTC-SPX correlation {:.2} > 0.8 — reducing all position sizes by 30%", btc_spx_corr);
                }
                
                // ── FEATURE 3: Kelly Criterion Position Sizing ──
                // Calculate position size using Kelly criterion with ATR-based stop distance
                let base_qty = intent.size as f64;
                let mut kelly_qty = if let Some(ref cb) = circuit_breaker {
                    let cb_state = cb.get_state();
                    let current_equity = cb_state.current_equity as f64 / 1e8;
                    let win_rate = intent.confidence.clamp(0.4, 0.7); // Use signal confidence as win rate proxy
                    let atr_stop_distance = exit_evaluator.get_position_atr(snapshot.symbol_id);
                    
                    if current_equity > 0.0 && atr_stop_distance > 0.0 {
                        let stop_price = if is_buy {
                            effective_price - atr_stop_distance
                        } else {
                            effective_price + atr_stop_distance
                        };
                        
                        let kelly_size_fp = risk_calculator::risk_based_position_size(
                            current_equity,
                            win_rate * 0.02, // 2% base risk scaled by confidence
                            FixedPrice::from_f64(effective_price).raw(),
                            FixedPrice::from_f64(stop_price).raw(),
                        );
                        
                        let kelly_contracts = fixed_point::FixedQty(kelly_size_fp).to_f64();
                        if kelly_contracts > 0.0 {
                            // FEATURE 13: Apply drawdown scalar to Kelly-sized position
                            (kelly_contracts * drawdown_scalar).min(base_qty)
                        } else {
                            base_qty * drawdown_scalar * impact_scalar
                        }
                    } else {
                        base_qty * drawdown_scalar * impact_scalar
                    }
                } else {
                    base_qty * drawdown_scalar * impact_scalar
                };

                // ── Phase 2 Feature 5: TSI-Based Position Sizing Adjustment ──
                let tsi_scale = if tsi_score > 0.7 {
                    1.5 // Strong trend: increase size by 50%
                } else if tsi_score < 0.3 {
                    0.5 // Weak trend: reduce size by 50%
                } else {
                    1.0 // Moderate trend: normal size
                };
                kelly_qty *= tsi_scale;
                
                if tsi_scale != 1.0 {
                    info!(
                        "[strategy] 📈 TSI={:.2} → size scaled by {:.1}x (final_qty={:.1})",
                        tsi_score, tsi_scale, kelly_qty
                    );
                }
                
                // Task 19: Apply realized volatility regime scaling to position size
                let vol_scale = realized_vol_calc.get_position_scale();
                kelly_qty *= vol_scale;
                
                if vol_scale != 1.0 {
                    let vol_regime = realized_vol_calc.get_regime();
                    let vol_pct = realized_vol_calc.get_volatility();
                    info!(
                        "[strategy] 📊 Realized Vol={:.1}% regime={} → size scaled by {:.2}x (final_qty={:.1})",
                        vol_pct, vol_regime.to_string(), vol_scale, kelly_qty
                    );
                }

                // ── FEATURE 8: Adaptive Leverage Calculation ──
                // Calculate leverage based on ATR/price ratio
                let atr = exit_evaluator.get_position_atr(snapshot.symbol_id);
                let adaptive_leverage = if atr > 0.0 {
                    position_sizer.calculate_adaptive_leverage(atr, effective_price)
                } else {
                    strategy.config().leverage.unwrap_or(5).clamp(1, 125)
                };

                let cmd = OrderCommand {
                    symbol_id: snapshot.symbol_id,
                    side: if is_buy { spsc::side::BUY } else { spsc::side::SELL },
                    order_type: cmd_order_type,
                    leverage: adaptive_leverage.clamp(1, 125) as u8,
                    _pad: [0; 3],
                    price: FixedPrice::from_f64(effective_price).raw(),
                    qty: {
                        // Institutional: Use DustTracker for fractional contract handling.
                        // Carries sub-contract remainders across trades to prevent drift.
                        let contracts = dust_tracker.float_to_contracts(
                            snapshot.symbol_id, kelly_qty, is_buy,
                        );
                        fixed_point::FixedQty::from_f64(contracts as f64).raw()
                    },
                    order_id: snapshot.sequence,
                    signal_ns: snapshot.timestamp_ns,
                    max_slippage_bps: 50,
                    ttl_ms: 5000,
                    stop_loss_fp: FixedPrice::from_f64(stop_loss_price).raw(),
                    take_profit_fp: FixedPrice::from_f64(take_profit_price).raw(),
                    placement_type: 0, // AtBest
                    post_only: cmd_post_only,
                    is_close: 0,
                    _pad2: [0; 5],
                };

                // ── Market Impact Pre-Trade Check ──
                let order_size_usdt = FixedPrice(cmd.price).to_f64()
                    * fixed_point::FixedQty(cmd.qty).to_f64();
                let mid = FixedPrice(snapshot.mid_price).to_f64();
                let bid_depth = snapshot.bid_depth_usdt as f64 / FixedPrice::PRECISION as f64;
                let ask_depth = snapshot.ask_depth_usdt as f64 / FixedPrice::PRECISION as f64;
                let impact = impact_model.estimate_impact(
                    (snapshot.symbol_id as usize).saturating_sub(1),
                    order_size_usdt,
                    mid,
                    bid_depth,
                    ask_depth,
                    snapshot.spread_bps as f64,
                    cmd.side == spsc::side::BUY,
                );

                if impact.total_bps > 50.0 {
                    warn!("[strategy] 🚫 Order rejected: market impact {:.1}bps exceeds 50bps cap", impact.total_bps);
                } else {
                    if impact.should_split {
                        info!("[strategy] ⚠️ High impact {:.1}bps — recommend TWAP {} slices",
                            impact.total_bps, impact.recommended_slices);
                    }

                    // ── Institutional: Pre-Trade Risk + Position Slot Checks ──
                    // BEFORE pushing to SPSC, validate through the risk engine and
                    // acquire a position slot. This prevents overexposure and ensures
                    // all trades meet institutional risk limits.
                    match cmd.validate() {
                        Ok(()) => {
                            // Step 1: Pre-trade risk check (leverage, margin, concentration)
                            if let Err(rejection) = pre_trade_risk.check(&cmd) {
                                warn!("[strategy] 🛡️ Pre-trade risk rejection: {:?}", rejection);
                                continue;
                            }

                            // Step 1b: Correlation-based exposure check (FEATURE 6)
                            let mut position_notional = FixedPrice(cmd.price).to_f64()
                                * fixed_point::FixedQty(cmd.qty).to_f64();
                            let mut cmd = cmd; // Make cmd mutable for potential resizing
                            if let Err(reason) = correlation_limiter.check_position_limit(symbol_name, position_notional) {
                                warn!("[strategy] 🔗 Correlation limit rejection: {}", reason);
                                // Try to reduce position size to fit within limit
                                let max_allowed = correlation_limiter.max_allowed_position_size(symbol_name);
                                if max_allowed > 0.0 {
                                    let scale_factor = max_allowed / position_notional;
                                    let reduced_qty = (fixed_point::FixedQty(cmd.qty).to_f64() * scale_factor).floor() as i64;
                                    if reduced_qty >= 1 {
                                        let mut cmd_resized = cmd;
                                        cmd_resized.qty = fixed_point::FixedQty::from_f64(reduced_qty as f64).raw();
                                        warn!(
                                            "[strategy] 🔗 Correlation limit: resized from {} to {} contracts",
                                            fixed_point::FixedQty(cmd.qty).to_f64(), reduced_qty
                                        );
                                        cmd = cmd_resized;
                                        position_notional = max_allowed; // Update for subsequent checks
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            }

                            // Step 2: Position slot acquisition (hard limit: 3 concurrent)
                            if !position_slots.try_acquire() {
                                warn!(
                                    "[strategy] 📍 Position slots full ({}/{}), skipping signal",
                                    position_slots.active_positions(),
                                    position_slots.max_slots()
                                );
                                continue;
                            }

                            // Step 3: Track position in ExitEvaluator for native exits
                            let entry_f64 = entry_price;
                            let sl_f64 = FixedPrice(cmd.stop_loss_fp).to_f64();
                            let tp_f64 = FixedPrice(cmd.take_profit_fp).to_f64();
                            let size_f64 = fixed_point::FixedQty(cmd.qty).to_f64();
                            let ttl_ns = (cmd.ttl_ms as u64) * 1_000_000;
                            exit_evaluator.track_position(
                                snapshot.symbol_id,
                                is_buy,
                                entry_f64,
                                sl_f64,
                                tp_f64,
                                size_f64.round() as i64,
                                ttl_ns,
                            );

                            // Upgrade 4: Initialize trailing stop for this position
                            if sl_f64 > 0.0 {
                                trailing_stops.insert(
                                    snapshot.symbol_id,
                                    exit_evaluator::TrailingStopState::new(entry_f64, sl_f64, is_buy),
                                );
                            }

                            // Directive 1: Also track in PositionLifecycleManager
                            lifecycle_mgr.track_position(
                                snapshot.symbol_id,
                                is_buy,
                                entry_f64,
                                size_f64.round() as i64,
                                cmd.leverage as i32,
                            );

                            // Step 4: Record book_to_signal latency
                            latency_tracker.book_to_signal.record_since(signal_start);

                            // Step 5: Push to execution ring
                            if !exec_ring.try_push(cmd) {
                                warn!("[strategy] Execution ring full — dropping signal");
                                // Release the slot we just acquired
                                position_slots.release();
                                exit_evaluator.untrack_position(snapshot.symbol_id);
                                // Record the drop in the overflow monitor
                                if overflow_monitor.record_drop() {
                                    if let Some(ref cb) = circuit_breaker {
                                        cb.trip(crate::circuit_breaker::TripReason::OrderRateAnomaly);
                                        error!("[strategy] 🚨 Circuit breaker tripped: SPSC overflow rate anomaly");
                                    }
                                }
                            }
                        }
                        Err(reason) => {
                            warn!("[strategy] {}", reason);
                        }
                    }
                }
            }
        } else {
            std::hint::spin_loop();
        }
    }
}

/// Execution router loop — Institutional Grade with Circuit Breaker & SL/TP.
///
/// Reads OrderCommands from strategy SPSC and processes them through the
/// full institutional execution pipeline:
///
/// 1. **Circuit Breaker Check**: Reject all orders if halted
/// 2. **Validate**: Ensure SL is set (reject unprotected trades)
/// 3. **Route**: SmartOrderRouter selects best venue (TASK 1: multi-exchange SOR when enabled)
/// 4. **Pre-flight**: Check margin & exposure limits
/// 5. **Submit**: Send order via REST gateway (or multi-venue parallel submission)
/// 6. **Protect**: Submit SL/TP as conditional orders
/// 7. **Monitor**: Track fills, PnL, and queue position
/// 8. **Circuit Breaker Update**: Record trade results
///
/// TASK 1: When USE_MULTI_EXCHANGE=on, uses the multi-exchange SOR to split orders
/// across Gate.io, Binance, and Bybit based on global book liquidity.
fn execution_router_loop(
    exec_ring: &'static SpscRingBuffer<OrderCommand, STRATEGY_TO_EXEC_CAPACITY>,
    manual_cmd_rx: crossbeam_channel::Receiver<dashboard_server::ManualTradeRequest>,
    gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>>,
    forex_gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>>,
    circuit_breaker: &'static CircuitBreaker,
    registry: Arc<SymbolRegistry>,
    lifecycle_tracker: &'static OrderLifecycleTracker,
    latency_tracker: &'static PipelineLatencyTracker,
    position_slots: &'static PositionSlotManager,
    dashboard_state: Arc<DashboardState>,
    funding_rates: Arc<parking_lot::RwLock<HashMap<String, f64>>>,
    exec_analytics: &'static parking_lot::Mutex<ExecutionAnalytics>,
    shared_prices: Arc<Vec<AtomicU64>>,
    // TASK 1: Multi-exchange SOR parameters
    global_book_registry: Option<Arc<multi_exchange::GlobalBookRegistry>>,
    multi_gateways: HashMap<multi_exchange::ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    multi_exchange_enabled: bool,
) {
    info!("[execution] Starting execution router on dedicated core (Institutional)");

    // Initialize execution context
    let mbo_book = mbo_book::MboBook::new();
    let adverse_detector = adverse_selection::AdverseSelectionDetector::with_defaults();
    
    // TASK 6: Build legacy SmartOrderRouter with multi-exchange venues when enabled
    let smart_router_inst = if multi_exchange_enabled {
        let venues = vec![
            smart_router::VenueState {
                exchange_id: 0,
                name: "gateio".to_string(),
                spread_bps: 2,
                taker_fee_bps: 5,
                maker_fee_bps: -1,
                bid_depth_usdt: 100_000.0,
                ask_depth_usdt: 100_000.0,
                at_rate_limit: false,
                last_latency_us: 0,
                last_update_ns: 0,
                enabled: true,
            },
            smart_router::VenueState {
                exchange_id: 1,
                name: "binance".to_string(),
                spread_bps: 1,
                taker_fee_bps: 4,
                maker_fee_bps: -2,
                bid_depth_usdt: 500_000.0,
                ask_depth_usdt: 500_000.0,
                at_rate_limit: false,
                last_latency_us: 0,
                last_update_ns: 0,
                enabled: true,
            },
            smart_router::VenueState {
                exchange_id: 2,
                name: "bybit".to_string(),
                spread_bps: 2,
                taker_fee_bps: 6,
                maker_fee_bps: -1,
                bid_depth_usdt: 200_000.0,
                ask_depth_usdt: 200_000.0,
                at_rate_limit: false,
                last_latency_us: 0,
                last_update_ns: 0,
                enabled: true,
            },
        ];
        info!("[execution] Legacy SmartOrderRouter initialized with {} venues (multi-exchange mode)", venues.len());
        smart_router::SmartOrderRouter::new(venues)
    } else {
        smart_router::SmartOrderRouter::default_venues()
    };
    
    let ws_mgr = ws_order_manager::WsOrderManager::new_paper();
    let mut exec_ctx = execution_gateway::ExecutionContext::new(
        mbo_book,
        adverse_detector,
        smart_router_inst,
        ws_mgr,
    );

    // TASK 5: Initialize CrossVenueMarginMonitor for multi-exchange margin health tracking
    let mut margin_monitor = multi_exchange::margin_monitor::CrossVenueMarginMonitor::with_defaults();
    if multi_exchange_enabled {
        info!("[execution] Multi-exchange margin monitor initialized (min_ratio=30%, critical=15%)");
    }
    
    // TASK 2: Initialize CrossExchangeFundingArb for funding rate arbitrage
    let mut cross_funding_arb = multi_exchange::funding_arb::CrossExchangeFundingArb::with_defaults();
    let mut last_funding_arb_check = std::time::Instant::now();
    let mut funding_arb_positions: HashMap<String, FundingArbPosition> = HashMap::new();
    if multi_exchange_enabled {
        info!("[execution] Cross-exchange funding arbitrage initialized (min_net_rate=0.005%, min_apr=10%)");
    }
    
    // TASK 3: Initialize CrossExchangeMarketMaker for hedged market making
    let mut cross_mm = multi_exchange::cross_exchange_mm::CrossExchangeMarketMaker::with_defaults();
    let mut last_mm_check = std::time::Instant::now();
    if multi_exchange_enabled {
        info!("[execution] Cross-exchange market maker initialized (Gate.io maker, Binance hedge)");
    }
    
    // TASK 4: Initialize StatArbEngine for statistical arbitrage
    let mut stat_arb_engine = multi_exchange::stat_arb::StatArbEngine::with_defaults();
    let mut last_stat_arb_check = std::time::Instant::now();
    if multi_exchange_enabled {
        info!("[execution] Statistical arbitrage engine initialized (2-sigma entry, 0.5-sigma exit)");
    }

    // Initialize event-sourced order state machine
    let mut order_state_machine = order_state_machine::OrderStateMachine::new();

    // Initialize PnL tracking for position entries
    let mut position_entries: HashMap<u16, (f64, i64, bool)> = HashMap::new();

    // Upgrade 3: Initialize TWAP executor for large order splitting
    let mut twap_exec = twap_executor::TwapExecutor::new();
    info!("[execution] TWAP/Iceberg executor initialized (max 5 concurrent)");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime for execution");
        
    let http_client = reqwest::Client::new();

    rt.block_on(async {
        let mut last_queue_check = std::time::Instant::now();
        let mut last_health_check = std::time::Instant::now();
        let mut orders_submitted: u64 = 0;
        let mut orders_rejected: u64 = 0;
        let mut total_pnl_fp: i64 = 0;

        // Fetch initial balance for daily drawdown tracking.
        //
        // This also serves as a **connectivity & auth diagnostic**: if this call
        // fails with INVALID_KEY, the engine logs detailed troubleshooting info
        // so the user can immediately identify the problem.
        if let Some(ref gw) = gateway {
            info!("[execution] Testing Gate.io authentication with balance check...");
            match gw.get_balance().await {
                Ok(balance) => {
                    let balance_fp = (balance * 1e8) as i64;
                    circuit_breaker.set_daily_start_balance(balance_fp);
                    info!("[execution] ✅ Auth OK — Initial balance: ${:.2} — circuit breaker armed", balance);
                }
                Err(e) => {
                    let err_str = format!("{}", e);
                    if err_str.contains("INVALID_KEY") {
                        error!("[execution] ❌ INVALID_KEY from Gate.io. Troubleshooting:");
                        error!("[execution]   1. Ensure GATEIO_TESTNET_API_KEY is a FUTURES testnet key (not spot, not mainnet)");
                        error!("[execution]   2. Gate.io has separate API keys for Spot vs Futures — you need the FUTURES key");
                        error!("[execution]   3. Generate keys at: https://www.gate.io/testnet/futures_trade/USDT/BTC_USDT (testnet site)");
                        error!("[execution]   4. Check for whitespace/newlines in your .env file around the key values");
                        error!("[execution]   5. Ensure the key has 'Futures' permission enabled");
                    }
                    warn!("[execution] Failed to fetch initial balance: {} — using $10k default", e);
                    circuit_breaker.set_daily_start_balance(10_000_0000_0000); // $10k default
                }
            }

            // FIX 5: Set margin mode to cross-margin for all symbols
            if let Some(ref gw) = gateway {
                info!("[execution] Setting margin mode to cross-margin for all symbols...");
                for id in registry.all_ids() {
                    let symbol = registry.get_name(id);
                    if let Err(e) = gw.set_margin_mode(symbol, "cross").await {
                        warn!("[execution] Failed to set margin mode for {}: {}", symbol, e);
                    }
                }
            }
        }

        loop {
            // Check for manual trade requests from dashboard
            while let Ok(manual_req) = manual_cmd_rx.try_recv() {
                let sym_upper = manual_req.symbol.replace('/', "_").to_uppercase();
                let sym_id = registry.get_id(&sym_upper);
                if sym_id == 0 {
                    warn!("[execution] Manual trade rejected: unknown symbol '{}'", sym_upper);
                    continue;
                }
                
                info!("[execution] Manual trade received: {} {} {} contracts lev={}x SL={:.4} TP={:.4}",
                    manual_req.side, sym_upper, manual_req.size, manual_req.leverage,
                    manual_req.stop_loss, manual_req.take_profit);
                
                let is_buy = manual_req.side.to_lowercase() == "buy";
                
                // Route to the appropriate gateway (default: Gate.io)
                // Note: binance_gateway and bybit_gateway would need to be passed to this function
                // for multi-exchange support. For now, use the primary gateway.
                let target_gw: Option<&Arc<dyn ExecutionGateway + Send + Sync>> = match manual_req.exchange.as_deref() {
                    Some("binance") => {
                        warn!("[execution] Manual trade: Binance gateway not yet wired to execution router");
                        gateway.as_ref() // Fallback to Gate.io
                    },
                    Some("bybit") => {
                        warn!("[execution] Manual trade: Bybit gateway not yet wired to execution router");
                        gateway.as_ref() // Fallback to Gate.io
                    },
                    _ => gateway.as_ref(), // Default to Gate.io
                };
                
                if let Some(gw) = target_gw {
                    // Step 1: Set leverage
                    if let Err(e) = gw.set_leverage(&sym_upper, manual_req.leverage as i32).await {
                        warn!("[execution] Manual trade: failed to set leverage: {}", e);
                    }
                    
                    // Step 2: Set margin mode to cross
                    if let Err(e) = gw.set_margin_mode(&sym_upper, "cross").await {
                        warn!("[execution] Manual trade: failed to set margin mode: {}", e);
                    }
                    
                    // Step 3: Build and submit the order
                    let side = if is_buy { execution_gateway::OrderSide::Buy } else { execution_gateway::OrderSide::Sell };
                    let order_type = if manual_req.price.is_some() {
                        execution_gateway::OrderType::Limit
                    } else {
                        execution_gateway::OrderType::Market
                    };
                    let tif = if manual_req.price.is_some() { "gtc" } else { "ioc" };
                    
                    let intent = execution_gateway::OrderIntent {
                        symbol: sym_upper.clone(),
                        side: side.clone(),
                        size: manual_req.size,
                        order_type,
                        price: manual_req.price,
                        reduce_only: false,
                        leverage: Some(manual_req.leverage as i32),
                        time_in_force: tif.to_string(),
                        slippage_cap_pct: Some(0.005),
                        placement: execution_state::PlacementType::AtBest,
                        stop_loss: Some(manual_req.stop_loss),
                        take_profit: Some(manual_req.take_profit),
                        confidence: 1.0,
                        signal_tag: "manual_dashboard".to_string(),
                    };
                    
                    match execution_gateway::submit_with_retry(&**gw, intent).await {
                        Ok(res) => {
                            info!("[execution] Manual trade filled: {} {} {} @ {:.4} (order_id={})",
                                manual_req.side, sym_upper, res.filled_size, res.avg_fill_price, res.order_id);
                            
                            orders_submitted += 1;
                            dashboard_state.orders_submitted.store(orders_submitted, Ordering::Relaxed);
                            dashboard_state.total_fills.fetch_add(1, Ordering::Relaxed);
                            
                            // Track position entry for PnL calculation
                            position_entries.insert(sym_id, (res.avg_fill_price, res.filled_size, is_buy));
                            
                            // Submit SL/TP conditional orders
                            let parent_side = if is_buy { execution_gateway::OrderSide::Buy } else { execution_gateway::OrderSide::Sell };
                            let gw_clone = gw.clone();
                            let sym_clone = sym_upper.clone();
                            let sl = manual_req.stop_loss;
                            let tp = manual_req.take_profit;
                            let filled = res.filled_size;
                            tokio::spawn(async move {
                                if sl > 0.0 {
                                    match gw_clone.submit_conditional_sl(&sym_clone, &parent_side, filled, sl).await {
                                        Ok(()) => info!("[execution] Manual trade SL placed: {} @ {:.4}", sym_clone, sl),
                                        Err(e) => warn!("[execution] Manual trade SL failed: {}", e),
                                    }
                                }
                                if tp > 0.0 {
                                    match gw_clone.submit_conditional_tp(&sym_clone, &parent_side, filled, tp).await {
                                        Ok(()) => info!("[execution] Manual trade TP placed: {} @ {:.4}", sym_clone, tp),
                                        Err(e) => warn!("[execution] Manual trade TP failed: {}", e),
                                    }
                                }
                            });
                            
                            // Acquire a position slot
                            if !position_slots.try_acquire() {
                                warn!("[execution] Manual trade: position slots full");
                            }
                        }
                        Err(e) => {
                            warn!("[execution] Manual trade failed: {}", e);
                        }
                    }
                } else {
                    warn!("[execution] Manual trade: no gateway available for exchange {:?}", manual_req.exchange);
                }
            }

            // ── Step 0: Check circuit breaker ──
            if circuit_breaker.is_trading_halted() {
                // Drain the ring to avoid stale orders when trading resumes.
                // CRITICAL: Release position slots for every drained order to
                // prevent permanent slot exhaustion during circuit breaker events.
                while exec_ring.try_pop().is_some() {
                    orders_rejected += 1;
                    position_slots.release();
                }
                // Check cooldown for auto-recovery
                circuit_breaker.check_cooldown();
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }

            if let Some(cmd) = exec_ring.try_pop() {
                // ── Step 1: Validate the command ──
                if let Err(reason) = cmd.validate() {
                    warn!("[execution] Order rejected: {}", reason);
                    orders_rejected += 1;
                    continue;
                }

                // ── Step 2: Record order submission in circuit breaker ──
                circuit_breaker.on_order_submitted();
                if circuit_breaker.is_trading_halted() {
                    warn!("[execution] Circuit breaker tripped during order rate check");
                    continue;
                }

                // ── Step 3: Route via SmartOrderRouter ──
                let order_size_usdt = FixedPrice(cmd.price).to_f64()
                    * fixed_point::FixedQty(cmd.qty).to_f64();
                let is_maker = cmd.order_type == spsc::order_cmd_type::LIMIT;
                let routing = exec_ctx.smart_router.route(order_size_usdt, is_maker);
                let symbol_name = registry.get_name(cmd.symbol_id);

                // ── TASK 1: Multi-Exchange SOR Routing ──
                // When multi-exchange is enabled and we have a global book, use the multi-exchange SOR
                // to split orders across venues for better execution quality.
                if multi_exchange_enabled && global_book_registry.is_some() && !config::is_forex_symbol(symbol_name) {
                    let gbr = global_book_registry.as_ref().unwrap();
                    if let Some(book_arc) = gbr.get(cmd.symbol_id) {
                        let book = book_arc.read();
                        let multi_sor = multi_exchange::sor::SmartOrderRouter::new(
                            multi_exchange::sor::SorConfig {
                                min_split_size_usdt: 5000.0,
                                max_venues: 3,
                                max_slippage_bps: 30.0,
                                prefer_maker: is_maker,
                            }
                        );
                        
                        let side = if cmd.side == spsc::side::BUY {
                            execution_gateway::OrderSide::Buy
                        } else {
                            execution_gateway::OrderSide::Sell
                        };
                        
                        let mid_price_fp = book.global_mid_fp().unwrap_or(cmd.price);
                        let total_size_fp = cmd.qty;
                        
                        let sor_result = multi_sor.route(&book, side.clone(), total_size_fp, mid_price_fp, symbol_name);
                        
                        // Store SOR result for dashboard preview
                        dashboard_state.set_funding_arb_json(serde_json::to_string(&sor_result.to_json()).unwrap_or_default());
                        
                        if !sor_result.slices.is_empty() {
                            info!(
                                "[execution] Multi-Exchange SOR: {} slices, reason='{}', savings={:.1}bps",
                                sor_result.slices.len(),
                                sor_result.routing_reason,
                                sor_result.estimated_savings_bps
                            );
                            
                            // Execute all slices in parallel using tokio::join!
                            let mut futures = Vec::new();
                            for slice in &sor_result.slices {
                                if let Some(gw) = multi_gateways.get(&slice.exchange).cloned() {
                                    let slice_side = slice.side.clone();
                                    let slice_size = (slice.size as f64 / 1e8) as i64;
                                    let slice_price = if slice.price_fp > 0 {
                                        Some(FixedPrice(slice.price_fp).to_f64())
                                    } else {
                                        Some(FixedPrice(cmd.price).to_f64())
                                    };
                                    let slice_symbol = slice.symbol.clone();
                                    let slice_exchange = slice.exchange;
                                    
                                    let order_type = if slice.is_maker {
                                        execution_gateway::OrderType::Limit
                                    } else {
                                        execution_gateway::OrderType::Market
                                    };
                                    
                                    let intent = execution_gateway::OrderIntent {
                                        symbol: slice_symbol.clone(),
                                        side: slice_side,
                                        size: slice_size.max(1),
                                        order_type,
                                        price: slice_price,
                                        reduce_only: false,
                                        leverage: Some(cmd.target_leverage()),
                                        time_in_force: if slice.is_maker { "gtc".to_string() } else { "ioc".to_string() },
                                        slippage_cap_pct: Some(cmd.max_slippage_bps as f64 / 10000.0),
                                        placement: execution_state::PlacementType::AtBest,
                                        stop_loss: if cmd.has_stop_loss() { Some(FixedPrice(cmd.stop_loss_fp).to_f64()) } else { None },
                                        take_profit: if cmd.has_take_profit() { Some(FixedPrice(cmd.take_profit_fp).to_f64()) } else { None },
                                        confidence: 0.0,
                                        signal_tag: "multi_exchange_sor".to_string(),
                                    };
                                    
                                    futures.push(async move {
                                        let result = execution_gateway::submit_with_retry(&*gw, intent).await;
                                        (slice_exchange, result)
                                    });
                                }
                            }
                            
                            // Execute all slices in parallel
                            let results = future::join_all(futures).await;
                            
                            // Aggregate results
                            let mut total_filled: i64 = 0;
                            let mut weighted_price_sum: f64 = 0.0;
                            let mut total_fees: f64 = 0.0;
                            let mut any_success = false;
                            
                            for (exchange, result) in results {
                                match result {
                                    Ok(res) => {
                                        total_filled += res.filled_size;
                                        weighted_price_sum += res.avg_fill_price * res.filled_size as f64;
                                        total_fees += res.fee;
                                        any_success = true;
                                        info!(
                                            "[execution] SOR slice on {}: filled={} @ {:.4}",
                                            exchange.name(), res.filled_size, res.avg_fill_price
                                        );
                                        orders_submitted += 1;
                                        dashboard_state.orders_submitted.store(orders_submitted, Ordering::Relaxed);
                                        dashboard_state.total_fills.fetch_add(1, Ordering::Relaxed);
                                    }
                                    Err(e) => {
                                        warn!("[execution] SOR slice on {} failed: {}", exchange.name(), e);
                                        // Don't rollback successful slices - partial fills are acceptable
                                    }
                                }
                            }
                            
                            if any_success {
                                let avg_fill_price = if total_filled > 0 {
                                    weighted_price_sum / total_filled as f64
                                } else {
                                    0.0
                                };
                                info!(
                                    "[execution] Multi-Exchange SOR complete: total_filled={}, avg_price={:.4}, fees={:.4}",
                                    total_filled, avg_fill_price, total_fees
                                );
                                
                                // Track position entry for PnL calculation
                                let is_buy = cmd.side == spsc::side::BUY;
                                position_entries.insert(cmd.symbol_id, (avg_fill_price, total_filled, is_buy));
                                
                                // Record in circuit breaker as successful trade
                                circuit_breaker.on_trade_result(0); // No PnL yet for new entry
                            }
                            
                            continue; // Skip single-exchange execution path
                        }
                    }
                }

                info!(
                    "[execution] Routing order #{}: sym={} side={} qty={} price={:.4} SL={:.4} TP={:.4} venue={} cost={}bps",
                    orders_submitted + 1,
                    registry.get_name(cmd.symbol_id),
                    if cmd.side == spsc::side::BUY { "BUY" } else { "SELL" },
                    fixed_point::FixedQty(cmd.qty).to_f64(),
                    FixedPrice(cmd.price).to_f64(),
                    FixedPrice(cmd.stop_loss_fp).to_f64(),
                    FixedPrice(cmd.take_profit_fp).to_f64(),
                    routing.exchange_id,
                    routing.expected_cost_bps,
                );

                // ── Step 4: Route to correct gateway (Mandate 3: Forex routing) ──
                // Forex symbols → forex_gateway, crypto symbols → crypto gateway
                let active_gw: Option<&Arc<dyn ExecutionGateway + Send + Sync>> =
                    if config::is_forex_symbol(symbol_name) {
                        if forex_gateway.is_some() {
                            info!("[execution] Routing {} through FOREX gateway", symbol_name);
                        }
                        forex_gateway.as_ref()
                    } else {
                        gateway.as_ref()
                    };

                if let Some(gw) = active_gw {
                    // FIX 2: Check available margin before order submission
                    let order_value_usdt = FixedPrice(cmd.price).to_f64()
                        * fixed_point::FixedQty(cmd.qty).to_f64();
                    let required_margin = order_value_usdt / cmd.target_leverage() as f64;
                    
                    match gw.get_balance().await {
                        Ok(available_balance) => {
                            if available_balance < required_margin * 1.1 {
                                warn!(
                                    "[execution] Insufficient margin: need ${:.2}, have ${:.2} — skipping order",
                                    required_margin, available_balance
                                );
                                orders_rejected += 1;
                                position_slots.release();
                                continue;
                            }
                            debug!(
                                "[execution] Margin check passed: ${:.2} available, ${:.2} required",
                                available_balance, required_margin
                            );
                        }
                        Err(e) => {
                            warn!("[execution] Balance check failed: {} — proceeding with order", e);
                        }
                    }

                    let side = if cmd.side == spsc::side::BUY {
                        execution_gateway::OrderSide::Buy
                    } else {
                        execution_gateway::OrderSide::Sell
                    };

                    let order_type = if cmd.post_only == 1 {
                        execution_gateway::OrderType::PostOnly
                    } else if cmd.order_type == spsc::order_cmd_type::MARKET {
                        execution_gateway::OrderType::Market
                    } else {
                        execution_gateway::OrderType::Limit
                    };

                    let tif = match order_type {
                        execution_gateway::OrderType::PostOnly => "poc",
                        execution_gateway::OrderType::Market => "ioc",
                        execution_gateway::OrderType::Limit => "gtc",
                    };

                    let intent = execution_gateway::OrderIntent {
                        symbol: symbol_name.to_string(),
                        side,
                        size: fixed_point::FixedQty(cmd.qty).to_f64() as i64,
                        order_type: order_type.clone(),
                        price: Some(FixedPrice(cmd.price).to_f64()),
                        reduce_only: false,
                        leverage: Some(cmd.target_leverage()),
                        time_in_force: tif.to_string(),
                        slippage_cap_pct: Some(cmd.max_slippage_bps as f64 / 10000.0),
                        placement: execution_state::PlacementType::AtBest,
                        stop_loss: if cmd.has_stop_loss() {
                            Some(FixedPrice(cmd.stop_loss_fp).to_f64())
                        } else {
                            None
                        },
                        take_profit: if cmd.has_take_profit() {
                            Some(FixedPrice(cmd.take_profit_fp).to_f64())
                        } else {
                            None
                        },
                        confidence: 0.0,
                        signal_tag: String::new(),
                    };

                    // Submit the main order with retry logic
                    let order_start = std::time::Instant::now();

                    // Record signal-to-order latency (time from strategy signal to execution receipt)
                    // cmd.signal_ns carries the book snapshot timestamp; compare to current time
                    {
                        let now_ns = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        let signal_to_order_us = now_ns.saturating_sub(cmd.signal_ns) / 1000;
                        latency_tracker.signal_to_order.record(signal_to_order_us.max(1));
                    }

                    // Register order in lifecycle tracker
                    let client_oid = format!("c-{}", orders_submitted + 1);
                    let lifecycle_order = crate::order_lifecycle::OrderLifecycle::new(
                        client_oid.clone(),
                        symbol_name.to_string(),
                        if cmd.side == spsc::side::BUY { "buy".into() } else { "sell".into() },
                        format!("{:?}", order_type),
                        fixed_point::FixedQty(cmd.qty).to_f64(),
                        FixedPrice(cmd.price).to_f64(),
                        "engine".into(),
                        0.0,
                    );
                    lifecycle_tracker.register_order(lifecycle_order);

                    match execution_gateway::submit_with_retry(&**gw, intent).await {
                        Ok(res) => {
                            // Record order-to-ack latency
                            latency_tracker.order_to_ack.record_since(order_start);

                            orders_submitted += 1;
                            // BUG 7 FIX: Update dashboard state with live data
                            dashboard_state.orders_submitted.store(orders_submitted, Ordering::Relaxed);
                            dashboard_state.total_fills.fetch_add(1, Ordering::Relaxed);
                            info!(
                                "[execution] ✅ Order {} filled: size={}, avg_price={:.4}, latency={}μs",
                                res.order_id, res.filled_size, res.avg_fill_price, res.latency_us
                            );
                            
                            // FIX 4 (Task 8): Post-only verification task
                            // If this was a post-only order, spawn a task to verify it wasn't immediately cancelled
                            if cmd.post_only == 1 && res.status == "open" {
                                let gw_clone = gw.clone();
                                let order_id_clone = res.order_id.clone();
                                let symbol_clone = symbol_name.to_string();
                                let sym_id = cmd.symbol_id;
                                tokio::spawn(async move {
                                    // Wait 3 seconds for the order to rest on the book
                                    tokio::time::sleep(Duration::from_secs(3)).await;
                                    
                                    // Query order status
                                    match gw_clone.get_order_status(&order_id_clone, &symbol_clone).await {
                                        Ok(Some(status)) if status.status == "cancelled" => {
                                            warn!(
                                                "[execution] ⚠️ Post-only order {} for {} was cancelled by exchange — releasing position slot",
                                                order_id_clone, symbol_clone
                                            );
                                            // Release the position slot that was acquired for this order
                                            position_slots.release();
                                            // Untrack from exit evaluator if it was tracked
                                            // Note: We can't directly call exit_evaluator.untrack_position here
                                            // because it's owned by the strategy thread. The next health check
                                            // will detect the discrepancy and clean up.
                                        }
                                        Ok(Some(status)) => {
                                            debug!(
                                                "[execution] Post-only order {} status: {}",
                                                order_id_clone, status.status
                                            );
                                        }
                                        Ok(None) => {
                                            debug!(
                                                "[execution] Post-only order {} no longer exists (filled or cancelled)",
                                                order_id_clone
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "[execution] Failed to query post-only order {} status: {}",
                                                order_id_clone, e
                                            );
                                        }
                                    }
                                });
                            }

                            // FIX 1: Only insert position entries on non-close fills
                            if cmd.is_close == 0 {
                                position_entries.insert(cmd.symbol_id, (res.avg_fill_price, res.filled_size, cmd.side == spsc::side::BUY));
                            }

                            // Task 1: Record fill in ExecutionAnalytics
                            // Use fill price as proxy for mid prices (signal_price from cmd, mid_at_signal = fill_price)
                            let fill_price = res.avg_fill_price;
                            let signal_price = FixedPrice(cmd.price).to_f64();
                            let mid_at_signal = fill_price; // Approximate mid at signal time
                            let size_usdt = fill_price * res.filled_size as f64;
                            
                            // Spawn async task to fetch mid_after_1s
                            let gw_clone = gw.clone();
                            let symbol_name_clone = symbol_name.to_string();
                            let exec_analytics_clone = exec_analytics;
                            tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_secs(1)).await;
                                let mid_after_1s = match gw_clone.get_ticker(&symbol_name_clone).await {
                                    Ok(ticker) => ticker.last,
                                    Err(_) => fill_price, // Fallback to fill price
                                };
                                
                                exec_analytics_clone.lock().record_fill(
                                    fill_price,
                                    signal_price,
                                    mid_at_signal,
                                    mid_after_1s,
                                    size_usdt,
                                );
                            });

                            // Map exchange ID and record fill in lifecycle tracker
                            lifecycle_tracker.map_exchange_id(&res.order_id, &client_oid);
                            lifecycle_tracker.record_fill(
                                &client_oid,
                                crate::order_lifecycle::Fill {
                                    fill_id: format!("f-{}", orders_submitted),
                                    price: res.avg_fill_price,
                                    quantity: res.filled_size as f64,
                                    fee: 0.0,
                                    fee_currency: "USDT".into(),
                                    timestamp_us: crate::order_lifecycle::now_micros(),
                                    is_maker: cmd.post_only == 1,
                                },
                            );

                            // FIX 1: Use is_close field instead of reduce_only for PnL calculation
                            let pnl_fp = if cmd.is_close == 1 {
                                if let Some((entry_price, size, is_long)) = position_entries.remove(&cmd.symbol_id) {
                                    let close_price = res.avg_fill_price;
                                    let pnl = if is_long {
                                        (close_price - entry_price) * size as f64
                                    } else {
                                        (entry_price - close_price) * size as f64
                                    };
                                    (pnl * 1e8) as i64
                                } else {
                                    0i64
                                }
                            } else {
                                0i64
                            };
                            total_pnl_fp += pnl_fp;
                            circuit_breaker.on_trade_result(pnl_fp);

                            // ── FIX 6: Submit SL/TP conditional orders to exchange ──
                            // Gate.io supports setting SL/TP via the price_trigger REST API.
                            // We submit them asynchronously via a spawned task so the execution
                            // loop is not blocked. The ExitEvaluator also monitors these locally
                            // as a safety net in case the exchange-side trigger fails.
                            {
                                let sl_price = if cmd.has_stop_loss() {
                                    Some(FixedPrice(cmd.stop_loss_fp).to_f64())
                                } else {
                                    None
                                };
                                let tp_price = if cmd.has_take_profit() {
                                    Some(FixedPrice(cmd.take_profit_fp).to_f64())
                                } else {
                                    None
                                };
                                if sl_price.is_some() || tp_price.is_some() {
                                    let parent_side = if cmd.side == spsc::side::BUY {
                                        execution_gateway::OrderSide::Buy
                                    } else {
                                        execution_gateway::OrderSide::Sell
                                    };
                                    let filled_size = res.filled_size;
                                    let sym_for_sltp = symbol_name.to_string();

                                    if let Some(ref gw) = gateway {
                                        let gw_clone = gw.clone();
                                        tokio::spawn(async move {
                                            // Submit SL as reduce-only conditional order
                                            if let Some(sl) = sl_price {
                                                let sl_intent = execution_gateway::OrderIntent {
                                                    symbol: sym_for_sltp.clone(),
                                                    side: if parent_side == execution_gateway::OrderSide::Buy {
                                                        execution_gateway::OrderSide::Sell
                                                    } else {
                                                        execution_gateway::OrderSide::Buy
                                                    },
                                                    size: filled_size.abs(),
                                                    order_type: execution_gateway::OrderType::Market,
                                                    price: Some(sl),
                                                    reduce_only: true,
                                                    leverage: None,
                                                    time_in_force: "ioc".to_string(),
                                                    slippage_cap_pct: Some(0.01),
                                                    placement: execution_state::PlacementType::AtBest,
                                                    stop_loss: None,
                                                    take_profit: None,
                                                    confidence: 0.0,
                                                    signal_tag: "sl_conditional".to_string(),
                                                };
                                                match gw_clone.submit_conditional_sl(
                                                    &sym_for_sltp, &parent_side, filled_size, sl
                                                ).await {
                                                    Ok(()) => info!(
                                                        "[execution] 🛡️ SL conditional order placed on exchange: {} @ {:.4}",
                                                        sym_for_sltp, sl
                                                    ),
                                                    Err(e) => warn!(
                                                        "[execution] SL conditional order failed (tracked locally): {} @ {:.4} — {}",
                                                        sym_for_sltp, sl, e
                                                    ),
                                                }
                                                let _ = sl_intent;
                                            }
                                            // Submit TP as reduce-only conditional order
                                            if let Some(tp) = tp_price {
                                                match gw_clone.submit_conditional_tp(
                                                    &sym_for_sltp, &parent_side, filled_size, tp
                                                ).await {
                                                    Ok(()) => info!(
                                                        "[execution] 🎯 TP conditional order placed on exchange: {} @ {:.4}",
                                                        sym_for_sltp, tp
                                                    ),
                                                    Err(e) => warn!(
                                                        "[execution] TP conditional order failed (tracked locally): {} @ {:.4} — {}",
                                                        sym_for_sltp, tp, e
                                                    ),
                                                }
                                            }
                                        });
                                    } else {
                                        if let Some(sl) = sl_price {
                                            info!("[execution] 🛡️ SL tracked locally: {} @ {:.4}", symbol_name, sl);
                                        }
                                        if let Some(tp) = tp_price {
                                            info!("[execution] 🎯 TP tracked locally: {} @ {:.4}", symbol_name, tp);
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("[execution] ❌ Order submission failed: {}", e);
                            orders_rejected += 1;
                            dashboard_state.orders_rejected.store(orders_rejected, Ordering::Relaxed);
                            lifecycle_tracker.reject_order(&client_oid, &format!("{}", e));

                            // CRITICAL: Release the position slot that was acquired by the
                            // strategy thread before pushing to the execution ring. Without
                            // this, failed orders permanently leak slots, eventually
                            // blocking all new trades.
                            position_slots.release();
                            info!(
                                "[execution] 📍 Released position slot after failed order (active={}/{})",
                                position_slots.active_positions(),
                                position_slots.max_slots()
                            );

                            // If it's a connectivity issue, trip the circuit breaker
                            if matches!(e, execution_gateway::ExchangeError::Timeout
                                | execution_gateway::ExchangeError::ConnectionReset) {
                                error!("[execution] Connectivity issue — considering circuit breaker trip");
                                // Don't trip immediately on a single timeout, but track it
                            }
                        }
                    }
                } else {
                    // No gateway — signal-only mode, just log
                    info!(
                        "[execution] 📝 Signal-only: sym={} side={} price={:.4} SL={:.4} TP={:.4}",
                        registry.get_name(cmd.symbol_id),
                        if cmd.side == spsc::side::BUY { "BUY" } else { "SELL" },
                        FixedPrice(cmd.price).to_f64(),
                        FixedPrice(cmd.stop_loss_fp).to_f64(),
                        FixedPrice(cmd.take_profit_fp).to_f64(),
                    );
                    orders_submitted += 1;
                }
            } else {
                // Idle backoff to prevent CPU burn
                std::hint::spin_loop();

                // ── Periodic maintenance ──

                // Check resting orders for queue position degradation (every 1s)
                if last_queue_check.elapsed() > Duration::from_secs(1) {
                    let cancels = exec_ctx.check_resting_orders();
                    for (idx, reason) in cancels {
                        info!("[execution] Canceling resting order {} due to {:?}", idx, reason);
                        exec_ctx.ws_order_mgr.cancel_by_lifecycle_idx(idx, reason);
                    }
                    last_queue_check = std::time::Instant::now();
                    
                    // TASK 6: Update legacy SmartOrderRouter venue states from GlobalBookRegistry
                    if let Some(ref gbr) = global_book_registry {
                        for sym_id in gbr.all_symbol_ids() {
                            if let Some(book_arc) = gbr.get(sym_id) {
                                let book = book_arc.read();
                                for exchange in multi_exchange::ExchangeId::all() {
                                    if let Some(snap) = book.get_exchange_snapshot(exchange) {
                                        let spread_bps = if snap.best_bid_fp > 0 {
                                            let mid = (snap.best_bid_fp + snap.best_ask_fp) / 2;
                                            ((snap.best_ask_fp - snap.best_bid_fp) * 10000) / mid.max(1)
                                        } else { 
                                            100 
                                        } as i64;
                                        
                                        // Calculate depth from levels (sum of top 5 levels)
                                        let bid_depth: f64 = snap.bid_levels.iter()
                                            .take(5)
                                            .map(|(_, qty)| *qty as f64 / 1e8)
                                            .sum();
                                        let ask_depth: f64 = snap.ask_levels.iter()
                                            .take(5)
                                            .map(|(_, qty)| *qty as f64 / 1e8)
                                            .sum();
                                        
                                        exec_ctx.smart_router.update_venue(
                                            exchange as u8,
                                            spread_bps,
                                            bid_depth * 50000.0, // Scale to USDT (approx)
                                            ask_depth * 50000.0,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Upgrade 3: Tick the TWAP executor to submit ready slices
                if twap_exec.active_count() > 0 {
                    let mut current_prices = HashMap::new();
                    for id in registry.all_ids() {
                        if (id as usize) > 0 && (id as usize) <= shared_prices.len() {
                            let price_bits = shared_prices[id as usize - 1].load(Ordering::Relaxed);
                            let price = f64::from_bits(price_bits);
                            if price > 0.0 {
                                current_prices.insert(registry.get_name(id).to_string(), price);
                            }
                        }
                    }
                    let ready_slices = twap_exec.tick(&current_prices);
                    for slice in ready_slices {
                        if let Some(ref gw) = gateway {
                            let side = if slice.side == 0 {
                                execution_gateway::OrderSide::Buy
                            } else {
                                execution_gateway::OrderSide::Sell
                            };
                            let intent = execution_gateway::OrderIntent {
                                symbol: slice.symbol.clone(),
                                side,
                                size: slice.size,
                                order_type: execution_gateway::OrderType::Market,
                                price: Some(slice.price),
                                reduce_only: false,
                                leverage: None,
                                time_in_force: "ioc".to_string(),
                                slippage_cap_pct: Some(0.005),
                                placement: execution_state::PlacementType::AtBest,
                                stop_loss: None,
                                take_profit: None,
                                confidence: 0.0,
                                signal_tag: "twap_slice".to_string(),
                            };
                            match gw.submit_order(intent).await {
                                Ok(res) => {
                                    orders_submitted += 1;
                                    info!(
                                        "[execution] TWAP slice filled: {} size={} @ {:.4}",
                                        slice.symbol, res.filled_size, res.avg_fill_price
                                    );
                                }
                                Err(e) => {
                                    warn!("[execution] TWAP slice failed: {} — {}", slice.symbol, e);
                                }
                            }
                        }
                    }
                    twap_exec.cleanup_completed();
                }

                // Periodic funding rate fetch (every 30s)
                if last_health_check.elapsed() > Duration::from_secs(30) {
                    if let Some(ref _gw) = gateway {
                        let rates_clone = funding_rates.clone();
                        let symbols_clone: Vec<String> = registry.all_ids().into_iter().map(|id| registry.get_name(id).to_string()).collect();
                        let client_clone = http_client.clone();
                        tokio::spawn(async move {
                            for symbol in symbols_clone {
                                if let Some(rate) = funding_rate::fetch_funding_rate(&client_clone, "https://api.gateio.ws", &symbol).await {
                                    rates_clone.write().insert(symbol, rate);
                                }
                            }
                        });
                    }
                }
                
                // TASK 2: Cross-Exchange Funding Rate Arbitrage (every 60s)
                if multi_exchange_enabled && last_funding_arb_check.elapsed() > Duration::from_secs(60) {
                    last_funding_arb_check = std::time::Instant::now();
                    
                    let symbols: Vec<String> = registry.all_ids().into_iter()
                        .map(|id| registry.get_name(id).to_string())
                        .collect();
                    
                    // Fetch funding rates from all exchanges
                    for symbol in &symbols {
                        cross_funding_arb.fetch_all_rates(
                            &http_client, 
                            symbol,
                            false, // gateio_testnet
                            false, // binance_testnet
                            false, // bybit_testnet
                        ).await;
                    }
                    
                    // Scan for actionable opportunities
                    let opportunities = cross_funding_arb.scan_opportunities();
                    
                    // Push opportunities to dashboard
                    dashboard_state.set_funding_arb_json(
                        serde_json::to_string(&cross_funding_arb.to_json()).unwrap_or_default()
                    );
                    
                    // Execute actionable funding arb trades
                    for opp in opportunities.iter().filter(|o| o.is_actionable) {
                        // Check if we already have an active funding arb position for this symbol
                        if funding_arb_positions.contains_key(&opp.symbol) {
                            continue;
                        }
                        
                        info!(
                            "[execution] Funding Arb Opportunity: {} SHORT@{} ({:.4}%) LONG@{} ({:.4}%) net={:.4}% APR={:.1}%",
                            opp.symbol,
                            opp.short_exchange.name(), opp.short_rate * 100.0,
                            opp.long_exchange.name(), opp.long_rate * 100.0,
                            opp.net_rate * 100.0,
                            opp.annualized_apr * 100.0
                        );
                        
                        // Get gateways for both exchanges
                        let short_gw = multi_gateways.get(&opp.short_exchange);
                        let long_gw = multi_gateways.get(&opp.long_exchange);
                        
                        if short_gw.is_none() || long_gw.is_none() {
                            warn!("[execution] Funding arb skipped: gateway not available for {} or {}", 
                                opp.short_exchange.name(), opp.long_exchange.name());
                            continue;
                        }
                        
                        let short_gw = short_gw.unwrap().clone();
                        let long_gw = long_gw.unwrap().clone();
                        
                        // Calculate position size (2% of total equity)
                        let total_equity = margin_monitor.total_equity();
                        let position_notional = total_equity * 0.02;
                        let mid_price = shared_prices.get(0)
                            .map(|p| f64::from_bits(p.load(Ordering::Relaxed)))
                            .unwrap_or(50000.0);
                        let position_size = (position_notional / mid_price).max(1.0) as i64;
                        
                        // Build SHORT intent (on high funding exchange)
                        let short_intent = execution_gateway::OrderIntent {
                            symbol: opp.symbol.clone(),
                            side: execution_gateway::OrderSide::Sell,
                            size: position_size,
                            order_type: execution_gateway::OrderType::Market,
                            price: Some(mid_price),
                            reduce_only: false,
                            leverage: Some(3),
                            time_in_force: "ioc".to_string(),
                            slippage_cap_pct: Some(0.002),
                            placement: execution_state::PlacementType::AtBest,
                            stop_loss: Some(mid_price * 1.02), // 2% SL
                            take_profit: None, // Hold for funding
                            confidence: 1.0,
                            signal_tag: "funding_arb_short".to_string(),
                        };
                        
                        // Build LONG intent (on low funding exchange)
                        let long_intent = execution_gateway::OrderIntent {
                            symbol: opp.symbol.clone(),
                            side: execution_gateway::OrderSide::Buy,
                            size: position_size,
                            order_type: execution_gateway::OrderType::Market,
                            price: Some(mid_price),
                            reduce_only: false,
                            leverage: Some(3),
                            time_in_force: "ioc".to_string(),
                            slippage_cap_pct: Some(0.002),
                            placement: execution_state::PlacementType::AtBest,
                            stop_loss: Some(mid_price * 0.98), // 2% SL
                            take_profit: None,
                            confidence: 1.0,
                            signal_tag: "funding_arb_long".to_string(),
                        };
                        
                        // Execute both legs in parallel
                        let (short_result, long_result) = tokio::join!(
                            execution_gateway::submit_with_retry(&*short_gw, short_intent),
                            execution_gateway::submit_with_retry(&*long_gw, long_intent)
                        );
                        
                        match (&short_result, &long_result) {
                            (Ok(short_res), Ok(long_res)) => {
                                info!(
                                    "[execution] Funding Arb OPENED: {} SHORT={} @ {:.4} on {} | LONG={} @ {:.4} on {}",
                                    opp.symbol,
                                    short_res.filled_size, short_res.avg_fill_price, opp.short_exchange.name(),
                                    long_res.filled_size, long_res.avg_fill_price, opp.long_exchange.name()
                                );
                                
                                // Track the funding arb position
                                let now_ns = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos() as u64;
                                
                                funding_arb_positions.insert(opp.symbol.clone(), FundingArbPosition {
                                    symbol: opp.symbol.clone(),
                                    short_exchange: opp.short_exchange,
                                    long_exchange: opp.long_exchange,
                                    short_entry_price: short_res.avg_fill_price,
                                    long_entry_price: long_res.avg_fill_price,
                                    size: short_res.filled_size.min(long_res.filled_size),
                                    entry_timestamp_ns: now_ns,
                                    entry_net_rate: opp.net_rate,
                                });
                                
                                orders_submitted += 2;
                                dashboard_state.orders_submitted.store(orders_submitted, Ordering::Relaxed);
                            }
                            (Err(e), Ok(_)) => {
                                warn!("[execution] Funding arb SHORT failed: {} — unwinding LONG", e);
                                // Close the long position since short failed
                                let close_intent = execution_gateway::OrderIntent {
                                    symbol: opp.symbol.clone(),
                                    side: execution_gateway::OrderSide::Sell,
                                    size: position_size,
                                    order_type: execution_gateway::OrderType::Market,
                                    price: Some(mid_price),
                                    reduce_only: true,
                                    leverage: None,
                                    time_in_force: "ioc".to_string(),
                                    slippage_cap_pct: Some(0.005),
                                    placement: execution_state::PlacementType::AtBest,
                                    stop_loss: None,
                                    take_profit: None,
                                    confidence: 0.0,
                                    signal_tag: "funding_arb_unwind".to_string(),
                                };
                                let _ = long_gw.submit_order(close_intent).await;
                            }
                            (Ok(_), Err(e)) => {
                                warn!("[execution] Funding arb LONG failed: {} — unwinding SHORT", e);
                                // Close the short position since long failed
                                let close_intent = execution_gateway::OrderIntent {
                                    symbol: opp.symbol.clone(),
                                    side: execution_gateway::OrderSide::Buy,
                                    size: position_size,
                                    order_type: execution_gateway::OrderType::Market,
                                    price: Some(mid_price),
                                    reduce_only: true,
                                    leverage: None,
                                    time_in_force: "ioc".to_string(),
                                    slippage_cap_pct: Some(0.005),
                                    placement: execution_state::PlacementType::AtBest,
                                    stop_loss: None,
                                    take_profit: None,
                                    confidence: 0.0,
                                    signal_tag: "funding_arb_unwind".to_string(),
                                };
                                let _ = short_gw.submit_order(close_intent).await;
                            }
                            (Err(e1), Err(e2)) => {
                                warn!("[execution] Funding arb BOTH legs failed: {} / {}", e1, e2);
                            }
                        }
                    }
                    
                    // Check exit conditions for existing funding arb positions
                    let mut positions_to_close = Vec::new();
                    let now_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    
                    for (symbol, pos) in &funding_arb_positions {
                        let hours_open = (now_ns - pos.entry_timestamp_ns) as f64 / 3_600_000_000_000.0;
                        
                        // Check if spread collapsed
                        let current_opp = cross_funding_arb.check_symbol(symbol);
                        let spread_collapsed = current_opp
                            .map(|o| o.net_rate < pos.entry_net_rate / 2.0)
                            .unwrap_or(true);
                        
                        // Exit conditions: spread collapsed OR >24 hours
                        if spread_collapsed || hours_open > 24.0 {
                            positions_to_close.push(symbol.clone());
                            info!(
                                "[execution] Funding Arb EXIT: {} after {:.1}h (spread_collapsed={})",
                                symbol, hours_open, spread_collapsed
                            );
                        }
                    }
                    
                    // Close marked positions
                    for symbol in positions_to_close {
                        if let Some(pos) = funding_arb_positions.remove(&symbol) {
                            let short_gw = multi_gateways.get(&pos.short_exchange);
                            let long_gw = multi_gateways.get(&pos.long_exchange);
                            
                            if let (Some(sgw), Some(lgw)) = (short_gw, long_gw) {
                                let mid_price = shared_prices.get(0)
                                    .map(|p| f64::from_bits(p.load(Ordering::Relaxed)))
                                    .unwrap_or(50000.0);
                                
                                // Close both legs in parallel
                                let close_short = execution_gateway::OrderIntent {
                                    symbol: symbol.clone(),
                                    side: execution_gateway::OrderSide::Buy,
                                    size: pos.size,
                                    order_type: execution_gateway::OrderType::Market,
                                    price: Some(mid_price),
                                    reduce_only: true,
                                    leverage: None,
                                    time_in_force: "ioc".to_string(),
                                    slippage_cap_pct: Some(0.005),
                                    placement: execution_state::PlacementType::AtBest,
                                    stop_loss: None,
                                    take_profit: None,
                                    confidence: 0.0,
                                    signal_tag: "funding_arb_close".to_string(),
                                };
                                
                                let close_long = execution_gateway::OrderIntent {
                                    symbol: symbol.clone(),
                                    side: execution_gateway::OrderSide::Sell,
                                    size: pos.size,
                                    order_type: execution_gateway::OrderType::Market,
                                    price: Some(mid_price),
                                    reduce_only: true,
                                    leverage: None,
                                    time_in_force: "ioc".to_string(),
                                    slippage_cap_pct: Some(0.005),
                                    placement: execution_state::PlacementType::AtBest,
                                    stop_loss: None,
                                    take_profit: None,
                                    confidence: 0.0,
                                    signal_tag: "funding_arb_close".to_string(),
                                };
                                
                                let sgw = sgw.clone();
                                let lgw = lgw.clone();
                                tokio::spawn(async move {
                                    let _ = tokio::join!(
                                        sgw.submit_order(close_short),
                                        lgw.submit_order(close_long)
                                    );
                                });
                            }
                        }
                    }
                }
                
                // TASK 3: Cross-Exchange Market Making (every 500ms)
                if multi_exchange_enabled && last_mm_check.elapsed() > Duration::from_millis(500) {
                    last_mm_check = std::time::Instant::now();
                    
                    // Skip if paused or no global book
                    if !cross_mm.is_paused() {
                        if let Some(ref gbr) = global_book_registry {
                            // Process each symbol
                            for sym_id in gbr.all_symbol_ids() {
                                let symbol = registry.get_name(sym_id).to_string();
                                
                                if let Some(book_arc) = gbr.get(sym_id) {
                                    // Check if spread is wide enough for market making
                                    let book = book_arc.read();
                                    let spread_bps = book.global_spread_bps().unwrap_or(0);
                                    
                                    // Only market make if spread is profitable (>3 bps)
                                    if spread_bps >= 3 && !cross_mm.inventory_limit_reached(&symbol) {
                                        // Generate maker orders
                                        let tick_size = 0.1; // Default tick size, should come from symbol config
                                        let maker_orders = cross_mm.generate_maker_orders(&symbol, &book_arc, tick_size);
                                        
                                        for intent in maker_orders {
                                            // Submit to maker exchange (Gate.io by default)
                                            if let Some(gw) = multi_gateways.get(&multi_exchange::ExchangeId::GateIo) {
                                                match execution_gateway::submit_with_retry(&**gw, intent.clone()).await {
                                                    Ok(res) => {
                                                        if res.filled_size > 0 {
                                                            // Fill detected! Generate hedge order
                                                            info!(
                                                                "[cross-mm] Maker fill: {} {} @ {:.4} — hedging",
                                                                intent.symbol,
                                                                if intent.side == execution_gateway::OrderSide::Buy { "BUY" } else { "SELL" },
                                                                res.avg_fill_price
                                                            );
                                                            
                                                            // Create maker order tracking
                                                            let maker_order = multi_exchange::cross_exchange_mm::MakerOrder {
                                                                order_id: res.order_id.clone(),
                                                                symbol: intent.symbol.clone(),
                                                                exchange: multi_exchange::ExchangeId::GateIo,
                                                                side: intent.side.clone(),
                                                                price: res.avg_fill_price,
                                                                size: res.filled_size,
                                                                original_size: intent.size,
                                                                filled_size: res.filled_size,
                                                                created_ns: std::time::SystemTime::now()
                                                                    .duration_since(std::time::UNIX_EPOCH)
                                                                    .unwrap_or_default()
                                                                    .as_nanos() as u64,
                                                                last_checked_ns: 0,
                                                                status: multi_exchange::cross_exchange_mm::MakerOrderStatus::Filled,
                                                            };
                                                            
                                                            // Generate and submit hedge order to Binance
                                                            let hedge_intent = cross_mm.generate_hedge_order(&maker_order, res.filled_size);
                                                            if let Some(hedge_gw) = multi_gateways.get(&multi_exchange::ExchangeId::Binance) {
                                                                match hedge_gw.submit_order(hedge_intent).await {
                                                                    Ok(hedge_res) => {
                                                                        cross_mm.on_hedge_fill(
                                                                            &intent.symbol,
                                                                            intent.side.clone(),
                                                                            res.avg_fill_price,
                                                                            hedge_res.avg_fill_price,
                                                                            res.filled_size,
                                                                        );
                                                                        info!(
                                                                            "[cross-mm] Hedge complete: {} @ {:.4} — PnL=${:.4}",
                                                                            intent.symbol,
                                                                            hedge_res.avg_fill_price,
                                                                            cross_mm.total_pnl()
                                                                        );
                                                                    }
                                                                    Err(e) => {
                                                                        warn!("[cross-mm] Hedge failed: {} — closing maker position", e);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        orders_submitted += 1;
                                                    }
                                                    Err(e) => {
                                                        debug!("[cross-mm] Maker order failed: {}", e);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Push MM state to dashboard
                    dashboard_state.set_funding_arb_json(
                        serde_json::to_string(&cross_mm.to_json()).unwrap_or_default()
                    );
                }
                
                // TASK 4: Statistical Arbitrage (every 1s)
                if multi_exchange_enabled && last_stat_arb_check.elapsed() > Duration::from_secs(1) {
                    last_stat_arb_check = std::time::Instant::now();
                    
                    // Update spread history from global book
                    if let Some(ref gbr) = global_book_registry {
                        for sym_id in gbr.all_symbol_ids() {
                            let symbol = registry.get_name(sym_id).to_string();
                            
                            if let Some(book_arc) = gbr.get(sym_id) {
                                let book = book_arc.read();
                                let now_ns = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos() as u64;
                                
                                // Get mid prices from each exchange pair
                                for (i, ex_a) in multi_exchange::ExchangeId::all().iter().enumerate() {
                                    for ex_b in multi_exchange::ExchangeId::all().iter().skip(i + 1) {
                                        let snap_a = book.get_exchange_snapshot(*ex_a);
                                        let snap_b = book.get_exchange_snapshot(*ex_b);
                                        
                                        if let (Some(a), Some(b)) = (snap_a, snap_b) {
                                            if a.best_bid_fp > 0 && b.best_bid_fp > 0 {
                                                let mid_a = (a.best_bid_fp + a.best_ask_fp) as f64 / 2.0 / 1e8;
                                                let mid_b = (b.best_bid_fp + b.best_ask_fp) as f64 / 2.0 / 1e8;
                                                
                                                stat_arb_engine.on_price_update(
                                                    &symbol, *ex_a, mid_a, *ex_b, mid_b, now_ns
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Check for entry opportunities
                    let opportunities = stat_arb_engine.scan_all_opportunities();
                    for (symbol, long_ex, short_ex, spread, mean, std_dev) in opportunities.iter().take(1) {
                        // Only take one opportunity at a time
                        info!(
                            "[stat-arb] Entry opportunity: {} z={:.2} (long={}, short={})",
                            symbol,
                            if *std_dev > 0.0 { (*spread - *mean) / *std_dev } else { 0.0 },
                            long_ex.name(),
                            short_ex.name()
                        );
                        
                        // Calculate position size (2% of total equity)
                        let total_equity = margin_monitor.total_equity();
                        let position_notional = total_equity * 0.02;
                        let mid_price = shared_prices.get(0)
                            .map(|p| f64::from_bits(p.load(Ordering::Relaxed)))
                            .unwrap_or(50000.0);
                        let position_size = (position_notional / mid_price).max(1.0) as i64;
                        
                        // Get gateways
                        let long_gw = multi_gateways.get(long_ex);
                        let short_gw = multi_gateways.get(short_ex);
                        
                        if let (Some(lg), Some(sg)) = (long_gw, short_gw) {
                            // Build entry intents
                            let (long_intent, short_intent) = multi_exchange::stat_arb::build_stat_arb_entry_intents(
                                symbol, *long_ex, *short_ex, position_size, mid_price, mid_price
                            );
                            
                            // Execute both legs in parallel
                            let lg = lg.clone();
                            let sg = sg.clone();
                            let (long_res, short_res) = tokio::join!(
                                lg.submit_order(long_intent),
                                sg.submit_order(short_intent)
                            );
                            
                            match (&long_res, &short_res) {
                                (Ok(lr), Ok(sr)) => {
                                    let now_ns = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_nanos() as u64;
                                    
                                    stat_arb_engine.record_entry(
                                        symbol,
                                        *long_ex,
                                        *short_ex,
                                        lr.avg_fill_price,
                                        sr.avg_fill_price,
                                        lr.filled_size.min(sr.filled_size),
                                        *spread,
                                        *mean,
                                        *std_dev,
                                        now_ns,
                                    );
                                    
                                    info!(
                                        "[stat-arb] ENTRY: {} long@{} ({:.4}) short@{} ({:.4})",
                                        symbol, long_ex.name(), lr.avg_fill_price,
                                        short_ex.name(), sr.avg_fill_price
                                    );
                                    orders_submitted += 2;
                                }
                                (Err(e), Ok(_)) => {
                                    warn!("[stat-arb] Long leg failed: {} — unwinding short", e);
                                    // Close short position
                                    let unwind = execution_gateway::OrderIntent {
                                        symbol: symbol.clone(),
                                        side: execution_gateway::OrderSide::Buy,
                                        size: position_size,
                                        order_type: execution_gateway::OrderType::Market,
                                        price: Some(mid_price),
                                        reduce_only: true,
                                        leverage: None,
                                        time_in_force: "ioc".to_string(),
                                        slippage_cap_pct: Some(0.005),
                                        placement: execution_state::PlacementType::AtBest,
                                        stop_loss: None,
                                        take_profit: None,
                                        confidence: 0.0,
                                        signal_tag: "stat_arb_unwind".to_string(),
                                    };
                                    let _ = sg.submit_order(unwind).await;
                                }
                                (Ok(_), Err(e)) => {
                                    warn!("[stat-arb] Short leg failed: {} — unwinding long", e);
                                    let unwind = execution_gateway::OrderIntent {
                                        symbol: symbol.clone(),
                                        side: execution_gateway::OrderSide::Sell,
                                        size: position_size,
                                        order_type: execution_gateway::OrderType::Market,
                                        price: Some(mid_price),
                                        reduce_only: true,
                                        leverage: None,
                                        time_in_force: "ioc".to_string(),
                                        slippage_cap_pct: Some(0.005),
                                        placement: execution_state::PlacementType::AtBest,
                                        stop_loss: None,
                                        take_profit: None,
                                        confidence: 0.0,
                                        signal_tag: "stat_arb_unwind".to_string(),
                                    };
                                    let _ = lg.submit_order(unwind).await;
                                }
                                (Err(e1), Err(e2)) => {
                                    warn!("[stat-arb] Both legs failed: {} / {}", e1, e2);
                                }
                            }
                        }
                    }
                    
                    // Check exit conditions for active positions
                    let now_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    
                    let exits = stat_arb_engine.check_exits(now_ns);
                    for (pos, reason) in exits {
                        info!(
                            "[stat-arb] EXIT: {} reason={:?} hours_open={:.1}",
                            pos.symbol,
                            reason,
                            pos.hours_open(now_ns)
                        );
                        
                        // Get current prices
                        let mid_price = shared_prices.get(0)
                            .map(|p| f64::from_bits(p.load(Ordering::Relaxed)))
                            .unwrap_or(50000.0);
                        
                        // Build exit intents
                        let (close_long, close_short) = multi_exchange::stat_arb::build_stat_arb_exit_intents(
                            &pos, mid_price, mid_price
                        );
                        
                        // Close both legs
                        if let (Some(lg), Some(sg)) = (
                            multi_gateways.get(&pos.long_exchange),
                            multi_gateways.get(&pos.short_exchange)
                        ) {
                            let lg = lg.clone();
                            let sg = sg.clone();
                            let symbol = pos.symbol.clone();
                            tokio::spawn(async move {
                                let _ = tokio::join!(
                                    lg.submit_order(close_long),
                                    sg.submit_order(close_short)
                                );
                            });
                        }
                        
                        stat_arb_engine.remove_position(&pos.symbol);
                    }
                    
                    // Push stat arb state to dashboard
                    dashboard_state.set_funding_arb_json(
                        serde_json::to_string(&stat_arb_engine.to_json()).unwrap_or_default()
                    );
                }

                // Periodic health check + position slot reconciliation (every 30s)
                if last_health_check.elapsed() > Duration::from_secs(30) {
                    let cb_state = circuit_breaker.get_state();
                    let active_slots = position_slots.active_positions();
                    let max_slots = position_slots.max_slots();
                    info!(
                        "[execution] Health: submitted={}, rejected={}, total_pnl={:.4}, cb_halted={}, consecutive_losses={}, slots={}/{}",
                        orders_submitted, orders_rejected,
                        total_pnl_fp as f64 / 1e8,
                        cb_state.halted, cb_state.consecutive_losses,
                        active_slots, max_slots
                    );
                    // BUG 7 FIX: Update dashboard state with periodic metrics
                    dashboard_state.active_positions.store(active_slots as u64, Ordering::Relaxed);
                    dashboard_state.set_realized_pnl(total_pnl_fp as f64 / 1e8);
                    dashboard_state.circuit_breaker_state.store(
                        if cb_state.halted { 1 } else { 0 }, Ordering::Relaxed);

                    // FIX 7: Fetch and sync balance/equity from exchange to dashboard
                    if let Some(ref gw) = gateway {
                        match gw.get_balance().await {
                            Ok(balance) => {
                                dashboard_state.balance_fp.store((balance * 1e8) as i64, Ordering::Relaxed);
                                dashboard_state.equity_fp.store((balance * 1e8) as i64, Ordering::Relaxed);
                                // TASK 3: Also update multi-exchange balance for Gate.io (index 0)
                                dashboard_state.set_exchange_balance(0, balance);
                            }
                            Err(e) => {
                                // Only log once per 60s to reduce noise
                                // (this runs every health check cycle)
                                debug!("[execution] Balance sync failed: {}", e);
                            }
                        }
                    }
                    
                // TASK 5: Multi-exchange margin monitor + balance fetching
                // Fetch balances and positions from all exchanges, update margin monitor and dashboard
                if multi_exchange_enabled && !multi_gateways.is_empty() {
                    // Refresh margin monitor with all gateways
                    margin_monitor.refresh_all(&multi_gateways).await;
                    
                    // Update dashboard with per-exchange balances
                    for (exchange_id, gw) in &multi_gateways {
                        let idx = *exchange_id as usize;
                        match gw.get_balance().await {
                            Ok(balance) => {
                                dashboard_state.set_exchange_balance(idx, balance);
                                let margin_ratio = margin_monitor.get_health(*exchange_id)
                                    .map(|h| h.margin_ratio)
                                    .unwrap_or(1.0);
                                dashboard_state.set_exchange_margin_ratio(idx, margin_ratio);
                            }
                            Err(e) => {
                                debug!("[execution] {} balance fetch failed: {}", exchange_id.name(), e);
                            }
                        }
                    }
                    
                    // Check for margin imbalances and log warnings
                    let alerts = margin_monitor.check_imbalances();
                    for alert in &alerts {
                        warn!(
                            "[execution] Margin imbalance: {} at {:.1}% — recommend transferring ${:.0} from {}",
                            alert.critical_exchange.name(),
                            alert.margin_ratio * 100.0,
                            alert.recommended_transfer_usdt,
                            alert.source_exchange.name()
                        );
                    }
                    
                    // If any exchange is critical (margin < 15%), halt new trades on that exchange
                    for health in margin_monitor.all_health() {
                        if health.is_critical {
                            warn!(
                                "[execution] CRITICAL: {} margin at {:.1}% — halting new trades",
                                health.exchange.name(),
                                health.margin_ratio * 100.0
                            );
                        }
                    }
                    
                    // Push margin monitor JSON to dashboard
                    dashboard_state.set_funding_arb_json(
                        serde_json::to_string(&margin_monitor.to_json()).unwrap_or_default()
                    );
                    
                    // Fetch positions from all exchanges and push to dashboard
                    let mut all_positions = Vec::new();
                    for (exchange_id, gw) in &multi_gateways {
                        if let Ok(positions) = gw.get_positions().await {
                            for p in positions {
                                all_positions.push(serde_json::json!({
                                    "exchange": exchange_id.name(),
                                    "symbol": p.symbol,
                                    "size": p.size,
                                    "entry_price": p.entry_price,
                                    "unrealized_pnl": p.unrealized_pnl,
                                    "leverage": p.leverage,
                                    "side": p.side,
                                }));
                            }
                        }
                    }
                    dashboard_state.set_multi_exchange_positions_json(
                        serde_json::to_string(&all_positions).unwrap_or_else(|_| "[]".to_string())
                    );
                    
                    // Update global book JSON for dashboard every 5 health checks (~2.5 min)
                    // Actually push every check since it's every 30s
                    if let Some(ref gbr) = global_book_registry {
                        dashboard_state.set_global_book_json(
                            serde_json::to_string(&gbr.to_json()).unwrap_or_else(|_| "{}".to_string())
                        );
                    }
                    
                    info!(
                        "[execution] Multi-exchange health: total_balance=${:.2}, exchanges={}, positions={}",
                        margin_monitor.total_balance(),
                        multi_gateways.len(),
                        all_positions.len()
                    );
                }
                    // }

                    // ── TASK 5: Position Sync + Slot Reconciliation ──
                    // Query exchange for actual positions and update dashboard
                    if let Some(ref gw) = gateway {
                        match gw.get_positions().await {
                            Ok(positions) => {
                                // TASK 5a: Update dashboard with real position data
                                let positions_json = serde_json::to_string(&positions).unwrap_or_else(|_| "[]".to_string());
                                dashboard_state.set_positions_json(positions_json);
                                
                                // Slot reconciliation: release orphaned slots
                                let actual = positions.len() as u32;
                                if active_slots > 0 && actual < active_slots {
                                    let leaked = active_slots - actual;
                                    for _ in 0..leaked {
                                        position_slots.release();
                                    }
                                    warn!(
                                        "[execution] Slot reconciliation: released {} orphaned slots (exchange={}, slots={})",
                                        leaked, actual, active_slots
                                    );
                                }
                                
                                // TASK 5b: Multi-exchange positions sync
                                // When multi-exchange is enabled, aggregate positions from all exchanges
                                if dashboard_state.multi_exchange_enabled.load(Ordering::Relaxed) {
                                    let mut all_positions = Vec::new();
                                    
                                    // Add Gate.io positions (tagged with exchange name)
                                    for p in &positions {
                                        all_positions.push(serde_json::json!({
                                            "exchange": "gateio",
                                            "symbol": p.symbol,
                                            "size": p.size,
                                            "entry_price": p.entry_price,
                                            "unrealized_pnl": p.unrealized_pnl,
                                            "leverage": p.leverage,
                                            "side": p.side,
                                        }));
                                    }
                                    
                                    // Note: binance_gateway and bybit_gateway positions would be added here
                                    // when those gateways are passed to the execution router
                                    
                                    let json = serde_json::to_string(&all_positions).unwrap_or_else(|_| "[]".to_string());
                                    dashboard_state.set_multi_exchange_positions_json(json);
                                }
                            }
                            Err(e) => {
                                warn!("[execution] Position sync failed: {}", e);
                            }
                        }
                    }

                    last_health_check = std::time::Instant::now();
                }

                // Small yield to avoid burning CPU when idle
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Main — OS Thread Spawning with Core Affinity
// ---------------------------------------------------------------------------

fn main() {
    // ── MANDATE 2: Load .env credentials BEFORE anything else ──────────────
    // The .env file lives in ../crypto_trading_bot/.env relative to the
    // rust_engine binary, or at the repo root.  We try several candidate
    // paths and fall back silently if none exist (Docker sets env vars
    // directly, so .env is optional in production).
    load_dotenv();

    // Initialize tracing (non-hot-path, can use std logging)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("trading_engine=info".parse().unwrap()),
        )
        .init();

    info!("Trading engine v{} starting (Institutional Refactor)", env!("CARGO_PKG_VERSION"));

    // Report available system resources
    let cpu_count = core_affinity::get_core_ids()
        .map(|ids| ids.len())
        .unwrap_or(0);
    info!("System: {} CPU cores detected", cpu_count);

    // 1. Load configuration (env vars are now populated by dotenvy)
    let config = load_config();

    // Resolve thread topology: use auto-detection if THREAD_TOPOLOGY=auto or
    // if the configured topology exceeds available cores
    let topology = {
        let use_auto = std::env::var("THREAD_TOPOLOGY")
            .map(|v| v.eq_ignore_ascii_case("auto"))
            .unwrap_or(false);

        let topo = if use_auto {
            info!("THREAD_TOPOLOGY=auto — auto-detecting optimal layout for {} cores", cpu_count);
            ThreadTopology::auto_detect()
        } else {
            config.thread_topology.clone()
        };

        // Validate topology against available cores — fall back to auto if invalid
        match topo.validate() {
            Ok(()) => {
                info!("Thread topology validated: ws_gateio={}, book={}, strategy={}, exec={}, telemetry={}, micro={:?}",
                    topo.ws_gateio_core, topo.book_builder_core,
                    topo.strategy_core, topo.execution_core, topo.telemetry_core,
                    topo.microstructure_cores);
                topo
            }
            Err(e) => {
                warn!("Thread topology invalid: {} — falling back to auto-detect", e);
                let auto_topo = ThreadTopology::auto_detect();
                info!("Auto-detected topology: ws_gateio={}, book={}, strategy={}, exec={}, telemetry={}, micro={:?}",
                    auto_topo.ws_gateio_core, auto_topo.book_builder_core,
                    auto_topo.strategy_core, auto_topo.execution_core, auto_topo.telemetry_core,
                    auto_topo.microstructure_cores);
                auto_topo
            }
        }
    };

    info!("Config loaded: {} exchanges, {} symbols",
          config.exchanges.len(), config.symbols.len());

    // 2. Build symbol registry
    let registry = Arc::new(build_symbol_registry(&config));
    info!("Symbol registry: {} symbols", registry.len());

    // 3. Allocate flat orderbooks (pre-allocated, no further heap alloc)
    let flat_configs = build_flat_book_configs(&config, &registry);
    let mut books: Vec<FlatOrderBook> = Vec::new();
    for id in registry.all_ids() {
        let name = registry.get_name(id);
        let book_config = flat_configs.get(&id)
            .copied()
            .unwrap_or_default();
        books.push(FlatOrderBook::new(book_config, name));
        info!("Allocated FlatOrderBook for {} (ID: {})", name, id);
    }

    // 4. Allocate SPSC ring buffers (Box::leak for 'static lifetime)
    let ws_gateio_to_book: &'static SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY> =
        Box::leak(Box::new(SpscRingBuffer::new()));

    let book_to_strategy: &'static SpscRingBuffer<BookSnapshot, BOOK_TO_STRATEGY_CAPACITY> =
        Box::leak(Box::new(SpscRingBuffer::new()));
    let strategy_to_exec: &'static SpscRingBuffer<OrderCommand, STRATEGY_TO_EXEC_CAPACITY> =
        Box::leak(Box::new(SpscRingBuffer::new()));
    let ws_to_strategy_trades: &'static SpscRingBuffer<spsc::TradeEvent, WS_TO_STRATEGY_TRADES_CAPACITY> =
        Box::leak(Box::new(SpscRingBuffer::new()));

    info!("SPSC ring buffers allocated: ws_to_book={}KB, book_to_strategy={}KB, strategy_to_exec={}KB",
          std::mem::size_of::<SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>>() / 1024,
          std::mem::size_of::<SpscRingBuffer<BookSnapshot, BOOK_TO_STRATEGY_CAPACITY>>() / 1024,
          std::mem::size_of::<SpscRingBuffer<OrderCommand, STRATEGY_TO_EXEC_CAPACITY>>() / 1024);

    // 5. Load shared memory regime reader
    let regime_shm_path = config.shared_mem.regime_shm_path.clone();

    // 6. Build strategy engine
    // Task 12: Pass &config instead of config.strategy to load pair profiles
    let strategy = Arc::new(StrategyEngine::new(&config));

    // 7a. Initialize global circuit breaker FIRST (required by gateway)
    // Must be created before the gateway so the WS connection loop can trip
    // it on disconnect via TripReason::ConnectivityLost.
    let cb_config = circuit_breaker::CircuitBreakerConfig {
        max_consecutive_losses: config.risk.max_consecutive_losses.unwrap_or(5) as u32,
        max_daily_drawdown_pct: config.risk.max_daily_drawdown.unwrap_or(0.05),
        max_spread_bps: 500,
        spread_zscore_threshold: 5.0,
        max_orders_per_second: 50,
        cooldown_seconds: 0, // manual reset required for safety
        flatten_on_trip: false,
        max_single_loss_usdt_fp: (config.risk.max_single_loss_usdt.unwrap_or(500.0) * 1e8) as i64,
        max_total_exposure_usdt_fp: (config.risk.max_position_usdt.unwrap_or(10_000.0) * 1e8) as i64,
    };
    let circuit_breaker_arc = Arc::new(CircuitBreaker::new(cb_config));
    // Leak a clone so hot-path threads get a zero-cost &'static reference.
    // Both the Arc and the &'static point to the SAME heap allocation,
    // so trip() from the gateway and is_tripped() from strategy share state.
    let circuit_breaker: &'static CircuitBreaker = {
        let ptr = Arc::into_raw(circuit_breaker_arc.clone());
        unsafe { &*ptr } // SAFETY: intentional leak — never dropped
    };
    info!("🛡️ Circuit breaker initialized and armed");

    // 7b. Build execution gateway with shared circuit breaker Arc.
    //
    // CRITICAL FIX: GateIoGateway::new_with_circuit_breaker calls tokio::spawn() for
    // WebSocket and reconciliation tasks. These MUST run on a long-lived runtime —
    // a temporary runtime would kill the spawned tasks when it drops.
    //
    // We create a PERSISTENT multi-threaded runtime that lives for the entire engine
    // lifetime. This runtime hosts the WS connection loop, reconciliation thread,
    // and all REST API calls from the execution thread.
    let gateway_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("gw-runtime")
        .enable_all()
        .build()
        .expect("Failed to build gateway Tokio runtime");
    let gateway_runtime = Box::leak(Box::new(gateway_runtime)); // intentional leak — lives forever

    let gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>> =
        config.exchanges.iter().find(|e| e.name == "gateio")
            .and_then(|cfg| {
                let ak = cfg.api_key.as_deref().unwrap_or("");
                let sk = cfg.secret_key.as_deref().unwrap_or("");
                if !is_valid_key(ak) || !is_valid_key(sk) {
                    info!("{}: no valid API credentials — signal-only mode", cfg.name);
                    return None;
                }
                info!("{}: live gateway initialising (with circuit breaker, persistent runtime)", cfg.name);

                // Initialize the gateway inside the PERSISTENT runtime.
                // The spawned WS/reconciliation tasks will keep running after block_on returns.
                let gateway = gateway_runtime.block_on(async {
                    GateIoGateway::new_with_circuit_breaker(
                        ak.to_string(), sk.to_string(), cfg.testnet,
                        Some(circuit_breaker_arc.clone()),
                    )
                });

                // Run comprehensive auth diagnostic at startup.
                // This prints detailed info to stderr so the user can immediately see
                // if their keys are valid, if the endpoint is reachable, etc.
                let gw_ref = &gateway;
                gateway_runtime.block_on(async {
                    match gw_ref.test_auth_diagnostic().await {
                        Ok(balance) => {
                            info!("[startup] ✅ Gate.io {} auth verified — balance: ${:.2}",
                                  if cfg.testnet { "testnet" } else { "live" }, balance);
                        }
                        Err(e) => {
                            error!("[startup] ❌ Gate.io auth diagnostic FAILED: {}", e);
                            error!("[startup] The engine will continue in signal-only mode until auth is fixed.");
                        }
                    }
                });

                Some(Arc::new(gateway) as Arc<dyn ExecutionGateway + Send + Sync>)
            });

    // 7c. Initialize Order Lifecycle Tracker (Institutional Feature)
    let lifecycle_tracker: &'static OrderLifecycleTracker =
        Box::leak(Box::new(OrderLifecycleTracker::new(5000)));
    info!("📋 Order Lifecycle Tracker initialized (capacity=5000)");

    // 7d. Initialize Pipeline Latency Tracker (Institutional Feature)
    let latency_tracker: &'static PipelineLatencyTracker =
        Box::leak(Box::new(PipelineLatencyTracker::new()));
    info!("⏱️ Pipeline Latency Tracker initialized (6 histograms)");

    // 7e. Initialize Market Impact Model (Institutional Feature)
    let impact_model: &'static MarketImpactModel = {
        let num_symbols = config.symbols.len().max(16);
        let mut model = MarketImpactModel::new(num_symbols);
        // Set default params for known symbol types
        for (idx, sym) in config.symbols.iter().enumerate() {
            if sym.contains("BTC") || sym.contains("ETH") {
                model.set_symbol_params(idx, ImpactParams::high_liquidity());
            } else if config::is_forex_symbol(sym) {
                model.set_symbol_params(idx, ImpactParams::forex());
            } else {
                model.set_symbol_params(idx, ImpactParams::medium_liquidity());
            }
        }
        Box::leak(Box::new(model))
    };
    info!("📊 Market Impact Model initialized ({} symbols calibrated)", config.symbols.len());

    // 7f. Initialize Pre-Trade Risk Engine (Institutional Feature)
    // Shared across strategy thread (check) and execution thread (update_balance,
    // on_position_opened, on_position_closed). All fields are AtomicI64/AtomicU32
    // except per_symbol_margin which uses parking_lot::RwLock.
    let pre_trade_risk_engine: &'static PreTradeRiskEngine =
        Box::leak(Box::new(PreTradeRiskEngine::with_defaults()));
    info!("🛡️ PreTradeRiskEngine initialized (max_leverage=125, max_positions=3)");

    // 7g. Initialize Position Slot Manager — hard limit of 3 concurrent positions.
    // Lock-free semaphore using AtomicU32. Strategy thread acquires slots before
    // pushing OrderCommand; execution thread releases on position close.
    // FIX 12: Use cfg.risk.max_open_positions instead of hardcoded 3
    let position_slot_manager: &'static PositionSlotManager =
        Box::leak(Box::new(PositionSlotManager::new(config.risk.max_open_positions as u32)));
    info!("📍 PositionSlotManager initialized (max_slots={}, lock-free AtomicU32)", config.risk.max_open_positions);

    // 7g2. Initialize Correlation Limiter (FEATURE 6)
    let correlation_limiter: &'static CorrelationLimiter =
        Box::leak(Box::new(CorrelationLimiter::default()));
    info!("🔗 CorrelationLimiter initialized (BTC-ETH=0.85, BTC-SOL=0.75, ETH-SOL=0.70, max_correlated=150%)");

    // 7h. Initialize Rate Limiter Pool (Directive 5)
    let is_testnet = config.exchanges.iter().any(|e| e.name == "gateio" && e.testnet);
    let _rate_limiter: &'static RateLimiterPool = Box::leak(Box::new(RateLimiterPool::new(is_testnet)));

    // 7i. Initialize Dashboard State (shared between hot-path and HTTP server)
    let dashboard_state = Arc::new(DashboardState::new());

    // 7i2. Initialize funding rates storage (shared between strategy and execution)
    let funding_rates: Arc<parking_lot::RwLock<HashMap<String, f64>>> = Arc::new(parking_lot::RwLock::new(HashMap::new()));

    // 7j. Initialize Position Sizer (Directive 2) — fetch contract specs at startup
    let position_sizer: &'static PositionSizer = {
        let mut sizer = PositionSizer::new();
        // Try to fetch contract specs from Gate.io REST API
        if let Some(ref gw_cfg) = config.exchanges.iter().find(|e| e.name == "gateio") {
            let ak = gw_cfg.api_key.as_deref().unwrap_or("");
            let sk = gw_cfg.secret_key.as_deref().unwrap_or("");
            if is_valid_key(ak) && is_valid_key(sk) {
                // Contract specs are PUBLIC data — always fetch from mainnet.
                // Gate.io testnet infrastructure is unreliable and returns HTTP 502
                // for public endpoints like /futures/usdt/contracts.
                let base_url = "https://api.gateio.ws/api/v4";
                let client = reqwest::Client::new();
                // Reuse the gateway runtime instead of creating a temporary one
                let specs = gateway_runtime.block_on(position_sizer::fetch_contract_specs(
                    &client, base_url, ak, sk.as_bytes(), &config.symbols,
                ));
                for spec in specs {
                    sizer.register_spec(spec);
                }
            }
        }
        Box::leak(Box::new(sizer))
    };
    info!("📏 PositionSizer initialized with contract specs");

    // 7k. Initialize Bridge IPC subsystem (Phase 4: Rust ↔ Python)
    // Tick broadcast: Rust → Python (normalized tick data via SHM ring buffer)
    let mut tick_broadcaster = bridge_ipc::tick_broadcast::TickBroadcaster::with_defaults();
    match tick_broadcaster.init() {
        Ok(()) => info!("📡 Bridge: Tick broadcaster initialized"),
        Err(e) => warn!("📡 Bridge: Tick broadcaster init failed (non-fatal): {}", e),
    }
    let _tick_broadcaster: &'static parking_lot::Mutex<bridge_ipc::tick_broadcast::TickBroadcaster> =
        Box::leak(Box::new(parking_lot::Mutex::new(tick_broadcaster)));

    // Portfolio receiver: Python → Rust (portfolio weight targets via SHM)
    let mut portfolio_rx = bridge_ipc::portfolio_receiver::PortfolioReceiver::with_defaults();
    match portfolio_rx.init() {
        Ok(()) => info!("📡 Bridge: Portfolio receiver initialized"),
        Err(e) => warn!("📡 Bridge: Portfolio receiver init failed (non-fatal): {}", e),
    }
    let _portfolio_rx: &'static parking_lot::Mutex<bridge_ipc::portfolio_receiver::PortfolioReceiver> =
        Box::leak(Box::new(parking_lot::Mutex::new(portfolio_rx)));

    // Execution confirmation broadcast: Rust → Python (fill/cancel notifications)
    let mut exec_broadcaster = bridge_ipc::exec_confirm_broadcast::ExecConfirmBroadcaster::with_defaults();
    match exec_broadcaster.init() {
        Ok(()) => info!("📡 Bridge: Execution confirmation broadcaster initialized"),
        Err(e) => warn!("📡 Bridge: Exec confirmation init failed (non-fatal): {}", e),
    }
    let _exec_broadcaster: &'static parking_lot::Mutex<bridge_ipc::exec_confirm_broadcast::ExecConfirmBroadcaster> =
        Box::leak(Box::new(parking_lot::Mutex::new(exec_broadcaster)));

    // Regime adapter: wraps regime_shm with typed interface
    let _regime_adapter: &'static parking_lot::Mutex<bridge_ipc::regime_adapter::RegimeAdapter> =
        Box::leak(Box::new(parking_lot::Mutex::new(
            bridge_ipc::regime_adapter::RegimeAdapter::with_defaults()
        )));

    // Signal adapter: wraps signal_queue with validation
    let _signal_adapter: &'static parking_lot::Mutex<bridge_ipc::signal_adapter::SignalAdapter> =
        Box::leak(Box::new(parking_lot::Mutex::new(
            bridge_ipc::signal_adapter::SignalAdapter::with_defaults()
        )));

    // Event bus: Internal MPMC event distribution (flume-based)
    let event_buses: &'static event_bus::EngineEventBuses =
        Box::leak(Box::new(event_bus::EngineEventBuses::new()));
    info!("🔌 Event bus initialized: market_data={}, execution={}, control={}",
        event_buses.market_data.capacity(),
        event_buses.execution.capacity(),
        event_buses.control.capacity());

    // Bridge health monitor: tracks IPC health metrics and exposes /health endpoint
    let mut health_monitor = bridge_ipc::health_monitor::BridgeHealthMonitor::new();
    // BridgeHealthMonitor is ready after new() — no separate init needed
    info!("📡 Bridge: Health monitor initialized");
    let health_monitor: &'static parking_lot::Mutex<bridge_ipc::health_monitor::BridgeHealthMonitor> =
        Box::leak(Box::new(parking_lot::Mutex::new(health_monitor)));

    info!("📡 Bridge IPC subsystem initialized (tick_broadcast + portfolio_rx + exec_confirm + regime + signal + event_bus + health_monitor)");

    // 7m. Initialize Execution Analytics (Phase 2 Feature 7)
    let exec_analytics_arc = Arc::new(parking_lot::Mutex::new(ExecutionAnalytics::new(1000)));
    let leaked_arc = exec_analytics_arc.clone();
    let exec_analytics: &'static parking_lot::Mutex<ExecutionAnalytics> =
        unsafe { &*Arc::into_raw(leaked_arc) };
    info!("📊 Execution Analytics initialized (slippage + shortfall + impact tracking, window=1000)");

    // Shared prices array for execution thread
    let num_symbols = registry.len();
    let mut prices_vec = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        prices_vec.push(AtomicU64::new(0));
    }
    let shared_prices = Arc::new(prices_vec);

    // Initialize ML Weight Reader
    let ml_weights_path = std::env::var("ML_WEIGHT_SHM_PATH").unwrap_or_else(|_| "/dev/shm/ml_weights".to_string());
    let ml_weights: &'static ml_weight_receiver::MlWeightReader = Box::leak(Box::new(ml_weight_receiver::MlWeightReader::new(&ml_weights_path)));

    // 8. Spawn OS threads with core affinity pinning
    let mut thread_handles = Vec::new();

    // ── Core 2: WS Ingestion (Gate.io) ──
    if let Some(gateio_cfg) = config.exchanges.iter().find(|e| e.name == "gateio").cloned() {
        let ws_ring = ws_gateio_to_book;
        let trades_ring = ws_to_strategy_trades;
        let reg = registry.clone();
        let core_id = topology.ws_gateio_core;
        let handle = thread::Builder::new()
            .name("ws-gateio".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[ws-gateio] Pinned to core {}", core_id);
                ws_ingestion_loop_gateio(ws_ring, trades_ring, gateio_cfg, reg);
            })
            .expect("Failed to spawn WS Gate.io thread");
        thread_handles.push(handle);
    }

    // ── Multi-Exchange Initialization (gated by USE_MULTI_EXCHANGE) ──────────────
    let global_book_registry: Option<Arc<multi_exchange::GlobalBookRegistry>> = if config.multi_exchange_enabled {
        info!("Multi-Exchange mode ENABLED - initializing Binance + Bybit");
        
        // Build global book registry
        let registry_me = Arc::new(multi_exchange::GlobalBookRegistry::new());
        
        // Build Binance gateway
        let binance_gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>> = {
            let ak = config.multi_exchange.binance_api_key.as_deref().unwrap_or("");
            let sk = config.multi_exchange.binance_secret_key.as_deref().unwrap_or("");
            if !ak.is_empty() && !sk.is_empty() && ak.len() >= 8 {
                info!("Binance: live execution gateway initialised (testnet={})", config.multi_exchange.binance_testnet);
                Some(Arc::new(binance_gateway::BinanceGateway::new(
                    ak.to_string(), sk.to_string(),
                    config.multi_exchange.binance_testnet,
                )) as Arc<dyn ExecutionGateway + Send + Sync>)
            } else {
                info!("Binance: no valid credentials - signal-only mode");
                None
            }
        };
        
        // Build Bybit gateway
        let bybit_gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>> = {
            let ak = config.multi_exchange.bybit_api_key.as_deref().unwrap_or("");
            let sk = config.multi_exchange.bybit_secret_key.as_deref().unwrap_or("");
            if !ak.is_empty() && !sk.is_empty() && ak.len() >= 8 {
                info!("Bybit: live execution gateway initialised (testnet={})", config.multi_exchange.bybit_testnet);
                Some(Arc::new(bybit_gateway::BybitGateway::new(
                    ak.to_string(), sk.to_string(),
                    config.multi_exchange.bybit_testnet,
                )) as Arc<dyn ExecutionGateway + Send + Sync>)
            } else {
                info!("Bybit: no valid credentials - signal-only mode");
                None
            }
        };
        
        // Store gateways for later use (e.g., margin monitor)
        let mut multi_gateways: HashMap<multi_exchange::ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>> = HashMap::new();
        if let Some(ref gw) = gateway {
            multi_gateways.insert(multi_exchange::ExchangeId::GateIo, gw.clone());
        }
        if let Some(gw) = binance_gateway {
            multi_gateways.insert(multi_exchange::ExchangeId::Binance, gw);
        }
        if let Some(gw) = bybit_gateway {
            multi_gateways.insert(multi_exchange::ExchangeId::Bybit, gw);
        }
        
        info!("Multi-exchange gateways initialized: {} connected", multi_gateways.len());
        
        // Spawn Binance WS ingestion thread
        {
            let reg = registry_me.clone();
            let sym_reg = registry.clone();
            let binance_ws_url = if config.multi_exchange.binance_testnet {
                "wss://stream.binancefuture.com/stream"
            } else {
                "wss://fstream.binance.com/stream"
            };
            eprintln!("[config] Binance WS URL: {}", binance_ws_url);
            let binance_cfg = crate::config::ExchangeConfig {
                name: "binance".to_string(),
                symbols: config.symbols.clone(),
                ws_url: binance_ws_url.to_string(),
                api_key: config.multi_exchange.binance_api_key.clone(),
                secret_key: config.multi_exchange.binance_secret_key.clone(),
                passphrase: None,
                testnet: config.multi_exchange.binance_testnet,
                rest_url: None,
                max_leverage: 125,
                enabled: true,
            };
            let core_id = topology.ws_binance_core;
            let handle = thread::Builder::new()
                .name("ws-binance".into())
                .spawn(move || {
                    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                    info!("[ws-binance] Pinned to core {}", core_id);
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to build tokio runtime for ws-binance");
                    rt.block_on(multi_exchange::ws_ingestion_multi::run_binance_ws_ingestion(
                        binance_cfg, reg, sym_reg,
                    ));
                })
                .expect("Failed to spawn Binance WS thread");
            thread_handles.push(handle);
        }
        
        // Spawn Bybit WS ingestion thread
        {
            let reg = registry_me.clone();
            let sym_reg = registry.clone();
            let bybit_ws_url = if config.multi_exchange.bybit_testnet {
                "wss://stream-testnet.bybit.com/v5/public/linear"
            } else {
                "wss://stream.bybit.com/v5/public/linear"
            };
            eprintln!("[config] Bybit WS URL: {}", bybit_ws_url);
            let bybit_cfg = crate::config::ExchangeConfig {
                name: "bybit".to_string(),
                symbols: config.symbols.clone(),
                ws_url: bybit_ws_url.to_string(),
                api_key: config.multi_exchange.bybit_api_key.clone(),
                secret_key: config.multi_exchange.bybit_secret_key.clone(),
                passphrase: None,
                testnet: config.multi_exchange.bybit_testnet,
                rest_url: None,
                max_leverage: 100,
                enabled: true,
            };
            let core_id = topology.ws_bybit_core;
            let handle = thread::Builder::new()
                .name("ws-bybit".into())
                .spawn(move || {
                    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                    info!("[ws-bybit] Pinned to core {}", core_id);
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to build tokio runtime for ws-bybit");
                    rt.block_on(multi_exchange::ws_ingestion_multi::run_bybit_ws_ingestion(
                        bybit_cfg, reg, sym_reg,
                    ));
                })
                .expect("Failed to spawn Bybit WS thread");
            thread_handles.push(handle);
        }
        
        // ── Spawn Funding Rate Arbitrage Engine ──────────────────────────────
        // Runs as a tokio task inside the gateway runtime. Monitors funding rates
        // across exchanges, validates opportunities, and executes delta-neutral
        // arb positions with full lifecycle management.
        //
        // FIX: Reuse the shared multi_gateways map (Arc clones) instead of
        // creating duplicate BinanceGateway/BybitGateway instances that would
        // open separate WebSocket connections and risk exchange rate limiting.
        //
        // FIX: Share a single CrossVenueMarginMonitor instance so the engine
        // sees the same margin health data as the rest of the system.
        {
            let fab_gbr = registry_me.clone();
            let fab_symbols = config.symbols.clone();
            let fab_gateio_testnet = config.exchanges.iter()
                .find(|e| e.name == "gateio")
                .map(|e| e.testnet)
                .unwrap_or(false);
            let fab_binance_testnet = config.multi_exchange.binance_testnet;
            let fab_bybit_testnet = config.multi_exchange.bybit_testnet;

            // Share the existing gateway instances instead of creating duplicates.
            // multi_gateways was built above at initialization — clone the Arc
            // references so all subsystems share the same connections.
            let fab_gateways = multi_gateways.clone();

            // Shared margin monitor — same instance used by execution router
            let fab_margin_monitor = Arc::new(
                parking_lot::RwLock::new(
                    multi_exchange::margin_monitor::CrossVenueMarginMonitor::with_defaults()
                )
            );

            // Dashboard state for operator visibility
            let fab_dashboard = dashboard_state.clone();

            // Shutdown signal — set by Ctrl+C handler to gracefully stop the engine
            let fab_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let fab_shutdown_clone = fab_shutdown.clone();

            info!("[funding-arb] Spawning engine with {} gateways, {} symbols",
                  fab_gateways.len(), fab_symbols.len());

            let handle = thread::Builder::new()
                .name("funding-arb".into())
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to build tokio runtime for funding-arb");

                    rt.block_on(async {
                        let mut engine = multi_exchange::FundingArbEngine::new(
                            multi_exchange::FundingArbEngineConfig::default(),
                            fab_shutdown_clone,
                        );

                        engine.run(
                            fab_gateways,
                            fab_gbr,
                            fab_margin_monitor,
                            Some(fab_dashboard),
                            fab_symbols,
                            fab_gateio_testnet,
                            fab_binance_testnet,
                            fab_bybit_testnet,
                        ).await;
                    });
                })
                .expect("Failed to spawn funding arb engine thread");
            thread_handles.push(handle);
        }

        Some(registry_me)
    } else {
        info!("Multi-Exchange mode DISABLED - Gate.io only");
        None
    };
    
    // Store multi-exchange enabled flag in dashboard state
    dashboard_state.set_multi_exchange_enabled(config.multi_exchange_enabled);

    // ── Core 4: Orderbook Builder ──
    {
        let ws_ring_g = ws_gateio_to_book;
        let strat_ring = book_to_strategy;
        let reg = registry.clone();
        let core_id = topology.book_builder_core;
        let sp = shared_prices.clone();
        // TASK 8: Pass global_book_registry to orderbook_builder_loop
        let gbr = global_book_registry.clone();
        let handle = thread::Builder::new()
            .name("book-builder".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[book-builder] Pinned to core {}", core_id);
                orderbook_builder_loop(ws_ring_g, strat_ring, &mut books, reg, sp, gbr);
            })
            .expect("Failed to spawn book builder thread");
        thread_handles.push(handle);
    }

    // ── Core 4: Strategy Evaluator + Market Impact ──
    // Allocate the SPSC overflow monitor as 'static (lives for program duration).
    // When the SPSC ring is full and drops exceed 10/sec, this trips the circuit
    // breaker with OrderRateAnomaly to back-pressure the strategy engine.
    let spsc_overflow_monitor: &'static SpscOverflowMonitor =
        Box::leak(Box::new(SpscOverflowMonitor::new(10))); // 10 drops/sec threshold

    // Manual trade channels: dashboard -> execution (command), execution -> strategy (position tracking)
    let (manual_cmd_tx, manual_cmd_rx) = crossbeam_channel::bounded::<dashboard_server::ManualTradeRequest>(32);
    let manual_cmd_tx_arc = Arc::new(manual_cmd_tx);
    // TODO: When the execution loop processes a manual trade fill, it sends the fill details
    // here so the strategy thread can register the position in exit_evaluator / lifecycle_mgr.
    // For now the sender is unused because the full gateway-call path is not yet wired.
    let (_manual_pos_tx, manual_pos_rx) = crossbeam_channel::bounded::<dashboard_server::ManualPositionTrack>(32);

    {
        let book_ring = book_to_strategy;
        let exec_ring = strategy_to_exec;
        let trades_ring = ws_to_strategy_trades;
        let strat = strategy.clone();
        let reg = registry.clone();
        let core_id = topology.strategy_core;
        let lat = latency_tracker;
        let impact = impact_model;
        let cb_ref = circuit_breaker; // &'static CircuitBreaker
        let funding_rates_strat = funding_rates.clone();
        let exec_analytics_strat = exec_analytics;
        let ml_weights_strat = ml_weights;
        let handle = thread::Builder::new()
            .name("strategy".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[strategy] Pinned to core {} (with Market Impact + Overflow Monitor)", core_id);
                let regime_reader = regime_shm::SharedMemRegimeReader::new(&regime_shm_path);
                strategy_evaluator_loop(
                    book_ring,
                    exec_ring,
                    trades_ring,
                    &regime_reader,
                    strat,
                    reg,
                    lat,
                    impact,
                    spsc_overflow_monitor,
                    Some(cb_ref),
                    pre_trade_risk_engine,
                    position_slot_manager,
                    correlation_limiter,
                    position_sizer,
                    funding_rates_strat,
                    exec_analytics_strat,
                    ml_weights_strat,
                    manual_pos_rx,
                );
            })
            .expect("Failed to spawn strategy thread");
        thread_handles.push(handle);
    }

    // ── Core 6: Execution Router (with Circuit Breaker & SL/TP) ──
    // Mandate 3: Build forex gateway if credentials are available
    let funding_rates_exec = funding_rates.clone();
    let exec_analytics_exec = exec_analytics;

    let forex_gw: Option<Arc<dyn ExecutionGateway + Send + Sync>> = {
        let login = config.forex_login.as_deref().unwrap_or("").trim();
        let password = config.forex_password.as_deref().unwrap_or("").trim();
        let server = config.forex_server.as_deref().unwrap_or("").trim();
        // FIX 8: Reject placeholder/sentinel forex credentials
        let login_valid = !login.is_empty() && !login.starts_with("your_") && login != "PLACEHOLDER";
        let password_valid = !password.is_empty() && !password.starts_with("your_") && password != "PLACEHOLDER";
        if login_valid && password_valid {
            info!("[main] Forex gateway enabled: login={}, server={}", login, server);
            Some(Arc::new(forex_gateway::ForexGateway::new(
                login.to_string(),
                password.to_string(),
                server.to_string(),
                None, // Use default MT5 bridge URL
                false,
            )))
        } else {
            info!("[main] Forex gateway disabled — no credentials");
            None
        }
    };
    
    // TASK 1: Build multi-exchange gateways map for execution router
    let exec_multi_gateways: HashMap<multi_exchange::ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>> = 
        if config.multi_exchange_enabled {
            let mut gateways_map = HashMap::new();
            if let Some(ref gw) = gateway {
                gateways_map.insert(multi_exchange::ExchangeId::GateIo, gw.clone());
            }
            // Build Binance gateway for execution
            {
                let ak = config.multi_exchange.binance_api_key.as_deref().unwrap_or("");
                let sk = config.multi_exchange.binance_secret_key.as_deref().unwrap_or("");
                if !ak.is_empty() && !sk.is_empty() && ak.len() >= 8 {
                    let binance_gw = Arc::new(binance_gateway::BinanceGateway::new(
                        ak.to_string(), sk.to_string(),
                        config.multi_exchange.binance_testnet,
                    )) as Arc<dyn ExecutionGateway + Send + Sync>;
                    gateways_map.insert(multi_exchange::ExchangeId::Binance, binance_gw);
                }
            }
            // Build Bybit gateway for execution
            {
                let ak = config.multi_exchange.bybit_api_key.as_deref().unwrap_or("");
                let sk = config.multi_exchange.bybit_secret_key.as_deref().unwrap_or("");
                if !ak.is_empty() && !sk.is_empty() && ak.len() >= 8 {
                    let bybit_gw = Arc::new(bybit_gateway::BybitGateway::new(
                        ak.to_string(), sk.to_string(),
                        config.multi_exchange.bybit_testnet,
                    )) as Arc<dyn ExecutionGateway + Send + Sync>;
                    gateways_map.insert(multi_exchange::ExchangeId::Bybit, bybit_gw);
                }
            }
            info!("[main] Multi-exchange execution gateways: {} connected", gateways_map.len());
            gateways_map
        } else {
            HashMap::new()
        };
    let multi_exchange_enabled_exec = config.multi_exchange_enabled;
    
    {
        let exec_ring = strategy_to_exec;
        let gw = gateway.clone();
        let fx_gw = forex_gw;
        let cb = circuit_breaker;
        let reg = registry.clone();
        let core_id = topology.execution_core;
        let lc = lifecycle_tracker;
        let lat = latency_tracker;
        let dash_exec = dashboard_state.clone();
        let sp = shared_prices.clone();
        // TASK 1: Pass global book registry and multi-gateways to execution router
        let gbr_exec = global_book_registry.clone();
        let handle = thread::Builder::new()
            .name("execution".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[execution] Pinned to core {} (with Lifecycle + Latency + Multi-Exchange SOR)", core_id);
                execution_router_loop(
                    exec_ring, manual_cmd_rx, gw, fx_gw, cb, reg, lc, lat, 
                    position_slot_manager, dash_exec, funding_rates_exec, exec_analytics_exec, sp,
                    gbr_exec, exec_multi_gateways, multi_exchange_enabled_exec,
                );
            })
            .expect("Failed to spawn execution thread");
        thread_handles.push(handle);
    }

    // ── Dashboard HTTP Server (runs on its own thread) ──
    {
        let dash_state = dashboard_state.clone();
        let exec_analytics_dash = exec_analytics_arc.clone();
        let bind_addr = std::env::var("DASHBOARD_BIND")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        // Set JOURNAL_DIR environment variable for Python dashboard to read
        let journal_dir_env = std::env::var("JOURNAL_DIR")
            .unwrap_or_else(|_| journal::JOURNAL_DIR.to_string());
        // SAFETY: Called before any threads read JOURNAL_DIR. The dashboard thread
        // hasn't started yet at this point in main().
        unsafe { std::env::set_var("JOURNAL_DIR", &journal_dir_env); }
        let manual_cmd_tx_dash = manual_cmd_tx_arc.clone();
        let symbol_registry_dash = registry.clone();
        let handle = thread::Builder::new()
            .name("dashboard-http".into())
            .spawn(move || {
                info!("[dashboard] Starting HTTP server on {}", bind_addr);
                dashboard_server::run_dashboard_server(&bind_addr, dash_state, exec_analytics_dash, manual_cmd_tx_dash, symbol_registry_dash);
            })
            .expect("Failed to spawn dashboard HTTP thread");
        thread_handles.push(handle);
    }

    // ── Core 7: Telemetry/Journaling (Issue 2 — Journal + SharedState) ──
    {
        let core_id = topology.telemetry_core;
        let journal_dir = std::env::var("JOURNAL_DIR")
            .unwrap_or_else(|_| journal::JOURNAL_DIR.to_string());
        let state_shm_path = std::env::var("STATE_SHM_PATH")
            .unwrap_or_else(|_| shared_state::STATE_SHM_PATH.to_string());
        let dash_telem = dashboard_state.clone();
        let health_mon = health_monitor;
        let handle = thread::Builder::new()
            .name("telemetry".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[telemetry] Pinned to core {} — Journal + SharedState + Health Monitor", core_id);

                // Create telemetry publisher with journal + shared state writers
                let mut telemetry = TelemetryPublisher::new(journal_dir, state_shm_path);
                if let Err(e) = telemetry.init() {
                    warn!("[telemetry] Init failed: {} — will retry lazily", e);
                }

                // Heartbeat loop: write heartbeat to journal + shared state every 500ms
                let interval = Duration::from_millis(500);
                let mut report_counter: u64 = 0;
                loop {
                    telemetry.publish_heartbeat();
                    // Write heartbeat file for Docker HEALTHCHECK
                    let ts = telemetry::now_micros();
                    let _ = std::fs::write("/tmp/engine_heartbeat", ts.to_string());

                    // BUG 7 FIX: Update dashboard uptime every heartbeat
                    let start = dash_telem.start_time.load(Ordering::Relaxed);
                    let now_epoch = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    dash_telem.uptime_secs.store(now_epoch.saturating_sub(start), Ordering::Relaxed);

                    // Every 60 heartbeats (~30s), log latency + lifecycle metrics
                    report_counter += 1;
                    if report_counter % 60 == 0 {
                        latency_tracker.log_report();
                        let lc_metrics = lifecycle_tracker.get_metrics();
                        info!("[telemetry] Lifecycle: orders={}, fills={}, rejections={}, active={}",
                            lc_metrics.total_orders, lc_metrics.total_fills,
                            lc_metrics.total_rejections, lc_metrics.active_orders);
                        // BUG 7 FIX: Sync lifecycle metrics to dashboard
                        dash_telem.total_fills.store(lc_metrics.total_fills, Ordering::Relaxed);
                        dash_telem.orders_submitted.store(lc_metrics.total_orders, Ordering::Relaxed);
                    }

                    // Every 120 heartbeats (~60s), log bridge health metrics
                    if report_counter % 120 == 0 {
                        let mut hm = health_mon.lock();
                        hm.update_rates();
                        let health = hm.get_status();
                        let metrics_json = hm.get_metrics_json();
                        info!("[telemetry] Bridge Health: status={}, metrics={}",
                            health, metrics_json);
                    }

                    thread::sleep(interval);
                }
            })
            .expect("Failed to spawn telemetry thread");
        thread_handles.push(handle);
    }

    info!("All {} threads spawned. Engine running. Press Ctrl+C to stop.",
          thread_handles.len());

    // 9. Block main thread on shutdown signal
    let (tx, rx) = std::sync::mpsc::channel();
    ctrlc::set_handler(move || {
        let _ = tx.send(());
    })
    .expect("Failed to set Ctrl+C handler");

    rx.recv().unwrap();
    info!("Shutdown signal received. Draining queues...");

    // Give threads a moment to drain
    thread::sleep(Duration::from_millis(500));
    info!("Trading engine stopped.");
}
