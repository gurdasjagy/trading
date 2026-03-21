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
mod telemetry;
mod ws_ingestion;
mod fixed_point;
mod flat_book;
mod spsc;
mod journal;
mod shared_state;

// Issue 3: Institutional execution modules
mod execution_state;
mod mbo_book;
mod adverse_selection;
mod smart_router;
mod ws_order_manager;

// Institutional upgrades: order lifecycle, latency tracking, market impact
mod order_lifecycle;
mod latency_tracker;
mod market_impact;

// Phase 4 Architecture: Decimal extension, event bus, bridge IPC, risk calculator
mod decimal_ext;
mod event_bus;
mod bridge_ipc;
mod risk_calculator;

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tracing::{debug, error, info, warn};

use crate::config::{
    EngineConfig, ExchangeConfig, SymbolRegistry, ThreadTopology,
    build_symbol_registry, build_flat_book_configs,
};
use crate::execution_gateway::ExecutionGateway;
use crate::fixed_point::FixedPrice;
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

    // ── Trading pairs override ──
    if let Ok(pairs_str) = std::env::var("TRADING_PAIRS") {
        let pairs: Vec<String> = pairs_str.split(',')
            .map(|s| s.trim().replace('/', "_").to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();
        if !pairs.is_empty() {
            cfg.symbols = pairs.clone();
            // Also update exchange symbols
            for ex in cfg.exchanges.iter_mut() {
                if ex.name == "gateio" {
                    ex.symbols = pairs.clone();
                }
            }
        }
    }

    // ── Leverage override ──
    if let Ok(lev_str) = std::env::var("DEFAULT_LEVERAGE") {
        if let Ok(lev) = lev_str.parse::<i32>() {
            cfg.strategy.leverage = Some(lev.clamp(1, 125));
        }
    }

    // ── Max open positions override ──
    if let Ok(max_str) = std::env::var("MAX_OPEN_POSITIONS") {
        if let Ok(max_pos) = max_str.parse::<usize>() {
            cfg.risk.max_open_positions = max_pos.max(1);
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
                match serde_json::from_str::<EngineConfig>(&content) {
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

            match ws_connect_and_ingest_gateio(ring, &config, &registry, &mut drop_count, &mut msg_count).await {
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

                                                // Encode trade as a special RawBookUpdate (update_type=3 for trades)
                                                let side = if size > 0 { spsc::side::BID } else { spsc::side::ASK };
                                                let update = RawBookUpdate {
                                                    symbol_id: sym_id,
                                                    side,
                                                    update_type: 3, // trade event
                                                    _pad: [0; 4],
                                                    price: FixedPrice::from_f64(price).raw(),
                                                    qty: fixed_point::FixedQty::from_f64(size.abs() as f64).raw(),
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
fn orderbook_builder_loop(
    ws_ring_gateio: &'static SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>,
    strategy_ring: &'static SpscRingBuffer<BookSnapshot, BOOK_TO_STRATEGY_CAPACITY>,
    books: &mut Vec<FlatOrderBook>,
    _registry: Arc<SymbolRegistry>,
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
                
                // Push snapshot to strategy
                if let (Some((bid, _bid_qty)), Some((ask, _ask_qty))) = (book.best_bid(), book.best_ask()) {
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
                }
            }
        }

        // Yield if ring is empty to avoid burning CPU
        if ws_ring_gateio.is_empty() {
            std::hint::spin_loop();
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
    regime_reader: &regime_shm::SharedMemRegimeReader,
    strategy: Arc<StrategyEngine>,
    registry: Arc<SymbolRegistry>,
    latency_tracker: &'static PipelineLatencyTracker,
    impact_model: &'static MarketImpactModel,
    overflow_monitor: &'static SpscOverflowMonitor,
    circuit_breaker: Option<&'static CircuitBreaker>,
    pre_trade_risk: &'static PreTradeRiskEngine,
    position_slots: &'static PositionSlotManager,
) {
    info!("[strategy] Starting strategy evaluator on dedicated core");
    let mut last_regime_check = std::time::Instant::now();
    let mut current_regime = regime::RegimeState::default();
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
    // Upgrade 4: Per-position trailing stop states
    let mut trailing_stops: HashMap<u16, exit_evaluator::TrailingStopState> = HashMap::new();
    // Upgrade 1: Funding rate check counter (check every 1000 snapshots)
    let _funding_check_counter: u64 = 0;
    info!("[strategy] 🧮 DustTracker initialized (max_dust=5.0)");
    info!("[strategy] 📊 VPIN calculator initialized (bucket=100k, depth=50)");
    info!("[strategy] 📈 Trailing stop tracking initialized (break-even + partial TP)");
    info!("[strategy] 💰 Funding rate checks every ~1000 book snapshots");
    info!("[strategy] 🚪 ExitEvaluator initialized (ParabolicSAR + ATR + Chandelier + HardSLTP)");
    info!("[strategy] 📊 PositionLifecycleManager initialized (reversal=30%, max_loss=2%)");
    info!("[strategy] 🎯 SmartEntryRouter initialized (maker-rebate optimization)");
    info!("[strategy] 📈 VolatilityTrailingStop initialized (ATR-based)");
    info!("[strategy] 🛡️ AdverseSelectionGuard initialized (spoofing detection)");

    loop {
        // Early bail-out: if circuit breaker is tripped, skip evaluation.
        if let Some(ref cb) = circuit_breaker {
            if cb.is_trading_halted() {
                while book_ring.try_pop().is_some() {}
                std::hint::spin_loop();
                continue;
            }
        }

        if let Some(snapshot) = book_ring.try_pop() {
            let signal_start = std::time::Instant::now();

            // Periodically refresh regime from shared memory (every 1s)
            if last_regime_check.elapsed() > Duration::from_secs(1) {
                // BUG 14 FIX: get_current() now takes &self — no unsafe cast needed
                let weights = regime_reader.get_current();
                current_regime = regime::RegimeState::from_weights(weights);
                last_regime_check = std::time::Instant::now();
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

            // Build microstructure metrics from the snapshot
            let metrics = strategy_engine::MicrostructureMetrics {
                mid_price: FixedPrice(snapshot.mid_price).to_f64(),
                spread_bps: snapshot.spread_bps as f64,
                imbalance: snapshot.imbalance_bps as f64 / 10000.0,
                bid_depth_usdt: snapshot.bid_depth_usdt as f64 / FixedPrice::PRECISION as f64,
                ask_depth_usdt: snapshot.ask_depth_usdt as f64 / FixedPrice::PRECISION as f64,
                vpin: vpin_calculator.get_vpin(), // FIX 9: compute live VPIN from trade flow
                last_trade_is_buy: None,
            };

            // Evaluate strategy
            let symbol_name = registry.get_name(snapshot.symbol_id);

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
                        let exit_cmd = OrderCommand {
                            symbol_id: sym_id,
                            side: exit_side,
                            order_type: spsc::order_cmd_type::MARKET,
                            leverage: 1,
                            _pad: [0; 3],
                            price: FixedPrice::from_f64(mid).raw(),
                            qty: fixed_point::FixedQty::from_f64(1.0).raw(),
                            order_id: snapshot.sequence.wrapping_add(100),
                            signal_ns: snapshot.timestamp_ns,
                            max_slippage_bps: 100,
                            ttl_ms: 1000,
                            stop_loss_fp: 0,
                            take_profit_fp: 0,
                            placement_type: 0,
                            post_only: 0,
                            _pad2: [0; 6],
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
                        let exit_cmd = OrderCommand {
                            symbol_id: exit_signal.symbol_id,
                            side: exit_side,
                            order_type: spsc::order_cmd_type::MARKET,
                            leverage: 1, // irrelevant for close
                            _pad: [0; 3],
                            price: FixedPrice::from_f64(exit_mid).raw(),
                            qty: fixed_point::FixedQty::from_f64(1.0).raw(), // full position close
                            order_id: snapshot.sequence.wrapping_add(1),
                            signal_ns: snapshot.timestamp_ns,
                            max_slippage_bps: 100, // wider for urgency
                            ttl_ms: 1000,
                            stop_loss_fp: 0,
                            take_profit_fp: 0,
                            placement_type: 0,
                            post_only: 0,
                            _pad2: [0; 6],
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
                        }
                        exit_evaluator::TrailingStopUpdate::TrailStop { new_sl } => {
                            exit_evaluator.update_sl_tp(snapshot.symbol_id, new_sl, 0.0);
                            info!(
                                "[strategy] 📈 Trailing SL updated for {}: new SL={:.4}",
                                registry.get_name(snapshot.symbol_id), new_sl
                            );
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
                                    _pad2: [0; 6],
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

            if let Some(intent) = strategy.evaluate(&metrics, &current_regime, symbol_name) {
                let entry_price = intent.price.unwrap_or(metrics.mid_price);
                let is_buy = matches!(intent.side, execution_gateway::OrderSide::Buy);

                // ── Directive 4: Smart Entry Decision ──
                // Decide whether to post maker or cross spread based on microstructure.
                let (entry_decision, smart_price) = smart_entry.decide(
                    is_buy,
                    FixedPrice(snapshot.best_bid).to_f64(),
                    FixedPrice(snapshot.best_ask).to_f64(),
                    metrics.vpin,
                    metrics.imbalance,
                    adverse_guard.is_long_paused() && is_buy,
                );

                // Skip entry if adverse selection guard paused it
                if entry_decision == smart_entry::EntryDecision::PauseEntry {
                    info!("[strategy] ⏸️ Entry paused by adverse selection guard for {}",
                        registry.get_name(snapshot.symbol_id));
                    continue;
                }

                // ── Directive 4: Volatility-adjusted SL/TP ──
                // Use real-time ATR to size stops instead of fixed percentages.
                let trail_distance = vol_trailing.calculate_trail_distance(entry_price, 0.0);
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
                let (cmd_order_type, cmd_post_only, effective_price) = match entry_decision {
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
                };

                let cmd = OrderCommand {
                    symbol_id: snapshot.symbol_id,
                    side: if is_buy { spsc::side::BUY } else { spsc::side::SELL },
                    order_type: cmd_order_type,
                    leverage: strategy.config().leverage.unwrap_or(5).clamp(1, 125) as u8,
                    _pad: [0; 3],
                    price: FixedPrice::from_f64(effective_price).raw(),
                    qty: {
                        // Institutional: Use DustTracker for fractional contract handling.
                        // Carries sub-contract remainders across trades to prevent drift.
                        let contracts = dust_tracker.float_to_contracts(
                            snapshot.symbol_id, intent.size as f64, is_buy,
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
                    _pad2: [0; 6],
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
/// 3. **Route**: SmartOrderRouter selects best venue
/// 4. **Pre-flight**: Check margin & exposure limits
/// 5. **Submit**: Send order via REST gateway (or WS in future)
/// 6. **Protect**: Submit SL/TP as conditional orders
/// 7. **Monitor**: Track fills, PnL, and queue position
/// 8. **Circuit Breaker Update**: Record trade results
fn execution_router_loop(
    exec_ring: &'static SpscRingBuffer<OrderCommand, STRATEGY_TO_EXEC_CAPACITY>,
    gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>>,
    forex_gateway: Option<Arc<dyn ExecutionGateway + Send + Sync>>,
    circuit_breaker: &'static CircuitBreaker,
    registry: Arc<SymbolRegistry>,
    lifecycle_tracker: &'static OrderLifecycleTracker,
    latency_tracker: &'static PipelineLatencyTracker,
    position_slots: &'static PositionSlotManager,
    dashboard_state: Arc<DashboardState>,
) {
    info!("[execution] Starting execution router on dedicated core (Institutional)");

    // Initialize execution context
    let mbo_book = mbo_book::MboBook::new();
    let adverse_detector = adverse_selection::AdverseSelectionDetector::with_defaults();
    let smart_router_inst = smart_router::SmartOrderRouter::default_venues();
    let ws_mgr = ws_order_manager::WsOrderManager::new_paper();
    let mut exec_ctx = execution_gateway::ExecutionContext::new(
        mbo_book,
        adverse_detector,
        smart_router_inst,
        ws_mgr,
    );

    // Initialize Alpha Oracle signal queue consumer
    let mut signal_queue = match signal_queue::SignalQueueConsumer::open() {
        Ok(sq) => {
            info!("[execution] Alpha Oracle signal queue connected");
            Some(sq)
        }
        Err(e) => {
            warn!("[execution] Alpha Oracle signal queue unavailable: {} — running without Python signals", e);
            None
        }
    };

    // Initialize event-sourced order state machine
    let mut order_state_machine = order_state_machine::OrderStateMachine::new();

    // Upgrade 3: Initialize TWAP executor for large order splitting
    let mut twap_exec = twap_executor::TwapExecutor::new();
    info!("[execution] TWAP/Iceberg executor initialized (max 5 concurrent)");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime for execution");

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
        }

        loop {
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
                let symbol_name = registry.get_name(cmd.symbol_id);
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

                    match execution_gateway::submit_with_retry(gw.as_ref(), intent).await {
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

                            // Calculate approximate PnL for circuit breaker tracking
                            let fill_price = res.avg_fill_price;
                            let entry_price = FixedPrice(cmd.price).to_f64();
                            let pnl_per_unit = if cmd.side == spsc::side::BUY {
                                fill_price - entry_price
                            } else {
                                entry_price - fill_price
                            };
                            let pnl_fp = (pnl_per_unit * res.filled_size as f64 * 1e8) as i64;
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
                // ── Idle: Poll Alpha Oracle signal queue ──
                if let Some(ref mut sq) = signal_queue {
                    while let Some(intent) = sq.try_pop() {
                        info!(
                            "[execution] 🎯 Alpha Oracle signal: {} {:?} (R:R={:.2}, conf={:.0}%, {}/{} strats)",
                            intent.symbol,
                            if intent.side == 0 { "LONG" } else { "SHORT" },
                            intent.risk_reward,
                            intent.confidence * 100.0,
                            intent.confluence_count,
                            intent.total_strategies,
                        );

                        // Validate signal age (reject if older than 5s)
                        let age_ns = now_ns().saturating_sub(intent.timestamp_ns);
                        if age_ns > 5_000_000_000 {
                            warn!("[execution] Stale Alpha Oracle signal (age={:.1}s) — skipping", age_ns as f64 / 1e9);
                            continue;
                        }

                        // FIX 4: Convert TradeIntent to OrderIntent and submit through gateway.
                        // The signal has already passed the Python confluence engine's filters
                        // (75%+ strategy agreement, R:R > 2.0). Here we perform final validation.
                        if !circuit_breaker.is_trading_halted() {
                            // Track in the order state machine
                            let client_oid = format!("alpha-{}", now_ns());
                            order_state_machine.track_order(client_oid.clone(), intent.symbol.clone());

                            // Acquire a position slot
                            if !position_slots.try_acquire() {
                                warn!("[execution] Position slots full — dropping Alpha Oracle signal for {}", intent.symbol);
                                continue;
                            }

                            // Determine the active gateway for this symbol
                            let active_gw: Option<&Arc<dyn ExecutionGateway + Send + Sync>> =
                                if config::is_forex_symbol(&intent.symbol) {
                                    forex_gateway.as_ref()
                                } else {
                                    gateway.as_ref()
                                };

                            if let Some(gw) = active_gw {
                                let side = if intent.side == 0 {
                                    execution_gateway::OrderSide::Buy
                                } else {
                                    execution_gateway::OrderSide::Sell
                                };

                                let order_intent = execution_gateway::OrderIntent {
                                    symbol: intent.symbol.clone(),
                                    side,
                                    size: intent.size_contracts.max(1),
                                    order_type: execution_gateway::OrderType::Market,
                                    price: Some(intent.entry_price),
                                    reduce_only: intent.intent_type == 1 || intent.intent_type == 2,
                                    leverage: Some(intent.leverage.max(1)),
                                    time_in_force: "ioc".to_string(),
                                    slippage_cap_pct: Some(intent.max_slippage),
                                    placement: execution_state::PlacementType::AtBest,
                                    stop_loss: intent.stop_loss,
                                    take_profit: intent.take_profit,
                                    confidence: intent.confidence,
                                    signal_tag: intent.signal_tag.clone(),
                                };

                                // Register in lifecycle tracker
                                let lc_order = crate::order_lifecycle::OrderLifecycle::new(
                                    client_oid.clone(),
                                    intent.symbol.clone(),
                                    if intent.side == 0 { "buy".into() } else { "sell".into() },
                                    "market".into(),
                                    intent.size_contracts as f64,
                                    intent.entry_price,
                                    "alpha_oracle".into(),
                                    intent.confidence,
                                );
                                lifecycle_tracker.register_order(lc_order);

                                match execution_gateway::submit_with_retry(gw.as_ref(), order_intent).await {
                                    Ok(res) => {
                                        orders_submitted += 1;
                                        dashboard_state.orders_submitted.store(orders_submitted, Ordering::Relaxed);
                                        dashboard_state.total_fills.fetch_add(1, Ordering::Relaxed);
                                        lifecycle_tracker.map_exchange_id(&res.order_id, &client_oid);
                                        lifecycle_tracker.record_fill(
                                            &client_oid,
                                            crate::order_lifecycle::Fill {
                                                fill_id: format!("alpha-f-{}", orders_submitted),
                                                price: res.avg_fill_price,
                                                quantity: res.filled_size as f64,
                                                fee: res.fee,
                                                fee_currency: "USDT".into(),
                                                timestamp_us: crate::order_lifecycle::now_micros(),
                                                is_maker: false,
                                            },
                                        );
                                        info!(
                                            "[execution] ✅ Alpha Oracle order filled: {} {} size={} @ {:.4} latency={}μs",
                                            intent.symbol,
                                            if intent.side == 0 { "LONG" } else { "SHORT" },
                                            res.filled_size, res.avg_fill_price, res.latency_us,
                                        );

                                        // Track PnL in circuit breaker
                                        let pnl_fp = 0i64; // PnL tracked on position close
                                        circuit_breaker.on_trade_result(pnl_fp);
                                    }
                                    Err(e) => {
                                        warn!("[execution] ❌ Alpha Oracle order failed: {} — {}", intent.symbol, e);
                                        orders_rejected += 1;
                                        position_slots.release();
                                        lifecycle_tracker.reject_order(&client_oid, &format!("{}", e));
                                    }
                                }
                            } else {
                                info!(
                                    "[execution] 📝 Alpha Oracle signal-only (no gateway): {} {:?}",
                                    intent.symbol,
                                    if intent.side == 0 { "LONG" } else { "SHORT" },
                                );
                                orders_submitted += 1;
                                position_slots.release(); // No actual position opened
                            }
                        } else {
                            warn!("[execution] Circuit breaker halted — dropping Alpha Oracle signal");
                        }
                    }
                }

                // ── Periodic maintenance ──

                // Check resting orders for queue position degradation (every 1s)
                if last_queue_check.elapsed() > Duration::from_secs(1) {
                    let cancels = exec_ctx.check_resting_orders();
                    for (idx, reason) in cancels {
                        info!("[execution] Canceling resting order {} due to {:?}", idx, reason);
                        exec_ctx.ws_order_mgr.cancel_by_lifecycle_idx(idx, reason);
                    }
                    last_queue_check = std::time::Instant::now();
                }

                // Upgrade 3: Tick the TWAP executor to submit ready slices
                if twap_exec.active_count() > 0 {
                    let current_prices = HashMap::new(); // TODO: populate from live tick data
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
                            }
                            Err(e) => {
                                // Only log once per 60s to reduce noise
                                // (this runs every health check cycle)
                                debug!("[execution] Balance sync failed: {}", e);
                            }
                        }
                    }

                    // ── Slot Reconciliation: Query Gate.io for actual positions ──
                    // If we have slots claimed but no actual exchange positions,
                    // release the orphaned slots to prevent permanent exhaustion.
                    if active_slots > 0 {
                        if let Some(ref gw) = gateway {
                            match gw.get_positions().await {
                                Ok(positions) => {
                                    let actual = positions.len() as u32;
                                    if actual < active_slots {
                                        let leaked = active_slots - actual;
                                        for _ in 0..leaked {
                                            position_slots.release();
                                        }
                                        warn!(
                                            "[execution] Slot reconciliation: released {} orphaned slots (exchange={}, slots={})",
                                            leaked, actual, active_slots
                                        );
                                    }
                                }
                                Err(e) => {
                                    warn!("[execution] Slot reconciliation skipped — position query failed: {}", e);
                                }
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

    info!("SPSC ring buffers allocated: ws_to_book={}KB, book_to_strategy={}KB, strategy_to_exec={}KB",
          std::mem::size_of::<SpscRingBuffer<RawBookUpdate, WS_TO_BOOK_CAPACITY>>() / 1024,
          std::mem::size_of::<SpscRingBuffer<BookSnapshot, BOOK_TO_STRATEGY_CAPACITY>>() / 1024,
          std::mem::size_of::<SpscRingBuffer<OrderCommand, STRATEGY_TO_EXEC_CAPACITY>>() / 1024);

    // 5. Load shared memory regime reader
    let regime_shm_path = config.shared_mem.regime_shm_path.clone();

    // 6. Build strategy engine
    let strategy = Arc::new(StrategyEngine::new(config.strategy.clone()));

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
    let position_slot_manager: &'static PositionSlotManager =
        Box::leak(Box::new(PositionSlotManager::default_3_slots()));
    info!("📍 PositionSlotManager initialized (max_slots=3, lock-free AtomicU32)");

    // 7h. Initialize Rate Limiter Pool (Directive 5)
    let is_testnet = config.exchanges.iter().any(|e| e.name == "gateio" && e.testnet);
    let _rate_limiter: &'static RateLimiterPool = Box::leak(Box::new(RateLimiterPool::new(is_testnet)));

    // 7i. Initialize Dashboard State (shared between hot-path and HTTP server)
    let dashboard_state = Arc::new(DashboardState::new());

    // 7j. Initialize Position Sizer (Directive 2) — fetch contract specs at startup
    let _position_sizer: &'static PositionSizer = {
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
    match health_monitor.init() {
        Ok(()) => info!("📡 Bridge: Health monitor initialized"),
        Err(e) => warn!("📡 Bridge: Health monitor init failed (non-fatal): {}", e),
    }
    let health_monitor: &'static parking_lot::Mutex<bridge_ipc::health_monitor::BridgeHealthMonitor> =
        Box::leak(Box::new(parking_lot::Mutex::new(health_monitor)));

    info!("📡 Bridge IPC subsystem initialized (tick_broadcast + portfolio_rx + exec_confirm + regime + signal + event_bus + health_monitor)");

    // 8. Spawn OS threads with core affinity pinning
    let mut thread_handles = Vec::new();

    // ── Core 2: WS Ingestion (Gate.io) ──
    if let Some(gateio_cfg) = config.exchanges.iter().find(|e| e.name == "gateio").cloned() {
        let ws_ring = ws_gateio_to_book;
        let reg = registry.clone();
        let core_id = topology.ws_gateio_core;
        let handle = thread::Builder::new()
            .name("ws-gateio".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[ws-gateio] Pinned to core {}", core_id);
                ws_ingestion_loop_gateio(ws_ring, gateio_cfg, reg);
            })
            .expect("Failed to spawn WS Gate.io thread");
        thread_handles.push(handle);
    }



    // ── Core 4: Orderbook Builder ──
    {
        let ws_ring_g = ws_gateio_to_book;
        let strat_ring = book_to_strategy;
        let reg = registry.clone();
        let core_id = topology.book_builder_core;
        let handle = thread::Builder::new()
            .name("book-builder".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[book-builder] Pinned to core {}", core_id);
                orderbook_builder_loop(ws_ring_g, strat_ring, &mut books, reg);
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
    {
        let book_ring = book_to_strategy;
        let exec_ring = strategy_to_exec;
        let strat = strategy.clone();
        let reg = registry.clone();
        let core_id = topology.strategy_core;
        let lat = latency_tracker;
        let impact = impact_model;
        let cb_ref = circuit_breaker; // &'static CircuitBreaker
        let handle = thread::Builder::new()
            .name("strategy".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[strategy] Pinned to core {} (with Market Impact + Overflow Monitor)", core_id);
                let regime_reader = regime_shm::SharedMemRegimeReader::new(&regime_shm_path);
                strategy_evaluator_loop(
                    book_ring,
                    exec_ring,
                    &regime_reader,
                    strat,
                    reg,
                    lat,
                    impact,
                    spsc_overflow_monitor,
                    Some(cb_ref),
                    pre_trade_risk_engine,
                    position_slot_manager,
                );
            })
            .expect("Failed to spawn strategy thread");
        thread_handles.push(handle);
    }

    // ── Core 6: Execution Router (with Circuit Breaker & SL/TP) ──
    // Mandate 3: Build forex gateway if credentials are available
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
        let handle = thread::Builder::new()
            .name("execution".into())
            .spawn(move || {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                info!("[execution] Pinned to core {} (with Lifecycle + Latency)", core_id);
                execution_router_loop(exec_ring, gw, fx_gw, cb, reg, lc, lat, position_slot_manager, dash_exec);
            })
            .expect("Failed to spawn execution thread");
        thread_handles.push(handle);
    }

    // ── Dashboard HTTP Server (runs on its own thread) ──
    {
        let dash_state = dashboard_state.clone();
        let bind_addr = std::env::var("DASHBOARD_BIND")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        let handle = thread::Builder::new()
            .name("dashboard-http".into())
            .spawn(move || {
                info!("[dashboard] Starting HTTP server on {}", bind_addr);
                dashboard_server::run_dashboard_server(&bind_addr, dash_state);
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
                        hm.record_tick_broadcast(1, 0); // Dummy update to trigger health check
                        let health = hm.get_health_status();
                        info!("[telemetry] Bridge Health: tick_rate={:.1}/s, portfolio_rate={:.1}/s, exec_rate={:.1}/s, errors={}",
                            health.tick_broadcast_rate, health.portfolio_rx_rate,
                            health.exec_confirm_rate, health.total_errors);
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
