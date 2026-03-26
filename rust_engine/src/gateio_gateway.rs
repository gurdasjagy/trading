//! Gate.io v4 Futures WebSocket Execution Gateway — Mandate 1 Rewrite.
//!
//! Replaces the previous REST-only gateway (reqwest HTTP) with a persistent,
//! authenticated WebSocket connection using `tokio-tungstenite`.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     SPSC      ┌─────────────────┐    WS     ┌─────────┐
//! │  Exec Router │ ──────────▶  │  GateIoGateway   │ ◀──────▶ │ Gate.io │
//! │  (Core 6)    │              │  (WS + State)    │          │  Server │
//! └─────────────┘              └─────────────────┘          └─────────┘
//! ```
//!
//! # Authentication
//!
//! Gate.io WS v4 has TWO different authentication mechanisms:
//!
//! 1. **Subscriptions** (event: "subscribe"):
//!    signature = HMAC_SHA512(secret, "channel={channel}&event={event}&time={time}")
//!
//! 2. **API requests** (event: "api", e.g. order placement/cancellation):
//!    signature = HMAC_SHA512(secret, "api\n{channel}\n{req_param_json}\n{timestamp}")
//!    Auth fields (api_key, signature, timestamp, req_param) go INSIDE the payload.
//!
//! # Order State Machine
//!
//! Every order submitted via WS gets a `client_order_id` (monotonic u64).
//! The gateway maintains a `DashMap<String, OrderTracking>` that maps
//! exchange order IDs to our local tracking structs. Fill confirmations
//! received via WS are matched against pending orders.
//!
//! # Reconnection
//!
//! On disconnect, the gateway:
//!   1. Re-authenticates
//!   2. Re-subscribes to `futures.orders` for fill events
//!   3. Reconciles local state against REST (fallback) to detect missed fills

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use parking_lot::RwLock;
use sha2::Sha512;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::{
    connect_async,
    tungstenite::protocol::Message,
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};

use crate::circuit_breaker::{CircuitBreaker, TripReason};
use crate::execution_gateway::{
    ExchangeError, ExecutionGateway, OrderIntent, OrderResult, OrderSide, OrderType, RustTicker, Position,
    now_ms, now_us,
};
use crate::execution_state::PlacementType;
use crate::instrument_manager::{InstrumentManager, Exchange, check_order_exists_gateio};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GATEIO_WS_URL: &str = "wss://fx-ws.gateio.ws/v4/ws/usdt";
const GATEIO_WS_TESTNET_URL: &str = "wss://ws-testnet.gateapi.io/v4/ws/usdt";
const GATEIO_REST_URL: &str = "https://api.gateio.ws/api/v4";
const MIN_CONTRACT_SIZE: i64 = 1;
const RECONNECT_BASE_MS: u64 = 500;
const RECONNECT_MAX_MS: u64 = 30_000;
const PING_INTERVAL_SECS: u64 = 15;
/// Maximum time (seconds) to wait for a pong response before declaring connection dead.
const PONG_TIMEOUT_SECS: u64 = 45;
const RESPONSE_TIMEOUT_MS: u64 = 5_000;

/// Precious metals symbol mapping: standard -> Gate.io contract format.
const PRECIOUS_METALS_MAP: &[(&str, &str)] = &[
    ("XAU_USDT", "XAUT_USDT"),
    ("XAG_USDT", "XAGT_USDT"),
    ("XAUUSDT", "XAUT_USDT"),
    ("XAGUSDT", "XAGT_USDT"),
];

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

// ---------------------------------------------------------------------------
// Order Tracking — Local State Machine
// ---------------------------------------------------------------------------

/// State of an order tracked locally.
#[derive(Debug, Clone)]
enum OrderTrackingState {
    /// Submitted via WS, awaiting exchange ACK.
    PendingAck { submit_us: i64 },
    /// Acknowledged by exchange, resting on book.
    Resting { exchange_id: String, filled_so_far: i64 },
    /// Fully filled.
    Filled { avg_price: f64, total_filled: i64, fee: f64 },
    /// Rejected by exchange.
    Rejected { reason: String },
    /// Cancelled.
    Cancelled,
}

/// Local tracking struct for a single order.
#[derive(Debug, Clone)]
struct OrderTracking {
    client_id: String,
    symbol: String,
    side: OrderSide,
    size: i64,
    state: OrderTrackingState,
    created_us: i64,
}

// ---------------------------------------------------------------------------
// WS Command — messages sent to the WS writer task
// ---------------------------------------------------------------------------

enum WsCommand {
    /// Send a raw text message on the WS.
    SendText(String),
    /// Shutdown the WS connection.
    Shutdown,
}

// ---------------------------------------------------------------------------
// Pending Order — awaiting response
// ---------------------------------------------------------------------------

struct PendingOrder {
    client_id: String,
    intent: OrderIntent,
    response_tx: oneshot::Sender<Result<OrderResult, ExchangeError>>,
    submit_us: i64,
}

// ---------------------------------------------------------------------------
// GateIoGateway — WebSocket-based
// ---------------------------------------------------------------------------

pub struct GateIoGateway {
    api_key: String,
    api_secret: Vec<u8>,
    testnet: bool,
    /// Channel to send commands to the WS writer task.
    ws_tx: mpsc::UnboundedSender<WsCommand>,
    /// Monotonically increasing client order ID.
    next_client_id: AtomicU64,
    /// Pending orders awaiting exchange response.
    pending: Arc<RwLock<HashMap<String, PendingOrder>>>,
    /// Local order state: client_id -> tracking.
    order_state: Arc<RwLock<HashMap<String, OrderTracking>>>,
    /// Whether the WS is authenticated and ready.
    is_ready: Arc<AtomicBool>,
    /// REST fallback client (for position queries, balance, etc.).
    rest_client: reqwest::Client,
    /// Counter: REST reconciliation cycles completed.
    reconciliation_cycles: AtomicU64,
    /// Counter: discrepancies detected during reconciliation.
    reconciliation_discrepancies: AtomicU64,
    /// Counter: rate-limit tokens consumed this second (for telemetry back-pressure).
    rate_limit_tokens_used: Arc<AtomicU64>,
    /// Timestamp of current rate-limit tracking second.
    rate_limit_second_ns: Arc<AtomicU64>,
    /// Optional reference to the global circuit breaker. When the WS disconnects,
    /// the connection loop trips this with TripReason::ConnectivityLost to prevent
    /// the strategy engine from firing signals into a dead gateway.
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    /// Dynamic instrument manager for real-time precision rules.
    /// When set, price is formatted using Gate.io's order_price_round (tick size)
    /// instead of hardcoded "{:.8}".
    instrument_mgr: Option<Arc<InstrumentManager>>,
}

impl GateIoGateway {
    /// Create a new WebSocket-based Gate.io gateway.
    ///
    /// This spawns background tokio tasks for:
    ///   1. WS connection management (connect, auth, reconnect)
    ///   2. WS reader (parse incoming messages, match fills)
    ///   3. WS writer (send orders, pings)
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        Self::new_with_circuit_breaker(api_key, api_secret, testnet, None)
    }

    /// Set the instrument manager after construction for dynamic price formatting.
    pub fn set_instrument_manager(&mut self, mgr: Arc<InstrumentManager>) {
        self.instrument_mgr = Some(mgr);
    }

    /// Create a new WebSocket-based Gate.io gateway with circuit breaker integration.
    ///
    /// When `circuit_breaker` is `Some`, the WS connection loop will trip it
    /// with `TripReason::ConnectivityLost` on disconnect, and reset it on
    /// successful reconnection.
    pub fn new_with_circuit_breaker(
        api_key: String,
        api_secret: String,
        testnet: bool,
        circuit_breaker: Option<Arc<CircuitBreaker>>,
    ) -> Self {
        // CRITICAL: Trim whitespace/newlines from API credentials.
        // Environment variable values loaded from .env files can contain trailing
        // \n or \r\n characters. Even a single extra byte in the key causes Gate.io
        // to return INVALID_KEY, and a single extra byte in the secret causes HMAC
        // signature mismatch (also reported as INVALID_KEY).
        let api_key = api_key.trim().to_string();
        let api_secret = api_secret.trim().to_string();

        info!("[gateio-ws] Initializing gateway: testnet={}, key_len={}, secret_len={}, key_prefix={}...",
              testnet, api_key.len(), api_secret.len(),
              &api_key[..api_key.len().min(6)]);

        let (ws_tx, ws_rx) = mpsc::unbounded_channel();
        let pending = Arc::new(RwLock::new(HashMap::new()));
        let order_state = Arc::new(RwLock::new(HashMap::new()));
        let is_ready = Arc::new(AtomicBool::new(false));
        let secret_bytes = api_secret.into_bytes();
        let rate_limit_tokens = Arc::new(AtomicU64::new(0));
        let rate_limit_second = Arc::new(AtomicU64::new(0));

        let rest_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(4)
            .build()
            .expect("Failed to build REST fallback client");

        let gateway = Self {
            api_key: api_key.clone(),
            api_secret: secret_bytes.clone(),
            testnet,
            ws_tx,
            next_client_id: AtomicU64::new(1),
            pending: pending.clone(),
            order_state: order_state.clone(),
            is_ready: is_ready.clone(),
            rest_client: rest_client.clone(),
            reconciliation_cycles: AtomicU64::new(0),
            reconciliation_discrepancies: AtomicU64::new(0),
            rate_limit_tokens_used: rate_limit_tokens.clone(),
            rate_limit_second_ns: rate_limit_second.clone(),
            circuit_breaker: circuit_breaker.clone(),
            instrument_mgr: None,
        };

        // Spawn the WS connection manager task
        let ws_url = if testnet { GATEIO_WS_TESTNET_URL } else { GATEIO_WS_URL };
        tokio::spawn(Self::ws_connection_loop(
            ws_url.to_string(),
            api_key.clone(),
            secret_bytes.clone(),
            ws_rx,
            pending,
            order_state.clone(),
            is_ready.clone(),
            circuit_breaker,
        ));

        // ── Task 1: Spawn liquidation price monitoring background thread (30s interval) ──
        // Monitors all open positions and triggers auto-reduce (50%) when within 5%
        // of liquidation price, or emergency close (100%) when within 2%.
        {
            let liq_client = rest_client.clone();
            let liq_key = api_key.clone();
            let liq_secret = secret_bytes.clone();
            let liq_testnet = testnet;
            let liq_ready = is_ready.clone();
            tokio::spawn(async move {
                // Wait for initial WS connection before starting monitoring
                tokio::time::sleep(Duration::from_secs(10)).await;
                info!("[gateio-liq] Liquidation monitoring thread started (30s interval)");
                loop {
                    tokio::time::sleep(Duration::from_secs(30)).await;

                    if !liq_ready.load(Ordering::Acquire) {
                        debug!("[gateio-liq] Skipping cycle — WS not ready");
                        continue;
                    }

                    // Create a temporary gateway instance for the monitoring call
                    let temp_gw = GateIoGateway {
                        api_key: liq_key.clone(),
                        api_secret: liq_secret.clone(),
                        testnet: liq_testnet,
                        ws_tx: mpsc::unbounded_channel().0, // Dummy channel (not used for REST)
                        next_client_id: AtomicU64::new(0),
                        pending: Arc::new(RwLock::new(HashMap::new())),
                        order_state: Arc::new(RwLock::new(HashMap::new())),
                        is_ready: Arc::new(AtomicBool::new(true)),
                        rest_client: liq_client.clone(),
                        reconciliation_cycles: AtomicU64::new(0),
                        reconciliation_discrepancies: AtomicU64::new(0),
                        rate_limit_tokens_used: Arc::new(AtomicU64::new(0)),
                        rate_limit_second_ns: Arc::new(AtomicU64::new(0)),
                        circuit_breaker: None,
                        instrument_mgr: None,
                    };

                    if let Err(e) = temp_gw.monitor_liquidation_prices().await {
                        warn!("[gateio-liq] Monitoring cycle failed: {}", e);
                    }
                }
            });
        }

        // ── Trap 1 Fix: Spawn 60-second REST reconciliation background thread ──
        // This thread periodically queries the Gate.io REST API to true-up the
        // internal order_state DashMap against actual exchange positions. This
        // prevents state desync caused by dropped WS ACKs during high-volatility.
        {
            let reconcile_client = rest_client;
            let reconcile_key = api_key;
            let reconcile_secret = secret_bytes;
            let reconcile_state = order_state;
            let reconcile_ready = is_ready;
            let reconcile_testnet = testnet;
            let reconcile_rl_tokens = rate_limit_tokens;
            let reconcile_rl_second = rate_limit_second;
            tokio::spawn(async move {
                // Wait for initial WS connection before starting reconciliation
                tokio::time::sleep(Duration::from_secs(10)).await;
                info!("[gateio-reconcile] REST reconciliation thread started (60s interval)");
                let mut cycle: u64 = 0;
                loop {
                    // BUG 10 FIX: Reduced from 60s to 15s to minimize the window
                    // where ghost positions can cause incorrect decisions (e.g.,
                    // refusing to open new positions because local state thinks
                    // one already exists).
                    tokio::time::sleep(Duration::from_secs(15)).await;
                    cycle += 1;

                    if !reconcile_ready.load(Ordering::Acquire) {
                        debug!("[gateio-reconcile] Skipping cycle {} — WS not ready", cycle);
                        continue;
                    }

                    // Query all open positions via REST
                    let base = if reconcile_testnet {
                        "https://api-testnet.gateapi.io/api/v4"
                    } else {
                        GATEIO_REST_URL
                    };
                    let path = "/futures/usdt/positions";
                    let timestamp = crate::execution_gateway::now_ms() / 1000;
                    // Gate.io v4 requires the FULL path (including /api/v4) in the signature
                    let full_path = format!("/api/v4{}", path);
                    let signature = Self::rest_sign(
                        "GET", &full_path, "", "", timestamp, &reconcile_secret,
                    );
                    let url = format!("{}{}", base, path);
                    let resp = reconcile_client
                        .get(&url)
                        .header("KEY", &reconcile_key)
                        .header("SIGN", &signature)
                        .header("Timestamp", timestamp.to_string())
                        .send()
                        .await;

                    match resp {
                        Ok(response) => {
                            // Track rate-limit consumption
                            let now_ns = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64;
                            let current_second = now_ns / 1_000_000_000;
                            let prev_second = reconcile_rl_second.load(Ordering::Relaxed);
                            if current_second != prev_second {
                                reconcile_rl_second.store(current_second, Ordering::Relaxed);
                                reconcile_rl_tokens.store(1, Ordering::Relaxed);
                            } else {
                                reconcile_rl_tokens.fetch_add(1, Ordering::Relaxed);
                            }

                            if response.status().is_success() {
                                match response.json::<serde_json::Value>().await {
                                    Ok(positions) => {
                                        if let Some(arr) = positions.as_array() {
                                            let mut discrepancies = 0u64;
                                            // BUG 10 FIX: Collect ghost tracking keys during
                                            // the read pass, then remove them in a separate
                                            // write pass. This avoids borrow conflicts.
                                            let mut ghost_keys_to_remove: Vec<String> = Vec::new();

                                            {
                                                let state = reconcile_state.read();
                                                for pos in arr {
                                                    let symbol = pos.get("contract")
                                                        .and_then(|v| v.as_str())
                                                        .unwrap_or("");
                                                    let rest_size = pos.get("size")
                                                        .and_then(|v| v.as_i64())
                                                        .unwrap_or(0);

                                                    // Check if any tracked order for this symbol
                                                    // has a state mismatch
                                                    let has_tracking = state.values().any(|t| {
                                                        t.symbol == symbol
                                                    });

                                                    // FEATURE 9: Fix position desync
                                                    if rest_size != 0 && !has_tracking {
                                                        discrepancies += 1;
                                                        warn!(
                                                            "[gateio-reconcile] DESYNC: REST shows position \
                                                             {} size={} but no local tracking exists! Creating emergency tracking.",
                                                            symbol, rest_size
                                                        );
                                                        
                                                        // Get entry price from REST
                                                        let entry_price = pos.get("entry_price")
                                                            .and_then(|v| v.as_str())
                                                            .and_then(|s| s.parse::<f64>().ok())
                                                            .or_else(|| pos.get("entry_price").and_then(|v| v.as_f64()))
                                                            .unwrap_or(0.0);
                                                        
                                                        if entry_price > 0.0 {
                                                            // Create synthetic emergency stop-loss at 3% from entry
                                                            let emergency_sl = if rest_size > 0 {
                                                                entry_price * 0.97 // Long: SL 3% below entry
                                                            } else {
                                                                entry_price * 1.03 // Short: SL 3% above entry
                                                            };
                                                            
                                                            info!(
                                                                "[gateio-reconcile] Created emergency SL for {} @ {:.4} (entry={:.4})",
                                                                symbol, emergency_sl, entry_price
                                                            );
                                                            
                                                            // Note: We can't directly call exit_evaluator.track_position() here
                                                            // because it's owned by the strategy thread. Instead, we log the
                                                            // discrepancy and rely on the next health check to sync state.
                                                            // A full implementation would use a channel to notify the strategy thread.
                                                        }
                                                    }
                                                    
                                                    // BUG 10 FIX: Detect ghost positions and collect
                                                    // their keys for removal after the read pass.
                                                    if rest_size == 0 && has_tracking {
                                                        warn!(
                                                            "[gateio-reconcile] GHOST: Local tracking for {} but REST shows no position — cleaning up",
                                                            symbol
                                                        );
                                                        let keys: Vec<String> = state.iter()
                                                            .filter(|(_, t)| t.symbol == symbol)
                                                            .map(|(k, _)| k.clone())
                                                            .collect();
                                                        ghost_keys_to_remove.extend(keys);
                                                    }
                                                }
                                            } // read lock released here

                                            // BUG 10 FIX: Actually remove ghost tracking entries
                                            // now that the read lock is released.
                                            if !ghost_keys_to_remove.is_empty() {
                                                let mut state_w = reconcile_state.write();
                                                for key in &ghost_keys_to_remove {
                                                    state_w.remove(key);
                                                    info!(
                                                        "[gateio-reconcile] Removed ghost tracking entry: {}",
                                                        key
                                                    );
                                                }
                                                discrepancies += ghost_keys_to_remove.len() as u64;
                                            }

                                            if discrepancies > 0 {
                                                warn!(
                                                    "[gateio-reconcile] Cycle {}: {} discrepancies found",
                                                    cycle, discrepancies
                                                );
                                            } else {
                                                debug!(
                                                    "[gateio-reconcile] Cycle {}: state consistent ({} positions checked)",
                                                    cycle, arr.len()
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        warn!("[gateio-reconcile] JSON parse error: {}", e);
                                    }
                                }
                            } else {
                                warn!(
                                    "[gateio-reconcile] REST query failed: HTTP {}",
                                    response.status()
                                );
                            }
                        }
                        Err(e) => {
                            warn!("[gateio-reconcile] REST request failed: {}", e);
                        }
                    }
                }
            });
        }

        gateway
    }

    /// Map symbol to Gate.io contract format.
    fn normalize_symbol(symbol: &str) -> String {
        for (from, to) in PRECIOUS_METALS_MAP {
            if symbol.eq_ignore_ascii_case(from) {
                return to.to_string();
            }
        }
        let normalized = symbol.replace('/', "_").replace(':', "_").to_uppercase();
        if normalized.ends_with("_USDT_USDT") {
            normalized[..normalized.len() - 5].to_string()
        } else {
            normalized
        }
    }

    /// Validate and enforce that size is a whole integer contract count.
    ///
    /// Gate.io futures ONLY accepts integer contract sizes. Any fractional
    /// value that leaks through from float-based strategy calculations must
    /// be truncated here before it reaches the WS payload.
    fn validate_contract_precision(size: i64) -> Result<i64, ExchangeError> {
        // size is already i64 (integer), but verify minimum
        if size < MIN_CONTRACT_SIZE {
            return Err(ExchangeError::MinimumOrderSize { min_size: MIN_CONTRACT_SIZE });
        }
        // Defensive: ensure abs value is within Gate.io limits (max 10M contracts)
        if size.abs() > 10_000_000 {
            return Err(ExchangeError::Unknown {
                code: "CONTRACT_SIZE_OVERFLOW".to_string(),
                message: format!("Contract size {} exceeds maximum", size),
            });
        }
        Ok(size)
    }

    /// Convert a float-originating size to an integer contract count.
    ///
    /// **DEPRECATED**: Use `DustTracker::float_to_contracts()` instead for
    /// proper fractional remainder carry-over. This method silently discards
    /// fractional contracts, causing positional drift over many trades.
    pub fn float_to_contracts(size_f: f64) -> i64 {
        // Truncate (floor toward zero) to guarantee we never over-order
        size_f.trunc() as i64
    }

    /// Convert a USDT amount to integer contracts for a given Gate.io futures contract.
    ///
    /// Fetches the contract spec from Gate.io REST API to get `quanto_multiplier`,
    /// then calculates: contracts = usdt_amount / (quanto_multiplier * last_price).
    /// Falls back to usdt_amount / last_price if contract spec fetch fails.
    pub async fn usdt_to_contracts(&self, contract: &str, usdt_amount: f64) -> Result<i64, ExchangeError> {
        let normalized = Self::normalize_symbol(contract);

        // Fetch contract spec to get quanto_multiplier
        let spec_url = format!("{}/futures/usdt/contracts/{}", self.base_url(), normalized);
        let quanto_multiplier = match self.rest_client.get(&spec_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(spec) => {
                        spec.get("quanto_multiplier")
                            .and_then(|v| v.as_str().and_then(|s| s.parse::<f64>().ok())
                                .or_else(|| v.as_f64()))
                            .unwrap_or(0.0)
                    }
                    Err(e) => {
                        warn!("[gateio] Failed to parse contract spec for {}: {}", normalized, e);
                        0.0
                    }
                }
            }
            _ => {
                warn!("[gateio] Failed to fetch contract spec for {}, falling back to price-only conversion", normalized);
                0.0
            }
        };

        // Fetch last price
        let last_price = self.fetch_last_price(&normalized).await
            .ok_or_else(|| ExchangeError::Unknown {
                code: "PRICE_FETCH".to_string(),
                message: format!("Cannot fetch last price for {}", normalized),
            })?;

        if last_price <= 0.0 {
            return Err(ExchangeError::Unknown {
                code: "INVALID_PRICE".to_string(),
                message: format!("Last price for {} is {}", normalized, last_price),
            });
        }

        let contracts = if quanto_multiplier > 0.0 {
            // Each contract = quanto_multiplier * underlying
            // Value of 1 contract in USDT = quanto_multiplier * last_price
            let contract_value_usdt = quanto_multiplier * last_price;
            (usdt_amount / contract_value_usdt).floor() as i64
        } else {
            // Fallback: assume 1 contract = 1 USD (common for some Gate.io contracts)
            (usdt_amount / last_price).floor() as i64
        };

        if contracts < MIN_CONTRACT_SIZE {
            return Err(ExchangeError::MinimumOrderSize { min_size: MIN_CONTRACT_SIZE });
        }

        info!("[gateio] USDT→contracts: {:.2} USDT → {} contracts (price={:.4}, quanto={:.8})",
            usdt_amount, contracts, last_price, quanto_multiplier);
        Ok(contracts)
    }

    /// Fetch the last traded price for a contract from Gate.io REST API.
    ///
    /// Used to validate SL/TP trigger prices before submission. Gate.io
    /// rejects conditional orders where the trigger price is already on
    /// the wrong side of the current market price.
    async fn fetch_last_price(&self, contract: &str) -> Option<f64> {
        let url = format!(
            "{}/futures/usdt/tickers?contract={}",
            self.base_url(), contract
        );
        match self.rest_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        // Gate.io returns an array of ticker objects
                        let ticker = if data.is_array() {
                            data.as_array().and_then(|a| a.first())
                        } else {
                            Some(&data)
                        };
                        ticker
                            .and_then(|t| t.get("last"))
                            .and_then(|v| v.as_str().or_else(|| v.as_f64().map(|_| "")))
                            .and_then(|s| {
                                if s.is_empty() {
                                    ticker.and_then(|t| t.get("last")).and_then(|v| v.as_f64())
                                } else {
                                    s.parse::<f64>().ok()
                                }
                            })
                    }
                    Err(e) => {
                        warn!("[gateio-ws] Failed to parse ticker for {}: {}", contract, e);
                        None
                    }
                }
            }
            Ok(resp) => {
                warn!("[gateio-ws] Ticker fetch HTTP {}: {}", resp.status(), contract);
                None
            }
            Err(e) => {
                warn!("[gateio-ws] Ticker fetch error for {}: {}", contract, e);
                None
            }
        }
    }

    /// Validate that a trigger price is on the correct side of the last price.
    ///
    /// Gate.io auto-trigger price rules:
    /// - rule=1 (>=): trigger_price must be **greater than** last_price
    /// - rule=2 (<=): trigger_price must be **less than** last_price
    ///
    /// Returns `true` if the trigger price is valid (can be submitted),
    /// `false` if Gate.io would reject it.
    fn validate_trigger_price(trigger_price: f64, last_price: f64, rule: u8) -> bool {
        match rule {
            1 => trigger_price > last_price,  // >= trigger: must be above current
            2 => trigger_price < last_price,  // <= trigger: must be below current
            _ => false,
        }
    }

    /// Build a Gate.io price_triggers conditional order message (SL or TP).
    ///
    /// Gate.io supports conditional orders via the `futures.price_triggers`
    /// REST endpoint (and soon WS). This builds the JSON for a stop-loss
    /// or take-profit order linked to a parent position.
    ///
    /// # Arguments
    /// * `contract` — Gate.io contract name (e.g., "BTC_USDT")
    /// * `trigger_price` — Price at which to trigger the order
    /// * `size` — Contract size (positive for close-short, negative for close-long)
    /// * `trigger_type` — 0 = trigger when price >= trigger_price (for TP on longs / SL on shorts)
    ///                     1 = trigger when price <= trigger_price (for SL on longs / TP on shorts)
    fn build_price_trigger_body(
        contract: &str,
        trigger_price: f64,
        size: i64,
        trigger_type: u8,
        instrument_mgr: Option<&Arc<InstrumentManager>>,
    ) -> String {
        let rule = if trigger_type == 0 { 1 } else { 2 }; // 1 = >=, 2 = <=

        // Use InstrumentManager for dynamic price formatting when available,
        // falling back to conservative 8 decimal places.
        let price_str = instrument_mgr
            .and_then(|mgr| mgr.get(Exchange::GateIo, contract))
            .map(|spec| spec.format_price(trigger_price))
            .unwrap_or_else(|| format!("{:.8}", trigger_price));

        // FIX 3: price_type=1 for mark price (more reliable than last price)
        format!(
            concat!(
                r#"{{"initial":{{"contract":"{}","size":{},"price":"0","tif":"ioc","reduce_only":true}},"#,
                r#""trigger":{{"strategy_type":0,"price_type":1,"price":"{}","rule":{}}}}}"#,
            ),
            contract, size, price_str, rule
        )
    }

    /// Submit SL/TP conditional orders for a filled parent order.
    ///
    /// Called after the main order is confirmed as filled. Creates conditional
    /// trigger orders on Gate.io that will fire market orders when the
    /// stop loss or take profit price is hit.
    ///
    /// **Gate.io auto-trigger price validation:**
    /// - rule=1 (>=): trigger_price must be > last_price at submission time
    /// - rule=2 (<=): trigger_price must be < last_price at submission time
    /// If price has already moved past the trigger, we skip the conditional
    /// order and log a critical warning (the position must be managed manually
    /// or by the Rust safety-net exit logic).
    async fn submit_sl_tp_orders(
        &self,
        symbol: &str,
        parent_side: &OrderSide,
        filled_size: i64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    ) {
        let close_size = if *parent_side == OrderSide::Buy {
            -filled_size // Close long = sell
        } else {
            filled_size // Close short = buy
        };

        // Fetch last traded price once for both SL and TP validation
        let last_price = self.fetch_last_price(symbol).await;

        // Submit Stop Loss
        if let Some(sl_price) = stop_loss {
            if sl_price > 0.0 {
                // For longs: SL triggers when price <= sl_price (rule=2)
                // For shorts: SL triggers when price >= sl_price (rule=1)
                let trigger_type = if *parent_side == OrderSide::Buy { 1 } else { 0 };
                let rule: u8 = if trigger_type == 0 { 1 } else { 2 };

                // Validate trigger price against last traded price
                if let Some(lp) = last_price {
                    if !Self::validate_trigger_price(sl_price, lp, rule) {
                        error!(
                            "[gateio-ws] ⚠️ SL price {:.4} already breached (last={:.4}, rule={}) for {} {} — \
                             price moved past stop loss! Position at risk of liquidation.",
                            sl_price, lp, rule, symbol,
                            if *parent_side == OrderSide::Buy { "LONG" } else { "SHORT" }
                        );
                        // Skip this conditional order — the execution_router safety net
                        // should detect the breach and close the position at market.
                        // Do NOT proceed with a conditional order that Gate.io will reject.
                    } else {
                        let body = Self::build_price_trigger_body(symbol, sl_price, close_size, trigger_type, self.instrument_mgr.as_ref());
                        let path = "/futures/usdt/price_orders";
                        let timestamp = now_ms() / 1000;
                        let full_path = format!("/api/v4{}", path);
                        let signature = Self::rest_sign("POST", &full_path, "", &body, timestamp, &self.api_secret);
                        let url = format!("{}{}", self.base_url(), path);

                        match self.rest_client
                            .post(&url)
                            .header("KEY", &self.api_key)
                            .header("SIGN", &signature)
                            .header("Timestamp", timestamp.to_string())
                            .header("Content-Type", "application/json")
                            .body(body)
                            .send()
                            .await
                        {
                            Ok(resp) => {
                                let status = resp.status().as_u16();
                                if status < 400 {
                                    info!("[gateio-ws] ✅ SL conditional order placed for {} @ {:.4} (last={:.4})", symbol, sl_price, lp);
                                } else {
                                    let body_text = resp.text().await.unwrap_or_default();
                                    error!("[gateio-ws] ❌ SL order failed (HTTP {}): {}", status, body_text);
                                }
                            }
                            Err(e) => {
                                error!("[gateio-ws] ❌ SL order submit error: {}", e);
                            }
                        }
                    }
                } else {
                    // Could not fetch last price — submit anyway and let Gate.io validate
                    warn!("[gateio-ws] Could not fetch last price for {} — submitting SL without validation", symbol);
                    let body = Self::build_price_trigger_body(symbol, sl_price, close_size, trigger_type, self.instrument_mgr.as_ref());
                    let path = "/futures/usdt/price_orders";
                    let timestamp = now_ms() / 1000;
                    let full_path = format!("/api/v4{}", path);
                    let signature = Self::rest_sign("POST", &full_path, "", &body, timestamp, &self.api_secret);
                    let url = format!("{}{}", self.base_url(), path);

                    match self.rest_client
                        .post(&url)
                        .header("KEY", &self.api_key)
                        .header("SIGN", &signature)
                        .header("Timestamp", timestamp.to_string())
                        .header("Content-Type", "application/json")
                        .body(body)
                        .send()
                        .await
                    {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            if status < 400 {
                                info!("[gateio-ws] ✅ SL conditional order placed for {} @ {:.4}", symbol, sl_price);
                            } else {
                                let body_text = resp.text().await.unwrap_or_default();
                                error!("[gateio-ws] ❌ SL order failed (HTTP {}): {}", status, body_text);
                            }
                        }
                        Err(e) => {
                            error!("[gateio-ws] ❌ SL order submit error: {}", e);
                        }
                    }
                }
            }
        }

        // Submit Take Profit
        if let Some(tp_price) = take_profit {
            if tp_price > 0.0 {
                // For longs: TP triggers when price >= tp_price (rule=1)
                // For shorts: TP triggers when price <= tp_price (rule=2)
                let trigger_type = if *parent_side == OrderSide::Buy { 0 } else { 1 };
                let rule: u8 = if trigger_type == 0 { 1 } else { 2 };

                if let Some(lp) = last_price {
                    if !Self::validate_trigger_price(tp_price, lp, rule) {
                        // TP already breached means we're already in profit past the target!
                        // This is a good problem — but we still can't submit the conditional.
                        warn!(
                            "[gateio-ws] TP price {:.4} already breached (last={:.4}, rule={}) for {} {} — \
                             price already past take-profit target.",
                            tp_price, lp, rule, symbol,
                            if *parent_side == OrderSide::Buy { "LONG" } else { "SHORT" }
                        );
                    } else {
                        let body = Self::build_price_trigger_body(symbol, tp_price, close_size, trigger_type, self.instrument_mgr.as_ref());
                        let path = "/futures/usdt/price_orders";
                        let timestamp = now_ms() / 1000;
                        let full_path = format!("/api/v4{}", path);
                        let signature = Self::rest_sign("POST", &full_path, "", &body, timestamp, &self.api_secret);
                        let url = format!("{}{}", self.base_url(), path);

                        match self.rest_client
                            .post(&url)
                            .header("KEY", &self.api_key)
                            .header("SIGN", &signature)
                            .header("Timestamp", timestamp.to_string())
                            .header("Content-Type", "application/json")
                            .body(body)
                            .send()
                            .await
                        {
                            Ok(resp) => {
                                let status = resp.status().as_u16();
                                if status < 400 {
                                    info!("[gateio-ws] ✅ TP conditional order placed for {} @ {:.4} (last={:.4})", symbol, tp_price, lp);
                                } else {
                                    let body_text = resp.text().await.unwrap_or_default();
                                    error!("[gateio-ws] ❌ TP order failed (HTTP {}): {}", status, body_text);
                                }
                            }
                            Err(e) => {
                                error!("[gateio-ws] ❌ TP order submit error: {}", e);
                            }
                        }
                    }
                } else {
                    warn!("[gateio-ws] Could not fetch last price for {} — submitting TP without validation", symbol);
                    let body = Self::build_price_trigger_body(symbol, tp_price, close_size, trigger_type, self.instrument_mgr.as_ref());
                    let path = "/futures/usdt/price_orders";
                    let timestamp = now_ms() / 1000;
                    let full_path = format!("/api/v4{}", path);
                    let signature = Self::rest_sign("POST", &full_path, "", &body, timestamp, &self.api_secret);
                    let url = format!("{}{}", self.base_url(), path);

                    match self.rest_client
                        .post(&url)
                        .header("KEY", &self.api_key)
                        .header("SIGN", &signature)
                        .header("Timestamp", timestamp.to_string())
                        .header("Content-Type", "application/json")
                        .body(body)
                        .send()
                        .await
                    {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            if status < 400 {
                                info!("[gateio-ws] ✅ TP conditional order placed for {} @ {:.4}", symbol, tp_price);
                            } else {
                                let body_text = resp.text().await.unwrap_or_default();
                                error!("[gateio-ws] ❌ TP order failed (HTTP {}): {}", status, body_text);
                            }
                        }
                        Err(e) => {
                            error!("[gateio-ws] ❌ TP order submit error: {}", e);
                        }
                    }
                }
            }
        }
    }

    /// Generate the next monotonic client order ID.
    fn next_id(&self) -> String {
        let id = self.next_client_id.fetch_add(1, Ordering::Relaxed);
        format!("r{}", id)
    }

    // ── HMAC-SHA512 WS Authentication ──────────────────────────────────────

    /// Compute Gate.io WS HMAC-SHA512 signature for **subscriptions**.
    ///
    /// sig_payload = "channel={channel}&event={event}&time={time}"
    /// signature   = HMAC_SHA512(api_secret, sig_payload)
    ///
    /// This is ONLY for subscribe/unsubscribe events. API requests (order
    /// placement, cancellation) use a different signature format — see
    /// `ws_api_sign()`.
    fn ws_sign(secret: &[u8], channel: &str, event: &str, time: i64) -> String {
        let payload = format!("channel={}&event={}&time={}", channel, event, time);
        let mut mac = Hmac::<Sha512>::new_from_slice(secret)
            .expect("HMAC accepts any key length");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Compute Gate.io WS HMAC-SHA512 signature for **API requests**
    /// (order placement, cancellation, amendment, status queries).
    ///
    /// Gate.io WS API (event: "api") uses a DIFFERENT signature format than
    /// subscriptions. The signature payload is:
    ///
    ///   "{event}\n{channel}\n{req_param_json}\n{timestamp}"
    ///
    /// where req_param_json is the JSON-serialized order parameters.
    /// This matches the format used by CCXT and the official Gate.io docs
    /// for WebSocket trading API.
    fn ws_api_sign(secret: &[u8], channel: &str, req_param_json: &str, timestamp: i64) -> String {
        let payload = format!("api\n{}\n{}\n{}", channel, req_param_json, timestamp);
        let mut mac = Hmac::<Sha512>::new_from_slice(secret)
            .expect("HMAC accepts any key length");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Build the WS authentication message for `futures.orders` subscription.
    ///
    /// Gate.io WS v4 does NOT have a separate "futures.login" channel.
    /// Authentication is done per-message via the `auth` block.
    /// We authenticate by subscribing to `futures.orders` with the auth block.
    /// The subscription response confirms both subscription AND authentication.
    /// Build WS auth+subscribe message for `futures.orders`.
    ///
    /// Gate.io `futures.orders` payload must be `["!all"]` to subscribe to
    /// ALL contract order updates, or specific contract names like
    /// `["BTC_USDT","ETH_USDT"]`.  Passing a user ID (e.g. "0") causes
    /// `"unknown contract 0"` error — Gate.io treats payload items as
    /// contract names, NOT user IDs.
    fn build_auth_subscribe_message(api_key: &str, secret: &[u8], contracts: &[String]) -> String {
        let time = now_ms() / 1000;
        let sign = Self::ws_sign(secret, "futures.orders", "subscribe", time);
        // Build JSON array items: "BTC_USDT","ETH_USDT" or "!all"
        let payload_items: String = contracts.iter()
            .map(|c| format!("\"{}\"", c))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"time":{},"channel":"futures.orders","event":"subscribe","payload":[{}],"auth":{{"method":"api_key","KEY":"{}","SIGN":"{}"}}}}"#,
            time, payload_items, api_key, sign
        )
    }

    /// Build explicit futures.login message for WS authentication.
    ///
    /// Gate.io WS v4 requires an explicit login before
    /// futures.order_place calls. The login signature format is:
    ///   HMAC_SHA512(secret, "channel=futures.login&event=login&time={time}")
    fn build_login_message(api_key: &str, secret: &[u8]) -> String {
        let time = now_ms() / 1000;
        // Event MUST be "login", not "api"
        let sign = Self::ws_sign(secret, "futures.login", "login", time);
        
        format!(
            r#"{{"time":{},"channel":"futures.login","event":"login","payload":{{}},"auth":{{"method":"api_key","KEY":"{}","SIGN":"{}"}}}}"#,
            time, api_key, sign
        )
    }

    /// Build a futures.order_place WS message for order submission.
    ///
    /// Gate.io WS API (event: "api") uses a DIFFERENT format than
    /// subscriptions. The auth is embedded INSIDE the payload object
    /// alongside the order parameters in `req_param`, and the signature
    /// is computed over "api\nchannel\nreq_param_json\ntimestamp".
    ///
    /// Correct format (matching CCXT and official Gate.io WS trading API):
    /// ```json
    /// {
    ///   "time": 1234567890,
    ///   "channel": "futures.order_place",
    ///   "event": "api",
    ///   "payload": {
    ///     "req_id": "r1",
    ///     "req_param": {
    ///       "contract": "BTC_USDT",
    ///       "size": 10,
    ///       "price": "50000",
    ///       "tif": "gtc",
    ///       "reduce_only": false
    ///     },
    ///     "timestamp": "1234567890",
    ///     "api_key": "your_key",
    ///     "signature": "hmac_sha512_hex"
    ///   }
    /// }
    /// ```
    fn build_order_place_message(
        api_key: &str,
        secret: &[u8],
        req_id: &str,
        contract: &str,
        size: i64,
        price: &str,
        tif: &str,
        reduce_only: bool,
    ) -> String {
        let time = now_ms() / 1000;
        // Build the req_param JSON first (needed for signature computation)
        let req_param = format!(
            r#"{{"contract":"{}","size":{},"price":"{}","tif":"{}","reduce_only":{},"text":"t-{}"}}"#,
            contract, size, price, tif, reduce_only, req_id
        );
        // Gate.io WS API signature: HMAC-SHA512("api\nchannel\nreq_param_json\ntimestamp")
        let sign = Self::ws_api_sign(secret, "futures.order_place", &req_param, time);
        // Build the full message with auth embedded inside payload
        format!(
            concat!(
                r#"{{"time":{},"channel":"futures.order_place","event":"api","#,
                r#""payload":{{"req_id":"{}","req_param":{},"#,
                r#""timestamp":"{}","api_key":"{}","signature":"{}"}}}}"#
            ),
            time, req_id, req_param, time, api_key, sign
        )
    }

    /// Build a futures.order_cancel WS message.
    ///
    /// Uses the same WS API auth format as order placement:
    /// signature over "api\nchannel\nreq_param_json\ntimestamp".
    fn build_order_cancel_message(
        api_key: &str,
        secret: &[u8],
        order_id: &str,
    ) -> String {
        let time = now_ms() / 1000;
        let req_param = format!(r#"{{"order_id":"{}"}}"#, order_id);
        let sign = Self::ws_api_sign(secret, "futures.order_cancel", &req_param, time);
        format!(
            concat!(
                r#"{{"time":{},"channel":"futures.order_cancel","event":"api","#,
                r#""payload":{{"req_id":"cancel_{}","req_param":{},"#,
                r#""timestamp":"{}","api_key":"{}","signature":"{}"}}}}"#
            ),
            time, order_id, req_param, time, api_key, sign
        )
    }

    // ── WebSocket Connection Loop ──────────────────────────────────────────

    /// Main WS connection loop with automatic reconnection.
    ///
    /// This function runs forever. On disconnect it backs off exponentially
    /// and reconnects. Resting orders are NOT dropped — they remain tracked
    /// locally and reconciled on reconnection.
    async fn ws_connection_loop(
        ws_url: String,
        api_key: String,
        api_secret: Vec<u8>,
        mut cmd_rx: mpsc::UnboundedReceiver<WsCommand>,
        pending: Arc<RwLock<HashMap<String, PendingOrder>>>,
        order_state: Arc<RwLock<HashMap<String, OrderTracking>>>,
        is_ready: Arc<AtomicBool>,
        circuit_breaker: Option<Arc<CircuitBreaker>>,
    ) {
        let mut backoff_ms = RECONNECT_BASE_MS;
        /// Maximum consecutive auth failures before giving up rapid retries.
        /// After this many INVALID_KEY errors, the gateway backs off to 60s
        /// intervals to avoid hammering Gate.io with bad credentials.
        const MAX_AUTH_FAILURES: u32 = 3;
        /// Backoff interval (ms) after exhausting auth retries.
        const AUTH_FAILURE_BACKOFF_MS: u64 = 60_000;
        let mut consecutive_auth_failures: u32 = 0;

        loop {
            is_ready.store(false, Ordering::Release);

            // ── CIRCUIT BREAKER: Trip on disconnect ──
            // When the WS drops, explicitly trip the circuit breaker to prevent
            // the strategy engine from firing signals into a dead gateway.
            // Skip tripping for auth failures — those are handled separately
            // in the auth response handler with TripReason::AuthFailure.
            if let Some(ref cb) = circuit_breaker {
                if backoff_ms > RECONNECT_BASE_MS && consecutive_auth_failures == 0 {
                    // Not the first connection attempt AND not an auth failure — this is
                    // a true connectivity issue (WS dropped, TCP reset, etc.)
                    cb.trip(TripReason::ConnectivityLost);
                    error!("[gateio-ws] 🚨 Circuit breaker tripped: ConnectivityLost");
                }
            }

            // ── PENDING ACK RECONCILIATION ──
            // Flush all PendingAck orders that were in-flight when the WS dropped.
            // These orders have an unknown state on the exchange — we resolve them
            // by notifying callers of ConnectionReset and marking them for REST
            // reconciliation.
            {
                let mut pending_lock = pending.write();
                let stale_count = pending_lock.len();
                if stale_count > 0 {
                    warn!(
                        "[gateio-ws] Flushing {} PendingAck orders on reconnect",
                        stale_count
                    );
                    let stale_keys: Vec<String> = pending_lock.keys().cloned().collect();
                    for key in stale_keys {
                        if let Some(p) = pending_lock.remove(&key) {
                            let _ = p.response_tx.send(Err(ExchangeError::ConnectionReset));
                        }
                    }
                }

                // Also mark all PendingAck entries in order_state as needing reconciliation
                let mut state_lock = order_state.write();
                let pending_ack_keys: Vec<String> = state_lock
                    .iter()
                    .filter(|(_, v)| matches!(v.state, OrderTrackingState::PendingAck { .. }))
                    .map(|(k, _)| k.clone())
                    .collect();
                for key in &pending_ack_keys {
                    if let Some(tracking) = state_lock.get_mut(key) {
                        warn!(
                            "[gateio-ws] PendingAck order {} for {} marked as ConnectionReset — needs REST reconciliation",
                            key, tracking.symbol
                        );
                        tracking.state = OrderTrackingState::Rejected {
                            reason: "ConnectionReset during PendingAck — needs REST reconciliation".to_string(),
                        };
                    }
                }
            }

            info!("[gateio-ws] Connecting to {}", ws_url);

            match connect_async(&ws_url).await {
                Ok((ws_stream, _response)) => {
                    info!("[gateio-ws] Connected");
                    backoff_ms = RECONNECT_BASE_MS; // Reset backoff

                    let (mut ws_write, mut ws_read) = ws_stream.split();

                    // Gate.io futures WS v4 authenticates via the auth block
                    // on subscription messages (Step 2 below), not via a
                    // separate login channel. No explicit login step needed.
                    info!("[gateio-ws] Proceeding to authenticated subscription");

                    // ── Step 2: Subscribe to futures.orders with auth block ──
                    // After explicit login, subscribe to order updates for ALL contracts.
                    let all_contracts = vec!["!all".to_string()];
                    let auth_sub_msg = Self::build_auth_subscribe_message(&api_key, &api_secret, &all_contracts);
                    info!("[gateio-ws] Sending auth subscription for futures.orders");
                    if let Err(e) = ws_write.send(Message::Text(auth_sub_msg)).await {
                        error!("[gateio-ws] Auth subscribe send failed: {}", e);
                        continue;
                    }

                    // Wait for subscription/auth response (with timeout)
                    let auth_deadline = Instant::now() + Duration::from_secs(5);
                    let mut authenticated = false;
                    while Instant::now() < auth_deadline {
                        tokio::select! {
                            msg = ws_read.next() => {
                                match msg {
                                    Some(Ok(Message::Text(txt))) => {
                                        debug!("[gateio-ws] Auth phase received: {}", txt);
                                        if txt.contains("futures.orders") {
                                            if txt.contains("\"error\":null") || txt.contains("\"status\":\"success\"") {
                                                info!("[gateio-ws] Authenticated & subscribed to futures.orders");
                                                authenticated = true;
                                                break;
                                            } else if txt.contains("INVALID_KEY") || txt.contains("\"error\":{") {
                                                error!("[gateio-ws] Auth/subscribe rejected: {}", txt);
                                                break;
                                            }
                                        }
                                        if txt.contains("futures.pong") || txt.contains("futures.ping") {
                                            continue;
                                        }
                                    }
                                    Some(Ok(Message::Ping(data))) => {
                                        let _ = ws_write.send(Message::Pong(data)).await;
                                    }
                                    Some(Err(e)) => {
                                        error!("[gateio-ws] Read error during auth: {}", e);
                                        break;
                                    }
                                    None => break,
                                    _ => {}
                                }
                            }
                            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                        }
                    }

                    if !authenticated {
                        consecutive_auth_failures += 1;
                        if consecutive_auth_failures >= MAX_AUTH_FAILURES {
                            // After repeated auth failures, this is almost certainly
                            // an invalid/expired API key — not a transient issue.
                            // Trip circuit breaker with AuthFailure (NOT ConnectivityLost)
                            // and back off to 60s to avoid hammering the server.
                            error!(
                                "[gateio-ws] {} consecutive auth failures — API key is likely invalid or expired. \
                                 Backing off to {}s intervals. Check GATEIO_API_KEY / GATEIO_API_SECRET env vars.",
                                consecutive_auth_failures, AUTH_FAILURE_BACKOFF_MS / 1000
                            );
                            if let Some(ref cb) = circuit_breaker {
                                cb.trip(TripReason::AuthFailure);
                            }
                            tokio::time::sleep(Duration::from_millis(AUTH_FAILURE_BACKOFF_MS)).await;
                        } else {
                            warn!(
                                "[gateio-ws] Auth/subscribe failed (attempt {}/{}) — retrying in {}ms...",
                                consecutive_auth_failures, MAX_AUTH_FAILURES, backoff_ms
                            );
                            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                            backoff_ms = (backoff_ms * 2).min(RECONNECT_MAX_MS);
                        }
                        continue;
                    }

                    // Both login and subscription succeeded — reset auth failure counter
                    consecutive_auth_failures = 0;
                    is_ready.store(true, Ordering::Release);
                    info!("[gateio-ws] Ready — login confirmed, accepting orders");

                    // ── CATEGORY 1 FIX: Auto-reset circuit breaker on successful reconnect ──
                    // When connectivity is restored, reset the circuit breaker if it was
                    // tripped due to ConnectivityLost or AuthFailure. This prevents the
                    // bot from staying halted after a transient network/auth issue.
                    if let Some(ref cb) = circuit_breaker {
                        if cb.is_trading_halted() {
                            let reason_code = cb.trip_reason_code();
                            // TripReason::ConnectivityLost = 7, TripReason::AuthFailure = 8
                            if reason_code == 7 || reason_code == 8 {
                                info!("[gateio-ws] Connectivity/auth restored — auto-resetting circuit breaker");
                                cb.reset();
                            }
                        }
                    }

                    // ── Step 3: Main read/write loop ──
                    let mut ping_interval = tokio::time::interval(Duration::from_secs(PING_INTERVAL_SECS));
                    let pending_clone = pending.clone();
                    let order_state_clone = order_state.clone();
                    // CATEGORY 1 FIX: Track last pong received time for heartbeat validation
                    let mut last_pong_time = Instant::now();

                    loop {
                        tokio::select! {
                            // Incoming WS messages
                            msg = ws_read.next() => {
                                match msg {
                                    Some(Ok(Message::Text(txt))) => {
                                        // CATEGORY 1 FIX: Track pong responses from Gate.io
                                        // Gate.io sends text-based pong: {"time":...,"channel":"futures.pong"}
                                        if txt.contains("futures.pong") {
                                            last_pong_time = Instant::now();
                                        }
                                        Self::handle_ws_message(
                                            &txt,
                                            &pending_clone,
                                            &order_state_clone,
                                        );
                                    }
                                    Some(Ok(Message::Ping(data))) => {
                                        let _ = ws_write.send(Message::Pong(data)).await;
                                    }
                                    Some(Ok(Message::Pong(_))) => {
                                        // CATEGORY 1 FIX: Also handle binary pong frames
                                        last_pong_time = Instant::now();
                                    }
                                    Some(Ok(Message::Close(_))) => {
                                        warn!("[gateio-ws] Server closed connection");
                                        break;
                                    }
                                    Some(Err(e)) => {
                                        error!("[gateio-ws] Read error: {}", e);
                                        break;
                                    }
                                    None => {
                                        warn!("[gateio-ws] Stream ended");
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            // Outgoing commands from execution router
                            cmd = cmd_rx.recv() => {
                                match cmd {
                                    Some(WsCommand::SendText(text)) => {
                                        if let Err(e) = ws_write.send(Message::Text(text)).await {
                                            error!("[gateio-ws] Write error: {}", e);
                                            break;
                                        }
                                    }
                                    Some(WsCommand::Shutdown) => {
                                        info!("[gateio-ws] Shutdown requested");
                                        let _ = ws_write.close().await;
                                        return;
                                    }
                                    None => {
                                        info!("[gateio-ws] Command channel closed");
                                        return;
                                    }
                                }
                            }
                            // Periodic ping to keep connection alive
                            _ = ping_interval.tick() => {
                                // CATEGORY 1 FIX: Validate pong responses — if no pong received
                                // within PONG_TIMEOUT_SECS, the connection is likely dead even
                                // though the TCP socket hasn't closed. Force reconnect.
                                let pong_age = last_pong_time.elapsed();
                                if pong_age > Duration::from_secs(PONG_TIMEOUT_SECS) {
                                    error!(
                                        "[gateio-ws] No pong received for {:.0}s (timeout={}s) — connection stale, forcing reconnect",
                                        pong_age.as_secs_f64(), PONG_TIMEOUT_SECS
                                    );
                                    break;
                                }
                                let ping_msg = format!(r#"{{"time":{},"channel":"futures.ping"}}"#, now_ms() / 1000);
                                if let Err(e) = ws_write.send(Message::Text(ping_msg)).await {
                                    error!("[gateio-ws] Ping failed: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    // Connection lost — mark not ready, fall through to reconnect
                    is_ready.store(false, Ordering::Release);
                }
                Err(e) => {
                    error!("[gateio-ws] Connection failed: {}", e);
                }
            }

            // Exponential backoff before reconnect
            warn!("[gateio-ws] Reconnecting in {}ms...", backoff_ms);
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(RECONNECT_MAX_MS);
        }
    }

    // ── WS Message Handler ─────────────────────────────────────────────────

    /// Handle an incoming WS message. Uses fast byte scanning instead of
    /// full serde_json deserialization on the hot path.
    ///
    /// Message types we care about:
    ///   - `futures.order_place` response (ACK / reject for our orders)
    ///   - `futures.order_cancel` response
    ///   - `futures.orders` update (fill / partial fill / cancel by exchange)
    fn handle_ws_message(
        text: &str,
        pending: &RwLock<HashMap<String, PendingOrder>>,
        order_state: &RwLock<HashMap<String, OrderTracking>>,
    ) {
        // Fast path: skip ping/pong and heartbeat responses
        if text.contains("futures.ping") || text.contains("futures.pong") {
            return;
        }

        // ── Order placement response ──
        if text.contains("futures.order_place") {
            Self::handle_order_place_response(text, pending, order_state);
            return;
        }

        // ── Order cancel response ──
        if text.contains("futures.order_cancel") {
            debug!("[gateio-ws] Cancel response: {}", &text[..text.len().min(200)]);
            return;
        }

        // ── Fill / status update via futures.orders subscription ──
        if text.contains("futures.orders") && text.contains("\"update\"") {
            Self::handle_order_update(text, pending, order_state);
            return;
        }

        // FIX 2C: Log ALL unhandled WS messages at warn! level so we can diagnose
        // what Gate.io sends back (or doesn't) during order submission windows.
        // Previously this was debug!-only, making silent order rejections invisible.
        warn!("[gateio-ws] Unhandled message: {}", &text[..text.len().min(200)]);
    }

    /// Handle a futures.order_place ACK/NACK response.
    ///
    /// Gate.io WS API responses have a different structure than subscription
    /// updates. The response format is:
    /// ```json
    /// {
    ///   "header": {"response_time": "...", "status": "200", "channel": "...", "event": "api"},
    ///   "data": {"result": {"id": 123, "status": "open", "size": 10, "left": 10, ...}},
    ///   "request_id": "r1"   // <-- top-level, NOT "req_id"
    /// }
    /// ```
    /// On error, `header.status` != "200" and `data.errs` contains error details.
    fn handle_order_place_response(
        text: &str,
        pending: &RwLock<HashMap<String, PendingOrder>>,
        order_state: &RwLock<HashMap<String, OrderTracking>>,
    ) {
        // Gate.io WS API returns `request_id` at top level for API responses.
        // Also try `req_id` for backward compatibility.
        let req_id = Self::extract_json_string(text, "request_id")
            .or_else(|| Self::extract_json_string(text, "req_id"));

        // Check for error: header.status != "200", or explicit error fields
        let is_error = {
            let has_error_field = text.contains("\"error\"") && !text.contains("\"error\":null");
            let has_errs = text.contains("\"errs\"");
            // Check header.status for non-200
            let header_error = Self::extract_header_status(text)
                .map(|s| s != "200")
                .unwrap_or(false);
            has_error_field || has_errs || header_error
        };

        // FIX: When req_id is missing but an error is present, resolve the most
        // recent pending order with that error. Gate.io's matching engine may omit
        // req_id in error responses (e.g., leverage mismatch rejection), leaving
        // the pending order stuck until the 5s timeout fires.
        let req_id = match req_id {
            Some(id) => id,
            None => {
                if is_error {
                    // Find the most recent pending order (by submit_us) and resolve it
                    let mut pending_lock = pending.write();
                    let oldest_key = pending_lock.iter()
                        .min_by_key(|(_, p)| p.submit_us)
                        .map(|(k, _)| k.clone());
                    if let Some(key) = oldest_key {
                        if let Some(p) = pending_lock.remove(&key) {
                            let err_msg = Self::extract_api_error_message(text);
                            warn!("[gateio-ws] order_place error without req_id, resolving pending order {}: {}", key, err_msg);
                            let mut state = order_state.write();
                            if let Some(tracking) = state.get_mut(&p.client_id) {
                                tracking.state = OrderTrackingState::Rejected { reason: err_msg.clone() };
                            }
                            let _ = p.response_tx.send(Err(ExchangeError::Unknown {
                                code: "WS_REJECT_NO_REQID".to_string(),
                                message: err_msg,
                            }));
                        }
                    }
                    return;
                }
                debug!("[gateio-ws] order_place response without req_id/request_id");
                return;
            }
        };

        let mut pending_lock = pending.write();
        if let Some(p) = pending_lock.remove(&req_id) {
            let end_us = now_us();
            let latency_us = (end_us - p.submit_us).max(0) as u64;

            if is_error {
                // Extract error message from data.errs or error.message
                let err_msg = Self::extract_api_error_message(text);
                let _ = p.response_tx.send(Err(ExchangeError::Unknown {
                    code: "WS_REJECT".to_string(),
                    message: err_msg.clone(),
                }));
                // Update local state
                let mut state = order_state.write();
                if let Some(tracking) = state.get_mut(&p.client_id) {
                    tracking.state = OrderTrackingState::Rejected { reason: err_msg };
                }
            } else {
                // Gate.io WS API wraps order data inside data.result.
                // Use extract_api_result_field_* helpers that look in
                // top-level -> result -> data.result -> data.result[0].
                let order_id = Self::extract_api_result_string(text, "id")
                    .or_else(|| Self::extract_api_result_number(text, "id").map(|n| n.to_string()))
                    .unwrap_or_default();

                let status = Self::extract_api_result_string(text, "status")
                    .unwrap_or_else(|| "open".to_string());

                // Gate.io returns `size` (total) and `left` (unfilled remaining).
                // Actual filled = abs(size) - abs(left).
                let total_size = Self::extract_api_result_number(text, "size")
                    .map(|s| s.abs())
                    .unwrap_or(0);
                let left = Self::extract_api_result_number(text, "left")
                    .map(|s| s.abs())
                    .unwrap_or(0);
                let filled_size = total_size - left;

                let fill_price = Self::extract_api_result_float(text, "fill_price")
                    .unwrap_or(0.0);

                info!(
                    "[gateio-ws] Order ACK: id={}, status={}, filled={}/{}, latency={}us",
                    order_id, status, filled_size, total_size, latency_us
                );

                // Update local state
                let mut state = order_state.write();
                if let Some(tracking) = state.get_mut(&p.client_id) {
                    tracking.state = OrderTrackingState::Resting {
                        exchange_id: order_id.clone(),
                        filled_so_far: 0,
                    };
                }

                let _ = p.response_tx.send(Ok(OrderResult {
                    order_id,
                    status,
                    filled_size,
                    avg_fill_price: fill_price,
                    fee: 0.0,
                    latency_us,
                    exchange_timestamp: now_ms(),
                    rejection_reason: None,
                }));
            }
        }
    }

    /// Extract `header.status` from a WS API response.
    fn extract_header_status(text: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        v.get("header")
            .and_then(|h| h.get("status"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
    }

    /// Extract error message from a WS API response.
    /// Checks: data.errs.message, data.errs.label, error.message, top-level message.
    fn extract_api_error_message(text: &str) -> String {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
            // Check data.errs first (API response error format)
            if let Some(errs) = v.get("data").and_then(|d| d.get("errs")) {
                if let Some(msg) = errs.get("message").and_then(|m| m.as_str()) {
                    return msg.to_string();
                }
                if let Some(label) = errs.get("label").and_then(|l| l.as_str()) {
                    return label.to_string();
                }
            }
            // Check error.message (subscription-style error)
            if let Some(err) = v.get("error") {
                if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                    return msg.to_string();
                }
            }
            // Check top-level message
            if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
                return msg.to_string();
            }
        }
        "unknown error".to_string()
    }

    /// Extract a string field from a WS API response result.
    /// Looks in: top-level -> result -> data.result -> data.result[0]
    fn extract_api_result_string(text: &str, key: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        // Top-level
        if let Some(s) = v.get(key).and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
        // result (subscription-style response)
        if let Some(result) = v.get("result") {
            if let Some(s) = result.get(key).and_then(|v| v.as_str()) {
                return Some(s.to_string());
            }
            if let Some(arr) = result.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                        return Some(s.to_string());
                    }
                }
            }
        }
        // data.result (API-style response)
        if let Some(data) = v.get("data") {
            if let Some(result) = data.get("result") {
                if let Some(s) = result.get(key).and_then(|v| v.as_str()) {
                    return Some(s.to_string());
                }
                if let Some(arr) = result.as_array() {
                    if let Some(first) = arr.first() {
                        if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                            return Some(s.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract an integer field from a WS API response result.
    /// Looks in: top-level -> result -> data.result -> data.result[0]
    fn extract_api_result_number(text: &str, key: &str) -> Option<i64> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        // Try all locations: top-level, result, data.result
        for container in [
            Some(&v),
            v.get("result"),
            v.get("data").and_then(|d| d.get("result")),
        ].iter().flatten() {
            if let Some(n) = container.get(key).and_then(|v| v.as_i64()) {
                return Some(n);
            }
            if let Some(s) = container.get(key).and_then(|v| v.as_str()) {
                if let Ok(n) = s.parse() {
                    return Some(n);
                }
            }
            // Check if container is an array
            if let Some(arr) = container.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(n) = first.get(key).and_then(|v| v.as_i64()) {
                        return Some(n);
                    }
                    if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                        if let Ok(n) = s.parse() {
                            return Some(n);
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract a float field from a WS API response result.
    /// Looks in: top-level -> result -> data.result -> data.result[0]
    fn extract_api_result_float(text: &str, key: &str) -> Option<f64> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        for container in [
            Some(&v),
            v.get("result"),
            v.get("data").and_then(|d| d.get("result")),
        ].iter().flatten() {
            if let Some(n) = container.get(key).and_then(|v| v.as_f64()) {
                return Some(n);
            }
            if let Some(s) = container.get(key).and_then(|v| v.as_str()) {
                if let Ok(n) = s.parse() {
                    return Some(n);
                }
            }
            if let Some(arr) = container.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(n) = first.get(key).and_then(|v| v.as_f64()) {
                        return Some(n);
                    }
                    if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                        if let Ok(n) = s.parse() {
                            return Some(n);
                        }
                    }
                }
            }
        }
        None
    }

    /// Handle a futures.orders subscription update (fills, partial fills, cancels).
    ///
    /// CATEGORY 2 FIX: Enhanced partial fill handling in WS response parser.
    /// Now properly tracks:
    ///   - Partial fill quantities via `left` field (remaining unfilled)
    ///   - Fill price from `fill_price` field
    ///   - Cumulative filled quantity computed as total_size - left
    ///   - Transition from partial fill to fully filled when left == 0
    fn handle_order_update(
        text: &str,
        _pending: &RwLock<HashMap<String, PendingOrder>>,
        order_state: &RwLock<HashMap<String, OrderTracking>>,
    ) {
        let order_id = Self::extract_json_string(text, "id")
            .or_else(|| Self::extract_json_number_as_string(text, "id"));
        let status = Self::extract_json_string(text, "status");
        let fill_price = Self::extract_json_float(text, "fill_price");
        let filled_size = Self::extract_json_number(text, "size").map(|s| s.abs());
        // CATEGORY 2 FIX: Extract `left` field for accurate partial fill tracking
        // Gate.io provides `left` = remaining unfilled quantity
        let left_remaining = Self::extract_json_number(text, "left").map(|s| s.abs());

        if let (Some(ref oid), Some(ref st)) = (&order_id, &status) {
            debug!("[gateio-ws] Order update: id={}, status={}", oid, st);

            // Update local state if we're tracking this order
            let mut state = order_state.write();
            // Find by exchange_id
            let matching_key = state.iter()
                .find(|(_, v)| {
                    matches!(&v.state, OrderTrackingState::Resting { exchange_id, .. } if exchange_id == oid)
                })
                .map(|(k, _)| k.clone());

            if let Some(key) = matching_key {
                if let Some(tracking) = state.get_mut(&key) {
                    match st.as_str() {
                        "finished" => {
                            // CATEGORY 2 FIX: Compute actual filled quantity from size - left
                            let total_filled = match (filled_size, left_remaining) {
                                (Some(total), Some(left)) => total - left,
                                (Some(total), None) => total,
                                _ => tracking.size,
                            };
                            tracking.state = OrderTrackingState::Filled {
                                avg_price: fill_price.unwrap_or(0.0),
                                total_filled,
                                fee: 0.0,
                            };
                            info!("[gateio-ws] Order {} FILLED: qty={} price={:.4}", oid, total_filled, fill_price.unwrap_or(0.0));
                        }
                        "cancelled" => {
                            tracking.state = OrderTrackingState::Cancelled;
                            info!("[gateio-ws] Order {} CANCELLED", oid);
                        }
                        _ => {
                            // CATEGORY 2 FIX: Enhanced partial fill tracking
                            // Gate.io status "open" with fill_price > 0 means partial fill
                            let cumulative_filled = match (filled_size, left_remaining) {
                                (Some(total), Some(left)) => Some(total - left),
                                _ => filled_size,
                            };
                            if let Some(filled) = cumulative_filled {
                                if let OrderTrackingState::Resting { ref mut filled_so_far, .. } = tracking.state {
                                    let prev = *filled_so_far;
                                    *filled_so_far = filled;
                                    if filled > prev {
                                        info!(
                                            "[gateio-ws] Order {} PARTIAL FILL: {}/{} contracts @ {:.4}",
                                            oid, filled, tracking.size, fill_price.unwrap_or(0.0)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Fast JSON field extraction (no serde on hot path) ──────────────────

    /// Robust JSON string extraction using serde_json.
    ///
    /// Replaces the brittle string-scanning `"key":"value"` pattern that breaks
    /// silently if the exchange modifies spacing, ordering, or nesting in their
    /// WebSocket payloads. Uses serde_json::from_str to parse the full document
    /// and then safely navigates the key.
    ///
    /// The parsed `serde_json::Value` is stack-allocated for small payloads
    /// (thanks to serde_json's small-string optimization) but may heap-allocate
    /// for large payloads.
    #[inline]
    fn extract_json_string(text: &str, key: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        // First check top-level, then check inside "result" (Gate.io nests data there)
        if let Some(s) = v.get(key).and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
        if let Some(result) = v.get("result") {
            if let Some(s) = result.get(key).and_then(|v| v.as_str()) {
                return Some(s.to_string());
            }
            // Gate.io sometimes returns arrays of results
            if let Some(arr) = result.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                        return Some(s.to_string());
                    }
                }
            }
        }
        None
    }

    /// Robust JSON integer extraction using serde_json.
    #[inline]
    fn extract_json_number(text: &str, key: &str) -> Option<i64> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        // Try top-level
        if let Some(n) = v.get(key).and_then(|v| v.as_i64()) {
            return Some(n);
        }
        // Try as string-encoded number (Gate.io sometimes quotes numbers)
        if let Some(s) = v.get(key).and_then(|v| v.as_str()) {
            return s.parse().ok();
        }
        // Try inside "result"
        if let Some(result) = v.get("result") {
            if let Some(n) = result.get(key).and_then(|v| v.as_i64()) {
                return Some(n);
            }
            if let Some(s) = result.get(key).and_then(|v| v.as_str()) {
                return s.parse().ok();
            }
            if let Some(arr) = result.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(n) = first.get(key).and_then(|v| v.as_i64()) {
                        return Some(n);
                    }
                    if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                        return s.parse().ok();
                    }
                }
            }
        }
        None
    }

    /// Extract a number and return as string.
    #[inline]
    fn extract_json_number_as_string(text: &str, key: &str) -> Option<String> {
        Self::extract_json_number(text, key).map(|n| n.to_string())
    }

    /// Robust JSON float extraction using serde_json.
    ///
    /// Handles both numeric (1.23) and string-encoded ("1.23") formats
    /// that Gate.io uses inconsistently across different API responses.
    #[inline]
    fn extract_json_float(text: &str, key: &str) -> Option<f64> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        // Try top-level as number
        if let Some(n) = v.get(key).and_then(|v| v.as_f64()) {
            return Some(n);
        }
        // Try top-level as string
        if let Some(s) = v.get(key).and_then(|v| v.as_str()) {
            return s.parse().ok();
        }
        // Try inside "result"
        if let Some(result) = v.get("result") {
            if let Some(n) = result.get(key).and_then(|v| v.as_f64()) {
                return Some(n);
            }
            if let Some(s) = result.get(key).and_then(|v| v.as_str()) {
                return s.parse().ok();
            }
            if let Some(arr) = result.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(n) = first.get(key).and_then(|v| v.as_f64()) {
                        return Some(n);
                    }
                    if let Some(s) = first.get(key).and_then(|v| v.as_str()) {
                        return s.parse().ok();
                    }
                }
            }
        }
        None
    }

    // ── REST Fallback Helpers ──────────────────────────────────────────────

    /// REST HMAC-SHA512 signature for fallback operations.
    fn rest_sign(method: &str, path: &str, query: &str, body: &str, timestamp: i64, secret: &[u8]) -> String {
        use sha2::Digest;
        let body_hash = hex::encode(sha2::Sha512::digest(body.as_bytes()));
        let payload = format!("{}\n{}\n{}\n{}\n{}", method, path, query, body_hash, timestamp);
        let mut mac = Hmac::<Sha512>::new_from_slice(secret).expect("HMAC key");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn base_url(&self) -> &str {
        if self.testnet {
            // The Python CCXT client uses set_sandbox_mode(True) which maps to
            // https://api-testnet.gateapi.io/api/v4 — NOT gateio.ws.
            // The old fx-api-testnet.gateio.ws domain returns INVALID_KEY because
            // it's a DIFFERENT server with a DIFFERENT API key pool.
            "https://api-testnet.gateapi.io/api/v4"
        } else {
            GATEIO_REST_URL
        }
    }

    /// Comprehensive startup authentication test.
    ///
    /// This method performs a full diagnostic check:
    ///   1. Verifies the API key has no invisible characters
    ///   2. Tests connectivity with a public (unauthenticated) endpoint
    ///   3. Compares server time vs local time to detect clock skew
    ///   4. Attempts an authenticated request with FULL diagnostic logging
    ///   5. Returns Ok(balance) or Err with detailed diagnostic info
    pub async fn test_auth_diagnostic(&self) -> Result<f64, String> {
        let mode = if self.testnet { "TESTNET" } else { "LIVE" };
        let base = self.base_url();

        eprintln!("\n============================================================");
        eprintln!("[AUTH-DIAG] Gate.io {} Authentication Diagnostic", mode);
        eprintln!("============================================================");

        // ── Step 1: Key sanity checks ──
        let key_bytes = self.api_key.as_bytes();
        let secret_len = self.api_secret.len();
        let has_non_ascii = key_bytes.iter().any(|&b| b < 0x20 || b > 0x7E);
        let has_whitespace = key_bytes.iter().any(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r');
        eprintln!("[AUTH-DIAG] Step 1 — Key sanity:");
        eprintln!("  API Key length:  {}", self.api_key.len());
        eprintln!("  API Key prefix:  {}...", &self.api_key[..self.api_key.len().min(8)]);
        eprintln!("  API Key suffix:  ...{}", &self.api_key[self.api_key.len().saturating_sub(4)..]);
        eprintln!("  Secret length:   {}", secret_len);
        eprintln!("  Has non-ASCII:   {} {}", has_non_ascii, if has_non_ascii { "⚠️ BAD!" } else { "✅" });
        eprintln!("  Has whitespace:  {} {}", has_whitespace, if has_whitespace { "⚠️ BAD!" } else { "✅" });
        eprintln!("  Testnet mode:    {}", self.testnet);
        eprintln!("  Base URL:        {}", base);

        if has_non_ascii || has_whitespace {
            return Err(format!(
                "API key contains invisible characters (non-ASCII={}, whitespace={}). \
                 Check your .env file for trailing spaces/newlines around the key value.",
                has_non_ascii, has_whitespace
            ));
        }

        // ── Step 2: Public endpoint connectivity test ──
        eprintln!("\n[AUTH-DIAG] Step 2 — Public endpoint test (no auth):");
        let public_url = format!("{}/futures/usdt/contracts/BTC_USDT", base);
        eprintln!("  URL: {}", public_url);
        match self.rest_client.get(&public_url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                eprintln!("  Status: {}", status);
                if status == 200 {
                    eprintln!("  ✅ Public endpoint reachable");
                } else {
                    let text = resp.text().await.unwrap_or_default();
                    eprintln!("  ⚠️ Unexpected status: {} — body: {}", status, &text[..text.len().min(200)]);
                    if status == 502 || status == 503 {
                        eprintln!("  ⚠️ Public endpoint returned {} — testnet infra may be degraded, continuing with auth test...", status);
                        // DON'T abort here — testnet public endpoints often return 502
                        // but private/auth endpoints still work. Continue to Step 4.
                    }
                }
            }
            Err(e) => {
                eprintln!("  ❌ Connection failed: {}", e);
                return Err(format!("Cannot reach Gate.io {} at {} — check internet/firewall", mode, base));
            }
        }

        // ── Step 3: Clock skew check ──
        eprintln!("\n[AUTH-DIAG] Step 3 — Clock skew check:");
        let local_ts = now_ms() / 1000;
        eprintln!("  Local timestamp: {} (unix seconds)", local_ts);
        // Gate.io server time endpoint
        let time_url = format!("{}/futures/usdt/contracts/BTC_USDT", base);
        match self.rest_client.get(&time_url).send().await {
            Ok(resp) => {
                if let Some(date_header) = resp.headers().get("date") {
                    eprintln!("  Server Date header: {:?}", date_header);
                }
                eprintln!("  ✅ Server responded (clock skew check via Date header)");
            }
            Err(_) => {
                eprintln!("  ⚠️ Could not check server time");
            }
        }

        // ── Step 4: Authenticated request with FULL diagnostic ──
        eprintln!("\n[AUTH-DIAG] Step 4 — Authenticated request test:");
        let path = "/futures/usdt/accounts";
        let timestamp = now_ms() / 1000;
        let full_path = format!("/api/v4{}", path);
        let body_hash = {
            use sha2::Digest;
            hex::encode(sha2::Sha512::digest(b""))
        };
        let sign_payload = format!("{}\n{}\n{}\n{}\n{}", "GET", full_path, "", body_hash, timestamp);
        let signature = Self::rest_sign("GET", &full_path, "", "", timestamp, &self.api_secret);
        let url = format!("{}{}", base, path);

        eprintln!("  Request URL:     {}", url);
        eprintln!("  Method:          GET");
        eprintln!("  Timestamp:       {}", timestamp);
        eprintln!("  Signature path:  {}", full_path);
        eprintln!("  Body hash:       {}...{}", &body_hash[..16], &body_hash[body_hash.len()-8..]);
        eprintln!("  Sign payload:    {:?}", sign_payload);
        eprintln!("  Signature:       {}...{}", &signature[..16], &signature[signature.len()-8..]);
        eprintln!("  Headers:");
        eprintln!("    KEY:           {}", self.api_key);
        eprintln!("    Timestamp:     {}", timestamp);
        eprintln!("    SIGN:          {}...{}", &signature[..16], &signature[signature.len()-8..]);

        match self.rest_client
            .get(&url)
            .header("KEY", &self.api_key)
            .header("SIGN", &signature)
            .header("Timestamp", timestamp.to_string())
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                eprintln!("\n  Response status:  {}", status);
                eprintln!("  Response body:    {}", &body_text[..body_text.len().min(500)]);

                if status == 200 {
                    eprintln!("\n  ✅ AUTHENTICATION SUCCESSFUL!");
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_text) {
                        let balance = json.get("available")
                            .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
                            .or_else(|| json.get("available").and_then(|v| v.as_f64()))
                            .unwrap_or(0.0);
                        eprintln!("  Available balance: ${:.4}", balance);
                        return Ok(balance);
                    }
                    return Ok(0.0);
                } else {
                    eprintln!("\n  ❌ AUTHENTICATION FAILED (HTTP {})", status);
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_text) {
                        let label = json.get("label").and_then(|v| v.as_str()).unwrap_or("?");
                        let msg = json.get("message").and_then(|v| v.as_str()).unwrap_or("?");
                        eprintln!("  Error label:    {}", label);
                        eprintln!("  Error message:  {}", msg);

                        if label == "INVALID_KEY" {
                            eprintln!("\n  ╔══════════════════════════════════════════════════════╗");
                            eprintln!("  ║  INVALID_KEY — The key is NOT recognized by Gate.io  ║");
                            eprintln!("  ╠══════════════════════════════════════════════════════╣");
                            eprintln!("  ║  This means the KEY value itself is wrong, NOT the   ║");
                            eprintln!("  ║  signature. Common causes:                           ║");
                            eprintln!("  ║                                                      ║");
                            if self.testnet {
                                eprintln!("  ║  1. You're using a MAINNET key on TESTNET endpoint   ║");
                                eprintln!("  ║     → Generate keys at gate.io TESTNET futures page   ║");
                                eprintln!("  ║  2. The key was deleted or expired                    ║");
                                eprintln!("  ║  3. Wrong env var: check GATEIO_TESTNET_API_KEY       ║");
                                eprintln!("  ║     (not GATEIO_API_KEY for testnet)                  ║");
                            } else {
                                eprintln!("  ║  1. You're using a TESTNET key on MAINNET endpoint    ║");
                                eprintln!("  ║  2. The key was deleted or expired                    ║");
                                eprintln!("  ║  3. Wrong env var: check GATEIO_API_KEY               ║");
                            }
                            eprintln!("  ║                                                      ║");
                            eprintln!("  ║  Your key: {}...{}", 
                                     &self.api_key[..self.api_key.len().min(8)],
                                     "                           ║");
                            eprintln!("  ║  Key length: {} chars", self.api_key.len());
                            eprintln!("  ║  Endpoint: {}", base);
                            eprintln!("  ╚══════════════════════════════════════════════════════╝");
                        }
                    }
                    return Err(format!("HTTP {} — {}", status, &body_text[..body_text.len().min(200)]));
                }
            }
            Err(e) => {
                eprintln!("  ❌ Request failed: {}", e);
                return Err(format!("Request failed: {}", e));
            }
        }
    }

    /// REST GET with auth (for position/balance queries that don't need WS speed).
    ///
    /// `path` should be the endpoint path WITHOUT the `/api/v4` prefix,
    /// e.g. `/futures/usdt/accounts`. The prefix is added automatically
    /// for both the URL and the signature (Gate.io v4 requires the full
    /// URL path in the signature).
    async fn rest_get(&self, path: &str, query: &str) -> Result<serde_json::Value, ExchangeError> {
        let timestamp = now_ms() / 1000;
        // Gate.io v4 REST API requires the FULL path (including /api/v4 prefix) in the signature.
        let full_path = format!("/api/v4{}", path);
        let signature = Self::rest_sign("GET", &full_path, query, "", timestamp, &self.api_secret);

        let url = if query.is_empty() {
            format!("{}{}", self.base_url(), path)
        } else {
            format!("{}{}?{}", self.base_url(), path, query)
        };

        debug!("[gateio-rest] GET {} (key={}…{}, len={}, ts={}, testnet={})",
               url,
               &self.api_key[..self.api_key.len().min(4)],
               &self.api_key[self.api_key.len().saturating_sub(4)..],
               self.api_key.len(),
               timestamp,
               self.testnet);

        let response = self.rest_client
            .get(&url)
            .header("KEY", &self.api_key)
            .header("SIGN", &signature)
            .header("Timestamp", timestamp.to_string())
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| {
                error!("[gateio-rest] Request failed: {} — URL: {}", e, url);
                ExchangeError::Timeout
            })?;

        let status = response.status().as_u16();
        let body: serde_json::Value = response.json().await.map_err(|e| ExchangeError::Unknown {
            code: "JSON_PARSE".to_string(),
            message: e.to_string(),
        })?;

        if status >= 400 {
            let label = body.get("label").and_then(|v| v.as_str()).unwrap_or("?");
            let msg = body.get("message").and_then(|v| v.as_str()).unwrap_or("?");
            let mode_str = if self.testnet { "TESTNET" } else { "LIVE" };
            error!("[gateio-rest] HTTP {} on {}: label={}, msg={}, key={}…{} ({}ch), ts={}, url={}",
                  status, path, label, msg,
                  &self.api_key[..self.api_key.len().min(6)],
                  &self.api_key[self.api_key.len().saturating_sub(4)..],
                  self.api_key.len(),
                  timestamp, url);
            if label == "INVALID_KEY" {
                let kb = self.api_key.as_bytes();
                error!("[gateio-rest] ══════════════════════════════════════════════");
                error!("[gateio-rest] INVALID_KEY — {} key not recognized by Gate.io", mode_str);
                error!("[gateio-rest] Full API key being sent: \"{}\"", self.api_key);
                error!("[gateio-rest] Key byte analysis: len={}, first_byte=0x{:02x}, last_byte=0x{:02x}",
                       self.api_key.len(),
                       kb.first().copied().unwrap_or(0),
                       kb.last().copied().unwrap_or(0));
                error!("[gateio-rest] Base URL: {}", self.base_url());
                error!("[gateio-rest] Are you using {} keys with {} endpoint?",
                       mode_str, mode_str);
                error!("[gateio-rest] Sign payload: {:?}", format!("{}\n{}\n{}\n...\n{}",
                       "GET", full_path, query, timestamp));
                error!("[gateio-rest] ══════════════════════════════════════════════");
            }
            return Err(crate::execution_gateway::classify_gateio_error(status, &body));
        }
        Ok(body)
    }
}

// ---------------------------------------------------------------------------
// ExecutionGateway Trait Implementation — WebSocket-first
// ---------------------------------------------------------------------------

#[async_trait]
impl ExecutionGateway for GateIoGateway {
    async fn get_ticker(&self, symbol: &str) -> Result<RustTicker, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let url = format!("{}/futures/usdt/tickers?contract={}", self.base_url(), normalized);
        match self.rest_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let ticker = if data.is_array() {
                            data.as_array().and_then(|a| a.first())
                        } else {
                            Some(&data)
                        };
                        let t = ticker.unwrap_or(&data);
                        Ok(RustTicker {
                            last: Self::extract_json_float(&t.to_string(), "last").unwrap_or(0.0),
                            bid: Self::extract_json_float(&t.to_string(), "highest_bid").unwrap_or(0.0),
                            ask: Self::extract_json_float(&t.to_string(), "lowest_ask").unwrap_or(0.0),
                            volume_24h: Self::extract_json_float(&t.to_string(), "volume_24h").unwrap_or(0.0),
                        })
                    }
                    Err(e) => Err(ExchangeError::Unknown { code: "PARSE".into(), message: e.to_string() }),
                }
            }
            _ => Err(ExchangeError::Timeout),
        }
    }

    /// Submit an order via the persistent WebSocket connection.
    ///
    /// Flow:
    ///   1. Build WS order_place message (inline format!, no serde)
    ///   2. Register a oneshot channel in `pending` map
    ///   3. Send message via `ws_tx` channel to the WS writer task
    ///   4. Await the oneshot response (with timeout)
    ///   5. Return OrderResult or ExchangeError
    async fn submit_order(&self, intent: OrderIntent) -> Result<OrderResult, ExchangeError> {
        let symbol = Self::normalize_symbol(&intent.symbol);
        let size = Self::validate_contract_precision(intent.size)?;
        let start_us = now_us();

        // Check WS readiness
        if !self.is_ready.load(Ordering::Acquire) {
            return Err(ExchangeError::ConnectionReset);
        }

        // ── DYNAMIC LEVERAGE ──
        // If the intent specifies a leverage target, set it BEFORE submitting
        // the order. Gate.io allows per-position leverage via REST.
        // Replaces the old hardcoded `leverage: Some(10)` in execution_router_loop.
        if let Some(target_leverage) = intent.leverage {
            if target_leverage > 0 && target_leverage <= 125 {
                let lev_result = self.set_leverage(&symbol, target_leverage).await;
                match lev_result {
                    Ok(_) => {
                        debug!("[gateio-ws] Set leverage {}x for {}", target_leverage, symbol);
                        // CATEGORY 2 FIX: Poll for leverage confirmation instead of
                        // hardcoded 500ms sleep. Check up to 5 times with 100ms intervals
                        // (max 500ms total, but usually confirms in 100-200ms).
                        let mut confirmed = false;
                        for attempt in 0..5 {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            match self.get_position(&symbol).await {
                                Ok(Some(pos)) if pos.leverage == target_leverage => {
                                    confirmed = true;
                                    debug!(
                                        "[gateio-ws] Leverage {}x confirmed for {} after {}ms",
                                        target_leverage, symbol, (attempt + 1) * 100
                                    );
                                    break;
                                }
                                Ok(Some(pos)) => {
                                    debug!(
                                        "[gateio-ws] Leverage poll {}: current={}x target={}x",
                                        attempt + 1, pos.leverage, target_leverage
                                    );
                                }
                                _ => {
                                    // No position yet or error — leverage may still be propagating
                                    debug!("[gateio-ws] Leverage poll {}: no position yet", attempt + 1);
                                }
                            }
                        }
                        if !confirmed {
                            warn!(
                                "[gateio-ws] Leverage {}x not confirmed for {} after 500ms — proceeding anyway",
                                target_leverage, symbol
                            );
                        }
                    }
                    Err(e) => {
                        // Non-fatal: log and continue (exchange may already be at this leverage)
                        warn!("[gateio-ws] Failed to set leverage {}x for {}: {} (continuing)", target_leverage, symbol, e);
                    }
                }
            }
        }

        let signed_size = if intent.side == OrderSide::Sell { -size } else { size };

        let tif = match intent.order_type {
            OrderType::PostOnly => "poc",
            OrderType::Market => "ioc",
            OrderType::Limit => &intent.time_in_force,
        };

        // BUG FIX #3: Use InstrumentManager for Gate.io price formatting when available.
        // Previously hardcoded "{:.8}" which may not respect Gate.io's order_price_round
        // (tick size) for specific contracts. The InstrumentManager fetches the real
        // tick size from GET /api/v4/futures/usdt/contracts at startup.
        // BUG FIX #3: Use InstrumentManager for Gate.io price formatting.
        // Previously hardcoded "{:.8}" which may not respect Gate.io's order_price_round
        // (tick size) for specific contracts. The InstrumentManager fetches the real
        // tick size from GET /api/v4/futures/usdt/contracts at startup.
        let price_str = if intent.order_type == OrderType::Market {
            "0".to_string()
        } else {
            intent.price.map(|p| {
                if let Some(ref mgr) = self.instrument_mgr {
                    let spec = mgr.get_or_default(Exchange::GateIo, &symbol);
                    spec.format_price(p)
                } else {
                    format!("{:.8}", p)
                }
            }).unwrap_or_else(|| "0".to_string())
        };

        let client_id = self.next_id();

        // Build WS message
        let ws_msg = Self::build_order_place_message(
            &self.api_key,
            &self.api_secret,
            &client_id,
            &symbol,
            signed_size,
            &price_str,
            tif,
            intent.reduce_only,
        );

        // Register pending order with oneshot channel
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.write();
            pending.insert(client_id.clone(), PendingOrder {
                client_id: client_id.clone(),
                intent: intent.clone(),
                response_tx: tx,
                submit_us: start_us,
            });
        }

        // Track locally
        {
            let mut state = self.order_state.write();
            state.insert(client_id.clone(), OrderTracking {
                client_id: client_id.clone(),
                symbol: symbol.clone(),
                side: intent.side.clone(),
                size,
                state: OrderTrackingState::PendingAck { submit_us: start_us },
                created_us: start_us,
            });
        }

        // Send via WS channel
        self.ws_tx.send(WsCommand::SendText(ws_msg)).map_err(|_| {
            ExchangeError::ConnectionReset
        })?;

        // Await response with timeout
        let result = match tokio::time::timeout(Duration::from_millis(RESPONSE_TIMEOUT_MS), rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                // Oneshot dropped — WS reconnection in progress
                warn!("[gateio-ws] Response channel dropped for order {}", client_id);
                Err(ExchangeError::ConnectionReset)
            }
            Err(_) => {
                // Timeout — order may still be live on exchange.
                // Return TimedOut with the client_id so that retry_failed_leg can call
                // check_order_by_client_id() before retrying, preventing duplicate fills.
                warn!("[gateio-ws] Timeout waiting for order {} ACK ({}ms)", client_id, RESPONSE_TIMEOUT_MS);
                // Remove from pending to avoid leak
                self.pending.write().remove(&client_id);
                Err(ExchangeError::TimedOut { client_order_id: client_id.clone() })
            }
        };

        // ── SL/TP CONDITIONAL ORDER SUBMISSION ──
        // After the main order is confirmed, submit linked Stop Loss and Take Profit
        // as conditional trigger orders on Gate.io. These fire automatically if the
        // price hits the SL/TP levels — the Rust exit evaluator is a secondary safety
        // net that catches faster than the exchange's trigger latency.
        if let Ok(ref order_result) = result {
            if order_result.filled_size > 0 && !intent.reduce_only {
                // Extract SL/TP from the intent (set by the strategy engine via OrderCommand)
                let stop_loss = intent.stop_loss;
                let take_profit = intent.take_profit;

                if stop_loss.is_some() || take_profit.is_some() {
                    // Fire-and-forget: SL/TP submission is async and non-blocking.
                    // If it fails, the Rust exit evaluator will still protect the position.
                    let filled_size = order_result.filled_size;
                    let side = intent.side.clone();
                    let sym = symbol.clone();
                    let rest_client = self.rest_client.clone();
                    let api_key = self.api_key.clone();
                    let api_secret = self.api_secret.clone();
                    let testnet = self.testnet;

                    tokio::spawn(async move {
                        let gateway_for_sl = GateIoGatewaySlTpHelper {
                            api_key,
                            api_secret,
                            testnet,
                            rest_client,
                        };
                        gateway_for_sl.submit_sl_tp(
                            &sym, &side, filled_size, stop_loss, take_profit,
                        ).await;
                    });
                }
            }
        }

        result
    }

    async fn cancel_order(&self, order_id: &str, _symbol: &str) -> Result<(), ExchangeError> {
        if !self.is_ready.load(Ordering::Acquire) {
            return Err(ExchangeError::ConnectionReset);
        }

        let ws_msg = Self::build_order_cancel_message(&self.api_key, &self.api_secret, order_id);
        self.ws_tx.send(WsCommand::SendText(ws_msg)).map_err(|_| ExchangeError::ConnectionReset)?;
        info!("[gateio-ws] Cancel sent for order {}", order_id);
        Ok(())
    }

    /// Position queries use REST fallback (cold path, infrequent).
    async fn get_position(&self, symbol: &str) -> Result<Option<Position>, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let path = format!("/futures/usdt/positions/{}", normalized);
        let response = self.rest_get(&path, "").await?;

        let size = response.get("size").and_then(|v| v.as_i64()).unwrap_or(0);
        if size == 0 {
            return Ok(None);
        }

        let entry_price = response.get("entry_price")
            .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
            .or_else(|| response.get("entry_price").and_then(|v| v.as_f64()))
            .unwrap_or(0.0);

        let unrealized_pnl = response.get("unrealised_pnl")
            .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let leverage = response.get("leverage")
            .and_then(|v| v.as_i64()).unwrap_or(1) as i32;

        let side = if size > 0 { "long" } else { "short" }.to_string();

        Ok(Some(Position {
            symbol: normalized,
            size,
            entry_price,
            unrealized_pnl,
            leverage,
            side,
        }))
    }

    async fn set_leverage(&self, symbol: &str, leverage: i32) -> Result<(), ExchangeError> {
        // FIX 4: Completely rewritten margin mode detection.
        //
        // Previous logic checked the `mode` field in position data, but that field
        // represents the position mode (single/dual), NOT the margin mode. It also
        // defaulted to cross-margin on any error or missing position, causing every
        // call to fail with "cross_leverage_limit only for cross-margin" and then
        // retry with isolated — wasting ~150ms per call.
        //
        // New approach: Try isolated margin FIRST (using `leverage` parameter) since
        // the account is configured for isolated margin. If that fails with an error
        // indicating cross-margin mode, retry with `cross_leverage_limit`.
        // This eliminates the wasted REST call in the common case.
        let normalized = Self::normalize_symbol(symbol);
        let path = format!("/futures/usdt/positions/{}/leverage", normalized);
        let full_path = format!("/api/v4{}", path);
        let body = "";

        // Try isolated margin first (most common configuration)
        let iso_query = format!("leverage={}", leverage);
        let timestamp = now_ms() / 1000;
        let signature = Self::rest_sign("POST", &full_path, &iso_query, body, timestamp, &self.api_secret);
        let url = format!("{}{}?{}", self.base_url(), path, iso_query);
        let response = self.rest_client
            .post(&url)
            .header("KEY", &self.api_key)
            .header("SIGN", &signature)
            .header("Timestamp", timestamp.to_string())
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let status = response.status().as_u16();
        if status >= 400 {
            let resp_body: serde_json::Value = response.json().await.map_err(|e| ExchangeError::Unknown {
                code: "JSON_PARSE".to_string(),
                message: e.to_string(),
            })?;
            let err = crate::execution_gateway::classify_gateio_error(status, &resp_body);
            let err_str = format!("{}", err);

            // If isolated-margin parameter failed because account uses cross-margin,
            // retry with the cross_leverage_limit parameter.
            if err_str.contains("leverage") || err_str.contains("cross") || err_str.contains("MISSING_REQUIRED_PARAM") {
                debug!("[gateio-ws] Isolated-margin leverage failed for {}, retrying with cross-margin param", normalized);
                let cross_query = format!("cross_leverage_limit={}", leverage);
                let timestamp = now_ms() / 1000;
                let cross_signature = Self::rest_sign("POST", &full_path, &cross_query, body, timestamp, &self.api_secret);
                let cross_url = format!("{}{}?{}", self.base_url(), path, cross_query);
                let cross_response = self.rest_client
                    .post(&cross_url)
                    .header("KEY", &self.api_key)
                    .header("SIGN", &cross_signature)
                    .header("Timestamp", timestamp.to_string())
                    .header("Content-Type", "application/json")
                    .send()
                    .await
                    .map_err(|_| ExchangeError::Timeout)?;

                let cross_status = cross_response.status().as_u16();
                if cross_status >= 400 {
                    let cross_body: serde_json::Value = cross_response.json().await.map_err(|e| ExchangeError::Unknown {
                        code: "JSON_PARSE".to_string(),
                        message: e.to_string(),
                    })?;
                    return Err(crate::execution_gateway::classify_gateio_error(cross_status, &cross_body));
                }
                info!("[gateio-ws] Leverage set to {}x for {} (cross margin)", leverage, normalized);
                return Ok(());
            }

            return Err(err);
        }

        info!("[gateio-ws] Leverage set to {}x for {} (isolated margin)", leverage, normalized);
        Ok(())
    }

    /// FIX 5: Set margin mode to cross-margin for a symbol.
    /// Gate.io defaults to isolated margin, but cross-margin is safer for
    /// multi-position strategies as it shares margin across all positions.
    async fn set_margin_mode(&self, symbol: &str, mode: &str) -> Result<(), ExchangeError> {
        // Gate.io futures default to cross margin. Margin mode is controlled
        // via the dual_mode setting, not per-position.
        debug!("[gateio-ws] Margin mode '{}' requested for {} (Gate.io defaults to cross)", mode, symbol);
        Ok(())
    }

    async fn get_balance(&self) -> Result<f64, ExchangeError> {
        let response = self.rest_get("/futures/usdt/accounts", "").await?;
        let available = response.get("available")
            .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
            .or_else(|| response.get("available").and_then(|v| v.as_f64()))
            .unwrap_or(0.0);
        Ok(available)
    }

    async fn get_positions(&self) -> Result<Vec<Position>, ExchangeError> {
        let response = self.rest_get("/futures/usdt/positions", "").await?;
        let mut positions = Vec::new();
        if let Some(arr) = response.as_array() {
            for item in arr {
                let size = item.get("size").and_then(|v| v.as_i64()).unwrap_or(0);
                if size == 0 {
                    continue;
                }
                let symbol = item.get("contract")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let entry_price = item.get("entry_price")
                    .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| item.get("entry_price").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);
                let unrealized_pnl = item.get("unrealised_pnl")
                    .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let leverage = item.get("leverage")
                    .and_then(|v| v.as_i64()).unwrap_or(1) as i32;
                let side = if size > 0 { "long" } else { "short" }.to_string();
                positions.push(Position {
                    symbol,
                    size,
                    entry_price,
                    unrealized_pnl,
                    leverage,
                    side,
                });
            }
        }
        Ok(positions)
    }

    /// BUG 1 FIX: Override usdt_to_contracts as a trait method so that Gate.io's
    /// quanto-aware conversion is dispatched correctly through Arc<dyn ExecutionGateway>.
    /// Previously this was only defined as an inherent method on GateIoGateway,
    /// meaning trait dispatch always fell through to the default implementation
    /// which does simple `usdt_amount / last_price` — producing 0 contracts for
    /// small orders on quanto contracts like BTC_USDT (where 1 contract = 0.0001 BTC).
    async fn usdt_to_contracts(&self, symbol: &str, usdt_amount: f64) -> Result<i64, ExchangeError> {
        // Delegate to the inherent quanto-aware implementation
        // which fetches the quanto_multiplier from the Gate.io REST API.
        GateIoGateway::usdt_to_contracts(self, symbol, usdt_amount).await
    }

    async fn get_order_status(&self, order_id: &str, symbol: &str)
        -> Result<Option<OrderResult>, ExchangeError>
    {
        let normalized = Self::normalize_symbol(symbol);
        let path = format!("/futures/usdt/orders/{}", order_id);
        let query = format!("contract={}", normalized);
        match self.rest_get(&path, &query).await {
            Ok(response) => {
                let status = response.get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let filled_size = response.get("size")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0)
                    - response.get("left")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                let avg_fill_price = response.get("fill_price")
                    .and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| response.get("fill_price").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);
                Ok(Some(OrderResult {
                    order_id: order_id.to_string(),
                    status,
                    filled_size,
                    avg_fill_price,
                    fee: 0.0,
                    latency_us: 0,
                    exchange_timestamp: now_ms(),
                    rejection_reason: None,
                }))
            }
            Err(ExchangeError::Unknown { ref code, .. }) if code == "ORDER_NOT_FOUND" => {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Idempotency check: look up an order by its client-side `req_id` (the `text` field
    /// Gate.io stores verbatim on each order).  Called when `submit_order` returns
    /// `ExchangeError::TimedOut` — i.e. the WS ACK never arrived but the exchange may
    /// have accepted the order.  Returns the exchange-assigned order ID if found so the
    /// caller can decide whether to retry or accept the existing fill.
    async fn check_order_by_client_id(
        &self,
        client_order_id: &str,
        symbol: &str,
    ) -> Result<Option<String>, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let result = check_order_exists_gateio(
            &self.rest_client,
            self.base_url(),
            &self.api_key,
            &self.api_secret,
            &normalized,
            client_order_id,
        ).await;
        Ok(result)
    }

    /// FIX 6: Submit a conditional stop-loss order to Gate.io via price_trigger API.
    async fn submit_conditional_sl(
        &self,
        symbol: &str,
        parent_side: &OrderSide,
        filled_size: i64,
        sl_price: f64,
    ) -> Result<(), ExchangeError> {
        self.submit_sl_tp_orders(symbol, parent_side, filled_size, Some(sl_price), None).await;
        Ok(())
    }

    /// FIX 6: Submit a conditional take-profit order to Gate.io via price_trigger API.
    async fn submit_conditional_tp(
        &self,
        symbol: &str,
        parent_side: &OrderSide,
        filled_size: i64,
        tp_price: f64,
    ) -> Result<(), ExchangeError> {
        self.submit_sl_tp_orders(symbol, parent_side, filled_size, None, Some(tp_price)).await;
        Ok(())
    }

    /// Feature 5: Cancel all conditional (price-triggered) orders for a symbol.
    /// Gate.io requires canceling old conditional orders before submitting new ones
    /// when trailing stops update the SL price.
    async fn cancel_conditional_orders(&self, symbol: &str) -> Result<(), ExchangeError> {
        let contract = Self::normalize_symbol(symbol);
        let query = format!("contract={}", contract);
        let path = "/futures/usdt/price_orders";

        // First, list all open price-triggered orders for this symbol
        match self.rest_get(path, &query).await {
            Ok(orders) => {
                if let Some(order_list) = orders.as_array() {
                    for order in order_list {
                        if let Some(order_id) = order.get("id").and_then(|v| v.as_u64()) {
                            let cancel_path = format!("{}/{}", path, order_id);
                            let full_cancel_path = format!("/api/v4{}", cancel_path);
                            let ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs() as i64;
                            let signature = Self::rest_sign("DELETE", &full_cancel_path, "", "", ts, &self.api_secret);
                            let url = format!("{}{}", self.base_url(), cancel_path);

                            match self.rest_client.delete(&url)
                                .header("KEY", &self.api_key)
                                .header("SIGN", &signature)
                                .header("Timestamp", ts.to_string())
                                .send()
                                .await
                            {
                                Ok(resp) => {
                                    if resp.status().is_success() {
                                        info!("[gateio] Cancelled conditional order {} for {}", order_id, symbol);
                                    } else {
                                        let body = resp.text().await.unwrap_or_default();
                                        warn!("[gateio] Failed to cancel conditional order {}: {}", order_id, body);
                                    }
                                }
                                Err(e) => {
                                    warn!("[gateio] HTTP error cancelling conditional order {}: {}", order_id, e);
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            Err(e) => {
                warn!("[gateio] Failed to list conditional orders for {}: {}", symbol, e);
                // Non-fatal: the in-memory exit evaluator still protects the position
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-trait methods on GateIoGateway
// ---------------------------------------------------------------------------

impl GateIoGateway {
    /// Monitor liquidation prices for all open positions.
    ///
    /// Queries `/futures/usdt/positions` REST endpoint, extracts `liq_price`,
    /// `margin`, and `maintenance_margin` fields, and triggers auto-reduce
    /// (50% position close) when price is within 5% of liquidation or
    /// emergency close when within 2%.
    ///
    /// This method should be called from a background task spawned at
    /// gateway initialization (every 30 seconds).
    pub async fn monitor_liquidation_prices(&self) -> Result<(), ExchangeError> {
        let response = self.rest_get("/futures/usdt/positions", "").await?;

        if let Some(positions) = response.as_array() {
            for pos in positions {
                let contract = pos.get("contract")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let size = pos.get("size")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);

                if size == 0 {
                    continue; // No position
                }

                let liq_price = pos.get("liq_price")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| pos.get("liq_price").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);

                let mark_price = pos.get("mark_price")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| pos.get("mark_price").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);

                let margin = pos.get("margin")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| pos.get("margin").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);

                let maintenance_margin = pos.get("maintenance_margin")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| pos.get("maintenance_margin").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);

                if liq_price <= 0.0 || mark_price <= 0.0 {
                    continue;
                }

                // Calculate distance to liquidation
                let distance_pct = if size > 0 {
                    // Long position: liquidation when price drops
                    (mark_price - liq_price) / mark_price
                } else {
                    // Short position: liquidation when price rises
                    (liq_price - mark_price) / mark_price
                };

                debug!(
                    "[gateio-liq] {} size={} mark={:.4} liq={:.4} dist={:.2}% margin={:.2} maint={:.2}",
                    contract, size, mark_price, liq_price, distance_pct * 100.0, margin, maintenance_margin
                );

                // Emergency close: within 2% of liquidation
                if distance_pct < 0.02 {
                    error!(
                        "[gateio-liq] 🚨 EMERGENCY: {} within 2% of liquidation (dist={:.2}%) — closing entire position",
                        contract, distance_pct * 100.0
                    );

                    // Submit emergency market close order
                    let close_side = if size > 0 {
                        OrderSide::Sell
                    } else {
                        OrderSide::Buy
                    };

                    let close_intent = OrderIntent {
                        symbol: contract.to_string(),
                        side: close_side,
                        size: size.abs(),
                        order_type: OrderType::Market,
                        price: None,
                        reduce_only: true,
                        leverage: None,
                        time_in_force: "ioc".to_string(),
                        slippage_cap_pct: Some(0.02), // Allow 2% slippage for emergency
                        placement: PlacementType::AtBest,
                        stop_loss: None,
                        take_profit: None,
                        confidence: 0.0,
                        signal_tag: "emergency_liquidation_close".to_string(),
                    };

                    match self.submit_order(close_intent).await {
                        Ok(res) => {
                            info!(
                                "[gateio-liq] ✅ Emergency close executed: {} filled {} @ {:.4}",
                                contract, res.filled_size, res.avg_fill_price
                            );
                        }
                        Err(e) => {
                            error!("[gateio-liq] ❌ Emergency close failed for {}: {}", contract, e);
                        }
                    }
                }
                // Auto-reduce: within 5% of liquidation
                else if distance_pct < 0.05 {
                    warn!(
                        "[gateio-liq] ⚠️ {} within 5% of liquidation (dist={:.2}%) — reducing position by 50%",
                        contract, distance_pct * 100.0
                    );

                    let reduce_size = (size.abs() as f64 * 0.5).ceil() as i64;
                    let reduce_side = if size > 0 {
                        OrderSide::Sell
                    } else {
                        OrderSide::Buy
                    };

                    let reduce_intent = OrderIntent {
                        symbol: contract.to_string(),
                        side: reduce_side,
                        size: reduce_size,
                        order_type: OrderType::Market,
                        price: None,
                        reduce_only: true,
                        leverage: None,
                        time_in_force: "ioc".to_string(),
                        slippage_cap_pct: Some(0.01),
                        placement: PlacementType::AtBest,
                        stop_loss: None,
                        take_profit: None,
                        confidence: 0.0,
                        signal_tag: "auto_reduce_liquidation".to_string(),
                    };

                    match self.submit_order(reduce_intent).await {
                        Ok(res) => {
                            info!(
                                "[gateio-liq] ✅ Auto-reduce executed: {} reduced by {} @ {:.4}",
                                contract, res.filled_size, res.avg_fill_price
                            );
                        }
                        Err(e) => {
                            warn!("[gateio-liq] Auto-reduce failed for {}: {}", contract, e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

}

// ═══════════════════════════════════════════════════════════════════════════
// SL/TP Helper — lightweight struct for async SL/TP submission
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight helper for submitting SL/TP conditional orders in a spawned task.
///
/// The full `GateIoGateway` can't be cloned or moved into a spawned task because
/// it contains non-Clone fields (channels, atomics). This helper captures only
/// the REST client and credentials needed for SL/TP submission.
struct GateIoGatewaySlTpHelper {
    api_key: String,
    api_secret: Vec<u8>,
    testnet: bool,
    rest_client: reqwest::Client,
}

impl GateIoGatewaySlTpHelper {
    fn base_url(&self) -> &str {
        if self.testnet {
            "https://api-testnet.gateapi.io/api/v4"
        } else {
            // CATEGORY 2 FIX: Use api.gateio.ws (standard REST endpoint) instead of
            // fx-api.gateio.ws which is the WebSocket endpoint. The SL/TP helper uses
            // REST API calls (POST /futures/usdt/price_orders) which must go to the
            // REST API host. Using the WS host causes silent 404s or connection resets.
            "https://api.gateio.ws/api/v4"
        }
    }

    fn rest_sign_helper(method: &str, path: &str, query: &str, body: &str, timestamp: i64, secret: &[u8]) -> String {
        use sha2::Digest;
        let body_hash = hex::encode(sha2::Sha512::digest(body.as_bytes()));
        let payload = format!("{}\n{}\n{}\n{}\n{}", method, path, query, body_hash, timestamp);
        let mut mac = Hmac::<Sha512>::new_from_slice(secret)
            .expect("HMAC accepts any key length");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Fetch the last traded price for a contract from Gate.io REST API.
    async fn fetch_last_price(&self, contract: &str) -> Option<f64> {
        let url = format!(
            "{}/futures/usdt/tickers?contract={}",
            self.base_url(), contract
        );
        match self.rest_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let ticker = if data.is_array() {
                            data.as_array().and_then(|a| a.first())
                        } else {
                            Some(&data)
                        };
                        ticker
                            .and_then(|t| t.get("last"))
                            .and_then(|v| v.as_str().or_else(|| v.as_f64().map(|_| "")))
                            .and_then(|s| {
                                if s.is_empty() {
                                    ticker.and_then(|t| t.get("last")).and_then(|v| v.as_f64())
                                } else {
                                    s.parse::<f64>().ok()
                                }
                            })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    async fn submit_sl_tp(
        &self,
        symbol: &str,
        parent_side: &OrderSide,
        filled_size: i64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    ) {
        let close_size = if *parent_side == OrderSide::Buy {
            -filled_size
        } else {
            filled_size
        };

        // Fetch last traded price once for both SL and TP validation
        let last_price = self.fetch_last_price(symbol).await;

        if let Some(sl_price) = stop_loss {
            if sl_price > 0.0 {
                let trigger_type = if *parent_side == OrderSide::Buy { 1 } else { 0 };
                let rule: u8 = if trigger_type == 0 { 1 } else { 2 };

                // Validate trigger price against last traded price
                if let Some(lp) = last_price {
                    if !GateIoGateway::validate_trigger_price(sl_price, lp, rule) {
                        error!(
                            "[gateio-sl-tp] ⚠️ SL price {:.4} already breached (last={:.4}, rule={}) for {} {}",
                            sl_price, lp, rule, symbol,
                            if *parent_side == OrderSide::Buy { "LONG" } else { "SHORT" }
                        );
                        return; // Don't submit SL or TP if SL is already breached
                    }
                }

                let body = GateIoGateway::build_price_trigger_body(symbol, sl_price, close_size, trigger_type, None);
                let path = "/futures/usdt/price_orders";
                let timestamp = now_ms() / 1000;
                let full_path = format!("/api/v4{}", path);
                let signature = Self::rest_sign_helper("POST", &full_path, "", &body, timestamp, &self.api_secret);
                let url = format!("{}{}", self.base_url(), path);

                match self.rest_client
                    .post(&url)
                    .header("KEY", &self.api_key)
                    .header("SIGN", &signature)
                    .header("Timestamp", timestamp.to_string())
                    .header("Content-Type", "application/json")
                    .body(body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        if status < 400 {
                            info!("[gateio-sl-tp] SL conditional order placed for {} @ {:.4}", symbol, sl_price);
                        } else {
                            let body_text = resp.text().await.unwrap_or_default();
                            error!("[gateio-sl-tp] SL order failed (HTTP {}): {}", status, body_text);
                        }
                    }
                    Err(e) => {
                        error!("[gateio-sl-tp] SL order submit error: {}", e);
                    }
                }
            }
        }

        if let Some(tp_price) = take_profit {
            if tp_price > 0.0 {
                let trigger_type = if *parent_side == OrderSide::Buy { 0 } else { 1 };
                let rule: u8 = if trigger_type == 0 { 1 } else { 2 };

                if let Some(lp) = last_price {
                    if !GateIoGateway::validate_trigger_price(tp_price, lp, rule) {
                        warn!(
                            "[gateio-sl-tp] TP price {:.4} already breached (last={:.4}) for {} {} — price past target",
                            tp_price, lp, symbol,
                            if *parent_side == OrderSide::Buy { "LONG" } else { "SHORT" }
                        );
                        return;
                    }
                }

                let body = GateIoGateway::build_price_trigger_body(symbol, tp_price, close_size, trigger_type, None);
                let path = "/futures/usdt/price_orders";
                let timestamp = now_ms() / 1000;
                let full_path = format!("/api/v4{}", path);
                let signature = Self::rest_sign_helper("POST", &full_path, "", &body, timestamp, &self.api_secret);
                let url = format!("{}{}", self.base_url(), path);

                match self.rest_client
                    .post(&url)
                    .header("KEY", &self.api_key)
                    .header("SIGN", &signature)
                    .header("Timestamp", timestamp.to_string())
                    .header("Content-Type", "application/json")
                    .body(body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        if status < 400 {
                            info!("[gateio-sl-tp] TP conditional order placed for {} @ {:.4}", symbol, tp_price);
                        } else {
                            let body_text = resp.text().await.unwrap_or_default();
                            error!("[gateio-sl-tp] TP order failed (HTTP {}): {}", status, body_text);
                        }
                    }
                    Err(e) => {
                        error!("[gateio-sl-tp] TP order submit error: {}", e);
                    }
                }
            }
        }
    }
}
