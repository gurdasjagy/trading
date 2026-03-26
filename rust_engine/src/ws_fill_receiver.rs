//! FEATURE 4: WebSocket Fill Receiver for Binance & Bybit.
//!
//! Binance and Bybit gateways currently use REST API for order submission,
//! adding 50-200ms latency per fill confirmation. This module adds WebSocket
//! connections to receive fill/order updates in real-time (~5-15ms).
//!
//! # Architecture
//! - Orders are still submitted via REST (Binance doesn't support WS order placement for futures)
//! - Fill confirmations arrive via WebSocket with much lower latency
//! - A shared `FillUpdateSink` receives parsed fill events from both exchanges
//! - The execution router consumes fills from the sink for position tracking
//!
//! # Binance Futures User Data Stream
//! 1. POST /fapi/v1/listenKey to get a listen key
//! 2. Connect to wss://fstream.binance.com/ws/{listenKey}
//! 3. Receive ORDER_TRADE_UPDATE events for fills
//! 4. PUT /fapi/v1/listenKey every 30 minutes to keep alive
//!
//! # Bybit v5 Private WebSocket
//! 1. Connect to wss://stream.bybit.com/v5/private
//! 2. Authenticate with HMAC-SHA256 signature
//! 3. Subscribe to "order" topic for fill updates

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

/// A parsed fill event from any exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    /// Source exchange identifier.
    pub exchange: String,
    /// Trading symbol (normalized, e.g. "BTCUSDT").
    pub symbol: String,
    /// Exchange-assigned order ID.
    pub order_id: String,
    /// Client-assigned order ID (for idempotency matching).
    pub client_order_id: String,
    /// Order side: "Buy" or "Sell".
    pub side: String,
    /// Order status: "New", "PartiallyFilled", "Filled", "Cancelled", etc.
    pub status: String,
    /// Cumulative filled quantity.
    pub filled_qty: f64,
    /// Average fill price.
    pub avg_price: f64,
    /// Fee paid for this fill.
    pub fee: f64,
    /// Fee asset (e.g. "USDT").
    pub fee_asset: String,
    /// Whether this is a maker fill.
    pub is_maker: bool,
    /// Timestamp of the fill event (milliseconds).
    pub timestamp_ms: i64,
    /// Latency from exchange event to our processing (microseconds).
    pub receive_latency_us: u64,
}

/// Thread-safe sink for fill events from all exchanges.
///
/// The WS receiver tasks push fill events here, and the execution
/// router drains them on each tick for position/PnL updates.
#[derive(Clone)]
pub struct FillUpdateSink {
    inner: Arc<Mutex<FillSinkInner>>,
}

struct FillSinkInner {
    /// Pending fill events (FIFO queue).
    pending: VecDeque<FillEvent>,
    /// Total fills received (for telemetry).
    total_received: u64,
    /// Total fills from Binance.
    binance_count: u64,
    /// Total fills from Bybit.
    bybit_count: u64,
}

impl FillUpdateSink {
    /// Create a new fill update sink.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(FillSinkInner {
                pending: VecDeque::with_capacity(256),
                total_received: 0,
                binance_count: 0,
                bybit_count: 0,
            })),
        }
    }

    /// Push a new fill event into the sink.
    pub fn push(&self, event: FillEvent) {
        let mut inner = self.inner.lock();
        match event.exchange.as_str() {
            "binance" => inner.binance_count += 1,
            "bybit" => inner.bybit_count += 1,
            _ => {}
        }
        inner.total_received += 1;
        inner.pending.push_back(event);
    }

    /// Drain all pending fill events.
    pub fn drain(&self) -> Vec<FillEvent> {
        let mut inner = self.inner.lock();
        inner.pending.drain(..).collect()
    }

    /// Check if there are pending fills.
    pub fn has_pending(&self) -> bool {
        let inner = self.inner.lock();
        !inner.pending.is_empty()
    }

    /// Get telemetry statistics.
    pub fn stats(&self) -> (u64, u64, u64) {
        let inner = self.inner.lock();
        (inner.total_received, inner.binance_count, inner.bybit_count)
    }
}

impl Default for FillUpdateSink {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Binance User Data Stream
// ---------------------------------------------------------------------------

/// Configuration for the Binance WebSocket fill receiver.
#[derive(Debug, Clone)]
pub struct BinanceWsConfig {
    /// API key for listen key management.
    pub api_key: String,
    /// Whether to use testnet endpoints.
    pub testnet: bool,
}

impl BinanceWsConfig {
    /// Get the REST base URL for listen key management.
    fn rest_base_url(&self) -> &str {
        if self.testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        }
    }

    /// Get the WebSocket base URL for user data stream.
    fn ws_base_url(&self) -> &str {
        if self.testnet {
            "wss://stream.binancefuture.com/ws"
        } else {
            "wss://fstream.binance.com/ws"
        }
    }
}

/// Create a listen key for the Binance Futures user data stream.
///
/// POST /fapi/v1/listenKey
/// Returns the listen key string on success.
pub async fn binance_create_listen_key(config: &BinanceWsConfig) -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/fapi/v1/listenKey", config.rest_base_url());

    let response = client
        .post(&url)
        .header("X-MBX-APIKEY", &config.api_key)
        .send()
        .await
        .map_err(|e| format!("Failed to create Binance listen key: {}", e))?;

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Binance listen key response: {}", e))?;

    body.get("listenKey")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("No listenKey in Binance response: {:?}", body))
}

/// Keep-alive for the Binance Futures user data stream.
///
/// PUT /fapi/v1/listenKey — must be called every 30 minutes.
pub async fn binance_keepalive_listen_key(config: &BinanceWsConfig) -> Result<(), String> {
    let client = reqwest::Client::new();
    let url = format!("{}/fapi/v1/listenKey", config.rest_base_url());

    client
        .put(&url)
        .header("X-MBX-APIKEY", &config.api_key)
        .send()
        .await
        .map_err(|e| format!("Failed to keep-alive Binance listen key: {}", e))?;

    Ok(())
}

/// Parse a Binance ORDER_TRADE_UPDATE WebSocket message into a FillEvent.
///
/// Message format:
/// ```json
/// {
///   "e": "ORDER_TRADE_UPDATE",
///   "T": 1234567890123,
///   "o": {
///     "s": "BTCUSDT",       // symbol
///     "c": "rte00000001",   // client order ID
///     "S": "BUY",           // side
///     "i": 12345678,        // order ID
///     "X": "FILLED",        // order status
///     "z": "0.001",         // cumulative filled qty
///     "ap": "60000.0",      // average price
///     "n": "0.024",         // commission amount
///     "N": "USDT",          // commission asset
///     "m": true,            // is maker
///     "T": 1234567890123    // order trade time
///   }
/// }
/// ```
pub fn parse_binance_fill(msg: &serde_json::Value, receive_time_us: i64) -> Option<FillEvent> {
    let event_type = msg.get("e").and_then(|v| v.as_str())?;
    if event_type != "ORDER_TRADE_UPDATE" {
        return None;
    }

    let order = msg.get("o")?;

    let symbol = order.get("s").and_then(|v| v.as_str())?.to_string();
    let order_id = order
        .get("i")
        .and_then(|v| v.as_u64())
        .map(|id| id.to_string())
        .unwrap_or_default();
    let client_order_id = order
        .get("c")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let side = order
        .get("S")
        .and_then(|v| v.as_str())
        .unwrap_or("BUY")
        .to_string();
    let status = order
        .get("X")
        .and_then(|v| v.as_str())
        .unwrap_or("NEW")
        .to_string();
    let filled_qty = order
        .get("z")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let avg_price = order
        .get("ap")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let fee = order
        .get("n")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let fee_asset = order
        .get("N")
        .and_then(|v| v.as_str())
        .unwrap_or("USDT")
        .to_string();
    let is_maker = order.get("m").and_then(|v| v.as_bool()).unwrap_or(false);
    let event_time_ms = order
        .get("T")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    // Calculate receive latency
    let event_time_us = event_time_ms * 1000;
    let receive_latency_us = if receive_time_us > event_time_us {
        (receive_time_us - event_time_us) as u64
    } else {
        0
    };

    Some(FillEvent {
        exchange: "binance".to_string(),
        symbol,
        order_id,
        client_order_id,
        side,
        status,
        filled_qty,
        avg_price,
        fee,
        fee_asset,
        is_maker,
        timestamp_ms: event_time_ms,
        receive_latency_us,
    })
}

/// Spawn a Binance user data stream WebSocket task.
///
/// This task:
/// 1. Creates a listen key via REST
/// 2. Connects to the WS endpoint with the listen key
/// 3. Parses ORDER_TRADE_UPDATE messages into FillEvents
/// 4. Pushes fills to the shared FillUpdateSink
/// 5. Sends keepalive pings every 25 minutes
/// 6. Reconnects on disconnect with exponential backoff
pub async fn spawn_binance_ws_fill_receiver(
    config: BinanceWsConfig,
    sink: FillUpdateSink,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;

    let mut backoff_ms = 1000u64;
    let max_backoff_ms = 60_000u64;

    loop {
        // Step 1: Create listen key
        let listen_key = match binance_create_listen_key(&config).await {
            Ok(key) => {
                info!("[binance-ws] Listen key created successfully");
                backoff_ms = 1000; // Reset backoff on success
                key
            }
            Err(e) => {
                error!("[binance-ws] Failed to create listen key: {}. Retrying in {}ms", e, backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                continue;
            }
        };

        // Step 2: Connect to WebSocket
        let ws_url = format!("{}/{}", config.ws_base_url(), listen_key);
        let (ws_stream, _) = match connect_async(&ws_url).await {
            Ok(conn) => {
                info!("[binance-ws] Connected to user data stream");
                conn
            }
            Err(e) => {
                error!("[binance-ws] WS connection failed: {}. Retrying in {}ms", e, backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                continue;
            }
        };

        let (mut ws_write, mut ws_read) = ws_stream.split();

        // Step 3: Spawn keepalive task (every 25 minutes)
        let keepalive_config = config.clone();
        let keepalive_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(25 * 60)).await;
                if let Err(e) = binance_keepalive_listen_key(&keepalive_config).await {
                    warn!("[binance-ws] Listen key keepalive failed: {}", e);
                } else {
                    info!("[binance-ws] Listen key keepalive sent");
                }
            }
        });

        // Step 4: Process messages
        let mut disconnect = false;
        while let Some(msg_result) = ws_read.next().await {
            match msg_result {
                Ok(msg) => {
                    if msg.is_text() {
                        let receive_time_us = crate::execution_gateway::now_us();
                        let text = msg.to_text().unwrap_or_default();
                        match serde_json::from_str::<serde_json::Value>(text) {
                            Ok(json) => {
                                if let Some(fill) = parse_binance_fill(&json, receive_time_us) {
                                    info!(
                                        "[binance-ws] Fill: {} {} {} qty={:.6} avg_price={:.2} status={} latency={}µs",
                                        fill.side, fill.symbol, fill.order_id,
                                        fill.filled_qty, fill.avg_price, fill.status,
                                        fill.receive_latency_us
                                    );
                                    sink.push(fill);
                                }
                            }
                            Err(e) => {
                                warn!("[binance-ws] Failed to parse message: {}", e);
                            }
                        }
                    } else if msg.is_ping() {
                        // Respond to ping with pong
                        if let Err(e) = ws_write
                            .send(tokio_tungstenite::tungstenite::Message::Pong(msg.into_data()))
                            .await
                        {
                            warn!("[binance-ws] Failed to send pong: {}", e);
                        }
                    } else if msg.is_close() {
                        info!("[binance-ws] Received close frame, reconnecting...");
                        disconnect = true;
                        break;
                    }
                }
                Err(e) => {
                    error!("[binance-ws] WebSocket error: {}. Reconnecting...", e);
                    disconnect = true;
                    break;
                }
            }
        }

        // Cleanup
        keepalive_handle.abort();

        if disconnect {
            info!("[binance-ws] Disconnected, reconnecting in {}ms...", backoff_ms);
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
        }
    }
}

// ---------------------------------------------------------------------------
// Bybit v5 Private WebSocket
// ---------------------------------------------------------------------------

/// Configuration for the Bybit WebSocket fill receiver.
#[derive(Debug, Clone)]
pub struct BybitWsConfig {
    /// API key for authentication.
    pub api_key: String,
    /// API secret for HMAC signature.
    pub api_secret: String,
    /// Whether to use testnet/demo endpoints.
    pub testnet: bool,
}

impl BybitWsConfig {
    /// Get the WebSocket URL for private stream.
    fn ws_url(&self) -> &str {
        if self.testnet {
            "wss://stream-demo.bybit.com/v5/private"
        } else {
            "wss://stream.bybit.com/v5/private"
        }
    }

    /// Build the authentication message for Bybit private WebSocket.
    ///
    /// Auth message format:
    /// ```json
    /// {
    ///   "op": "auth",
    ///   "args": ["<api_key>", <expires>, "<signature>"]
    /// }
    /// ```
    /// where signature = HMAC-SHA256("GET/realtime{expires}", api_secret)
    fn build_auth_message(&self) -> serde_json::Value {
        let expires = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64)
            + 10_000; // 10 seconds in the future

        let sign_input = format!("GET/realtime{}", expires);

        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let mut mac = Hmac::<Sha256>::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(sign_input.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        serde_json::json!({
            "op": "auth",
            "args": [self.api_key, expires, signature]
        })
    }

    /// Build the subscription message for order updates.
    fn build_subscribe_message() -> serde_json::Value {
        serde_json::json!({
            "op": "subscribe",
            "args": ["order"]
        })
    }
}

/// Parse a Bybit order update WebSocket message into a FillEvent.
///
/// Message format:
/// ```json
/// {
///   "topic": "order",
///   "data": [{
///     "symbol": "BTCUSDT",
///     "orderId": "1234567890",
///     "orderLinkId": "rte00000001",
///     "side": "Buy",
///     "orderStatus": "Filled",
///     "cumExecQty": "0.001",
///     "avgPrice": "60000.0",
///     "cumExecFee": "0.024",
///     "isLeverage": "1",
///     "updatedTime": "1234567890123"
///   }]
/// }
/// ```
pub fn parse_bybit_fill(msg: &serde_json::Value, receive_time_us: i64) -> Vec<FillEvent> {
    let mut fills = Vec::new();

    let topic = match msg.get("topic").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return fills,
    };

    if topic != "order" {
        return fills;
    }

    let data = match msg.get("data").and_then(|v| v.as_array()) {
        Some(d) => d,
        None => return fills,
    };

    for order in data {
        let symbol = order
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let order_id = order
            .get("orderId")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let client_order_id = order
            .get("orderLinkId")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let side = order
            .get("side")
            .and_then(|v| v.as_str())
            .unwrap_or("Buy")
            .to_string();
        let status = order
            .get("orderStatus")
            .and_then(|v| v.as_str())
            .unwrap_or("New")
            .to_string();
        let filled_qty = order
            .get("cumExecQty")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let avg_price = order
            .get("avgPrice")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let fee = order
            .get("cumExecFee")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let updated_time_ms = order
            .get("updatedTime")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        // Calculate receive latency
        let event_time_us = updated_time_ms * 1000;
        let receive_latency_us = if receive_time_us > event_time_us {
            (receive_time_us - event_time_us) as u64
        } else {
            0
        };

        fills.push(FillEvent {
            exchange: "bybit".to_string(),
            symbol,
            order_id,
            client_order_id,
            side,
            status,
            filled_qty,
            avg_price,
            fee,
            fee_asset: "USDT".to_string(),
            is_maker: false, // Bybit doesn't include this in WS updates
            timestamp_ms: updated_time_ms,
            receive_latency_us,
        });
    }

    fills
}

/// Spawn a Bybit private WebSocket task for fill updates.
///
/// This task:
/// 1. Connects to the Bybit v5 private WebSocket
/// 2. Authenticates with HMAC-SHA256
/// 3. Subscribes to the "order" topic
/// 4. Parses order updates into FillEvents
/// 5. Pushes fills to the shared FillUpdateSink
/// 6. Reconnects on disconnect with exponential backoff
pub async fn spawn_bybit_ws_fill_receiver(
    config: BybitWsConfig,
    sink: FillUpdateSink,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;

    let mut backoff_ms = 1000u64;
    let max_backoff_ms = 60_000u64;

    loop {
        // Step 1: Connect to WebSocket
        let (ws_stream, _) = match connect_async(config.ws_url()).await {
            Ok(conn) => {
                info!("[bybit-ws] Connected to private stream");
                backoff_ms = 1000; // Reset backoff on success
                conn
            }
            Err(e) => {
                error!("[bybit-ws] WS connection failed: {}. Retrying in {}ms", e, backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                continue;
            }
        };

        let (mut ws_write, mut ws_read) = ws_stream.split();

        // Step 2: Authenticate
        let auth_msg = config.build_auth_message();
        let auth_text = serde_json::to_string(&auth_msg).unwrap_or_default();
        if let Err(e) = ws_write
            .send(tokio_tungstenite::tungstenite::Message::Text(auth_text.into()))
            .await
        {
            error!("[bybit-ws] Failed to send auth message: {}", e);
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
            continue;
        }

        // Wait for auth response
        let mut authenticated = false;
        if let Some(Ok(msg)) = ws_read.next().await {
            if msg.is_text() {
                let text = msg.to_text().unwrap_or_default();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                    let success = json.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
                    let op = json.get("op").and_then(|v| v.as_str()).unwrap_or_default();
                    if op == "auth" && success {
                        info!("[bybit-ws] Authentication successful");
                        authenticated = true;
                    } else {
                        let ret_msg = json.get("ret_msg").and_then(|v| v.as_str()).unwrap_or("unknown");
                        error!("[bybit-ws] Authentication failed: {}", ret_msg);
                    }
                }
            }
        }

        if !authenticated {
            error!("[bybit-ws] Authentication failed, retrying in {}ms", backoff_ms);
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
            continue;
        }

        // Step 3: Subscribe to order topic
        let sub_msg = BybitWsConfig::build_subscribe_message();
        let sub_text = serde_json::to_string(&sub_msg).unwrap_or_default();
        if let Err(e) = ws_write
            .send(tokio_tungstenite::tungstenite::Message::Text(sub_text.into()))
            .await
        {
            error!("[bybit-ws] Failed to subscribe: {}", e);
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
            continue;
        }

        info!("[bybit-ws] Subscribed to order topic");

        // Step 4: Spawn heartbeat task (every 20 seconds)
        let ws_write_shared = Arc::new(tokio::sync::Mutex::new(ws_write));
        let heartbeat_writer = ws_write_shared.clone();
        let heartbeat_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(20)).await;
                let ping_msg = serde_json::json!({"op": "ping"});
                let ping_text = serde_json::to_string(&ping_msg).unwrap_or_default();
                let mut writer = heartbeat_writer.lock().await;
                if let Err(e) = writer
                    .send(tokio_tungstenite::tungstenite::Message::Text(ping_text.into()))
                    .await
                {
                    warn!("[bybit-ws] Heartbeat ping failed: {}", e);
                    break;
                }
            }
        });

        // Step 5: Process messages
        let mut disconnect = false;
        while let Some(msg_result) = ws_read.next().await {
            match msg_result {
                Ok(msg) => {
                    if msg.is_text() {
                        let receive_time_us = crate::execution_gateway::now_us();
                        let text = msg.to_text().unwrap_or_default();
                        match serde_json::from_str::<serde_json::Value>(text) {
                            Ok(json) => {
                                // Skip pong responses and subscription confirmations
                                let op = json.get("op").and_then(|v| v.as_str()).unwrap_or_default();
                                if op == "pong" || op == "subscribe" {
                                    continue;
                                }

                                let bybit_fills = parse_bybit_fill(&json, receive_time_us);
                                for fill in bybit_fills {
                                    info!(
                                        "[bybit-ws] Fill: {} {} {} qty={:.6} avg_price={:.2} status={} latency={}µs",
                                        fill.side, fill.symbol, fill.order_id,
                                        fill.filled_qty, fill.avg_price, fill.status,
                                        fill.receive_latency_us
                                    );
                                    sink.push(fill);
                                }
                            }
                            Err(e) => {
                                warn!("[bybit-ws] Failed to parse message: {}", e);
                            }
                        }
                    } else if msg.is_ping() {
                        let mut writer = ws_write_shared.lock().await;
                        if let Err(e) = writer
                            .send(tokio_tungstenite::tungstenite::Message::Pong(msg.into_data()))
                            .await
                        {
                            warn!("[bybit-ws] Failed to send pong: {}", e);
                        }
                    } else if msg.is_close() {
                        info!("[bybit-ws] Received close frame, reconnecting...");
                        disconnect = true;
                        break;
                    }
                }
                Err(e) => {
                    error!("[bybit-ws] WebSocket error: {}. Reconnecting...", e);
                    disconnect = true;
                    break;
                }
            }
        }

        // Cleanup
        heartbeat_handle.abort();

        if disconnect {
            info!("[bybit-ws] Disconnected, reconnecting in {}ms...", backoff_ms);
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_sink_push_drain() {
        let sink = FillUpdateSink::new();
        assert!(!sink.has_pending());

        sink.push(FillEvent {
            exchange: "binance".to_string(),
            symbol: "BTCUSDT".to_string(),
            order_id: "123".to_string(),
            client_order_id: "rte001".to_string(),
            side: "Buy".to_string(),
            status: "Filled".to_string(),
            filled_qty: 0.001,
            avg_price: 60000.0,
            fee: 0.024,
            fee_asset: "USDT".to_string(),
            is_maker: false,
            timestamp_ms: 1000000,
            receive_latency_us: 500,
        });

        assert!(sink.has_pending());
        let (total, binance, bybit) = sink.stats();
        assert_eq!(total, 1);
        assert_eq!(binance, 1);
        assert_eq!(bybit, 0);

        let fills = sink.drain();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].exchange, "binance");
        assert!(!sink.has_pending());
    }

    #[test]
    fn test_parse_binance_fill() {
        let msg = serde_json::json!({
            "e": "ORDER_TRADE_UPDATE",
            "T": 1234567890123i64,
            "o": {
                "s": "BTCUSDT",
                "c": "rte0000000000000001",
                "S": "BUY",
                "i": 12345678,
                "X": "FILLED",
                "z": "0.001",
                "ap": "60000.0",
                "n": "0.024",
                "N": "USDT",
                "m": true,
                "T": 1234567890123i64
            }
        });

        let fill = parse_binance_fill(&msg, 1234567890123000 + 5000);
        assert!(fill.is_some());
        let fill = fill.unwrap();
        assert_eq!(fill.exchange, "binance");
        assert_eq!(fill.symbol, "BTCUSDT");
        assert_eq!(fill.side, "BUY");
        assert_eq!(fill.status, "FILLED");
        assert!((fill.filled_qty - 0.001).abs() < 1e-9);
        assert!((fill.avg_price - 60000.0).abs() < 1e-6);
        assert!(fill.is_maker);
    }

    #[test]
    fn test_parse_binance_fill_wrong_event() {
        let msg = serde_json::json!({
            "e": "ACCOUNT_UPDATE",
            "T": 1234567890123i64
        });
        assert!(parse_binance_fill(&msg, 0).is_none());
    }

    #[test]
    fn test_parse_bybit_fill() {
        let msg = serde_json::json!({
            "topic": "order",
            "data": [{
                "symbol": "BTCUSDT",
                "orderId": "1234567890",
                "orderLinkId": "rte00000001",
                "side": "Buy",
                "orderStatus": "Filled",
                "cumExecQty": "0.001",
                "avgPrice": "60000.0",
                "cumExecFee": "0.024",
                "updatedTime": "1234567890123"
            }]
        });

        let fills = parse_bybit_fill(&msg, 1234567890123000 + 5000);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].exchange, "bybit");
        assert_eq!(fills[0].symbol, "BTCUSDT");
        assert_eq!(fills[0].side, "Buy");
        assert_eq!(fills[0].status, "Filled");
        assert!((fills[0].filled_qty - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_parse_bybit_fill_wrong_topic() {
        let msg = serde_json::json!({
            "topic": "position",
            "data": []
        });
        assert!(parse_bybit_fill(&msg, 0).is_empty());
    }
}
