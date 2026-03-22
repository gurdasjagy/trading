//! WebSocket-based Order Management — Issue 3.
//!
//! Replaces REST-based order submission (gateio_gateway.rs uses reqwest HTTP)
//! with WebSocket-based order management for sub-millisecond execution.
//!
//! Gate.io supports order operations via the same WS connection used for
//! market data. This eliminates DNS lookup, TCP handshake, and TLS overhead.
//!
//! # Order Flow
//!
//! 1. Execution router calls `submit_order_ws()` with an `OrderCommand`
//! 2. WsOrderManager builds the WS JSON message and assigns a `client_order_id`
//! 3. Message is sent via the WS write channel (SPSC to WS ingestion thread)
//! 4. WS ingestion thread sends it on the authenticated connection
//! 5. Exchange responds with confirmation/rejection via WS
//! 6. Response is forwarded back to execution router via SPSC
//! 7. Execution router updates `OrderLifecycle` state machine
//!
//! # Authentication
//!
//! Gate.io WS authentication uses HMAC-SHA512 signed with the API secret.
//! The auth token is pre-computed once at startup to avoid hot-path crypto.

use std::collections::HashMap;

use crate::execution_state::{OrderLifecycle, CancelReason};
use crate::fixed_point::FixedPrice;
use crate::spsc::OrderCommand;

// ═══════════════════════════════════════════════════════════════════════════
// WS Order Message Types
// ═══════════════════════════════════════════════════════════════════════════

/// A WS order message to be sent to the exchange.
#[derive(Debug, Clone)]
pub struct WsOrderMessage {
    /// The JSON payload to send via WebSocket.
    pub payload: String,
    /// Client-assigned order ID for correlation.
    pub client_id: String,
    /// Timestamp when this message was created.
    pub created_ns: u64,
}

/// Response from the exchange via WebSocket.
#[derive(Debug, Clone)]
pub struct WsOrderResponse {
    /// Client-assigned order ID for correlation.
    pub client_id: String,
    /// Exchange-assigned order ID (empty if rejected).
    pub exchange_order_id: String,
    /// Whether the order was accepted.
    pub accepted: bool,
    /// Rejection reason (if rejected).
    pub rejection_reason: Option<String>,
    /// Rejection code (if rejected).
    pub rejection_code: u32,
    /// Timestamp when we received this response.
    pub received_ns: u64,
}

/// A fill notification from the exchange.
#[derive(Debug, Clone)]
pub struct WsFillNotification {
    /// Exchange-assigned order ID.
    pub exchange_order_id: String,
    /// Fill price in FixedPrice.
    pub fill_price_fp: i64,
    /// Fill quantity.
    pub fill_qty: i64,
    /// Fee for this fill.
    pub fee_fp: i64,
    /// Whether this is the final fill (order fully filled).
    pub is_final: bool,
    /// Timestamp when we received this notification.
    pub received_ns: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// OrderCallback — pending order tracking
// ═══════════════════════════════════════════════════════════════════════════

/// Callback state for a pending order confirmation.
struct OrderCallback {
    /// The OrderLifecycle being tracked.
    pub lifecycle_idx: usize,
    /// Timestamp when the order was submitted.
    pub submit_ts_ns: u64,
    /// Symbol ID.
    pub symbol_id: u16,
    /// Side: 0 = buy, 1 = sell.
    pub side: u8,
}

// ═══════════════════════════════════════════════════════════════════════════
// WsOrderManager
// ═══════════════════════════════════════════════════════════════════════════

/// WS-based order management for Gate.io.
///
/// Runs on Core 6 (Execution Router). Builds WS messages for order
/// submission and cancellation, tracks pending confirmations, and
/// processes responses from the WS feed.
pub struct WsOrderManager {
    /// Pre-signed authentication header (computed once at startup).
    auth_token: String,
    /// API key for the exchange.
    api_key: String,
    /// Pending order confirmations: client_order_id → callback.
    pending: HashMap<String, OrderCallback>,
    /// Sequence counter for client order IDs.
    client_seq: u64,
    /// Active order lifecycles (indexed by position).
    pub lifecycles: Vec<OrderLifecycle>,
    /// Outbound message queue: messages to send via WS.
    /// The WS ingestion thread reads from this.
    pub outbound_queue: Vec<WsOrderMessage>,
    /// Total orders submitted.
    total_submitted: u64,
    /// Total orders confirmed.
    total_confirmed: u64,
    /// Total orders rejected.
    total_rejected: u64,
    /// Total orders canceled.
    total_canceled: u64,
}

impl WsOrderManager {
    /// Create a new WsOrderManager.
    pub fn new(api_key: String, api_secret: String) -> Self {
        // Pre-compute auth token (done once at startup, not on hot path).
        // Gate.io WS auth uses channel "futures.login" with HMAC-SHA512.
        let auth_token = Self::compute_auth_token(&api_key, &api_secret);

        Self {
            auth_token,
            api_key,
            pending: HashMap::with_capacity(64),
            client_seq: 0,
            lifecycles: Vec::with_capacity(64),
            outbound_queue: Vec::with_capacity(32),
            total_submitted: 0,
            total_confirmed: 0,
            total_rejected: 0,
            total_canceled: 0,
        }
    }

    /// Create with empty credentials (for testing / signal-only mode).
    pub fn new_paper() -> Self {
        Self {
            auth_token: String::new(),
            api_key: String::new(),
            pending: HashMap::with_capacity(16),
            client_seq: 0,
            lifecycles: Vec::with_capacity(16),
            outbound_queue: Vec::with_capacity(8),
            total_submitted: 0,
            total_confirmed: 0,
            total_rejected: 0,
            total_canceled: 0,
        }
    }

    /// Submit an order via WebSocket. Returns the client_order_id.
    ///
    /// The actual JSON message is queued in `outbound_queue` for the
    /// WS ingestion thread to send on the authenticated connection.
    pub fn submit_order_ws(&mut self, cmd: &OrderCommand, symbol_name: &str) -> String {
        self.client_seq += 1;
        let client_id = format!("t{}", self.client_seq);
        let now = now_ns();

        // Determine price and TIF
        let price_f64 = FixedPrice(cmd.price).to_f64();
        let tif = if cmd.order_type == 2 { "poc" } else { "gtc" };
        let size = if cmd.side == 0 {
            cmd.qty // positive for buy
        } else {
            -cmd.qty // negative for sell (Gate.io convention)
        };

        // Build the WS order message (Gate.io futures.order_place)
        let payload = serde_json::json!({
            "time": now_secs(),
            "channel": "futures.order_place",
            "event": "api",
            "payload": {
                "req_id": &client_id,
                "contract": symbol_name,
                "size": size,
                "price": format!("{:.8}", price_f64),
                "tif": tif,
                "reduce_only": false,
            }
        });

        let msg = WsOrderMessage {
            payload: payload.to_string(),
            client_id: client_id.clone(),
            created_ns: now,
        };

        self.outbound_queue.push(msg);

        // Create lifecycle entry
        let lifecycle_idx = self.lifecycles.len();
        let mut lifecycle = OrderLifecycle::new(cmd.symbol_id, cmd.side, self.client_seq);
        let _ = lifecycle.submit(&mut None);
        self.lifecycles.push(lifecycle);

        // Track pending confirmation
        self.pending.insert(
            client_id.clone(),
            OrderCallback {
                lifecycle_idx,
                submit_ts_ns: now,
                symbol_id: cmd.symbol_id,
                side: cmd.side,
            },
        );

        self.total_submitted += 1;
        client_id
    }

    /// Cancel an order via WebSocket. Returns the client cancel ID.
    pub fn cancel_order_ws(&mut self, exchange_order_id: &str) -> String {
        self.client_seq += 1;
        let client_id = format!("c{}", self.client_seq);
        let now = now_ns();

        let payload = serde_json::json!({
            "time": now_secs(),
            "channel": "futures.order_cancel",
            "event": "api",
            "payload": {
                "req_id": &client_id,
                "order_id": exchange_order_id,
            }
        });

        let msg = WsOrderMessage {
            payload: payload.to_string(),
            client_id: client_id.clone(),
            created_ns: now,
        };

        self.outbound_queue.push(msg);
        self.total_canceled += 1;
        client_id
    }

    /// Build the WS authentication message for Gate.io futures.
    pub fn build_auth_message(&self) -> String {
        serde_json::json!({
            "time": now_secs(),
            "channel": "futures.login",
            "event": "api",
            "payload": {
                "api_key": &self.api_key,
                "sign": &self.auth_token,
                "timestamp": now_secs().to_string(),
            }
        })
        .to_string()
    }

    /// Process a WS order response (confirmation or rejection).
    pub fn on_order_response(&mut self, response: &WsOrderResponse) {
        if let Some(callback) = self.pending.remove(&response.client_id) {
            if callback.lifecycle_idx < self.lifecycles.len() {
                let lifecycle = &mut self.lifecycles[callback.lifecycle_idx];

                if response.accepted {
                    // Convert exchange order ID to [u8; 32]
                    let mut oid = [0u8; 32];
                    let bytes = response.exchange_order_id.as_bytes();
                    let len = bytes.len().min(32);
                    oid[..len].copy_from_slice(&bytes[..len]);

                    let _ = lifecycle.on_placed(
                        oid,
                        0, // Price will be updated from book state
                        0, // Size from original command
                        &mut None,
                    );
                    self.total_confirmed += 1;
                } else {
                    let _ = lifecycle.on_rejected(
                        response.rejection_code,
                        &mut None,
                    );
                    self.total_rejected += 1;
                }
            }
        }
    }

    /// Process a fill notification from the WS feed.
    pub fn on_fill_notification(&mut self, fill: &WsFillNotification) {
        // Find the lifecycle by exchange order ID
        for lifecycle in &mut self.lifecycles {
            if let Some(oid) = lifecycle.order_id() {
                let oid_str = String::from_utf8_lossy(oid).trim_end_matches('\0').to_string();
                if oid_str == fill.exchange_order_id {
                    if fill.is_final {
                        let _ = lifecycle.on_filled(
                            fill.fill_price_fp,
                            fill.fill_qty,
                            fill.fee_fp,
                            &mut None,
                        );
                    } else {
                        let _ = lifecycle.on_partial_fill(fill.fill_qty);
                    }
                    break;
                }
            }
        }
    }

    /// Cancel a resting order by lifecycle index (used by adverse selection detector).
    pub fn cancel_by_lifecycle_idx(&mut self, idx: usize, reason: CancelReason) -> Option<String> {
        if idx >= self.lifecycles.len() {
            return None;
        }

        let lifecycle = &mut self.lifecycles[idx];
        if !lifecycle.is_resting() {
            return None;
        }

        // Get the exchange order ID
        let exchange_oid = lifecycle.order_id().map(|oid| {
            String::from_utf8_lossy(oid).trim_end_matches('\0').to_string()
        })?;

        // Transition to Canceling state
        let _ = lifecycle.start_cancel(reason);

        // Send cancel via WS
        Some(self.cancel_order_ws(&exchange_oid))
    }

    /// Drain the outbound message queue. Returns all pending messages.
    pub fn drain_outbound(&mut self) -> Vec<WsOrderMessage> {
        std::mem::take(&mut self.outbound_queue)
    }

    /// Get count of pending (unconfirmed) orders.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get count of active (non-terminal) lifecycles.
    pub fn active_lifecycle_count(&self) -> usize {
        self.lifecycles.iter().filter(|l| !l.is_terminal()).count()
    }

    /// Get count of resting orders.
    pub fn resting_count(&self) -> usize {
        self.lifecycles.iter().filter(|l| l.is_resting()).count()
    }

    /// Cleanup terminal lifecycles (keep history bounded).
    pub fn cleanup_terminal(&mut self, keep_last: usize) {
        if self.lifecycles.len() > keep_last * 2 {
            // Only remove terminal ones from the front
            self.lifecycles.retain(|l| !l.is_terminal());
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> WsOrderManagerStats {
        WsOrderManagerStats {
            total_submitted: self.total_submitted,
            total_confirmed: self.total_confirmed,
            total_rejected: self.total_rejected,
            total_canceled: self.total_canceled,
            pending: self.pending.len() as u64,
            active_lifecycles: self.active_lifecycle_count() as u64,
            resting: self.resting_count() as u64,
        }
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    /// Compute the authentication token for Gate.io WS.
    /// Uses HMAC-SHA512 signature of "channel=futures.login&event=api&time=<ts>".
    fn compute_auth_token(api_key: &str, api_secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha512;

        if api_key.is_empty() || api_secret.is_empty() {
            return String::new();
        }

        let ts = now_secs();
        let message = format!("channel=futures.login&event=api&time={}", ts);

        let mut mac = Hmac::<Sha512>::new_from_slice(api_secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(message.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

/// Statistics for the WsOrderManager.
#[derive(Debug, Clone)]
pub struct WsOrderManagerStats {
    pub total_submitted: u64,
    pub total_confirmed: u64,
    pub total_rejected: u64,
    pub total_canceled: u64,
    pub pending: u64,
    pub active_lifecycles: u64,
    pub resting: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[inline]
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spsc::{self, OrderCommand};

    fn make_order_cmd(symbol_id: u16, side: u8, price: i64, qty: i64) -> OrderCommand {
        OrderCommand {
            symbol_id,
            side,
            order_type: spsc::order_cmd_type::LIMIT,
            leverage: 0,
            _pad: [0; 3],
            price,
            qty,
            order_id: 1,
            signal_ns: 0,
            max_slippage_bps: 50,
            ttl_ms: 5000,
            stop_loss_fp: (price as f64 * 0.99) as i64,
            take_profit_fp: (price as f64 * 1.02) as i64,
            placement_type: 0,
            post_only: 0,
            is_close: 0,
            _pad2: [0; 5],
        }
    }

    #[test]
    fn test_submit_order_creates_lifecycle() {
        let mut mgr = WsOrderManager::new_paper();

        let cmd = make_order_cmd(1, 0, 5000_00000000, 10_0000);
        let client_id = mgr.submit_order_ws(&cmd, "BTC_USDT");

        assert!(client_id.starts_with("t"));
        assert_eq!(mgr.pending_count(), 1);
        assert_eq!(mgr.active_lifecycle_count(), 1);
        assert_eq!(mgr.outbound_queue.len(), 1);
    }

    #[test]
    fn test_order_confirmation() {
        let mut mgr = WsOrderManager::new_paper();

        let cmd = make_order_cmd(1, 0, 5000_00000000, 10_0000);
        let client_id = mgr.submit_order_ws(&cmd, "BTC_USDT");

        // Simulate exchange confirmation
        let response = WsOrderResponse {
            client_id: client_id.clone(),
            exchange_order_id: "12345".to_string(),
            accepted: true,
            rejection_reason: None,
            rejection_code: 0,
            received_ns: now_ns(),
        };
        mgr.on_order_response(&response);

        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.resting_count(), 1);
        assert_eq!(mgr.stats().total_confirmed, 1);
    }

    #[test]
    fn test_order_rejection() {
        let mut mgr = WsOrderManager::new_paper();

        let cmd = make_order_cmd(1, 0, 5000_00000000, 10_0000);
        let client_id = mgr.submit_order_ws(&cmd, "BTC_USDT");

        let response = WsOrderResponse {
            client_id: client_id.clone(),
            exchange_order_id: String::new(),
            accepted: false,
            rejection_reason: Some("Insufficient balance".to_string()),
            rejection_code: 110007,
            received_ns: now_ns(),
        };
        mgr.on_order_response(&response);

        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.resting_count(), 0);
        assert_eq!(mgr.stats().total_rejected, 1);
    }

    #[test]
    fn test_cancel_resting_order() {
        let mut mgr = WsOrderManager::new_paper();

        let cmd = make_order_cmd(1, 0, 5000_00000000, 10_0000);
        let client_id = mgr.submit_order_ws(&cmd, "BTC_USDT");

        // Confirm the order
        mgr.on_order_response(&WsOrderResponse {
            client_id: client_id.clone(),
            exchange_order_id: "12345".to_string(),
            accepted: true,
            rejection_reason: None,
            rejection_code: 0,
            received_ns: now_ns(),
        });

        // Cancel via adverse selection
        let cancel_id = mgr.cancel_by_lifecycle_idx(0, CancelReason::AdverseSelection);
        assert!(cancel_id.is_some());
        assert!(cancel_id.unwrap().starts_with("c"));
    }

    #[test]
    fn test_drain_outbound() {
        let mut mgr = WsOrderManager::new_paper();

        let cmd = make_order_cmd(1, 0, 5000_00000000, 10_0000);
        mgr.submit_order_ws(&cmd, "BTC_USDT");
        mgr.submit_order_ws(&cmd, "ETH_USDT");

        let messages = mgr.drain_outbound();
        assert_eq!(messages.len(), 2);
        assert!(mgr.outbound_queue.is_empty());
    }

    #[test]
    fn test_fill_notification() {
        let mut mgr = WsOrderManager::new_paper();

        let cmd = make_order_cmd(1, 0, 5000_00000000, 10_0000);
        let client_id = mgr.submit_order_ws(&cmd, "BTC_USDT");

        // Confirm
        mgr.on_order_response(&WsOrderResponse {
            client_id,
            exchange_order_id: "12345".to_string(),
            accepted: true,
            rejection_reason: None,
            rejection_code: 0,
            received_ns: now_ns(),
        });

        // Fill
        mgr.on_fill_notification(&WsFillNotification {
            exchange_order_id: "12345".to_string(),
            fill_price_fp: 5000_00000000,
            fill_qty: 10_0000,
            fee_fp: 500,
            is_final: true,
            received_ns: now_ns(),
        });

        // Should be terminal now
        assert!(mgr.lifecycles[0].is_terminal());
    }
}

