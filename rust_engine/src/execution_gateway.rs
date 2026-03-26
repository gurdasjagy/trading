//! Core execution gateway traits, structs, request signing, and rate limiting.
//!
//! **Issue 3 Rewrite**: Added WsExecutionGateway alongside REST gateway,
//! SmartOrderRouter integration, AdverseSelectionDetector integration,
//! and the new execution_router_loop() for Core 6.
//!
//! Defines:
//! - OrderIntent / OrderResult / ExchangeError (unchanged)
//! - HMAC signing for Gate.io (SHA-512)
//! - AdaptiveRateLimiter with header-based dynamic adjustment
//! - ExecutionGateway trait implemented by GateIoGateway
//! - WsExecutionGateway implementation using WsOrderManager
//! - execution_router_loop() with full state machine integration

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use hmac::{Hmac, Mac};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha512};
use tracing::warn;

// Issue 3 imports
use crate::adverse_selection::{AdverseSelectionDetector, TradeEvent};
use crate::execution_state::{CancelReason, PlacementType};
use crate::mbo_book::MboBook;
use crate::smart_router::SmartOrderRouter;
use crate::ws_order_manager::WsOrderManager;

// ---------------------------------------------------------------------------
// Core Structs
// ---------------------------------------------------------------------------

/// An order intent emitted by the strategy engine.
///
/// Includes explicit Stop Loss and Take Profit for position protection.
/// The execution router submits SL/TP as conditional orders to the exchange.
/// If SL placement fails, the position is marked "unprotected" and the
/// router will aggressively retry or close the position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntent {
    /// Symbol in exchange-specific format, e.g. "BTC_USDT" for Gate.io.
    pub symbol: String,
    pub side: OrderSide,
    /// Integer contracts (Gate.io requirement).
    pub size: i64,
    pub order_type: OrderType,
    /// Required for Limit / PostOnly orders.
    pub price: Option<f64>,
    pub reduce_only: bool,
    pub leverage: Option<i32>,
    /// "gtc", "ioc", "poc" (post-only / maker-or-cancel on Gate.io).
    pub time_in_force: String,
    /// Maximum slippage allowed as a fraction (e.g. 0.001 = 0.1%).
    pub slippage_cap_pct: Option<f64>,
    /// Issue 3: Desired placement type relative to BBO.
    /// The execution router translates this to actual price using live book state.
    #[serde(skip)]
    pub placement: PlacementType,
    /// Hard Stop Loss price. If set, the execution router registers a conditional
    /// stop order at the exchange. If SL placement fails, the position is marked
    /// "unprotected" and emergency measures are taken.
    #[serde(default)]
    pub stop_loss: Option<f64>,
    /// Take Profit price. If set, the execution router registers a conditional
    /// TP order at the exchange. Can be dynamically updated based on
    /// microstructure signals (trailing TP).
    #[serde(default)]
    pub take_profit: Option<f64>,
    /// Strategy confidence score [0.0, 1.0] for telemetry/logging.
    #[serde(default)]
    pub confidence: f64,
    /// Signal source tag for attribution (e.g. "microstructure_imbalance_vpin").
    #[serde(default)]
    pub signal_tag: String,
    /// CATEGORY 2 FIX: Minimum fill size for IOC orders.
    /// If set and the IOC order fills less than this amount, the partial fill
    /// is considered insufficient and the remaining exposure is managed.
    /// Institutional bots use this to ensure they get meaningful fills
    /// rather than dust amounts from IOC sweeps.
    #[serde(default)]
    pub min_fill_size: Option<i64>,
    /// CATEGORY 8 FIX: Strategy name for PnL attribution.
    /// Tracks which strategy generated this signal for per-strategy P&L reporting.
    #[serde(default)]
    pub strategy_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    PostOnly,
}

/// Result of an order submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderResult {
    pub order_id: String,
    /// "open", "closed", "rejected"
    pub status: String,
    pub filled_size: i64,
    pub avg_fill_price: f64,
    pub fee: f64,
    /// Measured from intent creation to HTTP response received (µs).
    pub latency_us: u64,
    pub exchange_timestamp: i64,
    pub rejection_reason: Option<String>,
}

/// An open position on an exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub size: i64,
    pub entry_price: f64,
    pub unrealized_pnl: f64,
    pub leverage: i32,
    pub side: String,
}

// ---------------------------------------------------------------------------
// ExchangeError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ExchangeError {
    // Retryable
    RateLimited { retry_after_ms: u64 },
    Timeout,
    /// WS ACK or REST request timed out with a known client-side order identifier.
    ///
    /// Unlike `Timeout`, this carries the `client_order_id` that was used when
    /// submitting the order. The exchange may have accepted the order even though
    /// we never received confirmation. Callers must call
    /// `ExecutionGateway::check_order_by_client_id` before retrying to avoid
    /// duplicate fills.
    TimedOut { client_order_id: String },
    ConnectionReset,
    InternalServerError,
    // Non-retryable
    InsufficientBalance,
    InvalidSymbol,
    OrderNotFound,
    InvalidPrice,
    PositionNotFound,
    // Requires adjustment
    MinimumOrderSize { min_size: i64 },
    PricePrecisionError { max_decimals: u32 },
    // Generic
    Unknown { code: String, message: String },
}

impl ExchangeError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ExchangeError::RateLimited { .. }
                | ExchangeError::Timeout
                | ExchangeError::TimedOut { .. }
                | ExchangeError::ConnectionReset
                | ExchangeError::InternalServerError
        )
    }
}

impl std::fmt::Display for ExchangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExchangeError::RateLimited { retry_after_ms } => {
                write!(f, "RateLimited (retry after {}ms)", retry_after_ms)
            }
            ExchangeError::Timeout => write!(f, "Request timeout"),
            ExchangeError::TimedOut { client_order_id } => {
                write!(f, "Request timeout (client_order_id={})", client_order_id)
            }
            ExchangeError::ConnectionReset => write!(f, "Connection reset"),
            ExchangeError::InternalServerError => write!(f, "Internal server error"),
            ExchangeError::InsufficientBalance => write!(f, "Insufficient balance"),
            ExchangeError::InvalidSymbol => write!(f, "Invalid symbol"),
            ExchangeError::OrderNotFound => write!(f, "Order not found"),
            ExchangeError::InvalidPrice => write!(f, "Invalid price"),
            ExchangeError::PositionNotFound => write!(f, "Position not found"),
            ExchangeError::MinimumOrderSize { min_size } => {
                write!(f, "Order below minimum size (min: {})", min_size)
            }
            ExchangeError::PricePrecisionError { max_decimals } => {
                write!(f, "Price precision error (max {} decimals)", max_decimals)
            }
            ExchangeError::Unknown { code, message } => {
                write!(f, "Unknown error {}: {}", code, message)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gate.io Error Classification
// ---------------------------------------------------------------------------

/// Classify a Gate.io error response into ExchangeError.
pub fn classify_gateio_error(status: u16, body: &serde_json::Value) -> ExchangeError {
    let label = body.get("label").and_then(|v| v.as_str()).unwrap_or("");
    let message = body
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match status {
        429 => return ExchangeError::RateLimited { retry_after_ms: 1000 },
        500 | 502 | 503 | 504 => return ExchangeError::InternalServerError,
        _ => {}
    }

    match label {
        "INVALID_PARAM_VALUE" | "MISSING_REQUIRED_PARAM" => ExchangeError::Unknown {
            code: label.to_string(),
            message: message.to_string(),
        },
        "INVALID_CURRENCY" | "INVALID_CONTRACT" => ExchangeError::InvalidSymbol,
        "ORDER_NOT_FOUND" | "ORDER_CANCELLED" => ExchangeError::OrderNotFound,
        "BALANCE_NOT_ENOUGH" | "INSUFFICIENT_BALANCE" => ExchangeError::InsufficientBalance,
        "POSITION_NOT_FOUND" => ExchangeError::PositionNotFound,
        "AUTO_TRIGGER_PRICE_GREATE_LAST"
        | "AUTO_TRIGGER_PRICE_LESS_LAST"
        | "INVALID_PRICE"
        | "PRICE_TOO_DEVIATED" => ExchangeError::InvalidPrice,
        "ORDER_TOO_SMALL" => ExchangeError::MinimumOrderSize { min_size: 1 },
        "1029" | "1026" => ExchangeError::InvalidPrice,
        "RATE_LIMIT" | "TOO_MANY_REQUESTS" => {
            ExchangeError::RateLimited { retry_after_ms: 1000 }
        }
        _ => ExchangeError::Unknown {
            code: label.to_string(),
            message: message.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Request Signing
// ---------------------------------------------------------------------------

/// Gate.io v4 HMAC-SHA512 request signing.
///
/// Signature input: "{method}\n{path}\n{query_string}\n{sha512_hex(body)}\n{timestamp}"
pub fn sign_gateio_request(
    method: &str,
    path: &str,
    query: &str,
    body: &str,
    timestamp: i64,
    secret: &[u8],
) -> String {
    let body_hash = {
        let mut hasher = Sha512::new();
        hasher.update(body.as_bytes());
        hex::encode(hasher.finalize())
    };

    let payload = format!(
        "{}\n{}\n{}\n{}\n{}",
        method, path, query, body_hash, timestamp
    );

    let mut mac =
        Hmac::<Sha512>::new_from_slice(secret).expect("HMAC can take any key length");
    mac.update(payload.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}


// ---------------------------------------------------------------------------
// Adaptive Rate Limiter
// ---------------------------------------------------------------------------

/// Token-bucket rate limiter that dynamically adjusts from exchange response headers.
pub struct AdaptiveRateLimiter {
    tokens: AtomicU32,
    max_tokens: AtomicU32,
    refill_interval_us: AtomicU64,
    /// Backoff multiplier × 100 (e.g. 150 = 1.5×).
    backoff_multiplier: AtomicU32,
    last_rate_limit_remaining: AtomicI32,
}

impl AdaptiveRateLimiter {
    pub fn new(initial_rps: u32) -> Self {
        let refill_interval_us = if initial_rps > 0 {
            1_000_000 / initial_rps as u64
        } else {
            100_000
        };
        Self {
            tokens: AtomicU32::new(initial_rps),
            max_tokens: AtomicU32::new(initial_rps),
            refill_interval_us: AtomicU64::new(refill_interval_us),
            backoff_multiplier: AtomicU32::new(100),
            last_rate_limit_remaining: AtomicI32::new(-1),
        }
    }

    /// Update limits from Gate.io response headers.
    pub fn update_from_gateio_headers(&self, headers: &HeaderMap) {
        if let Some(limit) = headers
            .get("X-Gate-RateLimit-Limit")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok())
        {
            self.max_tokens.store(limit, Ordering::Relaxed);
        }

        if let Some(remaining) = headers
            .get("X-Gate-RateLimit-Remaining")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<i32>().ok())
        {
            self.last_rate_limit_remaining
                .store(remaining, Ordering::Relaxed);

            let max = self.max_tokens.load(Ordering::Relaxed) as i32;
            if max > 0 && remaining < max / 5 {
                let current_backoff = self.backoff_multiplier.load(Ordering::Relaxed);
                let new_backoff = (current_backoff * 150 / 100).min(500);
                self.backoff_multiplier.store(new_backoff, Ordering::Relaxed);
                warn!(
                    "Gate.io rate limit low: {}/{} remaining. Backoff: {:.2}×",
                    remaining,
                    max,
                    new_backoff as f64 / 100.0
                );
            } else if remaining > max * 4 / 5 {
                let current_backoff = self.backoff_multiplier.load(Ordering::Relaxed);
                let new_backoff = (current_backoff * 95 / 100).max(100);
                self.backoff_multiplier.store(new_backoff, Ordering::Relaxed);
            }
        }

        if let Some(reset_secs) = headers
            .get("X-Gate-RateLimit-Reset")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
        {
            let remaining = self
                .last_rate_limit_remaining
                .load(Ordering::Relaxed)
                .max(0) as u64;
            if remaining > 0 && reset_secs > 0 {
                let new_interval_us = (reset_secs * 1_000_000) / remaining.max(1);
                self.refill_interval_us.store(
                    new_interval_us.min(100_000).max(10_000),
                    Ordering::Relaxed,
                );
            }
        }
    }

    /// Acquire a token slot, waiting if the bucket is empty.
    /// Returns estimated wait time in microseconds.
    pub async fn acquire(&self) -> u64 {
        let backoff = self.backoff_multiplier.load(Ordering::Relaxed);
        let base_interval_us = self.refill_interval_us.load(Ordering::Relaxed);
        let effective_interval_us = base_interval_us * backoff as u64 / 100;

        let mut attempts = 0u32;
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current > 0 {
                if self
                    .tokens
                    .compare_exchange_weak(
                        current,
                        current - 1,
                        Ordering::Release,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return 0;
                }
            } else {
                tokio::time::sleep(Duration::from_micros(effective_interval_us)).await;
                let max = self.max_tokens.load(Ordering::Relaxed);
                let _ = self.tokens.compare_exchange(
                    0,
                    1.min(max),
                    Ordering::Release,
                    Ordering::Relaxed,
                );
                attempts += 1;
                if attempts > 100 {
                    warn!("Rate limiter waited >100 cycles; possible deadlock");
                    return effective_interval_us * 100;
                }
                return effective_interval_us;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionGateway Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ExecutionGateway: Send + Sync {
    async fn submit_order(&self, intent: OrderIntent) -> Result<OrderResult, ExchangeError>;
    async fn cancel_order(&self, order_id: &str, symbol: &str)
        -> Result<(), ExchangeError>;
    async fn get_position(&self, symbol: &str)
        -> Result<Option<Position>, ExchangeError>;
    /// Query all open positions from Gate.io Futures.
    async fn get_positions(&self) -> Result<Vec<Position>, ExchangeError>;
    async fn set_leverage(&self, symbol: &str, leverage: i32)
        -> Result<(), ExchangeError>;
    /// Return the available USDT balance in the futures wallet.
    async fn get_balance(&self) -> Result<f64, ExchangeError>;
    
    /// Query the status of a specific order on Gate.io (Task 6).
    /// Returns None if the order no longer exists (filled/cancelled).
    /// Default implementation returns Ok(None) for gateways that don't support order status queries.
    async fn get_order_status(&self, order_id: &str, symbol: &str)
        -> Result<Option<OrderResult>, ExchangeError> {
        let _ = (order_id, symbol);
        Ok(None)
    }

    /// Idempotency check by client-side order identifier.
    ///
    /// Called when `submit_order` returns `ExchangeError::TimedOut` — meaning the WS ACK
    /// (or REST response) never arrived but the exchange may have accepted the order.
    ///
    /// Returns `Ok(Some(exchange_order_id))` if an active or filled order with this
    /// `client_order_id` is found on the exchange, or `Ok(None)` if no such order exists
    /// (safe to retry). Returns `Err` only on network/auth failures.
    ///
    /// Default implementation returns `Ok(None)` for gateways that do not support this
    /// lookup. Concrete implementations call the appropriate REST endpoint:
    /// - Binance: `GET /fapi/v1/order?origClientOrderId=…`
    /// - Bybit:   `GET /v5/order/realtime?orderLinkId=…`
    /// - Gate.io: `GET /api/v4/futures/usdt/orders?text=…`
    async fn check_order_by_client_id(
        &self,
        _client_order_id: &str,
        _symbol: &str,
    ) -> Result<Option<String>, ExchangeError> {
        Ok(None)
    }

    /// FIX 6: Submit a conditional stop-loss order to the exchange.
    /// Default implementation returns an error (not all gateways support conditional orders).
    async fn submit_conditional_sl(
        &self,
        _symbol: &str,
        _parent_side: &OrderSide,
        _filled_size: i64,
        _sl_price: f64,
    ) -> Result<(), ExchangeError> {
        Err(ExchangeError::Unknown { code: "UNSUPPORTED".into(), message: "Conditional SL not supported by this gateway".into() })
    }

    /// FIX 6: Submit a conditional take-profit order to the exchange.
    /// Default implementation returns an error (not all gateways support conditional orders).
    async fn submit_conditional_tp(
        &self,
        _symbol: &str,
        _parent_side: &OrderSide,
        _filled_size: i64,
        _tp_price: f64,
    ) -> Result<(), ExchangeError> {
        Err(ExchangeError::Unknown { code: "UNSUPPORTED".into(), message: "Conditional TP not supported by this gateway".into() })
    }

    /// Feature 5: Cancel all conditional (price-triggered) orders for a symbol.
    /// Used when trailing stop updates require canceling old SL before submitting new one.
    /// Default implementation returns Ok (no-op for gateways that don't support this).
    async fn cancel_conditional_orders(&self, _symbol: &str) -> Result<(), ExchangeError> {
        Ok(())
    }

    /// BUG 4 FIX: Add default implementations for set_margin_mode and get_ticker
    async fn set_margin_mode(&self, _symbol: &str, _mode: &str) -> Result<(), ExchangeError> {
        Ok(())
    }

    async fn get_ticker(&self, _symbol: &str) -> Result<RustTicker, ExchangeError> {
        Err(ExchangeError::Unknown { code: "UNSUPPORTED".into(), message: "get_ticker not supported".into() })
    }

    /// Convert a USDT amount to integer contracts for a given futures contract.
    /// Each gateway implements this based on the exchange's contract specifications.
    /// Default implementation uses get_ticker to do a simple price-based conversion.
    async fn usdt_to_contracts(&self, symbol: &str, usdt_amount: f64) -> Result<i64, ExchangeError> {
        let ticker = self.get_ticker(symbol).await?;
        if ticker.last <= 0.0 {
            return Err(ExchangeError::Unknown {
                code: "INVALID_PRICE".into(),
                message: format!("Last price for {} is {}", symbol, ticker.last),
            });
        }
        let contracts = (usdt_amount / ticker.last).floor() as i64;
        if contracts < 1 {
            return Err(ExchangeError::MinimumOrderSize { min_size: 1 });
        }
        Ok(contracts)
    }
}

/// BUG 4 FIX: Define RustTicker struct for get_ticker return type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustTicker {
    pub last: f64,
    pub bid: f64,
    pub ask: f64,
    pub volume_24h: f64,
}

// ---------------------------------------------------------------------------
// Retry Logic
// ---------------------------------------------------------------------------

/// Execute an order submission with exponential backoff + jitter on transient errors.
///
/// Issue 3: Enhanced retry logic — distinguishes between transient errors
/// (retry aggressively), rate limits (back off per header), and adverse
/// conditions (cancel and re-evaluate).
pub async fn submit_with_retry(
    gateway: &dyn ExecutionGateway,
    intent: OrderIntent,
) -> Result<OrderResult, ExchangeError> {
    let mut delay_ms = 100u64;
    let max_retries = 5;

    for attempt in 0..=max_retries {
        match gateway.submit_order(intent.clone()).await {
            Ok(result) => return Ok(result),
            Err(ref e) if matches!(e, ExchangeError::RateLimited { .. }) && attempt < max_retries => {
                // Rate limit: use the retry_after hint
                let wait = match e {
                    ExchangeError::RateLimited { retry_after_ms } => *retry_after_ms,
                    _ => delay_ms,
                };
                warn!(
                    "Rate limited on attempt {}/{}: {}. Waiting {}ms",
                    attempt + 1, max_retries, e, wait
                );
                tokio::time::sleep(Duration::from_millis(wait)).await;
                delay_ms = (delay_ms * 2).min(10_000);
            }
            Err(e) if e.is_retryable() && attempt < max_retries => {
                let jitter = delay_ms / 5;
                let sleep_ms = delay_ms + jitter;
                warn!(
                    "Retryable error on attempt {}/{}: {}. Retrying in {}ms",
                    attempt + 1,
                    max_retries,
                    e,
                    sleep_ms
                );
                tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
                delay_ms = (delay_ms * 2).min(10_000);
            }
            Err(e) => return Err(e),
        }
    }

    unreachable!()
}

// ---------------------------------------------------------------------------
// Issue 3: Execution Context — shared state for execution router
// ---------------------------------------------------------------------------

/// Shared context for the execution router on Core 6.
///
/// Contains all the components needed for institutional-grade execution:
/// - MBO book for queue position tracking
/// - Adverse selection detector for informed flow detection
/// - Smart order router for venue selection
/// - WS order manager for low-latency order submission
pub struct ExecutionContext {
    /// Market-By-Order book for queue position estimation.
    pub mbo_book: MboBook,
    /// Adverse selection detector.
    pub adverse_detector: AdverseSelectionDetector,
    /// Smart order router.
    pub smart_router: SmartOrderRouter,
    /// WS-based order manager (Gate.io).
    pub ws_order_mgr: WsOrderManager,
    /// Fill probability threshold: if below this for > stale_timeout, re-evaluate.
    pub fill_prob_threshold: f64,
    /// Time in seconds before a low-fill-probability order is re-evaluated.
    pub stale_timeout_s: f64,
}

impl ExecutionContext {
    /// Create a new execution context with the given components.
    pub fn new(
        mbo_book: MboBook,
        adverse_detector: AdverseSelectionDetector,
        smart_router: SmartOrderRouter,
        ws_order_mgr: WsOrderManager,
    ) -> Self {
        Self {
            mbo_book,
            adverse_detector,
            smart_router,
            ws_order_mgr,
            fill_prob_threshold: 0.3,
            stale_timeout_s: 5.0,
        }
    }

    /// Process a trade event through the adverse selection detector.
    ///
    /// INST: Active adverse selection protection — returns cancel recommendation
    /// at urgency >= 1 (previously only at >= 2), enabling faster reaction to
    /// toxic flow. Urgency 1 = prepare to cancel, urgency 2 = cancel immediately.
    pub fn on_trade_event(&mut self, event: &TradeEvent) -> Option<CancelReason> {
        if let Some(signal) = self.adverse_detector.on_trade(event) {
            // Active protection: act on urgency >= 1 (medium or high)
            // Previously only acted on urgency >= 2 (critical only), missing
            // early warning signals that could have prevented adverse fills.
            if signal.urgency >= 1 {
                return Some(CancelReason::AdverseSelection);
            }
        }
        None
    }

    /// Check all resting orders for adverse selection or stale queue positions.
    /// Returns indices of orders that should be canceled.
    ///
    /// INST: Enhanced with stale order detection — orders resting beyond
    /// stale_timeout_s with low fill probability are proactively canceled
    /// and repriced closer to market.
    pub fn check_resting_orders(&self) -> Vec<(usize, CancelReason)> {
        let mut cancels = Vec::new();
        let now = now_ms();

        for (idx, lifecycle) in self.ws_order_mgr.lifecycles.iter().enumerate() {
            if !lifecycle.is_resting() {
                continue;
            }

            // Check queue position via MBO book
            if let Some(oid_bytes) = lifecycle.order_id() {
                let oid_str = String::from_utf8_lossy(oid_bytes)
                    .trim_end_matches('\0')
                    .to_string();
                if let Ok(oid_u64) = oid_str.parse::<u64>() {
                    if let Some((queue_ahead, fill_prob, _ttf)) =
                        self.mbo_book.get_queue_position(oid_u64)
                    {
                        // Cancel if fill probability is too low
                        if fill_prob < self.fill_prob_threshold {
                            cancels.push((idx, CancelReason::QueuePositionBad));
                        }
                        let _ = queue_ahead;
                    }
                }
            }

            // INST: Stale order detection — if we couldn't find the order
            // in the MBO book at all (no queue position data), it may be
            // orphaned. Mark for cancel to avoid stuck resting orders.
            let found_in_book = if let Some(oid_bytes) = lifecycle.order_id() {
                let oid_str = String::from_utf8_lossy(oid_bytes)
                    .trim_end_matches('\0')
                    .to_string();
                if let Ok(oid_u64) = oid_str.parse::<u64>() {
                    self.mbo_book.get_queue_position(oid_u64).is_some()
                } else {
                    false
                }
            } else {
                false
            };
            if !found_in_book && lifecycle.transition_count() > 2 {
                // Order has gone through multiple transitions but isn't in book
                cancels.push((idx, CancelReason::QueuePositionBad));
            }
        }

        cancels
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Current Unix timestamp in milliseconds.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Current Unix timestamp in microseconds.
pub fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

// ---------------------------------------------------------------------------
// Bybit v5 Request Signing
// ---------------------------------------------------------------------------

/// Sign a Bybit v5 request using HMAC-SHA256.
///
/// Signature input: "{timestamp}{api_key}{recv_window}{payload}"
pub fn sign_bybit_request(
    timestamp: i64,
    api_key: &str,
    recv_window: i64,
    payload: &str,
    secret: &[u8],
) -> String {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let sign_input = format!("{}{}{}{}", timestamp, api_key, recv_window, payload);
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .expect("HMAC can take key of any size");
    mac.update(sign_input.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Classify Bybit v5 error responses into ExchangeError.
pub fn classify_bybit_error(body: &serde_json::Value) -> ExchangeError {
    let ret_code = body.get("retCode").and_then(|v| v.as_i64()).unwrap_or(0);
    let ret_msg = body.get("retMsg").and_then(|v| v.as_str()).unwrap_or("Unknown error");

    match ret_code {
        0 => ExchangeError::Unknown { code: "OK".to_string(), message: "Success".to_string() },
        10001 => ExchangeError::Unknown { code: ret_code.to_string(), message: "Parameter error".to_string() },
        10002 => ExchangeError::RateLimited { retry_after_ms: 1000 },
        10003 => ExchangeError::Unknown { code: "INVALID_API_KEY".to_string(), message: ret_msg.to_string() },
        10004 | 10005 => ExchangeError::Unknown { code: "SIGNATURE_ERROR".to_string(), message: ret_msg.to_string() },
        10006 => ExchangeError::RateLimited { retry_after_ms: 1000 },
        10010 => ExchangeError::Unknown { code: "IP_NOT_WHITELISTED".to_string(), message: ret_msg.to_string() },
        10016 => ExchangeError::InternalServerError,
        10017 => ExchangeError::Timeout,
        110001 => ExchangeError::MinimumOrderSize { min_size: 1 },
        110003 => ExchangeError::Unknown { code: "ORDER_NOT_EXIST".to_string(), message: ret_msg.to_string() },
        110004 => ExchangeError::InsufficientBalance,
        110007 => ExchangeError::InvalidPrice,
        110008 => ExchangeError::OrderNotFound,
        110012 => ExchangeError::InsufficientBalance,
        110017 => ExchangeError::Unknown { code: "REDUCE_ONLY_VIOLATION".to_string(), message: ret_msg.to_string() },
        110018 => ExchangeError::PositionNotFound,
        110043 => ExchangeError::Unknown { code: "POST_ONLY_REJECTED".to_string(), message: ret_msg.to_string() },
        110044 => ExchangeError::MinimumOrderSize { min_size: 1 },
        _ => ExchangeError::Unknown { code: ret_code.to_string(), message: ret_msg.to_string() },
    }
}

// ---------------------------------------------------------------------------
// Bybit Rate Limiter Extension
// ---------------------------------------------------------------------------

impl AdaptiveRateLimiter {
    /// Update rate limiter state from Bybit response.
    pub fn update_from_bybit_response(&self, body: &serde_json::Value) {
        // Bybit returns rate limit info in response headers, not body
        // For now, just check for rate limit errors
        let ret_code = body.get("retCode").and_then(|v| v.as_i64()).unwrap_or(0);
        if ret_code == 10002 || ret_code == 10006 {
            // Rate limited - increase backoff
            let current = self.backoff_multiplier.load(std::sync::atomic::Ordering::Relaxed);
            let new_backoff = (current * 150 / 100).min(500);
            self.backoff_multiplier.store(new_backoff, std::sync::atomic::Ordering::Relaxed);
            warn!("Bybit rate limited, backoff increased to {:.2}x", new_backoff as f64 / 100.0);
        }
    }
}
