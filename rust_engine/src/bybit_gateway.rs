//! Bybit v5 unified REST API gateway.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use hmac::{Hmac, Mac};
use reqwest::{
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
    Client,
};
use serde_json::{json, Value};
use sha2::Sha256;
use tracing::info;

use crate::execution_gateway::{
    classify_bybit_error, now_ms, now_us, sign_bybit_request, AdaptiveRateLimiter,
    ExchangeError, ExecutionGateway, OrderIntent, OrderResult, OrderSide, OrderType, Position,
    RustTicker,
};
use crate::instrument_manager::{InstrumentManager, Exchange, check_order_exists_bybit};

const BYBIT_BASE_URL: &str = "https://api.bybit.com";
const BYBIT_RECV_WINDOW: i64 = 5000;

pub struct BybitGateway {
    client: Client,
    api_key: String,
    api_secret: Vec<u8>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    testnet: bool,
    /// Monotonically increasing counter for generating unique orderLinkId values.
    /// Allows idempotency checks via check_order_exists_bybit when REST calls time out.
    next_client_id: AtomicU64,
    /// Dynamic instrument manager for real-time precision rules.
    /// When set, price and quantity are formatted using Bybit's
    /// tickSize and qtyStep instead of hardcoded "{:.8}" / "{:.3}".
    instrument_mgr: Option<Arc<InstrumentManager>>,
}

impl BybitGateway {
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            api_key,
            api_secret: api_secret.into_bytes(),
            rate_limiter: Arc::new(AdaptiveRateLimiter::new(10)),
            testnet,
            next_client_id: AtomicU64::new(1),
            instrument_mgr: None,
        }
    }

    /// Generate the next monotonic order link ID for idempotency tracking.
    ///
    /// The counter wraps to 0 after u64::MAX increments (effectively never in practice
    /// since a gateway instance is restarted long before 2^64 orders are submitted).
    fn next_order_link_id(&self) -> String {
        let id = self.next_client_id.fetch_add(1, Ordering::Relaxed);
        format!("rte{:016x}", id)
    }

    /// Create a new Bybit gateway with an InstrumentManager for real-time precision.
    pub fn new_with_instruments(
        api_key: String,
        api_secret: String,
        testnet: bool,
        instrument_mgr: Arc<InstrumentManager>,
    ) -> Self {
        let mut gw = Self::new(api_key, api_secret, testnet);
        gw.instrument_mgr = Some(instrument_mgr);
        gw
    }

    /// Set the instrument manager after construction.
    pub fn set_instrument_manager(&mut self, mgr: Arc<InstrumentManager>) {
        self.instrument_mgr = Some(mgr);
    }

    fn base_url(&self) -> &str {
        if self.testnet {
            // Demo trading uses mainnet infrastructure with simulated funds
            // API keys must be generated from the "Demo Trading" module inside mainnet Bybit account
            "https://api-demo.bybit.com"
        } else {
            BYBIT_BASE_URL
        }
    }

    /// Normalize symbol to Bybit format: no separators, uppercase (e.g. "BTCUSDT").
    fn normalize_symbol(symbol: &str) -> String {
        let normalized = symbol
            .replace('/', "")
            .replace('_', "")
            .replace(':', "")
            .to_uppercase();
        // Handle ccxt "BTC/USDT:USDT" → "BTCUSDTUSDT" → "BTCUSDT"
        if normalized.ends_with("USDTUSDT") {
            normalized[..normalized.len() - 4].to_string()
        } else {
            normalized
        }
    }

    fn build_auth_headers(&self, body_str: &str) -> HeaderMap {
        let timestamp = now_ms();
        let signature = sign_bybit_request(
            timestamp,
            &self.api_key,
            BYBIT_RECV_WINDOW,
            body_str,
            &self.api_secret,
        );

        let mut headers = HeaderMap::new();
        headers.insert(
            "X-BAPI-API-KEY",
            HeaderValue::from_str(&self.api_key)
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert(
            "X-BAPI-SIGN",
            HeaderValue::from_str(&signature)
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert(
            "X-BAPI-TIMESTAMP",
            HeaderValue::from_str(&timestamp.to_string())
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert(
            "X-BAPI-RECV-WINDOW",
            HeaderValue::from_str(&BYBIT_RECV_WINDOW.to_string())
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers
    }

    async fn post_signed(&self, path: &str, body: &Value) -> Result<Value, ExchangeError> {
        self.rate_limiter.acquire().await;

        let body_str = body.to_string();
        let auth_headers = self.build_auth_headers(&body_str);
        let url = format!("{}{}", self.base_url(), path);

        let response = self
            .client
            .post(&url)
            .headers(auth_headers)
            .body(body_str)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ExchangeError::Timeout
                } else {
                    ExchangeError::ConnectionReset
                }
            })?;

        let status = response.status().as_u16();
        let body: Value = response.json().await.map_err(|e| ExchangeError::Unknown {
            code: "JSON_PARSE".to_string(),
            message: e.to_string(),
        })?;

        self.rate_limiter.update_from_bybit_response(&body);

        if status >= 400 {
            return Err(classify_bybit_error(&body));
        }

        let ret_code = body.get("retCode").and_then(|v| v.as_i64()).unwrap_or(0);
        if ret_code != 0 {
            return Err(classify_bybit_error(&body));
        }

        Ok(body)
    }

    async fn get_signed(&self, path: &str, query: &str) -> Result<Value, ExchangeError> {
        self.rate_limiter.acquire().await;

        // For GET requests, Bybit signs the query string
        let timestamp = now_ms();
        let sign_input = format!(
            "{}{}{}{}",
            timestamp, &self.api_key, BYBIT_RECV_WINDOW, query
        );
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.api_secret)
            .expect("HMAC key error");
        mac.update(sign_input.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let url = if query.is_empty() {
            format!("{}{}", self.base_url(), path)
        } else {
            format!("{}{}?{}", self.base_url(), path, query)
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "X-BAPI-API-KEY",
            HeaderValue::from_str(&self.api_key)
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert(
            "X-BAPI-SIGN",
            HeaderValue::from_str(&signature)
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert(
            "X-BAPI-TIMESTAMP",
            HeaderValue::from_str(&timestamp.to_string())
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert(
            "X-BAPI-RECV-WINDOW",
            HeaderValue::from_str(&BYBIT_RECV_WINDOW.to_string())
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );

        let response = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let status = response.status().as_u16();
        let body: Value = response.json().await.map_err(|e| ExchangeError::Unknown {
            code: "JSON_PARSE".to_string(),
            message: e.to_string(),
        })?;

        if status >= 400 {
            return Err(classify_bybit_error(&body));
        }

        let ret_code = body.get("retCode").and_then(|v| v.as_i64()).unwrap_or(0);
        if ret_code != 0 {
            return Err(classify_bybit_error(&body));
        }

        Ok(body)
    }
}

#[async_trait]
impl ExecutionGateway for BybitGateway {
    async fn submit_order(&self, intent: OrderIntent) -> Result<OrderResult, ExchangeError> {
        let symbol = Self::normalize_symbol(&intent.symbol);
        let start_us = now_us();

        let side_str = match intent.side {
            OrderSide::Buy => "Buy",
            OrderSide::Sell => "Sell",
        };

        let order_type_str = match intent.order_type {
            OrderType::Market => "Market",
            OrderType::Limit | OrderType::PostOnly => "Limit",
        };

        let tif = match intent.order_type {
            OrderType::PostOnly => "PostOnly",
            OrderType::Market => "IOC",
            OrderType::Limit => "GTC",
        };

        // BUG FIX #2 & #3: Use InstrumentManager for proper precision formatting.
        // Previously used hardcoded "{:.3}" for qty and "{:.8}" for price, which
        // caused Bybit error 110007 (InvalidPrice) when the price didn't respect
        // the symbol's specific tickSize (e.g. BTCUSDT requires 0.5 increments).
        //
        // Now we fetch tickSize and qtyStep from Bybit's /v5/market/instruments-info
        // via the InstrumentManager and format accordingly.
        let spec: Option<crate::instrument_manager::ContractSpec> = self.instrument_mgr.as_ref()
            .and_then(|mgr| mgr.get(Exchange::Bybit, &symbol));

        let qty_f64 = intent.size as f64;
        let qty_str = if let Some(ref s) = spec {
            // Use real-time qtyStep precision from instruments-info lotSizeFilter
            let rounded = s.clamp_and_round_qty(qty_f64);
            if rounded < s.min_qty {
                tracing::warn!(
                    "Bybit qty {} below min {} for {} — clamping to min",
                    qty_f64, s.min_qty, symbol
                );
            }
            s.format_qty(rounded.max(s.min_qty))
        } else {
            // Fallback: conservative 8 decimal places
            format!("{:.8}", qty_f64.max(0.001))
        };

        // Generate a unique orderLinkId for idempotency.
        // Included in every order submission so that if this REST call times out
        // we can call check_order_exists_bybit() with this ID to determine whether
        // the order was accepted by the exchange before retrying.
        let link_id = self.next_order_link_id();

        let mut body = json!({
            "category": "linear",
            "symbol": symbol,
            "side": side_str,
            "orderType": order_type_str,
            "qty": qty_str,
            "timeInForce": tif,
            "reduceOnly": intent.reduce_only,
            "positionIdx": 0,  // one-way mode
            "orderLinkId": link_id,
        });

        // Only send price for non-MARKET orders
        if intent.order_type != OrderType::Market {
            if let Some(price) = intent.price {
                let price_str: String = if let Some(ref s) = spec {
                    // Use real-time tickSize precision from instruments-info priceFilter
                    s.format_price(price)
                } else {
                    format!("{:.8}", price)
                };
                body["price"] = json!(price_str);
            }
        }

        // Convert a generic Timeout into TimedOut so that retry_failed_leg can call
        // check_order_by_client_id() and avoid duplicate order submission.
        let response = self.post_signed("/v5/order/create", &body).await
            .map_err(|e| match e {
                ExchangeError::Timeout => ExchangeError::TimedOut { client_order_id: link_id.clone() },
                other => other,
            })?;
        let end_us = now_us();
        let latency_us = (end_us - start_us).max(0) as u64;

        let result_data = response.get("result").cloned().unwrap_or_default();

        let order_id = result_data
            .get("orderId")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let status = result_data
            .get("orderStatus")
            .and_then(|v| v.as_str())
            .unwrap_or("New")
            .to_string();

        info!(
            "Bybit order {} submitted (link_id={}): {} {} {} @ {:?} | {}µs",
            order_id, link_id, side_str, qty_str, symbol, intent.price, latency_us
        );

        Ok(OrderResult {
            order_id,
            status,
            filled_size: 0,
            avg_fill_price: 0.0,
            fee: 0.0,
            latency_us,
            exchange_timestamp: now_ms(),
            rejection_reason: None,
        })
    }

    async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<(), ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let body = json!({
            "category": "linear",
            "symbol": normalized,
            "orderId": order_id,
        });
        self.post_signed("/v5/order/cancel", &body).await?;
        info!("Bybit order {} cancelled ({})", order_id, normalized);
        Ok(())
    }

    async fn get_position(&self, symbol: &str) -> Result<Option<Position>, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let query = format!("category=linear&symbol={}", normalized);
        let response = self.get_signed("/v5/position/list", &query).await?;

        let positions = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .cloned()
            .unwrap_or_default();

        for pos in &positions {
            let size_str = pos.get("size").and_then(|v| v.as_str()).unwrap_or("0");
            let size: i64 = size_str.parse().unwrap_or(0);
            if size == 0 {
                continue;
            }

            let side = pos.get("side").and_then(|v| v.as_str()).unwrap_or("None");
            let signed_size = if side == "Sell" { -size } else { size };

            let entry_price = pos
                .get("avgPrice")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let unrealized_pnl = pos
                .get("unrealisedPnl")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let leverage = pos
                .get("leverage")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .map(|f| f as i32)
                .unwrap_or(1);

            return Ok(Some(Position {
                symbol: normalized,
                size: signed_size,
                entry_price,
                unrealized_pnl,
                leverage,
                side: side.to_lowercase(),
            }));
        }

        Ok(None)
    }

    async fn set_leverage(&self, symbol: &str, leverage: i32) -> Result<(), ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let body = json!({
            "category": "linear",
            "symbol": normalized,
            "buyLeverage": leverage.to_string(),
            "sellLeverage": leverage.to_string(),
        });
        match self.post_signed("/v5/position/set-leverage", &body).await {
            Ok(_) => {
                info!("Bybit leverage set to {}× for {}", leverage, normalized);
                Ok(())
            }
            Err(ref e) => {
                // Bybit returns retCode 110043 with "leverage not modified" when leverage
                // is already set to the requested value. This is NOT a real error — treat
                // it as a successful no-op so callers (stat-arb, funding-arb) don't abort.
                let err_str = format!("{}", e);
                if err_str.contains("leverage not modified")
                    || err_str.contains("Not modified")
                    || err_str.contains("110043")
                {
                    info!("Bybit leverage already at {}× for {} (no change needed)", leverage, normalized);
                    Ok(())
                } else {
                    Err(e.clone())
                }
            }
        }
    }

    async fn get_ticker(&self, symbol: &str) -> Result<RustTicker, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let query = format!("category=linear&symbol={}", normalized);
        let response = self.get_signed("/v5/market/tickers", &query).await?;

        let ticker = response
            .pointer("/result/list/0")
            .ok_or_else(|| ExchangeError::Unknown {
                code: "NO_TICKER".into(),
                message: format!("No ticker data for {}", normalized),
            })?;

        let last = ticker.get("lastPrice").and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
        let bid = ticker.get("bid1Price").and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok()).unwrap_or(last);
        let ask = ticker.get("ask1Price").and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok()).unwrap_or(last);
        let volume = ticker.get("volume24h").and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);

        Ok(RustTicker { last, bid, ask, volume_24h: volume })
    }

    async fn get_balance(&self) -> Result<f64, ExchangeError> {
        let query = "accountType=UNIFIED&coin=USDT";
        let response = self.get_signed("/v5/account/wallet-balance", query).await?;

        // Try multiple fields in priority order:
        // 1. totalAvailableBalance (account-level USD available, works for cross/portfolio margin)
        // 2. coin[].walletBalance (per-coin wallet balance)
        // 3. coin[].availableToWithdraw (DEPRECATED since Jan 2025, may return empty)
        // 4. totalEquity (account-level total equity in USD)

        // First try account-level totalAvailableBalance
        if let Some(total_avail) = response
            .pointer("/result/list/0/totalAvailableBalance")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
        {
            if total_avail > 0.0 {
                info!("Bybit balance (totalAvailableBalance): ${:.2}", total_avail);
                return Ok(total_avail);
            }
        }

        // Then try per-coin walletBalance
        if let Some(coins) = response.pointer("/result/list/0/coin").and_then(|v| v.as_array()) {
            for coin in coins {
                let coin_name = coin.get("coin").and_then(|v| v.as_str()).unwrap_or("");
                if coin_name == "USDT" {
                    // Try walletBalance first (most reliable)
                    if let Some(wb) = coin.get("walletBalance")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        if wb > 0.0 {
                            info!("Bybit balance (walletBalance): ${:.2}", wb);
                            return Ok(wb);
                        }
                    }
                    // Fallback to equity
                    if let Some(eq) = coin.get("equity")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        if eq > 0.0 {
                            info!("Bybit balance (equity): ${:.2}", eq);
                            return Ok(eq);
                        }
                    }
                    // Last resort: deprecated availableToWithdraw
                    if let Some(atw) = coin.get("availableToWithdraw")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        info!("Bybit balance (availableToWithdraw): ${:.2}", atw);
                        return Ok(atw);
                    }
                }
            }
        }

        // Final fallback: account-level totalEquity
        if let Some(equity) = response
            .pointer("/result/list/0/totalEquity")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
        {
            info!("Bybit balance (totalEquity): ${:.2}", equity);
            return Ok(equity);
        }

        info!("Bybit balance: could not extract balance from response: {}", 
            serde_json::to_string(&response).unwrap_or_default());
        Ok(0.0)
    }

    /// BUG 4 FIX: Implement get_order_status for Bybit to enable fill confirmation polling.
    /// Bybit's submit_order returns filled_size=0 immediately; actual fill data comes from
    /// querying /v5/order/realtime with the orderId.
    async fn get_order_status(&self, order_id: &str, symbol: &str)
        -> Result<Option<OrderResult>, ExchangeError>
    {
        let normalized = Self::normalize_symbol(symbol);
        let query = format!("category=linear&symbol={}&orderId={}", normalized, order_id);
        match self.get_signed("/v5/order/realtime", &query).await {
            Ok(response) => {
                let order = response
                    .pointer("/result/list/0")
                    .ok_or_else(|| ExchangeError::OrderNotFound)?;

                let status = order.get("orderStatus")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();

                let cum_exec_qty = order.get("cumExecQty")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let avg_price = order.get("avgPrice")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let cum_exec_fee = order.get("cumExecFee")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                // Convert cumExecQty (float string like "0.001") to milli-units i64
                // to match our position_size encoding from Bug 3 fix
                let filled_size = (cum_exec_qty * 1000.0).round() as i64;

                Ok(Some(OrderResult {
                    order_id: order_id.to_string(),
                    status,
                    filled_size,
                    avg_fill_price: avg_price,
                    fee: cum_exec_fee,
                    latency_us: 0,
                    exchange_timestamp: now_ms(),
                    rejection_reason: None,
                }))
            }
            Err(ExchangeError::OrderNotFound) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Idempotency check: look up an order by the `orderLinkId` that was sent with the
    /// original submission.  Called when `submit_order` returns
    /// `ExchangeError::TimedOut` — i.e. the REST call timed out and we don't know whether
    /// Bybit accepted the order.  Returns the exchange-assigned orderId if found and
    /// active, or `Ok(None)` if no matching order exists (safe to retry).
    async fn check_order_by_client_id(
        &self,
        client_order_id: &str,
        symbol: &str,
    ) -> Result<Option<String>, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let result = check_order_exists_bybit(
            &self.client,
            self.base_url(),
            &self.api_key,
            &self.api_secret,
            &normalized,
            client_order_id,
        ).await;
        Ok(result)
    }

    async fn get_positions(&self) -> Result<Vec<Position>, ExchangeError> {
        let query = "category=linear&settleCoin=USDT";
        let response = self.get_signed("/v5/position/list", query).await?;

        let raw_positions = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .cloned()
            .unwrap_or_default();

        let mut positions = Vec::new();

        for pos in &raw_positions {
            let size_str = pos.get("size").and_then(|v| v.as_str()).unwrap_or("0");
            let size: i64 = size_str.parse().unwrap_or(0);
            if size == 0 {
                continue;
            }

            let side = pos.get("side").and_then(|v| v.as_str()).unwrap_or("None");
            let signed_size = if side == "Sell" { -size } else { size };

            let symbol = pos.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string();

            let entry_price = pos
                .get("avgPrice")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let unrealized_pnl = pos
                .get("unrealisedPnl")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let leverage = pos
                .get("leverage")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .map(|f| f as i32)
                .unwrap_or(1);

            positions.push(Position {
                symbol,
                size: signed_size,
                entry_price,
                unrealized_pnl,
                leverage,
                side: side.to_lowercase(),
            });
        }

        Ok(positions)
    }
}
