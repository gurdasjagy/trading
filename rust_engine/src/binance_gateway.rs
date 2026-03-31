//! Binance Futures v1 REST API gateway.
//!
//! Provides execution gateway for Binance Futures (USDT-M perpetuals).
//! Used for multi-exchange arbitrage in Feature 5.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde_json::Value;
use sha2::Sha256;
use tracing::{info, warn};

use crate::execution_gateway::{
    AdaptiveRateLimiter, ExchangeError, ExecutionGateway, OrderIntent, OrderResult, OrderSide,
    OrderType, Position, RustTicker,
};
use crate::instrument_manager::{check_order_exists_binance, Exchange, InstrumentManager};

const BINANCE_FUTURES_BASE_URL: &str = "https://fapi.binance.com";
const BINANCE_FUTURES_TESTNET_URL: &str = "https://testnet.binancefuture.com";
const RECV_WINDOW: i64 = 5000;

/// Get current timestamp in milliseconds.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Get current timestamp in microseconds (for latency tracking).
fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

/// Sign a request using Binance's HMAC-SHA256 signature method.
fn sign_binance_request(query_string: &str, secret: &[u8]) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("HMAC can take key of any size");
    mac.update(query_string.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Classify Binance error responses into our error types.
fn classify_binance_error(body: &Value) -> ExchangeError {
    let code = body.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
    let msg = body
        .get("msg")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown error");

    match code {
        -1000 => ExchangeError::Unknown {
            code: code.to_string(),
            message: msg.to_string(),
        },
        -1001 | -1003 => ExchangeError::RateLimited {
            retry_after_ms: 1000,
        },
        -1002 | -2015 => ExchangeError::Unknown {
            code: "INVALID_API_KEY".to_string(),
            message: msg.to_string(),
        },
        -1013 | -4003 | -4014 | -4015 => ExchangeError::MinimumOrderSize { min_size: 1 },
        -1021 => ExchangeError::Timeout, // Timestamp outside recvWindow
        -2010 | -2011 | -2019 => ExchangeError::InsufficientBalance,
        -2021 | -2022 => ExchangeError::Unknown {
            code: "ORDER_REJECTED".to_string(),
            message: msg.to_string(),
        },
        -4028 | -4030 => ExchangeError::MinimumOrderSize { min_size: 1 },
        -4131 => ExchangeError::PositionNotFound,
        _ => ExchangeError::Unknown {
            code: code.to_string(),
            message: msg.to_string(),
        },
    }
}

/// Binance Futures gateway implementation.
pub struct BinanceGateway {
    client: Client,
    api_key: String,
    api_secret: Vec<u8>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    testnet: bool,
    /// Monotonically increasing counter for generating unique newClientOrderId values.
    /// Allows idempotency checks via check_order_exists_binance when REST calls time out.
    next_client_id: AtomicU64,
    /// Dynamic instrument manager for real-time precision rules.
    /// When set, price and quantity are formatted using exchange-specific
    /// tickSize and stepSize instead of hardcoded "{:.8}" / "{:.3}".
    instrument_mgr: Option<Arc<InstrumentManager>>,
}

impl BinanceGateway {
    /// Create a new Binance gateway instance.
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let client = Client::builder()
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
            next_client_id: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            ),
            instrument_mgr: None,
        }
    }

    /// Generate the next monotonic client order ID for idempotency tracking.
    ///
    /// The counter wraps to 0 after u64::MAX increments (effectively never in practice
    /// since a gateway instance is restarted long before 2^64 orders are submitted).
    fn next_client_id(&self) -> String {
        let id = self.next_client_id.fetch_add(1, Ordering::Relaxed);
        format!("rte{:016x}", id)
    }

    /// Create a new Binance gateway with an InstrumentManager for real-time precision.
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
            BINANCE_FUTURES_TESTNET_URL
        } else {
            BINANCE_FUTURES_BASE_URL
        }
    }

    /// Normalize symbol to Binance format: no separators, uppercase (e.g. "BTCUSDT").
    fn normalize_symbol(symbol: &str) -> String {
        symbol
            .replace('/', "")
            .replace('_', "")
            .replace(':', "")
            .to_uppercase()
    }

    /// Build signed query string with timestamp and signature.
    fn sign_query(&self, params: &str) -> String {
        let timestamp = now_ms();
        let query = if params.is_empty() {
            format!("timestamp={}&recvWindow={}", timestamp, RECV_WINDOW)
        } else {
            format!(
                "{}&timestamp={}&recvWindow={}",
                params, timestamp, RECV_WINDOW
            )
        };
        let signature = sign_binance_request(&query, &self.api_secret);
        format!("{}&signature={}", query, signature)
    }

    async fn post_signed(&self, path: &str, params: &str) -> Result<Value, ExchangeError> {
        self.rate_limiter.acquire().await;

        let signed_params = self.sign_query(params);
        let url = format!("{}{}?{}", self.base_url(), path, signed_params);

        let response = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
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

        if status >= 400 || body.get("code").is_some() {
            return Err(classify_binance_error(&body));
        }

        Ok(body)
    }

    async fn get_signed(&self, path: &str, params: &str) -> Result<Value, ExchangeError> {
        self.rate_limiter.acquire().await;

        let signed_params = self.sign_query(params);
        let url = format!("{}{}?{}", self.base_url(), path, signed_params);

        let response = self
            .client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let status = response.status().as_u16();
        let body: Value = response.json().await.map_err(|e| ExchangeError::Unknown {
            code: "JSON_PARSE".to_string(),
            message: e.to_string(),
        })?;

        if status >= 400 || body.get("code").is_some() {
            return Err(classify_binance_error(&body));
        }

        Ok(body)
    }

    async fn delete_signed(&self, path: &str, params: &str) -> Result<Value, ExchangeError> {
        self.rate_limiter.acquire().await;

        let signed_params = self.sign_query(params);
        let url = format!("{}{}?{}", self.base_url(), path, signed_params);

        let response = self
            .client
            .delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let status = response.status().as_u16();
        let body: Value = response.json().await.map_err(|e| ExchangeError::Unknown {
            code: "JSON_PARSE".to_string(),
            message: e.to_string(),
        })?;

        if status >= 400 || body.get("code").is_some() {
            return Err(classify_binance_error(&body));
        }

        Ok(body)
    }
}

#[async_trait]
impl ExecutionGateway for BinanceGateway {
    async fn submit_order(&self, intent: OrderIntent) -> Result<OrderResult, ExchangeError> {
        let symbol = Self::normalize_symbol(&intent.symbol);
        let start_us = now_us();

        let side = match intent.side {
            OrderSide::Buy => "BUY",
            OrderSide::Sell => "SELL",
        };

        let (order_type, time_in_force) = match intent.order_type {
            OrderType::Market => ("MARKET", ""),
            OrderType::Limit => ("LIMIT", "GTC"),
            OrderType::PostOnly => ("LIMIT", "GTX"), // GTX = Post-only on Binance
        };

        // BUG FIX #2 & #3: Use InstrumentManager for proper precision formatting.
        // Previously used hardcoded "{:.3}" for qty and "{:.8}" for price, which
        // caused InvalidPrice errors on Bybit and InsufficientBalance on Binance
        // (because size=1 meant 1 whole BTC instead of a fractional amount).
        //
        // Now we fetch tickSize and stepSize from Binance's /fapi/v1/exchangeInfo
        // via the InstrumentManager and format accordingly.
        let spec: Option<crate::instrument_manager::ContractSpec> = self
            .instrument_mgr
            .as_ref()
            .and_then(|mgr| mgr.get(Exchange::Binance, &symbol));

        let qty_f64 = intent.size as f64;
        let qty_str = if let Some(ref s) = spec {
            // Use real-time stepSize precision from exchangeInfo
            let rounded = s.clamp_and_round_qty(qty_f64);
            if rounded < s.min_qty {
                warn!(
                    "Binance qty {} below min {} for {} — clamping to min",
                    qty_f64, s.min_qty, symbol
                );
            }
            s.format_qty(rounded.max(s.min_qty))
        } else {
            // Fallback: conservative 8 decimal places
            format!("{:.8}", qty_f64.max(0.001))
        };

        // Generate a unique client order ID for idempotency.
        // Included in every order submission as `newClientOrderId` so that
        // if this REST call times out we can later call check_order_exists_binance()
        // with this ID to determine whether the order was accepted.
        let client_oid = self.next_client_id();

        let mut params = format!(
            "symbol={}&side={}&type={}&quantity={}&newClientOrderId={}",
            symbol, side, order_type, qty_str, client_oid
        );

        if !time_in_force.is_empty() {
            params.push_str(&format!("&timeInForce={}", time_in_force));
        }

        // Only send price for non-MARKET orders — Binance rejects MARKET orders
        // that include a price parameter with error -1102.
        if intent.order_type != OrderType::Market {
            if let Some(price) = intent.price {
                let price_str: String = if let Some(ref s) = spec {
                    // Use real-time tickSize precision from exchangeInfo PRICE_FILTER
                    s.format_price(price)
                } else {
                    format!("{:.8}", price)
                };
                params.push_str(&format!("&price={}", price_str));
            }
        }

        if intent.reduce_only {
            params.push_str("&reduceOnly=true");
        }

        // Convert a generic Timeout into TimedOut so that retry_failed_leg can call
        // check_order_by_client_id() and avoid duplicate order submission.
        let response = self
            .post_signed("/fapi/v1/order", &params)
            .await
            .map_err(|e| match e {
                ExchangeError::Timeout => ExchangeError::TimedOut {
                    client_order_id: client_oid.clone(),
                },
                other => other,
            })?;
        let end_us = now_us();
        let latency_us = (end_us - start_us).max(0) as u64;

        let order_id = response
            .get("orderId")
            .and_then(|v| v.as_u64())
            .map(|id| id.to_string())
            .unwrap_or_default();

        let status = response
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("NEW")
            .to_string();

        let filled_qty = response
            .get("executedQty")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        let avg_price = response
            .get("avgPrice")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        info!(
            "Binance order {} submitted (client_oid={}): {} {} {} @ {:?} | {}us",
            order_id, client_oid, side, qty_str, symbol, intent.price, latency_us
        );

        Ok(OrderResult {
            order_id,
            status,
            filled_size: filled_qty,
            avg_fill_price: avg_price,
            fee: 0.0,
            latency_us,
            exchange_timestamp: now_ms(),
            rejection_reason: None,
        })
    }

    async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<(), ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let params = format!("symbol={}&orderId={}", normalized, order_id);
        self.delete_signed("/fapi/v1/order", &params).await?;
        info!("Binance order {} cancelled ({})", order_id, normalized);
        Ok(())
    }

    async fn get_position(&self, symbol: &str) -> Result<Option<Position>, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let params = format!("symbol={}", normalized);
        let response = self.get_signed("/fapi/v2/positionRisk", &params).await?;

        let positions = response.as_array().cloned().unwrap_or_default();

        for pos in &positions {
            let sym = pos.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
            if sym != normalized {
                continue;
            }

            let size_str = pos
                .get("positionAmt")
                .and_then(|v| v.as_str())
                .unwrap_or("0");
            let size: f64 = size_str.parse().unwrap_or(0.0);

            if size.abs() < 1e-8 {
                continue;
            }

            let entry_price = pos
                .get("entryPrice")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let unrealized_pnl = pos
                .get("unRealizedProfit")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let leverage = pos
                .get("leverage")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(1);

            let side = if size > 0.0 { "long" } else { "short" };

            return Ok(Some(Position {
                symbol: normalized,
                size: (size.abs() * 1e8) as i64 * if size > 0.0 { 1 } else { -1 },
                entry_price,
                unrealized_pnl,
                leverage,
                side: side.to_string(),
            }));
        }

        Ok(None)
    }

    async fn get_ticker(&self, symbol: &str) -> Result<RustTicker, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        // Binance public endpoint — no signature needed, use raw GET
        let url = format!(
            "{}/fapi/v1/ticker/bookTicker?symbol={}",
            self.base_url(),
            normalized
        );
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;
        let body: Value = response.json().await.map_err(|e| ExchangeError::Unknown {
            code: "JSON_PARSE".into(),
            message: e.to_string(),
        })?;

        let bid = body
            .get("bidPrice")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let ask = body
            .get("askPrice")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let last = if bid > 0.0 && ask > 0.0 {
            (bid + ask) / 2.0
        } else {
            bid.max(ask)
        };

        Ok(RustTicker {
            last,
            bid,
            ask,
            volume_24h: 0.0,
        })
    }

    async fn set_leverage(&self, symbol: &str, leverage: i32) -> Result<(), ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let params = format!("symbol={}&leverage={}", normalized, leverage);
        self.post_signed("/fapi/v1/leverage", &params).await?;
        info!("Binance leverage set to {}x for {}", leverage, normalized);
        Ok(())
    }

    async fn get_balance(&self) -> Result<f64, ExchangeError> {
        let response = self.get_signed("/fapi/v2/balance", "").await?;

        let balances = response.as_array().cloned().unwrap_or_default();

        for balance in &balances {
            let asset = balance.get("asset").and_then(|v| v.as_str()).unwrap_or("");
            if asset == "USDT" {
                let available = balance
                    .get("availableBalance")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Ok(available);
            }
        }

        Ok(0.0)
    }

    async fn get_positions(&self) -> Result<Vec<Position>, ExchangeError> {
        let response = self.get_signed("/fapi/v2/positionRisk", "").await?;

        let raw_positions = response.as_array().cloned().unwrap_or_default();
        let mut positions = Vec::new();

        for pos in &raw_positions {
            let size_str = pos
                .get("positionAmt")
                .and_then(|v| v.as_str())
                .unwrap_or("0");
            let size: f64 = size_str.parse().unwrap_or(0.0);

            if size.abs() < 1e-8 {
                continue;
            }

            let symbol = pos
                .get("symbol")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let entry_price = pos
                .get("entryPrice")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let unrealized_pnl = pos
                .get("unRealizedProfit")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let leverage = pos
                .get("leverage")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(1);

            let side = if size > 0.0 { "long" } else { "short" };

            positions.push(Position {
                symbol,
                size: (size.abs() * 1e8) as i64 * if size > 0.0 { 1 } else { -1 },
                entry_price,
                unrealized_pnl,
                leverage,
                side: side.to_string(),
            });
        }

        Ok(positions)
    }

    /// Idempotency check: look up an order by the `newClientOrderId` that was sent with
    /// the original submission.  Called when `submit_order` returns
    /// `ExchangeError::TimedOut` — i.e. the REST call timed out and we don't know whether
    /// Binance accepted the order.  Returns the exchange-assigned orderId if found and
    /// active, or `Ok(None)` if no matching order exists (safe to retry).
    async fn check_order_by_client_id(
        &self,
        client_order_id: &str,
        symbol: &str,
    ) -> Result<Option<String>, ExchangeError> {
        let normalized = Self::normalize_symbol(symbol);
        let result = check_order_exists_binance(
            &self.client,
            self.base_url(),
            &self.api_key,
            &self.api_secret,
            &normalized,
            client_order_id,
        )
        .await;
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Spot-Futures Arbitrage: Spot wallet & order methods
    // -----------------------------------------------------------------------

    /// Get available balance of an asset in the Binance Spot wallet.
    /// Endpoint: GET /api/v3/account (signed)
    /// Mainnet: https://api.binance.com, Testnet: https://testnet.binance.vision
    async fn get_spot_asset_balance(&self, asset: &str) -> Result<f64, ExchangeError> {
        let spot_base = if self.testnet {
            "https://testnet.binance.vision"
        } else {
            "https://api.binance.com"
        };

        let ts = now_ms();
        let query = format!("timestamp={}&recvWindow={}", ts, RECV_WINDOW);
        let signature = sign_binance_request(&query, &self.api_secret);
        let url = format!(
            "{}/api/v3/account?{}&signature={}",
            spot_base, query, signature
        );

        self.rate_limiter.acquire().await;

        let resp = self
            .client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let body: Value = resp.json().await.map_err(|_| ExchangeError::Timeout)?;

        if body.get("code").is_some() {
            return Err(classify_binance_error(&body));
        }

        if let Some(balances) = body.get("balances").and_then(|v| v.as_array()) {
            for bal in balances {
                let a = bal.get("asset").and_then(|v| v.as_str()).unwrap_or("");
                if a.eq_ignore_ascii_case(asset) {
                    let free = bal
                        .get("free")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    return Ok(free);
                }
            }
        }

        Ok(0.0) // Asset not found = zero balance
    }

    /// Transfer between Binance Spot and Futures wallets.
    /// Endpoint: POST /sapi/v1/asset/transfer
    /// NOT available on testnet (/sapi/ endpoints don't exist on testnet).
    async fn transfer_between_wallets(
        &self,
        from: &str,
        to: &str,
        amount: f64,
        asset: &str,
    ) -> Result<(), ExchangeError> {
        // Check trading mode -- only live allows transfers
        let mode = std::env::var("TRADING_MODE").unwrap_or_else(|_| "paper".to_string());
        if mode != "live" {
            info!("Transfer skipped: not in live mode (TRADING_MODE={})", mode);
            return Ok(());
        }

        if self.testnet {
            info!("Transfer skipped: /sapi/ endpoints not available on Binance testnet");
            return Ok(());
        }

        let transfer_type = match (from, to) {
            ("spot", "futures") => "MAIN_UMFUTURE",
            ("futures", "spot") => "UMFUTURE_MAIN",
            _ => {
                return Err(ExchangeError::Unknown {
                    code: "INVALID_TRANSFER".into(),
                    message: format!("Invalid transfer direction: {} -> {}", from, to),
                });
            }
        };

        let ts = now_ms();
        let query = format!(
            "type={}&asset={}&amount={}&timestamp={}&recvWindow={}",
            transfer_type, asset, amount, ts, RECV_WINDOW
        );
        let signature = sign_binance_request(&query, &self.api_secret);
        let url = format!(
            "https://api.binance.com/sapi/v1/asset/transfer?{}&signature={}",
            query, signature
        );

        self.rate_limiter.acquire().await;

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let body: Value = resp.json().await.map_err(|_| ExchangeError::Timeout)?;

        if let Some(tran_id) = body.get("tranId") {
            info!(
                "[binance-spot] Transfer {} {} from {} to {}: tranId={}",
                amount, asset, from, to, tran_id
            );
            Ok(())
        } else {
            Err(classify_binance_error(&body))
        }
    }

    /// Submit a Spot order on Binance.
    /// Endpoint: POST /api/v3/order (signed)
    /// Mainnet: https://api.binance.com, Testnet: https://testnet.binance.vision
    async fn submit_spot_order(
        &self,
        intent: crate::execution_gateway::SpotOrderIntent,
    ) -> Result<crate::execution_gateway::SpotOrderResult, ExchangeError> {
        let spot_base = if self.testnet {
            "https://testnet.binance.vision"
        } else {
            "https://api.binance.com"
        };

        let start_us = now_us();
        let normalized = Self::normalize_symbol(&intent.symbol);

        let side_str = match intent.side {
            OrderSide::Buy => "BUY",
            OrderSide::Sell => "SELL",
        };

        let type_str = match intent.order_type {
            OrderType::Market => "MARKET",
            OrderType::Limit => "LIMIT",
            OrderType::PostOnly => "LIMIT", // Binance PostOnly = LIMIT + timeInForce=GTX
        };

        let tif = if matches!(intent.order_type, OrderType::PostOnly) {
            "GTX" // Binance Post-Only
        } else {
            &intent.time_in_force
        };

        let ts = now_ms();
        let client_oid = self.next_client_id();

        // Build query parameters
        let mut params = format!(
            "symbol={}&side={}&type={}&newClientOrderId={}&timestamp={}&recvWindow={}",
            normalized, side_str, type_str, client_oid, ts, RECV_WINDOW
        );

        // For market buys, prefer quoteOrderQty (spend X USDT)
        if matches!(intent.order_type, OrderType::Market) && intent.side == OrderSide::Buy {
            if let Some(quote_qty) = intent.quote_order_qty {
                params.push_str(&format!("&quoteOrderQty={:.8}", quote_qty));
            } else if intent.qty > 0.0 {
                params.push_str(&format!("&quantity={:.8}", intent.qty));
            }
        } else {
            if intent.qty > 0.0 {
                params.push_str(&format!("&quantity={:.8}", intent.qty));
            }
        }

        // Price for limit orders
        if let Some(price) = intent.price {
            params.push_str(&format!("&price={:.8}", price));
            params.push_str(&format!("&timeInForce={}", tif));
        }

        let signature = sign_binance_request(&params, &self.api_secret);
        let url = format!(
            "{}/api/v3/order?{}&signature={}",
            spot_base, params, signature
        );

        self.rate_limiter.acquire().await;

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .map_err(|_| ExchangeError::Timeout)?;

        let end_us = now_us();
        let body: Value = resp.json().await.map_err(|_| ExchangeError::Timeout)?;

        if body.get("code").is_some() && body.get("orderId").is_none() {
            return Err(classify_binance_error(&body));
        }

        let order_id = body
            .get("orderId")
            .and_then(|v| v.as_i64())
            .map(|id| id.to_string())
            .unwrap_or_default();
        let status = body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();

        // Parse fills
        let mut filled_qty = 0.0f64;
        let mut total_cost = 0.0f64;
        let mut total_fee = 0.0f64;

        if let Some(fills) = body.get("fills").and_then(|v| v.as_array()) {
            for fill in fills {
                let qty: f64 = fill
                    .get("qty")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let price: f64 = fill
                    .get("price")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let fee: f64 = fill
                    .get("commission")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                filled_qty += qty;
                total_cost += qty * price;
                total_fee += fee;
            }
        } else {
            // Fallback: use executedQty and cummulativeQuoteQty
            filled_qty = body
                .get("executedQty")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            total_cost = body
                .get("cummulativeQuoteQty")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
        }

        let avg_price = if filled_qty > 0.0 {
            total_cost / filled_qty
        } else {
            0.0
        };

        info!(
            "[binance-spot] Order {}: status={}, filled={}, avg_price={}, fee={}, latency={}us",
            order_id,
            status,
            filled_qty,
            avg_price,
            total_fee,
            (end_us - start_us)
        );

        Ok(crate::execution_gateway::SpotOrderResult {
            order_id,
            status: status.to_lowercase(),
            filled_qty,
            avg_fill_price: avg_price,
            fee: total_fee,
            latency_us: (end_us - start_us) as u64,
            rejection_reason: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_symbol_normalization() {
        assert_eq!(BinanceGateway::normalize_symbol("BTC/USDT"), "BTCUSDT");
        assert_eq!(BinanceGateway::normalize_symbol("BTC_USDT"), "BTCUSDT");
        assert_eq!(BinanceGateway::normalize_symbol("btcusdt"), "BTCUSDT");
        assert_eq!(
            BinanceGateway::normalize_symbol("BTC/USDT:USDT"),
            "BTCUSDTUSDT"
        );
    }

    #[test]
    fn test_error_classification() {
        let rate_limit = json!({"code": -1003, "msg": "Too many requests"});
        assert!(matches!(
            classify_binance_error(&rate_limit),
            ExchangeError::RateLimited { .. }
        ));

        let invalid_key = json!({"code": -2015, "msg": "Invalid API key"});
        assert!(matches!(
            classify_binance_error(&invalid_key),
            ExchangeError::Unknown { .. }
        ));
    }
}
