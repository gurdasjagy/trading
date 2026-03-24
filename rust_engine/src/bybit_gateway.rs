//! Bybit v5 unified REST API gateway.

use std::sync::Arc;

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
};

const BYBIT_BASE_URL: &str = "https://api.bybit.com";
const BYBIT_RECV_WINDOW: i64 = 5000;

pub struct BybitGateway {
    client: Client,
    api_key: String,
    api_secret: Vec<u8>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    testnet: bool,
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
        }
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

        // Ensure quantity is at least 1 and properly formatted as a decimal string.
        // Bybit v5 linear contracts require qty as a string in base currency units
        // (e.g. "0.01" for ETH). Sending "0" causes error 10001 (Parameter error).
        let qty = intent.size.max(1);
        let qty_str = format!("{:.3}", qty as f64);
        
        let mut body = json!({
            "category": "linear",
            "symbol": symbol,
            "side": side_str,
            "orderType": order_type_str,
            "qty": qty_str,
            "timeInForce": tif,
            "reduceOnly": intent.reduce_only,
            "positionIdx": 0,  // one-way mode
        });

        // Only send price for non-MARKET orders
        if intent.order_type != OrderType::Market {
            if let Some(price) = intent.price {
                body["price"] = json!(format!("{:.8}", price));
            }
        }

        let response = self.post_signed("/v5/order/create", &body).await?;
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
            "Bybit order {} submitted: {} {} {} @ {:?} | {}µs",
            order_id, side_str, qty_str, symbol, intent.price, latency_us
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
        self.post_signed("/v5/position/set-leverage", &body).await?;
        info!("Bybit leverage set to {}× for {}", leverage, normalized);
        Ok(())
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
