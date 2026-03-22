//! Binance Futures v1 REST API gateway.
//!
//! Provides execution gateway for Binance Futures (USDT-M perpetuals).
//! Used for multi-exchange arbitrage in Feature 5.

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
    AdaptiveRateLimiter, ExchangeError, ExecutionGateway, OrderIntent, OrderResult, 
    OrderSide, OrderType, Position,
};

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
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .expect("HMAC can take key of any size");
    mac.update(query_string.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Classify Binance error responses into our error types.
fn classify_binance_error(body: &Value) -> ExchangeError {
    let code = body.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
    let msg = body.get("msg").and_then(|v| v.as_str()).unwrap_or("Unknown error");
    
    match code {
        -1000 => ExchangeError::Unknown { code: code.to_string(), message: msg.to_string() },
        -1001 => ExchangeError::RateLimited,
        -1002 | -2015 => ExchangeError::InvalidApiKey,
        -1003 => ExchangeError::RateLimited,
        -1013 | -4003 | -4014 | -4015 => ExchangeError::MinSizeViolation,
        -1021 => ExchangeError::Timeout, // Timestamp outside recvWindow
        -2010 | -2011 => ExchangeError::InsufficientBalance,
        -2019 => ExchangeError::InsufficientMargin,
        -2021 => ExchangeError::PostOnlyWouldTake,
        -2022 => ExchangeError::ReduceOnlyViolation,
        -4028 | -4030 => ExchangeError::NotionalTooSmall,
        -4131 => ExchangeError::PositionNotFound,
        _ => ExchangeError::Unknown { code: code.to_string(), message: msg.to_string() },
    }
}

/// Binance Futures gateway implementation.
pub struct BinanceGateway {
    client: Client,
    api_key: String,
    api_secret: Vec<u8>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    testnet: bool,
}

impl BinanceGateway {
    /// Create a new Binance gateway instance.
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
            format!("{}&timestamp={}&recvWindow={}", params, timestamp, RECV_WINDOW)
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
        
        let mut params = format!(
            "symbol={}&side={}&type={}&quantity={}",
            symbol, side, order_type, intent.size
        );
        
        if !time_in_force.is_empty() {
            params.push_str(&format!("&timeInForce={}", time_in_force));
        }
        
        if let Some(price) = intent.price {
            params.push_str(&format!("&price={:.8}", price));
        }
        
        if intent.reduce_only {
            params.push_str("&reduceOnly=true");
        }
        
        let response = self.post_signed("/fapi/v1/order", &params).await?;
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
            "Binance order {} submitted: {} {} {} @ {:?} | {}us",
            order_id, side, intent.size, symbol, intent.price, latency_us
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
            
            let size_str = pos.get("positionAmt").and_then(|v| v.as_str()).unwrap_or("0");
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbol_normalization() {
        assert_eq!(BinanceGateway::normalize_symbol("BTC/USDT"), "BTCUSDT");
        assert_eq!(BinanceGateway::normalize_symbol("BTC_USDT"), "BTCUSDT");
        assert_eq!(BinanceGateway::normalize_symbol("btcusdt"), "BTCUSDT");
        assert_eq!(BinanceGateway::normalize_symbol("BTC/USDT:USDT"), "BTCUSDTUSDT");
    }
    
    #[test]
    fn test_error_classification() {
        let rate_limit = json!({"code": -1003, "msg": "Too many requests"});
        assert!(matches!(classify_binance_error(&rate_limit), ExchangeError::RateLimited));
        
        let invalid_key = json!({"code": -2015, "msg": "Invalid API key"});
        assert!(matches!(classify_binance_error(&invalid_key), ExchangeError::InvalidApiKey));
    }
}
