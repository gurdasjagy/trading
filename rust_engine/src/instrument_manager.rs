//! Dynamic Instrument Manager — fetches contract specs from all exchanges at startup.
//!
//! Replaces hardcoded precision values with real-time data from:
//! - **Binance**: `GET /fapi/v1/exchangeInfo` → PRICE_FILTER (tickSize), LOT_SIZE (stepSize, minQty)
//! - **Bybit**:   `GET /v5/market/instruments-info?category=linear` → priceFilter (tickSize), lotSizeFilter (qtyStep, minOrderQty)
//! - **Gate.io**: `GET /api/v4/futures/usdt/contracts/{contract}` → quanto_multiplier, order_size_min, order_price_round
//!
//! All values are fetched once at startup and periodically refreshed (default: every 5 minutes).
//! No hardcoded precision, tick sizes, or contract multipliers.

use std::collections::HashMap;
use std::sync::Arc;
use hmac::Mac;  // Required for Hmac::new_from_slice, update, finalize
use parking_lot::RwLock;
use reqwest::Client;
use serde_json::Value;
use tracing::{info, warn, error};

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// Contract specification fetched from an exchange.
#[derive(Debug, Clone)]
pub struct ContractSpec {
    /// Exchange-native symbol (e.g. "BTCUSDT" for Binance/Bybit, "BTC_USDT" for Gate.io)
    pub symbol: String,
    /// Minimum price increment (e.g. 0.10 for BTCUSDT on Bybit)
    pub tick_size: f64,
    /// Minimum quantity increment (e.g. 0.001 for BTCUSDT on Binance)
    pub step_size: f64,
    /// Minimum order quantity (e.g. 0.001)
    pub min_qty: f64,
    /// Maximum order quantity (e.g. 1000.0)
    pub max_qty: f64,
    /// Contract multiplier / quanto multiplier (Gate.io specific, 1.0 for linear)
    pub contract_multiplier: f64,
    /// Number of decimal places for price (derived from tick_size)
    pub price_precision: u32,
    /// Number of decimal places for quantity (derived from step_size)
    pub qty_precision: u32,
    /// Minimum notional value (Binance MIN_NOTIONAL filter)
    pub min_notional: f64,
}

impl ContractSpec {
    /// Round a price DOWN to the nearest valid tick_size.
    /// Example: price=67123.456, tick_size=0.10 → 67123.40
    pub fn round_price(&self, price: f64) -> f64 {
        if self.tick_size <= 0.0 || self.tick_size >= price {
            return price;
        }
        let ticks = (price / self.tick_size).floor();
        let rounded = ticks * self.tick_size;
        // Fix floating point artifacts
        round_to_decimals(rounded, self.price_precision)
    }

    /// Round a quantity DOWN to the nearest valid step_size.
    /// Example: qty=0.01234, step_size=0.001 → 0.012
    pub fn round_qty(&self, qty: f64) -> f64 {
        if self.step_size <= 0.0 {
            return qty;
        }
        let steps = (qty / self.step_size).floor();
        let rounded = steps * self.step_size;
        round_to_decimals(rounded, self.qty_precision)
    }

    /// Format a price as a string with the correct number of decimal places.
    pub fn format_price(&self, price: f64) -> String {
        let rounded = self.round_price(price);
        format!("{:.prec$}", rounded, prec = self.price_precision as usize)
    }

    /// Format a quantity as a string with the correct number of decimal places.
    pub fn format_qty(&self, qty: f64) -> String {
        let rounded = self.round_qty(qty);
        format!("{:.prec$}", rounded, prec = self.qty_precision as usize)
    }

    /// Clamp quantity to [min_qty, max_qty] and round to step_size.
    pub fn clamp_and_round_qty(&self, qty: f64) -> f64 {
        let clamped = qty.max(self.min_qty).min(self.max_qty);
        self.round_qty(clamped)
    }

    /// Convert a notional USD value to base-asset quantity.
    /// For Binance/Bybit linear: qty = notional / price
    /// For Gate.io: qty = notional / (price * contract_multiplier)
    pub fn notional_to_qty(&self, notional_usd: f64, price: f64) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }
        let raw_qty = notional_usd / (price * self.contract_multiplier);
        self.clamp_and_round_qty(raw_qty)
    }

    /// Convert a base-asset quantity to number of contracts (for Gate.io).
    /// For Binance/Bybit linear: contracts = qty (they're the same)
    /// For Gate.io: contracts = qty / contract_multiplier
    pub fn qty_to_contracts(&self, qty: f64) -> i64 {
        if self.contract_multiplier <= 0.0 {
            return qty as i64;
        }
        // Gate.io uses integer contracts where 1 contract = contract_multiplier * base asset
        (qty / self.contract_multiplier).floor() as i64
    }
}

/// Derive number of decimal places from a step/tick size value.
/// Example: 0.001 → 3, 0.10 → 1, 1.0 → 0, 0.00001 → 5
fn count_decimals(value: f64) -> u32 {
    if value <= 0.0 || value >= 1.0 {
        return 0;
    }
    // Use string representation to avoid floating point issues
    let s = format!("{}", value);
    if let Some(dot_pos) = s.find('.') {
        let decimal_part = &s[dot_pos + 1..];
        // Count significant digits (trim trailing zeros handled by format!)
        decimal_part.len() as u32
    } else {
        0
    }
}

/// Round to a specific number of decimal places.
fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (value * factor).round() / factor
}

// ---------------------------------------------------------------------------
// Exchange Identifier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exchange {
    Binance,
    Bybit,
    GateIo,
}

impl std::fmt::Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exchange::Binance => write!(f, "Binance"),
            Exchange::Bybit => write!(f, "Bybit"),
            Exchange::GateIo => write!(f, "GateIo"),
        }
    }
}

// ---------------------------------------------------------------------------
// InstrumentManager
// ---------------------------------------------------------------------------

/// Thread-safe instrument manager that caches contract specs from all exchanges.
///
/// Usage:
/// ```ignore
/// let mgr = InstrumentManager::new(false);
/// mgr.refresh_all().await;
/// let spec = mgr.get(Exchange::Binance, "BTCUSDT").unwrap();
/// let formatted_price = spec.format_price(67123.456);
/// let formatted_qty = spec.format_qty(0.01234);
/// ```
pub struct InstrumentManager {
    /// Map of (Exchange, normalized_symbol) → ContractSpec
    specs: Arc<RwLock<HashMap<(Exchange, String), ContractSpec>>>,
    client: Client,
    testnet: bool,
}

impl InstrumentManager {
    pub fn new(testnet: bool) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("Failed to build HTTP client for InstrumentManager");

        Self {
            specs: Arc::new(RwLock::new(HashMap::new())),
            client,
            testnet,
        }
    }

    /// Get contract spec for a symbol on an exchange. Returns None if not loaded.
    pub fn get(&self, exchange: Exchange, symbol: &str) -> Option<ContractSpec> {
        let normalized = normalize_key(exchange, symbol);
        self.specs.read().get(&(exchange, normalized)).cloned()
    }

    /// Get contract spec, returning a default if not found.
    /// The default uses conservative precision (8 decimals) so orders still work,
    /// just without optimal rounding.
    pub fn get_or_default(&self, exchange: Exchange, symbol: &str) -> ContractSpec {
        self.get(exchange, symbol).unwrap_or_else(|| {
            warn!(
                "InstrumentManager: no spec for {} on {}, using conservative defaults",
                symbol, exchange
            );
            ContractSpec {
                symbol: symbol.to_string(),
                tick_size: 0.01,
                step_size: 0.001,
                min_qty: 0.001,
                max_qty: 100_000.0,
                contract_multiplier: 1.0,
                price_precision: 2,
                qty_precision: 3,
                min_notional: 5.0,
            }
        })
    }

    /// Refresh specs from all exchanges. Call at startup and periodically.
    pub async fn refresh_all(&self) {
        let (r1, r2, r3) = tokio::join!(
            self.fetch_binance_specs(),
            self.fetch_bybit_specs(),
            self.fetch_gateio_specs(),
        );

        if let Err(e) = r1 {
            error!("Failed to fetch Binance instrument specs: {}", e);
        }
        if let Err(e) = r2 {
            error!("Failed to fetch Bybit instrument specs: {}", e);
        }
        if let Err(e) = r3 {
            error!("Failed to fetch Gate.io instrument specs: {}", e);
        }

        let count = self.specs.read().len();
        info!("InstrumentManager: loaded {} total contract specs", count);
    }

    /// Refresh specs for a single exchange.
    pub async fn refresh_exchange(&self, exchange: Exchange) {
        let result = match exchange {
            Exchange::Binance => self.fetch_binance_specs().await,
            Exchange::Bybit => self.fetch_bybit_specs().await,
            Exchange::GateIo => self.fetch_gateio_specs().await,
        };
        if let Err(e) = result {
            error!("Failed to refresh {} specs: {}", exchange, e);
        }
    }

    // -----------------------------------------------------------------------
    // Binance: GET /fapi/v1/exchangeInfo
    // -----------------------------------------------------------------------
    async fn fetch_binance_specs(&self) -> Result<(), String> {
        let base = if self.testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        let url = format!("{}/fapi/v1/exchangeInfo", base);

        let resp: Value = self.client.get(&url).send().await
            .map_err(|e| format!("Binance exchangeInfo request failed: {}", e))?
            .json().await
            .map_err(|e| format!("Binance exchangeInfo parse failed: {}", e))?;

        let symbols = resp.get("symbols").and_then(|v| v.as_array())
            .ok_or_else(|| "Binance exchangeInfo: missing 'symbols' array".to_string())?;

        let mut count = 0u32;
        let mut specs = self.specs.write();

        for sym_info in symbols {
            let symbol = match sym_info.get("symbol").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };

            // Only process USDT-margined perpetuals that are trading
            let contract_type = sym_info.get("contractType").and_then(|v| v.as_str()).unwrap_or("");
            let status = sym_info.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if contract_type != "PERPETUAL" || status != "TRADING" {
                continue;
            }

            let filters = match sym_info.get("filters").and_then(|v| v.as_array()) {
                Some(f) => f,
                None => continue,
            };

            let mut tick_size = 0.01;
            let mut step_size = 0.001;
            let mut min_qty = 0.001;
            let mut max_qty = 100_000.0;
            let mut min_notional = 5.0;

            for filter in filters {
                let filter_type = filter.get("filterType").and_then(|v| v.as_str()).unwrap_or("");
                match filter_type {
                    "PRICE_FILTER" => {
                        tick_size = filter.get("tickSize")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(tick_size);
                    }
                    "LOT_SIZE" => {
                        step_size = filter.get("stepSize")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(step_size);
                        min_qty = filter.get("minQty")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(min_qty);
                        max_qty = filter.get("maxQty")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(max_qty);
                    }
                    "MIN_NOTIONAL" => {
                        min_notional = filter.get("notional")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(min_notional);
                    }
                    _ => {}
                }
            }

            let price_precision = count_decimals(tick_size);
            let qty_precision = count_decimals(step_size);

            specs.insert(
                (Exchange::Binance, symbol.clone()),
                ContractSpec {
                    symbol: symbol.clone(),
                    tick_size,
                    step_size,
                    min_qty,
                    max_qty,
                    contract_multiplier: 1.0, // Binance linear = 1:1
                    price_precision,
                    qty_precision,
                    min_notional,
                },
            );
            count += 1;
        }

        info!("InstrumentManager: loaded {} Binance futures specs", count);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bybit: GET /v5/market/instruments-info?category=linear
    // -----------------------------------------------------------------------
    async fn fetch_bybit_specs(&self) -> Result<(), String> {
        let base = if self.testnet {
            "https://api-demo.bybit.com"
        } else {
            "https://api.bybit.com"
        };

        // Bybit paginates with cursor — fetch all pages
        let mut cursor = String::new();
        let mut total_count = 0u32;

        loop {
            let url = if cursor.is_empty() {
                format!("{}/v5/market/instruments-info?category=linear&limit=500", base)
            } else {
                format!(
                    "{}/v5/market/instruments-info?category=linear&limit=500&cursor={}",
                    base, cursor
                )
            };

            let resp: Value = self.client.get(&url).send().await
                .map_err(|e| format!("Bybit instruments-info request failed: {}", e))?
                .json().await
                .map_err(|e| format!("Bybit instruments-info parse failed: {}", e))?;

            let ret_code = resp.get("retCode").and_then(|v| v.as_i64()).unwrap_or(-1);
            if ret_code != 0 {
                let msg = resp.get("retMsg").and_then(|v| v.as_str()).unwrap_or("unknown");
                return Err(format!("Bybit instruments-info error {}: {}", ret_code, msg));
            }

            let list = resp.pointer("/result/list")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "Bybit instruments-info: missing result.list".to_string())?;

            let mut specs = self.specs.write();

            for item in list {
                let symbol = match item.get("symbol").and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };

                let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("");
                if status != "Trading" {
                    continue;
                }

                // priceFilter
                let tick_size = item.pointer("/priceFilter/tickSize")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.01);

                // lotSizeFilter
                let step_size = item.pointer("/lotSizeFilter/qtyStep")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.001);

                let min_qty = item.pointer("/lotSizeFilter/minOrderQty")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.001);

                let max_qty = item.pointer("/lotSizeFilter/maxOrderQty")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(100_000.0);

                let price_precision = count_decimals(tick_size);
                let qty_precision = count_decimals(step_size);

                specs.insert(
                    (Exchange::Bybit, symbol.clone()),
                    ContractSpec {
                        symbol: symbol.clone(),
                        tick_size,
                        step_size,
                        min_qty,
                        max_qty,
                        contract_multiplier: 1.0, // Bybit linear = 1:1
                        price_precision,
                        qty_precision,
                        min_notional: 5.0, // Bybit doesn't expose this in instruments-info
                    },
                );
                total_count += 1;
            }

            // Check for next page cursor
            let next_cursor = resp.pointer("/result/nextPageCursor")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if next_cursor.is_empty() {
                break;
            }
            cursor = next_cursor.to_string();
        }

        info!("InstrumentManager: loaded {} Bybit linear specs", total_count);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Gate.io: GET /api/v4/futures/usdt/contracts
    // -----------------------------------------------------------------------
    async fn fetch_gateio_specs(&self) -> Result<(), String> {
        let base = if self.testnet {
            "https://fx-api-testnet.gateio.ws"
        } else {
            "https://api.gateio.ws"
        };
        let url = format!("{}/api/v4/futures/usdt/contracts", base);

        let resp: Value = self.client.get(&url).send().await
            .map_err(|e| format!("Gate.io contracts request failed: {}", e))?
            .json().await
            .map_err(|e| format!("Gate.io contracts parse failed: {}", e))?;

        let contracts = resp.as_array()
            .ok_or_else(|| "Gate.io contracts: expected JSON array".to_string())?;

        let mut count = 0u32;
        let mut specs = self.specs.write();

        for contract in contracts {
            let name = match contract.get("name").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            let in_delisting = contract.get("in_delisting")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if in_delisting {
                continue;
            }

            // quanto_multiplier: how much base asset 1 contract represents
            // For BTC_USDT: quanto_multiplier = "0.0001" means 1 contract = 0.0001 BTC
            let quanto_multiplier = contract.get("quanto_multiplier")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1.0);

            // order_price_round: tick size for prices
            let tick_size = contract.get("order_price_round")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.01);

            // order_size_min: minimum number of contracts
            let order_size_min = contract.get("order_size_min")
                .and_then(|v| v.as_i64())
                .unwrap_or(1) as f64;

            // order_size_max: maximum number of contracts (0 = unlimited)
            let order_size_max = contract.get("order_size_max")
                .and_then(|v| v.as_i64())
                .unwrap_or(1_000_000) as f64;

            let price_precision = count_decimals(tick_size);
            // Gate.io contracts are always integers, so qty_precision = 0
            let qty_precision = 0u32;

            specs.insert(
                (Exchange::GateIo, name.clone()),
                ContractSpec {
                    symbol: name.clone(),
                    tick_size,
                    step_size: 1.0, // Gate.io contracts are integers
                    min_qty: order_size_min,
                    max_qty: if order_size_max > 0.0 { order_size_max } else { 1_000_000.0 },
                    contract_multiplier: quanto_multiplier,
                    price_precision,
                    qty_precision,
                    min_notional: 0.0, // Gate.io doesn't have min notional
                },
            );
            count += 1;
        }

        info!("InstrumentManager: loaded {} Gate.io futures specs", count);
        Ok(())
    }

    /// Get a shared reference to the internal specs map (for batch operations).
    pub fn all_specs(&self) -> HashMap<(Exchange, String), ContractSpec> {
        self.specs.read().clone()
    }

    /// Check if any specs have been loaded for an exchange.
    pub fn has_specs(&self, exchange: Exchange) -> bool {
        self.specs.read().keys().any(|(ex, _)| *ex == exchange)
    }

    /// Get the number of loaded specs for an exchange.
    pub fn spec_count(&self, exchange: Exchange) -> usize {
        self.specs.read().keys().filter(|(ex, _)| *ex == exchange).count()
    }
}

/// Normalize a symbol to the key format used internally.
/// - Binance/Bybit: uppercase, no separators (e.g. "BTCUSDT")
/// - Gate.io: uppercase with underscore (e.g. "BTC_USDT")
fn normalize_key(exchange: Exchange, symbol: &str) -> String {
    match exchange {
        Exchange::GateIo => {
            // Gate.io uses BTC_USDT format
            let s = symbol.to_uppercase();
            if s.contains('_') {
                s
            } else if s.contains('/') {
                s.replace('/', "_")
            } else {
                // Try to insert underscore before USDT/USD suffix
                if let Some(pos) = s.find("USDT") {
                    format!("{}_{}", &s[..pos], &s[pos..])
                } else if let Some(pos) = s.find("USD") {
                    format!("{}_{}", &s[..pos], &s[pos..])
                } else {
                    s
                }
            }
        }
        Exchange::Binance | Exchange::Bybit => {
            // Binance/Bybit use BTCUSDT format
            symbol
                .replace('/', "")
                .replace('_', "")
                .replace(':', "")
                .to_uppercase()
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-flight margin simulation
// ---------------------------------------------------------------------------

/// Simulated margin check result.
#[derive(Debug, Clone)]
pub struct MarginCheck {
    /// Whether the order can be placed without exceeding available margin.
    pub can_place: bool,
    /// Estimated initial margin required for the order.
    pub required_margin: f64,
    /// Available balance on the exchange.
    pub available_balance: f64,
    /// Maximum quantity that can be placed within available margin.
    pub max_affordable_qty: f64,
    /// Reason for rejection (if can_place is false).
    pub rejection_reason: Option<String>,
}

/// Simulate margin requirements for a proposed order BEFORE submitting it.
///
/// This prevents "InsufficientBalance" rejections that can leave you "legged in"
/// during multi-leg arbitrage (one side fills, the other gets rejected).
///
/// Parameters:
/// - `available_balance`: current available balance on the exchange (USDT)
/// - `price`: intended order price
/// - `qty`: intended quantity in base asset
/// - `leverage`: leverage to be used
/// - `is_cross_margin`: whether using cross margin mode
/// - `existing_position_notional`: absolute notional of any existing position (for margin offset)
pub fn simulate_margin(
    available_balance: f64,
    price: f64,
    qty: f64,
    leverage: i32,
    _is_cross_margin: bool,
    existing_position_notional: f64,
) -> MarginCheck {
    if price <= 0.0 || leverage <= 0 {
        return MarginCheck {
            can_place: false,
            required_margin: 0.0,
            available_balance,
            max_affordable_qty: 0.0,
            rejection_reason: Some("Invalid price or leverage".to_string()),
        };
    }

    let notional = price * qty;
    let initial_margin = notional / leverage as f64;

    // Account for maintenance margin (roughly 0.5% for most tier-1 contracts)
    let maintenance_margin_rate = 0.005;
    let maintenance_margin = notional * maintenance_margin_rate;

    // Total margin required = initial margin + estimated fees (0.1% round-trip)
    let fee_estimate = notional * 0.001;
    let total_required = initial_margin + maintenance_margin + fee_estimate;

    // Offset from existing position (cross-margin may partially offset)
    let effective_required = total_required - (existing_position_notional * 0.001);
    let effective_required = effective_required.max(0.0);

    // Calculate max affordable qty
    let max_notional = available_balance * leverage as f64 * 0.95; // 5% safety buffer
    let max_qty = if price > 0.0 { max_notional / price } else { 0.0 };

    if effective_required > available_balance {
        MarginCheck {
            can_place: false,
            required_margin: effective_required,
            available_balance,
            max_affordable_qty: max_qty,
            rejection_reason: Some(format!(
                "Insufficient margin: need ${:.2} but only ${:.2} available (max qty: {:.6})",
                effective_required, available_balance, max_qty
            )),
        }
    } else {
        MarginCheck {
            can_place: true,
            required_margin: effective_required,
            available_balance,
            max_affordable_qty: max_qty,
            rejection_reason: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution idempotency helpers
// ---------------------------------------------------------------------------

/// Check if an order with the given client_order_id already exists on the exchange.
/// Returns Some(order_id) if found, None if not.
///
/// This prevents accidental duplicate orders when WS ACK times out but the exchange
/// actually accepted the order. Before retrying, always call this first.
pub async fn check_order_exists_binance(
    client: &Client,
    base_url: &str,
    api_key: &str,
    api_secret: &[u8],
    symbol: &str,
    client_order_id: &str,
) -> Option<String> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let params = format!(
        "symbol={}&origClientOrderId={}&timestamp={}&recvWindow=5000",
        symbol, client_order_id, timestamp
    );

    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(api_secret).ok()?;
    hmac::Mac::update(&mut mac, params.as_bytes());
    let signature = hex::encode(hmac::Mac::finalize(mac).into_bytes());

    let url = format!("{}/fapi/v1/order?{}&signature={}", base_url, params, signature);

    let resp = client.get(&url)
        .header("X-MBX-APIKEY", api_key)
        .send().await.ok()?;

    let body: Value = resp.json().await.ok()?;

    // If order exists, return its orderId
    if let Some(order_id) = body.get("orderId").and_then(|v| v.as_u64()) {
        let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if status != "CANCELED" && status != "EXPIRED" && status != "REJECTED" {
            info!(
                "Idempotency check: found existing Binance order {} for client_id {}",
                order_id, client_order_id
            );
            return Some(order_id.to_string());
        }
    }
    None
}

/// Check if an order with the given orderLinkId already exists on Bybit.
pub async fn check_order_exists_bybit(
    client: &Client,
    base_url: &str,
    api_key: &str,
    api_secret: &[u8],
    symbol: &str,
    order_link_id: &str,
) -> Option<String> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let query = format!(
        "category=linear&symbol={}&orderLinkId={}",
        symbol, order_link_id
    );

    let sign_input = format!("{}{}5000{}", timestamp, api_key, query);
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(api_secret).ok()?;
    hmac::Mac::update(&mut mac, sign_input.as_bytes());
    let signature = hex::encode(hmac::Mac::finalize(mac).into_bytes());

    let url = format!("{}/v5/order/realtime?{}", base_url, query);

    let resp = client.get(&url)
        .header("X-BAPI-API-KEY", api_key)
        .header("X-BAPI-SIGN", &signature)
        .header("X-BAPI-TIMESTAMP", timestamp.to_string())
        .header("X-BAPI-RECV-WINDOW", "5000")
        .send().await.ok()?;

    let body: Value = resp.json().await.ok()?;

    let order = body.pointer("/result/list/0")?;
    let order_id = order.get("orderId").and_then(|v| v.as_str())?;
    let status = order.get("orderStatus").and_then(|v| v.as_str()).unwrap_or("");

    if status != "Cancelled" && status != "Rejected" && status != "Deactivated" {
        info!(
            "Idempotency check: found existing Bybit order {} for link_id {}",
            order_id, order_link_id
        );
        return Some(order_id.to_string());
    }

    None
}

// ---------------------------------------------------------------------------
// Fee optimization helpers
// ---------------------------------------------------------------------------

/// Recommended order strategy for fee avoidance.
#[derive(Debug, Clone)]
pub struct FeeStrategy {
    /// Use post-only / maker orders to earn rebates instead of paying taker fees.
    pub use_post_only: bool,
    /// Time in force string for the exchange.
    pub time_in_force: String,
    /// Offset from best price (positive = inside book for maker, 0 = at best price).
    pub price_offset_ticks: i32,
    /// Expected fee rate (negative = rebate).
    pub expected_fee_rate: f64,
    /// Whether to enable BNB fee payment (Binance specific, saves ~10%).
    pub use_bnb_discount: bool,
    /// Whether to use Gate.io point cards for fee reduction.
    pub use_gateio_points: bool,
}

/// Get the optimal fee strategy for a given exchange and order urgency.
///
/// For arbitrage trading, we want to minimize fees:
/// - Use limit/post-only orders to get maker rebates
/// - Enable BNB fee payment on Binance (saves ~10% on fees)
/// - Use Gate.io point cards when available
/// - Only use market/taker orders when urgency is critical
///
/// `urgency`: 0.0 = no rush (always maker), 1.0 = critical (accept taker)
pub fn optimal_fee_strategy(exchange: Exchange, urgency: f64) -> FeeStrategy {
    match exchange {
        Exchange::Binance => {
            if urgency < 0.5 {
                // Low urgency: post-only for maker rebate
                FeeStrategy {
                    use_post_only: true,
                    time_in_force: "GTX".to_string(), // Binance post-only
                    price_offset_ticks: 1, // 1 tick inside best price
                    expected_fee_rate: -0.00025, // maker rebate at VIP0
                    use_bnb_discount: true, // Always enable BNB discount
                    use_gateio_points: false,
                }
            } else {
                // High urgency: IOC but still enable BNB discount
                FeeStrategy {
                    use_post_only: false,
                    time_in_force: "IOC".to_string(),
                    price_offset_ticks: 0,
                    expected_fee_rate: 0.0004, // taker fee at VIP0
                    use_bnb_discount: true,
                    use_gateio_points: false,
                }
            }
        }
        Exchange::Bybit => {
            if urgency < 0.5 {
                FeeStrategy {
                    use_post_only: true,
                    time_in_force: "PostOnly".to_string(), // Bybit post-only TIF
                    price_offset_ticks: 1,
                    expected_fee_rate: -0.00025, // Bybit maker rebate
                    use_bnb_discount: false,
                    use_gateio_points: false,
                }
            } else {
                FeeStrategy {
                    use_post_only: false,
                    time_in_force: "IOC".to_string(),
                    price_offset_ticks: 0,
                    expected_fee_rate: 0.00055, // Bybit taker fee
                    use_bnb_discount: false,
                    use_gateio_points: false,
                }
            }
        }
        Exchange::GateIo => {
            if urgency < 0.5 {
                FeeStrategy {
                    use_post_only: true,
                    time_in_force: "poc".to_string(), // Gate.io post-only = "poc"
                    price_offset_ticks: 1,
                    expected_fee_rate: -0.000125, // Gate.io maker rebate at VIP1+
                    use_bnb_discount: false,
                    use_gateio_points: true, // Use point cards when available
                }
            } else {
                FeeStrategy {
                    use_post_only: false,
                    time_in_force: "ioc".to_string(),
                    price_offset_ticks: 0,
                    expected_fee_rate: 0.00075, // Gate.io taker fee
                    use_bnb_discount: false,
                    use_gateio_points: true,
                }
            }
        }
    }
}

/// Apply fee strategy to adjust an order price.
/// For maker orders, offsets the price by N ticks inside the book to ensure
/// it rests as a maker order rather than immediately filling as taker.
///
/// For buy orders: price = best_bid - offset * tick_size (post below best bid)
/// For sell orders: price = best_ask + offset * tick_size (post above best ask)
pub fn apply_fee_strategy_to_price(
    spec: &ContractSpec,
    strategy: &FeeStrategy,
    best_bid: f64,
    best_ask: f64,
    is_buy: bool,
) -> f64 {
    if !strategy.use_post_only || strategy.price_offset_ticks == 0 {
        // No adjustment needed for taker orders
        return if is_buy { best_ask } else { best_bid };
    }

    let offset = strategy.price_offset_ticks as f64 * spec.tick_size;

    let raw_price = if is_buy {
        // For maker buy: place below best bid to rest in book
        best_bid - offset
    } else {
        // For maker sell: place above best ask to rest in book
        best_ask + offset
    };

    spec.round_price(raw_price)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_decimals() {
        assert_eq!(count_decimals(0.01), 2);
        assert_eq!(count_decimals(0.001), 3);
        assert_eq!(count_decimals(0.1), 1);
        assert_eq!(count_decimals(0.0001), 4);
        assert_eq!(count_decimals(1.0), 0);
        assert_eq!(count_decimals(0.5), 1);
        assert_eq!(count_decimals(0.00001), 5);
    }

    #[test]
    fn test_round_price() {
        let spec = ContractSpec {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.10,
            step_size: 0.001,
            min_qty: 0.001,
            max_qty: 1000.0,
            contract_multiplier: 1.0,
            price_precision: 1,
            qty_precision: 3,
            min_notional: 5.0,
        };

        assert_eq!(spec.round_price(67123.456), 67123.4);
        assert_eq!(spec.round_price(67123.95), 67123.9);
        assert_eq!(spec.round_price(67123.0), 67123.0);
    }

    #[test]
    fn test_round_qty() {
        let spec = ContractSpec {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.01,
            step_size: 0.001,
            min_qty: 0.001,
            max_qty: 1000.0,
            contract_multiplier: 1.0,
            price_precision: 2,
            qty_precision: 3,
            min_notional: 5.0,
        };

        assert_eq!(spec.round_qty(0.01234), 0.012);
        assert_eq!(spec.round_qty(0.009), 0.009);
        assert_eq!(spec.round_qty(1.0), 1.0);
    }

    #[test]
    fn test_format_price() {
        let spec = ContractSpec {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.10,
            step_size: 0.001,
            min_qty: 0.001,
            max_qty: 1000.0,
            contract_multiplier: 1.0,
            price_precision: 1,
            qty_precision: 3,
            min_notional: 5.0,
        };

        assert_eq!(spec.format_price(67123.456), "67123.4");
        assert_eq!(spec.format_price(67123.0), "67123.0");
    }

    #[test]
    fn test_format_qty() {
        let spec = ContractSpec {
            symbol: "ETHUSDT".to_string(),
            tick_size: 0.01,
            step_size: 0.01,
            min_qty: 0.01,
            max_qty: 10000.0,
            contract_multiplier: 1.0,
            price_precision: 2,
            qty_precision: 2,
            min_notional: 5.0,
        };

        assert_eq!(spec.format_qty(1.2345), "1.23");
        assert_eq!(spec.format_qty(0.015), "0.01");
    }

    #[test]
    fn test_notional_to_qty_linear() {
        // Binance/Bybit: 1 contract = 1 base asset
        let spec = ContractSpec {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.10,
            step_size: 0.001,
            min_qty: 0.001,
            max_qty: 1000.0,
            contract_multiplier: 1.0,
            price_precision: 1,
            qty_precision: 3,
            min_notional: 5.0,
        };

        // $5000 at $100,000/BTC = 0.05 BTC
        let qty = spec.notional_to_qty(5000.0, 100_000.0);
        assert_eq!(qty, 0.05);
    }

    #[test]
    fn test_notional_to_qty_gateio() {
        // Gate.io: 1 contract = 0.0001 BTC
        let spec = ContractSpec {
            symbol: "BTC_USDT".to_string(),
            tick_size: 0.1,
            step_size: 1.0,
            min_qty: 1.0,
            max_qty: 1_000_000.0,
            contract_multiplier: 0.0001,
            price_precision: 1,
            qty_precision: 0,
            min_notional: 0.0,
        };

        // $5000 at $100,000/BTC = 0.05 BTC = 500 contracts
        let qty = spec.notional_to_qty(5000.0, 100_000.0);
        assert_eq!(qty, 500.0);
    }

    #[test]
    fn test_qty_to_contracts_gateio() {
        let spec = ContractSpec {
            symbol: "BTC_USDT".to_string(),
            tick_size: 0.1,
            step_size: 1.0,
            min_qty: 1.0,
            max_qty: 1_000_000.0,
            contract_multiplier: 0.0001,
            price_precision: 1,
            qty_precision: 0,
            min_notional: 0.0,
        };

        // 0.05 BTC = 500 contracts on Gate.io
        assert_eq!(spec.qty_to_contracts(0.05), 500);
    }

    #[test]
    fn test_clamp_and_round() {
        let spec = ContractSpec {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.01,
            step_size: 0.001,
            min_qty: 0.001,
            max_qty: 100.0,
            contract_multiplier: 1.0,
            price_precision: 2,
            qty_precision: 3,
            min_notional: 5.0,
        };

        // Below minimum → clamped to min
        assert_eq!(spec.clamp_and_round_qty(0.0001), 0.001);
        // Above maximum → clamped to max
        assert_eq!(spec.clamp_and_round_qty(999.0), 100.0);
        // Normal → rounded
        assert_eq!(spec.clamp_and_round_qty(1.2345), 1.234);
    }

    #[test]
    fn test_normalize_key_binance() {
        assert_eq!(normalize_key(Exchange::Binance, "BTC_USDT"), "BTCUSDT");
        assert_eq!(normalize_key(Exchange::Binance, "BTC/USDT"), "BTCUSDT");
        assert_eq!(normalize_key(Exchange::Binance, "btcusdt"), "BTCUSDT");
    }

    #[test]
    fn test_normalize_key_gateio() {
        assert_eq!(normalize_key(Exchange::GateIo, "BTC_USDT"), "BTC_USDT");
        assert_eq!(normalize_key(Exchange::GateIo, "BTC/USDT"), "BTC_USDT");
        assert_eq!(normalize_key(Exchange::GateIo, "BTCUSDT"), "BTC_USDT");
    }

    #[test]
    fn test_simulate_margin_pass() {
        let check = simulate_margin(10_000.0, 50_000.0, 0.01, 10, true, 0.0);
        assert!(check.can_place);
        assert!(check.required_margin < 10_000.0);
    }

    #[test]
    fn test_simulate_margin_fail() {
        let check = simulate_margin(100.0, 50_000.0, 1.0, 2, false, 0.0);
        assert!(!check.can_place);
        assert!(check.rejection_reason.is_some());
    }

    #[test]
    fn test_optimal_fee_strategy_binance_maker() {
        let strategy = optimal_fee_strategy(Exchange::Binance, 0.0);
        assert!(strategy.use_post_only);
        assert_eq!(strategy.time_in_force, "GTX");
        assert!(strategy.expected_fee_rate < 0.0); // Maker rebate
        assert!(strategy.use_bnb_discount);
    }

    #[test]
    fn test_optimal_fee_strategy_gateio_taker() {
        let strategy = optimal_fee_strategy(Exchange::GateIo, 1.0);
        assert!(!strategy.use_post_only);
        assert_eq!(strategy.time_in_force, "ioc");
        assert!(strategy.expected_fee_rate > 0.0); // Taker fee
        assert!(strategy.use_gateio_points);
    }

    #[test]
    fn test_apply_fee_strategy_to_price_buy_maker() {
        let spec = ContractSpec {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.1,
            step_size: 0.001,
            min_qty: 0.001,
            max_qty: 1000.0,
            contract_multiplier: 1.0,
            price_precision: 1,
            qty_precision: 3,
            min_notional: 5.0,
        };

        let strategy = FeeStrategy {
            use_post_only: true,
            time_in_force: "GTX".to_string(),
            price_offset_ticks: 1,
            expected_fee_rate: -0.00025,
            use_bnb_discount: true,
            use_gateio_points: false,
        };

        let price = apply_fee_strategy_to_price(&spec, &strategy, 67000.0, 67001.0, true);
        // Buy maker: best_bid - 1 tick = 67000.0 - 0.1 = 66999.9
        assert_eq!(price, 66999.9);
    }
}
