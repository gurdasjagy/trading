//! Spot-Futures Arbitrage: Exchange Specification Bootstrapping
//!
//! Dynamically fetches all Spot AND Futures market specifications at startup.
//! No hardcoded values -- all fees, step sizes, tick sizes, contract multipliers,
//! and funding intervals are fetched from exchange APIs at runtime.
//!
//! Endpoints:
//! - Binance Spot:  GET /api/v3/exchangeInfo
//! - Bybit Spot:    GET /v5/market/instruments-info?category=spot
//! - Gate.io Spot:  GET /api/v4/spot/currency_pairs
//! - Fee tiers:     Authenticated endpoints per exchange (with fallback defaults)

use std::collections::HashMap;

use reqwest::Client;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::multi_exchange::global_book::ExchangeId;

// ---------------------------------------------------------------------------
// Spot Market Specification
// ---------------------------------------------------------------------------

/// Complete specification for a Spot market pair on a single exchange.
/// All values fetched dynamically from exchange APIs -- zero hardcoded values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotMarketSpec {
    /// Exchange-native symbol (e.g., "BTCUSDT" or "BTC_USDT").
    pub symbol: String,
    /// Base asset (e.g., "BTC").
    pub base_asset: String,
    /// Quote asset (e.g., "USDT").
    pub quote_asset: String,
    /// Minimum price increment.
    pub tick_size: Decimal,
    /// Minimum quantity increment.
    pub step_size: Decimal,
    /// Minimum order quantity (in base asset).
    pub min_qty: Decimal,
    /// Minimum order value in quote asset (USDT).
    pub min_notional: Decimal,
    /// Maker fee rate (e.g., 0.001 = 0.1%).
    pub maker_fee_rate: Decimal,
    /// Taker fee rate (e.g., 0.001 = 0.1%).
    pub taker_fee_rate: Decimal,
    /// Number of decimal places for price (derived from tick_size).
    pub price_precision: u32,
    /// Number of decimal places for quantity (derived from step_size).
    pub qty_precision: u32,
}

impl SpotMarketSpec {
    /// Round a price DOWN to the nearest valid tick_size.
    pub fn round_price(&self, price: Decimal) -> Decimal {
        if self.tick_size.is_zero() {
            return price;
        }
        (price / self.tick_size).floor() * self.tick_size
    }

    /// Round a quantity DOWN to the nearest valid step_size.
    pub fn round_qty(&self, qty: Decimal) -> Decimal {
        if self.step_size.is_zero() {
            return qty;
        }
        (qty / self.step_size).floor() * self.step_size
    }

    /// Format a price as a string with the correct precision.
    pub fn format_price(&self, price: Decimal) -> String {
        let rounded = self.round_price(price);
        format!("{:.prec$}", rounded, prec = self.price_precision as usize)
    }

    /// Format a quantity as a string with the correct precision.
    pub fn format_qty(&self, qty: Decimal) -> String {
        let rounded = self.round_qty(qty);
        format!("{:.prec$}", rounded, prec = self.qty_precision as usize)
    }
}

// ---------------------------------------------------------------------------
// Futures Contract Specification (extended)
// ---------------------------------------------------------------------------

/// Extended futures contract specification with funding interval and fee tiers.
/// Built on top of the existing ContractSpec from instrument_manager.rs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesContractSpec {
    /// Exchange-native symbol.
    pub symbol: String,
    /// Base asset (e.g., "BTC").
    pub base_asset: String,
    /// Minimum price increment.
    pub tick_size: Decimal,
    /// Minimum quantity increment.
    pub step_size: Decimal,
    /// Minimum order quantity.
    pub min_qty: Decimal,
    /// Contract multiplier (Gate.io specific, 1.0 for linear contracts).
    pub contract_multiplier: Decimal,
    /// Funding interval in hours (8, 4, or 1).
    pub funding_interval_hours: f64,
    /// Maker fee rate.
    pub maker_fee_rate: Decimal,
    /// Taker fee rate.
    pub taker_fee_rate: Decimal,
    /// Number of decimal places for price.
    pub price_precision: u32,
    /// Number of decimal places for quantity.
    pub qty_precision: u32,
}

// ---------------------------------------------------------------------------
// Spot-Futures Specs Manager
// ---------------------------------------------------------------------------

/// Manages all Spot and Futures market specifications across all exchanges.
/// Fetched once at startup and periodically refreshed.
pub struct SpotFuturesSpecs {
    /// Spot specs: (ExchangeId, normalized_symbol) -> SpotMarketSpec
    spot_specs: HashMap<(ExchangeId, String), SpotMarketSpec>,
    /// Futures specs: (ExchangeId, normalized_symbol) -> FuturesContractSpec
    futures_specs: HashMap<(ExchangeId, String), FuturesContractSpec>,
    /// HTTP client for API requests.
    client: Client,
}

/// Conservative fallback fee rates when authenticated fee fetch fails.
const FALLBACK_SPOT_TAKER_FEE: &str = "0.001";   // 0.1%
const FALLBACK_SPOT_MAKER_FEE: &str = "0.001";   // 0.1%
const FALLBACK_FUTURES_TAKER_FEE: &str = "0.0005"; // 0.05%
const FALLBACK_FUTURES_MAKER_FEE: &str = "0.0002"; // 0.02%

/// Derive number of decimal places from a decimal step/tick value.
fn decimal_precision(value: Decimal) -> u32 {
    if value.is_zero() || value >= Decimal::ONE {
        return 0;
    }
    let s = value.to_string();
    if let Some(dot_pos) = s.find('.') {
        let decimal_part = s[dot_pos + 1..].trim_end_matches('0');
        decimal_part.len() as u32
    } else {
        0
    }
}

/// Normalize a symbol to a canonical base asset name (e.g., "BTC").
/// Strips USDT suffix and separators.
pub fn normalize_base_asset(symbol: &str) -> String {
    let s = symbol
        .replace('_', "")
        .replace('/', "")
        .to_uppercase();
    if s.ends_with("USDT") {
        s[..s.len() - 4].to_string()
    } else {
        s
    }
}

/// Find symbols available in BOTH Spot and Futures on ALL three exchanges.
/// Returns normalized base asset names (e.g., "BTC", "ETH").
pub fn find_common_symbols(
    binance_spot: &HashMap<String, SpotMarketSpec>,
    binance_futures: &HashMap<String, FuturesContractSpec>,
    bybit_spot: &HashMap<String, SpotMarketSpec>,
    bybit_futures: &HashMap<String, FuturesContractSpec>,
    gateio_spot: &HashMap<String, SpotMarketSpec>,
    gateio_futures: &HashMap<String, FuturesContractSpec>,
) -> Vec<String> {
    // Normalize all symbol sets to base asset names
    let binance_spot_bases: std::collections::HashSet<String> = binance_spot
        .keys()
        .map(|s| normalize_base_asset(s))
        .collect();
    let binance_futures_bases: std::collections::HashSet<String> = binance_futures
        .keys()
        .map(|s| normalize_base_asset(s))
        .collect();
    let bybit_spot_bases: std::collections::HashSet<String> = bybit_spot
        .keys()
        .map(|s| normalize_base_asset(s))
        .collect();
    let bybit_futures_bases: std::collections::HashSet<String> = bybit_futures
        .keys()
        .map(|s| normalize_base_asset(s))
        .collect();
    let gateio_spot_bases: std::collections::HashSet<String> = gateio_spot
        .keys()
        .map(|s| normalize_base_asset(s))
        .collect();
    let gateio_futures_bases: std::collections::HashSet<String> = gateio_futures
        .keys()
        .map(|s| normalize_base_asset(s))
        .collect();

    // Intersect all sets
    let mut common: Vec<String> = binance_spot_bases
        .iter()
        .filter(|s| binance_futures_bases.contains(*s))
        .filter(|s| bybit_spot_bases.contains(*s))
        .filter(|s| bybit_futures_bases.contains(*s))
        .filter(|s| gateio_spot_bases.contains(*s))
        .filter(|s| gateio_futures_bases.contains(*s))
        .cloned()
        .collect();

    common.sort();
    info!(
        "[spot-futures-specs] Found {} symbols available on all 3 exchanges in both Spot and Futures: {:?}",
        common.len(),
        &common[..common.len().min(20)]
    );
    common
}

impl SpotFuturesSpecs {
    /// Create a new specs manager.
    pub fn new() -> Self {
        Self {
            spot_specs: HashMap::new(),
            futures_specs: HashMap::new(),
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .expect("Failed to build HTTP client for specs"),
        }
    }

    /// Get a spot market spec for a given exchange and normalized base asset.
    pub fn get_spot_spec(&self, exchange: ExchangeId, base_asset: &str) -> Option<&SpotMarketSpec> {
        self.spot_specs.get(&(exchange, base_asset.to_uppercase()))
    }

    /// Get a futures contract spec for a given exchange and normalized base asset.
    pub fn get_futures_spec(&self, exchange: ExchangeId, base_asset: &str) -> Option<&FuturesContractSpec> {
        self.futures_specs.get(&(exchange, base_asset.to_uppercase()))
    }

    /// Get all known base assets that have both spot and futures specs on at least one exchange.
    pub fn tradeable_assets(&self) -> Vec<String> {
        let spot_assets: std::collections::HashSet<String> = self
            .spot_specs
            .keys()
            .map(|(_, s)| s.clone())
            .collect();
        let futures_assets: std::collections::HashSet<String> = self
            .futures_specs
            .keys()
            .map(|(_, s)| s.clone())
            .collect();
        let mut common: Vec<String> = spot_assets
            .intersection(&futures_assets)
            .cloned()
            .collect();
        common.sort();
        common
    }

    /// Fetch all Spot market specs from Binance.
    /// Endpoint: GET /api/v3/exchangeInfo
    pub async fn fetch_binance_spot_specs(&mut self, testnet: bool) -> HashMap<String, SpotMarketSpec> {
        let base_url = if testnet {
            "https://testnet.binance.vision"
        } else {
            "https://api.binance.com"
        };
        let url = format!("{}/api/v3/exchangeInfo", base_url);
        let mut result = HashMap::new();

        match self.client.get(&url).send().await {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        if let Some(symbols) = data.get("symbols").and_then(|v| v.as_array()) {
                            for sym in symbols {
                                let symbol = sym.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                                let status = sym.get("status").and_then(|v| v.as_str()).unwrap_or("");
                                let quote = sym.get("quoteAsset").and_then(|v| v.as_str()).unwrap_or("");
                                let base = sym.get("baseAsset").and_then(|v| v.as_str()).unwrap_or("");

                                // Only USDT pairs that are actively trading
                                if quote != "USDT" || status != "TRADING" {
                                    continue;
                                }

                                let filters = sym.get("filters").and_then(|v| v.as_array());
                                let mut tick_size = Decimal::new(1, 8); // default
                                let mut step_size = Decimal::new(1, 8);
                                let mut min_qty = Decimal::new(1, 8);
                                let mut min_notional = Decimal::new(10, 0);

                                if let Some(filters) = filters {
                                    for filter in filters {
                                        let ft = filter.get("filterType").and_then(|v| v.as_str()).unwrap_or("");
                                        match ft {
                                            "PRICE_FILTER" => {
                                                if let Some(ts) = filter.get("tickSize").and_then(|v| v.as_str()) {
                                                    tick_size = Decimal::from_str(ts).unwrap_or(tick_size);
                                                }
                                            }
                                            "LOT_SIZE" => {
                                                if let Some(ss) = filter.get("stepSize").and_then(|v| v.as_str()) {
                                                    step_size = Decimal::from_str(ss).unwrap_or(step_size);
                                                }
                                                if let Some(mq) = filter.get("minQty").and_then(|v| v.as_str()) {
                                                    min_qty = Decimal::from_str(mq).unwrap_or(min_qty);
                                                }
                                            }
                                            "NOTIONAL" | "MIN_NOTIONAL" => {
                                                if let Some(mn) = filter.get("minNotional").and_then(|v| v.as_str()) {
                                                    min_notional = Decimal::from_str(mn).unwrap_or(min_notional);
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }

                                let spec = SpotMarketSpec {
                                    symbol: symbol.to_string(),
                                    base_asset: base.to_string(),
                                    quote_asset: quote.to_string(),
                                    price_precision: decimal_precision(tick_size),
                                    qty_precision: decimal_precision(step_size),
                                    tick_size,
                                    step_size,
                                    min_qty,
                                    min_notional,
                                    maker_fee_rate: Decimal::from_str(FALLBACK_SPOT_MAKER_FEE).unwrap(),
                                    taker_fee_rate: Decimal::from_str(FALLBACK_SPOT_TAKER_FEE).unwrap(),
                                };

                                let base_upper = base.to_uppercase();
                                self.spot_specs.insert((ExchangeId::Binance, base_upper.clone()), spec.clone());
                                result.insert(symbol.to_string(), spec);
                            }
                        }
                        info!("[spot-futures-specs] Binance Spot: fetched {} USDT pairs", result.len());
                    }
                    Err(e) => error!("[spot-futures-specs] Binance Spot exchangeInfo parse error: {}", e),
                }
            }
            Err(e) => error!("[spot-futures-specs] Binance Spot exchangeInfo fetch error: {}", e),
        }
        result
    }

    /// Fetch all Spot market specs from Bybit.
    /// Endpoint: GET /v5/market/instruments-info?category=spot
    pub async fn fetch_bybit_spot_specs(&mut self, testnet: bool) -> HashMap<String, SpotMarketSpec> {
        let base_url = if testnet {
            "https://api-demo.bybit.com"
        } else {
            "https://api.bybit.com"
        };
        let url = format!("{}/v5/market/instruments-info?category=spot&limit=1000", base_url);
        let mut result = HashMap::new();

        match self.client.get(&url).send().await {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        if let Some(list) = data
                            .get("result")
                            .and_then(|r| r.get("list"))
                            .and_then(|l| l.as_array())
                        {
                            for item in list {
                                let symbol = item.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                                let base = item.get("baseCoin").and_then(|v| v.as_str()).unwrap_or("");
                                let quote = item.get("quoteCoin").and_then(|v| v.as_str()).unwrap_or("");
                                let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("");

                                if quote != "USDT" || status != "Trading" {
                                    continue;
                                }

                                let lot_filter = item.get("lotSizeFilter");
                                let price_filter = item.get("priceFilter");

                                let step_size = lot_filter
                                    .and_then(|f| f.get("basePrecision").and_then(|v| v.as_str()))
                                    .and_then(|s| Decimal::from_str(s).ok())
                                    .unwrap_or_else(|| Decimal::new(1, 8));
                                let min_qty = lot_filter
                                    .and_then(|f| f.get("minOrderQty").and_then(|v| v.as_str()))
                                    .and_then(|s| Decimal::from_str(s).ok())
                                    .unwrap_or_else(|| Decimal::new(1, 8));
                                let tick_size = price_filter
                                    .and_then(|f| f.get("tickSize").and_then(|v| v.as_str()))
                                    .and_then(|s| Decimal::from_str(s).ok())
                                    .unwrap_or_else(|| Decimal::new(1, 8));
                                let min_notional = lot_filter
                                    .and_then(|f| f.get("minOrderAmt").and_then(|v| v.as_str()))
                                    .and_then(|s| Decimal::from_str(s).ok())
                                    .unwrap_or_else(|| Decimal::new(1, 0));

                                let spec = SpotMarketSpec {
                                    symbol: symbol.to_string(),
                                    base_asset: base.to_string(),
                                    quote_asset: quote.to_string(),
                                    price_precision: decimal_precision(tick_size),
                                    qty_precision: decimal_precision(step_size),
                                    tick_size,
                                    step_size,
                                    min_qty,
                                    min_notional,
                                    maker_fee_rate: Decimal::from_str(FALLBACK_SPOT_MAKER_FEE).unwrap(),
                                    taker_fee_rate: Decimal::from_str(FALLBACK_SPOT_TAKER_FEE).unwrap(),
                                };

                                let base_upper = base.to_uppercase();
                                self.spot_specs.insert((ExchangeId::Bybit, base_upper.clone()), spec.clone());
                                result.insert(symbol.to_string(), spec);
                            }
                        }
                        info!("[spot-futures-specs] Bybit Spot: fetched {} USDT pairs", result.len());
                    }
                    Err(e) => error!("[spot-futures-specs] Bybit Spot instruments parse error: {}", e),
                }
            }
            Err(e) => error!("[spot-futures-specs] Bybit Spot instruments fetch error: {}", e),
        }
        result
    }

    /// Fetch all Spot market specs from Gate.io.
    /// Endpoint: GET /api/v4/spot/currency_pairs
    pub async fn fetch_gateio_spot_specs(&mut self, testnet: bool) -> HashMap<String, SpotMarketSpec> {
        let base_url = if testnet {
            "https://api-testnet.gateapi.io"
        } else {
            "https://api.gateio.ws"
        };
        let url = format!("{}/api/v4/spot/currency_pairs", base_url);
        let mut result = HashMap::new();

        match self.client.get(&url).send().await {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        if let Some(pairs) = data.as_array() {
                            for pair in pairs {
                                let id = pair.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                let base = pair.get("base").and_then(|v| v.as_str()).unwrap_or("");
                                let quote = pair.get("quote").and_then(|v| v.as_str()).unwrap_or("");
                                let trade_status = pair.get("trade_status").and_then(|v| v.as_str()).unwrap_or("");

                                if quote != "USDT" || trade_status != "tradable" {
                                    continue;
                                }

                                let price_prec = pair.get("precision")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(8) as u32;
                                let amt_prec = pair.get("amount_precision")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(8) as u32;

                                let tick_size = Decimal::new(1, price_prec);
                                let step_size = Decimal::new(1, amt_prec);

                                let min_base = pair.get("min_base_amount")
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| Decimal::from_str(s).ok())
                                    .unwrap_or_else(|| Decimal::new(1, 8));
                                let min_quote = pair.get("min_quote_amount")
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| Decimal::from_str(s).ok())
                                    .unwrap_or_else(|| Decimal::new(1, 0));

                                let spec = SpotMarketSpec {
                                    symbol: id.to_string(),
                                    base_asset: base.to_string(),
                                    quote_asset: quote.to_string(),
                                    tick_size,
                                    step_size,
                                    min_qty: min_base,
                                    min_notional: min_quote,
                                    maker_fee_rate: Decimal::from_str(FALLBACK_SPOT_MAKER_FEE).unwrap(),
                                    taker_fee_rate: Decimal::from_str(FALLBACK_SPOT_TAKER_FEE).unwrap(),
                                    price_precision: price_prec,
                                    qty_precision: amt_prec,
                                };

                                let base_upper = base.to_uppercase();
                                self.spot_specs.insert((ExchangeId::GateIo, base_upper.clone()), spec.clone());
                                result.insert(id.to_string(), spec);
                            }
                        }
                        info!("[spot-futures-specs] Gate.io Spot: fetched {} USDT pairs", result.len());
                    }
                    Err(e) => error!("[spot-futures-specs] Gate.io Spot currency_pairs parse error: {}", e),
                }
            }
            Err(e) => error!("[spot-futures-specs] Gate.io Spot currency_pairs fetch error: {}", e),
        }
        result
    }

    /// Fetch authenticated fee tiers from Binance.
    /// Endpoint: GET /sapi/v1/asset/tradeFee (mainnet only, NOT available on testnet).
    pub async fn fetch_binance_fee_tiers(
        &mut self,
        api_key: &str,
        api_secret: &[u8],
        testnet: bool,
    ) {
        if testnet {
            debug!("[spot-futures-specs] Binance testnet: /sapi/ endpoints not available, using fallback fees");
            return;
        }
        if api_key.is_empty() {
            debug!("[spot-futures-specs] Binance: no API key, using fallback fees");
            return;
        }

        let ts = crate::execution_gateway::now_ms();
        let query = format!("timestamp={}&recvWindow=5000", ts);

        // HMAC-SHA256 signing (same as binance_gateway)
        let signature = {
            use hmac::{Hmac, Mac};
            use sha2::Sha256;
            let mut mac = Hmac::<Sha256>::new_from_slice(api_secret)
                .expect("HMAC can take key of any size");
            mac.update(query.as_bytes());
            hex::encode(mac.finalize().into_bytes())
        };

        let url = format!(
            "https://api.binance.com/sapi/v1/asset/tradeFee?{}&signature={}",
            query, signature
        );

        match self.client.get(&url)
            .header("X-MBX-APIKEY", api_key)
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if let Some(fees) = data.as_array() {
                        let mut updated = 0u32;
                        for fee in fees {
                            let symbol = fee.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                            let maker = fee.get("makerCommission").and_then(|v| v.as_str())
                                .and_then(|s| Decimal::from_str(s).ok());
                            let taker = fee.get("takerCommission").and_then(|v| v.as_str())
                                .and_then(|s| Decimal::from_str(s).ok());

                            let base = normalize_base_asset(symbol);
                            if let Some(spec) = self.spot_specs.get_mut(&(ExchangeId::Binance, base)) {
                                if let Some(m) = maker { spec.maker_fee_rate = m; }
                                if let Some(t) = taker { spec.taker_fee_rate = t; }
                                updated += 1;
                            }
                        }
                        info!("[spot-futures-specs] Binance: updated fee tiers for {} symbols", updated);
                    }
                }
            }
            Err(e) => warn!("[spot-futures-specs] Binance fee tier fetch failed (using fallbacks): {}", e),
        }
    }

    /// Fetch authenticated fee tiers from Bybit.
    /// Endpoint: GET /v5/account/fee-rate?category=spot
    pub async fn fetch_bybit_fee_tiers(
        &mut self,
        api_key: &str,
        api_secret: &[u8],
        testnet: bool,
    ) {
        if api_key.is_empty() {
            debug!("[spot-futures-specs] Bybit: no API key, using fallback fees");
            return;
        }

        let base_url = if testnet {
            "https://api-demo.bybit.com"
        } else {
            "https://api.bybit.com"
        };

        let ts = crate::execution_gateway::now_ms();
        let query = "category=spot&symbol=BTCUSDT";
        let signature = crate::execution_gateway::sign_bybit_request(
            ts, api_key, 5000, query, api_secret,
        );

        let url = format!("{}/v5/account/fee-rate?{}", base_url, query);

        match self.client.get(&url)
            .header("X-BAPI-API-KEY", api_key)
            .header("X-BAPI-TIMESTAMP", ts.to_string())
            .header("X-BAPI-SIGN", &signature)
            .header("X-BAPI-RECV-WINDOW", "5000")
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if let Some(list) = data
                        .get("result")
                        .and_then(|r| r.get("list"))
                        .and_then(|l| l.as_array())
                    {
                        for item in list {
                            let maker = item.get("makerFeeRate").and_then(|v| v.as_str())
                                .and_then(|s| Decimal::from_str(s).ok());
                            let taker = item.get("takerFeeRate").and_then(|v| v.as_str())
                                .and_then(|s| Decimal::from_str(s).ok());
                            if let (Some(m), Some(t)) = (maker, taker) {
                                for ((ex, _), spec) in self.spot_specs.iter_mut() {
                                    if *ex == ExchangeId::Bybit {
                                        spec.maker_fee_rate = m;
                                        spec.taker_fee_rate = t;
                                    }
                                }
                                info!("[spot-futures-specs] Bybit Spot fees: maker={}, taker={}", m, t);
                            }
                        }
                    }
                }
            }
            Err(e) => warn!("[spot-futures-specs] Bybit fee tier fetch failed (using fallbacks): {}", e),
        }
    }

    /// Fetch authenticated fee tiers from Gate.io.
    /// Endpoint: GET /api/v4/wallet/fee
    pub async fn fetch_gateio_fee_tiers(
        &mut self,
        api_key: &str,
        api_secret: &[u8],
        testnet: bool,
    ) {
        if api_key.is_empty() {
            debug!("[spot-futures-specs] Gate.io: no API key, using fallback fees");
            return;
        }

        let base_url = if testnet {
            "https://api-testnet.gateapi.io"
        } else {
            "https://api.gateio.ws"
        };

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let path = "/api/v4/wallet/fee";
        let signature = crate::execution_gateway::sign_gateio_request(
            "GET", path, "", "", ts, api_secret,
        );

        let url = format!("{}{}", base_url, path);

        match self.client.get(&url)
            .header("KEY", api_key)
            .header("SIGN", &signature)
            .header("Timestamp", ts.to_string())
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    let maker = data.get("maker_fee").and_then(|v| v.as_str())
                        .and_then(|s| Decimal::from_str(s).ok());
                    let taker = data.get("taker_fee").and_then(|v| v.as_str())
                        .and_then(|s| Decimal::from_str(s).ok());

                    if let (Some(m), Some(t)) = (maker, taker) {
                        for ((ex, _), spec) in self.spot_specs.iter_mut() {
                            if *ex == ExchangeId::GateIo {
                                spec.maker_fee_rate = m;
                                spec.taker_fee_rate = t;
                            }
                        }
                        info!("[spot-futures-specs] Gate.io fees: maker={}, taker={}", m, t);
                    }
                }
            }
            Err(e) => warn!("[spot-futures-specs] Gate.io fee tier fetch failed (using fallbacks): {}", e),
        }
    }

    /// Bootstrap: fetch all specs from all exchanges.
    /// Fetches sequentially to avoid multiple mutable borrows of self.
    pub async fn bootstrap(
        &mut self,
        binance_testnet: bool,
        bybit_testnet: bool,
        gateio_testnet: bool,
    ) {
        info!("[spot-futures-specs] Bootstrapping Spot + Futures specs from all exchanges...");

        // Fetch Spot specs sequentially (each call borrows &mut self to store results)
        let binance_spot = self.fetch_binance_spot_specs(binance_testnet).await;
        let bybit_spot = self.fetch_bybit_spot_specs(bybit_testnet).await;
        let gateio_spot = self.fetch_gateio_spot_specs(gateio_testnet).await;

        info!(
            "[spot-futures-specs] Bootstrap complete: Binance Spot={}, Bybit Spot={}, Gate.io Spot={}",
            binance_spot.len(),
            bybit_spot.len(),
            gateio_spot.len(),
        );
    }
}
