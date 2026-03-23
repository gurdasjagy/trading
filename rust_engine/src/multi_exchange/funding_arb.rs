//! Cross-Exchange Perpetual Funding Rate Arbitrage
//!
//! Monitors funding rates across Gate.io, Binance, and Bybit to detect
//! arbitrage opportunities where:
//! - Go SHORT on the exchange with high positive funding (shorts receive payment)
//! - Go LONG on the exchange with low/negative funding (longs receive payment)
//!
//! The net funding income minus trading fees can provide risk-free yield.

use std::collections::HashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::multi_exchange::global_book::ExchangeId;

// ---------------------------------------------------------------------------
// Funding Rate Data
// ---------------------------------------------------------------------------

/// Funding rate data for a single exchange + symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRateData {
    pub exchange: ExchangeId,
    pub symbol: String,
    /// Funding rate (e.g. 0.0001 = 0.01%)
    pub rate: f64,
    /// Unix timestamp of next funding
    pub next_funding_ts: u64,
    /// Timestamp when this data was fetched (nanoseconds)
    pub updated_ns: u64,
}

impl FundingRateData {
    /// Convert rate to basis points.
    pub fn rate_bps(&self) -> f64 {
        self.rate * 10000.0
    }

    /// Convert rate to percentage string.
    pub fn rate_pct_str(&self) -> String {
        format!("{:.4}%", self.rate * 100.0)
    }
}

// ---------------------------------------------------------------------------
// Funding Arbitrage Opportunity
// ---------------------------------------------------------------------------

/// A detected funding rate arbitrage opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbOpportunity {
    /// Exchange to go SHORT on (high positive funding = shorts receive payment).
    pub short_exchange: ExchangeId,
    /// Exchange to go LONG on (low or negative funding = longs receive payment).
    pub long_exchange: ExchangeId,
    pub symbol: String,
    /// Net funding rate spread (short_rate - long_rate).
    pub net_rate: f64,
    /// Annualized APR of the spread.
    pub annualized_apr: f64,
    /// Estimated net profit after fees (basis points per funding period).
    pub net_profit_bps: f64,
    /// Whether this opportunity is currently actionable.
    pub is_actionable: bool,
    /// Short exchange funding rate.
    pub short_rate: f64,
    /// Long exchange funding rate.
    pub long_rate: f64,
}

impl FundingArbOpportunity {
    /// Serialize to JSON for dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "short_exchange": self.short_exchange.name(),
            "long_exchange": self.long_exchange.name(),
            "symbol": self.symbol,
            "net_rate": self.net_rate,
            "net_rate_pct": format!("{:.4}%", self.net_rate * 100.0),
            "annualized_apr": self.annualized_apr,
            "annualized_apr_pct": format!("{:.2}%", self.annualized_apr * 100.0),
            "net_profit_bps": self.net_profit_bps,
            "is_actionable": self.is_actionable,
            "short_rate_pct": format!("{:.4}%", self.short_rate * 100.0),
            "long_rate_pct": format!("{:.4}%", self.long_rate * 100.0),
        })
    }
}

// ---------------------------------------------------------------------------
// Cross-Exchange Funding Arbitrage Monitor
// ---------------------------------------------------------------------------

/// Cross-exchange funding rate arbitrage monitor.
pub struct CrossExchangeFundingArb {
    /// Latest funding rates per exchange per symbol.
    rates: HashMap<(ExchangeId, String), FundingRateData>,
    /// Minimum net rate spread to consider actionable (default: 0.005% = 0.5bps).
    min_net_rate: f64,
    /// Minimum annualized APR to consider actionable (default: 10%).
    min_annualized_apr: f64,
}

impl CrossExchangeFundingArb {
    /// Create a new funding arb monitor.
    ///
    /// # Arguments
    /// * `min_net_rate` - Minimum net rate spread (e.g., 0.00005 = 0.005%)
    /// * `min_annualized_apr` - Minimum annualized APR (e.g., 0.10 = 10%)
    pub fn new(min_net_rate: f64, min_annualized_apr: f64) -> Self {
        Self {
            rates: HashMap::new(),
            min_net_rate,
            min_annualized_apr,
        }
    }

    /// Create with default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(0.00005, 0.10)
    }

    /// Update funding rate data for a specific exchange and symbol.
    pub fn update_rate(&mut self, data: FundingRateData) {
        let key = (data.exchange, data.symbol.clone());
        self.rates.insert(key, data);
    }

    /// Get the current funding rate for an exchange and symbol.
    pub fn get_rate(&self, exchange: ExchangeId, symbol: &str) -> Option<&FundingRateData> {
        self.rates.get(&(exchange, symbol.to_string()))
    }

    /// Get all known symbols.
    pub fn known_symbols(&self) -> Vec<String> {
        let mut symbols: Vec<String> = self.rates.keys()
            .map(|(_, sym)| sym.clone())
            .collect();
        symbols.sort();
        symbols.dedup();
        symbols
    }

    /// Scan all known symbols for cross-exchange funding arbitrage opportunities.
    /// Returns a list of actionable opportunities sorted by net_profit_bps descending.
    pub fn scan_opportunities(&self) -> Vec<FundingArbOpportunity> {
        let mut opportunities = Vec::new();
        let symbols = self.known_symbols();

        for symbol in symbols {
            if let Some(opp) = self.check_symbol(&symbol) {
                opportunities.push(opp);
            }
        }

        // Sort by net profit descending
        opportunities.sort_by(|a, b| {
            b.net_profit_bps.partial_cmp(&a.net_profit_bps).unwrap_or(std::cmp::Ordering::Equal)
        });

        opportunities
    }

    /// Check if a specific symbol has an actionable opportunity.
    pub fn check_symbol(&self, symbol: &str) -> Option<FundingArbOpportunity> {
        // Gather rates from all exchanges for this symbol
        let mut rates: Vec<&FundingRateData> = Vec::new();
        
        for exchange in ExchangeId::all() {
            if let Some(rate) = self.get_rate(exchange, symbol) {
                rates.push(rate);
            }
        }

        if rates.len() < 2 {
            return None;  // Need at least 2 exchanges
        }

        // Find the highest and lowest funding rates
        let mut highest: Option<&FundingRateData> = None;
        let mut lowest: Option<&FundingRateData> = None;

        for rate in &rates {
            if highest.is_none() || rate.rate > highest.unwrap().rate {
                highest = Some(rate);
            }
            if lowest.is_none() || rate.rate < lowest.unwrap().rate {
                lowest = Some(rate);
            }
        }

        let highest = highest?;
        let lowest = lowest?;

        // Skip if same exchange (shouldn't happen but safety check)
        if highest.exchange == lowest.exchange {
            return None;
        }

        // Calculate net rate (short high, long low)
        let net_rate = highest.rate - lowest.rate;

        // Calculate annualized APR
        // Funding is typically every 8 hours = 3x per day = 1095x per year
        let funding_periods_per_year = 1095.0;
        let annualized_apr = net_rate * funding_periods_per_year;

        // Calculate net profit after fees
        // Entry: pay taker fee on both exchanges
        // Exit: pay taker fee on both exchanges
        // Total round-trip fee cost = 2 * (fee_short + fee_long)
        let short_fee_bps = highest.exchange.taker_fee_bps() as f64;
        let long_fee_bps = lowest.exchange.taker_fee_bps() as f64;
        let total_fee_bps = 2.0 * (short_fee_bps + long_fee_bps);

        // Net profit = net_rate in bps - fees (amortized over holding period)
        // Assuming 1 funding period holding
        let net_rate_bps = net_rate * 10000.0;
        let net_profit_bps = net_rate_bps - (total_fee_bps / 3.0);  // Assume 3 funding periods to break even on fees

        // Check if actionable
        let is_actionable = net_rate >= self.min_net_rate 
            && annualized_apr >= self.min_annualized_apr
            && net_profit_bps > 0.0;

        Some(FundingArbOpportunity {
            short_exchange: highest.exchange,
            long_exchange: lowest.exchange,
            symbol: symbol.to_string(),
            net_rate,
            annualized_apr,
            net_profit_bps,
            is_actionable,
            short_rate: highest.rate,
            long_rate: lowest.rate,
        })
    }

    /// Serialize all opportunities to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let opportunities = self.scan_opportunities();
        serde_json::json!({
            "opportunities": opportunities.iter().map(|o| o.to_json()).collect::<Vec<_>>(),
            "count": opportunities.len(),
            "actionable_count": opportunities.iter().filter(|o| o.is_actionable).count(),
        })
    }

    /// Fetch funding rates from Binance Futures REST API.
    /// Endpoint: GET /fapi/v1/premiumIndex?symbol={symbol}
    pub async fn fetch_binance_funding_rate(
        client: &Client,
        symbol: &str,
        testnet: bool,
    ) -> Option<FundingRateData> {
        let base_url = if testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };

        // Normalize symbol: "BTC_USDT" -> "BTCUSDT"
        let binance_symbol = symbol.replace('_', "").to_uppercase();
        let url = format!("{}/fapi/v1/premiumIndex?symbol={}", base_url, binance_symbol);

        let response = match client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!("[funding-arb] Binance request failed: {}", e);
                return None;
            }
        };

        let json: serde_json::Value = match response.json().await {
            Ok(j) => j,
            Err(e) => {
                warn!("[funding-arb] Binance parse failed: {}", e);
                return None;
            }
        };

        let rate = json.get("lastFundingRate")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())?;

        let next_funding_ts = json.get("nextFundingTime")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) / 1000;  // Convert ms to seconds

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        debug!("[funding-arb] Binance {} rate: {:.6}%", symbol, rate * 100.0);

        Some(FundingRateData {
            exchange: ExchangeId::Binance,
            symbol: symbol.to_string(),
            rate,
            next_funding_ts,
            updated_ns: now_ns,
        })
    }

    /// Fetch funding rates from Bybit v5 REST API.
    /// Endpoint: GET /v5/market/tickers?category=linear&symbol={symbol}
    pub async fn fetch_bybit_funding_rate(
        client: &Client,
        symbol: &str,
        testnet: bool,
    ) -> Option<FundingRateData> {
        let base_url = if testnet {
            "https://api-testnet.bybit.com"
        } else {
            "https://api.bybit.com"
        };

        // Normalize symbol: "BTC_USDT" -> "BTCUSDT"
        let bybit_symbol = symbol.replace('_', "").to_uppercase();
        let url = format!("{}/v5/market/tickers?category=linear&symbol={}", base_url, bybit_symbol);

        let response = match client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!("[funding-arb] Bybit request failed: {}", e);
                return None;
            }
        };

        let json: serde_json::Value = match response.json().await {
            Ok(j) => j,
            Err(e) => {
                warn!("[funding-arb] Bybit parse failed: {}", e);
                return None;
            }
        };

        let result = json.get("result")?.get("list")?.as_array()?.first()?;

        let rate = result.get("fundingRate")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())?;

        let next_funding_ts = result.get("nextFundingTime")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0) / 1000;

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        debug!("[funding-arb] Bybit {} rate: {:.6}%", symbol, rate * 100.0);

        Some(FundingRateData {
            exchange: ExchangeId::Bybit,
            symbol: symbol.to_string(),
            rate,
            next_funding_ts,
            updated_ns: now_ns,
        })
    }

    /// Fetch funding rates from Gate.io REST API.
    /// Endpoint: GET /api/v4/futures/usdt/contracts/{contract}
    pub async fn fetch_gateio_funding_rate(
        client: &Client,
        symbol: &str,
        testnet: bool,
    ) -> Option<FundingRateData> {
        let base_url = if testnet {
            "https://api-testnet.gateapi.io/api/v4"
        } else {
            "https://api.gateio.ws/api/v4"
        };

        // Gate.io uses underscore format: "BTC_USDT"
        let url = format!("{}/futures/usdt/contracts/{}", base_url, symbol);

        let response = match client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!("[funding-arb] Gate.io request failed: {}", e);
                return None;
            }
        };

        let json: serde_json::Value = match response.json().await {
            Ok(j) => j,
            Err(e) => {
                warn!("[funding-arb] Gate.io parse failed: {}", e);
                return None;
            }
        };

        let rate = json.get("funding_rate")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())?;

        let next_funding_ts = json.get("funding_next_apply")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        debug!("[funding-arb] Gate.io {} rate: {:.6}%", symbol, rate * 100.0);

        Some(FundingRateData {
            exchange: ExchangeId::GateIo,
            symbol: symbol.to_string(),
            rate,
            next_funding_ts,
            updated_ns: now_ns,
        })
    }

    /// Fetch funding rates from all exchanges for a symbol.
    pub async fn fetch_all_rates(
        &mut self,
        client: &Client,
        symbol: &str,
        gateio_testnet: bool,
        binance_testnet: bool,
        bybit_testnet: bool,
    ) {
        // Fetch in parallel
        let (gateio, binance, bybit) = tokio::join!(
            Self::fetch_gateio_funding_rate(client, symbol, gateio_testnet),
            Self::fetch_binance_funding_rate(client, symbol, binance_testnet),
            Self::fetch_bybit_funding_rate(client, symbol, bybit_testnet),
        );

        if let Some(data) = gateio {
            self.update_rate(data);
        }
        if let Some(data) = binance {
            self.update_rate(data);
        }
        if let Some(data) = bybit {
            self.update_rate(data);
        }
    }
}

impl Default for CrossExchangeFundingArb {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    #[test]
    fn test_funding_arb_detection() {
        let mut arb = CrossExchangeFundingArb::new(0.0001, 0.05);

        // Gate.io has high funding (shorts get paid)
        arb.update_rate(FundingRateData {
            exchange: ExchangeId::GateIo,
            symbol: "BTC_USDT".to_string(),
            rate: 0.001,  // 0.1%
            next_funding_ts: 0,
            updated_ns: now_ns(),
        });

        // Binance has low funding (longs pay less)
        arb.update_rate(FundingRateData {
            exchange: ExchangeId::Binance,
            symbol: "BTC_USDT".to_string(),
            rate: 0.0001,  // 0.01%
            next_funding_ts: 0,
            updated_ns: now_ns(),
        });

        let opportunities = arb.scan_opportunities();
        assert_eq!(opportunities.len(), 1);
        
        let opp = &opportunities[0];
        assert_eq!(opp.short_exchange, ExchangeId::GateIo);
        assert_eq!(opp.long_exchange, ExchangeId::Binance);
        assert!(opp.net_rate > 0.0);
        assert!(opp.annualized_apr > 0.0);
    }

    #[test]
    fn test_no_opportunity_same_rates() {
        let mut arb = CrossExchangeFundingArb::new(0.0001, 0.05);

        // Same rates on both exchanges
        arb.update_rate(FundingRateData {
            exchange: ExchangeId::GateIo,
            symbol: "BTC_USDT".to_string(),
            rate: 0.0001,
            next_funding_ts: 0,
            updated_ns: now_ns(),
        });

        arb.update_rate(FundingRateData {
            exchange: ExchangeId::Binance,
            symbol: "BTC_USDT".to_string(),
            rate: 0.0001,
            next_funding_ts: 0,
            updated_ns: now_ns(),
        });

        let opportunities = arb.scan_opportunities();
        // Opportunity exists but not actionable (net rate = 0)
        assert!(!opportunities.iter().any(|o| o.is_actionable));
    }
}
