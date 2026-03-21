//! Funding Rate Arbitrage Strategy — Upgrade 1.
//!
//! Monitors Gate.io futures funding rates and generates signals when rates are
//! extreme (>0.05% per period). Shorts collect funding when rate is positive,
//! longs collect funding when rate is negative.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};
use hmac::{Hmac, Mac};
use sha2::Sha512;

type HmacSha512 = Hmac<Sha512>;

#[derive(Debug, Clone)]
pub struct FundingRateInfo {
    pub rate: f64,
    pub next_funding_time: i64,
    pub annualized_pct: f64,
    pub fetched_at: Instant,
}

impl FundingRateInfo {
    pub fn new(rate: f64, next_funding_time: i64) -> Self {
        let annualized_pct = rate * 365.0 * 3.0 * 100.0; // 3 times per day * 365 days * 100 for percentage
        Self {
            rate,
            next_funding_time,
            annualized_pct,
            fetched_at: Instant::now(),
        }
    }

    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.fetched_at.elapsed() > max_age
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingSignal {
    pub symbol: String,
    pub is_short: bool, // true for SHORT (when rate > 0), false for LONG (when rate < 0)
    pub rate_pct: f64,
    pub annualized_pct: f64,
    pub confidence: f64, // 0.5 to 1.0 based on rate magnitude
    pub size_pct: f64, // 1-2% of portfolio
}

impl FundingSignal {
    pub fn new(symbol: String, rate: f64, annualized_pct: f64) -> Self {
        let is_short = rate > 0.0;
        let rate_pct = rate * 100.0;
        let abs_rate = rate.abs();
        
        // Scale confidence from 0.5 to 1.0 based on rate magnitude
        // 0.05% -> 0.5 confidence, 0.5% -> 1.0 confidence
        let confidence = (0.5 + (abs_rate - 0.0005) / 0.0045 * 0.5).clamp(0.5, 1.0);
        
        // Size scales with confidence: 1% to 2% of portfolio
        let size_pct = 1.0 + confidence;
        
        Self {
            symbol,
            is_short,
            rate_pct,
            annualized_pct,
            confidence,
            size_pct,
        }
    }

    pub fn direction(&self) -> &str {
        if self.is_short { "SHORT" } else { "LONG" }
    }
}

#[derive(Deserialize)]
struct GateIoFundingRateResponse {
    t: i64, // funding time
    r: String, // funding rate as string
}

pub struct FundingRateMonitor {
    rates: HashMap<String, FundingRateInfo>,
    api_key: String,
    api_secret: Vec<u8>,
    testnet: bool,
    last_fetch: Instant,
    fetch_interval: Duration,
    rest_client: reqwest::Client,
}

impl FundingRateMonitor {
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let rest_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            rates: HashMap::new(),
            api_key,
            api_secret: api_secret.into_bytes(),
            testnet,
            last_fetch: Instant::now() - Duration::from_secs(3600), // Force initial fetch
            fetch_interval: Duration::from_secs(300), // 5 minutes
            rest_client,
        }
    }

    pub fn should_fetch(&self) -> bool {
        self.last_fetch.elapsed() >= self.fetch_interval
    }

    fn get_base_url(&self) -> &str {
        // Funding rates are public data — always use mainnet.
        // Gate.io testnet infrastructure is unreliable (HTTP 502).
        "https://api.gateio.ws"
    }

    fn generate_signature(&self, method: &str, path: &str, query: &str, body: &str, timestamp: i64) -> String {
        use sha2::Digest;
        // Gate.io v4 signature: HMAC-SHA512 of "METHOD\nPATH\nQUERY\nSHA512(BODY)\nTIMESTAMP"
        let body_hash = hex::encode(sha2::Sha512::digest(body.as_bytes()));
        let payload = format!("{}\n{}\n{}\n{}\n{}", method, path, query, body_hash, timestamp);
        let mut mac = HmacSha512::new_from_slice(&self.api_secret)
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    pub async fn fetch_rates(&mut self, symbols: &[String]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.should_fetch() {
            return Ok(());
        }

        info!("Fetching funding rates for {} symbols", symbols.len());
        
        for symbol in symbols {
            match self.fetch_single_rate(symbol).await {
                Ok(rate_info) => {
                    debug!("Fetched funding rate for {}: {:.6}%", symbol, rate_info.rate * 100.0);
                    self.rates.insert(symbol.clone(), rate_info);
                }
                Err(e) => {
                    warn!("Failed to fetch funding rate for {}: {}", symbol, e);
                }
            }
            
            // Small delay between requests to avoid rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        self.last_fetch = Instant::now();
        info!("Funding rate fetch completed. Total symbols: {}", self.rates.len());
        
        Ok(())
    }

    async fn fetch_single_rate(&self, symbol: &str) -> Result<FundingRateInfo, Box<dyn std::error::Error + Send + Sync>> {
        let base_url = self.get_base_url();
        let uri = "/api/v4/futures/usdt/funding_rate";
        let query = format!("contract={}&limit=1", symbol);
        let url = format!("{}{}?{}", base_url, uri, query);
        
        // NOTE: /api/v4/futures/usdt/funding_rate is a PUBLIC endpoint.
        // Sending authentication headers to the testnet proxy causes
        // INVALID_KEY / HTTP 502 Bad Gateway errors. Use a plain GET.
        let response = self.rest_client
            .get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("HTTP {} for {}: {}", status, symbol, text).into());
        }

        let rates: Vec<GateIoFundingRateResponse> = response.json().await?;
        
        if rates.is_empty() {
            return Err(format!("No funding rate data for symbol: {}", symbol).into());
        }

        let rate_data = &rates[0];
        let rate = rate_data.r.parse::<f64>()?;
        
        Ok(FundingRateInfo::new(rate, rate_data.t))
    }

    pub fn check_funding_opportunity(&self, symbol: &str) -> Option<FundingSignal> {
        let rate_info = self.rates.get(symbol)?;
        
        // Check if data is stale (older than 10 minutes)
        if rate_info.is_stale(Duration::from_secs(600)) {
            debug!("Funding rate data for {} is stale", symbol);
            return None;
        }

        let abs_rate = rate_info.rate.abs();
        
        // Only signal if absolute funding rate > 0.05% (0.0005)
        if abs_rate > 0.0005 {
            let signal = FundingSignal::new(
                symbol.to_string(),
                rate_info.rate,
                rate_info.annualized_pct,
            );
            
            info!(
                "Funding opportunity detected: {} {} at {:.4}% (annualized: {:.2}%), confidence: {:.2}, size: {:.1}%",
                symbol,
                signal.direction(),
                signal.rate_pct,
                signal.annualized_pct,
                signal.confidence,
                signal.size_pct
            );
            
            Some(signal)
        } else {
            None
        }
    }

    pub fn get_funding_rate(&self, symbol: &str) -> Option<&FundingRateInfo> {
        self.rates.get(symbol)
    }

    pub fn get_all_rates(&self) -> &HashMap<String, FundingRateInfo> {
        &self.rates
    }

    pub fn clear_stale_rates(&mut self, max_age: Duration) {
        let initial_count = self.rates.len();
        self.rates.retain(|symbol, rate_info| {
            if rate_info.is_stale(max_age) {
                debug!("Removing stale funding rate data for {}", symbol);
                false
            } else {
                true
            }
        });
        
        let removed_count = initial_count - self.rates.len();
        if removed_count > 0 {
            info!("Cleared {} stale funding rate entries", removed_count);
        }
    }

    pub fn check_all_opportunities(&self) -> Vec<FundingSignal> {
        let mut signals = Vec::new();
        
        for symbol in self.rates.keys() {
            if let Some(signal) = self.check_funding_opportunity(symbol) {
                signals.push(signal);
            }
        }
        
        // Sort by confidence (highest first)
        signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        if !signals.is_empty() {
            info!("Found {} funding opportunities", signals.len());
        }
        
        signals
    }

    pub fn stats(&self) -> (usize, usize) {
        let total_rates = self.rates.len();
        let stale_rates = self.rates.values()
            .filter(|rate| rate.is_stale(Duration::from_secs(600)))
            .count();
        
        (total_rates, stale_rates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_rate_info() {
        let rate = 0.001; // 0.1%
        let next_funding = 1640995200; // timestamp
        let info = FundingRateInfo::new(rate, next_funding);
        
        assert_eq!(info.rate, 0.001);
        assert_eq!(info.next_funding_time, next_funding);
        assert!((info.annualized_pct - 109.5).abs() < 0.1); // 0.1% * 365 * 3 ≈ 109.5%
    }

    #[test]
    fn test_funding_signal_positive_rate() {
        let signal = FundingSignal::new("BTC_USDT".to_string(), 0.002, 219.0);
        
        assert_eq!(signal.symbol, "BTC_USDT");
        assert!(signal.is_short); // Positive rate -> SHORT
        assert_eq!(signal.direction(), "SHORT");
        assert_eq!(signal.rate_pct, 0.2); // 0.002 * 100
        assert_eq!(signal.annualized_pct, 219.0);
        assert!(signal.confidence > 0.5 && signal.confidence <= 1.0);
        assert!(signal.size_pct >= 1.0 && signal.size_pct <= 2.0);
    }

    #[test]
    fn test_funding_signal_negative_rate() {
        let signal = FundingSignal::new("ETH_USDT".to_string(), -0.001, -109.5);
        
        assert_eq!(signal.symbol, "ETH_USDT");
        assert!(!signal.is_short); // Negative rate -> LONG
        assert_eq!(signal.direction(), "LONG");
        assert_eq!(signal.rate_pct, -0.1); // -0.001 * 100
    }

    #[test]
    fn test_funding_monitor_opportunity_detection() {
        let mut monitor = FundingRateMonitor::new(
            "test_key".to_string(),
            "test_secret".to_string(),
            true,
        );

        // Add a rate that should trigger opportunity (> 0.05%)
        let high_rate = FundingRateInfo::new(0.001, 1640995200); // 0.1%
        monitor.rates.insert("BTC_USDT".to_string(), high_rate);

        // Add a rate that should NOT trigger opportunity (< 0.05%)
        let low_rate = FundingRateInfo::new(0.0003, 1640995200); // 0.03%
        monitor.rates.insert("ETH_USDT".to_string(), low_rate);

        let btc_signal = monitor.check_funding_opportunity("BTC_USDT");
        let eth_signal = monitor.check_funding_opportunity("ETH_USDT");

        assert!(btc_signal.is_some());
        assert!(eth_signal.is_none());

        if let Some(signal) = btc_signal {
            assert_eq!(signal.symbol, "BTC_USDT");
            assert!(signal.is_short);
        }
    }

    #[test]
    fn test_stale_rate_detection() {
        let old_instant = Instant::now() - Duration::from_secs(700); // 11+ minutes ago
        let mut rate_info = FundingRateInfo::new(0.001, 1640995200);
        rate_info.fetched_at = old_instant;

        assert!(rate_info.is_stale(Duration::from_secs(600))); // 10 minutes max age
        assert!(!rate_info.is_stale(Duration::from_secs(800))); // 13+ minutes max age
    }
}