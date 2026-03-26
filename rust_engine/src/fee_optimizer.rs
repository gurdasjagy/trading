//! Fee Tier and Maker Rebate Optimization.
//!
//! Tracks current fee tier based on 30-day volume and optimizes order
//! placement to maximize maker rebates.
//!
//! # Overview
//!
//! This module provides:
//! - Fee tier tracking based on 30-day trading volume
//! - Maker/taker fee calculation for different VIP levels
//! - Recommendations for order types (limit vs market)
//! - Break-even calculation for profitable trades
//!
//! # Exchange Support
//!
//! Currently implements Gate.io VIP tier structure.
//! Can be extended to support other exchanges.

use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Fee tier definition.
#[derive(Debug, Clone)]
pub struct FeeTier {
    /// Tier name (e.g., "VIP0", "VIP5").
    pub tier_name: String,
    /// 30-day volume threshold in USDT.
    pub volume_threshold_usdt: f64,
    /// Maker fee in basis points (negative = rebate).
    pub maker_fee_bps: f64,
    /// Taker fee in basis points.
    pub taker_fee_bps: f64,
}

/// Fee optimizer for a single exchange.
pub struct FeeOptimizer {
    /// Current 30-day volume in USDT.
    volume_30d: f64,
    /// Fee tiers for the exchange.
    tiers: Vec<FeeTier>,
    /// Current tier index.
    current_tier: usize,
    /// Volume needed for next tier.
    volume_to_next_tier: f64,
    /// Daily volume per symbol.
    daily_volume: HashMap<u16, f64>,
    /// Total fees paid (USDT).
    total_fees_paid: f64,
    /// Total rebates earned (USDT).
    total_rebates_earned: f64,
    /// Last 30-day volume update timestamp.
    last_volume_update_ns: u64,
}

impl FeeOptimizer {
    /// Create a new fee optimizer for Gate.io.
    pub fn new_gateio() -> Self {
        // Gate.io VIP tiers (as of 2024)
        let tiers = vec![
            FeeTier {
                tier_name: "VIP0".to_string(),
                volume_threshold_usdt: 0.0,
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
            },
            FeeTier {
                tier_name: "VIP1".to_string(),
                volume_threshold_usdt: 100_000.0,
                maker_fee_bps: 1.6,
                taker_fee_bps: 4.5,
            },
            FeeTier {
                tier_name: "VIP2".to_string(),
                volume_threshold_usdt: 500_000.0,
                maker_fee_bps: 1.4,
                taker_fee_bps: 4.0,
            },
            FeeTier {
                tier_name: "VIP3".to_string(),
                volume_threshold_usdt: 1_000_000.0,
                maker_fee_bps: 1.2,
                taker_fee_bps: 3.5,
            },
            FeeTier {
                tier_name: "VIP4".to_string(),
                volume_threshold_usdt: 2_500_000.0,
                maker_fee_bps: 1.0,
                taker_fee_bps: 3.0,
            },
            FeeTier {
                tier_name: "VIP5".to_string(),
                volume_threshold_usdt: 5_000_000.0,
                maker_fee_bps: 0.8,
                taker_fee_bps: 2.5,
            },
            FeeTier {
                tier_name: "VIP6".to_string(),
                volume_threshold_usdt: 10_000_000.0,
                maker_fee_bps: 0.6,
                taker_fee_bps: 2.0,
            },
            FeeTier {
                tier_name: "VIP7".to_string(),
                volume_threshold_usdt: 25_000_000.0,
                maker_fee_bps: 0.4,
                taker_fee_bps: 1.8,
            },
            FeeTier {
                tier_name: "VIP8".to_string(),
                volume_threshold_usdt: 50_000_000.0,
                maker_fee_bps: 0.2,
                taker_fee_bps: 1.6,
            },
            FeeTier {
                tier_name: "VIP9".to_string(),
                volume_threshold_usdt: 100_000_000.0,
                maker_fee_bps: 0.0, // Zero maker fee
                taker_fee_bps: 1.4,
            },
            FeeTier {
                tier_name: "VIP10".to_string(),
                volume_threshold_usdt: 200_000_000.0,
                maker_fee_bps: -1.0, // Maker rebate!
                taker_fee_bps: 1.2,
            },
        ];
        
        Self {
            volume_30d: 0.0,
            tiers,
            current_tier: 0,
            volume_to_next_tier: 100_000.0,
            daily_volume: HashMap::new(),
            total_fees_paid: 0.0,
            total_rebates_earned: 0.0,
            last_volume_update_ns: 0,
        }
    }
    
    /// Create a fee optimizer for Binance.
    pub fn new_binance() -> Self {
        let tiers = vec![
            FeeTier {
                tier_name: "Regular".to_string(),
                volume_threshold_usdt: 0.0,
                maker_fee_bps: 2.0,
                taker_fee_bps: 4.0,
            },
            FeeTier {
                tier_name: "VIP1".to_string(),
                volume_threshold_usdt: 250_000.0,
                maker_fee_bps: 1.6,
                taker_fee_bps: 4.0,
            },
            FeeTier {
                tier_name: "VIP2".to_string(),
                volume_threshold_usdt: 2_500_000.0,
                maker_fee_bps: 1.4,
                taker_fee_bps: 3.5,
            },
            FeeTier {
                tier_name: "VIP3".to_string(),
                volume_threshold_usdt: 5_000_000.0,
                maker_fee_bps: 1.2,
                taker_fee_bps: 3.2,
            },
            FeeTier {
                tier_name: "VIP4".to_string(),
                volume_threshold_usdt: 10_000_000.0,
                maker_fee_bps: 1.0,
                taker_fee_bps: 3.0,
            },
            FeeTier {
                tier_name: "VIP5".to_string(),
                volume_threshold_usdt: 25_000_000.0,
                maker_fee_bps: 0.8,
                taker_fee_bps: 2.7,
            },
            FeeTier {
                tier_name: "VIP6".to_string(),
                volume_threshold_usdt: 100_000_000.0,
                maker_fee_bps: 0.6,
                taker_fee_bps: 2.5,
            },
            FeeTier {
                tier_name: "VIP7".to_string(),
                volume_threshold_usdt: 250_000_000.0,
                maker_fee_bps: 0.4,
                taker_fee_bps: 2.2,
            },
            FeeTier {
                tier_name: "VIP8".to_string(),
                volume_threshold_usdt: 500_000_000.0,
                maker_fee_bps: 0.2,
                taker_fee_bps: 2.0,
            },
            FeeTier {
                tier_name: "VIP9".to_string(),
                volume_threshold_usdt: 1_000_000_000.0,
                maker_fee_bps: 0.0,
                taker_fee_bps: 1.7,
            },
        ];
        
        Self {
            volume_30d: 0.0,
            tiers,
            current_tier: 0,
            volume_to_next_tier: 250_000.0,
            daily_volume: HashMap::new(),
            total_fees_paid: 0.0,
            total_rebates_earned: 0.0,
            last_volume_update_ns: 0,
        }
    }
    
    /// Create a fee optimizer for Bybit perpetual futures.
    /// Bybit VIP fee tiers for USDT Perpetual & Expiry contracts (as of 2026).
    /// Source: https://www.bybit.com/en/help-center/article/Trading-Fee-Structure
    pub fn new_bybit() -> Self {
        let tiers = vec![
            FeeTier {
                tier_name: "VIP0".to_string(),
                volume_threshold_usdt: 0.0,
                maker_fee_bps: 2.0,   // 0.0200%
                taker_fee_bps: 5.5,   // 0.0550%
            },
            FeeTier {
                tier_name: "VIP1".to_string(),
                volume_threshold_usdt: 1_000_000.0,
                maker_fee_bps: 1.8,   // 0.0180%
                taker_fee_bps: 4.0,   // 0.0400%
            },
            FeeTier {
                tier_name: "VIP2".to_string(),
                volume_threshold_usdt: 5_000_000.0,
                maker_fee_bps: 1.6,   // 0.0160%
                taker_fee_bps: 3.75,  // 0.0375%
            },
            FeeTier {
                tier_name: "VIP3".to_string(),
                volume_threshold_usdt: 10_000_000.0,
                maker_fee_bps: 1.4,   // 0.0140%
                taker_fee_bps: 3.5,   // 0.0350%
            },
            FeeTier {
                tier_name: "VIP4".to_string(),
                volume_threshold_usdt: 25_000_000.0,
                maker_fee_bps: 1.0,   // 0.0100%
                taker_fee_bps: 3.2,   // 0.0320%
            },
            FeeTier {
                tier_name: "VIP5".to_string(),
                volume_threshold_usdt: 50_000_000.0,
                maker_fee_bps: 0.5,   // 0.0050%
                taker_fee_bps: 3.0,   // 0.0300%
            },
            FeeTier {
                tier_name: "VIP Supreme".to_string(),
                volume_threshold_usdt: 100_000_000.0,
                maker_fee_bps: 0.0,   // 0.0000% (zero maker)
                taker_fee_bps: 3.0,   // 0.0300%
            },
        ];

        Self {
            volume_30d: 0.0,
            tiers,
            current_tier: 0,
            volume_to_next_tier: 1_000_000.0,
            daily_volume: HashMap::new(),
            total_fees_paid: 0.0,
            total_rebates_earned: 0.0,
            last_volume_update_ns: 0,
        }
    }

    /// Fetch the real-time fee rate from Bybit API and return (maker_bps, taker_bps).
    /// Uses GET /v5/account/fee-rate?category=linear&symbol={symbol}.
    /// Falls back to VIP0 defaults if the API call fails.
    pub async fn fetch_bybit_fee_rate(
        client: &reqwest::Client,
        api_key: &str,
        api_secret: &[u8],
        symbol: &str,
        testnet: bool,
    ) -> (f64, f64) {
        let base = if testnet {
            "https://api-demo.bybit.com"
        } else {
            "https://api.bybit.com"
        };
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let recv_window = 5000i64;
        let params = format!("category=linear&symbol={}", symbol);
        let sign_input = format!("{}{}{}{}", timestamp, api_key, recv_window, params);
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(api_secret)
            .expect("HMAC can take key of any size");
        mac.update(sign_input.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let url = format!("{}/v5/account/fee-rate?{}", base, params);
        match client.get(&url)
            .header("X-BAPI-API-KEY", api_key)
            .header("X-BAPI-SIGN", &signature)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-RECV-WINDOW", recv_window.to_string())
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    if let Some(list) = body["result"]["list"].as_array() {
                        if let Some(item) = list.first() {
                            let maker = item["makerFeeRate"].as_str()
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.0002) * 10000.0; // to bps
                            let taker = item["takerFeeRate"].as_str()
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.00055) * 10000.0; // to bps
                            return (maker, taker);
                        }
                    }
                }
                (2.0, 5.5) // VIP0 defaults
            }
            Err(_) => (2.0, 5.5),
        }
    }

    /// Update 30-day volume from exchange API.
    pub fn update_volume(&mut self, volume: f64) {
        self.volume_30d = volume;
        self.recalculate_tier();
        self.last_volume_update_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        info!(
            "Volume updated: ${:.2} -> Tier {} ({})",
            volume,
            self.current_tier,
            self.tiers[self.current_tier].tier_name
        );
    }
    
    /// Record a trade for volume tracking.
    pub fn record_trade(&mut self, symbol_id: u16, volume_usdt: f64, is_maker: bool) {
        *self.daily_volume.entry(symbol_id).or_insert(0.0) += volume_usdt;
        
        let fee_bps = if is_maker { self.maker_fee_bps() } else { self.taker_fee_bps() };
        let fee_usdt = volume_usdt * fee_bps / 10000.0;
        
        if fee_usdt >= 0.0 {
            self.total_fees_paid += fee_usdt;
        } else {
            self.total_rebates_earned += fee_usdt.abs();
        }
        
        debug!(
            "Trade recorded: ${:.2} volume, ${:.4} {} ({})",
            volume_usdt,
            fee_usdt.abs(),
            if fee_usdt >= 0.0 { "fee" } else { "rebate" },
            if is_maker { "maker" } else { "taker" }
        );
    }
    
    fn recalculate_tier(&mut self) {
        for (i, tier) in self.tiers.iter().enumerate().rev() {
            if self.volume_30d >= tier.volume_threshold_usdt {
                self.current_tier = i;
                if i < self.tiers.len() - 1 {
                    self.volume_to_next_tier = 
                        self.tiers[i + 1].volume_threshold_usdt - self.volume_30d;
                } else {
                    self.volume_to_next_tier = 0.0;
                }
                return;
            }
        }
        self.current_tier = 0;
        self.volume_to_next_tier = self.tiers[1].volume_threshold_usdt - self.volume_30d;
    }
    
    /// Get current maker fee in basis points.
    pub fn maker_fee_bps(&self) -> f64 {
        self.tiers[self.current_tier].maker_fee_bps
    }
    
    /// Get current taker fee in basis points.
    pub fn taker_fee_bps(&self) -> f64 {
        self.tiers[self.current_tier].taker_fee_bps
    }
    
    /// Get current tier name.
    pub fn tier_name(&self) -> &str {
        &self.tiers[self.current_tier].tier_name
    }
    
    /// Should we aggressively try to be maker?
    /// Returns true if maker fee is significantly lower than taker.
    pub fn prefer_maker(&self) -> bool {
        let tier = &self.tiers[self.current_tier];
        (tier.taker_fee_bps - tier.maker_fee_bps) >= 2.0
    }
    
    /// Is maker rebate available (negative maker fee)?
    pub fn has_maker_rebate(&self) -> bool {
        self.tiers[self.current_tier].maker_fee_bps < 0.0
    }
    
    /// Calculate fee-adjusted break-even for a round-trip trade.
    pub fn break_even_bps(&self, is_maker: bool) -> f64 {
        if is_maker {
            self.maker_fee_bps() * 2.0 // Entry + exit as maker
        } else {
            self.taker_fee_bps() * 2.0 // Entry + exit as taker
        }
    }
    
    /// Calculate mixed break-even (maker entry, taker exit).
    pub fn mixed_break_even_bps(&self) -> f64 {
        self.maker_fee_bps() + self.taker_fee_bps()
    }
    
    /// Get fee savings from using maker vs taker orders.
    pub fn maker_savings_bps(&self) -> f64 {
        self.taker_fee_bps() - self.maker_fee_bps()
    }
    
    /// Calculate fee for a trade.
    pub fn calculate_fee(&self, volume_usdt: f64, is_maker: bool) -> f64 {
        let fee_bps = if is_maker { self.maker_fee_bps() } else { self.taker_fee_bps() };
        volume_usdt * fee_bps / 10000.0
    }
    
    /// Get volume needed to reach next tier.
    pub fn volume_to_next_tier(&self) -> f64 {
        self.volume_to_next_tier
    }
    
    /// Get tier progression percentage to next level.
    pub fn tier_progress_pct(&self) -> f64 {
        if self.current_tier >= self.tiers.len() - 1 {
            return 100.0; // Already at max tier
        }
        
        let current_threshold = self.tiers[self.current_tier].volume_threshold_usdt;
        let next_threshold = self.tiers[self.current_tier + 1].volume_threshold_usdt;
        let range = next_threshold - current_threshold;
        
        if range > 0.0 {
            ((self.volume_30d - current_threshold) / range * 100.0).clamp(0.0, 100.0)
        } else {
            100.0
        }
    }
    
    /// Get daily volume total.
    pub fn total_daily_volume(&self) -> f64 {
        self.daily_volume.values().sum()
    }
    
    /// Reset daily volume counters.
    pub fn reset_daily_volume(&mut self) {
        self.daily_volume.clear();
    }

    /// CATEGORY 8 FIX: Auto-detect current VIP level from exchange API response.
    ///
    /// Instead of relying solely on tracked 30-day volume (which may be inaccurate
    /// after restarts), this method accepts the actual fee rates returned by the
    /// exchange and determines the VIP tier from those rates.
    ///
    /// Call this after fetching account info from the exchange.
    pub fn detect_tier_from_fees(&mut self, actual_maker_bps: f64, actual_taker_bps: f64) {
        // Find the closest matching tier based on actual fees
        let mut best_match = 0usize;
        let mut best_distance = f64::MAX;

        for (i, tier) in self.tiers.iter().enumerate() {
            let distance = (tier.maker_fee_bps - actual_maker_bps).abs()
                + (tier.taker_fee_bps - actual_taker_bps).abs();
            if distance < best_distance {
                best_distance = distance;
                best_match = i;
            }
        }

        if best_match != self.current_tier {
            let old_tier = self.tiers[self.current_tier].tier_name.clone();
            self.current_tier = best_match;
            let new_tier = &self.tiers[best_match].tier_name;
            tracing::info!(
                "[fee-optimizer] Auto-detected tier change: {} -> {} (maker={:.1}bps, taker={:.1}bps)",
                old_tier, new_tier, actual_maker_bps, actual_taker_bps
            );

            // Also update volume to reflect detected tier
            self.volume_30d = self.tiers[best_match].volume_threshold_usdt;
            if best_match + 1 < self.tiers.len() {
                self.volume_to_next_tier =
                    self.tiers[best_match + 1].volume_threshold_usdt - self.volume_30d;
            } else {
                self.volume_to_next_tier = 0.0;
            }
        }
    }

    /// CATEGORY 8 FIX: Fetch and auto-detect Gate.io fee tier from REST API.
    ///
    /// Queries GET /api/v4/futures/usdt/accounts and extracts the fee tier
    /// from the response to keep the fee optimizer in sync with the exchange.
    pub async fn auto_detect_gateio_tier(
        &mut self,
        client: &reqwest::Client,
        base_url: &str,
        api_key: &str,
        api_secret: &[u8],
    ) {
        let path = "/futures/usdt/accounts";
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let full_path = format!("/api/v4{}", path);

        // Compute Gate.io signature
        use sha2::Digest;
        use hmac::Mac;
        let body_hash = hex::encode(sha2::Sha512::digest(b""));
        let payload = format!("GET\n{}\n\n{}\n{}", full_path, body_hash, timestamp);
        let mut mac = hmac::Hmac::<sha2::Sha512>::new_from_slice(api_secret)
            .expect("HMAC key");
        hmac::Mac::update(&mut mac, payload.as_bytes());
        let signature = hex::encode(hmac::Mac::finalize(mac).into_bytes());

        let url = format!("{}{}", base_url, path);
        match client.get(&url)
            .header("KEY", api_key)
            .header("SIGN", &signature)
            .header("Timestamp", timestamp.to_string())
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    // Gate.io returns "tier" field in account info
                    if let Some(tier_num) = json.get("tier").and_then(|v| v.as_u64()) {
                        let tier_idx = (tier_num as usize).min(self.tiers.len() - 1);
                        if tier_idx != self.current_tier {
                            let old = self.tiers[self.current_tier].tier_name.clone();
                            self.current_tier = tier_idx;
                            tracing::info!(
                                "[fee-optimizer] Gate.io tier auto-detected: {} -> {} (tier={})",
                                old, self.tiers[tier_idx].tier_name, tier_num
                            );
                        }
                    }
                }
            }
            _ => {
                tracing::debug!("[fee-optimizer] Could not auto-detect Gate.io tier");
            }
        }
    }
    
    /// Get statistics.
    pub fn stats(&self) -> FeeStats {
        FeeStats {
            current_tier: self.tier_name().to_string(),
            tier_index: self.current_tier,
            volume_30d: self.volume_30d,
            volume_to_next_tier: self.volume_to_next_tier,
            tier_progress_pct: self.tier_progress_pct(),
            maker_fee_bps: self.maker_fee_bps(),
            taker_fee_bps: self.taker_fee_bps(),
            total_fees_paid: self.total_fees_paid,
            total_rebates_earned: self.total_rebates_earned,
            prefer_maker: self.prefer_maker(),
            has_maker_rebate: self.has_maker_rebate(),
        }
    }

    /// FEATURE 3: Get the best exchange for a trade based on fee optimization.
    /// Returns the exchange with lowest effective fee for the given order type.
    pub fn best_exchange_for_fee(
        optimizers: &HashMap<String, FeeOptimizer>,
        is_maker: bool,
    ) -> Option<String> {
        let mut best: Option<(String, f64)> = None;
        for (exchange, opt) in optimizers {
            let fee = if is_maker { opt.maker_fee_bps() } else { opt.taker_fee_bps() };
            match &best {
                None => best = Some((exchange.clone(), fee)),
                Some((_, best_fee)) if fee < *best_fee => {
                    best = Some((exchange.clone(), fee));
                }
                _ => {}
            }
        }
        best.map(|(name, _)| name)
    }

    /// FEATURE 3: Calculate net fee savings across all exchanges for a round trip.
    /// Compares maker entry + taker exit vs taker entry + taker exit.
    pub fn round_trip_savings_bps(&self) -> f64 {
        let maker_entry_taker_exit = self.maker_fee_bps() + self.taker_fee_bps();
        let full_taker = self.taker_fee_bps() * 2.0;
        full_taker - maker_entry_taker_exit
    }

    /// FEATURE 3: Estimate annual fee savings based on current daily volume.
    pub fn estimated_annual_savings_usdt(&self) -> f64 {
        let daily_vol = self.total_daily_volume();
        let savings_per_dollar = self.round_trip_savings_bps() / 10000.0;
        daily_vol * savings_per_dollar * 365.0
    }
}

impl Default for FeeOptimizer {
    fn default() -> Self {
        Self::new_gateio()
    }
}

/// Fee optimizer statistics.
#[derive(Debug, Clone)]
pub struct FeeStats {
    pub current_tier: String,
    pub tier_index: usize,
    pub volume_30d: f64,
    pub volume_to_next_tier: f64,
    pub tier_progress_pct: f64,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub total_fees_paid: f64,
    pub total_rebates_earned: f64,
    pub prefer_maker: bool,
    pub has_maker_rebate: bool,
}

/// Order type recommendation based on fee optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderTypeRecommendation {
    /// Use limit order (post-only) to ensure maker fee.
    LimitPostOnly,
    /// Use limit order with standard behavior.
    Limit,
    /// Use market order (urgent execution needed).
    Market,
    /// Use iceberg order for large sizes.
    Iceberg,
}

impl FeeOptimizer {
    /// Get order type recommendation based on urgency and size.
    pub fn recommend_order_type(&self, urgency: f64, size_usdt: f64) -> OrderTypeRecommendation {
        let savings = self.maker_savings_bps();
        
        // Large orders benefit from iceberg
        if size_usdt > 50_000.0 {
            return OrderTypeRecommendation::Iceberg;
        }
        
        // High urgency = market order
        if urgency > 0.8 {
            return OrderTypeRecommendation::Market;
        }
        
        // If maker savings are significant, use post-only
        if savings >= 2.0 {
            return OrderTypeRecommendation::LimitPostOnly;
        }
        
        OrderTypeRecommendation::Limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tier_calculation() {
        let mut optimizer = FeeOptimizer::new_gateio();
        
        // VIP0 by default
        assert_eq!(optimizer.current_tier, 0);
        assert_eq!(optimizer.maker_fee_bps(), 2.0);
        
        // Update to VIP3 volume
        optimizer.update_volume(1_500_000.0);
        assert_eq!(optimizer.current_tier, 3);
        assert_eq!(optimizer.maker_fee_bps(), 1.2);
        
        // Update to VIP10 volume (maker rebate)
        optimizer.update_volume(250_000_000.0);
        assert_eq!(optimizer.current_tier, 10);
        assert!(optimizer.maker_fee_bps() < 0.0);
        assert!(optimizer.has_maker_rebate());
    }
    
    #[test]
    fn test_break_even_calculation() {
        let mut optimizer = FeeOptimizer::new_gateio();
        optimizer.update_volume(0.0); // VIP0
        
        // VIP0: maker=2bps, taker=5bps
        let maker_be = optimizer.break_even_bps(true);
        let taker_be = optimizer.break_even_bps(false);
        
        assert_eq!(maker_be, 4.0); // 2 * 2 = 4 bps
        assert_eq!(taker_be, 10.0); // 2 * 5 = 10 bps
        assert_eq!(optimizer.mixed_break_even_bps(), 7.0); // 2 + 5 = 7 bps
    }
    
    #[test]
    fn test_order_recommendation() {
        let optimizer = FeeOptimizer::new_gateio();
        
        // Normal trade, low urgency
        let rec = optimizer.recommend_order_type(0.3, 5000.0);
        assert_eq!(rec, OrderTypeRecommendation::LimitPostOnly);
        
        // High urgency
        let rec = optimizer.recommend_order_type(0.9, 5000.0);
        assert_eq!(rec, OrderTypeRecommendation::Market);
        
        // Large size
        let rec = optimizer.recommend_order_type(0.3, 100_000.0);
        assert_eq!(rec, OrderTypeRecommendation::Iceberg);
    }
}
