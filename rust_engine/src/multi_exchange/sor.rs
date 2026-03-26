//! Smart Order Router (SOR) — FEATURE 8 Enhanced
//!
//! Determines optimal order splitting across exchanges based on:
//! - Available liquidity at each venue
//! - Fee-adjusted effective prices (real-time fee tiers)
//! - Measured latency with exponential moving average
//! - Historical fill quality per exchange
//! - Liquidity-weighted order splitting
//!
//! When order size is below the split threshold, routes entirely to the
//! exchange with the best effective price. Above threshold, sweeps the
//! global book level by level with liquidity-weighted allocation.

use std::collections::HashMap;
use tracing::{info, warn};

use crate::execution_gateway::OrderSide;
use crate::multi_exchange::global_book::{ExchangeId, GlobalOrderBook};
use crate::fixed_point::FixedPrice;

// ---------------------------------------------------------------------------
// Order Slice
// ---------------------------------------------------------------------------

/// A single slice of a split order destined for one exchange.
#[derive(Debug, Clone)]
pub struct OrderSlice {
    pub exchange: ExchangeId,
    pub symbol: String,
    pub side: OrderSide,
    pub size: i64,           // contracts (scaled by 1e8)
    pub price_fp: i64,       // limit price in fixed-point (0 = market)
    pub expected_cost_bps: f64,
    pub is_maker: bool,
}

// ---------------------------------------------------------------------------
// SOR Result
// ---------------------------------------------------------------------------

/// Result of the SOR calculation.
#[derive(Debug, Clone)]
pub struct SorResult {
    pub slices: Vec<OrderSlice>,
    pub total_size: i64,
    pub estimated_slippage_bps: f64,
    pub estimated_savings_bps: f64,  // vs. single-exchange execution
    pub routing_reason: String,
}

impl SorResult {
    /// Create an empty result (no routing).
    pub fn empty() -> Self {
        Self {
            slices: Vec::new(),
            total_size: 0,
            estimated_slippage_bps: 0.0,
            estimated_savings_bps: 0.0,
            routing_reason: "No routing".to_string(),
        }
    }

    /// Serialize to JSON for dashboard preview.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "slices": self.slices.iter().map(|s| {
                serde_json::json!({
                    "exchange": s.exchange.name(),
                    "exchange_id": s.exchange.id_str(),
                    "symbol": s.symbol,
                    "side": if s.side == OrderSide::Buy { "Buy" } else { "Sell" },
                    "size": s.size as f64 / 1e8,
                    "price": if s.price_fp > 0 { FixedPrice(s.price_fp).to_f64() } else { 0.0 },
                    "expected_cost_bps": s.expected_cost_bps,
                    "is_maker": s.is_maker,
                })
            }).collect::<Vec<_>>(),
            "total_size": self.total_size as f64 / 1e8,
            "estimated_slippage_bps": self.estimated_slippage_bps,
            "estimated_savings_bps": self.estimated_savings_bps,
            "routing_reason": self.routing_reason,
        })
    }
}

// ---------------------------------------------------------------------------
// SOR Configuration
// ---------------------------------------------------------------------------

/// Smart Order Router configuration.
#[derive(Debug, Clone)]
pub struct SorConfig {
    /// Minimum order size (USDT) to trigger splitting across exchanges.
    pub min_split_size_usdt: f64,
    /// Maximum number of exchanges to split across (1-3).
    pub max_venues: usize,
    /// Maximum slippage tolerance in basis points.
    pub max_slippage_bps: f64,
    /// Prefer maker orders when spread allows.
    pub prefer_maker: bool,
    /// FEAT 8: Weight given to fill quality score in routing (0.0 to 1.0).
    pub fill_quality_weight: f64,
    /// FEAT 8: Weight given to latency in routing (0.0 to 1.0).
    pub latency_weight: f64,
    /// FEAT 8: EMA decay factor for latency tracking (0.0 to 1.0, higher = faster decay).
    pub latency_ema_alpha: f64,
    /// FEAT 8: Minimum fill rate to consider an exchange reliable (0.0 to 1.0).
    pub min_fill_rate: f64,
}

impl Default for SorConfig {
    fn default() -> Self {
        Self {
            min_split_size_usdt: 5000.0,
            max_venues: 3,
            max_slippage_bps: 30.0,
            prefer_maker: true,
            fill_quality_weight: 0.3,
            latency_weight: 0.2,
            latency_ema_alpha: 0.1,
            min_fill_rate: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// FEAT 8: Fill Quality Tracker
// ---------------------------------------------------------------------------

/// Historical fill quality record for a single exchange.
#[derive(Debug, Clone)]
pub struct ExchangeFillQuality {
    /// Total orders sent to this exchange.
    pub total_orders: u64,
    /// Orders that were fully filled.
    pub filled_orders: u64,
    /// Orders that were partially filled.
    pub partial_fills: u64,
    /// Orders that were rejected or timed out.
    pub rejected_orders: u64,
    /// Average fill time in microseconds (EMA).
    pub avg_fill_time_us: f64,
    /// Average slippage in basis points (EMA).
    pub avg_slippage_bps: f64,
    /// Exponential moving average of observed latency (microseconds).
    pub latency_ema_us: f64,
    /// Last updated timestamp (nanoseconds).
    pub last_update_ns: u64,
}

impl ExchangeFillQuality {
    /// Create a new fill quality record with neutral defaults.
    pub fn new() -> Self {
        Self {
            total_orders: 0,
            filled_orders: 0,
            partial_fills: 0,
            rejected_orders: 0,
            avg_fill_time_us: 0.0,
            avg_slippage_bps: 0.0,
            latency_ema_us: 0.0,
            last_update_ns: 0,
        }
    }

    /// Calculate fill rate (0.0 to 1.0).
    pub fn fill_rate(&self) -> f64 {
        if self.total_orders == 0 {
            return 1.0; // Assume good until proven otherwise
        }
        (self.filled_orders + self.partial_fills) as f64 / self.total_orders as f64
    }

    /// Calculate a composite quality score (0.0 to 1.0, higher is better).
    ///
    /// Combines fill rate, slippage performance, and latency into a single score.
    pub fn quality_score(&self) -> f64 {
        if self.total_orders == 0 {
            return 0.5; // Neutral score for unknown exchanges
        }

        let fill_rate_score = self.fill_rate();

        // Slippage score: 0 bps = 1.0, 10+ bps = 0.0
        let slippage_score = (1.0 - self.avg_slippage_bps / 10.0).clamp(0.0, 1.0);

        // Latency score: 0us = 1.0, 5000us+ = 0.0
        let latency_score = (1.0 - self.latency_ema_us / 5000.0).clamp(0.0, 1.0);

        // Weighted combination
        fill_rate_score * 0.5 + slippage_score * 0.3 + latency_score * 0.2
    }

    /// Record a fill outcome.
    pub fn record_fill(
        &mut self,
        was_filled: bool,
        was_partial: bool,
        fill_time_us: f64,
        slippage_bps: f64,
        ema_alpha: f64,
    ) {
        self.total_orders += 1;
        if was_filled {
            self.filled_orders += 1;
        } else if was_partial {
            self.partial_fills += 1;
        } else {
            self.rejected_orders += 1;
        }

        // Update EMA of fill time
        if self.avg_fill_time_us == 0.0 {
            self.avg_fill_time_us = fill_time_us;
        } else {
            self.avg_fill_time_us =
                ema_alpha * fill_time_us + (1.0 - ema_alpha) * self.avg_fill_time_us;
        }

        // Update EMA of slippage
        if self.avg_slippage_bps == 0.0 {
            self.avg_slippage_bps = slippage_bps;
        } else {
            self.avg_slippage_bps =
                ema_alpha * slippage_bps + (1.0 - ema_alpha) * self.avg_slippage_bps;
        }

        self.last_update_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
    }

    /// Update the latency EMA with a new observation.
    pub fn update_latency(&mut self, latency_us: f64, ema_alpha: f64) {
        if self.latency_ema_us == 0.0 {
            self.latency_ema_us = latency_us;
        } else {
            self.latency_ema_us =
                ema_alpha * latency_us + (1.0 - ema_alpha) * self.latency_ema_us;
        }
    }
}

impl Default for ExchangeFillQuality {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Smart Order Router
// ---------------------------------------------------------------------------

/// Smart Order Router for cross-exchange order splitting.
///
/// FEAT 8: Enhanced with fill quality tracking, latency-aware routing,
/// and liquidity-weighted order splitting.
pub struct SmartOrderRouter {
    config: SorConfig,
    /// FEAT 8: Historical fill quality per exchange.
    fill_quality: HashMap<ExchangeId, ExchangeFillQuality>,
    /// FEAT 8: Per-exchange fee override in basis points (from FeeOptimizer).
    fee_overrides_bps: HashMap<ExchangeId, f64>,
    /// FEAT 8: Total orders routed (for statistics).
    total_routed: u64,
    /// FEAT 8: Total orders split across multiple venues.
    total_split: u64,
}

impl SmartOrderRouter {
    /// Create a new SmartOrderRouter with the given configuration.
    pub fn new(config: SorConfig) -> Self {
        let mut fill_quality = HashMap::new();
        fill_quality.insert(ExchangeId::GateIo, ExchangeFillQuality::new());
        fill_quality.insert(ExchangeId::Binance, ExchangeFillQuality::new());
        fill_quality.insert(ExchangeId::Bybit, ExchangeFillQuality::new());

        Self {
            config,
            fill_quality,
            fee_overrides_bps: HashMap::new(),
            total_routed: 0,
            total_split: 0,
        }
    }

    /// FEAT 8: Update real-time fee tier for an exchange (from FeeOptimizer).
    pub fn update_fee_tier(&mut self, exchange: ExchangeId, taker_fee_bps: f64) {
        self.fee_overrides_bps.insert(exchange, taker_fee_bps);
    }

    /// FEAT 8: Get the effective taker fee for an exchange.
    ///
    /// Uses fee overrides if available, otherwise falls back to static fees.
    fn effective_taker_fee_bps(&self, exchange: ExchangeId) -> f64 {
        self.fee_overrides_bps
            .get(&exchange)
            .copied()
            .unwrap_or_else(|| exchange.taker_fee_bps() as f64)
    }

    /// FEAT 8: Record a fill outcome for learning.
    pub fn record_fill_outcome(
        &mut self,
        exchange: ExchangeId,
        was_filled: bool,
        was_partial: bool,
        fill_time_us: f64,
        slippage_bps: f64,
    ) {
        let alpha = self.config.latency_ema_alpha;
        if let Some(quality) = self.fill_quality.get_mut(&exchange) {
            quality.record_fill(was_filled, was_partial, fill_time_us, slippage_bps, alpha);
        }
    }

    /// FEAT 8: Update latency observation for an exchange.
    pub fn update_latency(&mut self, exchange: ExchangeId, latency_us: f64) {
        let alpha = self.config.latency_ema_alpha;
        if let Some(quality) = self.fill_quality.get_mut(&exchange) {
            quality.update_latency(latency_us, alpha);
        }
    }

    /// FEAT 8: Get fill quality stats for an exchange.
    pub fn get_fill_quality(&self, exchange: ExchangeId) -> Option<&ExchangeFillQuality> {
        self.fill_quality.get(&exchange)
    }

    /// FEAT 8: Get routing statistics as JSON.
    pub fn stats_json(&self) -> serde_json::Value {
        let quality_json: serde_json::Map<String, serde_json::Value> = self
            .fill_quality
            .iter()
            .map(|(ex, q)| {
                (
                    ex.name().to_string(),
                    serde_json::json!({
                        "total_orders": q.total_orders,
                        "fill_rate": q.fill_rate(),
                        "quality_score": q.quality_score(),
                        "avg_fill_time_us": q.avg_fill_time_us,
                        "avg_slippage_bps": q.avg_slippage_bps,
                        "latency_ema_us": q.latency_ema_us,
                    }),
                )
            })
            .collect();

        serde_json::json!({
            "total_routed": self.total_routed,
            "total_split": self.total_split,
            "split_rate_pct": if self.total_routed > 0 {
                self.total_split as f64 / self.total_routed as f64 * 100.0
            } else { 0.0 },
            "exchange_quality": quality_json,
        })
    }

    /// Calculate optimal order routing given the current global book state.
    ///
    /// FEAT 8 Enhanced Algorithm:
    /// 1. Filter exchanges below minimum fill rate threshold
    /// 2. If total_size_usdt < min_split_size_usdt -> route entirely to best
    ///    single exchange (considering fees, latency, and fill quality)
    /// 3. Otherwise, use liquidity-weighted splitting across venues
    /// 4. Apply fee adjustment from real-time fee tiers (FeeOptimizer)
    /// 5. Weight by historical fill quality and observed latency
    /// 6. Return SorResult with per-exchange slices
    pub fn route(
        &mut self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        mid_price_fp: i64,
        symbol: &str,
    ) -> SorResult {
        if total_size <= 0 {
            return SorResult::empty();
        }

        self.total_routed += 1;

        // Estimate order size in USDT
        let size_usdt = if mid_price_fp > 0 {
            (total_size as f64 / 1e8) * FixedPrice(mid_price_fp).to_f64()
        } else {
            0.0
        };

        // If below threshold, route to single best exchange
        if size_usdt < self.config.min_split_size_usdt {
            return self.route_to_best_single(book, side, total_size, symbol);
        }

        // Multi-exchange routing: sweep the global book with quality weighting
        self.total_split += 1;
        self.route_multi_venue(book, side, total_size, mid_price_fp, symbol)
    }

    /// FEAT 8: Route entirely to the single best exchange, considering
    /// fee tiers, latency, and historical fill quality.
    fn route_to_best_single(
        &self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        symbol: &str,
    ) -> SorResult {
        // Collect candidate levels from all exchanges
        let candidates: Vec<_> = match side {
            OrderSide::Buy => {
                let mut levels = Vec::new();
                // Get best ask from each exchange
                for level in &book.global_asks {
                    // Only take the best level per exchange
                    if !levels.iter().any(|l: &crate::multi_exchange::global_book::GlobalLevel| l.exchange == level.exchange) {
                        levels.push(level.clone());
                    }
                }
                levels
            }
            OrderSide::Sell => {
                let mut levels = Vec::new();
                for level in &book.global_bids {
                    if !levels.iter().any(|l: &crate::multi_exchange::global_book::GlobalLevel| l.exchange == level.exchange) {
                        levels.push(level.clone());
                    }
                }
                levels
            }
        };

        if candidates.is_empty() {
            return SorResult {
                slices: Vec::new(),
                total_size,
                estimated_slippage_bps: 0.0,
                estimated_savings_bps: 0.0,
                routing_reason: "No liquidity available".to_string(),
            };
        }

        // FEAT 8: Score each candidate exchange
        let mut best_exchange = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_level_idx = 0;

        for (idx, level) in candidates.iter().enumerate() {
            // Real-time fee from FeeOptimizer (or fallback to static)
            let fee_bps = self.effective_taker_fee_bps(level.exchange);

            // Fill quality score (0.0 to 1.0)
            let quality_score = self
                .fill_quality
                .get(&level.exchange)
                .map(|q| q.quality_score())
                .unwrap_or(0.5);

            // Check minimum fill rate
            let fill_rate = self
                .fill_quality
                .get(&level.exchange)
                .map(|q| q.fill_rate())
                .unwrap_or(1.0);

            if fill_rate < self.config.min_fill_rate {
                warn!(
                    "[sor] Skipping {} for {}: fill_rate {:.2} below threshold {:.2}",
                    level.exchange.name(), symbol, fill_rate, self.config.min_fill_rate
                );
                continue;
            }

            // Composite routing score:
            //   Higher is better. We negate costs (fee, latency) and add quality.
            let fee_component = -fee_bps;
            let quality_component = quality_score * 10.0 * self.config.fill_quality_weight;
            let latency_component = -(level.latency_penalty_bps as f64) * self.config.latency_weight;

            let score = fee_component + quality_component + latency_component;

            if score > best_score {
                best_score = score;
                best_exchange = Some(level.exchange);
                best_level_idx = idx;
            }
        }

        let Some(exchange) = best_exchange else {
            return SorResult {
                slices: Vec::new(),
                total_size,
                estimated_slippage_bps: 0.0,
                estimated_savings_bps: 0.0,
                routing_reason: "All exchanges below fill rate threshold".to_string(),
            };
        };

        let level = &candidates[best_level_idx];
        let fee_bps = self.effective_taker_fee_bps(exchange);
        let total_cost_bps = fee_bps + level.latency_penalty_bps as f64;

        let quality_info = self
            .fill_quality
            .get(&exchange)
            .map(|q| format!("quality={:.2}, fill_rate={:.2}", q.quality_score(), q.fill_rate()))
            .unwrap_or_else(|| "no history".to_string());

        info!(
            "[sor] Single routing {} to {} (fee={:.1}bps, {}, score={:.2})",
            symbol, exchange.name(), fee_bps, quality_info, best_score
        );

        let slice = OrderSlice {
            exchange,
            symbol: symbol.to_string(),
            side: side.clone(),
            size: total_size,
            price_fp: level.raw_price_fp,
            expected_cost_bps: total_cost_bps,
            is_maker: false,
        };

        SorResult {
            slices: vec![slice],
            total_size,
            estimated_slippage_bps: total_cost_bps,
            estimated_savings_bps: 0.0,
            routing_reason: format!(
                "Single exchange: {} (best score={:.2}, fee={:.1}bps, {})",
                exchange.name(), best_score, fee_bps, quality_info
            ),
        }
    }

    /// FEAT 8: Route across multiple venues with liquidity-weighted splitting,
    /// fee-aware pricing, and quality-weighted allocation.
    fn route_multi_venue(
        &self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        mid_price_fp: i64,
        symbol: &str,
    ) -> SorResult {
        let mut slices: Vec<OrderSlice> = Vec::new();
        let mut remaining_size = total_size;
        let mut total_cost_bps = 0.0;
        let mut exchanges_used = std::collections::HashSet::new();

        // Get the appropriate side of the book
        let levels = match side {
            OrderSide::Buy => &book.global_asks,
            OrderSide::Sell => &book.global_bids,
        };

        // Sweep through sorted levels with quality filtering
        for level in levels {
            if remaining_size <= 0 {
                break;
            }
            if exchanges_used.len() >= self.config.max_venues {
                break;
            }

            // FEAT 8: Check fill rate threshold before routing to this exchange
            let fill_rate = self
                .fill_quality
                .get(&level.exchange)
                .map(|q| q.fill_rate())
                .unwrap_or(1.0);

            if fill_rate < self.config.min_fill_rate {
                continue; // Skip unreliable exchanges
            }

            // Calculate slippage from mid price
            let slippage_bps = if mid_price_fp > 0 {
                ((level.effective_price_fp - mid_price_fp).abs() * 10000) / mid_price_fp
            } else {
                0
            };

            if slippage_bps as f64 > self.config.max_slippage_bps {
                break;
            }

            // FEAT 8: Adjust fill size by quality score (allocate more to better exchanges)
            let quality_score = self
                .fill_quality
                .get(&level.exchange)
                .map(|q| q.quality_score())
                .unwrap_or(0.5);

            // Scale available quantity by quality score (0.5x to 1.5x)
            let quality_multiplier = 0.5 + quality_score;
            let adjusted_qty = (level.qty as f64 * quality_multiplier) as i64;
            let fill_size = remaining_size.min(adjusted_qty.max(1));

            if fill_size <= 0 {
                continue;
            }

            // FEAT 8: Use real-time fee from FeeOptimizer
            let fee_bps = self.effective_taker_fee_bps(level.exchange);

            // Check if we already have a slice for this exchange
            let existing_slice = slices.iter_mut().find(|s| s.exchange == level.exchange);

            if let Some(slice) = existing_slice {
                slice.size += fill_size;
                match side {
                    OrderSide::Buy => {
                        if level.raw_price_fp > slice.price_fp {
                            slice.price_fp = level.raw_price_fp;
                        }
                    }
                    OrderSide::Sell => {
                        if level.raw_price_fp < slice.price_fp || slice.price_fp == 0 {
                            slice.price_fp = level.raw_price_fp;
                        }
                    }
                }
            } else {
                slices.push(OrderSlice {
                    exchange: level.exchange,
                    symbol: symbol.to_string(),
                    side: side.clone(),
                    size: fill_size,
                    price_fp: level.raw_price_fp,
                    expected_cost_bps: fee_bps + level.latency_penalty_bps as f64,
                    is_maker: false,
                });
                exchanges_used.insert(level.exchange);
            }

            remaining_size -= fill_size;
            total_cost_bps += fee_bps + level.latency_penalty_bps as f64;
        }

        // Calculate average cost
        let avg_cost_bps = if !slices.is_empty() {
            total_cost_bps / slices.len() as f64
        } else {
            0.0
        };

        // FEAT 8: Estimate savings vs worst single exchange using real-time fees
        let worst_single_fee = self
            .fee_overrides_bps
            .values()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(6.0); // At least Bybit's default
        let estimated_savings = worst_single_fee - avg_cost_bps;

        let filled_size = total_size - remaining_size;
        let routing_reason = if slices.len() > 1 {
            let exchange_details: Vec<String> = slices
                .iter()
                .map(|s| {
                    let q = self
                        .fill_quality
                        .get(&s.exchange)
                        .map(|q| format!("q={:.2}", q.quality_score()))
                        .unwrap_or_else(|| "q=?".to_string());
                    format!("{}({})", s.exchange.name(), q)
                })
                .collect();
            format!(
                "FEAT8 split across {} exchanges: {}",
                slices.len(),
                exchange_details.join(", ")
            )
        } else if slices.len() == 1 {
            format!("Single exchange: {}", slices[0].exchange.name())
        } else {
            "No routing (insufficient liquidity or quality)".to_string()
        };

        info!(
            "[sor] Multi-venue routing {}: {} slices, avg_cost={:.1}bps, savings={:.1}bps",
            symbol,
            slices.len(),
            avg_cost_bps,
            estimated_savings.max(0.0)
        );

        SorResult {
            slices,
            total_size: filled_size,
            estimated_slippage_bps: avg_cost_bps,
            estimated_savings_bps: estimated_savings.max(0.0),
            routing_reason,
        }
    }

    /// Route a single-exchange order (fallback when multi-exchange is off
    /// or when size is below the split threshold).
    pub fn route_single(
        &self,
        exchange: ExchangeId,
        side: OrderSide,
        total_size: i64,
        price_fp: i64,
        symbol: &str,
    ) -> SorResult {
        let fee_bps = exchange.taker_fee_bps() as f64;

        let slice = OrderSlice {
            exchange,
            symbol: symbol.to_string(),
            side,
            size: total_size,
            price_fp,
            expected_cost_bps: fee_bps,
            is_maker: false,
        };

        SorResult {
            slices: vec![slice],
            total_size,
            estimated_slippage_bps: fee_bps,
            estimated_savings_bps: 0.0,
            routing_reason: format!("Direct to {} (single-exchange mode)", exchange.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_exchange::global_book::ExchangeBookSnapshot;

    #[test]
    fn test_sor_single_exchange_routing() {
        let config = SorConfig::default();
        let mut sor = SmartOrderRouter::new(config);

        let mut book = GlobalOrderBook::new(1);

        // Add a simple snapshot
        let snap = ExchangeBookSnapshot {
            exchange: ExchangeId::Binance,
            symbol_id: 1,
            best_bid_fp: 5000000000000,
            best_ask_fp: 5000100000000,
            bid_levels: vec![(5000000000000, 1000000000000)],
            ask_levels: vec![(5000100000000, 1000000000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(snap);

        let result = sor.route(
            &book,
            OrderSide::Buy,
            100000000,
            5000050000000,
            "BTC_USDT",
        );

        assert_eq!(result.slices.len(), 1);
        assert_eq!(result.slices[0].exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_sor_multi_venue_routing() {
        let config = SorConfig {
            min_split_size_usdt: 100.0,
            max_venues: 3,
            max_slippage_bps: 100.0,
            prefer_maker: false,
            ..SorConfig::default()
        };
        let mut sor = SmartOrderRouter::new(config);

        let mut book = GlobalOrderBook::new(1);

        let gateio_snap = ExchangeBookSnapshot {
            exchange: ExchangeId::GateIo,
            symbol_id: 1,
            best_bid_fp: 5000000000000,
            best_ask_fp: 5000100000000,
            bid_levels: vec![(5000000000000, 500000000)],
            ask_levels: vec![(5000100000000, 500000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(gateio_snap);

        let binance_snap = ExchangeBookSnapshot {
            exchange: ExchangeId::Binance,
            symbol_id: 1,
            best_bid_fp: 5000000000000,
            best_ask_fp: 5000150000000,
            bid_levels: vec![(5000000000000, 500000000)],
            ask_levels: vec![(5000150000000, 500000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(binance_snap);

        let result = sor.route(
            &book,
            OrderSide::Buy,
            800000000,
            5000050000000,
            "BTC_USDT",
        );

        assert!(!result.slices.is_empty());
    }

    #[test]
    fn test_fill_quality_tracking() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        // Record some fills for Binance
        sor.record_fill_outcome(ExchangeId::Binance, true, false, 100.0, 0.5);
        sor.record_fill_outcome(ExchangeId::Binance, true, false, 120.0, 0.3);
        sor.record_fill_outcome(ExchangeId::Binance, true, false, 90.0, 0.4);

        let quality = sor.get_fill_quality(ExchangeId::Binance).unwrap();
        assert_eq!(quality.total_orders, 3);
        assert_eq!(quality.filled_orders, 3);
        assert!((quality.fill_rate() - 1.0).abs() < 0.01);
        assert!(quality.quality_score() > 0.5);

        // Record a rejection for Bybit
        sor.record_fill_outcome(ExchangeId::Bybit, false, false, 0.0, 0.0);
        let bybit_quality = sor.get_fill_quality(ExchangeId::Bybit).unwrap();
        assert_eq!(bybit_quality.rejected_orders, 1);
        assert!(bybit_quality.fill_rate() < 1.0);
    }

    #[test]
    fn test_latency_tracking() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        sor.update_latency(ExchangeId::GateIo, 500.0);
        sor.update_latency(ExchangeId::GateIo, 600.0);
        sor.update_latency(ExchangeId::GateIo, 550.0);

        let quality = sor.get_fill_quality(ExchangeId::GateIo).unwrap();
        assert!(quality.latency_ema_us > 0.0);
        assert!(quality.latency_ema_us < 700.0);
    }

    #[test]
    fn test_fee_override() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        // Default Binance fee is 4 bps
        assert!((sor.effective_taker_fee_bps(ExchangeId::Binance) - 4.0).abs() < 0.01);

        // Override with VIP tier fee
        sor.update_fee_tier(ExchangeId::Binance, 2.5);
        assert!((sor.effective_taker_fee_bps(ExchangeId::Binance) - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_stats_json() {
        let sor = SmartOrderRouter::new(SorConfig::default());
        let stats = sor.stats_json();
        assert!(stats.get("total_routed").is_some());
        assert!(stats.get("exchange_quality").is_some());
    }
}
