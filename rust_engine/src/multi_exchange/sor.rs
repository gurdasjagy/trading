//! Smart Order Router (SOR)
//!
//! Determines optimal order splitting across exchanges based on:
//! - Available liquidity at each venue
//! - Fee-adjusted effective prices
//! - Measured latency
//! - Minimum split thresholds
//!
//! When order size is below the split threshold, routes entirely to the
//! exchange with the best effective price. Above threshold, sweeps the
//! global book level by level.

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
}

impl Default for SorConfig {
    fn default() -> Self {
        Self {
            min_split_size_usdt: 5000.0,
            max_venues: 3,
            max_slippage_bps: 30.0,
            prefer_maker: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Smart Order Router
// ---------------------------------------------------------------------------

/// Smart Order Router for cross-exchange order splitting.
pub struct SmartOrderRouter {
    config: SorConfig,
}

impl SmartOrderRouter {
    /// Create a new SmartOrderRouter with the given configuration.
    pub fn new(config: SorConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal order routing given the current global book state.
    ///
    /// Algorithm:
    /// 1. If total_size_usdt < min_split_size_usdt -> route entirely to best
    ///    single exchange (lowest effective ask for buys, highest effective bid
    ///    for sells).
    /// 2. Otherwise, sweep the global book level by level, allocating size to
    ///    each exchange until the order is fully filled or max_venues is reached.
    /// 3. Apply fee adjustment: prefer exchanges with maker rebates when the
    ///    order can rest on the book (is_maker=true).
    /// 4. Return SorResult with per-exchange slices.
    pub fn route(
        &self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        mid_price_fp: i64,
        symbol: &str,
    ) -> SorResult {
        if total_size <= 0 {
            return SorResult::empty();
        }

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

        // Multi-exchange routing: sweep the global book
        self.route_multi_venue(book, side, total_size, mid_price_fp, symbol)
    }

    /// Route entirely to the single best exchange.
    fn route_to_best_single(
        &self,
        book: &GlobalOrderBook,
        side: OrderSide,
        total_size: i64,
        symbol: &str,
    ) -> SorResult {
        let (best_level, slippage_bps) = match side {
            OrderSide::Buy => {
                // For buys, find the best (lowest) effective ask
                match book.best_ask() {
                    Some(level) => {
                        let fee_bps = level.exchange.taker_fee_bps() as f64;
                        (Some(level), fee_bps + level.latency_penalty_bps as f64)
                    }
                    None => (None, 0.0),
                }
            }
            OrderSide::Sell => {
                // For sells, find the best (highest) effective bid
                match book.best_bid() {
                    Some(level) => {
                        let fee_bps = level.exchange.taker_fee_bps() as f64;
                        (Some(level), fee_bps + level.latency_penalty_bps as f64)
                    }
                    None => (None, 0.0),
                }
            }
        };

        let Some(level) = best_level else {
            return SorResult {
                slices: Vec::new(),
                total_size,
                estimated_slippage_bps: 0.0,
                estimated_savings_bps: 0.0,
                routing_reason: "No liquidity available".to_string(),
            };
        };

        let slice = OrderSlice {
            exchange: level.exchange,
            symbol: symbol.to_string(),
            side: side.clone(),
            size: total_size,
            price_fp: level.raw_price_fp,
            expected_cost_bps: slippage_bps,
            is_maker: false,  // Taker for immediate execution
        };

        SorResult {
            slices: vec![slice],
            total_size,
            estimated_slippage_bps: slippage_bps,
            estimated_savings_bps: 0.0,  // No savings vs single exchange (this IS single)
            routing_reason: format!("Single exchange: {} (best effective price)", level.exchange.name()),
        }
    }

    /// Route across multiple venues by sweeping the global book.
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

        // Sweep through sorted levels
        for level in levels {
            if remaining_size <= 0 {
                break;
            }
            if exchanges_used.len() >= self.config.max_venues {
                break;
            }

            // Calculate slippage from mid price
            let slippage_bps = if mid_price_fp > 0 {
                ((level.effective_price_fp - mid_price_fp).abs() * 10000) / mid_price_fp
            } else {
                0
            };

            if slippage_bps as f64 > self.config.max_slippage_bps {
                // Too much slippage, stop sweeping
                break;
            }

            let fill_size = remaining_size.min(level.qty);
            if fill_size <= 0 {
                continue;
            }

            // Check if we already have a slice for this exchange
            let existing_slice = slices.iter_mut().find(|s| s.exchange == level.exchange);
            
            if let Some(slice) = existing_slice {
                // Add to existing slice
                slice.size += fill_size;
                // Update price to worst-case (for buys: higher, for sells: lower)
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
                // Create new slice for this exchange
                let fee_bps = level.exchange.taker_fee_bps() as f64;
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
            total_cost_bps += (level.exchange.taker_fee_bps() + level.latency_penalty_bps) as f64;
        }

        // Calculate average cost
        let avg_cost_bps = if !slices.is_empty() {
            total_cost_bps / slices.len() as f64
        } else {
            0.0
        };

        // Estimate savings vs. single exchange (simplistic: assume worst single exchange fee)
        let worst_single_fee = 6.0;  // Bybit's fee
        let estimated_savings = worst_single_fee - avg_cost_bps;

        let filled_size = total_size - remaining_size;
        let routing_reason = if slices.len() > 1 {
            format!(
                "Split across {} exchanges: {}",
                slices.len(),
                slices.iter().map(|s| s.exchange.name()).collect::<Vec<_>>().join(", ")
            )
        } else if slices.len() == 1 {
            format!("Single exchange: {}", slices[0].exchange.name())
        } else {
            "No routing (insufficient liquidity)".to_string()
        };

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
        let sor = SmartOrderRouter::new(config);

        let mut book = GlobalOrderBook::new(1);
        
        // Add a simple snapshot
        let snap = ExchangeBookSnapshot {
            exchange: ExchangeId::Binance,
            symbol_id: 1,
            best_bid_fp: 5000000000000,
            best_ask_fp: 5000100000000,
            bid_levels: vec![(5000000000000, 1000000000000)],  // Large liquidity
            ask_levels: vec![(5000100000000, 1000000000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(snap);

        // Small order should route to single exchange
        let result = sor.route(
            &book,
            OrderSide::Buy,
            100000000,  // 1 contract
            5000050000000,  // mid price
            "BTC_USDT",
        );

        assert_eq!(result.slices.len(), 1);
        assert_eq!(result.slices[0].exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_sor_multi_venue_routing() {
        let config = SorConfig {
            min_split_size_usdt: 100.0,  // Low threshold for testing
            max_venues: 3,
            max_slippage_bps: 100.0,
            prefer_maker: false,
        };
        let sor = SmartOrderRouter::new(config);

        let mut book = GlobalOrderBook::new(1);
        
        // Add snapshots from two exchanges
        let gateio_snap = ExchangeBookSnapshot {
            exchange: ExchangeId::GateIo,
            symbol_id: 1,
            best_bid_fp: 5000000000000,
            best_ask_fp: 5000100000000,
            bid_levels: vec![(5000000000000, 500000000)],
            ask_levels: vec![(5000100000000, 500000000)],  // Limited liquidity
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(gateio_snap);

        let binance_snap = ExchangeBookSnapshot {
            exchange: ExchangeId::Binance,
            symbol_id: 1,
            best_bid_fp: 5000000000000,
            best_ask_fp: 5000150000000,  // Slightly worse price
            bid_levels: vec![(5000000000000, 500000000)],
            ask_levels: vec![(5000150000000, 500000000)],
            sequence: 1,
            timestamp_ns: 1000,
        };
        book.update_exchange_snapshot(binance_snap);

        // Large order should potentially split
        let result = sor.route(
            &book,
            OrderSide::Buy,
            800000000,  // 8 contracts - more than single exchange liquidity
            5000050000000,
            "BTC_USDT",
        );

        // Should have routing result
        assert!(!result.slices.is_empty());
    }
}
