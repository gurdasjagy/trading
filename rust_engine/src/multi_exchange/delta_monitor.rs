//! FEATURE 5: Position Delta Neutrality Monitor.
//!
//! Monitors net delta exposure across all exchanges for funding arb positions.
//! When the net position deviates from delta-neutral (e.g., due to partial fills,
//! leg failures, or execution slippage), generates hedge orders to restore neutrality.
//!
//! # Algorithm
//! 1. Query positions from all active gateways
//! 2. Sum signed position sizes to compute net delta in USDT
//! 3. If |net_delta| > max_delta_usdt, generate a hedge order on the most liquid exchange
//! 4. Log all delta checks for audit trail

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{error, info, warn};

use crate::execution_gateway::{
    ExecutionGateway, OrderIntent, OrderSide, OrderType,
};
use crate::execution_state::PlacementType;
use crate::multi_exchange::global_book::ExchangeId;

/// Delta neutrality monitor for cross-exchange funding arbitrage.
pub struct DeltaNeutralityMonitor {
    /// Maximum allowed net delta in USDT before triggering a hedge.
    max_delta_usdt: f64,
    /// Interval between delta checks in seconds.
    check_interval_secs: u64,
    /// Last computed net delta (for telemetry).
    last_net_delta: f64,
    /// Total number of hedge orders generated.
    hedge_count: u64,
}

/// Result of a delta neutrality check.
#[derive(Debug, Clone)]
pub struct DeltaCheckResult {
    /// Net delta in USDT across all exchanges.
    pub net_delta_usdt: f64,
    /// Per-exchange deltas.
    pub per_exchange: Vec<(ExchangeId, f64)>,
    /// Whether the portfolio is within acceptable delta bounds.
    pub is_neutral: bool,
    /// Hedge order to execute if not neutral (None if neutral).
    pub hedge_needed: bool,
}

impl DeltaNeutralityMonitor {
    /// Create a new delta neutrality monitor.
    ///
    /// # Arguments
    /// * `max_delta_usdt` - Maximum allowed net delta in USDT (e.g., 50.0)
    /// * `check_interval_secs` - How often to check delta (e.g., 30)
    pub fn new(max_delta_usdt: f64, check_interval_secs: u64) -> Self {
        Self {
            max_delta_usdt,
            check_interval_secs,
            last_net_delta: 0.0,
            hedge_count: 0,
        }
    }

    /// Create with default parameters (max delta $50, check every 30s).
    pub fn with_defaults() -> Self {
        Self::new(50.0, 30)
    }

    /// Get the check interval in seconds.
    pub fn check_interval_secs(&self) -> u64 {
        self.check_interval_secs
    }

    /// Get the last computed net delta.
    pub fn last_net_delta(&self) -> f64 {
        self.last_net_delta
    }

    /// Get total hedge orders generated.
    pub fn hedge_count(&self) -> u64 {
        self.hedge_count
    }

    /// Calculate net delta across all exchanges for a symbol.
    ///
    /// Queries positions from each gateway and sums the signed USDT value.
    /// Long positions contribute positive delta, short positions contribute negative.
    pub async fn calculate_net_delta(
        &mut self,
        symbol: &str,
        gateways: &HashMap<ExchangeId, Arc<dyn ExecutionGateway + Send + Sync>>,
    ) -> DeltaCheckResult {
        let mut net_delta = 0.0_f64;
        let mut per_exchange = Vec::new();

        for (exchange_id, gateway) in gateways {
            match gateway.get_position(symbol).await {
                Ok(Some(pos)) => {
                    // pos.size is in contracts (scaled), entry_price is the entry price
                    // For delta calculation: delta = size * entry_price
                    // size is already signed (positive=long, negative=short)
                    let delta_usdt = pos.size as f64 * pos.entry_price / 1e8;
                    net_delta += delta_usdt;
                    per_exchange.push((*exchange_id, delta_usdt));
                    info!(
                        "[delta-monitor] {} {}: size={} entry={:.2} delta=${:.2}",
                        symbol, exchange_id.name(), pos.size, pos.entry_price, delta_usdt
                    );
                }
                Ok(None) => {
                    per_exchange.push((*exchange_id, 0.0));
                }
                Err(e) => {
                    warn!(
                        "[delta-monitor] Failed to query {} position on {}: {:?}",
                        symbol, exchange_id.name(), e
                    );
                    // Don't add to per_exchange - we couldn't query this exchange
                }
            }
        }

        self.last_net_delta = net_delta;

        let is_neutral = net_delta.abs() < self.max_delta_usdt;

        if !is_neutral {
            warn!(
                "[delta-monitor] {} DELTA BREACH: net=${:.2} (max=${:.2})",
                symbol, net_delta, self.max_delta_usdt
            );
        } else {
            info!(
                "[delta-monitor] {} delta OK: net=${:.2} (max=${:.2})",
                symbol, net_delta, self.max_delta_usdt
            );
        }

        DeltaCheckResult {
            net_delta_usdt: net_delta,
            per_exchange,
            is_neutral,
            hedge_needed: !is_neutral,
        }
    }

    /// Check if the given net delta is within acceptable bounds.
    pub fn is_delta_neutral(&self, net_delta: f64) -> bool {
        net_delta.abs() < self.max_delta_usdt
    }

    /// Generate a hedge order to restore delta neutrality.
    ///
    /// If net_delta > 0, we're net long → sell to hedge.
    /// If net_delta < 0, we're net short → buy to hedge.
    ///
    /// Returns None if already delta neutral.
    pub fn generate_hedge_order(
        &mut self,
        net_delta: f64,
        symbol: &str,
        current_price: f64,
    ) -> Option<OrderIntent> {
        if self.is_delta_neutral(net_delta) {
            return None;
        }

        if current_price <= 0.0 {
            error!("[delta-monitor] Cannot generate hedge: invalid price {:.2}", current_price);
            return None;
        }

        let side = if net_delta > 0.0 {
            OrderSide::Sell
        } else {
            OrderSide::Buy
        };

        // Convert USDT delta to contract size
        let size = (net_delta.abs() / current_price).ceil() as i64;
        if size < 1 {
            return None;
        }

        self.hedge_count += 1;

        info!(
            "[delta-monitor] Generating hedge order: {:?} {} {} contracts @ ~{:.2} (delta=${:.2})",
            side, symbol, size, current_price, net_delta
        );

        Some(OrderIntent {
            symbol: symbol.to_string(),
            side,
            size,
            order_type: OrderType::Market,
            price: None,
            reduce_only: false,
            leverage: Some(3),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.002),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: "delta_hedge".to_string(),
        })
    }

    /// Get statistics as JSON for the dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "max_delta_usdt": self.max_delta_usdt,
            "check_interval_secs": self.check_interval_secs,
            "last_net_delta": self.last_net_delta,
            "is_neutral": self.is_delta_neutral(self.last_net_delta),
            "hedge_count": self.hedge_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_neutral_check() {
        let monitor = DeltaNeutralityMonitor::new(50.0, 30);

        assert!(monitor.is_delta_neutral(0.0));
        assert!(monitor.is_delta_neutral(49.9));
        assert!(monitor.is_delta_neutral(-49.9));
        assert!(!monitor.is_delta_neutral(50.1));
        assert!(!monitor.is_delta_neutral(-50.1));
    }

    #[test]
    fn test_hedge_order_generation() {
        let mut monitor = DeltaNeutralityMonitor::new(50.0, 30);

        // No hedge needed when neutral
        assert!(monitor.generate_hedge_order(30.0, "BTC_USDT", 60000.0).is_none());

        // Hedge needed when net long
        let order = monitor.generate_hedge_order(100.0, "BTC_USDT", 60000.0);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::Sell);
        assert_eq!(order.signal_tag, "delta_hedge");

        // Hedge needed when net short
        let order = monitor.generate_hedge_order(-100.0, "BTC_USDT", 60000.0);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::Buy);

        assert_eq!(monitor.hedge_count(), 2);
    }

    #[test]
    fn test_invalid_price() {
        let mut monitor = DeltaNeutralityMonitor::new(50.0, 30);
        assert!(monitor.generate_hedge_order(100.0, "BTC_USDT", 0.0).is_none());
        assert!(monitor.generate_hedge_order(100.0, "BTC_USDT", -1.0).is_none());
    }
}
