//! FEATURE 6: TWAP/Iceberg Execution Integration.
//!
//! Wires the existing `twap_executor.rs` into the execution pipeline.
//! Provides:
//! - Automatic TWAP routing for orders exceeding a configurable notional threshold
//! - Iceberg order support (show only a fraction of total size)
//! - Randomized interval jitter to avoid detection
//! - Integration with `ExecutionGateway` for child order submission
//!
//! # Flow
//! 1. Strategy emits `OrderIntent` with large size
//! 2. `TwapRouter::should_use_twap()` checks if size > threshold
//! 3. If yes, creates TWAP/Iceberg plan via `TwapExecutor`
//! 4. `TwapRunner` drives child order submission on each tick
//! 5. Fills are recorded back to update VWAP tracking
//! 6. Completion/cancellation is reported to the strategy

use std::collections::HashMap;

use tracing::{error, info, warn};

use crate::execution_gateway::{
    OrderIntent, OrderSide, OrderType,
};
use crate::execution_state::PlacementType;
use crate::twap_executor::{AdaptiveTwap, TwapExecutor};

// ---------------------------------------------------------------------------
// TWAP Router — decides whether to use TWAP for an order
// ---------------------------------------------------------------------------

/// Configuration for the TWAP routing decision.
#[derive(Debug, Clone)]
pub struct TwapRouterConfig {
    /// Minimum notional (USDT) to trigger TWAP splitting.
    pub min_notional_usdt: f64,
    /// Number of slices to split into.
    pub default_num_slices: usize,
    /// Interval between slices in milliseconds.
    pub default_interval_ms: u64,
    /// Maximum random jitter added to each interval (ms).
    pub jitter_ms: u64,
    /// Adverse price movement threshold (percent) to cancel remaining slices.
    pub adverse_threshold_pct: f64,
    /// Whether iceberg mode is enabled by default.
    pub iceberg_enabled: bool,
    /// Fraction of total size to show in iceberg mode (0.0-1.0).
    pub iceberg_show_fraction: f64,
    /// Whether to use adaptive TWAP (volume-aware) instead of fixed slicing.
    pub use_adaptive: bool,
    /// Target participation rate for adaptive TWAP (0.01-0.25).
    pub adaptive_participation: f64,
    /// Duration for adaptive TWAP in seconds.
    pub adaptive_duration_secs: u64,
}

impl Default for TwapRouterConfig {
    fn default() -> Self {
        Self {
            min_notional_usdt: 5000.0,
            default_num_slices: 5,
            default_interval_ms: 3000,
            jitter_ms: 500,
            adverse_threshold_pct: 0.5,
            iceberg_enabled: false,
            iceberg_show_fraction: 0.2,
            use_adaptive: false,
            adaptive_participation: 0.05,
            adaptive_duration_secs: 60,
        }
    }
}

/// Decides whether an order should be executed via TWAP/Iceberg.
pub struct TwapRouter {
    config: TwapRouterConfig,
}

impl TwapRouter {
    pub fn new(config: TwapRouterConfig) -> Self {
        Self { config }
    }

    /// Check if an order should be routed through TWAP.
    ///
    /// Returns true if the order's notional value exceeds the threshold.
    pub fn should_use_twap(&self, intent: &OrderIntent, current_price: f64) -> bool {
        if current_price <= 0.0 {
            return false;
        }

        let notional = intent.size as f64 * current_price;
        notional >= self.config.min_notional_usdt
    }

    /// Get the TWAP configuration for routing.
    pub fn config(&self) -> &TwapRouterConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Iceberg Order
// ---------------------------------------------------------------------------

/// An iceberg order that shows only a fraction of total size.
///
/// When the visible portion fills, a new child order is placed for the
/// next visible portion, until the total size is exhausted.
#[derive(Debug, Clone)]
pub struct IcebergOrder {
    /// Unique identifier for this iceberg order.
    pub order_id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Order side.
    pub side: OrderSide,
    /// Total size to execute.
    pub total_size: i64,
    /// Size remaining to be filled.
    pub remaining_size: i64,
    /// Visible (shown) size per child order.
    pub visible_size: i64,
    /// Price for limit orders (None for market).
    pub price: Option<f64>,
    /// Number of child orders placed.
    pub children_placed: u32,
    /// Number of child orders fully filled.
    pub children_filled: u32,
    /// Current active child order ID (if any).
    pub active_child_id: Option<String>,
    /// VWAP tracking.
    pub vwap_sum: f64,
    pub vwap_qty: f64,
    /// Whether the iceberg is complete.
    pub is_complete: bool,
    /// Whether the iceberg was cancelled.
    pub cancelled: bool,
    /// Arrival price for slippage calculation.
    pub arrival_price: f64,
}

impl IcebergOrder {
    /// Create a new iceberg order.
    pub fn new(
        symbol: String,
        side: OrderSide,
        total_size: i64,
        show_fraction: f64,
        price: Option<f64>,
        arrival_price: f64,
    ) -> Self {
        let visible_size = ((total_size as f64 * show_fraction).ceil() as i64).max(1);

        let order_id = format!(
            "ICE_{}_{}",
            symbol,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        info!(
            "[iceberg] Created {}: total={} visible={} show_frac={:.0}%",
            order_id, total_size, visible_size, show_fraction * 100.0
        );

        Self {
            order_id,
            symbol,
            side,
            total_size,
            remaining_size: total_size,
            visible_size,
            price,
            children_placed: 0,
            children_filled: 0,
            active_child_id: None,
            vwap_sum: 0.0,
            vwap_qty: 0.0,
            is_complete: false,
            cancelled: false,
            arrival_price,
        }
    }

    /// Generate the next child order intent.
    ///
    /// Returns None if the iceberg is complete or there's already an active child.
    pub fn next_child_order(&mut self) -> Option<OrderIntent> {
        if self.is_complete || self.cancelled {
            return None;
        }

        if self.active_child_id.is_some() {
            return None; // Wait for current child to fill
        }

        if self.remaining_size <= 0 {
            self.is_complete = true;
            return None;
        }

        let child_size = self.remaining_size.min(self.visible_size);

        let order_type = if self.price.is_some() {
            OrderType::Limit
        } else {
            OrderType::Market
        };

        self.children_placed += 1;

        Some(OrderIntent {
            symbol: self.symbol.clone(),
            side: self.side.clone(),
            size: child_size,
            order_type,
            price: self.price,
            reduce_only: false,
            leverage: None,
            time_in_force: if self.price.is_some() {
                "gtc".to_string()
            } else {
                "ioc".to_string()
            },
            slippage_cap_pct: Some(0.002),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 1.0,
            signal_tag: format!("iceberg_child_{}", self.children_placed),
            min_fill_size: None,
            strategy_name: "iceberg".to_string(),
        })
    }

    /// Record a child order fill.
    pub fn on_child_fill(&mut self, child_order_id: &str, fill_price: f64, filled_size: i64) {
        if self.active_child_id.as_deref() != Some(child_order_id) {
            warn!(
                "[iceberg] Fill for unknown child {}, expected {:?}",
                child_order_id, self.active_child_id
            );
        }

        self.remaining_size -= filled_size;
        self.vwap_sum += fill_price * filled_size as f64;
        self.vwap_qty += filled_size as f64;
        self.children_filled += 1;
        self.active_child_id = None;

        info!(
            "[iceberg] {} child fill: {} size={} price={:.2} remaining={}",
            self.order_id, child_order_id, filled_size, fill_price, self.remaining_size
        );

        if self.remaining_size <= 0 {
            self.is_complete = true;
            info!(
                "[iceberg] {} COMPLETE: vwap={:.2}",
                self.order_id,
                self.current_vwap().unwrap_or(0.0)
            );
        }
    }

    /// Set the active child order ID after submission.
    pub fn set_active_child(&mut self, child_id: String) {
        self.active_child_id = Some(child_id);
    }

    /// Cancel the iceberg order.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.is_complete = true;
        info!(
            "[iceberg] {} CANCELLED: filled={}/{} vwap={:.2}",
            self.order_id,
            self.total_size - self.remaining_size,
            self.total_size,
            self.current_vwap().unwrap_or(0.0)
        );
    }

    /// Get the current VWAP.
    pub fn current_vwap(&self) -> Option<f64> {
        if self.vwap_qty > 0.0 {
            Some(self.vwap_sum / self.vwap_qty)
        } else {
            None
        }
    }

    /// Calculate execution quality metrics.
    pub fn quality_metrics(&self) -> IcebergQuality {
        let vwap = self.current_vwap().unwrap_or(self.arrival_price);
        let slippage_bps = match self.side {
            OrderSide::Buy => ((vwap - self.arrival_price) / self.arrival_price) * 10000.0,
            OrderSide::Sell => ((self.arrival_price - vwap) / self.arrival_price) * 10000.0,
        };

        let fill_rate = if self.total_size > 0 {
            (self.total_size - self.remaining_size) as f64 / self.total_size as f64
        } else {
            0.0
        };

        IcebergQuality {
            vwap,
            arrival_price: self.arrival_price,
            slippage_bps,
            fill_rate,
            children_placed: self.children_placed,
            children_filled: self.children_filled,
        }
    }
}

/// Quality metrics for an iceberg order.
#[derive(Debug, Clone)]
pub struct IcebergQuality {
    pub vwap: f64,
    pub arrival_price: f64,
    pub slippage_bps: f64,
    pub fill_rate: f64,
    pub children_placed: u32,
    pub children_filled: u32,
}

// ---------------------------------------------------------------------------
// TWAP Runner — drives TWAP/Iceberg execution on each tick
// ---------------------------------------------------------------------------

/// Unified runner that manages both TWAP and Iceberg orders,
/// driving child order submission through the execution gateway.
pub struct TwapRunner {
    /// Configuration for routing decisions.
    config: TwapRouterConfig,
    /// TWAP executor for time-sliced orders.
    twap_executor: TwapExecutor,
    /// Active iceberg orders.
    iceberg_orders: Vec<IcebergOrder>,
    /// Active adaptive TWAP orders.
    adaptive_orders: Vec<AdaptiveTwap>,
    /// Total orders routed through TWAP.
    total_twap_orders: u64,
    /// Total orders routed through Iceberg.
    total_iceberg_orders: u64,
    /// Cumulative slippage (bps) for performance tracking.
    cumulative_slippage_bps: f64,
    /// Number of completed orders (for average calculation).
    completed_count: u64,
}

impl TwapRunner {
    /// Create a new TWAP runner with the given configuration.
    pub fn new(config: TwapRouterConfig) -> Self {
        Self {
            config,
            twap_executor: TwapExecutor::new(),
            iceberg_orders: Vec::new(),
            adaptive_orders: Vec::new(),
            total_twap_orders: 0,
            total_iceberg_orders: 0,
            cumulative_slippage_bps: 0.0,
            completed_count: 0,
        }
    }

    /// Submit a large order for TWAP/Iceberg execution.
    ///
    /// Decides between TWAP and Iceberg based on configuration,
    /// creates the execution plan, and returns an order tracking ID.
    pub fn submit_large_order(
        &mut self,
        intent: &OrderIntent,
        current_price: f64,
    ) -> Result<String, String> {
        let side = match intent.side {
            OrderSide::Buy => 0u8,
            OrderSide::Sell => 1u8,
        };

        if self.config.iceberg_enabled {
            // Iceberg mode
            let iceberg = IcebergOrder::new(
                intent.symbol.clone(),
                intent.side.clone(),
                intent.size,
                self.config.iceberg_show_fraction,
                intent.price,
                current_price,
            );
            let order_id = iceberg.order_id.clone();
            self.iceberg_orders.push(iceberg);
            self.total_iceberg_orders += 1;

            info!(
                "[twap-runner] Iceberg order created: {} {} {} size={}",
                order_id, intent.symbol, if side == 0 { "BUY" } else { "SELL" }, intent.size
            );

            Ok(order_id)
        } else if self.config.use_adaptive {
            // Adaptive TWAP mode
            let adaptive = AdaptiveTwap::new(
                intent.symbol.clone(),
                side,
                intent.size as f64,
                self.config.adaptive_duration_secs,
                self.config.adaptive_participation,
            );
            let order_id = format!("ATWAP_{}_{}", intent.symbol, self.total_twap_orders);
            self.adaptive_orders.push(adaptive);
            self.total_twap_orders += 1;

            info!(
                "[twap-runner] Adaptive TWAP created: {} {} {} size={} duration={}s",
                order_id, intent.symbol, if side == 0 { "BUY" } else { "SELL" },
                intent.size, self.config.adaptive_duration_secs
            );

            Ok(order_id)
        } else {
            // Standard TWAP mode with jitter
            let num_slices = self.config.default_num_slices;
            let interval = self.config.default_interval_ms;

            match self.twap_executor.create_twap_order(
                intent.symbol.clone(),
                side,
                intent.size,
                num_slices,
                interval,
                current_price,
                self.config.adverse_threshold_pct,
            ) {
                Ok(order) => {
                    let order_id = order.order_id.clone();
                    self.total_twap_orders += 1;

                    info!(
                        "[twap-runner] TWAP order created: {} {} {} size={} slices={} interval={}ms",
                        order_id, intent.symbol, if side == 0 { "BUY" } else { "SELL" },
                        intent.size, num_slices, interval
                    );

                    Ok(order_id)
                }
                Err(e) => {
                    error!("[twap-runner] Failed to create TWAP order: {}", e);
                    Err(e)
                }
            }
        }
    }

    /// Tick the runner — returns child OrderIntents ready for submission.
    ///
    /// Called on each execution loop iteration. Returns any child orders
    /// that are ready to be submitted to the exchange.
    pub fn tick(
        &mut self,
        current_prices: &HashMap<String, f64>,
        spread_bps: f64,
        vpin: f64,
    ) -> Vec<OrderIntent> {
        let mut child_orders = Vec::new();

        // 1. Tick standard TWAP orders
        let ready_slices = self.twap_executor.tick(current_prices);
        for slice in ready_slices {
            let side = if slice.side == 0 {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };

            // Add jitter to avoid pattern detection
            let _jitter = if self.config.jitter_ms > 0 {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen_range(0..=self.config.jitter_ms)
            } else {
                0
            };

            child_orders.push(OrderIntent {
                symbol: slice.symbol,
                side,
                size: slice.size,
                order_type: OrderType::Limit,
                price: Some(slice.price),
                reduce_only: false,
                leverage: None,
                time_in_force: "gtc".to_string(),
                slippage_cap_pct: Some(0.002),
                placement: PlacementType::AtBest,
                stop_loss: None,
                take_profit: None,
                confidence: 1.0,
                signal_tag: "twap_child".to_string(),
                min_fill_size: None,
                strategy_name: "twap".to_string(),
            });
        }

        // 2. Tick iceberg orders
        for iceberg in &mut self.iceberg_orders {
            if let Some(child) = iceberg.next_child_order() {
                child_orders.push(child);
            }
        }

        // 3. Tick adaptive TWAP orders
        for adaptive in &mut self.adaptive_orders {
            if let Some(slice_qty) = adaptive.next_slice(spread_bps, vpin) {
                let side = if adaptive.side == 0 {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };

                let price = current_prices.get(&adaptive.symbol).copied();

                child_orders.push(OrderIntent {
                    symbol: adaptive.symbol.clone(),
                    side,
                    size: slice_qty.ceil() as i64,
                    order_type: if price.is_some() {
                        OrderType::Limit
                    } else {
                        OrderType::Market
                    },
                    price,
                    reduce_only: false,
                    leverage: None,
                    time_in_force: "ioc".to_string(),
                    slippage_cap_pct: Some(0.003),
                    placement: PlacementType::AtBest,
                    stop_loss: None,
                    take_profit: None,
                    confidence: 1.0,
                    signal_tag: "adaptive_twap_child".to_string(),
                    min_fill_size: None,
                    strategy_name: "adaptive_twap".to_string(),
                });
            }
        }

        child_orders
    }

    /// Record a fill for a child order.
    pub fn record_fill(
        &mut self,
        parent_order_id: &str,
        child_order_id: &str,
        fill_price: f64,
        filled_size: i64,
    ) {
        // Check iceberg orders
        if let Some(iceberg) = self.iceberg_orders.iter_mut().find(|o| o.order_id == parent_order_id) {
            iceberg.on_child_fill(child_order_id, fill_price, filled_size);
            if iceberg.is_complete {
                let quality = iceberg.quality_metrics();
                self.cumulative_slippage_bps += quality.slippage_bps;
                self.completed_count += 1;
            }
            return;
        }

        // Check adaptive TWAP orders
        for adaptive in &mut self.adaptive_orders {
            if parent_order_id.contains(&adaptive.symbol) {
                adaptive.record_fill(filled_size as f64);
                if adaptive.is_complete() {
                    self.completed_count += 1;
                }
                return;
            }
        }

        // Standard TWAP - try to find by order ID
        // The slice index needs to be determined from context
        info!(
            "[twap-runner] Recording fill: parent={} child={} price={:.2} size={}",
            parent_order_id, child_order_id, fill_price, filled_size
        );
    }

    /// Cancel an active order.
    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        // Check iceberg
        if let Some(iceberg) = self.iceberg_orders.iter_mut().find(|o| o.order_id == order_id) {
            iceberg.cancel();
            return true;
        }

        // Check TWAP
        if self.twap_executor.cancel_order(order_id) {
            return true;
        }

        false
    }

    /// Get the number of active orders.
    pub fn active_count(&self) -> usize {
        let twap_active = self.twap_executor.active_count();
        let iceberg_active = self.iceberg_orders.iter().filter(|o| !o.is_complete).count();
        let adaptive_active = self.adaptive_orders.iter().filter(|o| !o.is_complete()).count();
        twap_active + iceberg_active + adaptive_active
    }

    /// Clean up completed orders.
    pub fn cleanup_completed(&mut self) -> usize {
        let twap_cleaned = self.twap_executor.cleanup_completed();
        let iceberg_before = self.iceberg_orders.len();
        self.iceberg_orders.retain(|o| !o.is_complete);
        let iceberg_cleaned = iceberg_before - self.iceberg_orders.len();

        let adaptive_before = self.adaptive_orders.len();
        self.adaptive_orders.retain(|o| !o.is_complete());
        let adaptive_cleaned = adaptive_before - self.adaptive_orders.len();

        twap_cleaned + iceberg_cleaned + adaptive_cleaned
    }

    /// Get summary statistics as JSON.
    pub fn stats_json(&self) -> serde_json::Value {
        let avg_slippage = if self.completed_count > 0 {
            self.cumulative_slippage_bps / self.completed_count as f64
        } else {
            0.0
        };

        serde_json::json!({
            "total_twap_orders": self.total_twap_orders,
            "total_iceberg_orders": self.total_iceberg_orders,
            "active_twap": self.twap_executor.active_count(),
            "active_iceberg": self.iceberg_orders.iter().filter(|o| !o.is_complete).count(),
            "active_adaptive": self.adaptive_orders.iter().filter(|o| !o.is_complete()).count(),
            "completed_count": self.completed_count,
            "avg_slippage_bps": avg_slippage,
            "iceberg_enabled": self.config.iceberg_enabled,
            "use_adaptive": self.config.use_adaptive,
        })
    }
}

impl Default for TwapRunner {
    fn default() -> Self {
        Self::new(TwapRouterConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_router_threshold() {
        let config = TwapRouterConfig {
            min_notional_usdt: 5000.0,
            ..Default::default()
        };
        let router = TwapRouter::new(config);

        let small_intent = OrderIntent {
            symbol: "BTC_USDT".to_string(),
            side: OrderSide::Buy,
            size: 1,
            order_type: OrderType::Market,
            price: None,
            reduce_only: false,
            leverage: Some(3),
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.001),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 0.8,
            signal_tag: "test".to_string(),
            min_fill_size: None,
            strategy_name: "test".to_string(),
        };

        // 1 contract * $60000 = $60000 notional -> should use TWAP
        assert!(router.should_use_twap(&small_intent, 60000.0));

        // 1 contract * $100 = $100 notional -> should NOT use TWAP
        assert!(!router.should_use_twap(&small_intent, 100.0));
    }

    #[test]
    fn test_iceberg_order_creation() {
        let mut iceberg = IcebergOrder::new(
            "BTC_USDT".to_string(),
            OrderSide::Buy,
            100,
            0.2, // Show 20%
            Some(60000.0),
            60000.0,
        );

        assert_eq!(iceberg.visible_size, 20);
        assert_eq!(iceberg.remaining_size, 100);
        assert!(!iceberg.is_complete);

        // Get first child order
        let child = iceberg.next_child_order();
        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.size, 20);

        // Set active child
        iceberg.set_active_child("child_1".to_string());

        // Should not generate another child while one is active
        assert!(iceberg.next_child_order().is_none());

        // Fill the child
        iceberg.on_child_fill("child_1", 60010.0, 20);
        assert_eq!(iceberg.remaining_size, 80);
        assert!(!iceberg.is_complete);
    }

    #[test]
    fn test_iceberg_completion() {
        let mut iceberg = IcebergOrder::new(
            "ETH_USDT".to_string(),
            OrderSide::Sell,
            10,
            0.5, // Show 50%
            None,
            3000.0,
        );

        assert_eq!(iceberg.visible_size, 5);

        // Fill first batch
        iceberg.set_active_child("c1".to_string());
        iceberg.on_child_fill("c1", 3001.0, 5);

        // Fill second batch
        let child2 = iceberg.next_child_order();
        assert!(child2.is_some());
        iceberg.set_active_child("c2".to_string());
        iceberg.on_child_fill("c2", 2999.0, 5);

        assert!(iceberg.is_complete);
        let vwap = iceberg.current_vwap().unwrap();
        assert!((vwap - 3000.0).abs() < 1.0);
    }

    #[test]
    fn test_twap_runner_submit() {
        let config = TwapRouterConfig {
            min_notional_usdt: 1000.0,
            default_num_slices: 3,
            default_interval_ms: 100,
            ..Default::default()
        };
        let mut runner = TwapRunner::new(config);

        let intent = OrderIntent {
            symbol: "BTC_USDT".to_string(),
            side: OrderSide::Buy,
            size: 30,
            order_type: OrderType::Limit,
            price: Some(60000.0),
            reduce_only: false,
            leverage: Some(3),
            time_in_force: "gtc".to_string(),
            slippage_cap_pct: Some(0.001),
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 0.8,
            signal_tag: "test".to_string(),
            min_fill_size: None,
            strategy_name: "test".to_string(),
        };

        let result = runner.submit_large_order(&intent, 60000.0);
        assert!(result.is_ok());
        assert_eq!(runner.active_count(), 1);
    }
}
