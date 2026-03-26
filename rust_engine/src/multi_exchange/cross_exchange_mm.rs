//! Cross-Exchange Market Making (Hedged Maker Strategy)
//!
//! Places passive limit orders (maker) on a wide-spread exchange and instantly
//! hedges fills with aggressive market orders (taker) on a liquid exchange.
//!
//! Profit = maker spread captured - hedge taker fee + maker rebate earned

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::execution_gateway::{ExecutionGateway, OrderIntent, OrderResult, OrderSide, OrderType};
use crate::execution_state::PlacementType;
use crate::multi_exchange::global_book::{ExchangeId, GlobalBookRegistry, SharedGlobalBook};

// ---------------------------------------------------------------------------
// Maker Order Tracking
// ---------------------------------------------------------------------------

/// A resting maker order on an exchange.
#[derive(Debug, Clone)]
pub struct MakerOrder {
    pub order_id: String,
    pub symbol: String,
    pub exchange: ExchangeId,
    pub side: OrderSide,
    pub price: f64,
    pub size: i64,
    pub original_size: i64,
    pub filled_size: i64,
    pub created_ns: u64,
    pub last_checked_ns: u64,
    pub status: MakerOrderStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MakerOrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
}

// ---------------------------------------------------------------------------
// Hedged Position
// ---------------------------------------------------------------------------

/// A pair of hedged positions across two exchanges.
#[derive(Debug, Clone)]
pub struct HedgePosition {
    pub symbol: String,
    pub maker_exchange: ExchangeId,
    pub hedge_exchange: ExchangeId,
    pub maker_side: OrderSide,
    pub maker_fill_price: f64,
    pub hedge_fill_price: f64,
    pub size: i64,
    pub opened_ns: u64,
    pub realized_pnl: f64,
}

// ---------------------------------------------------------------------------
// Cross-Exchange Market Maker Configuration
// ---------------------------------------------------------------------------

/// Configuration for the cross-exchange market maker.
#[derive(Debug, Clone)]
pub struct CrossExchangeMMConfig {
    /// Minimum spread in basis points to consider market making profitable.
    pub min_spread_bps: i64,
    /// Maximum inventory per symbol (contracts).
    pub max_inventory: i64,
    /// Hedge fill timeout (milliseconds). If hedge doesn't fill within this time,
    /// close the maker position at market.
    pub hedge_timeout_ms: u64,
    /// Fast poll interval (milliseconds) for fill detection.
    pub fill_poll_interval_ms: u64,
    /// Maximum retries for hedge orders.
    pub max_hedge_retries: u32,
}

impl Default for CrossExchangeMMConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: 3,
            max_inventory: 3,
            hedge_timeout_ms: 100,
            fill_poll_interval_ms: 500,
            max_hedge_retries: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-Exchange Market Maker
// ---------------------------------------------------------------------------

/// Cross-exchange market maker that posts passive orders and hedges fills.
pub struct CrossExchangeMarketMaker {
    config: CrossExchangeMMConfig,
    /// Exchange where we post passive maker orders (typically wider spread).
    maker_exchange: ExchangeId,
    /// Exchange where we hedge fills immediately (typically more liquid).
    hedge_exchange: ExchangeId,
    /// Active maker orders by order_id.
    active_maker_orders: HashMap<String, MakerOrder>,
    /// Hedged position pairs.
    hedge_positions: Vec<HedgePosition>,
    /// Current inventory per symbol (positive = long, negative = short).
    inventory: HashMap<String, i64>,
    /// Last tick timestamp for polling.
    last_tick_ns: u64,
    /// Is market making paused.
    paused: bool,
    /// Pause reason.
    pause_reason: Option<String>,
}

impl CrossExchangeMarketMaker {
    /// Create a new cross-exchange market maker.
    pub fn new(
        config: CrossExchangeMMConfig,
        maker_exchange: ExchangeId,
        hedge_exchange: ExchangeId,
    ) -> Self {
        Self {
            config,
            maker_exchange,
            hedge_exchange,
            active_maker_orders: HashMap::new(),
            hedge_positions: Vec::new(),
            inventory: HashMap::new(),
            last_tick_ns: 0,
            paused: false,
            pause_reason: None,
        }
    }

    /// Create with default config (Gate.io maker, Binance hedge).
    pub fn with_defaults() -> Self {
        Self::new(
            CrossExchangeMMConfig::default(),
            ExchangeId::GateIo,  // Maker (wider spreads)
            ExchangeId::Binance, // Hedge (most liquid)
        )
    }

    /// Set the maker exchange.
    pub fn set_maker_exchange(&mut self, exchange: ExchangeId) {
        self.maker_exchange = exchange;
    }

    /// Set the hedge exchange.
    pub fn set_hedge_exchange(&mut self, exchange: ExchangeId) {
        self.hedge_exchange = exchange;
    }

    /// Check if market making is currently paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Pause market making with a reason.
    pub fn pause(&mut self, reason: &str) {
        self.paused = true;
        self.pause_reason = Some(reason.to_string());
        warn!("[cross-mm] Market making PAUSED: {}", reason);
    }

    /// Resume market making.
    pub fn resume(&mut self) {
        self.paused = false;
        self.pause_reason = None;
        info!("[cross-mm] Market making RESUMED");
    }

    /// Get current inventory for a symbol.
    pub fn get_inventory(&self, symbol: &str) -> i64 {
        self.inventory.get(symbol).copied().unwrap_or(0)
    }

    /// Check if inventory limit is reached for a symbol.
    pub fn inventory_limit_reached(&self, symbol: &str) -> bool {
        self.get_inventory(symbol).abs() >= self.config.max_inventory
    }

    /// Find the exchange with the widest spread from the global book.
    pub fn find_widest_spread_exchange(
        &self,
        global_book: &SharedGlobalBook,
    ) -> Option<(ExchangeId, i64)> {
        let book = global_book.read();
        let mut widest = None;
        let mut max_spread = 0i64;

        for exchange in ExchangeId::all() {
            if let Some(snap) = book.get_exchange_snapshot(exchange) {
                if snap.best_bid_fp > 0 && snap.best_ask_fp > snap.best_bid_fp {
                    let mid = (snap.best_bid_fp + snap.best_ask_fp) / 2;
                    let spread_bps = ((snap.best_ask_fp - snap.best_bid_fp) * 10000) / mid.max(1);
                    if spread_bps > max_spread {
                        max_spread = spread_bps;
                        widest = Some((exchange, spread_bps));
                    }
                }
            }
        }

        widest
    }

    /// Generate maker order intents for a symbol.
    ///
    /// Places a passive BUY limit at best_bid + 1 tick and
    /// a passive SELL limit at best_ask - 1 tick.
    pub fn generate_maker_orders(
        &self,
        symbol: &str,
        global_book: &SharedGlobalBook,
        tick_size: f64,
    ) -> Vec<OrderIntent> {
        if self.paused {
            return Vec::new();
        }

        let book = global_book.read();
        let maker_snap = book.get_exchange_snapshot(self.maker_exchange);

        let Some(snap) = maker_snap else {
            return Vec::new();
        };

        // Check if spread is profitable
        let spread_bps = book.global_spread_bps().unwrap_or(0);
        if spread_bps < self.config.min_spread_bps {
            debug!(
                "[cross-mm] Spread too narrow: {} bps < {} bps min — skipping",
                spread_bps, self.config.min_spread_bps
            );
            return Vec::new();
        }

        // Check if any exchange book is stale
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        if book.is_exchange_stale(self.maker_exchange, now_ns) ||
           book.is_exchange_stale(self.hedge_exchange, now_ns) {
            debug!("[cross-mm] Exchange book stale — pausing market making");
            return Vec::new();
        }

        let best_bid = snap.best_bid_fp as f64 / 1e8;
        let best_ask = snap.best_ask_fp as f64 / 1e8;

        let mut orders = Vec::new();
        let inventory = self.get_inventory(symbol);

        // BUY order at best_bid + 1 tick (if not at max long inventory)
        if inventory < self.config.max_inventory {
            let buy_price = best_bid + tick_size;
            orders.push(OrderIntent {
                symbol: symbol.to_string(),
                side: OrderSide::Buy,
                size: 1,
                order_type: OrderType::PostOnly,
                price: Some(buy_price),
                reduce_only: false,
                leverage: None,
                time_in_force: "poc".to_string(),
                slippage_cap_pct: None,
                placement: PlacementType::AtBest,
                stop_loss: None,
                take_profit: None,
                confidence: 0.0,
                signal_tag: "cross_mm_buy".to_string(),
                min_fill_size: None,
                strategy_name: "cross_exchange_mm".to_string(),
            });
        }

        // SELL order at best_ask - 1 tick (if not at max short inventory)
        if inventory > -self.config.max_inventory {
            let sell_price = best_ask - tick_size;
            orders.push(OrderIntent {
                symbol: symbol.to_string(),
                side: OrderSide::Sell,
                size: 1,
                order_type: OrderType::PostOnly,
                price: Some(sell_price),
                reduce_only: false,
                leverage: None,
                time_in_force: "poc".to_string(),
                slippage_cap_pct: None,
                placement: PlacementType::AtBest,
                stop_loss: None,
                take_profit: None,
                confidence: 0.0,
                signal_tag: "cross_mm_sell".to_string(),
                min_fill_size: None,
                strategy_name: "cross_exchange_mm".to_string(),
            });
        }

        orders
    }

    /// Track a new maker order.
    pub fn track_maker_order(&mut self, order: MakerOrder) {
        self.active_maker_orders.insert(order.order_id.clone(), order);
    }

    /// Generate a hedge order for a filled maker order.
    pub fn generate_hedge_order(&self, maker_fill: &MakerOrder, fill_size: i64) -> OrderIntent {
        let hedge_side = match maker_fill.side {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        };

        OrderIntent {
            symbol: maker_fill.symbol.clone(),
            side: hedge_side,
            size: fill_size,
            order_type: OrderType::Market,
            price: Some(maker_fill.price), // Reference price
            reduce_only: false,
            leverage: None,
            time_in_force: "ioc".to_string(),
            slippage_cap_pct: Some(0.002), // 0.2% max slippage
            placement: PlacementType::AtBest,
            stop_loss: None,
            take_profit: None,
            confidence: 0.0,
            signal_tag: "cross_mm_hedge".to_string(),
            min_fill_size: None,
            strategy_name: "cross_exchange_mm".to_string(),
        }
    }

    /// Process a maker fill and update inventory.
    pub fn on_maker_fill(
        &mut self,
        order_id: &str,
        fill_size: i64,
        fill_price: f64,
    ) -> Option<OrderIntent> {
        // Snapshot order for hedge generation without holding mutable borrow.
        let (inventory_change, hedge) = {
            let order_view = self.active_maker_orders.get(order_id)?;
            let inventory_change = match order_view.side {
                OrderSide::Buy => fill_size,
                OrderSide::Sell => -fill_size,
            };
            let hedge = self.generate_hedge_order(order_view, fill_size);
            (inventory_change, hedge)
        };

        let order = self.active_maker_orders.get_mut(order_id)?;
        order.filled_size += fill_size;
        
        // Update inventory
        *self.inventory.entry(order.symbol.clone()).or_insert(0) += inventory_change;

        // Update status
        if order.filled_size >= order.original_size {
            order.status = MakerOrderStatus::Filled;
        } else {
            order.status = MakerOrderStatus::PartiallyFilled;
        }

        info!(
            "[cross-mm] Maker fill: {} {} {} @ {:.4} — hedging on {}",
            order.symbol,
            if order.side == OrderSide::Buy { "BUY" } else { "SELL" },
            fill_size,
            fill_price,
            self.hedge_exchange.name()
        );

        Some(hedge)
    }

    /// Record a successful hedge and calculate realized PnL.
    pub fn on_hedge_fill(
        &mut self,
        symbol: &str,
        maker_side: OrderSide,
        maker_price: f64,
        hedge_price: f64,
        size: i64,
    ) {
        // Calculate PnL: for a maker BUY at price P, hedge SELL at price H
        // PnL = (H - P) * size for buys, (P - H) * size for sells
        let pnl = match maker_side {
            OrderSide::Buy => (hedge_price - maker_price) * size as f64,
            OrderSide::Sell => (maker_price - hedge_price) * size as f64,
        };

        // Update inventory (hedge reverses the maker position)
        let inventory_change = match maker_side {
            OrderSide::Buy => -size,
            OrderSide::Sell => size,
        };
        *self.inventory.entry(symbol.to_string()).or_insert(0) += inventory_change;

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.hedge_positions.push(HedgePosition {
            symbol: symbol.to_string(),
            maker_exchange: self.maker_exchange,
            hedge_exchange: self.hedge_exchange,
            maker_side,
            maker_fill_price: maker_price,
            hedge_fill_price: hedge_price,
            size,
            opened_ns: now_ns,
            realized_pnl: pnl,
        });

        info!(
            "[cross-mm] Hedge complete: {} maker@{:.4} hedge@{:.4} PnL=${:.4}",
            symbol, maker_price, hedge_price, pnl
        );
    }

    /// Clean up completed and cancelled orders.
    pub fn cleanup_orders(&mut self) {
        self.active_maker_orders.retain(|_, order| {
            !matches!(order.status, MakerOrderStatus::Filled | MakerOrderStatus::Cancelled)
        });
    }

    /// Get total realized PnL.
    pub fn total_pnl(&self) -> f64 {
        self.hedge_positions.iter().map(|p| p.realized_pnl).sum()
    }

    /// Get count of active maker orders.
    pub fn active_order_count(&self) -> usize {
        self.active_maker_orders.len()
    }

    /// Serialize state to JSON for dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "maker_exchange": self.maker_exchange.name(),
            "hedge_exchange": self.hedge_exchange.name(),
            "active_orders": self.active_maker_orders.len(),
            "hedged_positions": self.hedge_positions.len(),
            "total_pnl": self.total_pnl(),
            "inventory": self.inventory,
            "config": {
                "min_spread_bps": self.config.min_spread_bps,
                "max_inventory": self.config.max_inventory,
                "hedge_timeout_ms": self.config.hedge_timeout_ms,
            }
        })
    }
}

impl Default for CrossExchangeMarketMaker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_maker_creation() {
        let mm = CrossExchangeMarketMaker::with_defaults();
        assert_eq!(mm.maker_exchange, ExchangeId::GateIo);
        assert_eq!(mm.hedge_exchange, ExchangeId::Binance);
        assert!(!mm.is_paused());
    }

    #[test]
    fn test_inventory_tracking() {
        let mut mm = CrossExchangeMarketMaker::with_defaults();
        assert_eq!(mm.get_inventory("BTC_USDT"), 0);
        assert!(!mm.inventory_limit_reached("BTC_USDT"));
    }

    #[test]
    fn test_pause_resume() {
        let mut mm = CrossExchangeMarketMaker::with_defaults();
        mm.pause("Test pause");
        assert!(mm.is_paused());
        mm.resume();
        assert!(!mm.is_paused());
    }
}
