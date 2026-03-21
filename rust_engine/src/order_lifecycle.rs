//! Order Lifecycle Tracker — Institutional-Grade Order State Management
//!
//! Tracks every order from submission through fill/cancel/reject with full
//! state history. Enables:
//!   - Position reconciliation with exchange REST state
//!   - Partial fill tracking and average price computation
//!   - Order state recovery after WebSocket reconnection
//!   - Audit trail for compliance and debugging
//!
//! # Thread Safety
//! All methods use interior mutability via parking_lot::Mutex for
//! thread-safe access from both the execution router and telemetry threads.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use parking_lot::Mutex;
use tracing::{info, warn, error, debug};

/// Unique order identifier (exchange-assigned or client-generated).
pub type OrderId = String;

/// Fill record for a single execution against an order.
#[derive(Debug, Clone)]
pub struct Fill {
    pub fill_id: String,
    pub price: f64,
    pub quantity: f64,
    pub fee: f64,
    pub fee_currency: String,
    pub timestamp_us: u64,
    pub is_maker: bool,
}

/// Current state of an order in its lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderState {
    /// Order submitted but not yet acknowledged by exchange.
    Pending,
    /// Order acknowledged by exchange, resting in book.
    Open,
    /// Order partially filled.
    PartiallyFilled {
        filled_qty: f64,
        remaining_qty: f64,
        avg_fill_price: f64,
    },
    /// Order fully filled.
    Filled {
        total_qty: f64,
        avg_fill_price: f64,
        total_fees: f64,
    },
    /// Order cancelled (by user or system).
    Cancelled {
        reason: String,
        filled_qty: f64,
    },
    /// Order rejected by exchange.
    Rejected {
        reason: String,
    },
    /// Order expired (GTT/IOC timeout).
    Expired {
        filled_qty: f64,
    },
}

impl std::fmt::Display for OrderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderState::Pending => write!(f, "PENDING"),
            OrderState::Open => write!(f, "OPEN"),
            OrderState::PartiallyFilled { filled_qty, .. } => {
                write!(f, "PARTIAL({:.4})", filled_qty)
            }
            OrderState::Filled { total_qty, avg_fill_price, .. } => {
                write!(f, "FILLED({:.4}@{:.4})", total_qty, avg_fill_price)
            }
            OrderState::Cancelled { reason, .. } => write!(f, "CANCELLED({})", reason),
            OrderState::Rejected { reason } => write!(f, "REJECTED({})", reason),
            OrderState::Expired { .. } => write!(f, "EXPIRED"),
        }
    }
}

/// State transition record for audit trail.
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: OrderState,
    pub to: OrderState,
    pub timestamp_us: u64,
    pub reason: Option<String>,
}

/// Complete order lifecycle record.
#[derive(Debug, Clone)]
pub struct OrderLifecycle {
    /// Exchange-assigned order ID.
    pub order_id: OrderId,
    /// Client-generated order ID (for matching before exchange ack).
    pub client_order_id: String,
    /// Symbol (e.g., "BTC_USDT").
    pub symbol: String,
    /// Side: "buy" or "sell".
    pub side: String,
    /// Order type: "limit", "market", "post_only".
    pub order_type: String,
    /// Original requested quantity.
    pub original_qty: f64,
    /// Original requested price (0 for market orders).
    pub original_price: f64,
    /// Current order state.
    pub state: OrderState,
    /// All fills received for this order.
    pub fills: Vec<Fill>,
    /// Complete state transition history.
    pub state_history: Vec<StateTransition>,
    /// Strategy that generated this order.
    pub strategy_tag: String,
    /// Confidence score at time of signal (0.0-1.0).
    pub signal_confidence: f64,
    /// Time order was created locally.
    pub created_at_us: u64,
    /// Time order was last updated.
    pub updated_at_us: u64,
    /// Stop-loss price (if attached).
    pub stop_loss: Option<f64>,
    /// Take-profit price (if attached).
    pub take_profit: Option<f64>,
    /// Leverage used.
    pub leverage: Option<i32>,
}

impl OrderLifecycle {
    pub fn new(
        client_order_id: String,
        symbol: String,
        side: String,
        order_type: String,
        qty: f64,
        price: f64,
        strategy_tag: String,
        signal_confidence: f64,
    ) -> Self {
        let now_us = now_micros();
        Self {
            order_id: String::new(),
            client_order_id,
            symbol,
            side,
            order_type,
            original_qty: qty,
            original_price: price,
            state: OrderState::Pending,
            fills: Vec::new(),
            state_history: Vec::new(),
            strategy_tag,
            signal_confidence,
            created_at_us: now_us,
            updated_at_us: now_us,
            stop_loss: None,
            take_profit: None,
            leverage: None,
        }
    }

    /// Transition to a new state, recording the transition in history.
    pub fn transition_to(&mut self, new_state: OrderState, reason: Option<String>) {
        let now_us = now_micros();
        let transition = StateTransition {
            from: self.state.clone(),
            to: new_state.clone(),
            timestamp_us: now_us,
            reason,
        };
        self.state_history.push(transition);
        self.state = new_state;
        self.updated_at_us = now_us;
    }

    /// Record a fill and update the order state.
    pub fn record_fill(&mut self, fill: Fill) {
        let fill_qty = fill.quantity;
        let fill_price = fill.price;
        let fill_fee = fill.fee;
        self.fills.push(fill);

        // Compute new totals
        let total_filled: f64 = self.fills.iter().map(|f| f.quantity).sum();
        let total_fees: f64 = self.fills.iter().map(|f| f.fee).sum();
        let avg_price = if total_filled > 0.0 {
            self.fills.iter().map(|f| f.price * f.quantity).sum::<f64>() / total_filled
        } else {
            0.0
        };

        let remaining = self.original_qty - total_filled;

        if remaining <= 1e-10 {
            // Fully filled
            self.transition_to(
                OrderState::Filled {
                    total_qty: total_filled,
                    avg_fill_price: avg_price,
                    total_fees,
                },
                Some(format!("fill: {:.4}@{:.4} fee={:.6}", fill_qty, fill_price, fill_fee)),
            );
        } else {
            // Partially filled
            self.transition_to(
                OrderState::PartiallyFilled {
                    filled_qty: total_filled,
                    remaining_qty: remaining,
                    avg_fill_price: avg_price,
                },
                Some(format!("partial fill: {:.4}@{:.4}", fill_qty, fill_price)),
            );
        }
    }

    /// Get the average fill price across all fills.
    pub fn avg_fill_price(&self) -> f64 {
        let total_filled: f64 = self.fills.iter().map(|f| f.quantity).sum();
        if total_filled > 0.0 {
            self.fills.iter().map(|f| f.price * f.quantity).sum::<f64>() / total_filled
        } else {
            0.0
        }
    }

    /// Get total filled quantity.
    pub fn filled_qty(&self) -> f64 {
        self.fills.iter().map(|f| f.quantity).sum()
    }

    /// Get total fees paid.
    pub fn total_fees(&self) -> f64 {
        self.fills.iter().map(|f| f.fee).sum()
    }

    /// Check if this order is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            OrderState::Filled { .. }
                | OrderState::Cancelled { .. }
                | OrderState::Rejected { .. }
                | OrderState::Expired { .. }
        )
    }

    /// Duration from creation to last update.
    pub fn age_us(&self) -> u64 {
        self.updated_at_us.saturating_sub(self.created_at_us)
    }
}

/// Manages all order lifecycles with thread-safe access.
pub struct OrderLifecycleTracker {
    /// Active (non-terminal) orders indexed by client_order_id.
    active: Mutex<HashMap<String, OrderLifecycle>>,
    /// Recently completed orders (ring buffer, last 1000).
    completed: Mutex<Vec<OrderLifecycle>>,
    /// Exchange order ID -> client order ID mapping.
    exchange_id_map: Mutex<HashMap<String, String>>,
    /// Max completed orders to retain.
    max_completed: usize,
    /// Metrics
    total_orders: std::sync::atomic::AtomicU64,
    total_fills: std::sync::atomic::AtomicU64,
    total_rejections: std::sync::atomic::AtomicU64,
}

impl OrderLifecycleTracker {
    pub fn new(max_completed: usize) -> Self {
        Self {
            active: Mutex::new(HashMap::new()),
            completed: Mutex::new(Vec::with_capacity(max_completed)),
            exchange_id_map: Mutex::new(HashMap::new()),
            max_completed,
            total_orders: std::sync::atomic::AtomicU64::new(0),
            total_fills: std::sync::atomic::AtomicU64::new(0),
            total_rejections: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Register a new order (called when order is submitted).
    pub fn register_order(&self, order: OrderLifecycle) -> String {
        let coid = order.client_order_id.clone();
        self.total_orders.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug!("[lifecycle] Registered order: {} sym={} side={} qty={:.4}",
            coid, order.symbol, order.side, order.original_qty);
        self.active.lock().insert(coid.clone(), order);
        coid
    }

    /// Map exchange order ID to client order ID (called on order ack).
    pub fn map_exchange_id(&self, exchange_id: &str, client_order_id: &str) {
        self.exchange_id_map.lock()
            .insert(exchange_id.to_string(), client_order_id.to_string());
        if let Some(order) = self.active.lock().get_mut(client_order_id) {
            order.order_id = exchange_id.to_string();
            order.transition_to(OrderState::Open, Some("exchange ack".into()));
        }
    }

    /// Record a fill for an order.
    pub fn record_fill(&self, client_order_id: &str, fill: Fill) {
        self.total_fills.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut active = self.active.lock();
        if let Some(order) = active.get_mut(client_order_id) {
            order.record_fill(fill);
            info!("[lifecycle] Fill recorded: {} state={}", client_order_id, order.state);
            // Move to completed if terminal
            if order.is_terminal() {
                let completed_order = order.clone();
                drop(active);
                self.move_to_completed(client_order_id, completed_order);
            }
        } else {
            warn!("[lifecycle] Fill for unknown order: {}", client_order_id);
        }
    }

    /// Record a fill using exchange order ID.
    pub fn record_fill_by_exchange_id(&self, exchange_id: &str, fill: Fill) {
        let client_id = self.exchange_id_map.lock().get(exchange_id).cloned();
        if let Some(coid) = client_id {
            self.record_fill(&coid, fill);
        } else {
            warn!("[lifecycle] Fill for unknown exchange order: {}", exchange_id);
        }
    }

    /// Cancel an order.
    pub fn cancel_order(&self, client_order_id: &str, reason: &str) {
        let mut active = self.active.lock();
        if let Some(order) = active.get_mut(client_order_id) {
            let filled_qty = order.filled_qty();
            order.transition_to(
                OrderState::Cancelled {
                    reason: reason.to_string(),
                    filled_qty,
                },
                Some(reason.to_string()),
            );
            info!("[lifecycle] Order cancelled: {} reason={}", client_order_id, reason);
            let completed_order = order.clone();
            drop(active);
            self.move_to_completed(client_order_id, completed_order);
        }
    }

    /// Reject an order.
    pub fn reject_order(&self, client_order_id: &str, reason: &str) {
        self.total_rejections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut active = self.active.lock();
        if let Some(order) = active.get_mut(client_order_id) {
            order.transition_to(
                OrderState::Rejected { reason: reason.to_string() },
                Some(reason.to_string()),
            );
            warn!("[lifecycle] Order rejected: {} reason={}", client_order_id, reason);
            let completed_order = order.clone();
            drop(active);
            self.move_to_completed(client_order_id, completed_order);
        }
    }

    /// Get active order count.
    pub fn active_count(&self) -> usize {
        self.active.lock().len()
    }

    /// Get all active orders (snapshot).
    pub fn get_active_orders(&self) -> Vec<OrderLifecycle> {
        self.active.lock().values().cloned().collect()
    }

    /// Get metrics summary.
    pub fn get_metrics(&self) -> LifecycleMetrics {
        LifecycleMetrics {
            total_orders: self.total_orders.load(std::sync::atomic::Ordering::Relaxed),
            total_fills: self.total_fills.load(std::sync::atomic::Ordering::Relaxed),
            total_rejections: self.total_rejections.load(std::sync::atomic::Ordering::Relaxed),
            active_orders: self.active_count(),
            completed_orders: self.completed.lock().len(),
        }
    }

    /// Reconcile local state with exchange REST response.
    /// Returns a list of discrepancies found.
    pub fn reconcile_with_exchange(
        &self,
        exchange_orders: &[(String, f64, String)], // (order_id, filled_qty, status)
    ) -> Vec<ReconciliationDiscrepancy> {
        let active = self.active.lock();
        let exchange_map = self.exchange_id_map.lock();
        let mut discrepancies = Vec::new();

        for (ex_id, ex_filled, ex_status) in exchange_orders {
            if let Some(coid) = exchange_map.get(ex_id) {
                if let Some(local_order) = active.get(coid) {
                    let local_filled = local_order.filled_qty();
                    let qty_diff = (ex_filled - local_filled).abs();
                    if qty_diff > 1e-8 {
                        discrepancies.push(ReconciliationDiscrepancy {
                            order_id: ex_id.clone(),
                            client_order_id: coid.clone(),
                            discrepancy_type: DiscrepancyType::QuantityMismatch {
                                local_filled,
                                exchange_filled: *ex_filled,
                            },
                        });
                    }
                }
            } else if ex_status != "closed" && ex_status != "cancelled" {
                // Exchange has an order we don't know about
                discrepancies.push(ReconciliationDiscrepancy {
                    order_id: ex_id.clone(),
                    client_order_id: String::new(),
                    discrepancy_type: DiscrepancyType::UnknownOrder,
                });
            }
        }

        if !discrepancies.is_empty() {
            error!("[lifecycle] Reconciliation found {} discrepancies", discrepancies.len());
        }

        discrepancies
    }

    /// Move a terminal order from active to completed.
    fn move_to_completed(&self, client_order_id: &str, order: OrderLifecycle) {
        self.active.lock().remove(client_order_id);
        let mut completed = self.completed.lock();
        if completed.len() >= self.max_completed {
            completed.remove(0); // Drop oldest
        }
        completed.push(order);
    }
}

/// Reconciliation discrepancy types.
#[derive(Debug, Clone)]
pub enum DiscrepancyType {
    QuantityMismatch {
        local_filled: f64,
        exchange_filled: f64,
    },
    UnknownOrder,
    MissingFromExchange,
    StateMismatch {
        local_state: String,
        exchange_state: String,
    },
}

/// A single reconciliation discrepancy.
#[derive(Debug, Clone)]
pub struct ReconciliationDiscrepancy {
    pub order_id: String,
    pub client_order_id: String,
    pub discrepancy_type: DiscrepancyType,
}

/// Lifecycle metrics for monitoring.
#[derive(Debug, Clone)]
pub struct LifecycleMetrics {
    pub total_orders: u64,
    pub total_fills: u64,
    pub total_rejections: u64,
    pub active_orders: usize,
    pub completed_orders: usize,
}

/// Get current timestamp in microseconds.
pub fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle_basic() {
        let tracker = OrderLifecycleTracker::new(100);
        let order = OrderLifecycle::new(
            "test-001".into(), "BTC_USDT".into(), "buy".into(),
            "limit".into(), 1.0, 50000.0, "imbalance".into(), 0.85,
        );
        tracker.register_order(order);
        assert_eq!(tracker.active_count(), 1);

        tracker.map_exchange_id("EX-123", "test-001");

        let fill = Fill {
            fill_id: "f1".into(),
            price: 49990.0,
            quantity: 0.5,
            fee: 0.01,
            fee_currency: "USDT".into(),
            timestamp_us: now_micros(),
            is_maker: true,
        };
        tracker.record_fill("test-001", fill);
        assert_eq!(tracker.active_count(), 1); // Still active (partial)

        let fill2 = Fill {
            fill_id: "f2".into(),
            price: 50010.0,
            quantity: 0.5,
            fee: 0.01,
            fee_currency: "USDT".into(),
            timestamp_us: now_micros(),
            is_maker: false,
        };
        tracker.record_fill("test-001", fill2);
        assert_eq!(tracker.active_count(), 0); // Moved to completed
    }

    #[test]
    fn test_order_rejection() {
        let tracker = OrderLifecycleTracker::new(100);
        let order = OrderLifecycle::new(
            "test-002".into(), "ETH_USDT".into(), "sell".into(),
            "limit".into(), 10.0, 3000.0, "momentum".into(), 0.7,
        );
        tracker.register_order(order);
        tracker.reject_order("test-002", "insufficient_margin");
        assert_eq!(tracker.active_count(), 0);

        let metrics = tracker.get_metrics();
        assert_eq!(metrics.total_rejections, 1);
    }
}
