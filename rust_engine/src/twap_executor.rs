//! TWAP/Iceberg Order Execution Engine — Upgrade 3.
//!
//! Splits large orders into N time-sliced sub-orders to reduce market impact.
//! Supports both TWAP (equal time intervals) and Iceberg (show-only fraction) modes.

use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn, error};

/// Represents a single slice of a TWAP order
#[derive(Debug, Clone)]
pub struct TwapSlice {
    pub symbol: String,
    pub side: u8,  // 0 = buy, 1 = sell
    pub size: i64,
    pub price: f64,
    pub scheduled_at: Instant,
    pub executed: bool,
    pub fill_price: Option<f64>,
}

impl TwapSlice {
    pub fn new(symbol: String, side: u8, size: i64, price: f64, scheduled_at: Instant) -> Self {
        Self {
            symbol,
            side,
            size,
            price,
            scheduled_at,
            executed: false,
            fill_price: None,
        }
    }

    pub fn is_ready(&self, now: Instant) -> bool {
        !self.executed && now >= self.scheduled_at
    }

    pub fn mark_executed(&mut self, fill_price: f64) {
        self.executed = true;
        self.fill_price = Some(fill_price);
    }
}

/// Represents a complete TWAP order with all its slices
#[derive(Debug)]
pub struct TwapOrder {
    pub order_id: String,
    pub symbol: String,
    pub total_size: i64,
    pub remaining_size: i64,
    pub slices: Vec<TwapSlice>,
    pub arrival_price: f64,
    pub vwap_sum: f64,      // Sum of (fill_price * filled_size)
    pub vwap_qty: f64,      // Total filled quantity
    pub created_at: Instant,
    pub adverse_threshold_pct: f64,
    pub side: u8,           // 0 = buy, 1 = sell
    pub is_complete: bool,
    pub cancelled: bool,
}

impl TwapOrder {
    pub fn new(
        order_id: String,
        symbol: String,
        side: u8,
        total_size: i64,
        arrival_price: f64,
        adverse_threshold_pct: f64,
    ) -> Self {
        Self {
            order_id,
            symbol,
            total_size,
            remaining_size: total_size,
            slices: Vec::new(),
            arrival_price,
            vwap_sum: 0.0,
            vwap_qty: 0.0,
            created_at: Instant::now(),
            adverse_threshold_pct,
            side,
            is_complete: false,
            cancelled: false,
        }
    }

    /// Check if price has moved adversely beyond threshold
    pub fn is_adverse_price(&self, current_price: f64) -> bool {
        let price_change_pct = if self.side == 0 {
            // Buy order: adverse if price moved up significantly
            (current_price - self.arrival_price) / self.arrival_price
        } else {
            // Sell order: adverse if price moved down significantly
            (self.arrival_price - current_price) / self.arrival_price
        };
        
        price_change_pct > self.adverse_threshold_pct / 100.0
    }

    /// Record a fill and update VWAP calculation
    pub fn record_fill(&mut self, slice_idx: usize, fill_price: f64, filled_size: i64) {
        if slice_idx >= self.slices.len() {
            error!("Invalid slice index {} for order {}", slice_idx, self.order_id);
            return;
        }

        let slice = &mut self.slices[slice_idx];
        if slice.executed {
            warn!("Slice {} already executed for order {}", slice_idx, self.order_id);
            return;
        }

        slice.mark_executed(fill_price);
        
        // Update VWAP calculation
        self.vwap_sum += fill_price * filled_size as f64;
        self.vwap_qty += filled_size as f64;
        self.remaining_size -= filled_size;

        // Check if order is complete
        if self.remaining_size <= 0 || self.slices.iter().all(|s| s.executed || s.scheduled_at > Instant::now()) {
            self.is_complete = true;
        }

        debug!(
            "Fill recorded for order {}: slice={}, price={}, size={}, remaining={}",
            self.order_id, slice_idx, fill_price, filled_size, self.remaining_size
        );
    }

    /// Calculate current VWAP
    pub fn current_vwap(&self) -> Option<f64> {
        if self.vwap_qty > 0.0 {
            Some(self.vwap_sum / self.vwap_qty)
        } else {
            None
        }
    }

    /// Cancel all remaining unexecuted slices
    pub fn cancel_remaining_slices(&mut self) {
        self.cancelled = true;
        self.is_complete = true;
        
        let cancelled_count = self.slices.iter().filter(|s| !s.executed).count();
        warn!(
            "Cancelled {} remaining slices for order {} due to adverse price movement",
            cancelled_count, self.order_id
        );
    }
}

/// Execution quality metrics for completed orders
#[derive(Debug, Clone)]
pub struct ExecutionQuality {
    pub vwap: f64,
    pub arrival_price: f64,
    pub slippage_bps: f64,
    pub fill_rate: f64,
    pub num_slices_executed: u32,
    pub num_slices_cancelled: u32,
    pub duration_ms: u64,
}

impl ExecutionQuality {
    pub fn from_order(order: &TwapOrder) -> Self {
        let vwap = order.current_vwap().unwrap_or(order.arrival_price);
        let slippage_bps = if order.side == 0 {
            // Buy: positive slippage if we paid more than arrival price
            ((vwap - order.arrival_price) / order.arrival_price) * 10000.0
        } else {
            // Sell: positive slippage if we received less than arrival price
            ((order.arrival_price - vwap) / order.arrival_price) * 10000.0
        };

        let num_slices_executed = order.slices.iter().filter(|s| s.executed).count() as u32;
        let num_slices_cancelled = order.slices.len() as u32 - num_slices_executed;
        let fill_rate = if order.total_size > 0 {
            (order.total_size - order.remaining_size) as f64 / order.total_size as f64
        } else {
            0.0
        };

        Self {
            vwap,
            arrival_price: order.arrival_price,
            slippage_bps,
            fill_rate,
            num_slices_executed,
            num_slices_cancelled,
            duration_ms: order.created_at.elapsed().as_millis() as u64,
        }
    }
}

/// Main TWAP execution engine
#[derive(Debug)]
pub struct TwapExecutor {
    pub active_orders: Vec<TwapOrder>,
    pub max_concurrent: usize,
}

impl TwapExecutor {
    /// Create a new TWAP executor
    pub fn new() -> Self {
        Self {
            active_orders: Vec::new(),
            max_concurrent: 5,
        }
    }

    /// Create a new TWAP order and add it to the active orders
    pub fn create_twap_order(
        &mut self,
        symbol: String,
        side: u8,
        total_size: i64,
        num_slices: usize,
        interval_ms: u64,
        arrival_price: f64,
        adverse_threshold_pct: f64,
    ) -> Result<&TwapOrder, String> {
        if self.active_orders.len() >= self.max_concurrent {
            return Err("Maximum concurrent orders reached".to_string());
        }

        if num_slices == 0 || total_size <= 0 {
            return Err("Invalid order parameters".to_string());
        }

        let order_id = format!("TWAP_{}_{}", symbol, Instant::now().elapsed().as_nanos());
        let mut order = TwapOrder::new(order_id, symbol.clone(), side, total_size, arrival_price, adverse_threshold_pct);

        // Calculate slice size (distribute remainder across first few slices)
        let base_slice_size = total_size / num_slices as i64;
        let remainder = total_size % num_slices as i64;
        
        let now = Instant::now();
        
        // Create slices with scheduled execution times
        for i in 0..num_slices {
            let slice_size = if i < remainder as usize {
                base_slice_size + 1
            } else {
                base_slice_size
            };
            
            let scheduled_at = now + std::time::Duration::from_millis(i as u64 * interval_ms);
            let slice = TwapSlice::new(
                symbol.clone(),
                side,
                slice_size,
                arrival_price, // Initial reference price
                scheduled_at,
            );
            
            order.slices.push(slice);
        }

        info!(
            "Created TWAP order {}: {} {} slices of {} shares each, interval={}ms",
            order.order_id, num_slices, symbol, base_slice_size, interval_ms
        );

        self.active_orders.push(order);
        Ok(self.active_orders.last().unwrap())
    }

    /// Main execution loop - returns slices ready to execute
    pub fn tick(&mut self, current_prices: &HashMap<String, f64>) -> Vec<TwapSlice> {
        let now = Instant::now();
        let mut ready_slices = Vec::new();

        for order in &mut self.active_orders {
            if order.is_complete || order.cancelled {
                continue;
            }

            // Check for adverse price movement
            if let Some(&current_price) = current_prices.get(&order.symbol) {
                if order.is_adverse_price(current_price) {
                    order.cancel_remaining_slices();
                    continue;
                }
            }

            // Find slices ready for execution
            for (idx, slice) in order.slices.iter().enumerate() {
                if slice.is_ready(now) {
                    debug!(
                        "Slice {} ready for execution: {} {} {} @ {}",
                        idx, slice.symbol, if slice.side == 0 { "BUY" } else { "SELL" }, slice.size, slice.price
                    );
                    ready_slices.push(slice.clone());
                }
            }
        }

        ready_slices
    }

    /// Record a fill for a specific slice
    pub fn record_fill(&mut self, order_id: &str, slice_idx: usize, fill_price: f64, filled_size: i64) {
        if let Some(order) = self.active_orders.iter_mut().find(|o| o.order_id == order_id) {
            order.record_fill(slice_idx, fill_price, filled_size);
        } else {
            error!("Order {} not found for fill recording", order_id);
        }
    }

    /// Get execution quality metrics for a completed order
    pub fn get_quality(&self, order_id: &str) -> Option<ExecutionQuality> {
        self.active_orders
            .iter()
            .find(|o| o.order_id == order_id && o.is_complete)
            .map(ExecutionQuality::from_order)
    }

    /// Cancel an active order
    #[allow(dead_code)]
    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        if let Some(order) = self.active_orders.iter_mut().find(|o| o.order_id == order_id) {
            if !order.is_complete {
                order.cancel_remaining_slices();
                info!("Cancelled TWAP order {}", order_id);
                return true;
            }
        }
        false
    }

    /// Get number of active orders
    pub fn active_count(&self) -> usize {
        self.active_orders.iter().filter(|o| !o.is_complete).count()
    }

    /// Remove completed orders from memory
    pub fn cleanup_completed(&mut self) -> usize {
        let initial_count = self.active_orders.len();
        self.active_orders.retain(|order| !order.is_complete);
        let cleaned_count = initial_count - self.active_orders.len();
        
        if cleaned_count > 0 {
            debug!("Cleaned up {} completed TWAP orders", cleaned_count);
        }
        
        cleaned_count
    }

    /// Get summary statistics
    #[allow(dead_code)]
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        let total_orders = self.active_orders.len();
        let active_orders = self.active_count();
        let completed_orders = total_orders - active_orders;
        
        stats.insert("total_orders".to_string(), total_orders as f64);
        stats.insert("active_orders".to_string(), active_orders as f64);
        stats.insert("completed_orders".to_string(), completed_orders as f64);
        
        // Calculate average fill rates and slippage for completed orders
        let completed: Vec<_> = self.active_orders.iter()
            .filter(|o| o.is_complete)
            .collect();
            
        if !completed.is_empty() {
            let avg_fill_rate = completed.iter()
                .map(|o| (o.total_size - o.remaining_size) as f64 / o.total_size as f64)
                .sum::<f64>() / completed.len() as f64;
                
            let avg_slippage = completed.iter()
                .filter_map(|o| o.current_vwap())
                .zip(completed.iter().map(|o| o.arrival_price))
                .map(|(vwap, arrival)| ((vwap - arrival) / arrival * 10000.0).abs())
                .sum::<f64>() / completed.len() as f64;
                
            stats.insert("avg_fill_rate".to_string(), avg_fill_rate);
            stats.insert("avg_slippage_bps".to_string(), avg_slippage);
        }
        
        stats
    }
}

impl Default for TwapExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_twap_order_creation() {
        let mut executor = TwapExecutor::new();
        
        let order = executor.create_twap_order(
            "AAPL".to_string(),
            0, // buy
            1000,
            5,
            1000, // 1 second intervals
            150.0,
            0.5,
        ).unwrap();
        
        assert_eq!(order.total_size, 1000);
        assert_eq!(order.slices.len(), 5);
        assert_eq!(order.side, 0);
        assert!(!order.is_complete);
    }

    #[test]
    fn test_slice_scheduling() {
        let mut executor = TwapExecutor::new();
        
        executor.create_twap_order(
            "AAPL".to_string(),
            0,
            1000,
            3,
            100, // 100ms intervals
            150.0,
            0.5,
        ).unwrap();

        // Initially no slices should be ready
        let prices = HashMap::from([("AAPL".to_string(), 150.0)]);
        let ready = executor.tick(&prices);
        assert!(ready.len() <= 1); // Only first slice might be ready immediately

        // Wait and check again
        thread::sleep(Duration::from_millis(150));
        let ready = executor.tick(&prices);
        assert!(ready.len() >= 1);
    }

    #[test]
    fn test_adverse_price_cancellation() {
        let mut executor = TwapExecutor::new();
        
        executor.create_twap_order(
            "AAPL".to_string(),
            0, // buy order
            1000,
            5,
            1000,
            100.0, // arrival price
            0.5,   // 0.5% threshold
        ).unwrap();

        // Price moves up by 1% - should trigger cancellation
        let prices = HashMap::from([("AAPL".to_string(), 101.0)]);
        executor.tick(&prices);
        
        let order = &executor.active_orders[0];
        assert!(order.cancelled);
        assert!(order.is_complete);
    }

    #[test]
    fn test_fill_recording_and_vwap() {
        let mut executor = TwapExecutor::new();
        
        let order = executor.create_twap_order(
            "AAPL".to_string(),
            0,
            1000,
            2,
            1000,
            100.0,
            0.5,
        ).unwrap();
        
        let order_id = order.order_id.clone();
        
        // Record fills
        executor.record_fill(&order_id, 0, 99.5, 500);
        executor.record_fill(&order_id, 1, 100.5, 500);
        
        let order = &executor.active_orders[0];
        let vwap = order.current_vwap().unwrap();
        assert!((vwap - 100.0).abs() < 0.01); // Should be close to 100.0
        assert!(order.is_complete);
    }
}