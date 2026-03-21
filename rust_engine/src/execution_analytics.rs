//! Execution Quality Analytics — Phase 2 Feature 7.
//!
//! Tracks slippage, implementation shortfall, and market impact for all fills.
//! Provides real-time feedback to the execution router for order type selection
//! and position sizing adjustments.
//!
//! Metrics:
//! - **Slippage**: (fill_price - signal_price) / signal_price * 10000 (bps)
//! - **Implementation Shortfall**: (mid_after_1s - mid_at_signal) / mid_at_signal * 10000 (bps)
//! - **Market Impact**: (mid_after_1s - fill_price) / fill_price * 10000 (bps)
//!
//! Follows the pattern from trade_flow_analyzer.rs for ring buffer management.

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Execution Analytics
// ═══════════════════════════════════════════════════════════════════════════

/// Execution quality analytics tracker.
///
/// Maintains rolling windows of slippage, shortfall, and impact metrics
/// to provide real-time feedback on execution quality.
pub struct ExecutionAnalytics {
    /// Ring buffer of slippage values (in bps).
    slippage_history: VecDeque<f64>,
    /// Ring buffer of implementation shortfall values (in bps).
    shortfall_history: VecDeque<f64>,
    /// Ring buffer of market impact values (in bps).
    impact_history: VecDeque<f64>,
    /// Maximum number of fills to track.
    window_size: usize,
    /// Cached average slippage (bps).
    avg_slippage_bps: f64,
    /// Cached average shortfall (bps).
    avg_shortfall_bps: f64,
    /// Cached average impact (bps).
    avg_impact_bps: f64,
}

impl ExecutionAnalytics {
    /// Create a new execution analytics tracker.
    ///
    /// # Arguments
    /// * `window_size` — Number of recent fills to track (e.g., 1000)
    pub fn new(window_size: usize) -> Self {
        Self {
            slippage_history: VecDeque::with_capacity(window_size),
            shortfall_history: VecDeque::with_capacity(window_size),
            impact_history: VecDeque::with_capacity(window_size),
            window_size,
            avg_slippage_bps: 0.0,
            avg_shortfall_bps: 0.0,
            avg_impact_bps: 0.0,
        }
    }

    /// Record a fill and update metrics.
    ///
    /// # Arguments
    /// * `fill_price` — Actual fill price
    /// * `signal_price` — Price at signal generation time
    /// * `mid_at_signal` — Mid price when signal was generated
    /// * `mid_after_1s` — Mid price 1 second after fill
    /// * `size_usdt` — Order size in USDT (for weighted averages in future)
    pub fn record_fill(
        &mut self,
        fill_price: f64,
        signal_price: f64,
        mid_at_signal: f64,
        mid_after_1s: f64,
        _size_usdt: f64,
    ) {
        // Calculate slippage: (fill - signal) / signal * 10000
        let slippage_bps = if signal_price > 0.0 {
            ((fill_price - signal_price) / signal_price * 10000.0).abs()
        } else {
            0.0
        };

        // Calculate implementation shortfall: (mid_after - mid_at_signal) / mid_at_signal * 10000
        let shortfall_bps = if mid_at_signal > 0.0 {
            ((mid_after_1s - mid_at_signal) / mid_at_signal * 10000.0).abs()
        } else {
            0.0
        };

        // Calculate market impact: (mid_after - fill) / fill * 10000
        let impact_bps = if fill_price > 0.0 {
            ((mid_after_1s - fill_price) / fill_price * 10000.0).abs()
        } else {
            0.0
        };

        // Add to ring buffers
        if self.slippage_history.len() >= self.window_size {
            self.slippage_history.pop_front();
        }
        self.slippage_history.push_back(slippage_bps);

        if self.shortfall_history.len() >= self.window_size {
            self.shortfall_history.pop_front();
        }
        self.shortfall_history.push_back(shortfall_bps);

        if self.impact_history.len() >= self.window_size {
            self.impact_history.pop_front();
        }
        self.impact_history.push_back(impact_bps);

        // Update cached averages
        self.avg_slippage_bps = self.slippage_history.iter().sum::<f64>() / self.slippage_history.len() as f64;
        self.avg_shortfall_bps = self.shortfall_history.iter().sum::<f64>() / self.shortfall_history.len() as f64;
        self.avg_impact_bps = self.impact_history.iter().sum::<f64>() / self.impact_history.len() as f64;
    }

    /// Get average slippage in basis points.
    #[inline]
    pub fn get_avg_slippage_bps(&self) -> f64 {
        self.avg_slippage_bps
    }

    /// Get average implementation shortfall in basis points.
    #[inline]
    pub fn get_avg_shortfall_bps(&self) -> f64 {
        self.avg_shortfall_bps
    }

    /// Get average market impact in basis points.
    #[inline]
    pub fn get_avg_impact_bps(&self) -> f64 {
        self.avg_impact_bps
    }

    /// Check if we should use limit orders instead of market orders.
    ///
    /// Returns `true` if average slippage exceeds 5 bps.
    #[inline]
    pub fn should_use_limit_orders(&self) -> bool {
        self.avg_slippage_bps > 5.0
    }

    /// Check if we should reduce position size due to high market impact.
    ///
    /// Returns `true` if average impact exceeds 10 bps.
    #[inline]
    pub fn should_reduce_size(&self) -> bool {
        self.avg_impact_bps > 10.0
    }

    /// Check if the analytics are warmed up.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.slippage_history.len() >= self.window_size / 2
    }
}

impl Default for ExecutionAnalytics {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_calculation() {
        let mut analytics = ExecutionAnalytics::new(100);
        
        // Fill at 100.1, signal at 100.0 → 10 bps slippage
        analytics.record_fill(100.1, 100.0, 100.0, 100.0, 1000.0);
        
        assert!((analytics.get_avg_slippage_bps() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_impact_calculation() {
        let mut analytics = ExecutionAnalytics::new(100);
        
        // Fill at 100.0, mid moves to 100.2 → 20 bps impact
        analytics.record_fill(100.0, 100.0, 100.0, 100.2, 1000.0);
        
        assert!((analytics.get_avg_impact_bps() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_limit_order_recommendation() {
        let mut analytics = ExecutionAnalytics::new(10);
        
        // Add 10 fills with high slippage (10 bps each)
        for _ in 0..10 {
            analytics.record_fill(100.1, 100.0, 100.0, 100.0, 1000.0);
        }
        
        assert!(analytics.should_use_limit_orders());
    }

    #[test]
    fn test_size_reduction_recommendation() {
        let mut analytics = ExecutionAnalytics::new(10);
        
        // Add 10 fills with high impact (15 bps each)
        for _ in 0..10 {
            analytics.record_fill(100.0, 100.0, 100.0, 100.15, 1000.0);
        }
        
        assert!(analytics.should_reduce_size());
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut analytics = ExecutionAnalytics::new(5);
        
        // Add 10 fills (should only keep last 5)
        for i in 0..10 {
            let slippage = i as f64;
            analytics.record_fill(100.0 + slippage, 100.0, 100.0, 100.0, 1000.0);
        }
        
        assert_eq!(analytics.slippage_history.len(), 5);
        // Average should be (5+6+7+8+9)*10 / 5 = 70 bps
        assert!((analytics.get_avg_slippage_bps() - 70.0).abs() < 1.0);
    }
}
