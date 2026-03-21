//! Trade Flow Analyzer — FEATURE 4.
//!
//! Processes trade events (update_type=3 from SPSC ring) to compute:
//! 1. Buy/sell volume ratio over last 100 trades
//! 2. Large trade detection (>10x average size)
//! 3. Trade arrival rate acceleration using exponential moving average
//!
//! Follows the pattern from adverse_selection.rs:TradeRing (lines 50-150)
//! for fixed-size circular buffer with zero-allocation updates.

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Trade Event
// ═══════════════════════════════════════════════════════════════════════════

/// A trade event from the public trade tape.
#[derive(Debug, Clone, Copy)]
pub struct TradeEvent {
    /// Trade price.
    pub price: f64,
    /// Trade size (absolute value).
    pub size: f64,
    /// Side: 0 = buy (taker bought), 1 = sell (taker sold).
    pub side: u8,
    /// Nanosecond timestamp.
    pub timestamp_ns: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Trade Flow Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Metrics computed from recent trade flow.
#[derive(Debug, Clone, Copy)]
pub struct TradeFlowMetrics {
    /// Buy/sell volume ratio (buy_vol / sell_vol). > 1.0 = buying pressure.
    pub buy_sell_ratio: f64,
    /// Whether a large trade was detected in the last N trades.
    pub large_trade_detected: bool,
    /// Trade arrival rate (trades per second, exponentially smoothed).
    pub arrival_rate: f64,
    /// Average trade size over the window.
    pub avg_trade_size: f64,
    /// Total buy volume in the window.
    pub buy_volume: f64,
    /// Total sell volume in the window.
    pub sell_volume: f64,
}

impl Default for TradeFlowMetrics {
    fn default() -> Self {
        Self {
            buy_sell_ratio: 1.0,
            large_trade_detected: false,
            arrival_rate: 0.0,
            avg_trade_size: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trade Flow Analyzer
// ═══════════════════════════════════════════════════════════════════════════

/// Analyzes trade flow to detect directional pressure and large trades.
///
/// Uses a fixed-size circular buffer (VecDeque) to track the last N trades.
/// All operations are O(1) amortized with no heap allocations after construction.
pub struct TradeFlowAnalyzer {
    /// Ring buffer of recent trades.
    trades: VecDeque<TradeEvent>,
    /// Maximum number of trades to track.
    max_trades: usize,
    /// Running buy volume sum.
    buy_volume: f64,
    /// Running sell volume sum.
    sell_volume: f64,
    /// Running total volume sum.
    total_volume: f64,
    /// Exponential moving average of trade arrival rate (trades/sec).
    arrival_rate_ema: f64,
    /// Timestamp of the last trade (for rate calculation).
    last_trade_ns: u64,
    /// Large trade threshold multiplier (e.g., 10.0 = 10x average).
    large_trade_multiplier: f64,
    /// EMA smoothing factor for arrival rate.
    arrival_rate_alpha: f64,
}

impl TradeFlowAnalyzer {
    /// Create a new trade flow analyzer.
    ///
    /// # Arguments
    /// * `max_trades` — Number of recent trades to track (e.g., 100)
    /// * `large_trade_multiplier` — Threshold for large trade detection (e.g., 10.0)
    pub fn new(max_trades: usize, large_trade_multiplier: f64) -> Self {
        Self {
            trades: VecDeque::with_capacity(max_trades),
            max_trades,
            buy_volume: 0.0,
            sell_volume: 0.0,
            total_volume: 0.0,
            arrival_rate_ema: 0.0,
            last_trade_ns: 0,
            large_trade_multiplier,
            arrival_rate_alpha: 0.1, // 10% weight to new samples
        }
    }

    /// Process a new trade event.
    ///
    /// # Arguments
    /// * `price` — Trade price
    /// * `size` — Trade size (absolute value)
    /// * `side` — 0 = buy, 1 = sell
    /// * `timestamp_ns` — Trade timestamp in nanoseconds
    pub fn on_trade(&mut self, price: f64, size: f64, side: u8, timestamp_ns: u64) {
        let event = TradeEvent {
            price,
            size,
            side,
            timestamp_ns,
        };

        // Update arrival rate EMA
        if self.last_trade_ns > 0 {
            let interval_ns = timestamp_ns.saturating_sub(self.last_trade_ns);
            if interval_ns > 0 {
                let rate = 1_000_000_000.0 / interval_ns as f64; // trades per second
                if self.arrival_rate_ema == 0.0 {
                    self.arrival_rate_ema = rate;
                } else {
                    self.arrival_rate_ema = self.arrival_rate_alpha * rate
                        + (1.0 - self.arrival_rate_alpha) * self.arrival_rate_ema;
                }
            }
        }
        self.last_trade_ns = timestamp_ns;

        // Add to ring buffer
        if self.trades.len() >= self.max_trades {
            // Remove oldest trade and update running sums
            if let Some(old) = self.trades.pop_front() {
                if old.side == 0 {
                    self.buy_volume -= old.size;
                } else {
                    self.sell_volume -= old.size;
                }
                self.total_volume -= old.size;
            }
        }

        // Add new trade
        self.trades.push_back(event);
        if side == 0 {
            self.buy_volume += size;
        } else {
            self.sell_volume += size;
        }
        self.total_volume += size;
    }

    /// Get current trade flow metrics.
    pub fn get_metrics(&self) -> TradeFlowMetrics {
        let buy_sell_ratio = if self.sell_volume > 0.0 {
            self.buy_volume / self.sell_volume
        } else if self.buy_volume > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        let avg_trade_size = if !self.trades.is_empty() {
            self.total_volume / self.trades.len() as f64
        } else {
            0.0
        };

        // Check for large trades in the last 10 trades
        let large_trade_detected = self
            .trades
            .iter()
            .rev()
            .take(10)
            .any(|t| t.size > avg_trade_size * self.large_trade_multiplier);

        TradeFlowMetrics {
            buy_sell_ratio,
            large_trade_detected,
            arrival_rate: self.arrival_rate_ema,
            avg_trade_size,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
        }
    }

    /// Check if the analyzer is warmed up (has enough data).
    pub fn is_ready(&self) -> bool {
        self.trades.len() >= self.max_trades / 2
    }

    /// Reset all state (useful for testing or symbol changes).
    pub fn reset(&mut self) {
        self.trades.clear();
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        self.total_volume = 0.0;
        self.arrival_rate_ema = 0.0;
        self.last_trade_ns = 0;
    }
    
    /// Task 22: Calculate order flow toxicity score.
    ///
    /// Composite score combining:
    /// - VPIN (40% weight)
    /// - CVD divergence score (30% weight)
    /// - Large trade ratio (30% weight)
    ///
    /// Returns a score in [0.0, 1.0] where higher = more toxic flow.
    ///
    /// # Arguments
    /// * `vpin` — Current VPIN value [0.0, 1.0]
    /// * `cvd_divergence_score` — CVD divergence score [-1.0, 1.0]
    pub fn calculate_toxicity_score(&self, vpin: f64, cvd_divergence_score: f64) -> f64 {
        let metrics = self.get_metrics();
        
        // Calculate large trade ratio: volume from trades > 2x median / total volume
        let large_trade_ratio = if self.total_volume > 0.0 {
            // Calculate median trade size
            let mut sizes: Vec<f64> = self.trades.iter().map(|t| t.size).collect();
            sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if sizes.is_empty() {
                0.0
            } else {
                sizes[sizes.len() / 2]
            };
            
            // Sum volume from large trades (> 2x median)
            let large_threshold = median * 2.0;
            let large_volume: f64 = self.trades.iter()
                .filter(|t| t.size > large_threshold)
                .map(|t| t.size)
                .sum();
            
            large_volume / self.total_volume
        } else {
            0.0
        };
        
        // Composite toxicity score
        let toxicity = 0.4 * vpin 
            + 0.3 * cvd_divergence_score.abs() 
            + 0.3 * large_trade_ratio;
        
        toxicity.clamp(0.0, 1.0)
    }
}

impl Default for TradeFlowAnalyzer {
    fn default() -> Self {
        Self::new(100, 10.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_sell_ratio() {
        let mut analyzer = TradeFlowAnalyzer::new(10, 10.0);
        let base_ts = 1_000_000_000_000u64;

        // Add 7 buy trades and 3 sell trades
        for i in 0..10 {
            let side = if i < 7 { 0 } else { 1 };
            analyzer.on_trade(100.0, 10.0, side, base_ts + i * 1_000_000_000);
        }

        let metrics = analyzer.get_metrics();
        // Buy volume = 70, Sell volume = 30
        // Ratio = 70/30 = 2.333...
        assert!((metrics.buy_sell_ratio - 2.333).abs() < 0.01);
        assert_eq!(metrics.buy_volume, 70.0);
        assert_eq!(metrics.sell_volume, 30.0);
    }

    #[test]
    fn test_large_trade_detection() {
        let mut analyzer = TradeFlowAnalyzer::new(100, 10.0);
        let base_ts = 1_000_000_000_000u64;

        // Add 50 normal trades of size 10
        for i in 0..50 {
            analyzer.on_trade(100.0, 10.0, 0, base_ts + i * 1_000_000_000);
        }

        let metrics = analyzer.get_metrics();
        assert!(!metrics.large_trade_detected);
        assert_eq!(metrics.avg_trade_size, 10.0);

        // Add a large trade (150 = 15x average)
        analyzer.on_trade(100.0, 150.0, 0, base_ts + 51 * 1_000_000_000);

        let metrics = analyzer.get_metrics();
        assert!(metrics.large_trade_detected);
    }

    #[test]
    fn test_arrival_rate() {
        let mut analyzer = TradeFlowAnalyzer::new(100, 10.0);
        let base_ts = 1_000_000_000_000u64;

        // Add trades 100ms apart (10 trades/sec)
        for i in 0..20 {
            analyzer.on_trade(100.0, 10.0, 0, base_ts + i * 100_000_000);
        }

        let metrics = analyzer.get_metrics();
        // Should be around 10 trades/sec
        assert!(metrics.arrival_rate > 5.0 && metrics.arrival_rate < 15.0);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut analyzer = TradeFlowAnalyzer::new(5, 10.0);
        let base_ts = 1_000_000_000_000u64;

        // Add 10 trades (should only keep last 5)
        for i in 0..10 {
            analyzer.on_trade(100.0, 10.0, 0, base_ts + i * 1_000_000_000);
        }

        assert_eq!(analyzer.trades.len(), 5);
        assert_eq!(analyzer.buy_volume, 50.0); // Last 5 trades
    }

    #[test]
    fn test_reset() {
        let mut analyzer = TradeFlowAnalyzer::new(100, 10.0);
        let base_ts = 1_000_000_000_000u64;

        for i in 0..10 {
            analyzer.on_trade(100.0, 10.0, 0, base_ts + i * 1_000_000_000);
        }

        analyzer.reset();

        assert_eq!(analyzer.trades.len(), 0);
        assert_eq!(analyzer.buy_volume, 0.0);
        assert_eq!(analyzer.sell_volume, 0.0);
        assert_eq!(analyzer.arrival_rate_ema, 0.0);
    }
}
