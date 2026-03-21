use std::collections::VecDeque;

/// Liquidation cascade state levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidationCascadeState {
    /// Normal market conditions (<$500k/s volume)
    Normal,
    /// Warning level ($500k-$1M/s volume)
    Warning,
    /// Active cascade ($1M-$5M/s volume)
    Active,
    /// Extreme cascade (>$5M/s volume)
    Extreme,
}

/// Liquidation cascade detector using volume-per-second analysis
pub struct LiquidationCascadeDetector {
    /// Ring buffer of recent trades (timestamp_ns, volume_usdt)
    trades: VecDeque<(u64, f64)>,
    
    /// Maximum entries to keep (1 second at ~1000 trades/sec = 1000)
    max_entries: usize,
    
    /// Exponential moving average of volume per second (alpha=0.1)
    ema_volume_per_sec: f64,
    
    /// EMA smoothing factor
    alpha: f64,
    
    /// Current cascade state
    current_state: LiquidationCascadeState,
}

impl LiquidationCascadeDetector {
    /// Create a new liquidation cascade detector
    pub fn new() -> Self {
        Self {
            trades: VecDeque::with_capacity(1000),
            max_entries: 1000,
            ema_volume_per_sec: 0.0,
            alpha: 0.1,
            current_state: LiquidationCascadeState::Normal,
        }
    }

    /// Update detector with a new trade
    /// 
    /// # Arguments
    /// * `timestamp_ns` - Trade timestamp in nanoseconds
    /// * `volume_usdt` - Trade volume in USDT
    pub fn on_trade(&mut self, timestamp_ns: u64, volume_usdt: f64) {
        // Add trade to ring buffer
        self.trades.push_back((timestamp_ns, volume_usdt));
        
        // Evict old entries if buffer is full
        if self.trades.len() > self.max_entries {
            self.trades.pop_front();
        }
        
        // Calculate volume per second over last 1 second window
        let volume_per_sec = self.calculate_volume_per_second(timestamp_ns);
        
        // Update EMA
        if self.ema_volume_per_sec == 0.0 {
            self.ema_volume_per_sec = volume_per_sec;
        } else {
            self.ema_volume_per_sec = self.alpha * volume_per_sec + (1.0 - self.alpha) * self.ema_volume_per_sec;
        }
        
        // Update state based on EMA volume
        self.current_state = self.classify_state(self.ema_volume_per_sec);
    }

    /// Calculate volume per second over the last 1 second window
    fn calculate_volume_per_second(&self, current_timestamp_ns: u64) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        
        // 1 second = 1_000_000_000 nanoseconds
        let one_second_ns = 1_000_000_000u64;
        let cutoff_ts = current_timestamp_ns.saturating_sub(one_second_ns);
        
        let mut total_volume = 0.0;
        for (ts, volume) in self.trades.iter().rev() {
            if *ts < cutoff_ts {
                break;
            }
            total_volume += volume;
        }
        
        total_volume
    }

    /// Classify cascade state based on volume per second
    fn classify_state(&self, volume_per_sec: f64) -> LiquidationCascadeState {
        if volume_per_sec > 5_000_000.0 {
            LiquidationCascadeState::Extreme
        } else if volume_per_sec > 1_000_000.0 {
            LiquidationCascadeState::Active
        } else if volume_per_sec > 500_000.0 {
            LiquidationCascadeState::Warning
        } else {
            LiquidationCascadeState::Normal
        }
    }

    /// Get current cascade state
    pub fn get_state(&self) -> LiquidationCascadeState {
        self.current_state
    }

    /// Get adjusted thresholds based on cascade state
    /// 
    /// Returns (imbalance_multiplier, trailing_stop_multiplier)
    /// - imbalance_multiplier: multiply imbalance threshold (require stronger signal)
    /// - trailing_stop_multiplier: multiply trailing stop distance (tighten stops)
    pub fn get_adjusted_thresholds(&self) -> (f64, f64) {
        match self.current_state {
            LiquidationCascadeState::Normal => (1.0, 1.0),
            LiquidationCascadeState::Warning => (1.5, 0.75),
            LiquidationCascadeState::Active => (2.0, 0.5),
            LiquidationCascadeState::Extreme => (3.0, 0.3),
        }
    }

    /// Get current EMA volume per second
    pub fn get_volume_per_sec(&self) -> f64 {
        self.ema_volume_per_sec
    }
}

impl Default for LiquidationCascadeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_state() {
        let mut detector = LiquidationCascadeDetector::new();
        
        // Add small trades
        let base_ts = 1_000_000_000_000u64;
        for i in 0..10 {
            detector.on_trade(base_ts + i * 100_000_000, 10_000.0); // $10k per trade
        }
        
        assert_eq!(detector.get_state(), LiquidationCascadeState::Normal);
    }

    #[test]
    fn test_warning_state() {
        let mut detector = LiquidationCascadeDetector::new();
        
        // Add medium-sized trades to reach warning level
        let base_ts = 1_000_000_000_000u64;
        for i in 0..100 {
            detector.on_trade(base_ts + i * 10_000_000, 7_000.0); // $7k per trade, ~$700k/s
        }
        
        let state = detector.get_state();
        assert!(
            state == LiquidationCascadeState::Warning || state == LiquidationCascadeState::Active,
            "Expected Warning or Active, got {:?}",
            state
        );
    }

    #[test]
    fn test_extreme_state() {
        let mut detector = LiquidationCascadeDetector::new();
        
        // Add large trades to reach extreme level
        let base_ts = 1_000_000_000_000u64;
        for i in 0..100 {
            detector.on_trade(base_ts + i * 10_000_000, 60_000.0); // $60k per trade, ~$6M/s
        }
        
        assert_eq!(detector.get_state(), LiquidationCascadeState::Extreme);
    }

    #[test]
    fn test_threshold_adjustments() {
        let mut detector = LiquidationCascadeDetector::new();
        
        // Normal state
        let (imb_mult, trail_mult) = detector.get_adjusted_thresholds();
        assert_eq!(imb_mult, 1.0);
        assert_eq!(trail_mult, 1.0);
        
        // Force extreme state
        let base_ts = 1_000_000_000_000u64;
        for i in 0..100 {
            detector.on_trade(base_ts + i * 10_000_000, 60_000.0);
        }
        
        let (imb_mult, trail_mult) = detector.get_adjusted_thresholds();
        assert_eq!(imb_mult, 3.0);
        assert_eq!(trail_mult, 0.3);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut detector = LiquidationCascadeDetector::new();
        
        let base_ts = 1_000_000_000_000u64;
        
        // Add a spike
        for i in 0..50 {
            detector.on_trade(base_ts + i * 10_000_000, 50_000.0);
        }
        
        let vol_after_spike = detector.get_volume_per_sec();
        
        // Add normal trades
        for i in 50..150 {
            detector.on_trade(base_ts + i * 10_000_000, 5_000.0);
        }
        
        let vol_after_normal = detector.get_volume_per_sec();
        
        // EMA should smooth out the spike
        assert!(vol_after_normal < vol_after_spike);
    }
}
