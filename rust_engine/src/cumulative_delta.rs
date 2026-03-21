use std::collections::VecDeque;

/// Kahan-compensated sum for numerical stability
#[derive(Debug, Clone, Copy)]
struct KahanSum {
    sum: f64,
    compensation: f64,
}

impl KahanSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    fn get(&self) -> f64 {
        self.sum
    }
}

/// Time-bucketed CVD entry
#[derive(Debug, Clone)]
struct CvdEntry {
    timestamp_ns: u64,
    delta: f64,
}

/// Cumulative Volume Delta tracker with multi-timeframe windows
pub struct CumulativeDeltaTracker {
    /// Ring buffer of delta entries (timestamp, delta)
    entries: VecDeque<CvdEntry>,
    /// Running CVD sum with Kahan compensation
    running_cvd: KahanSum,
    /// Maximum entries to keep (1 hour at ~1 trade/sec = 3600)
    max_entries: usize,
}

impl CumulativeDeltaTracker {
    /// Create a new CVD tracker
    pub fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(3600),
            running_cvd: KahanSum::new(),
            max_entries: 3600,
        }
    }

    /// Update CVD with a new trade
    /// 
    /// # Arguments
    /// * `timestamp_ns` - Trade timestamp in nanoseconds
    /// * `volume` - Trade volume
    /// * `is_buy` - True if buy (taker bought), false if sell (taker sold)
    pub fn on_trade(&mut self, timestamp_ns: u64, volume: f64, is_buy: bool) {
        let delta = if is_buy { volume } else { -volume };
        
        // Update running CVD with Kahan compensation
        self.running_cvd.add(delta);
        
        // Add to ring buffer
        self.entries.push_back(CvdEntry {
            timestamp_ns,
            delta,
        });
        
        // Evict old entries if buffer is full
        if self.entries.len() > self.max_entries {
            if let Some(old_entry) = self.entries.pop_front() {
                // Subtract old delta from running sum
                self.running_cvd.add(-old_entry.delta);
            }
        }
    }

    /// Get CVD for the last 5 minutes
    pub fn get_cvd_5m(&self) -> f64 {
        self.get_cvd_for_window(5 * 60 * 1_000_000_000)
    }

    /// Get CVD for the last 15 minutes
    pub fn get_cvd_15m(&self) -> f64 {
        self.get_cvd_for_window(15 * 60 * 1_000_000_000)
    }

    /// Get CVD for the last 1 hour
    pub fn get_cvd_1h(&self) -> f64 {
        self.get_cvd_for_window(60 * 60 * 1_000_000_000)
    }

    /// Get CVD for a specific time window (in nanoseconds)
    fn get_cvd_for_window(&self, window_ns: u64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }

        let latest_ts = self.entries.back().unwrap().timestamp_ns;
        let cutoff_ts = latest_ts.saturating_sub(window_ns);

        let mut cvd = KahanSum::new();
        for entry in self.entries.iter().rev() {
            if entry.timestamp_ns < cutoff_ts {
                break;
            }
            cvd.add(entry.delta);
        }

        cvd.get()
    }

    /// Detect bullish divergence: price makes new low but CVD makes higher low
    /// 
    /// # Arguments
    /// * `price_lows` - Recent price lows as (timestamp_ns, price) tuples
    /// 
    /// # Returns
    /// True if bullish divergence detected
    pub fn detect_bullish_divergence(&self, price_lows: &[(u64, f64)]) -> bool {
        if price_lows.len() < 2 {
            return false;
        }

        // Get the two most recent lows
        let recent_low = price_lows[price_lows.len() - 1];
        let previous_low = price_lows[price_lows.len() - 2];

        // Price makes new low
        if recent_low.1 >= previous_low.1 {
            return false;
        }

        // Get CVD at those timestamps
        let recent_cvd = self.get_cvd_at_time(recent_low.0);
        let previous_cvd = self.get_cvd_at_time(previous_low.0);

        // CVD makes higher low (bullish divergence)
        recent_cvd > previous_cvd
    }

    /// Detect bearish divergence: price makes new high but CVD makes lower high
    /// 
    /// # Arguments
    /// * `price_highs` - Recent price highs as (timestamp_ns, price) tuples
    /// 
    /// # Returns
    /// True if bearish divergence detected
    pub fn detect_bearish_divergence(&self, price_highs: &[(u64, f64)]) -> bool {
        if price_highs.len() < 2 {
            return false;
        }

        // Get the two most recent highs
        let recent_high = price_highs[price_highs.len() - 1];
        let previous_high = price_highs[price_highs.len() - 2];

        // Price makes new high
        if recent_high.1 <= previous_high.1 {
            return false;
        }

        // Get CVD at those timestamps
        let recent_cvd = self.get_cvd_at_time(recent_high.0);
        let previous_cvd = self.get_cvd_at_time(previous_high.0);

        // CVD makes lower high (bearish divergence)
        recent_cvd < previous_cvd
    }

    /// Get CVD value at a specific timestamp (approximate)
    fn get_cvd_at_time(&self, timestamp_ns: u64) -> f64 {
        let mut cvd = KahanSum::new();
        for entry in &self.entries {
            if entry.timestamp_ns > timestamp_ns {
                break;
            }
            cvd.add(entry.delta);
        }
        cvd.get()
    }

    /// Get the magnitude of divergence (for toxicity score)
    pub fn get_divergence_magnitude(&self) -> f64 {
        // Simple heuristic: compare 5m CVD slope vs 1h CVD slope
        let cvd_5m = self.get_cvd_5m();
        let cvd_1h = self.get_cvd_1h();
        
        if cvd_1h.abs() < 0.001 {
            return 0.0;
        }
        
        ((cvd_5m - cvd_1h) / cvd_1h).abs().min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cvd_accumulation() {
        let mut tracker = CumulativeDeltaTracker::new();
        
        // Add buy trades
        tracker.on_trade(1000, 10.0, true);
        tracker.on_trade(2000, 5.0, true);
        
        // Add sell trade
        tracker.on_trade(3000, 8.0, false);
        
        // CVD should be 10 + 5 - 8 = 7
        let cvd = tracker.get_cvd_5m();
        assert!((cvd - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_bullish_divergence() {
        let mut tracker = CumulativeDeltaTracker::new();
        
        // First low: price 100, CVD will be -10
        tracker.on_trade(1000, 10.0, false);
        let first_low_ts = 1000;
        
        // Some neutral activity
        tracker.on_trade(2000, 5.0, true);
        
        // Second low: price 95 (lower), but CVD will be -5 (higher)
        tracker.on_trade(3000, 5.0, true);
        let second_low_ts = 3000;
        
        let price_lows = vec![
            (first_low_ts, 100.0),
            (second_low_ts, 95.0),
        ];
        
        let divergence = tracker.detect_bullish_divergence(&price_lows);
        assert!(divergence);
    }

    #[test]
    fn test_bearish_divergence() {
        let mut tracker = CumulativeDeltaTracker::new();
        
        // First high: price 100, CVD will be +10
        tracker.on_trade(1000, 10.0, true);
        let first_high_ts = 1000;
        
        // Some neutral activity
        tracker.on_trade(2000, 5.0, false);
        
        // Second high: price 105 (higher), but CVD will be +5 (lower)
        tracker.on_trade(3000, 5.0, false);
        let second_high_ts = 3000;
        
        let price_highs = vec![
            (first_high_ts, 100.0),
            (second_high_ts, 105.0),
        ];
        
        let divergence = tracker.detect_bearish_divergence(&price_highs);
        assert!(divergence);
    }

    #[test]
    fn test_time_windows() {
        let mut tracker = CumulativeDeltaTracker::new();
        
        let base_ts = 1_000_000_000_000u64; // 1 second in nanoseconds
        
        // Add trades over 1 hour
        for i in 0..60 {
            let ts = base_ts + (i * 60 * 1_000_000_000); // Every minute
            tracker.on_trade(ts, 1.0, true);
        }
        
        // 5m window should have ~5 trades
        let cvd_5m = tracker.get_cvd_5m();
        assert!(cvd_5m >= 4.0 && cvd_5m <= 6.0);
        
        // 1h window should have all 60 trades
        let cvd_1h = tracker.get_cvd_1h();
        assert!((cvd_1h - 60.0).abs() < 1.0);
    }
}
