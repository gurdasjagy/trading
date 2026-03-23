use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct FibonacciLevels {
    pub swing_high: f64,
    pub swing_low: f64,
    pub levels: [f64; 5], // 23.6%, 38.2%, 50%, 61.8%, 78.6%
}

impl FibonacciLevels {
    const FIB_RATIOS: [f64; 5] = [0.236, 0.382, 0.500, 0.618, 0.786];

    pub fn new(swing_high: f64, swing_low: f64) -> Self {
        let range = swing_high - swing_low;
        let levels = [
            swing_high - range * Self::FIB_RATIOS[0],
            swing_high - range * Self::FIB_RATIOS[1],
            swing_high - range * Self::FIB_RATIOS[2],
            swing_high - range * Self::FIB_RATIOS[3],
            swing_high - range * Self::FIB_RATIOS[4],
        ];
        
        Self {
            swing_high,
            swing_low,
            levels,
        }
    }
}

struct SwingPoint {
    price: f64,
    timestamp: u64,
    is_high: bool,
}

pub struct FibonacciDetector {
    price_history_1h: VecDeque<(u64, f64)>,
    price_history_4h: VecDeque<(u64, f64)>,
    swing_points_1h: VecDeque<SwingPoint>,
    swing_points_4h: VecDeque<SwingPoint>,
    window_1h: usize,
    window_4h: usize,
    lookback_period: usize,
}

impl FibonacciDetector {
    pub fn new() -> Self {
        Self {
            price_history_1h: VecDeque::with_capacity(60),
            price_history_4h: VecDeque::with_capacity(240),
            swing_points_1h: VecDeque::with_capacity(20),
            swing_points_4h: VecDeque::with_capacity(20),
            window_1h: 60,   // 1 hour of minute data
            window_4h: 240,  // 4 hours of minute data
            lookback_period: 5, // Look back 5 periods for swing detection
        }
    }

    pub fn update(&mut self, timestamp: u64, price: f64) {
        // Update 1h history
        if self.price_history_1h.len() >= self.window_1h {
            self.price_history_1h.pop_front();
        }
        self.price_history_1h.push_back((timestamp, price));

        // Update 4h history
        if self.price_history_4h.len() >= self.window_4h {
            self.price_history_4h.pop_front();
        }
        self.price_history_4h.push_back((timestamp, price));

        // Detect swings periodically
        if self.price_history_1h.len() >= self.lookback_period * 2 + 1 {
            self.detect_swings();
        }
    }

    fn detect_swings(&mut self) {
        // Detect swing highs and lows in 1h window
        let history_1h: Vec<_> = self.price_history_1h.iter().cloned().collect();
        Self::detect_swing_in_window_static(&history_1h, &mut self.swing_points_1h, self.lookback_period);
        
        // Detect swing highs and lows in 4h window
        if self.price_history_4h.len() >= self.lookback_period * 2 + 1 {
            let history_4h: Vec<_> = self.price_history_4h.iter().cloned().collect();
            Self::detect_swing_in_window_static(&history_4h, &mut self.swing_points_4h, self.lookback_period);
        }
    }

    fn detect_swing_in_window_static(history: &[(u64, f64)], swing_points: &mut VecDeque<SwingPoint>, lookback_period: usize) {
        if history.len() < lookback_period * 2 + 1 {
            return;
        }

        let len = history.len();
        let idx = len - lookback_period - 1;
        
        if idx < lookback_period || idx + lookback_period >= len {
            return;
        }

        let (timestamp, price) = history[idx];
        
        // Check if it's a swing high
        let mut is_swing_high = true;
        for i in (idx - lookback_period)..=(idx + lookback_period) {
            if i != idx && history[i].1 >= price {
                is_swing_high = false;
                break;
            }
        }

        // Check if it's a swing low
        let mut is_swing_low = true;
        for i in (idx - lookback_period)..=(idx + lookback_period) {
            if i != idx && history[i].1 <= price {
                is_swing_low = false;
                break;
            }
        }

        if is_swing_high {
            if swing_points.len() >= 20 {
                swing_points.pop_front();
            }
            swing_points.push_back(SwingPoint {
                price,
                timestamp,
                is_high: true,
            });
        } else if is_swing_low {
            if swing_points.len() >= 20 {
                swing_points.pop_front();
            }
            swing_points.push_back(SwingPoint {
                price,
                timestamp,
                is_high: false,
            });
        }
    }

    pub fn calculate_fib_levels(&self) -> Option<FibonacciLevels> {
        // Use 4h swings if available, otherwise 1h
        let swing_points = if !self.swing_points_4h.is_empty() {
            &self.swing_points_4h
        } else {
            &self.swing_points_1h
        };

        if swing_points.len() < 2 {
            return None;
        }

        // Find most recent swing high and low
        let mut swing_high = f64::NEG_INFINITY;
        let mut swing_low = f64::INFINITY;

        for swing in swing_points.iter().rev().take(10) {
            if swing.is_high && swing.price > swing_high {
                swing_high = swing.price;
            } else if !swing.is_high && swing.price < swing_low {
                swing_low = swing.price;
            }
        }

        if swing_high > swing_low && swing_high.is_finite() && swing_low.is_finite() {
            Some(FibonacciLevels::new(swing_high, swing_low))
        } else {
            None
        }
    }

    pub fn get_nearest_level(&self, price: f64) -> Option<(f64, f64, bool)> {
        let fib_levels = self.calculate_fib_levels()?;
        
        let mut nearest_level = fib_levels.levels[0];
        let mut min_distance = (price - nearest_level).abs();

        for &level in &fib_levels.levels {
            let distance = (price - level).abs();
            if distance < min_distance {
                min_distance = distance;
                nearest_level = level;
            }
        }

        // Calculate which Fibonacci percentage this is
        let level_pct = if fib_levels.swing_high > fib_levels.swing_low {
            let range = fib_levels.swing_high - fib_levels.swing_low;
            let retracement = fib_levels.swing_high - nearest_level;
            (retracement / range * 100.0).round() / 100.0
        } else {
            0.0
        };

        // Distance in basis points
        let distance_bps = (min_distance / price * 10000.0).abs();
        
        // Is approaching if within 1% (100 bps)
        let is_approaching = distance_bps < 100.0;

        Some((level_pct, distance_bps, is_approaching))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_levels_calculation() {
        let fib = FibonacciLevels::new(100.0, 50.0);
        
        assert_eq!(fib.swing_high, 100.0);
        assert_eq!(fib.swing_low, 50.0);
        
        // Check 50% retracement
        assert!((fib.levels[2] - 75.0).abs() < 0.01);
        
        // Check 61.8% retracement
        assert!((fib.levels[3] - 69.1).abs() < 0.1);
    }

    #[test]
    fn test_fibonacci_detector_initialization() {
        let detector = FibonacciDetector::new();
        assert!(detector.calculate_fib_levels().is_none());
    }

    #[test]
    fn test_swing_detection() {
        let mut detector = FibonacciDetector::new();
        
        // Create a price pattern with clear swing high and low
        let prices = vec![
            50.0, 51.0, 52.0, 53.0, 54.0, 55.0, // Up trend
            56.0, 57.0, 58.0, 59.0, 60.0,       // Peak
            59.0, 58.0, 57.0, 56.0, 55.0,       // Down trend
            54.0, 53.0, 52.0, 51.0, 50.0,       // Trough
            51.0, 52.0, 53.0, 54.0, 55.0,       // Recovery
        ];

        for (i, price) in prices.iter().enumerate() {
            detector.update(i as u64, *price);
        }

        // Should have detected some swings
        assert!(detector.swing_points_1h.len() > 0);
    }

    #[test]
    fn test_get_nearest_level() {
        let mut detector = FibonacciDetector::new();
        
        // Build up enough history with clear swings
        for i in 0..30 {
            let price = if i < 10 {
                100.0 - i as f64
            } else if i < 20 {
                90.0 + (i - 10) as f64
            } else {
                100.0 - (i - 20) as f64
            };
            detector.update(i as u64, price);
        }

        // Manually add swing points for testing
        detector.swing_points_1h.push_back(SwingPoint {
            price: 100.0,
            timestamp: 0,
            is_high: true,
        });
        detector.swing_points_1h.push_back(SwingPoint {
            price: 50.0,
            timestamp: 10,
            is_high: false,
        });

        if let Some((level_pct, distance_bps, is_approaching)) = detector.get_nearest_level(75.0) {
            // 75.0 is the 50% retracement level
            assert!((level_pct - 0.5).abs() < 0.01);
            assert!(distance_bps < 10.0); // Very close
            assert!(is_approaching);
        }
    }

    #[test]
    fn test_fib_ratios() {
        let fib = FibonacciLevels::new(100.0, 0.0);
        
        // Verify standard Fibonacci ratios
        assert!((fib.levels[0] - 76.4).abs() < 0.1); // 23.6%
        assert!((fib.levels[1] - 61.8).abs() < 0.1); // 38.2%
        assert!((fib.levels[2] - 50.0).abs() < 0.1); // 50.0%
        assert!((fib.levels[3] - 38.2).abs() < 0.1); // 61.8%
        assert!((fib.levels[4] - 21.4).abs() < 0.1); // 78.6%
    }
}
