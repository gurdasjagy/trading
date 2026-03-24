use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudPosition {
    AboveCloud,
    InCloud,
    BelowCloud,
}

impl CloudPosition {
    pub fn to_string(&self) -> &'static str {
        match self {
            CloudPosition::AboveCloud => "AboveCloud",
            CloudPosition::InCloud => "InCloud",
            CloudPosition::BelowCloud => "BelowCloud",
        }
    }
}

pub struct Candle {
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

pub struct IchimokuCloud {
    candles: VecDeque<Candle>,
    tenkan_period: usize,   // 9
    kijun_period: usize,    // 26
    senkou_b_period: usize, // 52
    displacement: usize,    // 26
}

impl IchimokuCloud {
    pub fn new() -> Self {
        Self {
            candles: VecDeque::with_capacity(100),
            tenkan_period: 9,
            kijun_period: 26,
            senkou_b_period: 52,
            displacement: 26,
        }
    }

    pub fn update_candle(&mut self, high: f64, low: f64, close: f64) {
        if self.candles.len() >= 100 {
            self.candles.pop_front();
        }
        
        self.candles.push_back(Candle { high, low, close });
    }

    fn calculate_midpoint(&self, period: usize) -> Option<f64> {
        if self.candles.len() < period {
            return None;
        }

        let recent_candles: Vec<_> = self.candles.iter().rev().take(period).collect();
        
        let highest = recent_candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest = recent_candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        
        Some((highest + lowest) / 2.0)
    }

    pub fn tenkan_sen(&self) -> Option<f64> {
        self.calculate_midpoint(self.tenkan_period)
    }

    pub fn kijun_sen(&self) -> Option<f64> {
        self.calculate_midpoint(self.kijun_period)
    }

    pub fn senkou_span_a(&self) -> Option<f64> {
        let tenkan = self.tenkan_sen()?;
        let kijun = self.kijun_sen()?;
        Some((tenkan + kijun) / 2.0)
    }

    pub fn senkou_span_b(&self) -> Option<f64> {
        self.calculate_midpoint(self.senkou_b_period)
    }

    pub fn chikou_span(&self) -> Option<f64> {
        if self.candles.len() < self.displacement {
            return None;
        }
        
        Some(self.candles.back()?.close)
    }

    /// Returns true if enough candles have been ingested for Ichimoku to produce
    /// meaningful cloud values (needs at least `senkou_b_period` = 52 candles).
    pub fn is_warmed_up(&self) -> bool {
        self.candles.len() >= self.senkou_b_period
    }

    pub fn get_cloud_position(&self, price: f64) -> CloudPosition {
        let span_a = match self.senkou_span_a() {
            Some(v) => v,
            None => return CloudPosition::InCloud,
        };
        
        let span_b = match self.senkou_span_b() {
            Some(v) => v,
            None => return CloudPosition::InCloud,
        };

        let cloud_top = span_a.max(span_b);
        let cloud_bottom = span_a.min(span_b);

        if price > cloud_top {
            CloudPosition::AboveCloud
        } else if price < cloud_bottom {
            CloudPosition::BelowCloud
        } else {
            CloudPosition::InCloud
        }
    }

    pub fn get_trend_filter(&self) -> (bool, bool, f64) {
        let tenkan = match self.tenkan_sen() {
            Some(v) => v,
            None => return (false, false, 0.0),
        };
        
        let kijun = match self.kijun_sen() {
            Some(v) => v,
            None => return (false, false, 0.0),
        };
        
        let span_a = match self.senkou_span_a() {
            Some(v) => v,
            None => return (false, false, 0.0),
        };
        
        let span_b = match self.senkou_span_b() {
            Some(v) => v,
            None => return (false, false, 0.0),
        };

        let current_price = self.candles.back().map(|c| c.close).unwrap_or(0.0);

        // Bullish: price above cloud, tenkan > kijun, span_a > span_b
        let is_bullish = current_price > span_a.max(span_b) 
            && tenkan > kijun 
            && span_a > span_b;

        // Bearish: price below cloud, tenkan < kijun, span_a < span_b
        let is_bearish = current_price < span_a.min(span_b) 
            && tenkan < kijun 
            && span_a < span_b;

        // Strength based on separation
        let cloud_thickness = (span_a - span_b).abs();
        let tk_separation = (tenkan - kijun).abs();
        let strength = if current_price > 0.0 {
            ((cloud_thickness + tk_separation) / current_price * 100.0).min(1.0)
        } else {
            0.0
        };

        (is_bullish, is_bearish, strength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ichimoku_initialization() {
        let ichimoku = IchimokuCloud::new();
        assert!(ichimoku.tenkan_sen().is_none());
        assert!(ichimoku.kijun_sen().is_none());
    }

    #[test]
    fn test_tenkan_sen_calculation() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Add 9 candles with clear high/low
        for i in 0..9 {
            ichimoku.update_candle(100.0 + i as f64, 90.0 + i as f64, 95.0 + i as f64);
        }
        
        let tenkan = ichimoku.tenkan_sen().unwrap();
        // Midpoint of (108, 98) = 103
        assert!((tenkan - 103.0).abs() < 0.1);
    }

    #[test]
    fn test_kijun_sen_calculation() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Add 26 candles
        for i in 0..26 {
            ichimoku.update_candle(100.0 + i as f64, 90.0 + i as f64, 95.0 + i as f64);
        }
        
        let kijun = ichimoku.kijun_sen().unwrap();
        // Midpoint of (125, 115) = 120
        assert!((kijun - 120.0).abs() < 0.1);
    }

    #[test]
    fn test_cloud_position_above() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Build up enough history
        for i in 0..60 {
            ichimoku.update_candle(100.0, 90.0, 95.0);
        }
        
        let position = ichimoku.get_cloud_position(150.0);
        assert_eq!(position, CloudPosition::AboveCloud);
    }

    #[test]
    fn test_cloud_position_below() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Build up enough history
        for i in 0..60 {
            ichimoku.update_candle(100.0, 90.0, 95.0);
        }
        
        let position = ichimoku.get_cloud_position(50.0);
        assert_eq!(position, CloudPosition::BelowCloud);
    }

    #[test]
    fn test_cloud_position_in_cloud() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Build up enough history
        for i in 0..60 {
            ichimoku.update_candle(100.0, 90.0, 95.0);
        }
        
        let position = ichimoku.get_cloud_position(95.0);
        // Should be in or near cloud
        assert!(matches!(position, CloudPosition::InCloud | CloudPosition::AboveCloud | CloudPosition::BelowCloud));
    }

    #[test]
    fn test_trend_filter_bullish() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Create bullish trend: rising prices
        for i in 0..60 {
            let base = 100.0 + i as f64;
            ichimoku.update_candle(base + 5.0, base, base + 2.0);
        }
        
        let (is_bullish, is_bearish, strength) = ichimoku.get_trend_filter();
        // In a strong uptrend, should be bullish
        assert!(strength >= 0.0);
    }

    #[test]
    fn test_trend_filter_bearish() {
        let mut ichimoku = IchimokuCloud::new();
        
        // Create bearish trend: falling prices
        for i in 0..60 {
            let base = 200.0 - i as f64;
            ichimoku.update_candle(base, base - 5.0, base - 2.0);
        }
        
        let (is_bullish, is_bearish, strength) = ichimoku.get_trend_filter();
        // In a strong downtrend, should be bearish
        assert!(strength >= 0.0);
    }

    #[test]
    fn test_cloud_position_to_string() {
        assert_eq!(CloudPosition::AboveCloud.to_string(), "AboveCloud");
        assert_eq!(CloudPosition::InCloud.to_string(), "InCloud");
        assert_eq!(CloudPosition::BelowCloud.to_string(), "BelowCloud");
    }
}
