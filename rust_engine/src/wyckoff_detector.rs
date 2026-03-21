use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WyckoffPhase {
    Accumulation,
    Markup,
    Distribution,
    Markdown,
    Unknown,
}

impl WyckoffPhase {
    pub fn to_string(&self) -> &'static str {
        match self {
            WyckoffPhase::Accumulation => "Accumulation",
            WyckoffPhase::Markup => "Markup",
            WyckoffPhase::Distribution => "Distribution",
            WyckoffPhase::Markdown => "Markdown",
            WyckoffPhase::Unknown => "Unknown",
        }
    }
}

pub struct WyckoffDetector {
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    high_volume_at_highs: VecDeque<(f64, f64)>, // (price, volume)
    high_volume_at_lows: VecDeque<(f64, f64)>,  // (price, volume)
    window_size: usize,
    range_threshold: f64, // Price range tightness threshold
}

impl WyckoffDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            price_history: VecDeque::with_capacity(window_size),
            volume_history: VecDeque::with_capacity(window_size),
            high_volume_at_highs: VecDeque::with_capacity(50),
            high_volume_at_lows: VecDeque::with_capacity(50),
            window_size,
            range_threshold: 0.02, // 2% range tightness
        }
    }

    pub fn update(&mut self, price: f64, volume: f64) {
        // Update price history
        if self.price_history.len() >= self.window_size {
            self.price_history.pop_front();
        }
        self.price_history.push_back(price);

        // Update volume history
        if self.volume_history.len() >= self.window_size {
            self.volume_history.pop_front();
        }
        self.volume_history.push_back(volume);

        // Track high volume events at price extremes
        if self.price_history.len() >= 20 {
            let avg_volume = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
            
            if volume > avg_volume * 1.5 {
                let recent_high = self.price_history.iter().skip(self.price_history.len().saturating_sub(20)).copied().fold(f64::NEG_INFINITY, f64::max);
                let recent_low = self.price_history.iter().skip(self.price_history.len().saturating_sub(20)).copied().fold(f64::INFINITY, f64::min);
                
                // High volume at highs
                if price >= recent_high * 0.995 {
                    if self.high_volume_at_highs.len() >= 50 {
                        self.high_volume_at_highs.pop_front();
                    }
                    self.high_volume_at_highs.push_back((price, volume));
                }
                
                // High volume at lows
                if price <= recent_low * 1.005 {
                    if self.high_volume_at_lows.len() >= 50 {
                        self.high_volume_at_lows.pop_front();
                    }
                    self.high_volume_at_lows.push_back((price, volume));
                }
            }
        }
    }

    pub fn detect_phase(&self) -> (WyckoffPhase, f64) {
        if self.price_history.len() < self.window_size {
            return (WyckoffPhase::Unknown, 0.0);
        }

        let high = self.price_history.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low = self.price_history.iter().copied().fold(f64::INFINITY, f64::min);
        let current_price = *self.price_history.back().unwrap();
        
        // Calculate price range tightness
        let range = (high - low) / low;
        let is_tight_range = range < self.range_threshold;
        
        // Calculate volume distribution
        let total_volume: f64 = self.volume_history.iter().sum();
        let avg_volume = total_volume / self.volume_history.len() as f64;
        
        // Volume at highs vs lows
        let volume_at_highs: f64 = self.high_volume_at_highs.iter().map(|(_, v)| v).sum();
        let volume_at_lows: f64 = self.high_volume_at_lows.iter().map(|(_, v)| v).sum();
        
        // Price position in range
        let price_position = if high > low {
            (current_price - low) / (high - low)
        } else {
            0.5
        };
        
        // Detect phase based on Wyckoff principles
        let (phase, confidence) = if is_tight_range && volume_at_lows > volume_at_highs * 1.3 {
            // Tight range with high volume at lows = Accumulation
            let conf = (volume_at_lows / (volume_at_highs + 1.0)).min(1.0) * 0.8;
            (WyckoffPhase::Accumulation, conf)
        } else if is_tight_range && volume_at_highs > volume_at_lows * 1.3 {
            // Tight range with high volume at highs = Distribution
            let conf = (volume_at_highs / (volume_at_lows + 1.0)).min(1.0) * 0.8;
            (WyckoffPhase::Distribution, conf)
        } else if !is_tight_range && price_position > 0.7 && volume_at_lows > avg_volume {
            // Expanding range, price near highs, volume at lows = Markup
            let conf = (price_position * 0.7).min(0.85);
            (WyckoffPhase::Markup, conf)
        } else if !is_tight_range && price_position < 0.3 && volume_at_highs > avg_volume {
            // Expanding range, price near lows, volume at highs = Markdown
            let conf = ((1.0 - price_position) * 0.7).min(0.85);
            (WyckoffPhase::Markdown, conf)
        } else {
            (WyckoffPhase::Unknown, 0.0)
        };
        
        (phase, confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wyckoff_detector_initialization() {
        let detector = WyckoffDetector::new(100);
        let (phase, confidence) = detector.detect_phase();
        assert_eq!(phase, WyckoffPhase::Unknown);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_accumulation_detection() {
        let mut detector = WyckoffDetector::new(100);
        
        // Simulate tight range with high volume at lows
        for i in 0..100 {
            let price = 100.0 + (i % 10) as f64 * 0.5; // Tight range
            let volume = if price < 102.0 { 1000.0 } else { 500.0 }; // High volume at lows
            detector.update(price, volume);
        }
        
        let (phase, confidence) = detector.detect_phase();
        assert_eq!(phase, WyckoffPhase::Accumulation);
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_distribution_detection() {
        let mut detector = WyckoffDetector::new(100);
        
        // Simulate tight range with high volume at highs
        for i in 0..100 {
            let price = 100.0 + (i % 10) as f64 * 0.5; // Tight range
            let volume = if price > 103.0 { 1000.0 } else { 500.0 }; // High volume at highs
            detector.update(price, volume);
        }
        
        let (phase, confidence) = detector.detect_phase();
        assert_eq!(phase, WyckoffPhase::Distribution);
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_markup_detection() {
        let mut detector = WyckoffDetector::new(100);
        
        // Simulate expanding range with price trending up
        for i in 0..100 {
            let price = 100.0 + i as f64 * 0.5; // Expanding upward
            let volume = if i < 30 { 1000.0 } else { 600.0 }; // Volume at lows
            detector.update(price, volume);
        }
        
        let (phase, _) = detector.detect_phase();
        assert_eq!(phase, WyckoffPhase::Markup);
    }

    #[test]
    fn test_markdown_detection() {
        let mut detector = WyckoffDetector::new(100);
        
        // Simulate expanding range with price trending down
        for i in 0..100 {
            let price = 150.0 - i as f64 * 0.5; // Expanding downward
            let volume = if i < 30 { 1000.0 } else { 600.0 }; // Volume at highs
            detector.update(price, volume);
        }
        
        let (phase, _) = detector.detect_phase();
        assert_eq!(phase, WyckoffPhase::Markdown);
    }

    #[test]
    fn test_phase_to_string() {
        assert_eq!(WyckoffPhase::Accumulation.to_string(), "Accumulation");
        assert_eq!(WyckoffPhase::Markup.to_string(), "Markup");
        assert_eq!(WyckoffPhase::Distribution.to_string(), "Distribution");
        assert_eq!(WyckoffPhase::Markdown.to_string(), "Markdown");
        assert_eq!(WyckoffPhase::Unknown.to_string(), "Unknown");
    }
}
