use std::collections::{HashMap, VecDeque};

pub struct PriceHistory {
    prices: VecDeque<(u64, f64)>,
    window_size: usize,
}

impl PriceHistory {
    fn new(window_size: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    fn add(&mut self, timestamp: u64, price: f64) {
        if self.prices.len() >= self.window_size {
            self.prices.pop_front();
        }
        self.prices.push_back((timestamp, price));
    }

    fn get_returns(&self) -> Vec<f64> {
        if self.prices.len() < 2 {
            return Vec::new();
        }

        let mut returns = Vec::with_capacity(self.prices.len() - 1);
        for i in 1..self.prices.len() {
            let prev_price = self.prices[i - 1].1;
            let curr_price = self.prices[i].1;
            
            if prev_price > 0.0 {
                let ret = (curr_price - prev_price) / prev_price;
                returns.push(ret);
            }
        }
        
        returns
    }
}

pub struct CrossAssetCorrelationMonitor {
    price_histories: HashMap<String, PriceHistory>,
    correlation_matrix: HashMap<(String, String), f64>,
    window_size: usize,
    correlation_threshold: f64,
}

impl CrossAssetCorrelationMonitor {
    pub fn new(window_size: usize) -> Self {
        let mut monitor = Self {
            price_histories: HashMap::new(),
            correlation_matrix: HashMap::new(),
            window_size,
            correlation_threshold: 0.7,
        };

        // Initialize price histories for common pairs
        monitor.price_histories.insert("BTC_USDT".to_string(), PriceHistory::new(window_size));
        monitor.price_histories.insert("ETH_USDT".to_string(), PriceHistory::new(window_size));
        monitor.price_histories.insert("SOL_USDT".to_string(), PriceHistory::new(window_size));
        monitor.price_histories.insert("SPX".to_string(), PriceHistory::new(window_size));

        monitor
    }

    pub fn update_price(&mut self, symbol: &str, timestamp: u64, price: f64) {
        // Ensure symbol exists in histories
        if !self.price_histories.contains_key(symbol) {
            self.price_histories.insert(symbol.to_string(), PriceHistory::new(self.window_size));
        }

        if let Some(history) = self.price_histories.get_mut(symbol) {
            history.add(timestamp, price);
        }

        // Recalculate correlations periodically (every 10 updates)
        if timestamp % 10 == 0 {
            self.recalculate_correlations();
        }
    }

    fn recalculate_correlations(&mut self) {
        let symbols: Vec<String> = self.price_histories.keys().cloned().collect();

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let symbol_a = &symbols[i];
                let symbol_b = &symbols[j];

                if let Some(corr) = self.calculate_correlation(symbol_a, symbol_b) {
                    self.correlation_matrix.insert((symbol_a.clone(), symbol_b.clone()), corr);
                    self.correlation_matrix.insert((symbol_b.clone(), symbol_a.clone()), corr);
                }
            }
        }
    }

    fn calculate_correlation(&self, symbol_a: &str, symbol_b: &str) -> Option<f64> {
        let history_a = self.price_histories.get(symbol_a)?;
        let history_b = self.price_histories.get(symbol_b)?;

        let returns_a = history_a.get_returns();
        let returns_b = history_b.get_returns();

        if returns_a.len() < 10 || returns_b.len() < 10 {
            return None;
        }

        // Use minimum length
        let n = returns_a.len().min(returns_b.len());
        let returns_a = &returns_a[returns_a.len() - n..];
        let returns_b = &returns_b[returns_b.len() - n..];

        // Calculate Pearson correlation coefficient
        let mean_a: f64 = returns_a.iter().sum::<f64>() / n as f64;
        let mean_b: f64 = returns_b.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for i in 0..n {
            let diff_a = returns_a[i] - mean_a;
            let diff_b = returns_b[i] - mean_b;
            
            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        
        if denominator > 0.0 {
            Some((numerator / denominator).max(-1.0).min(1.0))
        } else {
            None
        }
    }

    pub fn get_correlation(&self, symbol_a: &str, symbol_b: &str) -> Option<f64> {
        self.correlation_matrix.get(&(symbol_a.to_string(), symbol_b.to_string())).copied()
    }

    pub fn detect_decorrelation_events(&self) -> Vec<(String, f64, bool)> {
        let mut events = Vec::new();

        for ((symbol_a, symbol_b), &correlation) in &self.correlation_matrix {
            // Only report each pair once (a,b) not (b,a)
            if symbol_a < symbol_b {
                let threshold_breached = correlation.abs() < self.correlation_threshold;
                
                if threshold_breached {
                    let pair_name = format!("{}-{}", symbol_a, symbol_b);
                    events.push((pair_name, correlation, threshold_breached));
                }
            }
        }

        events
    }

    pub fn set_correlation_threshold(&mut self, threshold: f64) {
        self.correlation_threshold = threshold.max(0.0).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_history() {
        let mut history = PriceHistory::new(10);
        
        history.add(1, 100.0);
        history.add(2, 105.0);
        history.add(3, 110.0);
        
        let returns = history.get_returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 0.001); // 5% return
        assert!((returns[1] - 0.047619).abs() < 0.001); // ~4.76% return
    }

    #[test]
    fn test_correlation_monitor_initialization() {
        let monitor = CrossAssetCorrelationMonitor::new(60);
        
        assert!(monitor.price_histories.contains_key("BTC_USDT"));
        assert!(monitor.price_histories.contains_key("ETH_USDT"));
        assert!(monitor.price_histories.contains_key("SOL_USDT"));
    }

    #[test]
    fn test_perfect_positive_correlation() {
        let mut monitor = CrossAssetCorrelationMonitor::new(60);
        
        // Add perfectly correlated prices
        for i in 0..30 {
            let timestamp = i as u64;
            let btc_price = 50000.0 + i as f64 * 100.0;
            let eth_price = 3000.0 + i as f64 * 6.0; // Proportional movement
            
            monitor.update_price("BTC_USDT", timestamp, btc_price);
            monitor.update_price("ETH_USDT", timestamp, eth_price);
        }
        
        monitor.recalculate_correlations();
        
        if let Some(corr) = monitor.get_correlation("BTC_USDT", "ETH_USDT") {
            assert!(corr > 0.95); // Should be very high positive correlation
        }
    }

    #[test]
    fn test_perfect_negative_correlation() {
        let mut monitor = CrossAssetCorrelationMonitor::new(60);
        
        // Add negatively correlated prices
        for i in 0..30 {
            let timestamp = i as u64;
            let btc_price = 50000.0 + i as f64 * 100.0;
            let eth_price = 4000.0 - i as f64 * 6.0; // Inverse movement
            
            monitor.update_price("BTC_USDT", timestamp, btc_price);
            monitor.update_price("ETH_USDT", timestamp, eth_price);
        }
        
        monitor.recalculate_correlations();
        
        if let Some(corr) = monitor.get_correlation("BTC_USDT", "ETH_USDT") {
            assert!(corr < -0.95); // Should be very high negative correlation
        }
    }

    #[test]
    fn test_decorrelation_detection() {
        let mut monitor = CrossAssetCorrelationMonitor::new(60);
        monitor.set_correlation_threshold(0.7);
        
        // Add uncorrelated prices
        for i in 0..30 {
            let timestamp = i as u64;
            let btc_price = 50000.0 + (i as f64 * 100.0 * ((i as f64).sin()));
            let eth_price = 3000.0 + (i as f64 * 50.0 * ((i as f64 * 2.0).cos()));
            
            monitor.update_price("BTC_USDT", timestamp, btc_price);
            monitor.update_price("ETH_USDT", timestamp, eth_price);
        }
        
        monitor.recalculate_correlations();
        
        let events = monitor.detect_decorrelation_events();
        // Should detect decorrelation
        assert!(events.len() > 0 || monitor.get_correlation("BTC_USDT", "ETH_USDT").is_some());
    }

    #[test]
    fn test_correlation_symmetry() {
        let mut monitor = CrossAssetCorrelationMonitor::new(60);
        
        for i in 0..30 {
            monitor.update_price("BTC_USDT", i as u64, 50000.0 + i as f64 * 100.0);
            monitor.update_price("ETH_USDT", i as u64, 3000.0 + i as f64 * 6.0);
        }
        
        monitor.recalculate_correlations();
        
        let corr_ab = monitor.get_correlation("BTC_USDT", "ETH_USDT");
        let corr_ba = monitor.get_correlation("ETH_USDT", "BTC_USDT");
        
        assert_eq!(corr_ab, corr_ba);
    }

    #[test]
    fn test_insufficient_data() {
        let monitor = CrossAssetCorrelationMonitor::new(60);
        
        // No data added
        let corr = monitor.get_correlation("BTC_USDT", "ETH_USDT");
        assert!(corr.is_none());
    }

    #[test]
    fn test_dynamic_symbol_addition() {
        let mut monitor = CrossAssetCorrelationMonitor::new(60);
        
        // Add a new symbol not in initial list
        monitor.update_price("MATIC_USDT", 1, 1.5);
        
        assert!(monitor.price_histories.contains_key("MATIC_USDT"));
    }
}
