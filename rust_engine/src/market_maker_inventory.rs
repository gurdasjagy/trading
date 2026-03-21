use std::collections::VecDeque;

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

pub struct MarketMakerInventoryModel {
    net_position: KahanSum,
    position_history: VecDeque<f64>,
    position_ema: f64,
    ema_alpha: f64,
    volume_history: VecDeque<f64>,
    window_size: usize,
}

impl MarketMakerInventoryModel {
    pub fn new(window_size: usize) -> Self {
        Self {
            net_position: KahanSum::new(),
            position_history: VecDeque::with_capacity(window_size),
            position_ema: 0.0,
            ema_alpha: 0.1, // 10% weight to new values
            volume_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn on_trade(&mut self, size: f64, is_buy: bool) {
        // Market maker takes opposite side
        // If trade is buy (aggressor buys), MM sells (negative position)
        // If trade is sell (aggressor sells), MM buys (positive position)
        let position_change = if is_buy { -size } else { size };
        
        self.net_position.add(position_change);
        
        // Update position history
        if self.position_history.len() >= self.window_size {
            self.position_history.pop_front();
        }
        self.position_history.push_back(self.net_position.get());
        
        // Update position EMA
        self.position_ema = self.ema_alpha * self.net_position.get() + (1.0 - self.ema_alpha) * self.position_ema;
        
        // Update volume history
        if self.volume_history.len() >= self.window_size {
            self.volume_history.pop_front();
        }
        self.volume_history.push_back(size);
    }

    pub fn get_inventory_pressure(&self) -> f64 {
        if self.volume_history.is_empty() {
            return 0.0;
        }

        let typical_volume: f64 = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
        
        if typical_volume == 0.0 {
            return 0.0;
        }

        // Normalize position by typical volume
        let pressure = self.position_ema / typical_volume;
        
        // Clamp to [-1.0, 1.0]
        pressure.max(-1.0).min(1.0)
    }

    pub fn get_inventory_signal(&self) -> (i8, f64) {
        let pressure = self.get_inventory_pressure();
        
        // High positive pressure = MM is long, wants to sell (bearish)
        // High negative pressure = MM is short, wants to buy (bullish)
        
        let direction = if pressure > 0.3 {
            -1 // Bearish
        } else if pressure < -0.3 {
            1 // Bullish
        } else {
            0 // Neutral
        };
        
        let pressure_score = pressure.abs();
        
        (direction, pressure_score)
    }

    pub fn reset_daily(&mut self) {
        self.net_position = KahanSum::new();
        self.position_ema = 0.0;
        self.position_history.clear();
        // Keep volume history for continuity
    }

    pub fn get_net_position(&self) -> f64 {
        self.net_position.get()
    }

    pub fn get_position_ema(&self) -> f64 {
        self.position_ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_accuracy() {
        let mut kahan = KahanSum::new();
        
        // Add many small values
        for _ in 0..1000 {
            kahan.add(0.001);
        }
        
        assert!((kahan.get() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inventory_model_initialization() {
        let model = MarketMakerInventoryModel::new(100);
        assert_eq!(model.get_net_position(), 0.0);
        assert_eq!(model.get_inventory_pressure(), 0.0);
    }

    #[test]
    fn test_buy_trades_create_negative_position() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Aggressor buys, MM sells
        model.on_trade(100.0, true);
        
        assert!(model.get_net_position() < 0.0);
    }

    #[test]
    fn test_sell_trades_create_positive_position() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Aggressor sells, MM buys
        model.on_trade(100.0, false);
        
        assert!(model.get_net_position() > 0.0);
    }

    #[test]
    fn test_inventory_pressure_calculation() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Create consistent buy pressure (MM accumulates short position)
        for _ in 0..50 {
            model.on_trade(100.0, true);
        }
        
        let pressure = model.get_inventory_pressure();
        assert!(pressure < 0.0); // Negative pressure (MM is short)
        assert!(pressure >= -1.0); // Clamped
    }

    #[test]
    fn test_inventory_signal_bearish() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // MM accumulates long position (wants to sell)
        for _ in 0..50 {
            model.on_trade(100.0, false);
        }
        
        let (direction, score) = model.get_inventory_signal();
        assert_eq!(direction, -1); // Bearish
        assert!(score > 0.0);
    }

    #[test]
    fn test_inventory_signal_bullish() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // MM accumulates short position (wants to buy)
        for _ in 0..50 {
            model.on_trade(100.0, true);
        }
        
        let (direction, score) = model.get_inventory_signal();
        assert_eq!(direction, 1); // Bullish
        assert!(score > 0.0);
    }

    #[test]
    fn test_inventory_signal_neutral() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Balanced trades
        for _ in 0..25 {
            model.on_trade(100.0, true);
            model.on_trade(100.0, false);
        }
        
        let (direction, _) = model.get_inventory_signal();
        assert_eq!(direction, 0); // Neutral
    }

    #[test]
    fn test_reset_daily() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Build up position
        for _ in 0..50 {
            model.on_trade(100.0, true);
        }
        
        assert!(model.get_net_position() != 0.0);
        
        // Reset
        model.reset_daily();
        
        assert_eq!(model.get_net_position(), 0.0);
        assert_eq!(model.get_position_ema(), 0.0);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Add trades and check EMA updates
        model.on_trade(100.0, true);
        let ema1 = model.get_position_ema();
        
        model.on_trade(100.0, true);
        let ema2 = model.get_position_ema();
        
        // EMA should be smoothing the position
        assert!(ema2 < ema1); // More negative
        assert!(ema2.abs() < model.get_net_position().abs()); // EMA lags
    }
}
