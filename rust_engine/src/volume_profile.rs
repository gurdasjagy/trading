use std::collections::HashMap;
use ordered_float::OrderedFloat;

/// Volume Profile tracker with VPOC (Volume Point of Control) and Value Area calculation
pub struct VolumeProfile {
    /// Volume accumulated at each price level (rounded to tick size)
    /// Key: price (OrderedFloat for HashMap compatibility)
    /// Value: cumulative volume at that price
    volume_by_price: HashMap<OrderedFloat<f64>, f64>,
    
    /// Tick size for price rounding
    tick_size: f64,
    
    /// Total volume across all price levels
    total_volume: f64,
    
    /// Cached VPOC (price with highest volume)
    cached_vpoc: Option<f64>,
    
    /// Cached Value Area High (70th percentile)
    cached_vah: Option<f64>,
    
    /// Cached Value Area Low (30th percentile)
    cached_val: Option<f64>,
    
    /// Flag to track if cache needs recalculation
    cache_dirty: bool,
}

impl VolumeProfile {
    /// Create a new VolumeProfile with specified tick size
    pub fn new(tick_size: f64) -> Self {
        Self {
            volume_by_price: HashMap::new(),
            tick_size,
            total_volume: 0.0,
            cached_vpoc: None,
            cached_vah: None,
            cached_val: None,
            cache_dirty: true,
        }
    }

    /// Round price to nearest tick
    #[inline]
    fn round_to_tick(&self, price: f64) -> f64 {
        if self.tick_size <= 0.0 {
            return price;
        }
        (price / self.tick_size).round() * self.tick_size
    }

    /// Update volume profile with a new trade
    /// 
    /// # Arguments
    /// * `price` - Trade price
    /// * `volume` - Trade volume
    /// * `tick_size` - Tick size for rounding (overrides constructor value if provided)
    pub fn update_trade(&mut self, price: f64, volume: f64, tick_size: f64) {
        if volume <= 0.0 || price <= 0.0 {
            return;
        }

        // Update tick size if provided
        if tick_size > 0.0 && (self.tick_size - tick_size).abs() > 1e-10 {
            self.tick_size = tick_size;
        }

        // Round price to tick
        let rounded_price = self.round_to_tick(price);
        let key = OrderedFloat(rounded_price);

        // Accumulate volume at this price level
        *self.volume_by_price.entry(key).or_insert(0.0) += volume;
        self.total_volume += volume;
        
        // Mark cache as dirty
        self.cache_dirty = true;
    }

    /// Get the VPOC (Volume Point of Control) - price level with highest volume
    /// 
    /// Returns None if no trades have been recorded
    pub fn get_vpoc(&mut self) -> Option<f64> {
        if self.cache_dirty {
            self.recalculate_cache();
        }
        self.cached_vpoc
    }

    /// Get the Value Area (70th percentile high, 30th percentile low)
    /// 
    /// Returns (Value Area Low, Value Area High) or None if insufficient data
    pub fn get_value_area(&mut self) -> Option<(f64, f64)> {
        if self.cache_dirty {
            self.recalculate_cache();
        }
        
        match (self.cached_val, self.cached_vah) {
            (Some(val), Some(vah)) => Some((val, vah)),
            _ => None,
        }
    }

    /// Recalculate cached values (VPOC and Value Area)
    fn recalculate_cache(&mut self) {
        if self.volume_by_price.is_empty() {
            self.cached_vpoc = None;
            self.cached_vah = None;
            self.cached_val = None;
            self.cache_dirty = false;
            return;
        }

        // Find VPOC (price with maximum volume)
        let mut max_volume = 0.0;
        let mut vpoc_price = 0.0;
        
        for (&price, &volume) in &self.volume_by_price {
            if volume > max_volume {
                max_volume = volume;
                vpoc_price = price.0;
            }
        }
        
        self.cached_vpoc = Some(vpoc_price);

        // Calculate Value Area (70% of volume around VPOC)
        // Sort price levels by volume (descending)
        let mut price_volume_pairs: Vec<(f64, f64)> = self.volume_by_price
            .iter()
            .map(|(&price, &volume)| (price.0, volume))
            .collect();
        
        price_volume_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Accumulate volume until we reach 70% of total
        let target_volume = self.total_volume * 0.70;
        let mut accumulated_volume = 0.0;
        let mut value_area_prices: Vec<f64> = Vec::new();

        for (price, volume) in price_volume_pairs {
            if accumulated_volume >= target_volume {
                break;
            }
            value_area_prices.push(price);
            accumulated_volume += volume;
        }

        // Value Area High = max price in value area
        // Value Area Low = min price in value area
        if !value_area_prices.is_empty() {
            let vah = value_area_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let val = value_area_prices.iter().cloned().fold(f64::INFINITY, f64::min);
            self.cached_vah = Some(vah);
            self.cached_val = Some(val);
        } else {
            self.cached_vah = None;
            self.cached_val = None;
        }

        self.cache_dirty = false;
    }

    /// Reset the volume profile (called daily or per session)
    pub fn reset_profile(&mut self) {
        self.volume_by_price.clear();
        self.total_volume = 0.0;
        self.cached_vpoc = None;
        self.cached_vah = None;
        self.cached_val = None;
        self.cache_dirty = true;
    }

    /// Get total volume tracked
    pub fn total_volume(&self) -> f64 {
        self.total_volume
    }

    /// Get number of unique price levels
    pub fn price_level_count(&self) -> usize {
        self.volume_by_price.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpoc_calculation() {
        let mut profile = VolumeProfile::new(0.1);
        
        // Add trades at different price levels
        profile.update_trade(100.0, 50.0, 0.1);
        profile.update_trade(100.1, 30.0, 0.1);
        profile.update_trade(100.2, 100.0, 0.1); // Highest volume
        profile.update_trade(100.3, 20.0, 0.1);
        
        let vpoc = profile.get_vpoc().unwrap();
        assert!((vpoc - 100.2).abs() < 0.01, "VPOC should be 100.2, got {}", vpoc);
    }

    #[test]
    fn test_value_area() {
        let mut profile = VolumeProfile::new(1.0);
        
        // Create a distribution
        profile.update_trade(100.0, 10.0, 1.0);
        profile.update_trade(101.0, 20.0, 1.0);
        profile.update_trade(102.0, 40.0, 1.0); // VPOC
        profile.update_trade(103.0, 20.0, 1.0);
        profile.update_trade(104.0, 10.0, 1.0);
        
        let (val, vah) = profile.get_value_area().unwrap();
        
        // Value area should contain the 70% of volume around VPOC
        assert!(val <= 102.0, "VAL should be <= VPOC");
        assert!(vah >= 102.0, "VAH should be >= VPOC");
        assert!(val < vah, "VAL should be less than VAH");
    }

    #[test]
    fn test_reset_profile() {
        let mut profile = VolumeProfile::new(0.1);
        
        profile.update_trade(100.0, 50.0, 0.1);
        profile.update_trade(100.1, 30.0, 0.1);
        
        assert!(profile.get_vpoc().is_some());
        assert!(profile.total_volume() > 0.0);
        
        profile.reset_profile();
        
        assert!(profile.get_vpoc().is_none());
        assert_eq!(profile.total_volume(), 0.0);
        assert_eq!(profile.price_level_count(), 0);
    }

    #[test]
    fn test_tick_rounding() {
        let mut profile = VolumeProfile::new(0.5);
        
        // Prices should be rounded to nearest 0.5
        profile.update_trade(100.23, 10.0, 0.5);
        profile.update_trade(100.27, 20.0, 0.5);
        profile.update_trade(100.73, 15.0, 0.5);
        
        // Both 100.23 and 100.27 should round to 100.0
        // 100.73 should round to 101.0
        let vpoc = profile.get_vpoc().unwrap();
        // The level with most volume should be 100.0 (10 + 20 = 30)
        assert!((vpoc - 100.0).abs() < 0.01 || (vpoc - 101.0).abs() < 0.01);
    }
}
