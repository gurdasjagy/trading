//! Real-Time Value at Risk Engine using Historical Simulation.
//!
//! Maintains a rolling window of returns and computes VaR at 95%/99% confidence.
//! Updates on every position change.
//!
//! # VaR Methodologies
//!
//! This engine implements:
//! - **Historical Simulation**: Non-parametric, uses actual return distribution
//! - **CVaR (Expected Shortfall)**: Average loss beyond VaR threshold
//! - **Marginal VaR**: Sensitivity of portfolio VaR to position changes
//!
//! # Usage
//!
//! ```ignore
//! let mut var_engine = VaREngine::new();
//! var_engine.set_portfolio_value(100_000.0);
//! 
//! // Record returns from each period
//! var_engine.record_return(0.002);  // 0.2% return
//! var_engine.record_return(-0.001); // -0.1% return
//!
//! // Compute VaR metrics
//! let metrics = var_engine.compute();
//! println!("99% VaR: ${:.2}", metrics.var_99);
//! ```

use std::collections::VecDeque;
use tracing::{debug, warn};

/// Number of return observations to maintain in the rolling window.
/// 1000 observations at 1-minute frequency = ~16.7 hours of data.
const RETURN_WINDOW: usize = 1000;

/// Annualization factor for daily VaR.
const TRADING_DAYS: f64 = 252.0;

/// Maximum number of symbols supported.
const MAX_SYMBOLS: usize = 16;

/// VaR metrics computed by the engine.
#[derive(Debug, Clone, Default)]
pub struct VaRMetrics {
    /// Value at Risk at 95% confidence (USDT).
    pub var_95: f64,
    /// Value at Risk at 99% confidence (USDT).
    pub var_99: f64,
    /// Conditional VaR (Expected Shortfall) at 95% (USDT).
    pub cvar_95: f64,
    /// Conditional VaR at 99% (USDT).
    pub cvar_99: f64,
    /// Marginal VaR for incremental position sizing (USDT).
    pub marginal_var: f64,
    /// Component VaR breakdown by position.
    pub component_var: Vec<f64>,
    /// Number of observations used in calculation.
    pub observations: usize,
    /// Timestamp of last update (nanoseconds since epoch).
    pub last_update_ns: u64,
}

/// Real-time VaR calculation engine.
pub struct VaREngine {
    /// Rolling window of portfolio returns (as percentage).
    returns: VecDeque<f64>,
    /// Current portfolio value in USDT.
    portfolio_value: f64,
    /// Cached sorted returns for percentile calculation.
    sorted_returns: Vec<f64>,
    /// Flag indicating sorted_returns needs update.
    dirty: bool,
    /// Per-symbol volatility estimates (annualized).
    symbol_volatilities: [f64; MAX_SYMBOLS],
    /// Per-symbol current exposure (USDT).
    symbol_exposures: [f64; MAX_SYMBOLS],
    /// Correlation matrix (simplified: assume correlation = 0.5 for all pairs).
    avg_correlation: f64,
    /// Configuration: VaR limit as percentage of portfolio.
    var_limit_pct: f64,
    /// Configuration: Confidence level for primary VaR calculation.
    confidence_level: f64,
    /// Statistics: Total returns recorded.
    total_returns_recorded: u64,
}

impl VaREngine {
    /// Create a new VaR engine with default configuration.
    pub fn new() -> Self {
        Self {
            returns: VecDeque::with_capacity(RETURN_WINDOW),
            portfolio_value: 0.0,
            sorted_returns: Vec::with_capacity(RETURN_WINDOW),
            dirty: true,
            symbol_volatilities: [0.02; MAX_SYMBOLS], // Default 2% daily vol
            symbol_exposures: [0.0; MAX_SYMBOLS],
            avg_correlation: 0.5,
            var_limit_pct: 0.05, // 5% default VaR limit
            confidence_level: 0.99,
            total_returns_recorded: 0,
        }
    }
    
    /// Create a new VaR engine with custom configuration.
    pub fn with_config(var_limit_pct: f64, confidence_level: f64) -> Self {
        let mut engine = Self::new();
        engine.var_limit_pct = var_limit_pct.clamp(0.01, 0.50);
        engine.confidence_level = confidence_level.clamp(0.90, 0.999);
        engine
    }
    
    /// Record a new return observation.
    /// 
    /// # Arguments
    /// * `return_pct` - Portfolio return for the period (e.g., 0.001 = 0.1%)
    pub fn record_return(&mut self, return_pct: f64) {
        // Sanity check: reject unrealistic returns (>50% per period)
        if return_pct.abs() > 0.50 {
            warn!("Rejected unrealistic return: {:.4}", return_pct);
            return;
        }
        
        if self.returns.len() >= RETURN_WINDOW {
            self.returns.pop_front();
        }
        self.returns.push_back(return_pct);
        self.dirty = true;
        self.total_returns_recorded += 1;
    }
    
    /// Update portfolio value for VaR calculation.
    pub fn set_portfolio_value(&mut self, value: f64) {
        if value > 0.0 {
            self.portfolio_value = value;
        }
    }
    
    /// Update exposure for a specific symbol.
    pub fn set_symbol_exposure(&mut self, symbol_id: u16, exposure_usdt: f64) {
        if (symbol_id as usize) < MAX_SYMBOLS {
            self.symbol_exposures[symbol_id as usize] = exposure_usdt;
        }
    }
    
    /// Update volatility estimate for a symbol.
    /// 
    /// # Arguments
    /// * `symbol_id` - Symbol identifier (0-15)
    /// * `daily_vol` - Daily volatility (e.g., 0.02 = 2%)
    pub fn update_symbol_volatility(&mut self, symbol_id: u16, daily_vol: f64) {
        if (symbol_id as usize) < MAX_SYMBOLS {
            self.symbol_volatilities[symbol_id as usize] = daily_vol.clamp(0.001, 1.0);
        }
    }
    
    /// Set the average correlation assumption for portfolio VaR.
    pub fn set_average_correlation(&mut self, correlation: f64) {
        self.avg_correlation = correlation.clamp(-1.0, 1.0);
    }
    
    /// Compute VaR metrics using historical simulation.
    pub fn compute(&mut self) -> VaRMetrics {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Check for sufficient data
        if self.returns.len() < 30 {
            // Insufficient data - return conservative estimate based on parametric VaR
            let conservative_vol = 0.03; // Assume 3% daily volatility
            return VaRMetrics {
                var_95: self.portfolio_value * conservative_vol * 1.645,
                var_99: self.portfolio_value * conservative_vol * 2.326,
                cvar_95: self.portfolio_value * conservative_vol * 2.063,
                cvar_99: self.portfolio_value * conservative_vol * 2.665,
                marginal_var: self.portfolio_value * conservative_vol * 0.1,
                component_var: Vec::new(),
                observations: self.returns.len(),
                last_update_ns: now_ns,
            };
        }
        
        // Rebuild sorted returns if dirty
        if self.dirty {
            self.sorted_returns = self.returns.iter().cloned().collect();
            self.sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.dirty = false;
        }
        
        let n = self.sorted_returns.len();
        
        // VaR at 95% (5th percentile of losses)
        let idx_95 = ((n as f64) * 0.05).floor() as usize;
        let var_95_pct = -self.sorted_returns[idx_95].min(0.0);
        
        // VaR at 99% (1st percentile of losses)
        let idx_99 = ((n as f64) * 0.01).floor() as usize;
        let var_99_pct = -self.sorted_returns[idx_99].min(0.0);
        
        // CVaR: Average of returns below VaR threshold
        let cvar_95_pct = if idx_95 > 0 {
            self.sorted_returns[..=idx_95]
                .iter()
                .map(|r| (-r).max(0.0))
                .sum::<f64>() / (idx_95 + 1) as f64
        } else {
            var_95_pct
        };
        
        let cvar_99_pct = if idx_99 > 0 {
            self.sorted_returns[..=idx_99]
                .iter()
                .map(|r| (-r).max(0.0))
                .sum::<f64>() / (idx_99 + 1) as f64
        } else {
            var_99_pct
        };
        
        // Scale from 1-period to 1-day (sqrt(N) rule)
        // Assuming 1-minute returns, scale to daily
        let periods_per_day = 1440.0; // Minutes per day
        let scale_factor = (periods_per_day / n as f64).sqrt().min(10.0);
        
        // Calculate component VaR for each position
        let component_var = self.calculate_component_var(var_99_pct);
        
        // Marginal VaR: Sensitivity to additional exposure
        let marginal_var = self.calculate_marginal_var();
        
        VaRMetrics {
            var_95: self.portfolio_value * var_95_pct * scale_factor,
            var_99: self.portfolio_value * var_99_pct * scale_factor,
            cvar_95: self.portfolio_value * cvar_95_pct * scale_factor,
            cvar_99: self.portfolio_value * cvar_99_pct * scale_factor,
            marginal_var,
            component_var,
            observations: n,
            last_update_ns: now_ns,
        }
    }
    
    /// Calculate component VaR for each position.
    fn calculate_component_var(&self, portfolio_var_pct: f64) -> Vec<f64> {
        let total_exposure: f64 = self.symbol_exposures.iter().sum();
        if total_exposure <= 0.0 {
            return Vec::new();
        }
        
        self.symbol_exposures
            .iter()
            .enumerate()
            .filter(|(_, &exp)| exp > 0.0)
            .map(|(i, &exposure)| {
                let weight = exposure / total_exposure;
                let vol = self.symbol_volatilities[i];
                // Simplified: component VaR proportional to weight and volatility
                self.portfolio_value * portfolio_var_pct * weight * (vol / 0.02)
            })
            .collect()
    }
    
    /// Calculate marginal VaR for position sizing.
    fn calculate_marginal_var(&self) -> f64 {
        // Use the average symbol volatility weighted by exposure
        let total_exposure: f64 = self.symbol_exposures.iter().sum();
        if total_exposure <= 0.0 {
            return self.portfolio_value * 0.02 * 2.326 * 0.01;
        }
        
        let weighted_vol: f64 = self.symbol_exposures
            .iter()
            .enumerate()
            .map(|(i, &exp)| exp * self.symbol_volatilities[i])
            .sum::<f64>() / total_exposure;
        
        // Marginal VaR: derivative of portfolio VaR with respect to position size
        // Approximated as: Vol * Z-score * correlation factor
        let z_score_99 = 2.326;
        let correlation_factor = (1.0 + (MAX_SYMBOLS as f64 - 1.0) * self.avg_correlation).sqrt();
        
        weighted_vol * z_score_99 / correlation_factor * 0.01 * total_exposure
    }
    
    /// Check if a proposed position would breach VaR limits.
    ///
    /// # Arguments
    /// * `additional_exposure` - Additional exposure in USDT
    ///
    /// # Returns
    /// * `true` if the position is within VaR limits
    /// * `false` if adding the position would breach limits
    pub fn check_position(&mut self, additional_exposure: f64) -> bool {
        let metrics = self.compute();
        
        if self.portfolio_value <= 0.0 {
            return additional_exposure < 10000.0; // Allow small positions with no portfolio value
        }
        
        // Estimate new VaR with additional exposure
        let exposure_factor = 1.0 + additional_exposure / self.portfolio_value;
        let proposed_var = metrics.var_99 * exposure_factor;
        
        let limit = self.portfolio_value * self.var_limit_pct;
        let within_limit = proposed_var <= limit;
        
        if !within_limit {
            debug!(
                "VaR check failed: proposed=${:.2} > limit=${:.2} ({}%)",
                proposed_var, limit, self.var_limit_pct * 100.0
            );
        }
        
        within_limit
    }
    
    /// Get the maximum position size that stays within VaR limits.
    pub fn max_position_size(&mut self) -> f64 {
        let metrics = self.compute();
        
        if self.portfolio_value <= 0.0 || metrics.var_99 <= 0.0 {
            return 0.0;
        }
        
        let limit = self.portfolio_value * self.var_limit_pct;
        let headroom = (limit - metrics.var_99).max(0.0);
        
        // Estimate how much additional exposure fits in headroom
        // Using marginal VaR for sizing
        if metrics.marginal_var > 0.0 {
            (headroom / metrics.marginal_var) * self.portfolio_value * 0.01
        } else {
            headroom * 10.0 // Conservative fallback
        }
    }
    
    /// Get current VaR utilization as a percentage.
    pub fn var_utilization(&mut self) -> f64 {
        let metrics = self.compute();
        let limit = self.portfolio_value * self.var_limit_pct;
        
        if limit > 0.0 {
            (metrics.var_99 / limit * 100.0).min(100.0)
        } else {
            0.0
        }
    }
    
    /// Get summary statistics.
    pub fn stats(&self) -> VaRStats {
        VaRStats {
            observations: self.returns.len(),
            portfolio_value: self.portfolio_value,
            var_limit_pct: self.var_limit_pct,
            confidence_level: self.confidence_level,
            total_returns_recorded: self.total_returns_recorded,
        }
    }
    
    /// Reset all state.
    pub fn reset(&mut self) {
        self.returns.clear();
        self.sorted_returns.clear();
        self.dirty = true;
        self.symbol_exposures = [0.0; MAX_SYMBOLS];
    }
}

impl Default for VaREngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for the VaR engine.
#[derive(Debug, Clone)]
pub struct VaRStats {
    pub observations: usize,
    pub portfolio_value: f64,
    pub var_limit_pct: f64,
    pub confidence_level: f64,
    pub total_returns_recorded: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_var_engine_basic() {
        let mut engine = VaREngine::new();
        engine.set_portfolio_value(100_000.0);
        
        // Generate synthetic returns
        for i in 0..100 {
            let ret = if i % 10 == 0 { -0.02 } else { 0.001 };
            engine.record_return(ret);
        }
        
        let metrics = engine.compute();
        assert!(metrics.var_99 > 0.0, "VaR should be positive");
        assert!(metrics.cvar_99 >= metrics.var_99, "CVaR should be >= VaR");
        assert_eq!(metrics.observations, 100);
    }
    
    #[test]
    fn test_var_position_check() {
        let mut engine = VaREngine::with_config(0.05, 0.99);
        engine.set_portfolio_value(100_000.0);
        
        // Add stable returns
        for _ in 0..50 {
            engine.record_return(0.001);
        }
        
        // Should allow small positions
        assert!(engine.check_position(5_000.0));
        
        // Very large positions should be rejected
        // (this depends on the return distribution)
    }
    
    #[test]
    fn test_conservative_estimate() {
        let mut engine = VaREngine::new();
        engine.set_portfolio_value(100_000.0);
        
        // With insufficient data, should return conservative estimate
        engine.record_return(0.01);
        let metrics = engine.compute();
        
        assert!(metrics.var_99 > 0.0, "Should have positive VaR estimate");
        assert_eq!(metrics.observations, 1);
    }
}
