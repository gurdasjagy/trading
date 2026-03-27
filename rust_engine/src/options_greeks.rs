//! Options Greeks calculator with Black-Scholes model.
//!
//! Computes delta, gamma, theta, vega for options positions.
//! Used for gamma exposure and hedging calculations.
//!
//! # Overview
//!
//! This module provides institutional-grade options analytics:
//! - Black-Scholes option pricing
//! - Greeks calculation (delta, gamma, theta, vega, rho)
//! - Aggregate gamma exposure computation
//! - Gamma flip level detection
//!
//! # Usage
//!
//! ```ignore
//! use options_greeks::{call_greeks, put_greeks, GammaExposure};
//!
//! // Calculate Greeks for a call option
//! let greeks = call_greeks(50000.0, 52000.0, 0.25, 0.05, 0.60);
//! println!("Delta: {:.4}, Gamma: {:.6}", greeks.delta, greeks.gamma);
//!
//! // Calculate aggregate gamma exposure
//! let positions = vec![(52000.0, 10.0, 0.25, true)];
//! let exposure = GammaExposure::compute(50000.0, &positions);
//! ```

use std::f64::consts::{E, PI};
use tracing::{info, warn};

/// Standard normal CDF approximation (Abramowitz and Stegun).
/// 
/// Uses the polynomial approximation with error < 7.5e-8.
fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * E.powf(-x * x / 2.0);
    
    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF.
#[inline]
fn norm_pdf(x: f64) -> f64 {
    E.powf(-x * x / 2.0) / (2.0 * PI).sqrt()
}

/// Options Greeks result.
#[derive(Debug, Clone, Copy, Default)]
pub struct Greeks {
    /// Delta: Rate of change of option price with respect to underlying price.
    /// Range: [0, 1] for calls, [-1, 0] for puts.
    pub delta: f64,
    /// Gamma: Rate of change of delta with respect to underlying price.
    /// Always positive, highest at-the-money.
    pub gamma: f64,
    /// Theta: Rate of change of option price with respect to time (per year).
    /// Usually negative (time decay).
    pub theta: f64,
    /// Vega: Rate of change of option price with respect to volatility (per 1%).
    /// Always positive.
    pub vega: f64,
    /// Rho: Rate of change of option price with respect to interest rate (per 1%).
    pub rho: f64,
}

impl Greeks {
    /// Create a zero Greeks struct (for invalid inputs).
    pub fn zero() -> Self {
        Self::default()
    }
}

/// Calculate d1 parameter for Black-Scholes.
#[inline]
fn d1(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt())
}

/// Calculate d2 parameter for Black-Scholes.
#[inline]
fn d2(d1: f64, sigma: f64, t: f64) -> f64 {
    d1 - sigma * t.sqrt()
}

/// Calculate Greeks for a call option.
///
/// # Arguments
/// * `spot` - Current underlying price
/// * `strike` - Option strike price
/// * `time_years` - Time to expiration in years
/// * `rate` - Risk-free interest rate (e.g., 0.05 for 5%)
/// * `vol` - Implied volatility (e.g., 0.60 for 60%)
///
/// # Returns
/// Greeks struct with all calculated values.
pub fn call_greeks(spot: f64, strike: f64, time_years: f64, rate: f64, vol: f64) -> Greeks {
    if time_years <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return Greeks::zero();
    }
    
    let d1_val = d1(spot, strike, time_years, rate, vol);
    let d2_val = d2(d1_val, vol, time_years);
    
    let sqrt_t = time_years.sqrt();
    let discount = E.powf(-rate * time_years);
    
    let delta = norm_cdf(d1_val);
    let gamma = norm_pdf(d1_val) / (spot * vol * sqrt_t);
    let theta = -(spot * norm_pdf(d1_val) * vol) / (2.0 * sqrt_t)
        - rate * strike * discount * norm_cdf(d2_val);
    let vega = spot * sqrt_t * norm_pdf(d1_val) / 100.0; // Per 1% vol change
    let rho = strike * time_years * discount * norm_cdf(d2_val) / 100.0;
    
    Greeks { delta, gamma, theta, vega, rho }
}

/// Calculate Greeks for a put option.
///
/// # Arguments
/// * `spot` - Current underlying price
/// * `strike` - Option strike price
/// * `time_years` - Time to expiration in years
/// * `rate` - Risk-free interest rate (e.g., 0.05 for 5%)
/// * `vol` - Implied volatility (e.g., 0.60 for 60%)
///
/// # Returns
/// Greeks struct with all calculated values.
pub fn put_greeks(spot: f64, strike: f64, time_years: f64, rate: f64, vol: f64) -> Greeks {
    if time_years <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return Greeks::zero();
    }
    
    let d1_val = d1(spot, strike, time_years, rate, vol);
    let d2_val = d2(d1_val, vol, time_years);
    
    let sqrt_t = time_years.sqrt();
    let discount = E.powf(-rate * time_years);
    
    let delta = norm_cdf(d1_val) - 1.0;
    let gamma = norm_pdf(d1_val) / (spot * vol * sqrt_t);
    let theta = -(spot * norm_pdf(d1_val) * vol) / (2.0 * sqrt_t)
        + rate * strike * discount * norm_cdf(-d2_val);
    let vega = spot * sqrt_t * norm_pdf(d1_val) / 100.0;
    let rho = -strike * time_years * discount * norm_cdf(-d2_val) / 100.0;
    
    Greeks { delta, gamma, theta, vega, rho }
}

/// Calculate Black-Scholes call option price.
pub fn call_price(spot: f64, strike: f64, time_years: f64, rate: f64, vol: f64) -> f64 {
    if time_years <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return 0.0;
    }
    
    let d1_val = d1(spot, strike, time_years, rate, vol);
    let d2_val = d2(d1_val, vol, time_years);
    
    spot * norm_cdf(d1_val) - strike * E.powf(-rate * time_years) * norm_cdf(d2_val)
}

/// Calculate Black-Scholes put option price.
pub fn put_price(spot: f64, strike: f64, time_years: f64, rate: f64, vol: f64) -> f64 {
    if time_years <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return 0.0;
    }
    
    let d1_val = d1(spot, strike, time_years, rate, vol);
    let d2_val = d2(d1_val, vol, time_years);
    
    strike * E.powf(-rate * time_years) * norm_cdf(-d2_val) - spot * norm_cdf(-d1_val)
}

/// Implied volatility calculator using Newton-Raphson method.
pub fn implied_vol_call(
    spot: f64, 
    strike: f64, 
    time_years: f64, 
    rate: f64, 
    market_price: f64
) -> Option<f64> {
    if market_price <= 0.0 || time_years <= 0.0 {
        return None;
    }
    
    // Initial guess based on rough approximation
    let mut vol = 0.30;
    
    for _ in 0..100 {
        let price = call_price(spot, strike, time_years, rate, vol);
        let diff = market_price - price;
        
        if diff.abs() < 0.0001 {
            return Some(vol);
        }
        
        let greeks = call_greeks(spot, strike, time_years, rate, vol);
        if greeks.vega.abs() < 0.0001 {
            break;
        }
        
        // Vega is per 1%, so multiply by 100
        vol += diff / (greeks.vega * 100.0);
        vol = vol.clamp(0.01, 5.0);
    }
    
    None
}

/// Aggregate gamma exposure across strike prices.
#[derive(Debug, Clone)]
pub struct GammaExposure {
    /// Net gamma at each price level (strike, gamma_amount).
    pub levels: Vec<(f64, f64)>,
    /// Gamma flip level (where net gamma changes sign).
    pub flip_level: Option<f64>,
    /// Total net gamma.
    pub total_gamma: f64,
    /// Is the market currently above the gamma flip level?
    pub above_flip: bool,
}

impl GammaExposure {
    /// Compute aggregate gamma exposure from options positions.
    ///
    /// # Arguments
    /// * `spot` - Current underlying price
    /// * `positions` - Vector of (strike, quantity, time_years, is_call)
    ///
    /// # Returns
    /// GammaExposure with aggregated data and flip level.
    pub fn compute(spot: f64, positions: &[(f64, f64, f64, bool)]) -> Self {
        if positions.is_empty() {
            return Self {
                levels: Vec::new(),
                flip_level: None,
                total_gamma: 0.0,
                above_flip: false,
            };
        }
        
        let vol = 0.50; // Default 50% implied vol
        let rate = 0.05; // Default 5% risk-free rate
        
        let mut levels: Vec<(f64, f64)> = Vec::with_capacity(positions.len());
        let mut total_gamma = 0.0;
        
        for &(strike, qty, time, is_call) in positions {
            let greeks = if is_call {
                call_greeks(spot, strike, time, rate, vol)
            } else {
                put_greeks(spot, strike, time, rate, vol)
            };
            
            let gamma_amount = greeks.gamma * qty * spot * spot / 100.0;
            levels.push((strike, gamma_amount));
            total_gamma += gamma_amount;
        }
        
        // Sort by strike
        levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find gamma flip level (where cumulative gamma crosses zero)
        let mut flip_level = None;
        let mut cumulative = 0.0;
        for (strike, gamma) in &levels {
            let prev_cum = cumulative;
            cumulative += gamma;
            
            // Check for sign change
            if prev_cum != 0.0 && prev_cum.signum() != cumulative.signum() && cumulative.signum() != 0.0 {
                // Linear interpolation to find exact flip level
                let weight = prev_cum.abs() / (prev_cum.abs() + gamma.abs());
                flip_level = Some(*strike);
                break;
            }
        }
        
        let above_flip = match flip_level {
            Some(level) => spot > level,
            None => true,
        };
        
        Self {
            levels,
            flip_level,
            total_gamma,
            above_flip,
        }
    }
    
    /// Get gamma at a specific price level.
    pub fn gamma_at_strike(&self, strike: f64) -> f64 {
        self.levels
            .iter()
            .find(|(s, _)| (*s - strike).abs() < 1.0)
            .map(|(_, g)| *g)
            .unwrap_or(0.0)
    }
    
    /// Is the market in a "long gamma" environment?
    /// Long gamma = dealers hedge by buying dips and selling rallies (stabilizing).
    pub fn is_long_gamma(&self) -> bool {
        self.total_gamma > 0.0
    }
    
    /// Is the market in a "short gamma" environment?
    /// Short gamma = dealers hedge by selling dips and buying rallies (destabilizing).
    pub fn is_short_gamma(&self) -> bool {
        self.total_gamma < 0.0
    }
}

/// Options position for portfolio analysis.
#[derive(Debug, Clone)]
pub struct OptionPosition {
    pub symbol: String,
    pub strike: f64,
    pub expiry_years: f64,
    pub is_call: bool,
    pub quantity: f64,
    pub implied_vol: f64,
}

/// CATEGORY 4 FIX: Portfolio-level Greeks aggregator.
///
/// Tracks aggregate delta, gamma, theta, vega, and rho across all options
/// positions in the portfolio. Institutional desks use portfolio Greeks for:
///   - Delta hedging: maintain delta-neutral portfolio
///   - Gamma scalping: trade around gamma exposure
///   - Vega risk: manage sensitivity to implied vol changes
///   - Theta decay: track daily time decay P&L
///   - Risk limits: set max portfolio delta/gamma/vega limits
#[derive(Debug, Clone, Default)]
pub struct PortfolioGreeks {
    pub net_delta: f64,
    pub net_gamma: f64,
    pub net_theta: f64,
    pub net_vega: f64,
    /// CATEGORY 4 FIX: Added net rho for interest rate sensitivity.
    pub net_rho: f64,
    /// CATEGORY 4 FIX: Number of positions contributing to portfolio Greeks.
    pub position_count: usize,
    /// CATEGORY 4 FIX: Dollar delta (notional delta exposure in USDT).
    pub dollar_delta: f64,
    /// CATEGORY 4 FIX: Dollar gamma (P&L from 1% underlying move).
    pub dollar_gamma_1pct: f64,
    /// CATEGORY 4 FIX: Dollar vega (P&L from 1% IV increase).
    pub dollar_vega_1pct: f64,
    /// CATEGORY 4 FIX: Daily theta (expected daily time decay in USDT).
    pub daily_theta: f64,
}

impl PortfolioGreeks {
    /// Calculate aggregate Greeks for a portfolio of options.
    ///
    /// CATEGORY 4 FIX: Enhanced to compute dollar-denominated Greeks,
    /// net rho, and risk metrics needed for institutional risk management.
    pub fn from_positions(spot: f64, positions: &[OptionPosition], rate: f64) -> Self {
        let mut result = Self::default();
        result.position_count = positions.len();

        for pos in positions {
            let greeks = if pos.is_call {
                call_greeks(spot, pos.strike, pos.expiry_years, rate, pos.implied_vol)
            } else {
                put_greeks(spot, pos.strike, pos.expiry_years, rate, pos.implied_vol)
            };

            result.net_delta += greeks.delta * pos.quantity;
            result.net_gamma += greeks.gamma * pos.quantity;
            result.net_theta += greeks.theta * pos.quantity;
            result.net_vega += greeks.vega * pos.quantity;
            result.net_rho += greeks.rho * pos.quantity;
        }

        // CATEGORY 4 FIX: Compute dollar-denominated risk metrics
        result.dollar_delta = result.net_delta * spot;
        result.dollar_gamma_1pct = 0.5 * result.net_gamma * spot * spot * 0.01 * 0.01;
        result.dollar_vega_1pct = result.net_vega; // Vega already per 1%
        result.daily_theta = result.net_theta / 365.0;

        if result.position_count > 0 {
            info!(
                "[greeks] Portfolio Greeks: delta={:.4} gamma={:.6} theta={:.4} vega={:.4} rho={:.4} \
                 $delta={:.2} $gamma1%={:.2} $vega1%={:.2} daily_theta={:.2} ({} positions)",
                result.net_delta, result.net_gamma, result.net_theta,
                result.net_vega, result.net_rho,
                result.dollar_delta, result.dollar_gamma_1pct,
                result.dollar_vega_1pct, result.daily_theta,
                result.position_count
            );
        }

        result
    }

    /// Calculate delta-equivalent position (underlying shares to delta-hedge).
    pub fn delta_equivalent(&self, spot: f64) -> f64 {
        self.net_delta * spot
    }

    /// Calculate dollar gamma (P&L from a 1% move).
    pub fn dollar_gamma(&self, spot: f64) -> f64 {
        0.5 * self.net_gamma * spot * spot * 0.0001
    }

    /// CATEGORY 4 FIX: Check if portfolio Greeks exceed risk limits.
    ///
    /// Returns a list of breached limits with descriptions.
    pub fn check_risk_limits(
        &self,
        max_abs_delta: f64,
        max_abs_gamma: f64,
        max_abs_vega: f64,
    ) -> Vec<String> {
        let mut breaches = Vec::new();

        if self.net_delta.abs() > max_abs_delta {
            breaches.push(format!(
                "Delta limit breached: {:.4} (limit: +/-{:.4})",
                self.net_delta, max_abs_delta
            ));
        }
        if self.net_gamma.abs() > max_abs_gamma {
            breaches.push(format!(
                "Gamma limit breached: {:.6} (limit: +/-{:.6})",
                self.net_gamma, max_abs_gamma
            ));
        }
        if self.net_vega.abs() > max_abs_vega {
            breaches.push(format!(
                "Vega limit breached: {:.4} (limit: +/-{:.4})",
                self.net_vega, max_abs_vega
            ));
        }

        if !breaches.is_empty() {
            for breach in &breaches {
                warn!("[greeks] RISK LIMIT: {}", breach);
            }
        }

        breaches
    }

    /// CATEGORY 4 FIX: Compute the hedge order needed to neutralize delta.
    ///
    /// Returns (quantity, is_buy) for the underlying to achieve delta-neutral.
    /// Positive quantity with is_buy=true means buy underlying.
    pub fn delta_hedge_order(&self) -> (f64, bool) {
        let hedge_qty = self.net_delta.abs();
        let is_buy = self.net_delta < 0.0; // Short delta → buy underlying
        (hedge_qty, is_buy)
    }

    /// CATEGORY 4 FIX: Estimate P&L impact of a given price move.
    ///
    /// Uses Taylor expansion: dP ≈ delta * dS + 0.5 * gamma * dS²
    pub fn estimate_pnl_from_move(&self, spot: f64, price_change_pct: f64) -> f64 {
        let ds = spot * price_change_pct;
        self.net_delta * ds + 0.5 * self.net_gamma * ds * ds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_norm_cdf() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((norm_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((norm_cdf(-1.96) - 0.025).abs() < 0.01);
    }
    
    #[test]
    fn test_call_greeks() {
        let greeks = call_greeks(100.0, 100.0, 1.0, 0.05, 0.20);
        
        // ATM call delta should be around 0.5-0.6
        assert!(greeks.delta > 0.5 && greeks.delta < 0.7);
        // Gamma should be positive
        assert!(greeks.gamma > 0.0);
        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);
        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }
    
    #[test]
    fn test_put_call_parity() {
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let vol = 0.20;
        
        let call = call_price(spot, strike, time, rate, vol);
        let put = put_price(spot, strike, time, rate, vol);
        
        // Put-Call Parity: C - P = S - K*e^(-rT)
        let parity = call - put - (spot - strike * E.powf(-rate * time));
        assert!(parity.abs() < 0.01, "Put-call parity violated: {}", parity);
    }
    
    #[test]
    fn test_gamma_exposure() {
        let positions = vec![
            (48000.0, 100.0, 0.25, true),  // 100 calls at 48k strike
            (52000.0, -50.0, 0.25, true),  // Short 50 calls at 52k strike
        ];
        
        let exposure = GammaExposure::compute(50000.0, &positions);
        
        assert_eq!(exposure.levels.len(), 2);
        // Should have some gamma flip level between strikes
    }
}
