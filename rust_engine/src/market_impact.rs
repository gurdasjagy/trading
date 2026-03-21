//! Market Impact Model — Pre-Trade Estimation of Execution Cost.
//!
//! Implements a simplified Almgren-Chriss market impact model combined
//! with Kyle's lambda for real-time impact estimation from live orderbook data.
//!
//! Components:
//!   - **Temporary impact**: Price moves during execution (linear in rate)
//!   - **Permanent impact**: Price moves that persist after execution
//!   - **Orderbook-based liquidity**: Real depth analysis from FlatOrderBook
//!
//! Usage:
//!   Before every order, call `estimate_impact()` with the proposed order
//!   size and current orderbook state. Adjust order sizing or execution
//!   strategy (TWAP/VWAP split) if impact exceeds threshold.
//!
//! # Performance
//! All computations are O(1) with no allocation. Designed to run inline
//! in the execution router's hot path.


/// Market impact estimation result.
#[derive(Debug, Clone, Copy)]
pub struct ImpactEstimate {
    /// Temporary impact in basis points (execution cost that reverts).
    pub temporary_bps: f64,
    /// Permanent impact in basis points (persistent price displacement).
    pub permanent_bps: f64,
    /// Total expected impact in basis points.
    pub total_bps: f64,
    /// Estimated cost in USD for this order.
    pub cost_usd: f64,
    /// Liquidity ratio: order_size / available_depth (0-1+).
    pub liquidity_ratio: f64,
    /// Whether the order should be split (TWAPed).
    pub should_split: bool,
    /// Recommended number of child orders if splitting.
    pub recommended_slices: u32,
}

/// Per-symbol market impact parameters, calibrated from historical data.
#[derive(Debug, Clone, Copy)]
pub struct ImpactParams {
    /// Kyle's lambda: permanent impact coefficient.
    /// Impact = lambda * sqrt(order_size / daily_volume)
    pub lambda: f64,
    /// Temporary impact coefficient (related to bid-ask spread).
    pub eta: f64,
    /// Estimated daily volume in USD (used for normalization).
    pub daily_volume_usd: f64,
    /// Average spread in basis points (for baseline cost).
    pub avg_spread_bps: f64,
    /// Maximum acceptable impact before recommending TWAP.
    pub max_single_order_impact_bps: f64,
}

impl Default for ImpactParams {
    fn default() -> Self {
        Self {
            lambda: 0.1,          // Moderate permanent impact
            eta: 0.05,            // Half of lambda for temporary
            daily_volume_usd: 50_000_000.0, // $50M daily volume (BTC)
            avg_spread_bps: 2.0,
            max_single_order_impact_bps: 5.0,
        }
    }
}

impl ImpactParams {
    /// Create parameters for a high-liquidity asset (BTC, ETH).
    pub fn high_liquidity() -> Self {
        Self {
            lambda: 0.05,
            eta: 0.02,
            daily_volume_usd: 200_000_000.0,
            avg_spread_bps: 1.0,
            max_single_order_impact_bps: 3.0,
        }
    }

    /// Create parameters for a medium-liquidity asset.
    pub fn medium_liquidity() -> Self {
        Self {
            lambda: 0.15,
            eta: 0.08,
            daily_volume_usd: 10_000_000.0,
            avg_spread_bps: 3.0,
            max_single_order_impact_bps: 8.0,
        }
    }

    /// Create parameters for a low-liquidity asset.
    pub fn low_liquidity() -> Self {
        Self {
            lambda: 0.3,
            eta: 0.15,
            daily_volume_usd: 1_000_000.0,
            avg_spread_bps: 10.0,
            max_single_order_impact_bps: 15.0,
        }
    }

    /// Create parameters for forex pairs (very high liquidity).
    pub fn forex() -> Self {
        Self {
            lambda: 0.02,
            eta: 0.01,
            daily_volume_usd: 500_000_000.0,
            avg_spread_bps: 0.5,
            max_single_order_impact_bps: 2.0,
        }
    }
}

/// Market Impact Estimator.
///
/// Runs inline in the execution path. All operations are O(1), no heap allocation.
pub struct MarketImpactModel {
    /// Default parameters (overridden per-symbol if calibrated).
    default_params: ImpactParams,
    /// Per-symbol overrides (indexed by symbol_id).
    symbol_params: Vec<Option<ImpactParams>>,
}

impl MarketImpactModel {
    pub fn new(max_symbols: usize) -> Self {
        Self {
            default_params: ImpactParams::default(),
            symbol_params: vec![None; max_symbols],
        }
    }

    /// Set calibrated parameters for a specific symbol.
    pub fn set_symbol_params(&mut self, symbol_idx: usize, params: ImpactParams) {
        if symbol_idx < self.symbol_params.len() {
            self.symbol_params[symbol_idx] = Some(params);
        }
    }

    /// Estimate market impact for a proposed order.
    ///
    /// # Parameters
    /// - `symbol_idx`: Symbol index in the registry
    /// - `order_size_usd`: Order notional in USD
    /// - `mid_price`: Current mid price
    /// - `bid_depth_usd`: Total visible bid depth in USD (top N levels)
    /// - `ask_depth_usd`: Total visible ask depth in USD (top N levels)
    /// - `spread_bps`: Current bid-ask spread in basis points
    /// - `is_buy`: Whether this is a buy order
    #[inline]
    pub fn estimate_impact(
        &self,
        symbol_idx: usize,
        order_size_usd: f64,
        _mid_price: f64,
        bid_depth_usd: f64,
        ask_depth_usd: f64,
        spread_bps: f64,
        is_buy: bool,
    ) -> ImpactEstimate {
        let params = self.symbol_params.get(symbol_idx)
            .and_then(|p| p.as_ref())
            .unwrap_or(&self.default_params);

        // Liquidity on the relevant side
        let available_depth = if is_buy { ask_depth_usd } else { bid_depth_usd };
        let liquidity_ratio = if available_depth > 0.0 {
            order_size_usd / available_depth
        } else {
            10.0 // Very illiquid
        };

        // Participation rate (fraction of daily volume)
        let participation_rate = order_size_usd / params.daily_volume_usd.max(1.0);

        // Permanent impact: lambda * sqrt(participation_rate) * 10000 (to bps)
        let permanent_bps = params.lambda * participation_rate.sqrt() * 10000.0;

        // Temporary impact: eta * (order_size / depth) * 10000 + half-spread cost
        let depth_impact = params.eta * liquidity_ratio * 10000.0;
        let half_spread = spread_bps / 2.0;
        let temporary_bps = depth_impact + half_spread;

        let total_bps = permanent_bps + temporary_bps;
        let cost_usd = (total_bps / 10000.0) * order_size_usd;

        // Determine if we should split the order
        let should_split = total_bps > params.max_single_order_impact_bps;
        let recommended_slices = if should_split {
            // Split so each slice has impact <= max_single_order_impact_bps
            let ratio = total_bps / params.max_single_order_impact_bps;
            (ratio.ceil() as u32).max(2).min(20) // 2-20 slices
        } else {
            1
        };

        ImpactEstimate {
            temporary_bps,
            permanent_bps,
            total_bps,
            cost_usd,
            liquidity_ratio,
            should_split,
            recommended_slices,
        }
    }

    /// Quick check: should this order be sent as-is or TWAPed?
    #[inline]
    pub fn should_twap(
        &self,
        symbol_idx: usize,
        order_size_usd: f64,
        available_depth_usd: f64,
    ) -> bool {
        let params = self.symbol_params.get(symbol_idx)
            .and_then(|p| p.as_ref())
            .unwrap_or(&self.default_params);

        let participation = order_size_usd / params.daily_volume_usd.max(1.0);
        let depth_ratio = order_size_usd / available_depth_usd.max(1.0);

        // TWAP if we're taking >2% of visible depth or >0.1% of daily volume
        depth_ratio > 0.02 || participation > 0.001
    }

    /// Update daily volume estimate for a symbol (call periodically).
    pub fn update_daily_volume(&mut self, symbol_idx: usize, volume_usd: f64) {
        if let Some(Some(params)) = self.symbol_params.get_mut(symbol_idx) {
            params.daily_volume_usd = volume_usd;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_order_low_impact() {
        let model = MarketImpactModel::new(4);
        let impact = model.estimate_impact(
            0,
            1000.0,    // $1000 order
            50000.0,   // BTC at $50k
            500000.0,  // $500k bid depth
            500000.0,  // $500k ask depth
            2.0,       // 2 bps spread
            true,
        );
        assert!(impact.total_bps < 10.0, "Small order should have low impact: {:.2} bps", impact.total_bps);
    }

    #[test]
    fn test_large_order_high_impact() {
        let model = MarketImpactModel::new(4);
        let impact = model.estimate_impact(
            0,
            100000.0,  // $100k order
            50000.0,
            50000.0,   // Only $50k depth (2x order vs depth)
            50000.0,
            5.0,
            true,
        );
        assert!(impact.total_bps > 5.0, "Large order should have high impact: {:.2} bps", impact.total_bps);
        assert!(impact.should_split);
        assert!(impact.recommended_slices >= 2);
    }

    #[test]
    fn test_forex_params() {
        let mut model = MarketImpactModel::new(4);
        model.set_symbol_params(0, ImpactParams::forex());
        let impact = model.estimate_impact(
            0,
            10000.0,      // $10k order
            1.0850,       // EUR/USD
            1000000.0,    // $1M depth
            1000000.0,
            0.5,
            true,
        );
        assert!(impact.total_bps < 5.0, "Forex should have very low impact: {:.2} bps", impact.total_bps);
    }
}
