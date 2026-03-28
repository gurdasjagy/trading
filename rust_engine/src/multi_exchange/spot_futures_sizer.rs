//! Spot-Futures Arbitrage: Capital Allocation & Position Sizing
//!
//! Balance-aware sizing that works with any capital ($50 or $50,000).
//! All position sizing uses `rust_decimal::Decimal` for financial precision.
//! No minimum balance thresholds -- if allocation is below exchange minimums,
//! the opportunity is skipped with a log message (never errors out).

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use tracing::{debug, info, warn};

use crate::config::SpotFuturesConfig;
use crate::multi_exchange::global_book::ExchangeId;
use crate::multi_exchange::spot_futures_specs::{SpotMarketSpec, FuturesContractSpec};

// ---------------------------------------------------------------------------
// Sizing Result
// ---------------------------------------------------------------------------

/// Result of position sizing calculation.
#[derive(Debug, Clone)]
pub struct SpotFuturesSizingResult {
    /// Base asset quantity to buy on Spot (e.g., 0.05 BTC).
    pub spot_qty: Decimal,
    /// Base asset quantity to short on Futures.
    pub futures_qty: Decimal,
    /// USDT allocated to the Spot leg.
    pub spot_usdt_allocated: Decimal,
    /// USDT allocated as Futures margin.
    pub futures_usdt_allocated: Decimal,
    /// Estimated liquidation price for the Futures short.
    pub estimated_liquidation_price: Decimal,
    /// The exchange chosen for this trade.
    pub exchange: ExchangeId,
}

// ---------------------------------------------------------------------------
// Sizer
// ---------------------------------------------------------------------------

/// Position sizer for Spot-Futures arbitrage.
/// Determines optimal position size given available balances and exchange specs.
pub struct SpotFuturesSizer {
    config: SpotFuturesConfig,
}

impl SpotFuturesSizer {
    /// Create a new sizer with the given configuration.
    pub fn new(config: SpotFuturesConfig) -> Self {
        Self { config }
    }

    /// Calculate position size for a spot-futures arb trade.
    ///
    /// # Arguments
    /// * `exchange` - Target exchange for both legs
    /// * `spot_spec` - Spot market specification
    /// * `futures_spec` - Futures contract specification
    /// * `spot_usdt_balance` - Available USDT in Spot wallet
    /// * `futures_usdt_balance` - Available USDT in Futures wallet
    /// * `spot_ask_price` - Current spot ask price (what we'd pay to buy)
    /// * `futures_bid_price` - Current futures bid price (what we'd receive to short)
    ///
    /// Returns `None` if the position is too small for exchange minimums.
    pub fn calculate_size(
        &self,
        exchange: ExchangeId,
        spot_spec: &SpotMarketSpec,
        futures_spec: &FuturesContractSpec,
        spot_usdt_balance: Decimal,
        futures_usdt_balance: Decimal,
        spot_ask_price: Decimal,
        futures_bid_price: Decimal,
    ) -> Option<SpotFuturesSizingResult> {
        if spot_ask_price.is_zero() || futures_bid_price.is_zero() {
            warn!("[spot-futures-sizer] Zero price detected, cannot size position");
            return None;
        }

        let leverage = Decimal::from(self.config.short_leverage);
        let max_position_pct = Decimal::from_str(&self.config.max_position_pct.to_string())
            .unwrap_or_else(|_| Decimal::new(90, 2));

        // Total capital across both wallets
        let total_capital = spot_usdt_balance + futures_usdt_balance;
        if total_capital.is_zero() {
            warn!("[spot-futures-sizer] Zero total capital, cannot size");
            return None;
        }

        // Capital allocation ratio based on leverage:
        // 1x leverage -> 50/50 split (spot_ratio = 0.5)
        // 2x leverage -> 66/33 split (spot_ratio = 0.66)
        // 3x leverage -> 75/25 split (spot_ratio = 0.75)
        let one = Decimal::ONE;
        let spot_ratio = leverage / (leverage + one);
        let futures_ratio = one - spot_ratio;

        // Apply max position percentage
        let spot_allocation = total_capital * spot_ratio * max_position_pct;
        let futures_allocation = total_capital * futures_ratio * max_position_pct;

        // Clamp to actual available balances
        let effective_spot_alloc = spot_allocation.min(spot_usdt_balance);
        let effective_futures_alloc = futures_allocation.min(futures_usdt_balance);

        // The limiting factor is whichever wallet has less relative to its need
        let spot_limited_qty = effective_spot_alloc / spot_ask_price;
        let futures_limited_qty = (effective_futures_alloc * leverage) / futures_bid_price;

        // Take the minimum to ensure both legs can be filled
        let raw_qty = spot_limited_qty.min(futures_limited_qty);

        // Floor to the LARGEST step_size across both Spot and Futures
        // This ensures the quantity is valid on both markets
        let unified_step = spot_spec.step_size.max(futures_spec.step_size);
        if unified_step.is_zero() {
            warn!("[spot-futures-sizer] Zero step size detected");
            return None;
        }

        let final_qty = (raw_qty / unified_step).floor() * unified_step;

        if final_qty.is_zero() {
            warn!(
                "[spot-futures-sizer] Position too small after rounding: raw_qty={}, step={}",
                raw_qty, unified_step
            );
            return None;
        }

        // Verify minimum order sizes
        if final_qty < spot_spec.min_qty {
            warn!(
                "[spot-futures-sizer] Below Spot min_qty: {} < {} on {}",
                final_qty, spot_spec.min_qty, exchange.name()
            );
            return None;
        }
        if final_qty < futures_spec.min_qty {
            warn!(
                "[spot-futures-sizer] Below Futures min_qty: {} < {} on {}",
                final_qty, futures_spec.min_qty, exchange.name()
            );
            return None;
        }

        // Verify minimum notional
        let notional = final_qty * spot_ask_price;
        if notional < spot_spec.min_notional {
            warn!(
                "[spot-futures-sizer] Below Spot min_notional: {} < {} on {}",
                notional, spot_spec.min_notional, exchange.name()
            );
            return None;
        }

        // Calculate actual allocations
        let actual_spot_usdt = final_qty * spot_ask_price;
        let actual_futures_usdt = (final_qty * futures_bid_price) / leverage;

        // Estimate liquidation price for isolated margin short:
        // liq_price = entry_price * (1 + 1/leverage)
        let liq_price = futures_bid_price * (one + one / leverage);

        // Warn if liquidation is too close
        let liq_buffer = Decimal::new(15, 1); // 1.5
        if liq_price < futures_bid_price * liq_buffer {
            warn!(
                "[spot-futures-sizer] Liquidation price too close: liq={} vs entry={} (leverage={}x)",
                liq_price, futures_bid_price, self.config.short_leverage
            );
        }

        info!(
            "[spot-futures-sizer] Sized position on {}: qty={}, spot_alloc=${}, futures_alloc=${}, liq_price={}",
            exchange.name(), final_qty, actual_spot_usdt, actual_futures_usdt, liq_price
        );

        Some(SpotFuturesSizingResult {
            spot_qty: final_qty,
            futures_qty: final_qty,
            spot_usdt_allocated: actual_spot_usdt,
            futures_usdt_allocated: actual_futures_usdt,
            estimated_liquidation_price: liq_price,
            exchange,
        })
    }
}
