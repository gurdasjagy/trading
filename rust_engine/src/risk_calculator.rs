//! Risk Calculator — rust_decimal-based precise risk computation engine.
//!
//! # Purpose
//!
//! This module provides institutional-grade risk calculations using `rust_decimal`
//! for EXACT arithmetic. Unlike the hot-path modules that use FixedPrice/FixedQty
//! (i64 with implicit scaling) for nanosecond latency, this module prioritizes
//! correctness over speed.
//!
//! # When to Use This Module
//!
//! - Pre-trade margin and notional checks (before placing orders)
//! - Portfolio-level risk aggregation (total exposure, VaR proxy)
//! - PnL calculations (realized and unrealized)
//! - Drawdown monitoring (daily, weekly, peak-to-trough)
//! - Fee calculations (maker/taker rebates and commissions)
//!
//! # When NOT to Use This Module
//!
//! - Orderbook operations (use FixedPrice/FixedQty)
//! - SPSC ring buffer messages (use raw i64)
//! - Tick-by-tick price comparison (use FixedPrice)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐         ┌──────────────────┐
//! │ Hot Path     │  i64    │  Risk Calculator  │
//! │ (FixedPrice) │ ──────▶ │  (rust_decimal)   │ ──▶ Risk Decision
//! └──────────────┘ convert └──────────────────┘
//! ```

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use crate::decimal_ext::*;

// ═══════════════════════════════════════════════════════════════════════════
// Portfolio Risk State
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks overall portfolio risk state with decimal precision.
pub struct PortfolioRiskState {
    /// Starting equity for the current trading day (USDT).
    pub day_start_equity: Decimal,
    /// Current equity (USDT).
    pub current_equity: Decimal,
    /// High water mark equity (for drawdown calculation).
    pub high_water_mark: Decimal,
    /// Total realized PnL today (USDT).
    pub daily_realized_pnl: Decimal,
    /// Total unrealized PnL across all open positions (USDT).
    pub unrealized_pnl: Decimal,
    /// Total fees paid today (USDT).
    pub daily_fees: Decimal,
    /// Total notional exposure across all positions (USDT).
    pub total_notional_exposure: Decimal,
    /// Maximum allowed daily drawdown (fraction, e.g., 0.05 = 5%).
    pub max_daily_drawdown: Decimal,
    /// Maximum allowed total exposure (USDT).
    pub max_total_exposure: Decimal,
    /// Maximum allowed leverage.
    pub max_leverage: u32,
    /// Number of consecutive losses.
    pub consecutive_losses: u32,
    /// Maximum allowed consecutive losses.
    pub max_consecutive_losses: u32,
}

impl PortfolioRiskState {
    /// Create a new portfolio risk state with the given starting equity.
    pub fn new(starting_equity_usdt: f64) -> Self {
        let equity = f64_to_decimal(starting_equity_usdt);
        Self {
            day_start_equity: equity,
            current_equity: equity,
            high_water_mark: equity,
            daily_realized_pnl: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            daily_fees: Decimal::ZERO,
            total_notional_exposure: Decimal::ZERO,
            max_daily_drawdown: Decimal::new(5, 2),  // 5%
            max_total_exposure: f64_to_decimal(50_000.0),
            max_leverage: 125,
            consecutive_losses: 0,
            max_consecutive_losses: 5,
        }
    }

    /// Update equity after a realized trade.
    pub fn on_trade_closed(&mut self, realized_pnl_fp: i64, fee_fp: i64) {
        let pnl = Decimal::new(realized_pnl_fp, 8);
        let fee = Decimal::new(fee_fp.abs(), 8);

        self.daily_realized_pnl += pnl;
        self.daily_fees += fee;
        self.current_equity += pnl - fee;

        // Update high water mark
        if self.current_equity > self.high_water_mark {
            self.high_water_mark = self.current_equity;
        }

        // Track consecutive losses
        if pnl < Decimal::ZERO {
            self.consecutive_losses += 1;
        } else if pnl > Decimal::ZERO {
            self.consecutive_losses = 0;
        }
    }

    /// Update unrealized PnL (called on every tick for open positions).
    pub fn update_unrealized(&mut self, total_unrealized_fp: i64) {
        self.unrealized_pnl = Decimal::new(total_unrealized_fp, 8);
    }

    /// Update total notional exposure.
    pub fn update_exposure(&mut self, total_notional_fp: i64) {
        self.total_notional_exposure = Decimal::new(total_notional_fp, 8);
    }

    /// Get the current daily drawdown as a fraction.
    pub fn daily_drawdown(&self) -> Decimal {
        if self.day_start_equity.is_zero() {
            return Decimal::ZERO;
        }
        let change = self.current_equity - self.day_start_equity;
        (change / self.day_start_equity).abs()
    }

    /// Get the peak-to-trough drawdown as a fraction.
    pub fn peak_drawdown(&self) -> Decimal {
        if self.high_water_mark.is_zero() {
            return Decimal::ZERO;
        }
        let drawdown = self.high_water_mark - self.current_equity;
        if drawdown > Decimal::ZERO {
            drawdown / self.high_water_mark
        } else {
            Decimal::ZERO
        }
    }

    /// Get the current effective leverage.
    pub fn current_leverage(&self) -> Decimal {
        if self.current_equity.is_zero() {
            return Decimal::ZERO;
        }
        self.total_notional_exposure / self.current_equity
    }

    /// Reset for a new trading day.
    pub fn reset_daily(&mut self) {
        self.day_start_equity = self.current_equity;
        self.daily_realized_pnl = Decimal::ZERO;
        self.daily_fees = Decimal::ZERO;
        self.consecutive_losses = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-Trade Risk Check Result
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a pre-trade risk check.
#[derive(Debug, Clone)]
pub enum RiskCheckResult {
    /// Trade is allowed.
    Approved {
        /// Adjusted position size (may be reduced by risk limits).
        adjusted_qty_fp: i64,
        /// Reason for any adjustment.
        note: String,
    },
    /// Trade is rejected.
    Rejected {
        /// Reason for rejection.
        reason: String,
    },
}

impl RiskCheckResult {
    /// Check if the trade was approved.
    pub fn is_approved(&self) -> bool {
        matches!(self, RiskCheckResult::Approved { .. })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Risk Check Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Perform a comprehensive pre-trade risk check.
///
/// Checks all risk limits and returns whether the trade is allowed.
/// If the trade exceeds limits, it may return a reduced quantity
/// or outright rejection.
pub fn pre_trade_risk_check(
    state: &PortfolioRiskState,
    price_fp: i64,
    qty_fp: i64,
    leverage: u32,
    _is_long: bool,
) -> RiskCheckResult {
    // 1. Check daily drawdown limit
    let dd = state.daily_drawdown();
    if dd > state.max_daily_drawdown {
        return RiskCheckResult::Rejected {
            reason: format!(
                "Daily drawdown {:.2}% exceeds max {:.2}%",
                decimal_to_f64(dd * Decimal::new(100, 0)),
                decimal_to_f64(state.max_daily_drawdown * Decimal::new(100, 0)),
            ),
        };
    }

    // 2. Check consecutive loss limit
    if state.consecutive_losses >= state.max_consecutive_losses {
        return RiskCheckResult::Rejected {
            reason: format!(
                "Consecutive losses {} >= max {}",
                state.consecutive_losses, state.max_consecutive_losses,
            ),
        };
    }

    // 3. Check leverage limit
    if leverage > state.max_leverage {
        return RiskCheckResult::Rejected {
            reason: format!("Leverage {} exceeds max {}", leverage, state.max_leverage),
        };
    }

    // 4. Calculate proposed trade notional
    let trade_notional = compute_notional_decimal(price_fp, qty_fp);

    // 5. Check if new exposure would exceed limit
    let new_exposure = state.total_notional_exposure + trade_notional;
    if new_exposure > state.max_total_exposure {
        // Try to reduce quantity to fit within exposure limit
        let remaining_capacity = state.max_total_exposure - state.total_notional_exposure;
        if remaining_capacity <= Decimal::ZERO {
            return RiskCheckResult::Rejected {
                reason: format!(
                    "Total exposure ${:.2} would exceed max ${:.2}",
                    decimal_to_f64(new_exposure),
                    decimal_to_f64(state.max_total_exposure),
                ),
            };
        }

        // Calculate reduced quantity
        let price = Decimal::new(price_fp, 8);
        if price.is_zero() {
            return RiskCheckResult::Rejected {
                reason: "Price is zero".to_string(),
            };
        }
        let max_qty_decimal = remaining_capacity / price;
        let max_qty_fp = decimal_to_fixed_qty(max_qty_decimal)
            .map(|q| q.raw())
            .unwrap_or(0);

        if max_qty_fp <= 0 {
            return RiskCheckResult::Rejected {
                reason: "Insufficient exposure capacity for minimum size".to_string(),
            };
        }

        return RiskCheckResult::Approved {
            adjusted_qty_fp: max_qty_fp,
            note: format!(
                "Qty reduced from {} to {} to fit exposure limit",
                qty_fp, max_qty_fp,
            ),
        };
    }

    // 6. Calculate margin requirement
    let margin = compute_margin_decimal(price_fp, qty_fp, leverage);
    let margin_f64 = decimal_to_f64(margin);
    let equity_f64 = decimal_to_f64(state.current_equity);
    if margin_f64 > equity_f64 * 0.9 {
        // Margin would use >90% of equity
        return RiskCheckResult::Rejected {
            reason: format!(
                "Margin requirement ${:.2} exceeds 90% of equity ${:.2}",
                margin_f64, equity_f64,
            ),
        };
    }

    // All checks passed
    RiskCheckResult::Approved {
        adjusted_qty_fp: qty_fp,
        note: format!(
            "notional=${:.2}, margin=${:.2}, leverage={}x",
            decimal_to_f64(trade_notional), margin_f64, leverage,
        ),
    }
}

/// Calculate the Kelly criterion optimal position fraction.
///
/// Given a win rate and average win/loss ratio, computes the optimal
/// fraction of equity to risk on each trade.
///
/// `f* = (p * b - q) / b` where:
///   - p = probability of winning
///   - q = probability of losing (1 - p)
///   - b = ratio of average win to average loss
pub fn kelly_fraction(win_rate: f64, avg_win_loss_ratio: f64) -> Decimal {
    let p = f64_to_decimal(win_rate.clamp(0.01, 0.99));
    let q = Decimal::ONE - p;
    let b = f64_to_decimal(avg_win_loss_ratio.max(0.01));

    let numerator = p * b - q;
    let fraction = numerator / b;

    // Apply half-Kelly for safety
    let half_kelly = fraction / Decimal::new(2, 0);
    half_kelly.max(Decimal::ZERO).min(Decimal::new(25, 2)) // Cap at 25%
}

/// Calculate the position size based on risk budget and stop distance.
///
/// `size = (equity * risk_fraction) / stop_distance`
///
/// Returns the position size in contracts (FixedQty scale).
pub fn risk_based_position_size(
    equity_usdt: f64,
    risk_fraction: f64,
    entry_price_fp: i64,
    stop_price_fp: i64,
) -> i64 {
    let equity = f64_to_decimal(equity_usdt);
    let risk_frac = f64_to_decimal(risk_fraction.clamp(0.001, 0.1));
    let entry = Decimal::new(entry_price_fp, 8);
    let stop = Decimal::new(stop_price_fp, 8);

    let stop_distance = (entry - stop).abs();
    if stop_distance.is_zero() || entry.is_zero() {
        return 0;
    }

    let risk_budget = equity * risk_frac;
    let size = risk_budget / stop_distance;

    // Convert to FixedQty (i64 with 1e4 scale)
    let scaled = size * Decimal::new(10_000, 0);
    scaled.to_i64().unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_risk_state_basic() {
        let state = PortfolioRiskState::new(10_000.0);
        assert_eq!(decimal_to_f64(state.current_equity), 10_000.0);
        assert_eq!(state.consecutive_losses, 0);
    }

    #[test]
    fn test_on_trade_closed_profit() {
        let mut state = PortfolioRiskState::new(10_000.0);
        // Profit of $100, fee of $1
        let pnl_fp = (100.0 * 1e8) as i64;
        let fee_fp = (1.0 * 1e8) as i64;
        state.on_trade_closed(pnl_fp, fee_fp);

        let equity = decimal_to_f64(state.current_equity);
        assert!(equity > 10_098.0 && equity < 10_100.0);
        assert_eq!(state.consecutive_losses, 0);
    }

    #[test]
    fn test_on_trade_closed_loss() {
        let mut state = PortfolioRiskState::new(10_000.0);
        let pnl_fp = (-50.0 * 1e8) as i64;
        let fee_fp = (1.0 * 1e8) as i64;
        state.on_trade_closed(pnl_fp, fee_fp);

        let equity = decimal_to_f64(state.current_equity);
        assert!(equity < 9_950.0);
        assert_eq!(state.consecutive_losses, 1);
    }

    #[test]
    fn test_risk_check_approved() {
        let state = PortfolioRiskState::new(10_000.0);
        let price_fp = (50_000.0 * 1e8) as i64;
        let qty_fp = (0.01 * 1e4) as i64; // 0.01 BTC = $500
        let result = pre_trade_risk_check(&state, price_fp, qty_fp, 5, true);
        assert!(result.is_approved());
    }

    #[test]
    fn test_risk_check_rejected_drawdown() {
        let mut state = PortfolioRiskState::new(10_000.0);
        state.max_daily_drawdown = Decimal::new(1, 2); // 1%
        // Simulate large loss
        state.on_trade_closed((-200.0 * 1e8) as i64, 0);

        let price_fp = (50_000.0 * 1e8) as i64;
        let qty_fp = (0.01 * 1e4) as i64;
        let result = pre_trade_risk_check(&state, price_fp, qty_fp, 5, true);
        assert!(!result.is_approved());
    }

    #[test]
    fn test_risk_check_rejected_consecutive_losses() {
        let mut state = PortfolioRiskState::new(10_000.0);
        state.max_consecutive_losses = 3;
        state.consecutive_losses = 3;

        let result = pre_trade_risk_check(
            &state, (50_000.0 * 1e8) as i64, (0.01 * 1e4) as i64, 5, true,
        );
        assert!(!result.is_approved());
    }

    #[test]
    fn test_kelly_fraction() {
        let kelly = kelly_fraction(0.55, 1.5);
        let k = decimal_to_f64(kelly);
        assert!(k > 0.0 && k < 0.25);
    }

    #[test]
    fn test_risk_based_position_size() {
        let size = risk_based_position_size(
            10_000.0,
            0.02,              // 2% risk
            (50_000.0 * 1e8) as i64,  // entry
            (49_500.0 * 1e8) as i64,  // stop (-1%)
        );
        assert!(size > 0);
    }

    #[test]
    fn test_peak_drawdown() {
        let mut state = PortfolioRiskState::new(10_000.0);
        // Gain
        state.on_trade_closed((500.0 * 1e8) as i64, 0);
        assert!(decimal_to_f64(state.high_water_mark) > 10_400.0);

        // Loss
        state.on_trade_closed((-300.0 * 1e8) as i64, 0);
        let dd = decimal_to_f64(state.peak_drawdown());
        assert!(dd > 0.0 && dd < 0.1);
    }
}
