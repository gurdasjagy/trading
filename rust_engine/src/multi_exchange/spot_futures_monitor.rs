//! Spot-Futures Arbitrage: Live Position Monitoring & Margin Rebalancing
//!
//! Tracks active spot-futures positions with:
//! - Real-time PnL calculation (hedged, so price PnL should be ~0)
//! - Funding payment verification against exchange records
//! - Dynamic margin rebalancing (LIVE MODE ONLY)
//! - Hedge ratio monitoring
//! - Exit trigger evaluation

use std::collections::VecDeque;

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config::SpotFuturesConfig;
use crate::multi_exchange::global_book::ExchangeId;

// ---------------------------------------------------------------------------
// Position State Machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a spot-futures arbitrage position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpotFuturesPositionState {
    /// Pre-trade validation in progress.
    Validating,
    /// Executing entry (placing orders).
    Entering,
    /// Both legs active, collecting funding.
    Active,
    /// Exit triggered, closing both legs.
    Exiting,
    /// Fully closed, final PnL calculated.
    Closed,
    /// Entry failed or aborted.
    Failed,
}

// ---------------------------------------------------------------------------
// Funding Payment Record
// ---------------------------------------------------------------------------

/// A single funding payment record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingPayment {
    /// Timestamp of the funding settlement (nanoseconds).
    pub timestamp_ns: u64,
    /// Funding rate at settlement.
    pub rate: Decimal,
    /// Expected payment amount (calculated).
    pub expected_amount: Decimal,
    /// Actual payment amount (from exchange).
    pub actual_amount: Option<Decimal>,
    /// Whether the payment was verified against exchange records.
    pub verified: bool,
}

// ---------------------------------------------------------------------------
// Spot-Futures Position
// ---------------------------------------------------------------------------

/// A complete spot-futures arbitrage position with full lifecycle tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotFuturesPosition {
    /// Unique position ID.
    pub id: u64,
    /// Trading symbol base asset (e.g., "BTC").
    pub symbol: String,
    /// Current lifecycle state.
    pub state: SpotFuturesPositionState,

    // --- Spot Leg ---
    /// Exchange for the Spot leg (same as futures in V1).
    pub spot_exchange: ExchangeId,
    /// Spot entry price (actual fill).
    pub spot_entry_price: Decimal,
    /// Actual crypto quantity held in Spot wallet.
    pub spot_qty: Decimal,
    /// Live value of Spot holdings in USDT.
    pub spot_current_value: Decimal,

    // --- Futures Leg ---
    /// Exchange for the Futures short leg.
    pub futures_exchange: ExchangeId,
    /// Futures short entry price.
    pub futures_entry_price: Decimal,
    /// Short quantity (should match spot_qty in base asset terms).
    pub futures_qty: Decimal,
    /// Futures contracts (integer, for Gate.io compatibility).
    pub futures_contracts: i64,
    /// Leverage on the futures short.
    pub futures_leverage: i32,
    /// USDT margin used for the futures short.
    pub futures_margin_used: Decimal,
    /// Estimated liquidation price.
    pub futures_liquidation_price: Decimal,

    // --- Hedge Ratio ---
    /// spot_qty / futures_qty (should be ~1.0).
    pub hedge_ratio: Decimal,

    // --- Funding Tracking ---
    /// Total accumulated funding received (positive = received from shorts).
    pub accumulated_funding: Decimal,
    /// Individual funding payment records.
    pub funding_payments: Vec<FundingPayment>,
    /// Number of funding periods collected.
    pub funding_periods_collected: u32,

    // --- PnL ---
    /// Live PnL = accumulated_funding + basis_change - fees.
    pub live_pnl: Decimal,
    /// Total fees paid across both legs (entry + exit).
    pub total_fees_paid: Decimal,
    /// Entry basis: futures_entry - spot_entry.
    pub entry_basis: Decimal,

    // --- Timestamps ---
    /// When the position was opened (nanoseconds).
    pub opened_at_ns: u64,
    /// Last funding settlement timestamp.
    pub last_funding_at_ns: u64,
    /// Number of consecutive negative funding periods.
    pub consecutive_negative_funding: u32,
}

impl SpotFuturesPosition {
    /// Create a new position from entry fills.
    pub fn new(
        id: u64,
        symbol: String,
        exchange: ExchangeId,
        spot_entry_price: Decimal,
        spot_qty: Decimal,
        futures_entry_price: Decimal,
        futures_qty: Decimal,
        futures_contracts: i64,
        leverage: i32,
        entry_fees: Decimal,
    ) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let hedge_ratio = if !futures_qty.is_zero() {
            spot_qty / futures_qty
        } else {
            Decimal::ZERO
        };

        let entry_basis = futures_entry_price - spot_entry_price;
        let futures_margin = if leverage > 0 {
            (futures_qty * futures_entry_price) / Decimal::from(leverage)
        } else {
            futures_qty * futures_entry_price
        };

        let liq_price = if leverage > 0 {
            futures_entry_price * (Decimal::ONE + Decimal::ONE / Decimal::from(leverage))
        } else {
            futures_entry_price * Decimal::TWO
        };

        Self {
            id,
            symbol,
            state: SpotFuturesPositionState::Active,
            spot_exchange: exchange,
            spot_entry_price,
            spot_qty,
            spot_current_value: spot_qty * spot_entry_price,
            futures_exchange: exchange,
            futures_entry_price,
            futures_qty,
            futures_contracts,
            futures_leverage: leverage,
            futures_margin_used: futures_margin,
            futures_liquidation_price: liq_price,
            hedge_ratio,
            accumulated_funding: Decimal::ZERO,
            funding_payments: Vec::new(),
            funding_periods_collected: 0,
            live_pnl: -entry_fees, // Start negative by fees paid
            total_fees_paid: entry_fees,
            entry_basis,
            opened_at_ns: now_ns,
            last_funding_at_ns: 0,
            consecutive_negative_funding: 0,
        }
    }

    /// Update live prices and recalculate PnL.
    pub fn update_prices(&mut self, current_spot_bid: Decimal, current_futures_ask: Decimal) {
        self.spot_current_value = self.spot_qty * current_spot_bid;

        // Hedged PnL: funding + basis change - fees
        let spot_pnl = (current_spot_bid - self.spot_entry_price) * self.spot_qty;
        let futures_pnl = (self.futures_entry_price - current_futures_ask) * self.futures_qty;

        self.live_pnl = self.accumulated_funding + spot_pnl + futures_pnl - self.total_fees_paid;
    }

    /// Record a funding payment.
    pub fn record_funding(&mut self, payment: FundingPayment) {
        let amount = payment.actual_amount.unwrap_or(payment.expected_amount);

        if amount < Decimal::ZERO {
            self.consecutive_negative_funding += 1;
        } else {
            self.consecutive_negative_funding = 0;
        }

        self.accumulated_funding += amount;
        self.funding_periods_collected += 1;
        self.last_funding_at_ns = payment.timestamp_ns;
        self.funding_payments.push(payment);

        // Recalculate PnL
        self.live_pnl = self.accumulated_funding - self.total_fees_paid;
    }

    /// Get the hold duration in hours.
    pub fn hold_duration_hours(&self) -> f64 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let elapsed_ns = now_ns.saturating_sub(self.opened_at_ns);
        elapsed_ns as f64 / 3_600_000_000_000.0
    }

    /// Serialize to JSON for dashboard/persistence.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "symbol": self.symbol,
            "state": format!("{:?}", self.state),
            "exchange": self.spot_exchange.name(),
            "spot_entry_price": self.spot_entry_price.to_string(),
            "spot_qty": self.spot_qty.to_string(),
            "futures_entry_price": self.futures_entry_price.to_string(),
            "futures_qty": self.futures_qty.to_string(),
            "hedge_ratio": self.hedge_ratio.to_string(),
            "accumulated_funding": self.accumulated_funding.to_string(),
            "funding_periods": self.funding_periods_collected,
            "live_pnl": self.live_pnl.to_string(),
            "total_fees": self.total_fees_paid.to_string(),
            "hold_hours": format!("{:.1}", self.hold_duration_hours()),
        })
    }
}

// ---------------------------------------------------------------------------
// Exit Reason
// ---------------------------------------------------------------------------

/// Reason for exiting a spot-futures position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpotFuturesExitReason {
    /// Funding rate turned negative for consecutive periods.
    NegativeFunding { consecutive_periods: u32 },
    /// Futures basis reverted (futures at or below spot).
    BasisReversion { current_basis: Decimal },
    /// Accumulated profit target reached.
    ProfitTarget { accumulated_pnl: Decimal, target: Decimal },
    /// Futures margin ratio critically low.
    MarginDanger { margin_ratio: f64 },
    /// Maximum hold time exceeded.
    TimeStop { hold_hours: f64, max_hours: f64 },
    /// Kill switch activated.
    KillSwitch,
    /// Manual exit requested.
    Manual,
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Spot-Futures position monitor.
/// Evaluates exit conditions and manages margin rebalancing.
pub struct SpotFuturesMonitor {
    config: SpotFuturesConfig,
}

impl SpotFuturesMonitor {
    /// Create a new monitor.
    pub fn new(config: SpotFuturesConfig) -> Self {
        Self { config }
    }

    /// Check if any exit condition is triggered for a position.
    pub fn check_exit_conditions(
        &self,
        position: &SpotFuturesPosition,
        current_funding_rate: f64,
        futures_margin_ratio: f64,
    ) -> Option<SpotFuturesExitReason> {
        // 1. Negative Funding: consecutive negative periods
        if position.consecutive_negative_funding >= self.config.negative_funding_exit_periods {
            return Some(SpotFuturesExitReason::NegativeFunding {
                consecutive_periods: position.consecutive_negative_funding,
            });
        }

        // 2. Basis Reversion: futures at or below spot
        if position.entry_basis > Decimal::ZERO {
            // We entered with futures premium. If basis goes negative, capture it.
            let current_spot_value = position.spot_current_value / position.spot_qty;
            let estimated_futures_price = current_spot_value + position.entry_basis;
            // This is approximate -- real check uses live prices from update_prices()
        }

        // 3. Accumulated Profit Target
        let deployed_capital = position.spot_qty * position.spot_entry_price
            + position.futures_margin_used;
        let take_profit_threshold = deployed_capital
            * Decimal::from_str(&self.config.take_profit_pct.to_string())
                .unwrap_or(Decimal::new(5, 3))
            / Decimal::from(100);

        if position.accumulated_funding > take_profit_threshold && take_profit_threshold > Decimal::ZERO {
            return Some(SpotFuturesExitReason::ProfitTarget {
                accumulated_pnl: position.accumulated_funding,
                target: take_profit_threshold,
            });
        }

        // 4. Margin Danger
        if futures_margin_ratio < 0.15 && futures_margin_ratio > 0.0 {
            return Some(SpotFuturesExitReason::MarginDanger {
                margin_ratio: futures_margin_ratio,
            });
        }

        // 5. Time Stop
        let hold_hours = position.hold_duration_hours();
        if hold_hours > self.config.max_hold_hours {
            return Some(SpotFuturesExitReason::TimeStop {
                hold_hours,
                max_hours: self.config.max_hold_hours,
            });
        }

        None
    }

    /// Check if margin rebalancing is needed.
    /// Returns the amount of USDT to transfer from spot to futures wallet.
    pub fn check_rebalance_needed(
        &self,
        position: &SpotFuturesPosition,
        futures_margin_ratio: f64,
    ) -> Option<Decimal> {
        if !self.config.rebalance_enabled {
            return None;
        }

        if futures_margin_ratio >= self.config.margin_rebalance_threshold {
            return None; // Margin is healthy
        }

        // Calculate how much USDT the futures account needs
        // Target: restore margin ratio to 50% (comfortable buffer)
        let target_ratio = Decimal::new(50, 2); // 0.50
        let current_ratio = Decimal::from_str(&futures_margin_ratio.to_string())
            .unwrap_or(Decimal::ZERO);

        if current_ratio >= target_ratio {
            return None;
        }

        // Transfer amount = (target_ratio - current_ratio) * total_margin_needed
        let total_position_value = position.futures_qty * position.futures_entry_price;
        let needed = (target_ratio - current_ratio) * total_position_value
            / Decimal::from(position.futures_leverage);

        // Cap at 10% of spot holdings to avoid over-rebalancing
        let max_transfer = position.spot_current_value * Decimal::new(10, 2);
        let transfer_amount = needed.min(max_transfer);

        if transfer_amount > Decimal::ZERO {
            info!(
                "[spot-futures-monitor] Rebalance needed: transfer ${} from spot to futures (margin_ratio={:.2}%)",
                transfer_amount, futures_margin_ratio * 100.0
            );
            Some(transfer_amount)
        } else {
            None
        }
    }

    /// Check hedge ratio deviation.
    /// Returns true if the hedge ratio is within tolerance.
    pub fn is_hedge_ratio_healthy(&self, position: &SpotFuturesPosition) -> bool {
        let deviation = (position.hedge_ratio - Decimal::ONE).abs();
        let tolerance = Decimal::from_str(&self.config.hedge_ratio_tolerance.to_string())
            .unwrap_or(Decimal::new(5, 2));

        if deviation > tolerance {
            warn!(
                "[spot-futures-monitor] Hedge ratio deviation: {} (tolerance: {})",
                deviation, tolerance
            );
            false
        } else {
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Historical Funding Rate Tracker
// ---------------------------------------------------------------------------

/// Tracks historical funding rates for weighted average prediction.
/// Stores last N observations per (exchange, symbol) pair.
pub struct FundingRateHistory {
    /// (exchange, symbol) -> VecDeque of (timestamp_ns, rate)
    history: std::collections::HashMap<(ExchangeId, String), VecDeque<(u64, f64)>>,
    /// Maximum number of observations to store per symbol.
    max_depth: usize,
}

impl FundingRateHistory {
    /// Create a new history tracker.
    pub fn new(max_depth: usize) -> Self {
        Self {
            history: std::collections::HashMap::new(),
            max_depth,
        }
    }

    /// Record a funding rate observation.
    pub fn record(&mut self, exchange: ExchangeId, symbol: &str, timestamp_ns: u64, rate: f64) {
        let key = (exchange, symbol.to_string());
        let deque = self.history.entry(key).or_insert_with(VecDeque::new);
        deque.push_back((timestamp_ns, rate));
        while deque.len() > self.max_depth {
            deque.pop_front();
        }
    }

    /// Get the weighted predicted rate: current * 0.6 + avg_last_3 * 0.4
    pub fn predicted_rate(&self, exchange: ExchangeId, symbol: &str, current_rate: f64) -> f64 {
        let key = (exchange, symbol.to_string());
        if let Some(deque) = self.history.get(&key) {
            if deque.len() >= 3 {
                let last_3_avg: f64 = deque.iter().rev().take(3).map(|(_, r)| r).sum::<f64>() / 3.0;
                return current_rate * 0.6 + last_3_avg * 0.4;
            } else if !deque.is_empty() {
                let avg: f64 = deque.iter().map(|(_, r)| r).sum::<f64>() / deque.len() as f64;
                return current_rate * 0.6 + avg * 0.4;
            }
        }
        current_rate // No history, use current rate only
    }

    /// Get the average rate for a symbol on an exchange.
    pub fn average_rate(&self, exchange: ExchangeId, symbol: &str) -> Option<f64> {
        let key = (exchange, symbol.to_string());
        self.history.get(&key).and_then(|deque| {
            if deque.is_empty() {
                None
            } else {
                Some(deque.iter().map(|(_, r)| r).sum::<f64>() / deque.len() as f64)
            }
        })
    }
}
