//! CATEGORY 6 FIX: Kelly Criterion for Optimal Bet Sizing.
//!
//! Replaces the fixed confidence threshold in StrategyEngine::evaluate() with
//! a mathematically optimal position sizing formula. The Kelly Criterion
//! determines the fraction of capital to wager on each trade based on:
//!   - Historical win rate
//!   - Average win/loss ratio
//!   - Current edge estimation
//!
//! # Formula
//!
//! Full Kelly: f* = (p * b - q) / b
//! where:
//!   p = probability of winning (historical win rate)
//!   q = probability of losing (1 - p)
//!   b = average win / average loss ratio (payoff ratio)
//!
//! # Fractional Kelly
//!
//! Institutional desks typically use fractional Kelly (0.25-0.50 of full Kelly)
//! to reduce variance while maintaining positive expectancy.
//!
//! # Integration
//!
//! The Kelly fraction is used as a multiplier on the base position size
//! computed by the strategy engine, replacing the simple confidence threshold.

use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Rolling window size for Kelly calculation (number of trades).
const KELLY_WINDOW: usize = 100;

/// Maximum Kelly fraction (full Kelly can suggest absurd sizes).
const MAX_KELLY_FRACTION: f64 = 0.25;

/// Minimum number of trades before Kelly calculation is active.
const MIN_TRADES_FOR_KELLY: usize = 20;

/// A single trade outcome for Kelly calculation.
#[derive(Debug, Clone, Copy)]
pub struct TradeOutcome {
    /// PnL of the trade in USDT.
    pub pnl_usdt: f64,
    /// Whether the trade was a win (pnl > 0).
    pub is_win: bool,
    /// Signal confidence when the trade was opened.
    pub entry_confidence: f64,
    /// Strategy that generated the signal.
    pub strategy_id: u16,
}

/// Kelly Criterion calculator with per-strategy tracking.
pub struct KellyCriterion {
    /// Rolling window of recent trade outcomes.
    outcomes: VecDeque<TradeOutcome>,
    /// Per-strategy outcome tracking (strategy_id -> outcomes).
    strategy_outcomes: Vec<VecDeque<TradeOutcome>>,
    /// Fractional Kelly multiplier (0.25 = quarter Kelly, conservative).
    fractional_kelly: f64,
    /// Maximum allowed position size as fraction of equity.
    max_position_fraction: f64,
    /// Total trades recorded.
    total_trades: u64,
}

impl KellyCriterion {
    /// Create a new Kelly Criterion calculator.
    ///
    /// # Arguments
    /// * `fractional_kelly` - Fraction of full Kelly to use (0.25 recommended)
    /// * `max_position_fraction` - Max position size as fraction of equity (0.05 = 5%)
    pub fn new(fractional_kelly: f64, max_position_fraction: f64) -> Self {
        let mut strategy_outcomes = Vec::with_capacity(64);
        for _ in 0..64 {
            strategy_outcomes.push(VecDeque::with_capacity(KELLY_WINDOW));
        }

        Self {
            outcomes: VecDeque::with_capacity(KELLY_WINDOW),
            strategy_outcomes,
            fractional_kelly: fractional_kelly.clamp(0.1, 1.0),
            max_position_fraction: max_position_fraction.clamp(0.01, 0.25),
            total_trades: 0,
        }
    }

    /// Create with default institutional settings (quarter Kelly).
    pub fn with_defaults() -> Self {
        Self::new(0.25, 0.05)
    }

    /// Record a trade outcome.
    pub fn record_trade(&mut self, outcome: TradeOutcome) {
        // Global tracking
        if self.outcomes.len() >= KELLY_WINDOW {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back(outcome);

        // Per-strategy tracking
        let sid = outcome.strategy_id as usize;
        if sid < self.strategy_outcomes.len() {
            if self.strategy_outcomes[sid].len() >= KELLY_WINDOW {
                self.strategy_outcomes[sid].pop_front();
            }
            self.strategy_outcomes[sid].push_back(outcome);
        }

        self.total_trades += 1;
    }

    /// Compute the Kelly fraction for overall portfolio sizing.
    ///
    /// Returns the optimal fraction of equity to allocate per trade [0.0, max_position_fraction].
    pub fn compute_kelly_fraction(&self) -> f64 {
        self.compute_kelly_from_outcomes(&self.outcomes)
    }

    /// Compute Kelly fraction for a specific strategy.
    pub fn compute_strategy_kelly(&self, strategy_id: u16) -> f64 {
        let sid = strategy_id as usize;
        if sid >= self.strategy_outcomes.len() {
            return 0.0;
        }
        self.compute_kelly_from_outcomes(&self.strategy_outcomes[sid])
    }

    /// Internal Kelly calculation from a set of outcomes.
    fn compute_kelly_from_outcomes(&self, outcomes: &VecDeque<TradeOutcome>) -> f64 {
        if outcomes.len() < MIN_TRADES_FOR_KELLY {
            // Not enough data - return conservative default
            return 0.01; // 1% of equity
        }

        let total = outcomes.len() as f64;
        let wins: Vec<f64> = outcomes.iter()
            .filter(|o| o.is_win)
            .map(|o| o.pnl_usdt)
            .collect();
        let losses: Vec<f64> = outcomes.iter()
            .filter(|o| !o.is_win)
            .map(|o| o.pnl_usdt.abs())
            .collect();

        if wins.is_empty() || losses.is_empty() {
            return 0.01; // Edge case: all wins or all losses
        }

        let win_count = wins.len() as f64;
        let p = win_count / total; // Win probability
        let q = 1.0 - p;           // Loss probability

        let avg_win = wins.iter().sum::<f64>() / win_count;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;

        if avg_loss <= 0.0 {
            return 0.01;
        }

        let b = avg_win / avg_loss; // Payoff ratio

        // Kelly formula: f* = (p * b - q) / b
        let full_kelly = (p * b - q) / b;

        if full_kelly <= 0.0 {
            // Negative Kelly = negative edge, don't trade
            debug!(
                "[kelly] Negative edge: p={:.3}, b={:.3}, kelly={:.4}",
                p, b, full_kelly
            );
            return 0.0;
        }

        // Apply fractional Kelly and cap
        let fractional = full_kelly * self.fractional_kelly;
        let capped = fractional.min(self.max_position_fraction);

        debug!(
            "[kelly] p={:.3}, b={:.3}, full_kelly={:.4}, fractional={:.4}, capped={:.4} (n={})",
            p, b, full_kelly, fractional, capped, outcomes.len()
        );

        capped
    }

    /// Get the optimal position size in USDT given current equity.
    pub fn optimal_size_usdt(&self, equity: f64) -> f64 {
        let fraction = self.compute_kelly_fraction();
        (equity * fraction).max(0.0)
    }

    /// Get the optimal position size for a specific strategy.
    pub fn optimal_strategy_size_usdt(&self, equity: f64, strategy_id: u16) -> f64 {
        let fraction = self.compute_strategy_kelly(strategy_id);
        (equity * fraction).max(0.0)
    }

    /// Check if we have enough data for reliable Kelly estimates.
    pub fn is_warmed_up(&self) -> bool {
        self.outcomes.len() >= MIN_TRADES_FOR_KELLY
    }

    /// Get current win rate.
    pub fn win_rate(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        let wins = self.outcomes.iter().filter(|o| o.is_win).count() as f64;
        wins / self.outcomes.len() as f64
    }

    /// Get current payoff ratio (avg win / avg loss).
    pub fn payoff_ratio(&self) -> f64 {
        let wins: Vec<f64> = self.outcomes.iter()
            .filter(|o| o.is_win)
            .map(|o| o.pnl_usdt)
            .collect();
        let losses: Vec<f64> = self.outcomes.iter()
            .filter(|o| !o.is_win)
            .map(|o| o.pnl_usdt.abs())
            .collect();

        if wins.is_empty() || losses.is_empty() {
            return 1.0;
        }

        let avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;

        if avg_loss <= 0.0 { return 1.0; }
        avg_win / avg_loss
    }

    /// Get statistics for dashboard display.
    pub fn get_stats(&self) -> KellyStats {
        KellyStats {
            total_trades: self.total_trades,
            window_size: self.outcomes.len(),
            win_rate: self.win_rate(),
            payoff_ratio: self.payoff_ratio(),
            full_kelly: self.compute_kelly_from_outcomes(&self.outcomes) / self.fractional_kelly.max(0.01),
            fractional_kelly: self.compute_kelly_fraction(),
            is_warmed_up: self.is_warmed_up(),
        }
    }
}

/// Kelly Criterion statistics for dashboard/telemetry.
#[derive(Debug, Clone)]
pub struct KellyStats {
    pub total_trades: u64,
    pub window_size: usize,
    pub win_rate: f64,
    pub payoff_ratio: f64,
    pub full_kelly: f64,
    pub fractional_kelly: f64,
    pub is_warmed_up: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_positive_edge() {
        let mut kelly = KellyCriterion::new(0.5, 0.10);
        // 60% win rate, 2:1 payoff ratio
        for i in 0..100 {
            let is_win = i % 5 != 0 && i % 5 != 1; // 60% wins
            kelly.record_trade(TradeOutcome {
                pnl_usdt: if is_win { 200.0 } else { -100.0 },
                is_win,
                entry_confidence: 0.7,
                strategy_id: 0,
            });
        }
        let fraction = kelly.compute_kelly_fraction();
        assert!(fraction > 0.0, "Should have positive Kelly with 60% win rate and 2:1 payoff");
        assert!(fraction <= 0.10, "Should be capped at max_position_fraction");
    }

    #[test]
    fn test_kelly_negative_edge() {
        let mut kelly = KellyCriterion::new(0.5, 0.10);
        // 30% win rate, 1:1 payoff (negative expectancy)
        for i in 0..100 {
            let is_win = i % 10 < 3; // 30% wins
            kelly.record_trade(TradeOutcome {
                pnl_usdt: if is_win { 100.0 } else { -100.0 },
                is_win,
                entry_confidence: 0.5,
                strategy_id: 0,
            });
        }
        let fraction = kelly.compute_kelly_fraction();
        assert_eq!(fraction, 0.0, "Should return 0 for negative edge");
    }

    #[test]
    fn test_kelly_warmup() {
        let kelly = KellyCriterion::with_defaults();
        assert!(!kelly.is_warmed_up());
        assert_eq!(kelly.compute_kelly_fraction(), 0.01); // Default during warmup
    }

    #[test]
    fn test_per_strategy_kelly() {
        let mut kelly = KellyCriterion::with_defaults();
        // Strategy 0: good edge
        for _ in 0..30 {
            kelly.record_trade(TradeOutcome {
                pnl_usdt: 150.0,
                is_win: true,
                entry_confidence: 0.8,
                strategy_id: 0,
            });
        }
        // Strategy 1: bad edge
        for _ in 0..30 {
            kelly.record_trade(TradeOutcome {
                pnl_usdt: -100.0,
                is_win: false,
                entry_confidence: 0.5,
                strategy_id: 1,
            });
        }
        let s0 = kelly.compute_strategy_kelly(0);
        let s1 = kelly.compute_strategy_kelly(1);
        assert!(s0 > s1, "Strategy 0 (all wins) should have higher Kelly than strategy 1 (all losses)");
    }
}
