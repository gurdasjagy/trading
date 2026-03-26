//! CATEGORY 8: Trade Cost Analysis (TCA).
//!
//! Compares execution price vs arrival price vs VWAP to measure execution quality.
//! Institutional desks use TCA to:
//!   - Quantify slippage on every trade
//!   - Detect systematic execution problems
//!   - Optimize order routing decisions
//!   - Attribute costs to specific strategies/venues
//!
//! # Metrics
//!
//! - **Implementation Shortfall**: (execution_price - arrival_price) / arrival_price
//! - **VWAP Slippage**: (execution_price - vwap) / vwap
//! - **Market Impact**: Price movement caused by our order
//! - **Timing Cost**: Cost of delay between signal and execution
//! - **Spread Cost**: Half-spread paid on each trade

use std::collections::{HashMap, VecDeque};
use tracing::{debug, info};

/// Maximum TCA records to keep per symbol.
const MAX_TCA_RECORDS: usize = 1000;

/// A single TCA record for one executed trade.
#[derive(Debug, Clone)]
pub struct TcaRecord {
    /// Trading symbol.
    pub symbol: String,
    /// Exchange where executed.
    pub exchange: String,
    /// Strategy that generated the signal.
    pub strategy_name: String,
    /// Side: true = buy, false = sell.
    pub is_buy: bool,
    /// Order size (contracts).
    pub size: i64,
    /// Price when the signal was generated (arrival price).
    pub arrival_price: f64,
    /// Price when the order was submitted (decision price).
    pub decision_price: f64,
    /// Actual average execution price.
    pub execution_price: f64,
    /// VWAP during the execution window.
    pub vwap: f64,
    /// Mid price at time of execution.
    pub mid_price_at_exec: f64,
    /// Best bid at time of execution.
    pub best_bid: f64,
    /// Best ask at time of execution.
    pub best_ask: f64,
    /// Signal timestamp (microseconds).
    pub signal_time_us: i64,
    /// Order submission timestamp (microseconds).
    pub submit_time_us: i64,
    /// Fill confirmation timestamp (microseconds).
    pub fill_time_us: i64,
    /// Fees paid (USDT).
    pub fees_usdt: f64,
}

/// Computed TCA metrics for a single trade.
#[derive(Debug, Clone)]
pub struct TcaMetrics {
    /// Implementation shortfall: (exec - arrival) / arrival
    /// Positive = we paid more than arrival (bad for buys)
    pub implementation_shortfall_bps: f64,
    /// VWAP slippage: (exec - vwap) / vwap
    pub vwap_slippage_bps: f64,
    /// Spread cost: half-spread at time of execution
    pub spread_cost_bps: f64,
    /// Market impact estimate
    pub market_impact_bps: f64,
    /// Timing cost: price movement from signal to submit
    pub timing_cost_bps: f64,
    /// Total execution cost (implementation shortfall + fees)
    pub total_cost_bps: f64,
    /// Signal-to-fill latency in microseconds
    pub signal_to_fill_us: i64,
    /// Submit-to-fill latency in microseconds
    pub submit_to_fill_us: i64,
}

impl TcaRecord {
    /// Compute TCA metrics from a trade record.
    pub fn compute_metrics(&self) -> TcaMetrics {
        let direction = if self.is_buy { 1.0 } else { -1.0 };

        // Implementation shortfall (IS)
        // For buys: positive IS means we paid MORE than arrival (bad)
        // For sells: positive IS means we received LESS than arrival (bad)
        let is_raw = direction * (self.execution_price - self.arrival_price) / self.arrival_price;
        let implementation_shortfall_bps = is_raw * 10_000.0;

        // VWAP slippage
        let vwap_slip = if self.vwap > 0.0 {
            direction * (self.execution_price - self.vwap) / self.vwap * 10_000.0
        } else {
            0.0
        };

        // Spread cost
        let spread = self.best_ask - self.best_bid;
        let spread_cost_bps = if self.mid_price_at_exec > 0.0 {
            (spread / 2.0) / self.mid_price_at_exec * 10_000.0
        } else {
            0.0
        };

        // Market impact: price moved from our order
        // Approximation: diff between mid at exec and decision price
        let market_impact_bps = if self.decision_price > 0.0 {
            direction * (self.mid_price_at_exec - self.decision_price)
                / self.decision_price * 10_000.0
        } else {
            0.0
        };

        // Timing cost: price moved while we waited
        let timing_cost_bps = if self.arrival_price > 0.0 {
            direction * (self.decision_price - self.arrival_price)
                / self.arrival_price * 10_000.0
        } else {
            0.0
        };

        // Total cost including fees
        let fee_bps = if self.execution_price > 0.0 && self.size != 0 {
            self.fees_usdt / (self.execution_price * self.size.abs() as f64) * 10_000.0
        } else {
            0.0
        };
        let total_cost_bps = implementation_shortfall_bps + fee_bps;

        TcaMetrics {
            implementation_shortfall_bps,
            vwap_slippage_bps: vwap_slip,
            spread_cost_bps,
            market_impact_bps,
            timing_cost_bps,
            total_cost_bps,
            signal_to_fill_us: self.fill_time_us - self.signal_time_us,
            submit_to_fill_us: self.fill_time_us - self.submit_time_us,
        }
    }
}

/// Aggregate TCA statistics.
#[derive(Debug, Clone, Default)]
pub struct TcaAggregate {
    /// Number of trades analyzed.
    pub trade_count: u64,
    /// Average implementation shortfall (bps).
    pub avg_is_bps: f64,
    /// Average VWAP slippage (bps).
    pub avg_vwap_slip_bps: f64,
    /// Average spread cost (bps).
    pub avg_spread_cost_bps: f64,
    /// Average market impact (bps).
    pub avg_market_impact_bps: f64,
    /// Average total cost (bps).
    pub avg_total_cost_bps: f64,
    /// Average signal-to-fill latency (microseconds).
    pub avg_signal_to_fill_us: f64,
    /// Total fees paid (USDT).
    pub total_fees_usdt: f64,
    /// Best execution (lowest IS in bps).
    pub best_execution_bps: f64,
    /// Worst execution (highest IS in bps).
    pub worst_execution_bps: f64,
}

/// Trade Cost Analysis engine.
pub struct TcaEngine {
    /// All TCA records.
    records: VecDeque<TcaRecord>,
    /// Per-strategy aggregate stats.
    strategy_stats: HashMap<String, TcaAggregate>,
    /// Per-symbol aggregate stats.
    symbol_stats: HashMap<String, TcaAggregate>,
    /// Global aggregate stats.
    global_stats: TcaAggregate,
}

impl TcaEngine {
    /// Create a new TCA engine.
    pub fn new() -> Self {
        Self {
            records: VecDeque::with_capacity(MAX_TCA_RECORDS),
            strategy_stats: HashMap::new(),
            symbol_stats: HashMap::new(),
            global_stats: TcaAggregate::default(),
        }
    }

    /// Record a completed trade for TCA analysis.
    pub fn record_trade(&mut self, record: TcaRecord) {
        let metrics = record.compute_metrics();

        // Update global aggregate
        self.update_aggregate(&mut self.global_stats.clone(), &metrics, record.fees_usdt);

        // Update per-strategy aggregate
        let strategy_agg = self.strategy_stats
            .entry(record.strategy_name.clone())
            .or_insert_with(TcaAggregate::default);
        self.update_aggregate(strategy_agg, &metrics, record.fees_usdt);

        // Update per-symbol aggregate
        let symbol_agg = self.symbol_stats
            .entry(record.symbol.clone())
            .or_insert_with(TcaAggregate::default);
        self.update_aggregate(symbol_agg, &metrics, record.fees_usdt);

        info!(
            "[tca] {} {} {}: IS={:.2}bps, VWAP_slip={:.2}bps, total={:.2}bps, latency={}us",
            record.strategy_name,
            if record.is_buy { "BUY" } else { "SELL" },
            record.symbol,
            metrics.implementation_shortfall_bps,
            metrics.vwap_slippage_bps,
            metrics.total_cost_bps,
            metrics.signal_to_fill_us,
        );

        // Store record
        if self.records.len() >= MAX_TCA_RECORDS {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    fn update_aggregate(&self, agg: &mut TcaAggregate, metrics: &TcaMetrics, fees: f64) {
        let n = agg.trade_count as f64;
        let new_n = n + 1.0;

        // Running averages
        agg.avg_is_bps = (agg.avg_is_bps * n + metrics.implementation_shortfall_bps) / new_n;
        agg.avg_vwap_slip_bps = (agg.avg_vwap_slip_bps * n + metrics.vwap_slippage_bps) / new_n;
        agg.avg_spread_cost_bps = (agg.avg_spread_cost_bps * n + metrics.spread_cost_bps) / new_n;
        agg.avg_market_impact_bps = (agg.avg_market_impact_bps * n + metrics.market_impact_bps) / new_n;
        agg.avg_total_cost_bps = (agg.avg_total_cost_bps * n + metrics.total_cost_bps) / new_n;
        agg.avg_signal_to_fill_us = (agg.avg_signal_to_fill_us * n + metrics.signal_to_fill_us as f64) / new_n;
        agg.total_fees_usdt += fees;
        agg.trade_count += 1;

        // Track best/worst
        if agg.trade_count == 1 {
            agg.best_execution_bps = metrics.implementation_shortfall_bps;
            agg.worst_execution_bps = metrics.implementation_shortfall_bps;
        } else {
            if metrics.implementation_shortfall_bps < agg.best_execution_bps {
                agg.best_execution_bps = metrics.implementation_shortfall_bps;
            }
            if metrics.implementation_shortfall_bps > agg.worst_execution_bps {
                agg.worst_execution_bps = metrics.implementation_shortfall_bps;
            }
        }
    }

    /// Get global TCA statistics.
    pub fn global_stats(&self) -> &TcaAggregate {
        &self.global_stats
    }

    /// Get per-strategy TCA statistics.
    pub fn strategy_stats(&self, strategy: &str) -> Option<&TcaAggregate> {
        self.strategy_stats.get(strategy)
    }

    /// Get per-symbol TCA statistics.
    pub fn symbol_stats(&self, symbol: &str) -> Option<&TcaAggregate> {
        self.symbol_stats.get(symbol)
    }

    /// Get all strategy names with TCA data.
    pub fn strategies(&self) -> Vec<String> {
        self.strategy_stats.keys().cloned().collect()
    }

    /// Get the most recent N TCA records.
    pub fn recent_records(&self, n: usize) -> Vec<&TcaRecord> {
        self.records.iter().rev().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(is_buy: bool, arrival: f64, exec: f64) -> TcaRecord {
        TcaRecord {
            symbol: "BTC_USDT".to_string(),
            exchange: "gateio".to_string(),
            strategy_name: "microstructure".to_string(),
            is_buy,
            size: 10,
            arrival_price: arrival,
            decision_price: arrival,
            execution_price: exec,
            vwap: (arrival + exec) / 2.0,
            mid_price_at_exec: exec,
            best_bid: exec - 0.5,
            best_ask: exec + 0.5,
            signal_time_us: 1000,
            submit_time_us: 1100,
            fill_time_us: 1500,
            fees_usdt: 0.5,
        }
    }

    #[test]
    fn test_buy_slippage() {
        let record = make_record(true, 50000.0, 50010.0); // Paid 10 more than arrival
        let metrics = record.compute_metrics();
        assert!(metrics.implementation_shortfall_bps > 0.0, "Buying higher should be positive IS");
    }

    #[test]
    fn test_sell_slippage() {
        let record = make_record(false, 50000.0, 49990.0); // Sold 10 less than arrival
        let metrics = record.compute_metrics();
        assert!(metrics.implementation_shortfall_bps > 0.0, "Selling lower should be positive IS");
    }

    #[test]
    fn test_tca_engine_aggregation() {
        let mut engine = TcaEngine::new();
        engine.record_trade(make_record(true, 50000.0, 50010.0));
        engine.record_trade(make_record(true, 50000.0, 50005.0));
        assert_eq!(engine.global_stats().trade_count, 2);
        assert!(engine.global_stats().avg_is_bps > 0.0);
    }
}
