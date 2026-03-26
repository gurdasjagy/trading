//! CATEGORY 8: PnL Attribution by Strategy/Signal/Timeframe.
//!
//! Tracks realized and unrealized PnL broken down by:
//!   - Strategy name (e.g., "microstructure_imbalance", "funding_arb", "liquidation_counter")
//!   - Signal source (e.g., "vpin", "cvd_divergence", "ichimoku_cloud")
//!   - Timeframe (1m, 5m, 15m, 1h signals)
//!   - Trading session (Asian, London, NY)
//!
//! This enables the bot to identify which strategies are profitable
//! and dynamically allocate more capital to winners (Kelly integration).

use std::collections::HashMap;
use tracing::{debug, info};

/// PnL record for a single closed trade with attribution.
#[derive(Debug, Clone)]
pub struct AttributedTrade {
    /// Strategy that generated the signal.
    pub strategy_name: String,
    /// Signal source tag.
    pub signal_tag: String,
    /// Symbol traded.
    pub symbol: String,
    /// Trading session when the trade was opened.
    pub session: String,
    /// Entry confidence score.
    pub entry_confidence: f64,
    /// Realized PnL in USDT.
    pub pnl_usdt: f64,
    /// Fees paid in USDT.
    pub fees_usdt: f64,
    /// Net PnL (pnl - fees).
    pub net_pnl_usdt: f64,
    /// Entry timestamp (milliseconds).
    pub entry_time_ms: i64,
    /// Exit timestamp (milliseconds).
    pub exit_time_ms: i64,
    /// Hold duration in seconds.
    pub duration_secs: f64,
    /// Whether this was a winning trade.
    pub is_win: bool,
}

/// Aggregated PnL statistics for an attribution category.
#[derive(Debug, Clone, Default)]
pub struct AttributionStats {
    /// Total trades in this category.
    pub trade_count: u64,
    /// Number of winning trades.
    pub win_count: u64,
    /// Total realized PnL.
    pub total_pnl: f64,
    /// Total fees paid.
    pub total_fees: f64,
    /// Net PnL (total_pnl - total_fees).
    pub net_pnl: f64,
    /// Average PnL per trade.
    pub avg_pnl: f64,
    /// Win rate (0.0 to 1.0).
    pub win_rate: f64,
    /// Profit factor (gross_profit / gross_loss).
    pub profit_factor: f64,
    /// Largest winning trade.
    pub largest_win: f64,
    /// Largest losing trade.
    pub largest_loss: f64,
    /// Average hold duration in seconds.
    pub avg_duration_secs: f64,
    /// Gross profit (sum of winning trades).
    gross_profit: f64,
    /// Gross loss (sum of losing trades, positive number).
    gross_loss: f64,
    /// Sum of durations for averaging.
    total_duration: f64,
}

impl AttributionStats {
    /// Record a new trade in this attribution bucket.
    fn record(&mut self, trade: &AttributedTrade) {
        self.trade_count += 1;
        self.total_pnl += trade.pnl_usdt;
        self.total_fees += trade.fees_usdt;
        self.net_pnl += trade.net_pnl_usdt;
        self.total_duration += trade.duration_secs;

        if trade.is_win {
            self.win_count += 1;
            self.gross_profit += trade.pnl_usdt;
            if trade.pnl_usdt > self.largest_win {
                self.largest_win = trade.pnl_usdt;
            }
        } else {
            self.gross_loss += trade.pnl_usdt.abs();
            if trade.pnl_usdt < self.largest_loss {
                self.largest_loss = trade.pnl_usdt;
            }
        }

        // Update derived metrics
        let n = self.trade_count as f64;
        self.avg_pnl = self.total_pnl / n;
        self.win_rate = self.win_count as f64 / n;
        self.profit_factor = if self.gross_loss > 0.0 {
            self.gross_profit / self.gross_loss
        } else if self.gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        self.avg_duration_secs = self.total_duration / n;
    }
}

/// PnL attribution engine.
pub struct PnlAttribution {
    /// Per-strategy statistics.
    strategy_stats: HashMap<String, AttributionStats>,
    /// Per-signal statistics.
    signal_stats: HashMap<String, AttributionStats>,
    /// Per-session statistics.
    session_stats: HashMap<String, AttributionStats>,
    /// Per-symbol statistics.
    symbol_stats: HashMap<String, AttributionStats>,
    /// Global statistics.
    global_stats: AttributionStats,
    /// All recorded trades (limited to recent N).
    trades: Vec<AttributedTrade>,
    /// Maximum trades to store.
    max_trades: usize,
}

impl PnlAttribution {
    /// Create a new PnL attribution engine.
    pub fn new(max_trades: usize) -> Self {
        Self {
            strategy_stats: HashMap::new(),
            signal_stats: HashMap::new(),
            session_stats: HashMap::new(),
            symbol_stats: HashMap::new(),
            global_stats: AttributionStats::default(),
            trades: Vec::new(),
            max_trades,
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(10_000)
    }

    /// Record a closed trade for attribution.
    pub fn record_trade(&mut self, trade: AttributedTrade) {
        // Update all attribution buckets
        self.global_stats.record(&trade);

        self.strategy_stats
            .entry(trade.strategy_name.clone())
            .or_insert_with(AttributionStats::default)
            .record(&trade);

        self.signal_stats
            .entry(trade.signal_tag.clone())
            .or_insert_with(AttributionStats::default)
            .record(&trade);

        self.session_stats
            .entry(trade.session.clone())
            .or_insert_with(AttributionStats::default)
            .record(&trade);

        self.symbol_stats
            .entry(trade.symbol.clone())
            .or_insert_with(AttributionStats::default)
            .record(&trade);

        info!(
            "[pnl-attr] {} {} {} | PnL=${:.2} | Session={} | WR={:.1}%",
            trade.strategy_name,
            trade.symbol,
            if trade.is_win { "WIN" } else { "LOSS" },
            trade.net_pnl_usdt,
            trade.session,
            self.strategy_stats.get(&trade.strategy_name)
                .map(|s| s.win_rate * 100.0)
                .unwrap_or(0.0),
        );

        // Store trade
        if self.trades.len() >= self.max_trades {
            self.trades.remove(0);
        }
        self.trades.push(trade);
    }

    /// Get global PnL statistics.
    pub fn global(&self) -> &AttributionStats {
        &self.global_stats
    }

    /// Get per-strategy statistics.
    pub fn strategy(&self, name: &str) -> Option<&AttributionStats> {
        self.strategy_stats.get(name)
    }

    /// Get all strategy names with statistics.
    pub fn all_strategies(&self) -> Vec<(&String, &AttributionStats)> {
        let mut strategies: Vec<_> = self.strategy_stats.iter().collect();
        strategies.sort_by(|a, b| b.1.net_pnl.partial_cmp(&a.1.net_pnl).unwrap_or(std::cmp::Ordering::Equal));
        strategies
    }

    /// Get per-session statistics.
    pub fn session(&self, name: &str) -> Option<&AttributionStats> {
        self.session_stats.get(name)
    }

    /// Get all sessions with statistics.
    pub fn all_sessions(&self) -> Vec<(&String, &AttributionStats)> {
        self.session_stats.iter().collect()
    }

    /// Get per-symbol statistics.
    pub fn symbol(&self, name: &str) -> Option<&AttributionStats> {
        self.symbol_stats.get(name)
    }

    /// Get the best performing strategy.
    pub fn best_strategy(&self) -> Option<(&String, &AttributionStats)> {
        self.strategy_stats.iter()
            .filter(|(_, s)| s.trade_count >= 10) // Minimum sample size
            .max_by(|a, b| a.1.net_pnl.partial_cmp(&b.1.net_pnl).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the worst performing strategy.
    pub fn worst_strategy(&self) -> Option<(&String, &AttributionStats)> {
        self.strategy_stats.iter()
            .filter(|(_, s)| s.trade_count >= 10)
            .min_by(|a, b| a.1.net_pnl.partial_cmp(&b.1.net_pnl).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Serialize to JSON for dashboard.
    pub fn to_json(&self) -> serde_json::Value {
        let strategies: serde_json::Map<String, serde_json::Value> = self.strategy_stats.iter()
            .map(|(name, stats)| {
                (name.clone(), serde_json::json!({
                    "trades": stats.trade_count,
                    "win_rate": format!("{:.1}%", stats.win_rate * 100.0),
                    "net_pnl": format!("${:.2}", stats.net_pnl),
                    "avg_pnl": format!("${:.2}", stats.avg_pnl),
                    "profit_factor": format!("{:.2}", stats.profit_factor),
                    "avg_duration": format!("{:.0}s", stats.avg_duration_secs),
                }))
            })
            .collect();

        serde_json::json!({
            "global": {
                "trades": self.global_stats.trade_count,
                "win_rate": format!("{:.1}%", self.global_stats.win_rate * 100.0),
                "net_pnl": format!("${:.2}", self.global_stats.net_pnl),
                "profit_factor": format!("{:.2}", self.global_stats.profit_factor),
            },
            "strategies": strategies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(strategy: &str, pnl: f64) -> AttributedTrade {
        AttributedTrade {
            strategy_name: strategy.to_string(),
            signal_tag: "test_signal".to_string(),
            symbol: "BTC_USDT".to_string(),
            session: "London".to_string(),
            entry_confidence: 0.7,
            pnl_usdt: pnl,
            fees_usdt: 1.0,
            net_pnl_usdt: pnl - 1.0,
            entry_time_ms: 1000,
            exit_time_ms: 2000,
            duration_secs: 60.0,
            is_win: pnl > 0.0,
        }
    }

    #[test]
    fn test_pnl_attribution_basic() {
        let mut attr = PnlAttribution::with_defaults();
        attr.record_trade(make_trade("momentum", 100.0));
        attr.record_trade(make_trade("momentum", -50.0));
        attr.record_trade(make_trade("mean_rev", 30.0));

        assert_eq!(attr.global().trade_count, 3);
        let mom = attr.strategy("momentum").unwrap();
        assert_eq!(mom.trade_count, 2);
        assert_eq!(mom.win_count, 1);
    }

    #[test]
    fn test_profit_factor() {
        let mut attr = PnlAttribution::with_defaults();
        // 2:1 profit factor
        attr.record_trade(make_trade("test", 200.0));
        attr.record_trade(make_trade("test", -100.0));

        let stats = attr.strategy("test").unwrap();
        assert!((stats.profit_factor - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_best_worst_strategy() {
        let mut attr = PnlAttribution::with_defaults();
        for _ in 0..15 {
            attr.record_trade(make_trade("good", 100.0));
            attr.record_trade(make_trade("bad", -50.0));
        }
        let best = attr.best_strategy().unwrap();
        assert_eq!(best.0, "good");
        let worst = attr.worst_strategy().unwrap();
        assert_eq!(worst.0, "bad");
    }
}
