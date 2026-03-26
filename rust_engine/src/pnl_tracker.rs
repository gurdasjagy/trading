//! FEATURE 7: Real-Time PnL Tracking & Risk Dashboard.
//!
//! Tracks realized and unrealized PnL across all exchanges and symbols.
//! Computes Sharpe ratio, max drawdown, win rate, and other risk metrics
//! in real-time for dashboard consumption.
//!
//! # Architecture
//! - `PnLTracker` is the main struct, updated on every fill and price tick
//! - Realized PnL is recorded when positions are closed (full or partial)
//! - Unrealized PnL is recalculated on each price update
//! - Rolling metrics (Sharpe, drawdown) use a configurable lookback window
//! - JSON serialization for dashboard WebSocket streaming

use std::collections::HashMap;
use tracing::{info, warn};

/// A single closed trade record for PnL calculation.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Trading symbol.
    pub symbol: String,
    /// Exchange where the trade occurred.
    pub exchange: String,
    /// Entry price.
    pub entry_price: f64,
    /// Exit price.
    pub exit_price: f64,
    /// Position size (positive = long, negative = short).
    pub size: f64,
    /// Realized PnL in USDT.
    pub pnl_usdt: f64,
    /// Total fees paid (entry + exit).
    pub fees_usdt: f64,
    /// Net PnL (pnl_usdt - fees_usdt).
    pub net_pnl_usdt: f64,
    /// Entry timestamp (milliseconds).
    pub entry_time_ms: i64,
    /// Exit timestamp (milliseconds).
    pub exit_time_ms: i64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Whether this was a winning trade.
    pub is_win: bool,
}

/// An open position being tracked for unrealized PnL.
#[derive(Debug, Clone)]
pub struct TrackedPosition {
    /// Trading symbol.
    pub symbol: String,
    /// Exchange where the position is held.
    pub exchange: String,
    /// Entry price.
    pub entry_price: f64,
    /// Current mark price (updated on each tick).
    pub mark_price: f64,
    /// Position size (positive = long, negative = short).
    pub size: f64,
    /// Unrealized PnL at current mark price.
    pub unrealized_pnl: f64,
    /// Entry timestamp.
    pub entry_time_ms: i64,
    /// Leverage used.
    pub leverage: i32,
}

impl TrackedPosition {
    /// Recalculate unrealized PnL based on current mark price.
    pub fn update_mark_price(&mut self, mark_price: f64) {
        self.mark_price = mark_price;
        if self.size > 0.0 {
            // Long position
            self.unrealized_pnl = (mark_price - self.entry_price) * self.size;
        } else {
            // Short position
            self.unrealized_pnl = (self.entry_price - mark_price) * self.size.abs();
        }
    }
}

/// Real-time PnL tracker with rolling risk metrics.
pub struct PnLTracker {
    /// Closed trade records (most recent first).
    trades: Vec<TradeRecord>,
    /// Currently open positions keyed by "exchange:symbol".
    positions: HashMap<String, TrackedPosition>,
    /// Cumulative realized PnL.
    total_realized_pnl: f64,
    /// Cumulative fees paid.
    total_fees: f64,
    /// CATEGORY 4 FIX: Cumulative funding rate costs.
    /// Funding payments are deducted from PnL to reflect true cost of carry.
    total_funding_costs: f64,
    /// Peak equity (for drawdown calculation).
    peak_equity: f64,
    /// Current equity (balance + unrealized PnL).
    current_equity: f64,
    /// Starting balance for return calculations.
    starting_balance: f64,
    /// Maximum drawdown observed (percentage).
    max_drawdown_pct: f64,
    /// Daily PnL values for Sharpe calculation.
    daily_pnl_history: Vec<f64>,
    /// Current day's PnL accumulator.
    current_day_pnl: f64,
    /// Last day boundary (Unix day number).
    last_day: i64,
    /// Total number of trades.
    trade_count: u64,
    /// Number of winning trades.
    win_count: u64,
    /// Largest single win.
    largest_win: f64,
    /// Largest single loss.
    largest_loss: f64,
    /// Consecutive wins counter.
    consecutive_wins: u32,
    /// Consecutive losses counter.
    consecutive_losses: u32,
    /// Max consecutive wins.
    max_consecutive_wins: u32,
    /// Max consecutive losses.
    max_consecutive_losses: u32,
    /// Profit factor numerator (sum of wins).
    gross_profit: f64,
    /// Profit factor denominator (sum of losses).
    gross_loss: f64,
}

impl PnLTracker {
    /// Create a new PnL tracker with the given starting balance.
    pub fn new(starting_balance: f64) -> Self {
        Self {
            trades: Vec::new(),
            positions: HashMap::new(),
            total_realized_pnl: 0.0,
            total_fees: 0.0,
            total_funding_costs: 0.0,
            peak_equity: starting_balance,
            current_equity: starting_balance,
            starting_balance,
            max_drawdown_pct: 0.0,
            daily_pnl_history: Vec::new(),
            current_day_pnl: 0.0,
            last_day: 0,
            trade_count: 0,
            win_count: 0,
            largest_win: 0.0,
            largest_loss: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            gross_profit: 0.0,
            gross_loss: 0.0,
        }
    }

    /// Record a new position opening.
    pub fn open_position(
        &mut self,
        symbol: &str,
        exchange: &str,
        entry_price: f64,
        size: f64,
        leverage: i32,
    ) {
        let key = format!("{}:{}", exchange, symbol);
        let now_ms = now_ms();

        let position = TrackedPosition {
            symbol: symbol.to_string(),
            exchange: exchange.to_string(),
            entry_price,
            mark_price: entry_price,
            size,
            unrealized_pnl: 0.0,
            entry_time_ms: now_ms,
            leverage,
        };

        info!(
            "[pnl] Opened position: {} {} size={:.4} entry={:.2} lev={}x",
            exchange, symbol, size, entry_price, leverage
        );

        self.positions.insert(key, position);
    }

    /// Record a position close and calculate realized PnL.
    pub fn close_position(
        &mut self,
        symbol: &str,
        exchange: &str,
        exit_price: f64,
        fees: f64,
    ) -> Option<TradeRecord> {
        let key = format!("{}:{}", exchange, symbol);
        let position = self.positions.remove(&key)?;
        let now_ms = now_ms();

        let pnl_usdt = if position.size > 0.0 {
            (exit_price - position.entry_price) * position.size
        } else {
            (position.entry_price - exit_price) * position.size.abs()
        };

        let net_pnl = pnl_usdt - fees;
        let duration_secs = (now_ms - position.entry_time_ms) as f64 / 1000.0;
        let is_win = net_pnl > 0.0;

        let record = TradeRecord {
            symbol: symbol.to_string(),
            exchange: exchange.to_string(),
            entry_price: position.entry_price,
            exit_price,
            size: position.size,
            pnl_usdt,
            fees_usdt: fees,
            net_pnl_usdt: net_pnl,
            entry_time_ms: position.entry_time_ms,
            exit_time_ms: now_ms,
            duration_secs,
            is_win,
        };

        // Update statistics
        self.total_realized_pnl += net_pnl;
        self.total_fees += fees;
        self.trade_count += 1;

        if is_win {
            self.win_count += 1;
            self.gross_profit += net_pnl;
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
            self.max_consecutive_wins = self.max_consecutive_wins.max(self.consecutive_wins);
            if net_pnl > self.largest_win {
                self.largest_win = net_pnl;
            }
        } else {
            self.gross_loss += net_pnl.abs();
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
            self.max_consecutive_losses = self.max_consecutive_losses.max(self.consecutive_losses);
            if net_pnl < self.largest_loss {
                self.largest_loss = net_pnl;
            }
        }

        // Update daily PnL
        self.update_daily_pnl(net_pnl, now_ms);

        // Update equity and drawdown
        self.update_equity_and_drawdown();

        info!(
            "[pnl] Closed position: {} {} pnl=${:.2} fees=${:.2} net=${:.2} duration={:.0}s {}",
            exchange, symbol, pnl_usdt, fees, net_pnl, duration_secs,
            if is_win { "WIN" } else { "LOSS" }
        );

        self.trades.push(record.clone());
        Some(record)
    }

    /// Update mark prices for all open positions.
    pub fn update_prices(&mut self, prices: &HashMap<String, f64>) {
        for (_key, position) in &mut self.positions {
            if let Some(&price) = prices.get(&position.symbol) {
                position.update_mark_price(price);
            }
        }
        self.update_equity_and_drawdown();
    }

    /// Update a single symbol's mark price.
    pub fn update_price(&mut self, symbol: &str, exchange: &str, price: f64) {
        let key = format!("{}:{}", exchange, symbol);
        if let Some(position) = self.positions.get_mut(&key) {
            position.update_mark_price(price);
        }
        self.update_equity_and_drawdown();
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    /// Get win rate as a percentage.
    pub fn win_rate_pct(&self) -> f64 {
        if self.trade_count == 0 {
            return 0.0;
        }
        (self.win_count as f64 / self.trade_count as f64) * 100.0
    }

    /// Get profit factor (gross profit / gross loss).
    pub fn profit_factor(&self) -> f64 {
        if self.gross_loss == 0.0 {
            return if self.gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
        }
        self.gross_profit / self.gross_loss
    }

    /// Calculate annualized Sharpe ratio from daily PnL history.
    ///
    /// Sharpe = (mean_daily_pnl / std_daily_pnl) * sqrt(365)
    pub fn sharpe_ratio(&self) -> f64 {
        if self.daily_pnl_history.len() < 2 {
            return 0.0;
        }

        let n = self.daily_pnl_history.len() as f64;
        let mean = self.daily_pnl_history.iter().sum::<f64>() / n;
        let variance = self
            .daily_pnl_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        (mean / std_dev) * (365.0_f64).sqrt()
    }

    /// Get maximum drawdown percentage.
    pub fn max_drawdown_pct(&self) -> f64 {
        self.max_drawdown_pct
    }

    /// Get current drawdown percentage.
    pub fn current_drawdown_pct(&self) -> f64 {
        if self.peak_equity <= 0.0 {
            return 0.0;
        }
        ((self.peak_equity - self.current_equity) / self.peak_equity) * 100.0
    }

    /// Get total unrealized PnL across all open positions.
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }

    /// CATEGORY 4 FIX: Record a funding rate payment.
    ///
    /// Funding rates are periodic payments between long and short holders.
    /// Positive `amount` = we paid funding (cost), negative = we received.
    /// This is deducted from PnL to reflect true cost of carry.
    ///
    /// Called by the funding rate monitor when payments are detected.
    pub fn record_funding_payment(
        &mut self,
        symbol: &str,
        exchange: &str,
        amount: f64,
    ) {
        self.total_funding_costs += amount;
        // Funding costs affect realized PnL (they are settled immediately)
        self.total_realized_pnl -= amount;
        self.update_equity_and_drawdown();

        if amount.abs() > 0.01 {
            info!(
                "[pnl] Funding payment: {} {} ${:.4} (total funding costs: ${:.2})",
                exchange, symbol, amount, self.total_funding_costs
            );
        }
    }

    /// CATEGORY 4 FIX: Get total funding costs paid.
    pub fn total_funding_costs(&self) -> f64 {
        self.total_funding_costs
    }

    /// Get total PnL (realized + unrealized - funding costs).
    /// CATEGORY 4 FIX: Now includes funding rate costs in PnL calculation.
    pub fn total_pnl(&self) -> f64 {
        self.total_realized_pnl + self.total_unrealized_pnl()
    }

    /// Get return on starting balance (percentage).
    pub fn total_return_pct(&self) -> f64 {
        if self.starting_balance <= 0.0 {
            return 0.0;
        }
        (self.total_pnl() / self.starting_balance) * 100.0
    }

    /// Get average trade PnL.
    pub fn avg_trade_pnl(&self) -> f64 {
        if self.trade_count == 0 {
            return 0.0;
        }
        self.total_realized_pnl / self.trade_count as f64
    }

    /// Get the average win amount.
    pub fn avg_win(&self) -> f64 {
        if self.win_count == 0 {
            return 0.0;
        }
        self.gross_profit / self.win_count as f64
    }

    /// Get the average loss amount.
    pub fn avg_loss(&self) -> f64 {
        let loss_count = self.trade_count - self.win_count;
        if loss_count == 0 {
            return 0.0;
        }
        self.gross_loss / loss_count as f64
    }

    /// Get the expectancy (average expected PnL per trade).
    pub fn expectancy(&self) -> f64 {
        if self.trade_count == 0 {
            return 0.0;
        }
        let win_rate = self.win_count as f64 / self.trade_count as f64;
        let loss_rate = 1.0 - win_rate;
        (win_rate * self.avg_win()) - (loss_rate * self.avg_loss())
    }

    /// Get open positions count.
    pub fn open_position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get all open positions.
    pub fn open_positions(&self) -> Vec<&TrackedPosition> {
        self.positions.values().collect()
    }

    /// Get recent trade records (most recent N).
    pub fn recent_trades(&self, n: usize) -> &[TradeRecord] {
        let start = if self.trades.len() > n {
            self.trades.len() - n
        } else {
            0
        };
        &self.trades[start..]
    }

    // -----------------------------------------------------------------------
    // Dashboard JSON
    // -----------------------------------------------------------------------

    /// Serialize full PnL state as JSON for dashboard consumption.
    pub fn to_json(&self) -> serde_json::Value {
        let positions_json: Vec<serde_json::Value> = self
            .positions
            .values()
            .map(|p| {
                serde_json::json!({
                    "symbol": p.symbol,
                    "exchange": p.exchange,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "size": p.size,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": p.leverage,
                    "side": if p.size > 0.0 { "long" } else { "short" },
                })
            })
            .collect();

        let recent_trades_json: Vec<serde_json::Value> = self
            .recent_trades(10)
            .iter()
            .map(|t| {
                serde_json::json!({
                    "symbol": t.symbol,
                    "exchange": t.exchange,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "net_pnl": t.net_pnl_usdt,
                    "fees": t.fees_usdt,
                    "duration_secs": t.duration_secs,
                    "is_win": t.is_win,
                })
            })
            .collect();

        serde_json::json!({
            "realized_pnl": self.total_realized_pnl,
            "unrealized_pnl": self.total_unrealized_pnl(),
            "total_pnl": self.total_pnl(),
            "total_fees": self.total_fees,
            "total_return_pct": self.total_return_pct(),
            "equity": self.current_equity,
            "starting_balance": self.starting_balance,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "win_rate_pct": self.win_rate_pct(),
            "profit_factor": self.profit_factor(),
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_drawdown_pct": self.current_drawdown_pct(),
            "avg_trade_pnl": self.avg_trade_pnl(),
            "avg_win": self.avg_win(),
            "avg_loss": self.avg_loss(),
            "expectancy": self.expectancy(),
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "open_positions": positions_json,
            "recent_trades": recent_trades_json,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Update equity and drawdown calculations.
    fn update_equity_and_drawdown(&mut self) {
        let unrealized = self.total_unrealized_pnl();
        self.current_equity = self.starting_balance + self.total_realized_pnl + unrealized;

        if self.current_equity > self.peak_equity {
            self.peak_equity = self.current_equity;
        }

        let drawdown_pct = self.current_drawdown_pct();
        if drawdown_pct > self.max_drawdown_pct {
            self.max_drawdown_pct = drawdown_pct;
        }
    }

    /// Update daily PnL tracking for Sharpe calculation.
    fn update_daily_pnl(&mut self, pnl: f64, timestamp_ms: i64) {
        let day_number = timestamp_ms / (86_400 * 1000);

        if self.last_day == 0 {
            self.last_day = day_number;
        }

        if day_number > self.last_day {
            // New day — save previous day's PnL and start fresh
            self.daily_pnl_history.push(self.current_day_pnl);

            // Keep only last 365 days
            if self.daily_pnl_history.len() > 365 {
                self.daily_pnl_history.remove(0);
            }

            self.current_day_pnl = pnl;
            self.last_day = day_number;
        } else {
            self.current_day_pnl += pnl;
        }
    }
}

impl Default for PnLTracker {
    fn default() -> Self {
        Self::new(1000.0) // Default $1000 starting balance
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl_tracker_basic() {
        let mut tracker = PnLTracker::new(10000.0);

        // Open a long position
        tracker.open_position("BTC_USDT", "binance", 60000.0, 0.1, 10);
        assert_eq!(tracker.open_position_count(), 1);

        // Price goes up
        tracker.update_price("BTC_USDT", "binance", 61000.0);
        assert!((tracker.total_unrealized_pnl() - 100.0).abs() < 0.01);

        // Close the position
        let record = tracker.close_position("BTC_USDT", "binance", 61000.0, 2.0);
        assert!(record.is_some());
        let record = record.unwrap();
        assert!(record.is_win);
        assert!((record.net_pnl_usdt - 98.0).abs() < 0.01);

        assert_eq!(tracker.trade_count, 1);
        assert_eq!(tracker.win_count, 1);
        assert!((tracker.win_rate_pct() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_pnl_tracker_short() {
        let mut tracker = PnLTracker::new(10000.0);

        // Open a short position
        tracker.open_position("ETH_USDT", "bybit", 3000.0, -1.0, 5);

        // Price goes down (profitable for short)
        tracker.update_price("ETH_USDT", "bybit", 2900.0);
        assert!((tracker.total_unrealized_pnl() - 100.0).abs() < 0.01);

        // Close
        let record = tracker.close_position("ETH_USDT", "bybit", 2900.0, 1.5);
        assert!(record.is_some());
        let record = record.unwrap();
        assert!(record.is_win);
        assert!((record.net_pnl_usdt - 98.5).abs() < 0.01);
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut tracker = PnLTracker::new(10000.0);

        // Win first
        tracker.open_position("BTC_USDT", "gate", 60000.0, 0.1, 10);
        tracker.close_position("BTC_USDT", "gate", 62000.0, 1.0);
        // PnL = 200 - 1 = 199, equity = 10199

        // Then lose
        tracker.open_position("BTC_USDT", "gate", 62000.0, 0.1, 10);
        tracker.close_position("BTC_USDT", "gate", 59000.0, 1.0);
        // PnL = -300 - 1 = -301, equity = 10199 - 301 = 9898

        assert!(tracker.max_drawdown_pct() > 0.0);
        assert!(tracker.current_drawdown_pct() > 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let mut tracker = PnLTracker::new(10000.0);
        // Manually set daily PnL history
        tracker.daily_pnl_history = vec![10.0, -5.0, 15.0, -3.0, 20.0, 8.0, -2.0];

        let sharpe = tracker.sharpe_ratio();
        // Should be positive since net gains > losses
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_profit_factor() {
        let mut tracker = PnLTracker::new(10000.0);

        // Two wins totaling $200
        tracker.open_position("BTC_USDT", "gate", 60000.0, 0.1, 10);
        tracker.close_position("BTC_USDT", "gate", 61000.0, 0.0);
        tracker.open_position("BTC_USDT", "gate", 60000.0, 0.1, 10);
        tracker.close_position("BTC_USDT", "gate", 61000.0, 0.0);

        // One loss of $50
        tracker.open_position("BTC_USDT", "gate", 60000.0, 0.1, 10);
        tracker.close_position("BTC_USDT", "gate", 59500.0, 0.0);

        assert!(tracker.profit_factor() > 1.0);
        assert_eq!(tracker.trade_count, 3);
        assert_eq!(tracker.win_count, 2);
    }

    #[test]
    fn test_to_json() {
        let tracker = PnLTracker::new(10000.0);
        let json = tracker.to_json();
        assert!(json.get("realized_pnl").is_some());
        assert!(json.get("sharpe_ratio").is_some());
        assert!(json.get("open_positions").is_some());
    }
}
