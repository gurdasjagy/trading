use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
struct KahanSum {
    sum: f64,
    compensation: f64,
}

impl KahanSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    fn get(&self) -> f64 {
        self.sum
    }
}

pub struct MarketMakerInventoryModel {
    net_position: KahanSum,
    position_history: VecDeque<f64>,
    position_ema: f64,
    ema_alpha: f64,
    volume_history: VecDeque<f64>,
    window_size: usize,
}

impl MarketMakerInventoryModel {
    pub fn new(window_size: usize) -> Self {
        Self {
            net_position: KahanSum::new(),
            position_history: VecDeque::with_capacity(window_size),
            position_ema: 0.0,
            ema_alpha: 0.1, // 10% weight to new values
            volume_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn on_trade(&mut self, size: f64, is_buy: bool) {
        // Market maker takes opposite side
        // If trade is buy (aggressor buys), MM sells (negative position)
        // If trade is sell (aggressor sells), MM buys (positive position)
        let position_change = if is_buy { -size } else { size };
        
        self.net_position.add(position_change);
        
        // Update position history
        if self.position_history.len() >= self.window_size {
            self.position_history.pop_front();
        }
        self.position_history.push_back(self.net_position.get());
        
        // Update position EMA
        self.position_ema = self.ema_alpha * self.net_position.get() + (1.0 - self.ema_alpha) * self.position_ema;
        
        // Update volume history
        if self.volume_history.len() >= self.window_size {
            self.volume_history.pop_front();
        }
        self.volume_history.push_back(size);
    }

    pub fn get_inventory_pressure(&self) -> f64 {
        if self.volume_history.is_empty() {
            return 0.0;
        }

        let typical_volume: f64 = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
        
        if typical_volume == 0.0 {
            return 0.0;
        }

        // Normalize position by typical volume
        let pressure = self.position_ema / typical_volume;
        
        // Clamp to [-1.0, 1.0]
        pressure.max(-1.0).min(1.0)
    }

    pub fn get_inventory_signal(&self) -> (i8, f64) {
        let pressure = self.get_inventory_pressure();
        
        // High positive pressure = MM is long, wants to sell (bearish)
        // High negative pressure = MM is short, wants to buy (bullish)
        
        let direction = if pressure > 0.3 {
            -1 // Bearish
        } else if pressure < -0.3 {
            1 // Bullish
        } else {
            0 // Neutral
        };
        
        let pressure_score = pressure.abs();
        
        (direction, pressure_score)
    }

    pub fn reset_daily(&mut self) {
        self.net_position = KahanSum::new();
        self.position_ema = 0.0;
        self.position_history.clear();
        // Keep volume history for continuity
    }

    pub fn get_net_position(&self) -> f64 {
        self.net_position.get()
    }

    pub fn get_position_ema(&self) -> f64 {
        self.position_ema
    }

    /// INST: Calculate quote skew based on inventory pressure.
    ///
    /// Returns (bid_skew_bps, ask_skew_bps) adjustments to apply to quotes.
    /// When inventory is long, widen ask (make selling easier) and tighten bid.
    /// When inventory is short, widen bid (make buying easier) and tighten ask.
    ///
    /// This is the Avellaneda-Stoikov optimal market making approach:
    /// the market maker adjusts quotes to mean-revert inventory toward zero.
    pub fn get_quote_skew_bps(&self, base_spread_bps: f64) -> (f64, f64) {
        let pressure = self.get_inventory_pressure();
        // Skew is proportional to inventory pressure
        // Max skew is 50% of the base spread on each side
        let max_skew = base_spread_bps * 0.5;
        let skew = pressure * max_skew;

        // Positive pressure (long) → tighten bid, widen ask
        // Negative pressure (short) → widen bid, tighten ask
        let bid_skew = -skew; // Negative = tighter bid when long
        let ask_skew = skew;  // Positive = wider ask when long

        (bid_skew, ask_skew)
    }

    /// INST: Check if inventory exceeds rebalancing threshold.
    ///
    /// Returns true if the absolute inventory pressure exceeds the given
    /// threshold, indicating the position should be actively unwound.
    pub fn needs_rebalance(&self, threshold: f64) -> bool {
        self.get_inventory_pressure().abs() > threshold.clamp(0.1, 0.9)
    }

    /// INST: Get recommended rebalance direction and urgency.
    ///
    /// Returns (should_buy: bool, urgency: f64) where urgency [0,1]
    /// indicates how aggressively the position should be unwound.
    pub fn rebalance_recommendation(&self) -> (bool, f64) {
        let pressure = self.get_inventory_pressure();
        let should_buy = pressure < 0.0; // Short inventory → need to buy
        let urgency = (pressure.abs() - 0.3).max(0.0) / 0.7; // 0 at threshold, 1 at max
        (should_buy, urgency.clamp(0.0, 1.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Drawdown Kill Switch (INST 3)
// ═══════════════════════════════════════════════════════════════════════════

/// Monitors cumulative PnL and equity curve to halt trading when drawdown
/// exceeds configurable thresholds. This is a critical risk control used
/// by all institutional trading desks.
///
/// The kill switch has two modes:
/// - **Soft halt**: No new entries, existing positions managed normally
/// - **Hard halt**: No new entries, existing positions closed immediately
pub struct DrawdownKillSwitch {
    /// Peak equity value (high water mark).
    peak_equity: f64,
    /// Current equity value.
    current_equity: f64,
    /// Maximum allowed drawdown as a fraction (e.g., 0.05 = 5%).
    max_drawdown_pct: f64,
    /// Critical drawdown threshold for hard halt (e.g., 0.10 = 10%).
    critical_drawdown_pct: f64,
    /// Whether trading is currently halted.
    is_halted: bool,
    /// Whether hard halt is active (close all positions).
    is_hard_halted: bool,
    /// Timestamp when halt was triggered (0 if not halted).
    halt_triggered_at_ms: i64,
    /// Cooldown period before allowing trading to resume (ms).
    cooldown_ms: i64,
    /// Daily PnL tracking.
    daily_pnl: f64,
    /// Maximum daily loss before halt.
    max_daily_loss: f64,
    /// Number of consecutive losing trades.
    consecutive_losses: u32,
    /// Maximum consecutive losses before halt.
    max_consecutive_losses: u32,
}

impl DrawdownKillSwitch {
    /// Create a new kill switch with configurable thresholds.
    pub fn new(
        initial_equity: f64,
        max_drawdown_pct: f64,
        critical_drawdown_pct: f64,
        max_daily_loss: f64,
        max_consecutive_losses: u32,
    ) -> Self {
        Self {
            peak_equity: initial_equity,
            current_equity: initial_equity,
            max_drawdown_pct: max_drawdown_pct.clamp(0.01, 0.50),
            critical_drawdown_pct: critical_drawdown_pct.clamp(0.02, 0.50),
            is_halted: false,
            is_hard_halted: false,
            halt_triggered_at_ms: 0,
            cooldown_ms: 3600_000, // 1 hour default cooldown
            daily_pnl: 0.0,
            max_daily_loss,
            consecutive_losses: 0,
            max_consecutive_losses,
        }
    }

    /// Create with sensible production defaults.
    pub fn with_defaults(initial_equity: f64) -> Self {
        Self::new(
            initial_equity,
            0.05,  // 5% max drawdown → soft halt
            0.10,  // 10% max drawdown → hard halt
            initial_equity * 0.02, // 2% max daily loss
            5,     // 5 consecutive losses
        )
    }

    /// Update equity and check kill switch conditions.
    /// Returns true if trading should continue, false if halted.
    pub fn update_equity(&mut self, new_equity: f64, now_ms: i64) -> bool {
        self.current_equity = new_equity;

        // Update high water mark
        if new_equity > self.peak_equity {
            self.peak_equity = new_equity;
        }

        // Calculate current drawdown
        let drawdown = if self.peak_equity > 0.0 {
            (self.peak_equity - self.current_equity) / self.peak_equity
        } else {
            0.0
        };

        // Check critical drawdown (hard halt)
        if drawdown >= self.critical_drawdown_pct {
            if !self.is_hard_halted {
                self.is_hard_halted = true;
                self.is_halted = true;
                self.halt_triggered_at_ms = now_ms;
                tracing::error!(
                    "[kill-switch] HARD HALT: drawdown {:.2}% >= critical {:.2}%. Closing all positions.",
                    drawdown * 100.0, self.critical_drawdown_pct * 100.0
                );
            }
            return false;
        }

        // Check soft drawdown
        if drawdown >= self.max_drawdown_pct {
            if !self.is_halted {
                self.is_halted = true;
                self.halt_triggered_at_ms = now_ms;
                tracing::warn!(
                    "[kill-switch] SOFT HALT: drawdown {:.2}% >= max {:.2}%. No new entries.",
                    drawdown * 100.0, self.max_drawdown_pct * 100.0
                );
            }
            return false;
        }

        // Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss {
            if !self.is_halted {
                self.is_halted = true;
                self.halt_triggered_at_ms = now_ms;
                tracing::warn!(
                    "[kill-switch] SOFT HALT: daily loss ${:.2} exceeds max ${:.2}",
                    self.daily_pnl.abs(), self.max_daily_loss
                );
            }
            return false;
        }

        // Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses {
            if !self.is_halted {
                self.is_halted = true;
                self.halt_triggered_at_ms = now_ms;
                tracing::warn!(
                    "[kill-switch] SOFT HALT: {} consecutive losses >= max {}",
                    self.consecutive_losses, self.max_consecutive_losses
                );
            }
            return false;
        }

        // Check cooldown period
        if self.is_halted && !self.is_hard_halted {
            let elapsed = now_ms - self.halt_triggered_at_ms;
            if elapsed >= self.cooldown_ms && drawdown < self.max_drawdown_pct * 0.5 {
                self.is_halted = false;
                self.consecutive_losses = 0;
                tracing::info!(
                    "[kill-switch] Cooldown expired and drawdown recovered to {:.2}%. Resuming trading.",
                    drawdown * 100.0
                );
            }
        }

        !self.is_halted
    }

    /// Record a trade result (positive = profit, negative = loss).
    pub fn record_trade(&mut self, pnl: f64) {
        self.daily_pnl += pnl;
        if pnl < 0.0 {
            self.consecutive_losses += 1;
        } else {
            self.consecutive_losses = 0;
        }
    }

    /// Reset daily PnL tracking (call at start of each trading day).
    pub fn reset_daily(&mut self) {
        self.daily_pnl = 0.0;
        // Don't reset consecutive_losses - those span days
    }

    /// Check if new entries are allowed.
    #[inline]
    pub fn allows_new_entries(&self) -> bool {
        !self.is_halted
    }

    /// Check if all positions should be closed immediately (hard halt).
    #[inline]
    pub fn should_close_all(&self) -> bool {
        self.is_hard_halted
    }

    /// Get current drawdown as a percentage.
    pub fn current_drawdown_pct(&self) -> f64 {
        if self.peak_equity > 0.0 {
            (self.peak_equity - self.current_equity) / self.peak_equity
        } else {
            0.0
        }
    }

    /// Get current daily PnL.
    pub fn daily_pnl(&self) -> f64 {
        self.daily_pnl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_accuracy() {
        let mut kahan = KahanSum::new();
        
        // Add many small values
        for _ in 0..1000 {
            kahan.add(0.001);
        }
        
        assert!((kahan.get() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inventory_model_initialization() {
        let model = MarketMakerInventoryModel::new(100);
        assert_eq!(model.get_net_position(), 0.0);
        assert_eq!(model.get_inventory_pressure(), 0.0);
    }

    #[test]
    fn test_buy_trades_create_negative_position() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Aggressor buys, MM sells
        model.on_trade(100.0, true);
        
        assert!(model.get_net_position() < 0.0);
    }

    #[test]
    fn test_sell_trades_create_positive_position() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Aggressor sells, MM buys
        model.on_trade(100.0, false);
        
        assert!(model.get_net_position() > 0.0);
    }

    #[test]
    fn test_inventory_pressure_calculation() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Create consistent buy pressure (MM accumulates short position)
        for _ in 0..50 {
            model.on_trade(100.0, true);
        }
        
        let pressure = model.get_inventory_pressure();
        assert!(pressure < 0.0); // Negative pressure (MM is short)
        assert!(pressure >= -1.0); // Clamped
    }

    #[test]
    fn test_inventory_signal_bearish() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // MM accumulates long position (wants to sell)
        for _ in 0..50 {
            model.on_trade(100.0, false);
        }
        
        let (direction, score) = model.get_inventory_signal();
        assert_eq!(direction, -1); // Bearish
        assert!(score > 0.0);
    }

    #[test]
    fn test_inventory_signal_bullish() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // MM accumulates short position (wants to buy)
        for _ in 0..50 {
            model.on_trade(100.0, true);
        }
        
        let (direction, score) = model.get_inventory_signal();
        assert_eq!(direction, 1); // Bullish
        assert!(score > 0.0);
    }

    #[test]
    fn test_inventory_signal_neutral() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Balanced trades
        for _ in 0..25 {
            model.on_trade(100.0, true);
            model.on_trade(100.0, false);
        }
        
        let (direction, _) = model.get_inventory_signal();
        assert_eq!(direction, 0); // Neutral
    }

    #[test]
    fn test_reset_daily() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Build up position
        for _ in 0..50 {
            model.on_trade(100.0, true);
        }
        
        assert!(model.get_net_position() != 0.0);
        
        // Reset
        model.reset_daily();
        
        assert_eq!(model.get_net_position(), 0.0);
        assert_eq!(model.get_position_ema(), 0.0);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut model = MarketMakerInventoryModel::new(100);
        
        // Add trades and check EMA updates
        model.on_trade(100.0, true);
        let ema1 = model.get_position_ema();
        
        model.on_trade(100.0, true);
        let ema2 = model.get_position_ema();
        
        // EMA should be smoothing the position
        assert!(ema2 < ema1); // More negative
        assert!(ema2.abs() < model.get_net_position().abs()); // EMA lags
    }
}
