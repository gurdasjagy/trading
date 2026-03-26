//! Telegram Alert Sender — FEATURE 10.
//!
//! Sends HTTP POST requests to Telegram Bot API for real-time trade alerts,
//! stop-loss/take-profit notifications, circuit breaker trips, and daily P&L summaries.
//!
//! # Integration
//!
//! Integrated into telemetry.rs:TelemetryPublisher. Alerts are sent asynchronously
//! via tokio::spawn to avoid blocking the hot path.
//!
//! # Message Formatting
//!
//! - **Trade Opened**: Symbol, side, size, entry price, leverage
//! - **Trade Closed**: Symbol, exit price, PnL, duration
//! - **SL/TP Hit**: Symbol, trigger type, price, PnL
//! - **Circuit Breaker**: Reason, current equity, drawdown
//! - **Daily P&L**: Total PnL, win rate, largest win/loss
//!
//! Follows the pattern from gateio_gateway.rs REST calls (lines 800-850).

use reqwest::Client;
use serde_json::json;
use tracing::{debug, error, warn};

// ═══════════════════════════════════════════════════════════════════════════
// Telegram Alert Sender
// ═══════════════════════════════════════════════════════════════════════════

/// Sends alerts to a Telegram chat via the Bot API.
pub struct TelegramAlertSender {
    /// Telegram bot token.
    bot_token: String,
    /// Telegram chat ID to send messages to.
    chat_id: String,
    /// HTTP client for sending requests.
    client: Client,
    /// Whether the sender is enabled.
    enabled: bool,
}

impl TelegramAlertSender {
    /// Create a new Telegram alert sender.
    ///
    /// # Arguments
    /// * `bot_token` — Telegram bot token (from @BotFather)
    /// * `chat_id` — Chat ID to send messages to (can be user ID or group ID)
    pub fn new(bot_token: String, chat_id: String) -> Self {
        let enabled = !bot_token.is_empty() && !chat_id.is_empty();
        if enabled {
            debug!("[telegram] Alert sender initialized for chat {}", chat_id);
        } else {
            warn!("[telegram] Alert sender disabled (missing token or chat_id)");
        }

        Self {
            bot_token,
            chat_id,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            enabled,
        }
    }

    /// Send a message to the configured Telegram chat.
    ///
    /// # Arguments
    /// * `text` — Message text (supports Markdown formatting)
    pub async fn send_message(&self, text: &str) {
        if !self.enabled {
            return;
        }

        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.bot_token
        );

        let body = json!({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": true,
        });

        match self.client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    error!(
                        "[telegram] Send failed: HTTP {} — {}",
                        status,
                        &body_text[..body_text.len().min(200)]
                    );
                }
            }
            Err(e) => {
                error!("[telegram] Request failed: {}", e);
            }
        }
    }

    /// Format and send a trade opened alert.
    ///
    /// # Arguments
    /// * `symbol` — Trading symbol (e.g., "BTC_USDT")
    /// * `side` — "LONG" or "SHORT"
    /// * `size` — Position size in contracts
    /// * `entry_price` — Entry price
    /// * `leverage` — Leverage used
    /// * `stop_loss` — Stop loss price (optional)
    /// * `take_profit` — Take profit price (optional)
    pub async fn send_trade_opened_alert(
        &self,
        symbol: &str,
        side: &str,
        size: i64,
        entry_price: f64,
        leverage: i32,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    ) {
        let sl_str = stop_loss
            .map(|sl| format!("SL: ${:.4}", sl))
            .unwrap_or_else(|| "SL: None".to_string());
        let tp_str = take_profit
            .map(|tp| format!("TP: ${:.4}", tp))
            .unwrap_or_else(|| "TP: None".to_string());

        let text = format!(
            "🟢 *Trade Opened*\n\
             Symbol: `{}`\n\
             Side: *{}*\n\
             Size: {} contracts\n\
             Entry: ${:.4}\n\
             Leverage: {}x\n\
             {}\n\
             {}",
            symbol, side, size, entry_price, leverage, sl_str, tp_str
        );

        self.send_message(&text).await;
    }

    /// Format and send a trade closed alert.
    ///
    /// # Arguments
    /// * `symbol` — Trading symbol
    /// * `side` — "LONG" or "SHORT"
    /// * `entry_price` — Entry price
    /// * `exit_price` — Exit price
    /// * `pnl_usdt` — Realized PnL in USDT
    /// * `pnl_pct` — PnL as percentage
    /// * `duration_secs` — Trade duration in seconds
    pub async fn send_trade_closed_alert(
        &self,
        symbol: &str,
        side: &str,
        entry_price: f64,
        exit_price: f64,
        pnl_usdt: f64,
        pnl_pct: f64,
        duration_secs: u64,
    ) {
        let emoji = if pnl_usdt >= 0.0 { "✅" } else { "❌" };
        let pnl_sign = if pnl_usdt >= 0.0 { "+" } else { "" };

        let duration_str = if duration_secs < 60 {
            format!("{}s", duration_secs)
        } else if duration_secs < 3600 {
            format!("{}m", duration_secs / 60)
        } else {
            format!("{}h {}m", duration_secs / 3600, (duration_secs % 3600) / 60)
        };

        let text = format!(
            "{} *Trade Closed*\n\
             Symbol: `{}`\n\
             Side: *{}*\n\
             Entry: ${:.4}\n\
             Exit: ${:.4}\n\
             PnL: *{}{:.2} USDT* ({}{:.2}%)\n\
             Duration: {}",
            emoji, symbol, side, entry_price, exit_price, pnl_sign, pnl_usdt, pnl_sign, pnl_pct, duration_str
        );

        self.send_message(&text).await;
    }

    /// Format and send a stop-loss/take-profit hit alert.
    ///
    /// # Arguments
    /// * `symbol` — Trading symbol
    /// * `trigger_type` — "Stop Loss" or "Take Profit"
    /// * `trigger_price` — Price at which the trigger fired
    /// * `pnl_usdt` — Realized PnL
    pub async fn send_sl_tp_alert(
        &self,
        symbol: &str,
        trigger_type: &str,
        trigger_price: f64,
        pnl_usdt: f64,
    ) {
        let emoji = if trigger_type == "Take Profit" {
            "🎯"
        } else {
            "🛡️"
        };

        let text = format!(
            "{} *{} Hit*\n\
             Symbol: `{}`\n\
             Trigger Price: ${:.4}\n\
             PnL: {:.2} USDT",
            emoji, trigger_type, symbol, trigger_price, pnl_usdt
        );

        self.send_message(&text).await;
    }

    /// Format and send a circuit breaker trip alert.
    ///
    /// # Arguments
    /// * `reason` — Reason for the trip (e.g., "Daily Drawdown Exceeded")
    /// * `current_equity` — Current account equity
    /// * `drawdown_pct` — Current drawdown percentage
    /// * `consecutive_losses` — Number of consecutive losses
    pub async fn send_circuit_breaker_alert(
        &self,
        reason: &str,
        current_equity: f64,
        drawdown_pct: f64,
        consecutive_losses: u32,
    ) {
        let text = format!(
            "🚨 *Circuit Breaker Tripped*\n\
             Reason: *{}*\n\
             Current Equity: ${:.2}\n\
             Drawdown: {:.2}%\n\
             Consecutive Losses: {}\n\
             Trading halted until manual reset.",
            reason, current_equity, drawdown_pct, consecutive_losses
        );

        self.send_message(&text).await;
    }

    /// Format and send a daily P&L summary.
    ///
    /// # Arguments
    /// * `total_pnl` — Total realized PnL for the day
    /// * `total_trades` — Number of trades executed
    /// * `win_count` — Number of winning trades
    /// * `loss_count` — Number of losing trades
    /// * `largest_win` — Largest winning trade
    /// * `largest_loss` — Largest losing trade
    pub async fn send_daily_pnl_summary(
        &self,
        total_pnl: f64,
        total_trades: u64,
        win_count: u64,
        loss_count: u64,
        largest_win: f64,
        largest_loss: f64,
    ) {
        let win_rate = if total_trades > 0 {
            (win_count as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        let emoji = if total_pnl >= 0.0 { "📈" } else { "📉" };
        let pnl_sign = if total_pnl >= 0.0 { "+" } else { "" };

        let text = format!(
            "{} *Daily P&L Summary*\n\
             Total PnL: *{}{:.2} USDT*\n\
             Trades: {} (W: {}, L: {})\n\
             Win Rate: {:.1}%\n\
             Largest Win: +{:.2} USDT\n\
             Largest Loss: {:.2} USDT",
            emoji, pnl_sign, total_pnl, total_trades, win_count, loss_count, win_rate, largest_win, largest_loss
        );

        self.send_message(&text).await;
    }

    // ── CONFIG 2: Additional alert types ──────────────────────────────

    /// Format and send a funding arb opportunity alert.
    ///
    /// # Arguments
    /// * `symbol` — Trading symbol (e.g., "BTC_USDT")
    /// * `long_exchange` — Exchange for the long leg
    /// * `short_exchange` — Exchange for the short leg
    /// * `funding_diff_bps` — Funding rate differential in basis points
    /// * `estimated_apy` — Estimated annualized yield
    pub async fn send_funding_arb_alert(
        &self,
        symbol: &str,
        long_exchange: &str,
        short_exchange: &str,
        funding_diff_bps: f64,
        estimated_apy: f64,
    ) {
        let text = format!(
            "💰 *Funding Arb Opportunity*\n\
             Symbol: `{}`\n\
             Long: *{}* → Short: *{}*\n\
             Funding Diff: {:.2} bps\n\
             Est. APY: {:.1}%",
            symbol, long_exchange, short_exchange, funding_diff_bps, estimated_apy
        );

        self.send_message(&text).await;
    }

    /// Format and send a margin ratio warning alert.
    ///
    /// # Arguments
    /// * `exchange` — Exchange name
    /// * `margin_ratio_pct` — Current margin ratio as percentage
    /// * `threshold_pct` — Warning threshold percentage
    /// * `available_margin` — Available margin in USDT
    pub async fn send_margin_ratio_warning(
        &self,
        exchange: &str,
        margin_ratio_pct: f64,
        threshold_pct: f64,
        available_margin: f64,
    ) {
        let urgency = if margin_ratio_pct > 90.0 {
            "🔴 CRITICAL"
        } else if margin_ratio_pct > 80.0 {
            "🟠 HIGH"
        } else {
            "🟡 WARNING"
        };

        let text = format!(
            "⚠️ *Margin Ratio Warning* {}\n\
             Exchange: *{}*\n\
             Margin Ratio: *{:.1}%* (threshold: {:.1}%)\n\
             Available Margin: ${:.2}\n\
             Action: Reduce positions or add margin.",
            urgency, exchange, margin_ratio_pct, threshold_pct, available_margin
        );

        self.send_message(&text).await;
    }

    /// Format and send a delta neutrality breach alert.
    ///
    /// # Arguments
    /// * `net_delta` — Current net delta exposure in USDT
    /// * `max_delta` — Maximum allowed net delta
    /// * `positions` — Summary of positions contributing to delta
    pub async fn send_delta_neutrality_breach(
        &self,
        net_delta: f64,
        max_delta: f64,
        positions: &str,
    ) {
        let direction = if net_delta > 0.0 { "LONG" } else { "SHORT" };

        let text = format!(
            "⚖️ *Delta Neutrality Breach*\n\
             Net Delta: *${:.2}* ({})\n\
             Max Allowed: ${:.2}\n\
             Breach: {:.1}%\n\
             Positions:\n{}\n\
             Action: Rebalancing required.",
            net_delta.abs(),
            direction,
            max_delta,
            (net_delta.abs() / max_delta * 100.0),
            positions
        );

        self.send_message(&text).await;
    }

    /// Format and send an exchange connectivity issue alert.
    ///
    /// # Arguments
    /// * `exchange` — Exchange name
    /// * `issue` — Description of the connectivity issue
    /// * `last_seen_secs` — Seconds since last successful connection
    /// * `is_recovered` — Whether connectivity has been restored
    pub async fn send_exchange_connectivity_alert(
        &self,
        exchange: &str,
        issue: &str,
        last_seen_secs: u64,
        is_recovered: bool,
    ) {
        if is_recovered {
            let text = format!(
                "🟢 *Exchange Connectivity Restored*\n\
                 Exchange: *{}*\n\
                 Downtime: {}s\n\
                 Status: Back online",
                exchange, last_seen_secs
            );
            self.send_message(&text).await;
        } else {
            let text = format!(
                "🔴 *Exchange Connectivity Lost*\n\
                 Exchange: *{}*\n\
                 Issue: {}\n\
                 Last Seen: {}s ago\n\
                 Action: Orders may not be executing. Manual intervention may be required.",
                exchange, issue, last_seen_secs
            );
            self.send_message(&text).await;
        }
    }

    /// Check if the sender is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for TelegramAlertSender {
    fn default() -> Self {
        Self::new(String::new(), String::new())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telegram_sender_disabled_when_no_credentials() {
        let sender = TelegramAlertSender::new(String::new(), String::new());
        assert!(!sender.is_enabled());
    }

    #[test]
    fn test_telegram_sender_enabled_with_credentials() {
        let sender = TelegramAlertSender::new("test_token".to_string(), "12345".to_string());
        assert!(sender.is_enabled());
    }

    #[tokio::test]
    async fn test_send_message_when_disabled() {
        let sender = TelegramAlertSender::default();
        // Should not panic when disabled
        sender.send_message("Test message").await;
    }

    #[tokio::test]
    async fn test_trade_opened_alert_formatting() {
        let sender = TelegramAlertSender::default();
        // Should not panic
        sender
            .send_trade_opened_alert(
                "BTC_USDT",
                "LONG",
                10,
                50000.0,
                5,
                Some(49000.0),
                Some(52000.0),
            )
            .await;
    }

    #[tokio::test]
    async fn test_trade_closed_alert_formatting() {
        let sender = TelegramAlertSender::default();
        sender
            .send_trade_closed_alert("BTC_USDT", "LONG", 50000.0, 51000.0, 100.0, 2.0, 3600)
            .await;
    }

    #[tokio::test]
    async fn test_circuit_breaker_alert_formatting() {
        let sender = TelegramAlertSender::default();
        sender
            .send_circuit_breaker_alert("Daily Drawdown Exceeded", 9500.0, 5.0, 3)
            .await;
    }

    #[tokio::test]
    async fn test_daily_pnl_summary_formatting() {
        let sender = TelegramAlertSender::default();
        sender
            .send_daily_pnl_summary(250.0, 10, 7, 3, 150.0, -50.0)
            .await;
    }
}
