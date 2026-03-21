"""Multi-channel alert manager — Telegram, Discord, and Email."""

from __future__ import annotations

import asyncio
import smtplib
import time
from collections import deque
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Deque, Dict, Optional

import httpx
from loguru import logger

from config.settings import Settings

# ---------------------------------------------------------------------------
# Rate-limit constants: max 20 messages per 60-second sliding window
# ---------------------------------------------------------------------------
_RATE_LIMIT_MAX: int = 20
_RATE_LIMIT_WINDOW: float = 60.0  # seconds


class AlertType(str, Enum):
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    LIQUIDATION = "liquidation"
    MANUAL_TRADE = "manual_trade"
    POSITION_REDUCED = "position_reduced"
    EMERGENCY_CLOSE = "emergency_close"
    BREAK_EVEN_ACTIVATED = "break_even_activated"
    TRAILING_TP = "trailing_tp"
    DAILY_SUMMARY = "daily_summary"
    RISK_WARNING = "risk_warning"
    CIRCUIT_BREAKER = "circuit_breaker"
    SYSTEM_ERROR = "system_error"
    OPPORTUNITY = "opportunity"
    WHALE_ALERT = "whale_alert"
    NEWS_ALERT = "news_alert"


_ALERT_EMOJIS: Dict[AlertType, str] = {
    AlertType.TRADE_OPENED: "📈",
    AlertType.TRADE_CLOSED: "💰",
    AlertType.SL_HIT: "🛑",
    AlertType.TP_HIT: "🎯",
    AlertType.LIQUIDATION: "💥",
    AlertType.MANUAL_TRADE: "⚡",
    AlertType.POSITION_REDUCED: "✂️",
    AlertType.EMERGENCY_CLOSE: "🚨",
    AlertType.BREAK_EVEN_ACTIVATED: "🔒",
    AlertType.TRAILING_TP: "📐",
    AlertType.DAILY_SUMMARY: "📊",
    AlertType.RISK_WARNING: "⚠️",
    AlertType.CIRCUIT_BREAKER: "🚨",
    AlertType.SYSTEM_ERROR: "❌",
    AlertType.OPPORTUNITY: "🎯",
    AlertType.WHALE_ALERT: "🐋",
    AlertType.NEWS_ALERT: "📰",
}

# Discord embed colours keyed by severity level
_LEVEL_COLORS: Dict[str, int] = {
    "info": 0x3498DB,  # blue
    "warning": 0xF39C12,  # orange
    "error": 0xE74C3C,  # red
    "critical": 0x8B0000,  # dark red
    "success": 0x2ECC71,  # green
}


def _format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string (e.g. '2h 15m 3s')."""
    if not seconds or seconds <= 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def _strip_bot_prefix(token: str) -> str:
    """Remove a leading ``bot`` prefix from a Telegram bot token if present.

    Some users copy-paste their token including the ``bot`` prefix (e.g.
    ``bot123456:ABC...``).  When the token is then embedded in the API URL as
    ``/bot{token}/...`` the resulting path becomes ``/botbot123456:ABC.../``
    which returns a 404 from the Telegram API.
    """
    token = token.strip()
    if token[:3].lower() == "bot":
        token = token[3:]
    return token


class AlertManager:
    """Sends rich alerts via Telegram, Discord, and Email."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings.get_settings()
        # Sliding-window rate limiter: stores monotonic timestamps of sent msgs
        self._sent_times: Deque[float] = deque()
        self._rate_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _check_rate_limit(self, force: bool = False) -> bool:
        """Return True if within the rate limit, False (and log) if exceeded.

        Args:
            force: When ``True``, bypass the rate limit check for critical/pinned
                alerts.  The message is still counted toward the sliding window.
        """
        async with self._rate_lock:
            now = time.monotonic()
            # Evict timestamps outside the sliding window
            while self._sent_times and self._sent_times[0] < now - _RATE_LIMIT_WINDOW:
                self._sent_times.popleft()
            if not force and len(self._sent_times) >= _RATE_LIMIT_MAX:
                logger.warning(
                    "Alert rate limit reached ({} msgs/min) — dropping message",
                    _RATE_LIMIT_MAX,
                )
                return False
            self._sent_times.append(now)
            return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    async def send_alert(self, message: str, level: str = "info") -> bool:
        """Send a plain-text alert to all configured channels.

        This is the primary entry-point used by the engine and circuit breaker.
        If Telegram or Discord credentials are absent the channel is skipped with
        a warning but no exception is raised (graceful degradation).

        Args:
            message: The message text to send.
            level: Severity level — ``"info"``, ``"warning"``, ``"error"``,
                ``"critical"``, or ``"success"``.

        Returns:
            ``True`` if at least one channel delivered the alert successfully.
        """
        if not await self._check_rate_limit():
            return False

        success = False
        monitoring = self._settings.monitoring

        if monitoring.enable_telegram_alerts:
            tg_token = self._settings.alert_telegram_bot_token or self._settings.telegram_bot_token
            tg_chat = self._settings.alert_telegram_chat_id or self._settings.telegram_chat_id
            if tg_token and tg_chat:
                success = await self.send_telegram(message) or success
            else:
                logger.warning(
                    "Telegram credentials (ALERT_TELEGRAM_BOT_TOKEN / ALERT_TELEGRAM_CHAT_ID) "
                    "not configured — skipping Telegram alert"
                )

        if monitoring.enable_discord_alerts:
            if self._settings.discord_webhook_url:
                success = await self._send_discord_embed(message, level) or success
            else:
                logger.warning("DISCORD_WEBHOOK_URL not configured — skipping Discord alert")

        if monitoring.enable_email_alerts and self._settings.alert_email_to:
            subject = f"[TradingBot] {level.upper()} Alert"
            success = await self.send_email(subject, message) or success

        return success

    async def send_trade_alert(self, trade_result: dict) -> bool:
        """Send a formatted trade-execution alert to all configured channels.

        Args:
            trade_result: Trade result dict (keys: symbol, direction, size, price /
                entry_price, pnl, risk_reward, …).

        Returns:
            ``True`` if at least one channel delivered the alert successfully.
        """
        message = self._format_trade_alert(trade_result)
        pnl = trade_result.get("pnl") or 0.0
        level = "success" if pnl >= 0 else "warning"
        return await self.send_alert(message, level=level)

    async def send_daily_report(self, summary: dict) -> bool:
        """Send an end-of-day summary alert to all configured channels.

        Args:
            summary: Summary dict (keys: date, total_pnl, win_rate, trade_count, …).

        Returns:
            ``True`` if at least one channel delivered the alert successfully.
        """
        message = self._format_daily_summary(summary)
        total_pnl = summary.get("total_pnl") or 0.0
        level = "success" if total_pnl >= 0 else "warning"
        return await self.send_alert(message, level=level)

    async def send_typed_alert(self, alert_type: AlertType, data: dict) -> bool:
        """Send an alert using a typed ``AlertType`` and structured data payload.

        Formats the message via the built-in formatters then delegates to
        :meth:`send_alert`.

        Args:
            alert_type: The type of alert to send.
            data: Alert payload used to format the message.

        Returns:
            ``True`` if at least one channel delivered the alert successfully.
        """
        message = self._format_message(alert_type, data)
        level = "critical" if alert_type == AlertType.CIRCUIT_BREAKER else "info"
        return await self.send_alert(message, level=level)

    async def send_trade_open_alert(self, trade: dict, mode: str = "paper") -> bool:
        """Send a detailed trade-open alert to all configured channels.

        Args:
            trade: Trade result dict containing symbol, direction/side, size,
                leverage, price/entry_price, stop_loss, take_profit, strategy,
                pnl_pct, margin_used, order_id, exchange.
            mode: Trading mode string — ``"live"``, ``"testnet"``, or
                ``"paper"``/``"manual"``.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_trade_open(trade, mode)
        return await self.send_alert(message, level="info")

    async def send_telegram_and_pin(self, message: str) -> bool:
        """Send a Telegram message and pin it in the chat.

        Critical/important alerts (trade opens, liquidations, circuit breakers)
        are sent via this method so they remain visible at the top of the chat.
        The rate-limit check is bypassed (``force=True``) for pinned messages.

        Args:
            message: HTML-formatted message text.

        Returns:
            ``True`` on success.
        """
        token = self._settings.alert_telegram_bot_token or self._settings.telegram_bot_token
        chat_id = self._settings.alert_telegram_chat_id or self._settings.telegram_chat_id
        if not token or not chat_id:
            return False
        token = _strip_bot_prefix(token)

        send_url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(send_url, json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "Telegram send+pin failed: status={} body={}",
                        resp.status_code,
                        resp.text,
                    )
                    return False
                result = resp.json()
                message_id = result.get("result", {}).get("message_id")

                # Pin the message (disable_notification avoids an extra ping)
                if message_id:
                    pin_url = f"https://api.telegram.org/bot{token}/pinChatMessage"
                    pin_payload = {
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "disable_notification": True,
                    }
                    await client.post(pin_url, json=pin_payload)
            return True
        except Exception as exc:
            logger.error("Telegram send+pin error: {}", exc)
            return False

    async def send_trade_open_alert_and_pin(self, trade: dict, mode: str = "paper") -> bool:
        """Send a detailed trade-open alert, pin it in Telegram, and deliver to other channels.

        The Telegram message is sent-and-pinned so it stays visible.  Other
        channels (Discord, email) receive the message via the normal ``send_alert``
        path.  The rate-limit check is bypassed (``force=True``) because
        trade-open notifications are critical.

        Args:
            trade: Trade result dict with symbol, direction/side, size, leverage,
                price/entry_price, stop_loss, take_profit, take_profit_levels,
                position_size, strategy, margin_used, order_id, exchange.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_trade_open(trade, mode)

        success = False
        # Bypass rate limit for trade-open alerts (critical information)
        if await self._check_rate_limit(force=True):
            success = await self.send_telegram_and_pin(message)

        # Also push to Discord via normal path (rate-limited normally)
        if self._settings.discord_webhook_url:
            try:
                discord_ok = await self._send_discord_embed(message, level="info")
                success = success or discord_ok
            except Exception:
                pass

        return success

    async def send_sl_alert(self, trade: dict, mode: str = "paper") -> bool:
        """Send a detailed stop-loss hit alert to all configured channels.

        Args:
            trade: Trade data dict containing symbol, side, size, entry_price,
                exit_price/price, pnl, pnl_pct, strategy, duration_seconds,
                fees, leverage.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_sl_hit(trade, mode)
        return await self.send_alert(message, level="warning")

    async def send_tp_alert(self, trade: dict, mode: str = "paper") -> bool:
        """Send a detailed take-profit hit alert to all configured channels.

        Args:
            trade: Trade data dict containing symbol, side, size, entry_price,
                exit_price/price, pnl, pnl_pct, strategy, duration_seconds,
                fees, leverage, risk_reward.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_tp_hit(trade, mode)
        return await self.send_alert(message, level="success")

    async def send_liquidation_alert(self, position: dict, mode: str = "paper") -> bool:
        """Send a detailed liquidation alert to all configured channels.

        Args:
            position: Position data dict containing symbol, side, size,
                entry_price, liquidation_price, margin_used, leverage,
                unrealized_pnl, strategy.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_liquidation(position, mode)
        return await self.send_alert(message, level="critical")

    async def send_manual_trade_warning(self, trade: dict, mode: str = "paper") -> bool:
        """Send a manual-trade warning alert to all configured channels.

        Args:
            trade: Trade data dict.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_manual_trade(trade, mode)
        return await self.send_alert(message, level="warning")

    async def send_position_reduced_alert(self, trade: dict, mode: str = "paper") -> bool:
        """Send a position-reduced alert to all configured channels.

        Args:
            trade: Trade data dict with symbol, side, reduced_by, remaining,
                realized_pnl, strategy.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_position_reduced(trade, mode)
        pnl = trade.get("realized_pnl", 0.0) or 0.0
        return await self.send_alert(message, level="success" if pnl >= 0 else "warning")

    async def send_emergency_close_alert(self, trade: dict, mode: str = "paper") -> bool:
        """Send an emergency-close alert to all configured channels.

        Args:
            trade: Trade data dict with symbol, reason, pnl, strategy.
            mode: Trading mode string.

        Returns:
            ``True`` if at least one channel delivered the alert.
        """
        message = self._format_emergency_close(trade, mode)
        return await self.send_alert(message, level="critical")

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    async def send_telegram(self, message: str) -> bool:
        """Send a message via the configured Telegram bot (HTML parse mode).

        Uses the Telegram Bot API ``sendMessage`` endpoint.
        Reads ``ALERT_TELEGRAM_BOT_TOKEN`` and ``ALERT_TELEGRAM_CHAT_ID`` from
        settings (falls back to ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``).

        Args:
            message: HTML-formatted message text.

        Returns:
            ``True`` on success.
        """
        token = self._settings.alert_telegram_bot_token or self._settings.telegram_bot_token
        chat_id = self._settings.alert_telegram_chat_id or self._settings.telegram_chat_id
        if not token or not chat_id:
            logger.debug("Telegram credentials not configured")
            return False
        token = _strip_bot_prefix(token)
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.debug("Telegram alert sent")
                return True
            if resp.status_code == 400 and "chat not found" in resp.text.lower():
                logger.warning(
                    "Telegram send failed: chat not found (chat_id={!r}). "
                    "Fix: make sure you have sent /start to your bot, then obtain the "
                    "correct chat_id via https://api.telegram.org/bot{}/getUpdates and "
                    "set ALERT_TELEGRAM_CHAT_ID in your .env file.",
                    chat_id,
                    token[:8] + "…",
                )
                return False
            logger.warning("Telegram send failed: status={} body={}", resp.status_code, resp.text)
            return False
        except Exception as exc:
            logger.error("Telegram send error: {}", exc)
            return False

    async def _send_discord_embed(self, message: str, level: str = "info") -> bool:
        """POST an embed-formatted message to the configured Discord webhook.

        Reads ``DISCORD_WEBHOOK_URL`` from settings.

        Args:
            message: Message text for the embed description (max 4 096 chars).
            level: Severity level used to select the embed sidebar colour.

        Returns:
            ``True`` on success.
        """
        webhook_url = self._settings.discord_webhook_url
        if not webhook_url:
            logger.debug("Discord webhook URL not configured")
            return False
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
        payload = {
            "embeds": [
                {
                    "description": message[:4096],
                    "color": color,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            ]
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json=payload)
            if resp.status_code in (200, 204):
                logger.debug("Discord alert sent")
                return True
            logger.warning("Discord send failed: status={} body={}", resp.status_code, resp.text)
            return False
        except Exception as exc:
            logger.error("Discord send error: {}", exc)
            return False

    async def send_discord(self, message: str) -> bool:
        """Send a plain-content Discord webhook message (embed format).

        Args:
            message: Text message to send.

        Returns:
            ``True`` on success.
        """
        return await self._send_discord_embed(message)

    async def send_email(self, subject: str, body: str) -> bool:
        """Send an alert email via SMTP.

        Args:
            subject: Email subject line.
            body: Plain-text email body.

        Returns:
            ``True`` on success.
        """
        s = self._settings
        if not all([s.alert_email_from, s.alert_email_to, s.alert_email_smtp_host]):
            logger.debug("Email credentials not fully configured")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = s.alert_email_from  # type: ignore[arg-type]
            msg["To"] = s.alert_email_to  # type: ignore[arg-type]
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(s.alert_email_smtp_host, s.alert_email_smtp_port) as server:  # type: ignore[arg-type]
                server.starttls()
                if s.alert_email_smtp_user and s.alert_email_smtp_password:
                    server.login(s.alert_email_smtp_user, s.alert_email_smtp_password)
                server.sendmail(s.alert_email_from, s.alert_email_to, msg.as_string())
            logger.debug("Email alert sent to {}", s.alert_email_to)
            return True
        except Exception as exc:
            logger.error("Email send error: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Message formatters
    # ------------------------------------------------------------------

    def _format_message(self, alert_type: AlertType, data: dict) -> str:
        """Dispatch to the appropriate formatter."""
        formatters = {
            AlertType.TRADE_OPENED: lambda d: self._format_trade_open(d, d.get("mode", "paper")),
            AlertType.TRADE_CLOSED: self._format_trade_alert,
            AlertType.SL_HIT: lambda d: self._format_sl_hit(d, d.get("mode", "paper")),
            AlertType.TP_HIT: lambda d: self._format_tp_hit(d, d.get("mode", "paper")),
            AlertType.LIQUIDATION: lambda d: self._format_liquidation(d, d.get("mode", "paper")),
            AlertType.MANUAL_TRADE: lambda d: self._format_manual_trade(d, d.get("mode", "paper")),
            AlertType.POSITION_REDUCED: lambda d: self._format_position_reduced(d, d.get("mode", "paper")),
            AlertType.EMERGENCY_CLOSE: lambda d: self._format_emergency_close(d, d.get("mode", "paper")),
            AlertType.BREAK_EVEN_ACTIVATED: self._format_break_even,
            AlertType.TRAILING_TP: self._format_trailing_tp,
            AlertType.DAILY_SUMMARY: self._format_daily_summary,
            AlertType.CIRCUIT_BREAKER: self._format_circuit_breaker,
            AlertType.RISK_WARNING: self._format_risk_warning,
            AlertType.SYSTEM_ERROR: self._format_system_error,
            AlertType.WHALE_ALERT: self._format_generic,
            AlertType.NEWS_ALERT: self._format_generic,
            AlertType.OPPORTUNITY: self._format_generic,
        }
        formatter = formatters.get(alert_type, self._format_generic)
        emoji = _ALERT_EMOJIS.get(alert_type, "ℹ️")
        return f"{emoji} {formatter(data)}"

    # ── Detailed per-event formatters (delegate to module-level helpers) ──

    def _format_trade_open(self, trade: dict, mode: str = "paper") -> str:
        """Format a detailed trade-open message."""
        return _fmt_trade_open(trade, mode)

    def _format_trade_alert(self, trade: dict) -> str:
        """Format a generic trade-closed / trade-result message."""
        return _fmt_trade_closed(trade)

    def _format_sl_hit(self, trade: dict, mode: str = "paper") -> str:
        return _fmt_sl_hit(trade, mode)

    def _format_tp_hit(self, trade: dict, mode: str = "paper") -> str:
        return _fmt_tp_hit(trade, mode)

    def _format_liquidation(self, position: dict, mode: str = "paper") -> str:
        return _fmt_liquidation(position, mode)

    def _format_manual_trade(self, trade: dict, mode: str = "paper") -> str:
        return _fmt_manual_trade(trade, mode)

    def _format_position_reduced(self, trade: dict, mode: str = "paper") -> str:
        """Format a position-reduced message."""
        tag = _fmt_mode_tag(mode)
        symbol = trade.get("symbol", "N/A")
        reduced_by = trade.get("reduced_by") or trade.get("reduced_amount") or 0.0
        remaining = trade.get("remaining") or trade.get("remaining_amount") or 0.0
        realized_pnl = trade.get("realized_pnl") or trade.get("pnl") or 0.0
        pnl_pct = trade.get("pnl_pct") or 0.0
        strategy = trade.get("strategy") or "—"
        reason = (trade.get("reason") or "partial close").replace("_", " ").title()
        pnl_emoji = "✅" if realized_pnl >= 0 else "⚠️"

        return (
            f"{tag} ✂️ <b>POSITION REDUCED — {symbol}</b>\n"
            f"{'━' * 28}\n"
            f"Reason:       {reason}\n"
            f"Reduced By:   <code>{reduced_by:.4f}</code>\n"
            f"Remaining:    <code>{remaining:.4f}</code>\n"
            f"{'━' * 28}\n"
            f"{pnl_emoji} Realized PnL: <b><code>{realized_pnl:+.4f} USDT</code></b>"
            f"  (<code>{pnl_pct:+.2f}%</code>)\n"
            f"Strategy:     {strategy}"
        )

    def _format_emergency_close(self, trade: dict, mode: str = "paper") -> str:
        """Format an emergency-close message."""
        tag = _fmt_mode_tag(mode)
        symbol = trade.get("symbol", "N/A")
        reason = trade.get("reason") or "Emergency risk limit"
        pnl = trade.get("pnl") or 0.0
        strategy = trade.get("strategy") or "—"

        return (
            f"{tag} 🚨 <b>EMERGENCY CLOSE — {symbol}</b>\n"
            f"{'━' * 28}\n"
            f"⚠️ Position forcibly closed by risk system!\n"
            f"{'━' * 28}\n"
            f"Reason:    {reason}\n"
            f"PnL:       <code>{pnl:+.4f} USDT</code>\n"
            f"Strategy:  {strategy}"
        )

    def _format_break_even(self, data: dict) -> str:
        """Format a break-even stop activated message."""
        symbol = data.get("symbol", "N/A")
        entry = data.get("entry_price") or data.get("price") or 0.0
        new_sl = data.get("new_stop_loss") or data.get("sl") or 0.0
        profit = data.get("profit") or 0.0
        return (
            f"<b>Break-Even Stop Activated — {symbol}</b>\n"
            f"{'━' * 28}\n"
            f"Entry:     <code>{entry:.4f}</code>\n"
            f"New SL:    <code>{new_sl:.4f}</code> (at break-even)\n"
            f"Profit:    <code>{profit:+.4f} USDT</code> at activation"
        )

    def _format_trailing_tp(self, data: dict) -> str:
        """Format a trailing take-profit activated/updated message."""
        symbol = data.get("symbol", "N/A")
        price = data.get("price") or 0.0
        distance = data.get("distance") or data.get("trailing_distance") or 0.0
        return (
            f"<b>Trailing TP {'Activated' if data.get('activated') else 'Updated'} — {symbol}</b>\n"
            f"Price:     <code>{price:.4f}</code>\n"
            f"Distance:  <code>{distance:.4f}</code>"
        )

    def _format_daily_summary(self, summary: dict) -> str:
        date_ = summary.get("date", "N/A")
        total_pnl = summary.get("total_pnl", 0.0) or 0.0
        win_rate = summary.get("win_rate", 0.0) or 0.0
        trades = summary.get("trade_count") or summary.get("total_trades") or 0
        wins = summary.get("wins", 0) or 0
        losses = summary.get("losses", 0) or 0
        best = summary.get("best_trade", 0.0) or 0.0
        worst = summary.get("worst_trade", 0.0) or 0.0
        fees = summary.get("fees_paid", 0.0) or 0.0
        balance = summary.get("ending_balance") or summary.get("balance") or 0.0
        drawdown = summary.get("max_drawdown", 0.0) or 0.0
        pnl_emoji = "✅" if total_pnl >= 0 else "❌"

        msg = (
            f"<b>📊 Daily Summary — {date_}</b>\n"
            f"{'━' * 28}\n"
            f"{pnl_emoji} Total PnL:   <b><code>{total_pnl:+.4f} USDT</code></b>\n"
            f"Balance:      <code>${balance:.2f}</code>\n"
            f"{'━' * 28}\n"
            f"Trades:       {trades}  (✅ {wins} / ❌ {losses})\n"
            f"Win Rate:     <b>{win_rate:.1f}%</b>\n"
            f"Best Trade:   <code>{best:+.4f} USDT</code>\n"
            f"Worst Trade:  <code>{worst:+.4f} USDT</code>\n"
            f"Fees Paid:    <code>-{fees:.4f} USDT</code>\n"
        )
        if drawdown:
            msg += f"Max Drawdown: <code>{drawdown:.2f}%</code>\n"
        return msg

    def _format_circuit_breaker(self, data: dict) -> str:
        reason = data.get("reason", "Unknown")
        positions_closed = data.get("positions_closed", 0)
        total_pnl = data.get("total_pnl") or 0.0
        return (
            f"<b>🚨 CIRCUIT BREAKER TRIGGERED</b>\n"
            f"{'━' * 28}\n"
            f"Reason: {reason}\n"
            f"Positions Closed: {positions_closed}\n"
            f"Total PnL: <code>{total_pnl:+.4f} USDT</code>\n"
            f"{'━' * 28}\n"
            "⚠️ All positions closed. Manual reset required.\n"
            "Action: Investigate logs before resuming."
        )

    def _format_risk_warning(self, data: dict) -> str:
        warning = data.get("warning", "Risk threshold approached")
        drawdown = data.get("drawdown_pct") or 0.0
        exposure = data.get("exposure_pct") or 0.0
        msg = f"<b>⚠️ Risk Warning</b>\n{warning}"
        if drawdown:
            msg += f"\nDrawdown: <code>{drawdown:.2f}%</code>"
        if exposure:
            msg += f"\nExposure: <code>{exposure:.2f}%</code>"
        return msg

    def _format_system_error(self, data: dict) -> str:
        error = data.get("error", "Unknown error")
        component = data.get("component") or ""
        return (
            "<b>❌ System Error</b>\n"
            + (f"Component: {component}\n" if component else "")
            + f"Error: {error}"
        )

    def _format_generic(self, data: dict) -> str:
        lines = [f"{k}: {v}" for k, v in data.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level message formatting helpers (shared by AlertManager + TelegramAlerter)
# ---------------------------------------------------------------------------


def _fmt_mode_tag(mode: str) -> str:
    """Return a mode tag emoji string for a given trading mode."""
    m = mode.upper()
    if m == "LIVE":
        return "🔴 [LIVE]"
    if m == "TESTNET":
        return "🟡 [TESTNET]"
    if m == "MANUAL":
        return "⚡ [MANUAL]"
    return "⚪ [PAPER]"


def _fmt_trade_open(trade: dict, mode: str = "paper") -> str:
    """Format a detailed trade-open message (module-level helper)."""
    tag = _fmt_mode_tag(mode)
    symbol = trade.get("symbol", "N/A")
    side = (trade.get("direction") or trade.get("side") or "N/A").upper()
    side_emoji = "🟢 LONG" if "LONG" in side or "BUY" in side else "🔴 SHORT"
    price = trade.get("price") or trade.get("entry_price") or trade.get("filled_price") or 0.0
    size = trade.get("size") or trade.get("amount") or 0.0
    leverage = trade.get("leverage", 1)
    sl = trade.get("stop_loss")
    # Prefer explicit take_profit; fall back to first element of take_profit_levels list
    tp_levels_raw = trade.get("take_profit_levels") or []
    tp = trade.get("take_profit") or (tp_levels_raw[0] if tp_levels_raw else None)
    strategy = trade.get("strategy") or trade.get("strategy_name") or "—"
    margin = trade.get("margin_used") or trade.get("margin") or 0.0
    order_id = trade.get("order_id") or trade.get("id") or "—"
    exchange = trade.get("exchange") or "—"
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Position size in USDT
    position_size_usdt = trade.get("position_size") or trade.get("size_usdt") or 0.0
    if not position_size_usdt and price and size:
        position_size_usdt = float(price) * float(size)

    # Risk/reward ratio
    rr_str = "—"
    if sl and tp and price:
        try:
            entry_f = float(price)
            sl_f = float(sl)
            tp_f = float(tp)
            risk = abs(entry_f - sl_f)
            reward = abs(tp_f - entry_f)
            if risk > 1e-10:
                rr_str = f"{reward / risk:.2f}R"
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # Estimated liquidation price (simplified isolated margin approximation).
    # Assumes 100% margin efficiency with no maintenance margin adjustment.
    # Actual liq price depends on exchange tier, fees, and funding — treat as indicative.
    est_liq_str = "—"
    try:
        lev = float(leverage) if leverage else 1.0
        entry_f = float(price) if price else 0.0
        if lev > 1 and entry_f > 1e-10:
            # Approximate: liq = entry * (1 - 1/leverage) for LONG,
            #              liq = entry * (1 + 1/leverage) for SHORT
            if "LONG" in side or "BUY" in side:
                est_liq = entry_f * (1.0 - 1.0 / lev)
            else:
                est_liq = entry_f * (1.0 + 1.0 / lev)
            est_liq_str = f"{est_liq:.4f}"
    except (TypeError, ValueError):
        pass

    msg = (
        f"{tag} 📈 <b>NEW TRADE OPENED</b>\n"
        f"{'━' * 28}\n"
        f"Symbol:       <b>{symbol}</b>\n"
        f"Direction:    {side_emoji}\n"
        f"Price:        <code>{price:.4f}</code>\n"
        f"Size:         <code>{size:.4f}</code>"
    )
    if position_size_usdt:
        msg += f"  (<code>${position_size_usdt:.2f} USDT</code>)"
    msg += f"\nLeverage:     <b>{leverage}×</b>\n"
    if sl:
        sl_dist = abs(float(price) - float(sl)) / float(price) * 100 if price > 1e-10 else 0.0
        msg += f"Stop Loss:    <code>{float(sl):.4f}</code> ({sl_dist:.2f}% away)\n"
    else:
        msg += "Stop Loss:    <i>not set</i>\n"
    if tp:
        if tp_levels_raw and isinstance(tp_levels_raw, (list, tuple)) and len(tp_levels_raw) > 1:
            tp_str = " / ".join(f"<code>{float(t):.4f}</code>" for t in tp_levels_raw)
            msg += f"Take Profit:  {tp_str}\n"
        else:
            msg += f"Take Profit:  <code>{float(tp):.4f}</code>\n"
    else:
        msg += "Take Profit:  <i>not set</i>\n"
    msg += f"Risk/Reward:  <b>{rr_str}</b>\n"
    msg += f"Est. Liq.:    <code>{est_liq_str}</code>\n"
    if margin:
        msg += f"Margin:       <code>${margin:.2f}</code>\n"

    # Account balance / equity after trade
    balance = trade.get("balance") or trade.get("equity") or trade.get("account_balance")
    if balance:
        msg += f"Balance:      <code>${float(balance):.2f} USDT</code>\n"

    # Number of open positions
    open_positions = trade.get("open_positions")
    if open_positions is not None:
        msg += f"Open Pos.:    <b>{open_positions}</b>\n"

    msg += (
        f"Strategy:     {strategy}\n"
        f"Exchange:     {exchange}\n"
        f"Order ID:     <code>{order_id}</code>\n"
        f"Time:         {now}"
    )
    return msg


def _fmt_trade_closed(trade: dict) -> str:
    """Format a generic trade-closed / trade-result message (module-level helper)."""
    symbol = trade.get("symbol", "N/A")
    side = (trade.get("direction") or trade.get("side") or "N/A").upper()
    side_emoji = "🟢 LONG" if "LONG" in side or "BUY" in side else "🔴 SHORT"
    size = trade.get("size") or trade.get("amount") or 0.0
    entry = trade.get("price") or trade.get("entry_price") or 0.0
    exit_p = trade.get("exit_price") or trade.get("filled_price") or 0.0
    pnl = trade.get("pnl") or 0.0
    pnl_pct = trade.get("pnl_pct") or 0.0
    leverage = trade.get("leverage", 1)
    strategy = trade.get("strategy") or "—"
    exit_reason = (trade.get("exit_reason") or "closed").replace("_", " ").title()
    fees = trade.get("fees") or trade.get("fee") or 0.0
    net_pnl = pnl - fees
    pnl_emoji = "✅" if pnl >= 0 else "❌"

    msg = (
        f"{pnl_emoji} <b>TRADE CLOSED — {symbol}</b>\n"
        f"{'━' * 28}\n"
        f"Direction:   {side_emoji}\n"
        f"Size:        <code>{size:.4f}</code>  {leverage}×\n"
    )
    if entry:
        msg += f"Entry Price: <code>{entry:.4f}</code>\n"
    if exit_p:
        msg += f"Exit Price:  <code>{exit_p:.4f}</code>\n"
    msg += (
        f"{'━' * 28}\n"
        f"Gross PnL:   <code>{pnl:+.4f} USDT</code>\n"
    )
    if fees:
        msg += f"Fees:        <code>-{fees:.4f} USDT</code>\n"
    msg += f"Net PnL:     <b><code>{net_pnl:+.4f} USDT</code></b>"
    if pnl_pct:
        msg += f"  (<code>{pnl_pct:+.2f}%</code>)"
    msg += f"\n{'━' * 28}\n"
    msg += f"Reason:      {exit_reason}\n"
    msg += f"Strategy:    {strategy}"
    return msg


def _fmt_sl_hit(trade: dict, mode: str = "paper") -> str:
    """Format a detailed stop-loss hit message (module-level helper)."""
    tag = _fmt_mode_tag(mode)
    symbol = trade.get("symbol", "N/A")
    side = (trade.get("direction") or trade.get("side") or "N/A").upper()
    side_emoji = "🟢 LONG" if "LONG" in side or "BUY" in side else "🔴 SHORT"
    entry = trade.get("entry_price") or trade.get("price") or 0.0
    exit_p = trade.get("exit_price") or trade.get("filled_price") or 0.0
    size = trade.get("size") or trade.get("amount") or 0.0
    pnl = trade.get("pnl") or 0.0
    pnl_pct = trade.get("pnl_pct") or 0.0
    leverage = trade.get("leverage", 1)
    strategy = trade.get("strategy") or "—"
    fees = trade.get("fees") or trade.get("fee") or 0.0
    net_pnl = pnl - fees
    duration = trade.get("duration_seconds") or trade.get("duration") or 0
    duration_str = _format_duration(duration)

    return (
        f"{tag} 🛑 <b>STOP LOSS HIT — {symbol}</b>\n"
        f"{'━' * 28}\n"
        f"Direction:   {side_emoji}  {leverage}×\n"
        f"Size:        <code>{size:.4f}</code>\n"
        f"Entry:       <code>{entry:.4f}</code>\n"
        f"Exit (SL):   <code>{exit_p:.4f}</code>\n"
        f"{'━' * 28}\n"
        f"Gross PnL:   <code>{pnl:+.4f} USDT</code>\n"
        f"Fees:        <code>-{fees:.4f} USDT</code>\n"
        f"Net Loss:    <b><code>{net_pnl:+.4f} USDT</code></b>"
        f"  (<code>{pnl_pct:+.2f}%</code>)\n"
        f"{'━' * 28}\n"
        f"Duration:    {duration_str}\n"
        f"Strategy:    {strategy}"
    )


def _fmt_tp_hit(trade: dict, mode: str = "paper") -> str:
    """Format a detailed take-profit hit message (module-level helper)."""
    tag = _fmt_mode_tag(mode)
    symbol = trade.get("symbol", "N/A")
    side = (trade.get("direction") or trade.get("side") or "N/A").upper()
    side_emoji = "🟢 LONG" if "LONG" in side or "BUY" in side else "🔴 SHORT"
    entry = trade.get("entry_price") or trade.get("price") or 0.0
    exit_p = trade.get("exit_price") or trade.get("filled_price") or 0.0
    size = trade.get("size") or trade.get("amount") or 0.0
    pnl = trade.get("pnl") or 0.0
    pnl_pct = trade.get("pnl_pct") or 0.0
    leverage = trade.get("leverage", 1)
    strategy = trade.get("strategy") or "—"
    fees = trade.get("fees") or trade.get("fee") or 0.0
    net_pnl = pnl - fees
    rr = trade.get("risk_reward") or 0.0
    duration = trade.get("duration_seconds") or trade.get("duration") or 0
    duration_str = _format_duration(duration)
    tp_level = trade.get("tp_level") or ""

    msg = f"{tag} 🎯 <b>TAKE PROFIT HIT"
    if tp_level:
        msg += f" (TP{tp_level})"
    msg += (
        f" — {symbol}</b>\n"
        f"{'━' * 28}\n"
        f"Direction:   {side_emoji}  {leverage}×\n"
        f"Size:        <code>{size:.4f}</code>\n"
        f"Entry:       <code>{entry:.4f}</code>\n"
        f"Exit (TP):   <code>{exit_p:.4f}</code>\n"
        f"{'━' * 28}\n"
        f"Gross PnL:   <code>{pnl:+.4f} USDT</code>\n"
        f"Fees:        <code>-{fees:.4f} USDT</code>\n"
        f"Net Profit:  <b><code>{net_pnl:+.4f} USDT</code></b>"
        f"  (<code>{pnl_pct:+.2f}%</code>)\n"
        f"{'━' * 28}\n"
    )
    if rr:
        msg += f"R:R Ratio:   {rr:.2f}\n"
    msg += (
        f"Duration:    {duration_str}\n"
        f"Strategy:    {strategy}"
    )
    return msg


def _fmt_liquidation(position: dict, mode: str = "paper") -> str:
    """Format a detailed liquidation message (module-level helper)."""
    tag = _fmt_mode_tag(mode)
    symbol = position.get("symbol", "N/A")
    side = (position.get("side") or position.get("direction") or "N/A").upper()
    side_emoji = "🟢 LONG" if "LONG" in side or "BUY" in side else "🔴 SHORT"
    size = position.get("size") or position.get("amount") or 0.0
    entry = position.get("entry_price") or position.get("price") or 0.0
    liq_price = position.get("liquidation_price") or position.get("liq_price") or 0.0
    margin = position.get("margin_used") or position.get("margin") or 0.0
    leverage = position.get("leverage", 1)
    strategy = position.get("strategy") or "—"
    pnl = position.get("unrealized_pnl") or position.get("pnl") or -margin
    dist_pct = position.get("dist") or position.get("dist_pct") or 0.0

    return (
        f"{tag} 💥 <b>LIQUIDATION — {symbol}</b>\n"
        f"{'━' * 28}\n"
        f"⚠️ Position has been LIQUIDATED!\n"
        f"{'━' * 28}\n"
        f"Direction:    {side_emoji}\n"
        f"Size:         <code>{size:.4f}</code>\n"
        f"Leverage:     <b>{leverage}×</b>\n"
        f"Entry:        <code>{entry:.4f}</code>\n"
        f"Liq. Price:   <code>{liq_price:.4f}</code>\n"
        f"Dist at Liq:  <code>{dist_pct:.2f}%</code>\n"
        f"{'━' * 28}\n"
        f"Margin Lost:  <b><code>-${margin:.2f} USDT</code></b>\n"
        f"Final PnL:    <code>{pnl:+.4f} USDT</code>\n"
        f"{'━' * 28}\n"
        f"Strategy:     {strategy}\n"
        f"Action:       Review leverage & risk settings immediately!"
    )


def _fmt_manual_trade(trade: dict, mode: str = "paper") -> str:
    """Format a manual-trade warning message (module-level helper)."""
    tag = _fmt_mode_tag(mode)
    symbol = trade.get("symbol", "N/A")
    side = (trade.get("direction") or trade.get("side") or "N/A").upper()
    price = trade.get("price") or trade.get("entry_price") or 0.0
    size = trade.get("size") or trade.get("amount") or 0.0
    leverage = trade.get("leverage", 1)
    sl = trade.get("stop_loss")
    tp = trade.get("take_profit")
    source = trade.get("source") or "external"
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    msg = (
        f"{tag} ⚡ <b>MANUAL TRADE DETECTED — {symbol}</b>\n"
        f"{'━' * 28}\n"
        f"⚠️ A trade was opened outside the bot.\n"
        f"{'━' * 28}\n"
        f"Direction: {side}\n"
        f"Price:     <code>{price:.4f}</code>\n"
        f"Size:      <code>{size:.4f}</code>\n"
        f"Leverage:  {leverage}×\n"
    )
    if sl:
        msg += f"Stop Loss: <code>{float(sl):.4f}</code>\n"
    if tp:
        tp_val = tp[0] if isinstance(tp, (list, tuple)) else tp
        msg += f"Take Profit: <code>{float(tp_val):.4f}</code>\n"
    msg += (
        f"Source:    {source}\n"
        f"Time:      {now}\n"
        f"{'━' * 28}\n"
        f"⚡ Bot is now tracking this position."
    )
    return msg


# ---------------------------------------------------------------------------
# TelegramAlerter — specialised Telegram-only alerter with typed methods
# ---------------------------------------------------------------------------

#: PnL milestones (in percent) that trigger an alert, ordered from smallest.
_PNL_MILESTONES: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)


class TelegramAlerter:
    """Lightweight async Telegram alerter for order and PnL events.

    Uses ``aiohttp`` to POST to the Telegram Bot API and formats all messages
    as HTML for readable Telegram notifications.

    Args:
        settings: Application settings.  When *None*, the global singleton is
            used via :meth:`~config.settings.Settings.get_settings`.
    """

    #: Timeout in seconds for each Telegram API request.
    _SEND_TIMEOUT: float = 10.0

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings.get_settings()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def send_order_alert(
        self,
        order: dict,
        margin_remaining: float = 0.0,
        mode: str = "paper",
    ) -> bool:
        """Send a detailed order-execution alert to Telegram.

        Args:
            order: Trade result / order dict with keys such as ``symbol``,
                ``direction`` / ``side``, ``leverage``, ``price`` /
                ``entry_price``, ``amount`` / ``size``, ``stop_loss``,
                ``take_profit``, ``strategy``.
            margin_remaining: Free futures margin in USDT after the order.
            mode: Trading mode string — ``"live"``, ``"testnet"``, or
                ``"paper"`` / ``"manual"``.

        Returns:
            ``True`` if the message was sent successfully.
        """
        msg = _fmt_trade_open(order, mode)
        msg += f"\nFree Margin: <code>${margin_remaining:.2f}</code>"
        return await self._send(msg)

    async def send_pnl_alert(self, position: dict, milestone_hit: float) -> bool:
        """Send a PnL milestone alert to Telegram.

        Args:
            position: Position dict with keys ``symbol``, ``pnl_pct``,
                ``unrealized_pnl`` (USDT value), ``entry_price``,
                ``current_price``, ``side``, ``leverage``, ``strategy``.
            milestone_hit: The milestone percentage that was crossed
                (positive for profit, negative for loss).

        Returns:
            ``True`` if the message was sent successfully.
        """
        symbol = position.get("symbol", "N/A")
        pnl_pct = position.get("pnl_pct", 0.0) or 0.0
        pnl_usdt = float(position.get("unrealized_pnl") or position.get("pnl") or 0.0)
        entry = position.get("entry_price") or position.get("price") or 0.0
        current = position.get("current_price") or position.get("mark_price") or 0.0
        side = (position.get("side") or position.get("direction") or "").upper()
        leverage = position.get("leverage", 1)
        strategy = position.get("strategy") or "—"

        if milestone_hit > 0:
            emoji = "🟢"
            label = f"+{milestone_hit:.0f}% Profit Milestone"
        else:
            emoji = "🔴"
            label = f"{milestone_hit:.0f}% Loss Warning"

        msg = (
            f"{emoji} <b>{label} — {symbol}</b>\n"
            f"{'━' * 28}\n"
            f"Side:          {side}  {leverage}×\n"
        )
        if entry:
            msg += f"Entry Price:   <code>{entry:.4f}</code>\n"
        if current:
            msg += f"Current Price: <code>{current:.4f}</code>\n"
        msg += (
            f"Unrealized PnL: <b><code>{pnl_usdt:+.4f} USDT</code></b>"
            f"  (<code>{pnl_pct:+.2f}%</code>)\n"
            f"Strategy:      {strategy}"
        )
        return await self._send(msg)

    async def send_trade_closed(
        self,
        symbol: str,
        realized_pnl: float,
        reason: str = "closed",
        trade_data: Optional[dict] = None,
    ) -> bool:
        """Send a detailed trade-closed receipt with total realised PnL.

        Args:
            symbol: Trading pair symbol.
            realized_pnl: Total realised PnL in USDT for this position.
            reason: Human-readable reason / exit type (e.g. ``"stop_loss"``,
                ``"take_profit"``, ``"manual"``).
            trade_data: Optional full trade dict for additional details.

        Returns:
            ``True`` if the message was sent successfully.
        """
        emoji = "💰" if realized_pnl >= 0 else "📉"
        if trade_data is not None:
            td = dict(trade_data)
            td["pnl"] = td.get("pnl") or realized_pnl
            td["symbol"] = td.get("symbol") or symbol
            mode = td.pop("mode", "paper")
            if reason in ("stop_loss", "sl"):
                msg = _fmt_sl_hit(td, mode)
            elif reason in ("take_profit", "tp", "trailing_take_profit"):
                msg = _fmt_tp_hit(td, mode)
            else:
                msg = _fmt_trade_closed(td)
                msg = f"{emoji} {msg}"
        else:
            reason_clean = reason.replace("_", " ").title()
            msg = (
                f"{emoji} <b>Trade Closed: {symbol}</b>\n"
                f"{'━' * 28}\n"
                f"Reason:       {reason_clean}\n"
                f"Realised PnL: <b><code>{realized_pnl:+.4f} USDT</code></b>"
            )
        return await self._send(msg)

    async def send_sl_alert(
        self,
        position: dict,
        mode: str = "paper",
    ) -> bool:
        """Send a detailed stop-loss hit alert to Telegram.

        Args:
            position: Position dict with full trade data.
            mode: Trading mode string.

        Returns:
            ``True`` if the message was sent successfully.
        """
        return await self._send(_fmt_sl_hit(position, mode))

    async def send_tp_alert(
        self,
        position: dict,
        mode: str = "paper",
    ) -> bool:
        """Send a detailed take-profit hit alert to Telegram.

        Args:
            position: Position dict with full trade data.
            mode: Trading mode string.

        Returns:
            ``True`` if the message was sent successfully.
        """
        return await self._send(_fmt_tp_hit(position, mode))

    async def send_liquidation_alert(
        self,
        position: dict,
        mode: str = "paper",
    ) -> bool:
        """Send a detailed liquidation alert to Telegram.

        Args:
            position: Position dict with symbol, side, size, entry_price,
                liquidation_price, margin_used, leverage, unrealized_pnl,
                strategy, dist_pct.
            mode: Trading mode string.

        Returns:
            ``True`` if the message was sent successfully.
        """
        return await self._send(_fmt_liquidation(position, mode))

    async def send_manual_trade_alert(
        self,
        order: dict,
        mode: str = "paper",
    ) -> bool:
        """Send a manual-trade-detected warning to Telegram.

        Args:
            order: Trade dict for the manually-opened position.
            mode: Trading mode string.

        Returns:
            ``True`` if the message was sent successfully.
        """
        return await self._send(_fmt_manual_trade(order, mode))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send(self, message: str) -> bool:
        """POST *message* to the Telegram Bot API using aiohttp.

        Falls back to ``httpx`` if ``aiohttp`` is not available so the module
        works in both dependency configurations.
        """
        token = self._settings.alert_telegram_bot_token or self._settings.telegram_bot_token
        chat_id = self._settings.alert_telegram_chat_id or self._settings.telegram_chat_id
        if not token or not chat_id:
            logger.debug("TelegramAlerter: credentials not configured — skipping")
            return False

        token = _strip_bot_prefix(token)
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

        try:
            import aiohttp  # type: ignore[import]
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=self._SEND_TIMEOUT)) as resp:
                    if resp.status == 200:
                        logger.debug("TelegramAlerter: message sent")
                        return True
                    text = await resp.text()
                    if resp.status == 400 and "chat not found" in text.lower():
                        logger.warning(
                            "TelegramAlerter: chat not found (chat_id={!r}). "
                            "Fix: send /start to your bot, then get the correct chat_id via "
                            "https://api.telegram.org/bot.../getUpdates and set "
                            "ALERT_TELEGRAM_CHAT_ID in your .env file.",
                            chat_id,
                        )
                        return False
                    logger.warning("TelegramAlerter: send failed status={} body={}", resp.status, text)
                    return False
        except ImportError:
            # aiohttp not available — fall back to httpx (already a project dependency)
            try:
                async with httpx.AsyncClient(timeout=self._SEND_TIMEOUT) as client:
                    resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.debug("TelegramAlerter: message sent (httpx fallback)")
                    return True
                if resp.status_code == 400 and "chat not found" in resp.text.lower():
                    logger.warning(
                        "TelegramAlerter: chat not found (chat_id={!r}). "
                        "Fix: send /start to your bot, then get the correct chat_id via "
                        "https://api.telegram.org/bot.../getUpdates and set "
                        "ALERT_TELEGRAM_CHAT_ID in your .env file.",
                        chat_id,
                    )
                    return False
                logger.warning(
                    "TelegramAlerter: send failed status={} body={}",
                    resp.status_code,
                    resp.text,
                )
                return False
            except Exception as exc:
                logger.error("TelegramAlerter: send error: {}", exc)
                return False
        except Exception as exc:
            logger.error("TelegramAlerter: send error: {}", exc)
            return False
