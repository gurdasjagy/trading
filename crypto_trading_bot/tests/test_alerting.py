"""Tests for the updated AlertManager (httpx-based, rate-limited, new API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
from monitoring.alerting import _RATE_LIMIT_MAX, AlertManager, AlertType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**kwargs) -> Settings:
    return Settings(
        TRADING_MODE="paper",
        SECRET_KEY="test-secret-key-32-chars-padding!",
        **kwargs,
    )


def _make_alert_manager(**kwargs) -> AlertManager:
    return AlertManager(settings=_make_settings(**kwargs))


# ---------------------------------------------------------------------------
# Tests — send_alert (plain message)
# ---------------------------------------------------------------------------


class TestSendAlert:
    """send_alert(message, level) dispatches to configured channels."""

    @pytest.mark.asyncio
    async def test_no_credentials_warns_and_returns_false(self, caplog):
        """When no Telegram/Discord credentials are set, warn but don't crash."""
        am = _make_alert_manager()
        # Ensure channels are enabled but no credentials
        am._settings.monitoring.enable_telegram_alerts = True
        am._settings.monitoring.enable_discord_alerts = True
        am._settings.monitoring.enable_email_alerts = False

        result = await am.send_alert("test message")

        assert result is False

    @pytest.mark.asyncio
    async def test_telegram_called_when_credentials_set(self):
        """send_telegram is called when token and chat_id are configured."""
        am = _make_alert_manager(
            TELEGRAM_BOT_TOKEN="test-token",
            TELEGRAM_CHAT_ID="123456",
        )
        am._settings.monitoring.enable_telegram_alerts = True
        am._settings.monitoring.enable_discord_alerts = False
        am._settings.monitoring.enable_email_alerts = False

        with patch.object(am, "send_telegram", new=AsyncMock(return_value=True)) as mock_tg:
            result = await am.send_alert("hello", level="info")

        mock_tg.assert_called_once_with("hello")
        assert result is True

    @pytest.mark.asyncio
    async def test_discord_embed_called_when_webhook_set(self):
        """_send_discord_embed is called when the webhook URL is configured."""
        am = _make_alert_manager(DISCORD_WEBHOOK_URL="https://discord.example/hook")
        am._settings.monitoring.enable_telegram_alerts = False
        am._settings.monitoring.enable_discord_alerts = True
        am._settings.monitoring.enable_email_alerts = False

        with patch.object(am, "_send_discord_embed", new=AsyncMock(return_value=True)) as mock_dc:
            result = await am.send_alert("embed msg", level="warning")

        mock_dc.assert_called_once_with("embed msg", "warning")
        assert result is True

    @pytest.mark.asyncio
    async def test_both_channels_attempted(self):
        """Both Telegram and Discord are tried when both are configured."""
        am = _make_alert_manager(
            TELEGRAM_BOT_TOKEN="tok",
            TELEGRAM_CHAT_ID="cid",
            DISCORD_WEBHOOK_URL="https://discord.example/hook",
        )
        am._settings.monitoring.enable_telegram_alerts = True
        am._settings.monitoring.enable_discord_alerts = True
        am._settings.monitoring.enable_email_alerts = False

        with (
            patch.object(am, "send_telegram", new=AsyncMock(return_value=False)),
            patch.object(am, "_send_discord_embed", new=AsyncMock(return_value=True)),
        ):
            result = await am.send_alert("msg")

        assert result is True


# ---------------------------------------------------------------------------
# Tests — send_trade_alert
# ---------------------------------------------------------------------------


class TestSendTradeAlert:
    """send_trade_alert formats trade info and delegates to send_alert."""

    @pytest.mark.asyncio
    async def test_positive_pnl_uses_success_level(self):
        am = _make_alert_manager()
        am._settings.monitoring.enable_telegram_alerts = False
        am._settings.monitoring.enable_discord_alerts = False
        am._settings.monitoring.enable_email_alerts = False

        calls = []

        async def _capture(msg, level="info"):
            calls.append((msg, level))
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_trade_alert(
            {
                "symbol": "BTC/USDT",
                "direction": "long",
                "size": 0.01,
                "price": 50_000.0,
                "pnl": 50.0,
            }
        )

        assert calls, "send_alert should have been called"
        _, level = calls[0]
        assert level == "success"

    @pytest.mark.asyncio
    async def test_negative_pnl_uses_warning_level(self):
        am = _make_alert_manager()
        calls = []

        async def _capture(msg, level="info"):
            calls.append((msg, level))
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_trade_alert({"symbol": "ETH/USDT", "direction": "short", "pnl": -25.0})

        _, level = calls[0]
        assert level == "warning"

    @pytest.mark.asyncio
    async def test_message_contains_symbol(self):
        am = _make_alert_manager()
        received = []

        async def _capture(msg, level="info"):
            received.append(msg)
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_trade_alert({"symbol": "SOL/USDT", "direction": "long"})

        assert received and "SOL/USDT" in received[0]


# ---------------------------------------------------------------------------
# Tests — send_daily_report
# ---------------------------------------------------------------------------


class TestSendDailyReport:
    """send_daily_report formats and sends an end-of-day summary."""

    @pytest.mark.asyncio
    async def test_positive_pnl_uses_success_level(self):
        am = _make_alert_manager()
        calls = []

        async def _capture(msg, level="info"):
            calls.append((msg, level))
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_daily_report(
            {"date": "2024-01-01", "total_pnl": 200.0, "win_rate": 60.0, "trade_count": 5}
        )

        _, level = calls[0]
        assert level == "success"

    @pytest.mark.asyncio
    async def test_negative_pnl_uses_warning_level(self):
        am = _make_alert_manager()
        calls = []

        async def _capture(msg, level="info"):
            calls.append((msg, level))
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_daily_report({"date": "2024-01-01", "total_pnl": -50.0})

        _, level = calls[0]
        assert level == "warning"

    @pytest.mark.asyncio
    async def test_message_contains_date(self):
        am = _make_alert_manager()
        received = []

        async def _capture(msg, level="info"):
            received.append(msg)
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_daily_report({"date": "2024-03-15", "total_pnl": 0.0})

        assert received and "2024-03-15" in received[0]


# ---------------------------------------------------------------------------
# Tests — Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    """AlertManager enforces a 30-message-per-minute cap."""

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess_messages(self):
        """After _RATE_LIMIT_MAX messages the next one is dropped."""
        am = _make_alert_manager()
        am._settings.monitoring.enable_telegram_alerts = False
        am._settings.monitoring.enable_discord_alerts = False
        am._settings.monitoring.enable_email_alerts = False

        # Exhaust the rate limit bucket directly
        import time as _time

        now = _time.monotonic()
        for _ in range(_RATE_LIMIT_MAX):
            am._sent_times.append(now)

        result = await am.send_alert("should be blocked")
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_allows_after_window_expires(self):
        """Old timestamps outside the 60-second window are evicted."""
        am = _make_alert_manager()
        am._settings.monitoring.enable_telegram_alerts = False
        am._settings.monitoring.enable_discord_alerts = False
        am._settings.monitoring.enable_email_alerts = False

        import time as _time

        # Fill bucket with timestamps older than the window
        old_ts = _time.monotonic() - 70.0
        for _ in range(_RATE_LIMIT_MAX):
            am._sent_times.append(old_ts)

        # Should be allowed — old timestamps evicted
        allowed = await am._check_rate_limit()
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests — Telegram implementation (httpx)
# ---------------------------------------------------------------------------


class TestSendTelegram:
    """send_telegram calls the Telegram Bot API via httpx."""

    @pytest.mark.asyncio
    async def test_success_200(self):
        am = _make_alert_manager(TELEGRAM_BOT_TOKEN="token", TELEGRAM_CHAT_ID="chat")

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("monitoring.alerting.httpx.AsyncClient", return_value=mock_client):
            result = await am.send_telegram("hello")

        assert result is True
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "token/sendMessage" in call_kwargs.args[0]

    @pytest.mark.asyncio
    async def test_non_200_returns_false(self):
        am = _make_alert_manager(TELEGRAM_BOT_TOKEN="token", TELEGRAM_CHAT_ID="chat")

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("monitoring.alerting.httpx.AsyncClient", return_value=mock_client):
            result = await am.send_telegram("hello")

        assert result is False

    @pytest.mark.asyncio
    async def test_missing_credentials_returns_false(self):
        am = _make_alert_manager()  # no token / chat_id
        result = await am.send_telegram("msg")
        assert result is False

    @pytest.mark.asyncio
    async def test_network_exception_returns_false(self):
        am = _make_alert_manager(TELEGRAM_BOT_TOKEN="tok", TELEGRAM_CHAT_ID="cid")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("network error"))

        with patch("monitoring.alerting.httpx.AsyncClient", return_value=mock_client):
            result = await am.send_telegram("msg")

        assert result is False


# ---------------------------------------------------------------------------
# Tests — Discord implementation (httpx embed)
# ---------------------------------------------------------------------------


class TestSendDiscordEmbed:
    """_send_discord_embed sends an embed payload via httpx."""

    @pytest.mark.asyncio
    async def test_success_204(self):
        am = _make_alert_manager(DISCORD_WEBHOOK_URL="https://discord.example/hook")

        mock_resp = MagicMock()
        mock_resp.status_code = 204

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("monitoring.alerting.httpx.AsyncClient", return_value=mock_client):
            result = await am._send_discord_embed("hello", "info")

        assert result is True
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or {}
        assert "embeds" in payload
        assert payload["embeds"][0]["description"] == "hello"

    @pytest.mark.asyncio
    async def test_embed_uses_level_color(self):
        from monitoring.alerting import _LEVEL_COLORS

        am = _make_alert_manager(DISCORD_WEBHOOK_URL="https://discord.example/hook")

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("monitoring.alerting.httpx.AsyncClient", return_value=mock_client):
            await am._send_discord_embed("crit msg", "critical")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or {}
        assert payload["embeds"][0]["color"] == _LEVEL_COLORS["critical"]

    @pytest.mark.asyncio
    async def test_missing_webhook_returns_false(self):
        am = _make_alert_manager()
        result = await am._send_discord_embed("msg")
        assert result is False

    @pytest.mark.asyncio
    async def test_network_exception_returns_false(self):
        am = _make_alert_manager(DISCORD_WEBHOOK_URL="https://discord.example/hook")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))

        with patch("monitoring.alerting.httpx.AsyncClient", return_value=mock_client):
            result = await am._send_discord_embed("msg")

        assert result is False


# ---------------------------------------------------------------------------
# Tests — send_typed_alert (backward-compat wrapper)
# ---------------------------------------------------------------------------


class TestSendTypedAlert:
    """send_typed_alert formats via AlertType and delegates to send_alert."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_uses_critical_level(self):
        am = _make_alert_manager()
        calls = []

        async def _capture(msg, level="info"):
            calls.append((msg, level))
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_typed_alert(AlertType.CIRCUIT_BREAKER, {"reason": "test"})

        _, level = calls[0]
        assert level == "critical"

    @pytest.mark.asyncio
    async def test_trade_opened_uses_info_level(self):
        am = _make_alert_manager()
        calls = []

        async def _capture(msg, level="info"):
            calls.append((msg, level))
            return False

        am.send_alert = _capture  # type: ignore[method-assign]

        await am.send_typed_alert(
            AlertType.TRADE_OPENED, {"symbol": "BTC/USDT", "direction": "long"}
        )

        _, level = calls[0]
        assert level == "info"


# ---------------------------------------------------------------------------
# Tests — Engine wiring: trade alerts and circuit breaker
# ---------------------------------------------------------------------------


class TestEngineAlertWiring:
    """Engine initializes AlertManager and wires it to trade execution."""

    @pytest.mark.asyncio
    async def test_alert_manager_initialized_on_start(self):
        from core.engine import TradingEngine
        from core.state_manager import StateManager

        StateManager._instance = None
        settings = _make_settings()
        engine = TradingEngine(settings=settings)
        await engine._initialize_subsystems()

        assert engine.alert_manager is not None, "AlertManager must be initialized"
        assert isinstance(engine.alert_manager, AlertManager)

        await engine.stop()
        StateManager._instance = None

    @pytest.mark.asyncio
    async def test_pre_wired_alert_manager_not_overwritten(self):
        """alert_manager set before _initialize_subsystems is not replaced."""
        from core.engine import TradingEngine
        from core.state_manager import StateManager

        StateManager._instance = None
        settings = _make_settings()
        engine = TradingEngine(settings=settings)

        custom_am = AlertManager(settings=settings)
        engine.alert_manager = custom_am

        await engine._initialize_subsystems()

        assert engine.alert_manager is custom_am, "Pre-wired AlertManager must not be replaced"

        await engine.stop()
        StateManager._instance = None
