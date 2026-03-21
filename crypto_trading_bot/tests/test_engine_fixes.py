"""Tests for critical engine bug fixes (Bugs 1-5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import Settings
from core.engine import TradingEngine
from core.state_manager import StateManager
from risk.risk_manager import RiskApproval, RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**kwargs) -> Settings:
    return Settings(
        TRADING_MODE="paper",
        SECRET_KEY="test-secret-key-32-chars-padding!",
        **kwargs,
    )


def _make_engine(settings=None) -> TradingEngine:
    return TradingEngine(settings=settings or _make_settings())


# ---------------------------------------------------------------------------
# Bug 1 — Circuit breaker must NOT trigger at 0% daily loss
# ---------------------------------------------------------------------------


class TestCircuitBreakerZeroLoss:
    """Bug 1: 0% daily loss must never activate the circuit breaker."""

    @pytest.mark.asyncio
    async def test_zero_daily_loss_does_not_trigger(self):
        """Circuit breaker stays inactive when daily P&L is 0%."""
        # Reset singleton so state is fresh for this test
        StateManager._instance = None
        engine = _make_engine()
        await engine._initialize_subsystems()  # sets up state_manager
        # Skip per-symbol exchange calls in this unit test
        engine.strategy_manager = None

        # Confirm the state starts at 0% pnl
        state = await engine.state_manager.get_state()
        assert state.daily_pnl_pct == 0.0

        # Run one trading cycle — no signals, no per-symbol loop
        await engine._trading_cycle(1)

        state = await engine.state_manager.get_state()
        assert (
            state.circuit_breaker_active is False
        ), "Circuit breaker must not trigger when daily loss is 0%"
        # cleanup
        await engine.stop()
        StateManager._instance = None

    @pytest.mark.asyncio
    async def test_loss_below_limit_does_not_trigger(self):
        """Circuit breaker stays inactive when loss is below the configured limit."""
        StateManager._instance = None
        engine = _make_engine()
        await engine._initialize_subsystems()
        # Skip per-symbol exchange calls in this unit test
        engine.strategy_manager = None

        # Simulate a small 1% daily loss (limit is 2% in defaults)
        await engine.state_manager.update_state(daily_pnl_pct=-1.0)

        await engine._trading_cycle(1)

        state = await engine.state_manager.get_state()
        assert state.circuit_breaker_active is False
        await engine.stop()
        StateManager._instance = None

    @pytest.mark.asyncio
    async def test_loss_at_limit_triggers(self):
        """Circuit breaker activates when loss equals the configured limit."""
        StateManager._instance = None
        settings = _make_settings()
        engine = _make_engine(settings)
        await engine._initialize_subsystems()
        # Skip per-symbol exchange calls in this unit test
        engine.strategy_manager = None

        max_loss = settings.risk.max_daily_loss_pct  # e.g. 2.0
        await engine.state_manager.update_state(daily_pnl_pct=-max_loss)

        await engine._trading_cycle(1)

        state = await engine.state_manager.get_state()
        assert (
            state.circuit_breaker_active is True
        ), f"Circuit breaker must trigger when loss ({max_loss}%) reaches the limit"
        await engine.stop()
        StateManager._instance = None


# ---------------------------------------------------------------------------
# Bug 2 — Risk approval check uses .approved field, not object truthiness
# ---------------------------------------------------------------------------


class TestRiskApprovalCheck:
    """Bug 2: validate_trade returns RiskApproval (Pydantic model, always truthy);
    the engine must inspect the .approved field."""

    @staticmethod
    def _wire_exchange_and_strategy(engine, signals):
        """Helper: attach mock exchange + strategy_manager that return *signals*."""
        import time
        import pandas as pd

        # Mock exchange returning minimal OHLCV data and positions
        mock_exchange = AsyncMock()
        # Use recent timestamps (last 50 15-minute candles) to avoid stale-data checks
        now_ms = int(time.time() * 1000)
        interval_ms = 15 * 60 * 1000  # 15 minutes in milliseconds
        recent_timestamps = [now_ms - (49 - i) * interval_ms for i in range(50)]
        ohlcv_df = pd.DataFrame(
            {
                "timestamp": recent_timestamps,
                "open": [50_000.0] * 50,
                "high": [51_000.0] * 50,
                "low": [49_000.0] * 50,
                "close": [50_500.0] * 50,
                "volume": [100.0] * 50,
            }
        )
        mock_exchange.get_ohlcv = AsyncMock(return_value=ohlcv_df)
        mock_exchange.get_ticker = AsyncMock(return_value=MagicMock(last=50_500.0))
        mock_exchange.get_positions = AsyncMock(return_value=[])
        mock_exchange.get_balance = AsyncMock(
            return_value=MagicMock(usdt_total=10_000.0, usdt_free=10_000.0)
        )
        engine.exchange = mock_exchange

        # Mock strategy_manager returning the supplied signals for the first
        # symbol, empty list for all others.
        call_count = [0]

        async def _evaluate_all(symbol, market_data, **kwargs):
            call_count[0] += 1
            return signals if call_count[0] == 1 else []

        mock_sm = AsyncMock()
        mock_sm.evaluate_all = AsyncMock(side_effect=_evaluate_all)
        engine.strategy_manager = mock_sm

    @pytest.mark.asyncio
    async def test_rejected_signal_does_not_execute(self):
        """A signal rejected by the risk manager must not reach the executor."""
        StateManager._instance = None
        engine = _make_engine()
        await engine._initialize_subsystems()

        # Wire in a mock risk manager that always rejects
        mock_risk = AsyncMock(spec=RiskManager)
        rejection = RiskApproval(
            approved=False,
            symbol="BTC/USDT",
            direction="long",
            rejection_reason="test rejection",
        )
        mock_risk.validate_trade = AsyncMock(return_value=rejection)
        mock_risk.update_market_state = MagicMock()
        engine.risk_manager = mock_risk

        # Wire in a mock executor that records calls
        mock_executor = AsyncMock()
        engine.trade_executor = mock_executor

        # Inject a signal via exchange + strategy_manager mocks
        self._wire_exchange_and_strategy(
            engine,
            signals=[{"symbol": "BTC/USDT", "direction": "long", "entry_price": 50_000.0}],
        )

        await engine._trading_cycle(1)

        mock_executor.execute_trade.assert_not_called()
        await engine.stop()
        StateManager._instance = None

    @pytest.mark.asyncio
    async def test_approved_signal_reaches_executor(self):
        """A signal approved by the risk manager must be forwarded to the executor."""
        StateManager._instance = None
        engine = _make_engine()
        await engine._initialize_subsystems()

        mock_risk = AsyncMock(spec=RiskManager)
        approval = RiskApproval(
            approved=True,
            symbol="BTC/USDT",
            direction="long",
            position_size=100.0,
            stop_loss=49_000.0,
            take_profit_levels=[51_000.0, 52_000.0, 53_000.0],
            leverage=5,
            risk_reward=2.0,
        )
        mock_risk.validate_trade = AsyncMock(return_value=approval)
        mock_risk.update_market_state = MagicMock()
        engine.risk_manager = mock_risk

        mock_executor = AsyncMock()
        mock_executor.execute_trade = AsyncMock(return_value={"success": True})
        engine.trade_executor = mock_executor

        # Inject a signal via exchange + strategy_manager mocks
        self._wire_exchange_and_strategy(
            engine,
            signals=[{"symbol": "BTC/USDT", "direction": "long", "entry_price": 50_000.0}],
        )

        await engine._trading_cycle(1)

        mock_executor.execute_trade.assert_called_once()
        await engine.stop()
        StateManager._instance = None


# ---------------------------------------------------------------------------
# Bug 4 — Starting equity is set during initialization
# ---------------------------------------------------------------------------


class TestStartingEquityInitialization:
    """Bug 4: DailyPnLManager.set_starting_equity() must be called on startup."""

    @pytest.mark.asyncio
    async def test_starting_equity_set_on_init(self):
        """After engine initialization, DailyPnLManager has a non-default starting equity."""
        StateManager._instance = None
        settings = _make_settings()
        engine = _make_engine(settings)

        # Wire in a real risk manager so we can inspect its internal state
        risk_mgr = RiskManager(settings=settings)
        engine.risk_manager = risk_mgr

        await engine._initialize_subsystems()

        # DailyPnLManager should now have today's starting equity recorded
        from datetime import datetime, timezone

        today = str(datetime.now(tz=timezone.utc).date())
        assert (
            today in risk_mgr._daily_pnl._starting_equity
        ), "Starting equity must be set for today after engine initialization"
        equity = risk_mgr._daily_pnl._starting_equity[today]
        assert equity > 0, "Starting equity must be positive"

        await engine.stop()
        StateManager._instance = None

    @pytest.mark.asyncio
    async def test_paper_mode_uses_default_balance(self):
        """Paper-mode starting equity defaults to 10 000 when no exchange is wired."""
        import os
        from pathlib import Path

        # Clean up any existing paper state to avoid cross-test pollution
        paper_state = Path("data/paper_state.json")
        if paper_state.exists():
            paper_state.unlink()

        StateManager._instance = None
        settings = _make_settings()  # defaults to paper mode
        engine = _make_engine(settings)

        risk_mgr = RiskManager(settings=settings)
        engine.risk_manager = risk_mgr

        await engine._initialize_subsystems()

        from datetime import datetime, timezone

        today = str(datetime.now(tz=timezone.utc).date())
        equity = risk_mgr._daily_pnl._starting_equity.get(today, 0.0)
        assert equity == pytest.approx(10_000.0), "Paper mode starting equity should be 10 000"
        await engine.stop()
        StateManager._instance = None


# ---------------------------------------------------------------------------
# Bug 5 — Portfolio equity is updated in the trading cycle
# ---------------------------------------------------------------------------


class TestPortfolioEquityUpdate:
    """Bug 5: risk_manager.update_market_state() must be called in each cycle."""

    @pytest.mark.asyncio
    async def test_update_market_state_called_per_cycle(self):
        """Each trading cycle calls update_market_state on the risk manager."""
        StateManager._instance = None
        engine = _make_engine()
        await engine._initialize_subsystems()
        # Skip per-symbol exchange calls in this unit test
        engine.strategy_manager = None

        mock_risk = MagicMock(spec=RiskManager)
        mock_risk.update_market_state = MagicMock()
        engine.risk_manager = mock_risk

        await engine._trading_cycle(1)

        mock_risk.update_market_state.assert_called_once()
        call_kwargs = mock_risk.update_market_state.call_args
        # equity may be passed as positional or keyword arg; handle both
        equity_arg = call_kwargs.kwargs.get("equity")
        if equity_arg is None and call_kwargs.args:
            # positional: update_market_state(volatility_regime, market_regime, equity, positions)
            equity_arg = call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
        assert (
            equity_arg is not None and equity_arg > 0
        ), "Equity passed to update_market_state must be positive"

        await engine.stop()
        StateManager._instance = None

    @pytest.mark.asyncio
    async def test_portfolio_equity_not_zero_after_cycle(self):
        """After one cycle, RiskManager._portfolio_equity is updated to the starting balance."""
        StateManager._instance = None
        settings = _make_settings()
        engine = _make_engine(settings)
        risk_mgr = RiskManager(settings=settings)
        engine.risk_manager = risk_mgr

        await engine._initialize_subsystems()
        # Skip per-symbol exchange calls in this unit test
        engine.strategy_manager = None

        # Before first cycle, portfolio equity starts at 0 (update_market_state
        # has not been called yet — only DailyPnLManager has the equity set)
        assert risk_mgr._portfolio_equity == pytest.approx(0.0)

        # After a cycle, update_market_state is called and equity is populated
        await engine._trading_cycle(1)
        assert (
            risk_mgr._portfolio_equity > 0
        ), "Portfolio equity must be updated after the first trading cycle"

        await engine.stop()
        StateManager._instance = None
