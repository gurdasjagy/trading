"""Tests for risk management components."""

from __future__ import annotations

import pytest

from risk.circuit_breaker import CircuitBreaker
from risk.daily_pnl_manager import DailyPnLManager
from risk.drawdown_protector import DrawdownProtector
from risk.leverage_optimizer import LeverageOptimizer
from risk.position_sizer import PositionSizer
from risk.stop_loss_engine import StopLossEngine
from risk.take_profit_engine import TakeProfitEngine

# ── PositionSizer — Kelly ──────────────────────────────────────────────────


class TestPositionSizerKelly:
    def test_kelly_positive_edge(self):
        """Kelly criterion returns a positive size when there is positive edge."""
        sizer = PositionSizer()
        size = sizer.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0.02, capital=10_000)
        assert size > 0, "Expected positive position size for positive edge"

    def test_kelly_capped_at_25_pct(self):
        """Half-Kelly is capped at 25% of capital."""
        sizer = PositionSizer()
        size = sizer.kelly_size(win_rate=0.9, avg_win=0.5, avg_loss=0.01, capital=10_000)
        assert size <= 10_000 * 0.25

    def test_kelly_zero_capital(self):
        """Kelly returns 0 when capital is zero."""
        sizer = PositionSizer()
        assert sizer.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0.02, capital=0) == 0.0

    def test_kelly_zero_avg_loss(self):
        """Kelly returns 0 when avg_loss is zero (division guard)."""
        sizer = PositionSizer()
        assert sizer.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0, capital=10_000) == 0.0

    def test_kelly_negative_edge(self):
        """Kelly returns 0 (not negative) when edge is negative."""
        sizer = PositionSizer()
        size = sizer.kelly_size(win_rate=0.3, avg_win=0.01, avg_loss=0.05, capital=10_000)
        assert size >= 0.0


# ── PositionSizer — Fixed Fraction ─────────────────────────────────────────


class TestPositionSizerFixedFraction:
    def test_fixed_fraction_basic(self):
        """Fixed fraction: 10% of 10 000 = 1 000."""
        sizer = PositionSizer()
        size = sizer.fixed_fraction_size(capital=10_000, risk_pct=0.10)
        assert size == pytest.approx(1_000.0)

    def test_fixed_fraction_capped_at_100pct(self):
        """Risk percentage is capped at 100%."""
        sizer = PositionSizer()
        size = sizer.fixed_fraction_size(capital=10_000, risk_pct=2.0)
        assert size <= 10_000.0

    def test_fixed_fraction_zero_capital(self):
        """Returns 0 when capital is zero."""
        sizer = PositionSizer()
        assert sizer.fixed_fraction_size(capital=0, risk_pct=0.1) == 0.0


# ── StopLossEngine — ATR ──────────────────────────────────────────────────


class TestStopLossEngineATR:
    def test_long_stop_below_entry(self):
        """For a long position, ATR stop is below entry."""
        engine = StopLossEngine()
        stop = engine.calculate_initial_stop(
            entry=50_000, direction="long", atr=500, multiplier=2.0
        )
        assert stop < 50_000

    def test_short_stop_above_entry(self):
        """For a short position, ATR stop is above entry."""
        engine = StopLossEngine()
        stop = engine.calculate_initial_stop(
            entry=50_000, direction="short", atr=500, multiplier=2.0
        )
        assert stop > 50_000

    def test_atr_multiplier_applied(self):
        """Stop distance equals atr × multiplier."""
        engine = StopLossEngine()
        atr, mult = 200.0, 3.0
        stop = engine.calculate_initial_stop(
            entry=10_000, direction="long", atr=atr, multiplier=mult
        )
        assert stop == pytest.approx(10_000 - atr * mult)

    def test_zero_atr_fallback(self):
        """Zero ATR triggers fallback (2% below/above entry)."""
        engine = StopLossEngine()
        stop = engine.calculate_initial_stop(entry=1_000, direction="long", atr=0)
        assert stop == pytest.approx(1_000 * 0.98)

    def test_volatility_adjust_high(self):
        """High volatility widens the stop."""
        engine = StopLossEngine()
        base_stop = 48_000.0
        adj = engine.adjust_for_volatility(base_stop, "high")
        assert adj > base_stop

    def test_volatility_adjust_low(self):
        """Low volatility tightens the stop."""
        engine = StopLossEngine()
        base_stop = 48_000.0
        adj = engine.adjust_for_volatility(base_stop, "low")
        assert adj < base_stop


# ── TakeProfitEngine ──────────────────────────────────────────────────────


class TestTakeProfitLevels:
    def test_returns_three_levels(self):
        """Take-profit engine should return three TP levels."""
        engine = TakeProfitEngine()
        levels = engine.calculate_tp_levels(
            entry=50_000, direction="long", atr=500, risk_reward=2.0
        )
        assert len(levels) == 3

    def test_long_tp_above_entry(self):
        """All TP levels for a long trade should be above entry."""
        engine = TakeProfitEngine()
        levels = engine.calculate_tp_levels(entry=50_000, direction="long", atr=500)
        for lvl in levels:
            assert lvl["price"] > 50_000

    def test_short_tp_below_entry(self):
        """All TP levels for a short trade should be below entry."""
        engine = TakeProfitEngine()
        levels = engine.calculate_tp_levels(entry=50_000, direction="short", atr=500)
        for lvl in levels:
            assert lvl["price"] < 50_000

    def test_rr_ratio_calculation(self):
        """R:R ratio should be positive."""
        engine = TakeProfitEngine()
        rr = engine.calculate_rr_ratio(entry=100, stop=98, target=104)
        assert rr == pytest.approx(2.0)


# ── DrawdownProtector ─────────────────────────────────────────────────────


class TestDrawdownProtector:
    def test_no_breach_below_max(self):
        """No breach when equity is above peak by less than max_drawdown_pct."""
        dp = DrawdownProtector(max_drawdown_pct=10.0)
        dp.record_equity_peak(10_000.0)
        assert dp.check_max_drawdown_breach(9_500.0) is False  # only 5% down

    def test_breach_above_max(self):
        """Breach detected when equity is more than max_drawdown_pct below peak."""
        dp = DrawdownProtector(max_drawdown_pct=10.0)
        dp.record_equity_peak(10_000.0)
        assert dp.check_max_drawdown_breach(8_000.0) is True  # 20% down

    def test_exposure_multiplier_no_drawdown(self):
        """Exposure multiplier is 1.0 when drawdown is zero."""
        dp = DrawdownProtector(max_drawdown_pct=10.0)
        assert dp.get_exposure_multiplier(0.0) == pytest.approx(1.0)


# ── CircuitBreaker ────────────────────────────────────────────────────────


class TestCircuitBreakerTrigger:
    def test_not_triggered_initially(self):
        """Circuit breaker starts in an un-triggered state."""
        cb = CircuitBreaker()
        assert cb.is_triggered() is False

    @pytest.mark.asyncio
    async def test_trigger_sets_active(self):
        """Triggering the circuit breaker marks it as active."""
        cb = CircuitBreaker()
        await cb.trigger("Test trigger reason")
        assert cb.is_triggered() is True

    @pytest.mark.asyncio
    async def test_trigger_reason_stored(self):
        """Trigger reason is accessible via trigger_info."""
        cb = CircuitBreaker()
        reason = "Exceeded daily loss limit"
        await cb.trigger(reason)
        assert reason in str(cb.trigger_info)

    @pytest.mark.asyncio
    async def test_reset_clears_trigger(self):
        """Resetting the circuit breaker clears the triggered state."""
        cb = CircuitBreaker()
        await cb.trigger("test")
        await cb.reset()
        assert cb.is_triggered() is False


# ── DailyPnLManager ───────────────────────────────────────────────────────


class TestDailyPnLManager:
    @pytest.mark.asyncio
    async def test_record_and_check(self, settings):
        """Recording a PnL update is reflected in daily status."""
        mgr = DailyPnLManager(settings=settings)
        await mgr.record_pnl(100.0, "trade-001")
        status = await mgr.check_daily_status()
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_initial_status_not_limited(self, settings):
        """Fresh manager reports limit_reached=False."""
        mgr = DailyPnLManager(settings=settings)
        status = await mgr.check_daily_status()
        assert status.get("limit_reached") is False


# ── LeverageOptimizer ─────────────────────────────────────────────────────


class TestLeverageOptimizer:
    def test_normal_regime(self):
        """Leverage in normal regime is between 1 and max."""
        opt = LeverageOptimizer(default_leverage=5, max_leverage=20)
        lev = opt.calculate_optimal_leverage(
            symbol="BTC/USDT",
            max_leverage=20,
            volatility_regime="normal",
            market_regime="bull",
        )
        assert 1 <= lev <= 20

    def test_extreme_regime_reduces_leverage(self):
        """Extreme volatility reduces leverage compared to normal."""
        opt = LeverageOptimizer(default_leverage=5, max_leverage=20)
        normal_lev = opt.calculate_optimal_leverage("BTC/USDT", 20, "normal", "bull")
        extreme_lev = opt.calculate_optimal_leverage("BTC/USDT", 20, "extreme", "bull")
        assert extreme_lev <= normal_lev

    def test_max_safe_leverage(self):
        """Max safe leverage is at most max_leverage."""
        opt = LeverageOptimizer(default_leverage=5, max_leverage=20)
        safe = opt.get_max_safe_leverage("BTC/USDT")
        assert safe <= 20
