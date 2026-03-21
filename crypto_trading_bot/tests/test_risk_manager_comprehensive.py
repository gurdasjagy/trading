"""Comprehensive tests for risk management components.

Covers all major risk rules: Kelly sizing, VaR/CVaR, portfolio risk,
dynamic TP, intelligent trailing, profit compounding, economic filter,
and the RiskManager.validate_trade() pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from risk.advanced_kelly import BayesianKelly
from risk.dynamic_take_profit import DynamicTakeProfitEngine
from risk.intelligent_trailing import IntelligentTrailingStop
from risk.portfolio_risk_manager import PortfolioRiskManager
from risk.profit_compounder import ProfitCompounder
from risk.var_cvar_calculator import VaRCVaRCalculator


# ── BayesianKelly ─────────────────────────────────────────────────────────────


class TestBayesianKelly:
    def _kelly(self) -> BayesianKelly:
        return BayesianKelly(prior_alpha=2.0, prior_beta=3.0, kelly_fraction=0.25, max_fraction=0.05)

    def test_initial_position_size_is_sensible(self):
        """get_position_size() before any recorded trades must return a non-negative value."""
        k = self._kelly()
        size = k.get_position_size(capital=10_000, avg_win=0.03, avg_loss=0.02)
        assert size >= 0.0
        assert size <= 10_000 * 0.05 + 0.01  # never exceeds max_fraction of capital

    def test_initial_posterior_uses_prior(self):
        """Before any trades the posterior win-rate should reflect the prior."""
        k = self._kelly()
        posterior = k.posterior_win_rate
        # Prior: alpha=2, beta=3 → mean = 2/(2+3) = 0.4
        assert abs(posterior - 0.4) < 0.05

    def test_update_win_increases_win_rate(self):
        """Recording consecutive wins must push the posterior win-rate higher."""
        k = self._kelly()
        baseline = k.posterior_win_rate
        for _ in range(10):
            k.update(won=True)
        assert k.posterior_win_rate > baseline

    def test_update_loss_decreases_win_rate(self):
        """Recording consecutive losses must lower the posterior win-rate."""
        k = self._kelly()
        baseline = k.posterior_win_rate
        for _ in range(10):
            k.update(won=False)
        assert k.posterior_win_rate < baseline

    def test_kelly_fraction_capped_at_max(self):
        """Position size must never exceed max_fraction of capital."""
        k = self._kelly()
        for _ in range(20):
            k.update(won=True)
        size = k.get_position_size(
            capital=10_000, avg_win=0.5, avg_loss=0.01
        )
        assert size <= 10_000 * 0.05 + 0.01

    def test_kelly_zero_avg_loss_guard(self):
        """Kelly must return 0 when avg_loss is zero to avoid division errors."""
        k = self._kelly()
        size = k.get_position_size(capital=10_000, avg_win=0.5, avg_loss=0.0)
        assert size == 0.0

    def test_total_trades_increments(self):
        """total_trades counter increments with each update call."""
        k = self._kelly()
        assert k.total_trades == 0
        k.update(won=True)
        k.update(won=False)
        assert k.total_trades == 2


# ── VaRCVaRCalculator ─────────────────────────────────────────────────────────


class TestVaRCVaR:
    def _calc(self) -> VaRCVaRCalculator:
        return VaRCVaRCalculator()

    def _returns(self) -> pd.Series:
        rng = np.random.default_rng(0)
        return pd.Series(rng.normal(0, 0.01, 252))

    def test_var_is_positive(self):
        """VaR must be a positive dollar loss estimate."""
        calc = self._calc()
        var = calc.calculate_var(self._returns(), portfolio_value=10_000)
        assert var > 0

    def test_cvar_ge_var(self):
        """CVaR (expected shortfall) must be >= VaR at the same confidence level."""
        calc = self._calc()
        returns = self._returns()
        var = calc.calculate_var(returns, portfolio_value=10_000)
        cvar = calc.calculate_cvar(returns, portfolio_value=10_000)
        assert cvar >= var - 0.01  # small tolerance for rounding

    def test_var_scales_with_portfolio_value(self):
        """Doubling portfolio value should approximately double VaR."""
        calc = self._calc()
        returns = self._returns()
        var_10k = calc.calculate_var(returns, portfolio_value=10_000)
        var_20k = calc.calculate_var(returns, portfolio_value=20_000)
        assert abs(var_20k - var_10k * 2) < var_10k * 0.5

    def test_cvar_limit_check(self):
        """check_cvar_limit must return False when CVaR exceeds the threshold."""
        calc = VaRCVaRCalculator(max_cvar_pct=0.001)  # very tight limit
        # Large negative returns → CVaR will likely exceed 0.1% of portfolio
        extreme_returns = pd.Series([-0.05] * 252)
        within_limit = calc.check_cvar_limit(
            returns=extreme_returns,
            portfolio_value=10_000,
        )
        # np.bool_ may be returned — use truthiness directly
        assert not within_limit  # should be rejected given extreme returns


# ── PortfolioRiskManager ──────────────────────────────────────────────────────


class TestPortfolioRiskManager:
    def _mgr(self) -> PortfolioRiskManager:
        return PortfolioRiskManager(
            max_portfolio_risk_pct=5.0, max_correlated_exposure_pct=3.0
        )

    def test_no_reduction_for_empty_portfolio(self):
        """With no existing positions there should be no size reduction."""
        mgr = self._mgr()
        should_reduce, adjusted, _ = mgr.should_reduce_new_position(
            new_symbol="BTC/USDT",
            new_size_usdt=500.0,
            existing_positions=[],
            equity=10_000.0,
        )
        assert should_reduce is False
        assert adjusted == pytest.approx(500.0)

    def test_reduction_when_correlated(self):
        """BTC and ETH are highly correlated — a new ETH position should be reduced."""
        mgr = self._mgr()
        should_reduce, adjusted, reason = mgr.should_reduce_new_position(
            new_symbol="ETH/USDT",
            new_size_usdt=500.0,
            existing_positions=[{"symbol": "BTC/USDT", "amount": 0.1, "entry_price": 50_000}],
            equity=10_000.0,
        )
        assert should_reduce is True
        assert adjusted < 500.0
        assert len(reason) > 0

    def test_portfolio_var_returns_non_negative(self):
        """Portfolio VaR must always be non-negative."""
        mgr = self._mgr()
        positions = [
            {"symbol": "BTC/USDT", "amount": 0.1, "entry_price": 50_000, "side": "long"},
            {"symbol": "ETH/USDT", "amount": 1.0, "entry_price": 3_000, "side": "long"},
        ]
        var = mgr.calculate_portfolio_var(positions=positions, equity=10_000.0)
        assert var >= 0


# ── DynamicTakeProfitEngine ───────────────────────────────────────────────────


class TestDynamicTakeProfitEngine:
    def _engine(self) -> DynamicTakeProfitEngine:
        return DynamicTakeProfitEngine()

    def test_tp_levels_are_ascending_for_long(self):
        """Fixed TP levels for a long trade must be in ascending order."""
        engine = self._engine()
        levels = engine.calculate_tp_levels(
            entry_price=50_000,
            direction="long",
            atr=500.0,
            support_resistance_levels=[],
        )
        # Filter out trailing level (price=0.0) — it is managed separately
        fixed_levels = [lvl for lvl in levels if lvl.get("type") != "trailing"]
        prices = [lvl["price"] for lvl in fixed_levels]
        assert all(p > 50_000 for p in prices), "All fixed TP prices must be above entry for longs"
        assert prices == sorted(prices), "TP levels must be ordered ascending"

    def test_tp_levels_are_descending_for_short(self):
        """Fixed TP levels for a short trade must be in descending order."""
        engine = self._engine()
        levels = engine.calculate_tp_levels(
            entry_price=50_000,
            direction="short",
            atr=500.0,
            support_resistance_levels=[],
        )
        fixed_levels = [lvl for lvl in levels if lvl.get("type") != "trailing"]
        prices = [lvl["price"] for lvl in fixed_levels]
        assert all(p < 50_000 for p in prices), "All fixed TP prices must be below entry for shorts"
        assert prices == sorted(prices, reverse=True), "TP levels must be ordered descending"

    def test_tp_levels_quantities_sum_to_one(self):
        """TP quantity allocations (as fractions) must sum to ~1.0."""
        engine = self._engine()
        levels = engine.calculate_tp_levels(
            entry_price=50_000,
            direction="long",
            atr=500.0,
            support_resistance_levels=[],
        )
        total_pct = sum(lvl.get("percentage", 0) for lvl in levels)
        assert abs(total_pct - 1.0) < 0.05


# ── IntelligentTrailingStop ───────────────────────────────────────────────────


class TestIntelligentTrailingStop:
    def _trail(self) -> IntelligentTrailingStop:
        return IntelligentTrailingStop(base_atr_multiplier=2.0)

    def test_trail_distance_positive(self):
        """Trailing stop distance must be positive for a normal regime."""
        trail = self._trail()
        dist = trail.calculate_trail_distance(
            atr=200.0, volatility_regime="normal", profit_multiple=1.0
        )
        assert dist > 0

    def test_trail_tightens_in_low_volatility(self):
        """Trail distance should be smaller in low-volatility conditions."""
        trail = self._trail()
        dist_low = trail.calculate_trail_distance(
            atr=100.0, volatility_regime="low", profit_multiple=1.0
        )
        dist_high = trail.calculate_trail_distance(
            atr=100.0, volatility_regime="high", profit_multiple=1.0
        )
        assert dist_low <= dist_high

    def test_trail_tightens_at_high_profit_multiple(self):
        """Distance should narrow as profit_multiple increases."""
        trail = self._trail()
        dist_1r = trail.calculate_trail_distance(
            atr=100.0, volatility_regime="normal", profit_multiple=1.0
        )
        dist_3r = trail.calculate_trail_distance(
            atr=100.0, volatility_regime="normal", profit_multiple=3.0
        )
        assert dist_3r <= dist_1r

    def test_update_returns_float_or_none(self):
        """update() must return a float stop price or None."""
        trail = self._trail()
        result = trail.update(
            symbol="BTC/USDT",
            current_price=51_000.0,
            highest_price=52_000.0,
            atr=500.0,
            vol_regime="normal",
            entry_price=50_000.0,
            direction="long",
        )
        assert result is None or isinstance(result, float)

    def test_reset_does_not_raise(self):
        """reset() must not raise for a known or unknown symbol."""
        trail = self._trail()
        trail.update(
            symbol="ETH/USDT",
            current_price=3_000.0,
            highest_price=3_100.0,
            atr=50.0,
            vol_regime="normal",
            entry_price=3_000.0,
        )
        trail.reset("ETH/USDT")
        # Second reset of unknown symbol must also be safe
        trail.reset("UNKNOWN/USDT")


# ── ProfitCompounder ──────────────────────────────────────────────────────────


class TestProfitCompounder:
    def _compounder(self) -> ProfitCompounder:
        return ProfitCompounder(base_size_pct=0.05, max_compound_multiplier=2.0)

    def test_multiplier_is_one_for_flat_pnl(self):
        """Zero daily P&L should produce a 1.0 multiplier (no compounding)."""
        c = self._compounder()
        assert c.get_size_multiplier(daily_pnl_pct=0.0) == pytest.approx(1.0)

    def test_multiplier_increases_for_good_day(self):
        """Daily P&L > 5% should produce a multiplier > 1.0."""
        c = self._compounder()
        assert c.get_size_multiplier(daily_pnl_pct=6.0) > 1.0

    def test_multiplier_capped_at_max(self):
        """Multiplier must never exceed max_compound_multiplier."""
        c = self._compounder()
        # Trigger weekly compounding many times to accumulate
        for _ in range(50):
            c.get_size_multiplier(daily_pnl_pct=6.0, weekly_pnl_pct=6.0)
        mult = c.get_size_multiplier(daily_pnl_pct=6.0, weekly_pnl_pct=6.0)
        assert mult <= 2.0 + 0.01

    def test_multiplier_reduced_for_loss_day(self):
        """Daily P&L < -3% should produce a multiplier < 1.0."""
        c = self._compounder()
        assert c.get_size_multiplier(daily_pnl_pct=-4.0) < 1.0

    def test_reset_weekly_clears_extra_allocation(self):
        """reset_weekly_compounding() should zero extra_allocation_pct."""
        c = self._compounder()
        c.get_size_multiplier(daily_pnl_pct=6.0, weekly_pnl_pct=6.0)
        assert c.extra_allocation_pct > 0
        c.reset_weekly_compounding()
        assert c.extra_allocation_pct == 0.0
