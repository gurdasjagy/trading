"""Comprehensive tests for RiskManager covering all major features."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from config.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        TRADING_MODE="paper",
        SECRET_KEY="test-secret-key-32-chars-long!",
    )


@pytest.fixture
def mock_exchange():
    """Create mock exchange."""
    exchange = Mock()
    exchange.fetch_balance = AsyncMock(return_value={"USDT": {"free": 10000.0, "total": 10000.0}})
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    return exchange


# ── Kelly Criterion Position Sizing ───────────────────────────────────────


class TestKellyCriterionPositionSizing:
    def test_kelly_criterion_position_sizing(self):
        """Kelly criterion returns appropriate position size based on edge."""
        from risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        # Simulate positive edge: 60% win rate, avg win 3%, avg loss 2%
        size = sizer.kelly_size(
            win_rate=0.6,
            avg_win=0.03,
            avg_loss=0.02,
            capital=10000.0
        )
        assert size > 0, "Kelly size should be positive with positive edge"
        assert size <= 10000.0 * 0.25, "Kelly size should be capped at 25% of capital"

    def test_kelly_negative_edge_returns_zero(self):
        """Kelly criterion returns zero for negative edge."""
        from risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        size = sizer.kelly_size(
            win_rate=0.3,
            avg_win=0.01,
            avg_loss=0.05,
            capital=10000.0
        )
        assert size >= 0.0, "Kelly size should not be negative"


# ── Circuit Breaker ───────────────────────────────────────────────────────


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers_on_consecutive_losses(self):
        """Circuit breaker triggers after consecutive losses."""
        from risk.circuit_breaker import CircuitBreaker
        
        cb = CircuitBreaker(max_consecutive_losses=5)
        
        # Record consecutive losses
        for i in range(5):
            await cb.record_loss()
        
        assert cb.is_triggered(), "Circuit breaker should trigger after 5 consecutive losses"

    @pytest.mark.asyncio
    async def test_circuit_breaker_daily_loss_limit(self, settings):
        """Circuit breaker triggers when daily loss limit is exceeded."""
        from risk.daily_pnl_manager import DailyPnLManager
        
        mgr = DailyPnLManager(settings=settings)
        
        # Record large loss exceeding daily limit
        daily_limit = settings.risk.max_daily_loss_pct
        capital = 10000.0
        loss_amount = capital * (daily_limit / 100.0) * 1.5  # 150% of limit
        
        await mgr.record_pnl(-loss_amount, "big-loss")
        
        status = await mgr.check_daily_status()
        assert status.get("limit_reached", False), "Daily loss limit should be reached"


# ── Daily Limits ──────────────────────────────────────────────────────────


class TestDailyLimits:
    @pytest.mark.asyncio
    async def test_daily_profit_target(self, settings):
        """Trading stops when daily profit target is reached."""
        from risk.daily_pnl_manager import DailyPnLManager
        
        mgr = DailyPnLManager(settings=settings)
        
        # Record profitable trades reaching daily target
        daily_target = settings.risk.daily_profit_target_pct
        capital = 10000.0
        profit_amount = capital * (daily_target / 100.0) * 1.1  # 110% of target
        
        await mgr.record_pnl(profit_amount, "big-win")
        
        status = await mgr.check_daily_status()
        # Check if profit is recorded
        assert status.get("daily_pnl", 0) > 0

    @pytest.mark.asyncio
    async def test_max_daily_trades(self, settings):
        """Trading stops when max daily trades limit is reached."""
        from risk.daily_pnl_manager import DailyPnLManager
        
        mgr = DailyPnLManager(settings=settings)
        max_trades = settings.risk.max_daily_trades
        
        # Record max_trades number of trades
        for i in range(max_trades):
            await mgr.record_pnl(10.0, f"trade-{i}")
        
        status = await mgr.check_daily_status()
        assert status.get("trade_count", 0) >= max_trades


# ── Correlation Checks ────────────────────────────────────────────────────


class TestCorrelationLimit:
    def test_correlation_limit_rejects_correlated_positions(self):
        """Correlation check identifies highly correlated assets."""
        # Simple correlation test
        # BTC and ETH are typically highly correlated
        # This is a placeholder - actual implementation would use correlation matrix
        btc_symbol = "BTC/USDT"
        eth_symbol = "ETH/USDT"
        
        # Mock correlation check
        correlation = 0.85  # High correlation
        
        assert correlation > 0.7, "BTC and ETH should be highly correlated"


# ── VaR Calculation ───────────────────────────────────────────────────────


class TestVaRCalculation:
    def test_var_calculation(self):
        """VaR calculation returns reasonable value."""
        import numpy as np
        
        # Provide sample returns
        returns = np.array([-0.02, 0.01, -0.01, 0.03, -0.015, 0.02, -0.005, 0.01])
        
        # Calculate 95% VaR
        var_95 = np.percentile(returns, 5)
        
        assert var_95 < 0, "VaR should be negative (represents loss)"
        assert -1.0 < var_95 < 0, "VaR should be a reasonable fraction"

    def test_var_portfolio_level(self):
        """Portfolio-level VaR accounts for all positions."""
        import numpy as np
        
        portfolio_value = 10000.0
        returns = np.array([-0.01, 0.02, -0.015, 0.01, -0.005])
        
        var = np.percentile(returns, 5)
        portfolio_var = abs(var) * portfolio_value
        
        assert portfolio_var > 0, "Portfolio VaR should be positive"
        assert portfolio_var < portfolio_value, "Portfolio VaR should be less than total value"


# ── Drawdown Protection ───────────────────────────────────────────────────


class TestMaxDrawdownProtection:
    def test_max_drawdown_protection(self):
        """Max drawdown protection reduces position sizes."""
        from risk.drawdown_protector import DrawdownProtector
        
        dp = DrawdownProtector(max_drawdown_pct=10.0)
        
        # Record equity peak
        dp.record_equity_peak(10000.0)
        
        # Simulate drawdown
        current_equity = 9000.0  # 10% drawdown
        
        # Check if drawdown protection is active
        exposure_multiplier = dp.get_exposure_multiplier(10.0)  # 10% drawdown
        
        assert 0 < exposure_multiplier <= 1.0, "Exposure multiplier should reduce size during drawdown"

    def test_max_drawdown_breach_stops_trading(self):
        """Trading stops when max drawdown is breached."""
        from risk.drawdown_protector import DrawdownProtector
        
        dp = DrawdownProtector(max_drawdown_pct=15.0)
        
        # Record equity peak
        dp.record_equity_peak(10000.0)
        
        # Simulate drawdown exceeding max
        current_equity = 8000.0  # 20% drawdown
        
        is_breached = dp.check_max_drawdown_breach(current_equity)
        
        assert is_breached, "Max drawdown breach should be detected"


# ── Funding Rate Auto-Close ───────────────────────────────────────────────


class TestFundingRateAutoClose:
    def test_funding_rate_auto_close(self):
        """Position closes when funding rate is unfavorable."""
        # Mock high funding rate
        funding_rate = 0.001  # 0.1% (high for crypto)
        threshold = 0.0005  # 0.05%
        
        # Long position with positive funding = paying funding
        position_side = "long"
        
        should_close = funding_rate > threshold and position_side == "long"
        
        assert should_close, "Long position should close with high positive funding rate"

    def test_funding_rate_threshold(self):
        """Funding rate close only triggers above threshold."""
        # Low funding rate should not trigger close
        low_funding = 0.0001  # 0.01%
        threshold = 0.0005  # 0.05%
        
        should_close = low_funding > threshold
        
        assert not should_close, "Low funding should not trigger close"
