"""Tests for anti-liquidation integration."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_liquidation_critical_auto_closes():
    """When anti-liquidation returns close_position, the position should be closed."""
    from risk.anti_liquidation import AntiLiquidationManager, LiquidationRisk

    mgr = AntiLiquidationManager()

    # Build a critical risk (distance ≤ 5 %)
    risk = LiquidationRisk(
        symbol="BTC/USDT",
        entry_price=50_000.0,
        current_price=50_300.0,
        liquidation_price=50_000.0 * 0.97,  # very close
        leverage=20,
        side="long",
        notional_value=100_000.0,
        distance_pct=0.03,
        risk_level="critical",
    )

    action = mgr.get_action(risk)
    assert action == "close_position"


@pytest.mark.asyncio
async def test_liquidation_danger_reduces_position():
    """When anti-liquidation returns reduce_50pct, a 50 % reduction should be recommended."""
    from risk.anti_liquidation import AntiLiquidationManager, LiquidationRisk

    mgr = AntiLiquidationManager()

    risk = LiquidationRisk(
        symbol="ETH/USDT",
        entry_price=3_000.0,
        current_price=3_100.0,
        liquidation_price=2_850.0,
        leverage=10,
        side="long",
        notional_value=50_000.0,
        distance_pct=0.08,
        risk_level="danger",
    )

    action = mgr.get_action(risk)
    assert action == "reduce_50pct"


@pytest.mark.asyncio
async def test_liquidation_safe_no_action():
    """Safe position should return no action."""
    from risk.anti_liquidation import AntiLiquidationManager, LiquidationRisk

    mgr = AntiLiquidationManager()

    risk = LiquidationRisk(
        symbol="SOL/USDT",
        entry_price=100.0,
        current_price=110.0,
        liquidation_price=50.0,
        leverage=5,
        side="long",
        notional_value=10_000.0,
        distance_pct=0.50,
        risk_level="safe",
    )

    action = mgr.get_action(risk)
    assert action is None


@pytest.mark.asyncio
async def test_circuit_breaker_daily_loss_triggers():
    """Circuit breaker should trigger when daily loss exceeds threshold."""
    from risk.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker()
    cb.register_callbacks(
        close_positions=AsyncMock(),
        cancel_orders=AsyncMock(),
        send_alert=AsyncMock(),
    )
    triggered = await cb.check(daily_loss_pct=6.0)
    assert triggered is True
    assert cb.is_triggered()


@pytest.mark.asyncio
async def test_circuit_breaker_resets_after_manual_reset():
    """Circuit breaker should be resettable via reset()."""
    from risk.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker()
    cb.register_callbacks(
        close_positions=AsyncMock(),
        cancel_orders=AsyncMock(),
        send_alert=AsyncMock(),
    )
    await cb.check(daily_loss_pct=6.0)
    assert cb.is_triggered()

    await cb.reset()
    assert not cb.is_triggered()


@pytest.mark.asyncio
async def test_circuit_breaker_not_triggered_below_threshold():
    """Circuit breaker should NOT trigger when conditions are within limits."""
    from risk.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker()
    triggered = await cb.check(daily_loss_pct=2.0, consecutive_losses=3)
    assert triggered is False
    assert not cb.is_triggered()


@pytest.mark.asyncio
async def test_circuit_breaker_abnormal_market_triggers():
    """Circuit breaker should trigger on abnormal market movement."""
    from risk.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker()
    cb.register_callbacks(
        close_positions=AsyncMock(),
        cancel_orders=AsyncMock(),
        send_alert=AsyncMock(),
    )
    triggered = await cb.check(abnormal_market=True)
    assert triggered is True


def test_tiered_maintenance_margin():
    """Maintenance margin should increase with position size (Gate.io tiers)."""
    from risk.anti_liquidation import AntiLiquidationManager

    mgr = AntiLiquidationManager()

    # Small position: ≤50K → 0.5 %
    assert mgr._get_maintenance_margin(10_000) == 0.005
    # Medium position: ≤200K → 1.0 %
    assert mgr._get_maintenance_margin(100_000) == 0.01
    # Large position: ≤1M → 2.0 %
    assert mgr._get_maintenance_margin(500_000) == 0.02
    # Very large: ≤5M → 2.5 %
    assert mgr._get_maintenance_margin(2_000_000) == 0.025
    # Extremely large: >5M → 5.0 %
    assert mgr._get_maintenance_margin(10_000_000) == 0.05


def test_portfolio_assessment_exceeds_margin():
    """assess_portfolio should flag when margin usage exceeds 70%."""
    from risk.anti_liquidation import AntiLiquidationManager

    mgr = AntiLiquidationManager()
    positions = [
        {
            "symbol": "BTC/USDT",
            "entry_price": 50_000.0,
            "current_price": 50_000.0,
            "leverage": 10,
            "side": "long",
            "size": 1.0,
            "margin": 5_000.0,
        },
        {
            "symbol": "ETH/USDT",
            "entry_price": 3_000.0,
            "current_price": 3_000.0,
            "leverage": 5,
            "side": "long",
            "size": 10.0,
            "margin": 6_000.0,
        },
    ]
    # Use a small equity so margin usage is high
    result = mgr.assess_portfolio(positions, equity=1_000.0)
    assert "exceeds_margin_limit" in result
    assert result["exceeds_margin_limit"] is True
