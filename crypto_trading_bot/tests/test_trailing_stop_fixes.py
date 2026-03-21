"""Tests for trailing stop and position manager fixes."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_position(symbol: str, side: str, entry_price: float, amount: float = 1.0):
    """Helper to build a minimal Position mock."""
    from exchange.base_exchange import Position, PositionSide

    pos = MagicMock(spec=Position)
    pos.symbol = symbol
    pos.side = PositionSide.LONG if side == "long" else PositionSide.SHORT
    pos.entry_price = entry_price
    pos.amount = amount
    pos.leverage = 5
    pos.unrealized_pnl = 0.0
    return pos


def _make_tracker(symbol: str, side: str, entry_price: float, trailing_stop: float, amount: float = 1.0):
    """Helper to build a PositionTracker with a trailing stop set."""
    from exchange.position_manager import PositionTracker

    pos = _make_position(symbol, side, entry_price, amount)
    tracker = PositionTracker(
        position=pos,
        strategy="test",
        opened_at=datetime.now(tz=timezone.utc),
        trailing_stop=trailing_stop,
        highest_price=entry_price,
        lowest_price=entry_price,
    )
    return tracker


@pytest.mark.asyncio
async def test_trailing_stop_uses_high_water_mark():
    """Trailing stop for a long should trail from the highest price, not entry."""
    from exchange.position_manager import PositionManager

    mock_exchange = AsyncMock()
    pm = PositionManager(exchange=mock_exchange)

    symbol = "BTC/USDT"
    entry = 50_000.0
    distance = 1_000.0
    tracker = _make_tracker(symbol, "long", entry, distance)
    pm._positions[symbol] = tracker

    # Price rises: new high-water mark should set SL = 52000 - 1000 = 51000
    new_sl = await pm.update_trailing_stop(symbol, 52_000.0)
    assert new_sl == pytest.approx(51_000.0)
    assert tracker.highest_price == 52_000.0
    assert tracker.trailing_stop_active is True

    # Price drops back: SL should NOT move down
    no_change = await pm.update_trailing_stop(symbol, 50_500.0)
    assert no_change is None
    # SL stays at 51_000
    assert tracker.stop_loss == pytest.approx(51_000.0)


@pytest.mark.asyncio
async def test_trailing_stop_short_uses_low_water_mark():
    """Trailing stop for a short should trail from the lowest price."""
    from exchange.position_manager import PositionManager

    mock_exchange = AsyncMock()
    pm = PositionManager(exchange=mock_exchange)

    symbol = "ETH/USDT"
    entry = 3_000.0
    distance = 100.0
    tracker = _make_tracker(symbol, "short", entry, distance)
    pm._positions[symbol] = tracker

    # Price falls: new low-water mark → SL = 2800 + 100 = 2900
    new_sl = await pm.update_trailing_stop(symbol, 2_800.0)
    assert new_sl == pytest.approx(2_900.0)
    assert tracker.lowest_price == 2_800.0

    # Price rises: SL should NOT move up
    no_change = await pm.update_trailing_stop(symbol, 2_850.0)
    assert no_change is None
    assert tracker.stop_loss == pytest.approx(2_900.0)


@pytest.mark.asyncio
async def test_trailing_stop_no_position_returns_none():
    """update_trailing_stop should return None when the symbol is not tracked."""
    from exchange.position_manager import PositionManager

    pm = PositionManager(exchange=AsyncMock())
    result = await pm.update_trailing_stop("UNKNOWN/USDT", 50_000.0)
    assert result is None


@pytest.mark.asyncio
async def test_watchdog_sl_cooldown():
    """_is_sl_recently_placed should return True within cooldown period."""
    from exchange.position_manager import PositionManager

    pm = PositionManager(exchange=AsyncMock())
    symbol = "SOL/USDT"

    # Record an SL placement now
    pm._sl_placement_timestamps[symbol] = time.time()

    assert pm._is_sl_recently_placed(symbol) is True


@pytest.mark.asyncio
async def test_watchdog_sl_cooldown_expired():
    """_is_sl_recently_placed should return False after cooldown period."""
    from exchange.position_manager import PositionManager

    pm = PositionManager(exchange=AsyncMock())
    symbol = "SOL/USDT"

    # Record an SL placement in the distant past
    pm._sl_placement_timestamps[symbol] = time.time() - 9999.0

    assert pm._is_sl_recently_placed(symbol) is False


@pytest.mark.asyncio
async def test_mark_position_protected_sets_cooldown():
    """mark_position_protected should register the symbol and set an SL timestamp."""
    from exchange.position_manager import PositionManager

    pm = PositionManager(exchange=AsyncMock())
    symbol = "XRP/USDT"
    pm.mark_position_protected(symbol, "sl_order_abc")

    assert symbol in pm._protected_symbols
    assert pm._is_sl_recently_placed(symbol) is True


@pytest.mark.asyncio
async def test_sync_lock_prevents_race():
    """Concurrent sync_positions calls should be serialised (not overlap)."""
    from exchange.position_manager import PositionManager

    call_log: list[str] = []
    lock_held = False

    async def mock_sync_impl(*_args, **_kwargs):
        nonlocal lock_held
        # If lock_held is True, two calls are running concurrently — that's a bug
        assert not lock_held, "Concurrent sync_positions calls detected!"
        lock_held = True
        await asyncio.sleep(0.02)
        lock_held = False
        return []

    mock_exchange = AsyncMock()
    pm = PositionManager(exchange=mock_exchange)

    # Patch the internal implementation to use our fake
    pm._sync_positions_impl = mock_sync_impl

    # Fire three concurrent sync calls
    await asyncio.gather(
        pm.sync_positions(),
        pm.sync_positions(),
        pm.sync_positions(),
    )
    # If we reach here without assertion errors, serialisation is correct


@pytest.mark.asyncio
async def test_break_even_uses_current_amount():
    """activate_break_even should use the current position amount, not original."""
    from exchange.base_exchange import Order, OrderSide, PositionSide
    from exchange.position_manager import PositionManager, PositionTracker

    symbol = "BTC/USDT"
    entry_price = 50_000.0
    # Simulate a partial close: only 0.5 contracts remain
    remaining_amount = 0.5

    mock_pos = _make_position(symbol, "long", entry_price, amount=remaining_amount)
    mock_pos.side = PositionSide.LONG

    tracker = PositionTracker(
        position=mock_pos,
        strategy="test",
        opened_at=datetime.now(tz=timezone.utc),
        stop_loss=49_000.0,
    )

    mock_exchange = AsyncMock()
    mock_exchange.get_open_orders.return_value = []
    mock_sl_order = MagicMock(spec=Order)
    mock_sl_order.id = "be_sl_order_1"
    mock_exchange.create_stop_loss_order.return_value = mock_sl_order

    pm = PositionManager(exchange=mock_exchange)
    pm._positions[symbol] = tracker

    result = await pm.activate_break_even(symbol)
    assert result is True

    # Verify the SL order was placed with the current (partial) amount
    mock_exchange.create_stop_loss_order.assert_called_once()
    call_args = mock_exchange.create_stop_loss_order.call_args
    # Positional args: (symbol, side, amount, price)
    placed_amount = call_args[0][2] if call_args[0] else call_args[1].get("amount")
    assert placed_amount == pytest.approx(remaining_amount)
    assert tracker.break_even_activated is True
