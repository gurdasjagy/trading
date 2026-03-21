"""Tests for WebSocket-driven fill confirmation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_register_fill_waiter_creates_future():
    """register_fill_waiter should create and store an asyncio.Future."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    future = client.register_fill_waiter("order_123")
    assert "order_123" in client._pending_fills
    assert isinstance(future, asyncio.Future)
    assert not future.done()


@pytest.mark.asyncio
async def test_ws_fill_resolves_future():
    """When a futures.orders WS message arrives with status=finished, the pending future should resolve."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    client._ws_orders = {}
    client._ws_last_update = 0.0
    client._ws_positions = {}
    client._event_bus = None

    future = client.register_fill_waiter("12345")

    ws_message = {
        "channel": "futures.orders",
        "event": "update",
        "result": [{"id": 12345, "status": "finished", "size": 10, "price": "50000"}],
    }
    await client._handle_ws_user_data(ws_message)

    assert future.done()
    result = future.result()
    assert result["status"] == "finished"


@pytest.mark.asyncio
async def test_ws_fill_no_pending_future():
    """WS fill for an order with no pending future should not raise."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    client._ws_orders = {}
    client._ws_last_update = 0.0
    client._ws_positions = {}
    client._event_bus = None

    ws_message = {
        "channel": "futures.orders",
        "event": "update",
        "result": [{"id": 99999, "status": "finished"}],
    }
    # Should not raise
    await client._handle_ws_user_data(ws_message)


@pytest.mark.asyncio
async def test_ws_fill_cancelled_order_resolves_future():
    """A WS message with status=cancelled should also resolve the pending future."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    client._ws_orders = {}
    client._ws_last_update = 0.0
    client._ws_positions = {}
    client._event_bus = None

    future = client.register_fill_waiter("55555")

    ws_message = {
        "channel": "futures.orders",
        "event": "update",
        "result": [{"id": 55555, "status": "cancelled"}],
    }
    await client._handle_ws_user_data(ws_message)

    assert future.done()
    assert future.result()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_ws_fill_open_order_does_not_resolve_future():
    """A WS message for an open (not finished/cancelled) order must NOT resolve the future."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    client._ws_orders = {}
    client._ws_last_update = 0.0
    client._ws_positions = {}
    client._event_bus = None

    future = client.register_fill_waiter("77777")

    ws_message = {
        "channel": "futures.orders",
        "event": "update",
        "result": [{"id": 77777, "status": "open"}],
    }
    await client._handle_ws_user_data(ws_message)

    assert not future.done()


@pytest.mark.asyncio
async def test_ws_fill_ignores_unknown_event():
    """WS messages with unknown event types (not 'update' or 'subscribe') should be silently ignored."""
    from exchange.gateio_client import GateIOClient

    client = GateIOClient.__new__(GateIOClient)
    client._pending_fills = {}
    client._ws_orders = {}
    client._ws_last_update = 0.0
    client._ws_positions = {}
    client._event_bus = None

    future = client.register_fill_waiter("88888")

    ws_message = {
        "channel": "futures.orders",
        "event": "heartbeat",  # unknown event type — not 'update' or 'subscribe'
        "result": [{"id": 88888, "status": "finished"}],
    }
    await client._handle_ws_user_data(ws_message)

    # unknown event type must NOT resolve the future
    assert not future.done()


@pytest.mark.asyncio
async def test_wait_for_fill_ws_first_then_rest_fallback():
    """_wait_for_fill should try WS first (when fresh), then fall back to REST polling."""
    from exchange.base_exchange import OrderStatus
    from execution.trade_executor import TradeExecutor

    mock_exchange = AsyncMock()
    # is_ws_data_fresh is a synchronous method — override with a plain MagicMock
    mock_exchange.is_ws_data_fresh = MagicMock(return_value=False)

    mock_order = MagicMock()
    mock_order.status = OrderStatus.CLOSED
    mock_order.price = 50000.0
    mock_order.filled = 10
    mock_order.amount = 10
    mock_exchange.get_order.return_value = mock_order

    executor = TradeExecutor.__new__(TradeExecutor)
    executor._exchange = mock_exchange

    result = await executor._wait_for_fill("order_1", "BTC/USDT", timeout=2.0)
    assert result is not None
    assert result.status == OrderStatus.CLOSED


@pytest.mark.asyncio
async def test_wait_for_fill_returns_none_for_cancelled_order():
    """_wait_for_fill should return None when an order is cancelled."""
    from exchange.base_exchange import OrderStatus
    from execution.trade_executor import TradeExecutor

    mock_exchange = AsyncMock()
    mock_exchange.is_ws_data_fresh = MagicMock(return_value=False)

    mock_order = MagicMock()
    mock_order.status = OrderStatus.CANCELED
    mock_order.filled = 0
    mock_order.amount = 10
    mock_exchange.get_order.return_value = mock_order

    executor = TradeExecutor.__new__(TradeExecutor)
    executor._exchange = mock_exchange

    result = await executor._wait_for_fill("order_cancelled", "ETH/USDT", timeout=2.0)
    assert result is None


@pytest.mark.asyncio
async def test_wait_for_fill_ws_path_when_fresh():
    """_wait_for_fill should use the WebSocket path when WS data is fresh."""
    from exchange.base_exchange import OrderStatus
    from execution.trade_executor import TradeExecutor

    loop = asyncio.get_running_loop()
    mock_future: asyncio.Future = loop.create_future()
    mock_future.set_result({"id": "ws_order_1", "status": "finished"})

    mock_exchange = AsyncMock()
    mock_exchange.is_ws_data_fresh = MagicMock(return_value=True)
    mock_exchange.register_fill_waiter.return_value = mock_future

    ws_order = MagicMock()
    ws_order.status = OrderStatus.CLOSED
    ws_order.filled = 5
    ws_order.amount = 5
    mock_exchange.get_order.return_value = ws_order

    executor = TradeExecutor.__new__(TradeExecutor)
    executor._exchange = mock_exchange

    result = await executor._wait_for_fill("ws_order_1", "BTC/USDT", timeout=2.0)
    assert result is not None
    assert result.status == OrderStatus.CLOSED
    mock_exchange.register_fill_waiter.assert_called_once_with("ws_order_1")
