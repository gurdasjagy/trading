"""Tests for exchange client components."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from exchange.balance_manager import BalanceManager
from exchange.base_exchange import Balance
from exchange.order_manager import OrderManager
from exchange.position_manager import PositionManager
from exchange.gateio_client import GateIOClient


def _make_mock_exchange() -> MagicMock:
    """Return a MagicMock that satisfies the BaseExchange interface."""
    mock = MagicMock()
    mock.get_balance = AsyncMock(
        return_value=Balance(
            total={"USDT": 10_000.0},
            free={"USDT": 10_000.0},
            usdt_total=10_000.0,
            usdt_free=10_000.0,
        )
    )
    mock.get_positions = AsyncMock(return_value=[])
    mock.create_market_order = AsyncMock(return_value=None)
    return mock


# ── GateIOClient Precious Metals Symbol Mapping ──────────────────────────────


class TestGateIOPreciousMetalsMapping:
    """Test Gate.io precious metals symbol mapping for XAU/XAG TradFi section."""

    def test_xau_usdt_maps_to_xaut(self):
        """XAU/USDT should map to XAUT/USDT (Tether Gold)."""
        client = GateIOClient("test_key", "test_secret")
        result = client._resolve_swap_symbol("XAU/USDT")
        assert result == "XAUT/USDT:USDT", f"Expected XAUT/USDT:USDT, got {result}"

    def test_xaut_usdt_passthrough(self):
        """XAUT/USDT should pass through as XAUT/USDT:USDT."""
        client = GateIOClient("test_key", "test_secret")
        result = client._resolve_swap_symbol("XAUT/USDT")
        assert result == "XAUT/USDT:USDT", f"Expected XAUT/USDT:USDT, got {result}"

    def test_paxg_usdt_passthrough(self):
        """PAXG/USDT should pass through as PAXG/USDT:USDT."""
        client = GateIOClient("test_key", "test_secret")
        result = client._resolve_swap_symbol("PAXG/USDT")
        assert result == "PAXG/USDT:USDT", f"Expected PAXG/USDT:USDT, got {result}"

    def test_regular_crypto_unchanged(self):
        """Regular crypto symbols should convert normally to perpetual format."""
        client = GateIOClient("test_key", "test_secret")
        result = client._resolve_swap_symbol("BTC/USDT")
        assert result == "BTC/USDT:USDT", f"Expected BTC/USDT:USDT, got {result}"

    def test_already_formatted_symbol(self):
        """Symbols already in perpetual format should pass through unchanged."""
        client = GateIOClient("test_key", "test_secret")
        result = client._resolve_swap_symbol("ETH/USDT:USDT")
        assert result == "ETH/USDT:USDT", f"Expected ETH/USDT:USDT, got {result}"


# ── BalanceManager ────────────────────────────────────────────────────────


class TestBalanceManagerInit:
    def test_initializes_without_error(self):
        """BalanceManager can be instantiated with a mock exchange."""
        manager = BalanceManager(exchange=_make_mock_exchange())
        assert manager is not None

    @pytest.mark.asyncio
    async def test_refresh_returns_balance(self):
        """refresh() returns a Balance object."""
        manager = BalanceManager(exchange=_make_mock_exchange())
        balance = await manager.refresh()
        assert balance is not None

    @pytest.mark.asyncio
    async def test_get_balance_returns_balance(self):
        """get_balance() returns a Balance object."""
        manager = BalanceManager(exchange=_make_mock_exchange())
        balance = await manager.get_balance()
        assert balance is not None


# ── OrderManager ──────────────────────────────────────────────────────────


class TestOrderManagerInit:
    def test_initializes_without_error(self):
        """OrderManager can be instantiated with a mock exchange."""
        manager = OrderManager(exchange=_make_mock_exchange())
        assert manager is not None

    def test_no_active_orders_initially(self):
        """OrderManager has no active orders after creation."""
        manager = OrderManager(exchange=_make_mock_exchange())
        assert manager._orders == {}


# ── PositionManager ───────────────────────────────────────────────────────


class TestPositionManagerInit:
    def test_initializes_without_error(self):
        """PositionManager can be instantiated with a mock exchange."""
        manager = PositionManager(exchange=_make_mock_exchange())
        assert manager is not None

    def test_no_positions_initially(self):
        """PositionManager has no tracked positions after creation."""
        manager = PositionManager(exchange=_make_mock_exchange())
        assert manager._positions == {}

    @pytest.mark.asyncio
    async def test_sync_positions_returns_list(self):
        """sync_positions() returns a list."""
        manager = PositionManager(exchange=_make_mock_exchange())
        positions = await manager.sync_positions()
        assert isinstance(positions, list)
