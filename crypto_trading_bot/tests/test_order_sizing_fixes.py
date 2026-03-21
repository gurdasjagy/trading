"""Tests for the order-sizing bug fixes.

Covers:
- USDT → base currency conversion in PositionSizer
- Notional sanity limit in PaperExchange.create_market_order
- validate_risk_reward() in RiskManager
- USDT → base unit conversion in TradeExecutor.execute_trade
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from risk.position_sizer import PositionSizer
from risk.risk_manager import RiskManager


# ── PositionSizer.convert_to_base_units ──────────────────────────────────


class TestConvertToBaseUnits:
    def test_basic_conversion(self):
        """1000 USDT at 50_000 price = 0.02 BTC."""
        result = PositionSizer.convert_to_base_units(usdt_size=1000.0, price=50_000.0)
        assert result == pytest.approx(0.02)

    def test_zero_price_returns_zero(self):
        """Zero price returns 0.0 (division-by-zero guard)."""
        result = PositionSizer.convert_to_base_units(usdt_size=1000.0, price=0.0)
        assert result == 0.0

    def test_negative_price_returns_zero(self):
        """Negative price returns 0.0."""
        result = PositionSizer.convert_to_base_units(usdt_size=1000.0, price=-100.0)
        assert result == 0.0

    def test_zero_size_returns_zero(self):
        """Zero USDT size returns 0.0."""
        result = PositionSizer.convert_to_base_units(usdt_size=0.0, price=50_000.0)
        assert result == 0.0

    def test_high_price_small_base_amount(self):
        """At BTC price of 73_000, 1000 USDT yields small BTC amount."""
        result = PositionSizer.convert_to_base_units(usdt_size=1000.0, price=73_000.0)
        assert result == pytest.approx(1000.0 / 73_000.0)

    def test_low_price_alt_coin(self):
        """For a cheap alt-coin (0.50 USDT), 100 USDT = 200 units."""
        result = PositionSizer.convert_to_base_units(usdt_size=100.0, price=0.50)
        assert result == pytest.approx(200.0)

    def test_callable_as_instance_method(self):
        """convert_to_base_units works when called on an instance too."""
        sizer = PositionSizer()
        result = sizer.convert_to_base_units(500.0, 25_000.0)
        assert result == pytest.approx(0.02)


# ── RiskManager.validate_risk_reward ─────────────────────────────────────


class TestValidateRiskReward:
    @pytest.fixture
    def rm(self, settings) -> RiskManager:
        return RiskManager(settings=settings)

    def test_valid_long_rr(self, rm):
        """Long trade with adequate R:R (≥1.5) should pass."""
        # risk = 50_000 - 49_000 = 1_000  reward = 51_500 - 50_000 = 1_500  rr = 1.5
        assert rm.validate_risk_reward(50_000, 49_000, [51_500], "long") is True

    def test_insufficient_long_rr(self, rm):
        """Long trade with R:R below 1.5 should fail."""
        # risk = 1_000  reward = 500  rr = 0.5
        assert rm.validate_risk_reward(50_000, 49_000, [50_500], "long") is False

    def test_valid_short_rr(self, rm):
        """Short trade with adequate R:R should pass."""
        # risk = 51_000 - 50_000 = 1_000  reward = 50_000 - 48_500 = 1_500  rr = 1.5
        assert rm.validate_risk_reward(50_000, 51_000, [48_500], "short") is True

    def test_empty_tp_returns_false(self, rm):
        """No take-profit levels should return False."""
        assert rm.validate_risk_reward(50_000, 49_000, [], "long") is False

    def test_zero_stop_loss_returns_false(self, rm):
        """Zero stop-loss should return False."""
        assert rm.validate_risk_reward(50_000, 0.0, [51_500], "long") is False

    def test_zero_entry_returns_false(self, rm):
        """Zero entry price should return False."""
        assert rm.validate_risk_reward(0.0, 49_000, [51_500], "long") is False

    def test_zero_risk_returns_false(self, rm):
        """When stop-loss equals entry (zero risk), should return False."""
        assert rm.validate_risk_reward(50_000, 50_000, [51_500], "long") is False

    def test_floating_point_boundary_long(self, rm):
        """R:R that is numerically 1.5 with a non-round entry should pass (float precision fix)."""
        # entry=48_371.23, stop=47_000.0, stop_distance=1_371.23
        # tp = round(entry + stop_distance * 1.5, 8) — mirrors TakeProfitEngine rounding
        entry = 48_371.23
        stop = 47_000.0
        stop_distance = entry - stop
        tp = round(entry + stop_distance * 1.5, 8)
        # raw reward/risk evaluates to ~1.4999999999 due to float arithmetic
        assert rm.validate_risk_reward(entry, stop, [tp], "long") is True

    def test_floating_point_boundary_short(self, rm):
        """R:R that is numerically 1.5 with a non-round short entry should pass (float precision fix)."""
        entry = 48_371.23
        stop = 49_742.46  # stop above entry for short
        stop_distance = stop - entry
        tp = round(entry - stop_distance * 1.5, 8)
        assert rm.validate_risk_reward(entry, stop, [tp], "short") is True


# ── TradeExecutor unit conversion ─────────────────────────────────────────


class TestTradeExecutorSizeConversion:
    """Verify that execute_trade converts USDT size to base currency before ordering."""

    @pytest.mark.asyncio
    async def test_execute_trade_converts_usdt_to_base(self):
        """execute_trade must call place_market_order with base currency amount."""
        from execution.trade_executor import TradeExecutor
        from exchange.base_exchange import OrderSide, OrderType, OrderStatus
        from exchange.base_exchange import Ticker, Order

        # Set up mocks
        mock_exchange = AsyncMock()
        mock_exchange.get_ticker = AsyncMock(
            return_value=Ticker(
                symbol="BTC/USDT",
                bid=50_000.0,
                ask=50_000.0,
                last=50_000.0,
                high=51_000.0,
                low=49_000.0,
                volume=1000.0,
                timestamp=1_700_000_000_000,
            )
        )
        mock_exchange.set_leverage = AsyncMock(return_value=None)
        # Ensure the code uses get_markets() with an empty dict (no contract market data,
        # so amount_to_order falls back to spot calculation: size/price = 0.02 BTC)
        mock_exchange._client = None
        mock_exchange.get_markets = AsyncMock(return_value={})
        # _resolve_swap_symbol is sync; mock it to avoid returning a coroutine.
        # Markets dict is empty so no contract info is found for this symbol.
        mock_exchange._resolve_swap_symbol = MagicMock(return_value="BTC/USDT:USDT")
        # Return a real dict for orderbook to avoid AsyncMock.get() returning coroutines
        mock_exchange.get_orderbook = AsyncMock(return_value={"bids": [], "asks": []})

        dummy_order = Order(
            id="order-123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=0.02,
            price=50_000.0,
            filled=0.02,
            remaining=0.0,
            status=OrderStatus.CLOSED,
            timestamp=1_700_000_000_000,
            fee=0.0,
        )

        mock_order_manager = AsyncMock()
        mock_order_manager.place_market_order = AsyncMock(return_value=dummy_order)

        mock_position_manager = AsyncMock()
        # Return None so actual_amount falls back to amount_to_order (0.02 BTC)
        mock_position_manager.get_position = AsyncMock(return_value=None)

        executor = TradeExecutor(
            exchange=mock_exchange,
            order_manager=mock_order_manager,
            position_manager=mock_position_manager,
        )

        signal = {
            "symbol": "BTC/USDT",
            "direction": "long",
            "position_size": 1000.0,  # 1000 USDT
            "stop_loss": 49_000.0,
            "take_profit_levels": [51_500.0],
            "leverage": 5,
            "strategy": "test",
        }

        result = await executor.execute_trade(signal)

        assert result["success"] is True
        # The order must be placed with BASE currency units, not USDT
        call_kwargs = mock_order_manager.place_market_order.call_args
        placed_amount = call_kwargs.kwargs.get("amount") or call_kwargs.args[2]
        expected_base = 1000.0 / 50_000.0  # 0.02 BTC
        assert placed_amount == pytest.approx(expected_base, rel=1e-5)
        # result["size"] is in base units now
        assert result["size"] == pytest.approx(expected_base, rel=1e-5)
        # result["size_usdt"] preserves the original USDT value
        assert result["size_usdt"] == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_execute_trade_fails_on_zero_price(self):
        """execute_trade must return error dict when ticker price is zero."""
        from execution.trade_executor import TradeExecutor
        from exchange.base_exchange import Ticker

        mock_exchange = AsyncMock()
        mock_exchange.get_ticker = AsyncMock(
            return_value=Ticker(
                symbol="BTC/USDT",
                bid=0.0,
                ask=0.0,
                last=0.0,  # zero price
                high=0.0,
                low=0.0,
                volume=0.0,
                timestamp=1_700_000_000_000,
            )
        )

        executor = TradeExecutor(
            exchange=mock_exchange,
            order_manager=AsyncMock(),
            position_manager=AsyncMock(),
        )

        signal = {
            "symbol": "BTC/USDT",
            "direction": "long",
            "position_size": 1000.0,
            "stop_loss": 0.0,
            "take_profit_levels": [],
            "leverage": 5,
            "strategy": "test",
        }

        result = await executor.execute_trade(signal)
        assert result["success"] is False
        assert "price" in result["error"].lower()


# ── PaperExchange notional sanity limit ───────────────────────────────────


class TestPaperExchangeNotionalLimit:
    """Verify that PaperExchange rejects orders whose notional exceeds 50% of balance."""

    @pytest.mark.asyncio
    async def test_rejects_oversized_order(self):
        """Order notional (mid-price * amount) > 50% of balance should raise ValueError."""
        from exchange.paper_exchange import PaperExchange
        from exchange.base_exchange import OrderSide

        pe = PaperExchange(starting_balance=10_000.0)
        # Mock _get_live_price to return 50_000: notional = 50_000 * 1.0 = 50_000 USDT,
        # which far exceeds the 5_000 USDT limit (50% of 10_000 balance).
        pe._get_live_price = AsyncMock(return_value=50_000.0)

        with pytest.raises(ValueError, match="sanity limit"):
            await pe.create_market_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                amount=1.0,  # 1 BTC @ 50_000 = 50_000 USDT notional >> 5_000 USDT limit
            )

    @pytest.mark.asyncio
    async def test_allows_normal_order(self):
        """Order notional ≤ 50% of balance should succeed."""
        from exchange.paper_exchange import PaperExchange
        from exchange.base_exchange import OrderSide

        pe = PaperExchange(starting_balance=10_000.0)
        pe._get_live_price = AsyncMock(return_value=50_000.0)
        pe._apply_slippage = lambda price, side: price  # no slippage for test

        # 0.05 BTC @ 50_000 = 2_500 USDT notional (25% of 10_000 balance — OK)
        order = await pe.create_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=0.05,
        )
        assert order is not None
        assert order.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_reduce_only_bypasses_limit(self):
        """reduce_only=True (position close) should bypass the notional limit."""
        from exchange.paper_exchange import PaperExchange
        from exchange.base_exchange import OrderSide, PositionSide

        pe = PaperExchange(starting_balance=10_000.0)
        pe._get_live_price = AsyncMock(return_value=50_000.0)
        pe._apply_slippage = lambda price, side: price

        # Manually insert a position so _close_position_internal has something to close
        pe._positions["BTC/USDT"] = {
            "side": PositionSide.LONG.value,
            "amount": 1.0,
            "entry_price": 50_000.0,
            "leverage": 1,
            "opened_at": 0,
        }

        # Even though notional would be huge, reduce_only should bypass the check
        order = await pe.create_market_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            amount=1.0,
            params={"reduceOnly": True},
        )
        assert order is not None
