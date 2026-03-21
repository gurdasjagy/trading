"""Reusable MockExchange for tests — fully in-memory, zero network I/O.

Usage::

    from tests.mock_exchange import MockExchange, MockExchangeError

    exchange = MockExchange(initial_balance=10_000)
    exchange.set_price("BTC/USDT", 50_000.0)
    order = await exchange.create_market_order("BTC/USDT", OrderSide.BUY, 1)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# ------------------------------------------------------------------
# Synthetic OHLCV generation constants
# ------------------------------------------------------------------
_SECONDS_PER_HOUR = 3600         # seconds in one candle (1h timeframe default)
_CANDLE_HIGH_MULTIPLIER = 1.005  # high = close × this
_CANDLE_LOW_MULTIPLIER = 0.995   # low  = close × this

from exchange.base_exchange import (
    Balance,
    BaseExchange,
    MarginType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Ticker,
)


class MockExchangeError(Exception):
    """Raised by MockExchange when error injection is active."""


class MockExchange(BaseExchange):
    """Fully in-memory exchange mock for unit and integration tests.

    Features:
    - Configurable price feed per symbol
    - In-memory order book (synthetic, derived from the price feed)
    - Immediate fill at the configured price
    - Error injection hooks for every method
    - Realistic OHLCV generation seeded from the price feed
    - Balance tracking (USDT only for simplicity)
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        maker_fee: float = 0.0,
        taker_fee: float = 0.00075,
        api_key: str = "mock-key",
        secret_key: str = "mock-secret",
        testnet: bool = False,
    ) -> None:
        super().__init__(api_key, secret_key, "", testnet)
        self._usdt_balance: float = initial_balance
        self._maker_fee: float = maker_fee
        self._taker_fee: float = taker_fee

        # Symbol → price
        self._prices: Dict[str, float] = {}
        # order_id → Order
        self._orders: Dict[str, Order] = {}
        # symbol → Position
        self._positions: Dict[str, Position] = {}
        # Leverage settings per symbol
        self._leverages: Dict[str, int] = defaultdict(lambda: 1)
        # Error injection: method_name → exception to raise
        self._errors: Dict[str, Exception] = {}
        # Call counters for assertions
        self.call_counts: Dict[str, int] = defaultdict(int)

        self._order_counter = 0
        self._connected = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_price(self, symbol: str, price: float) -> None:
        """Set the simulated last price for *symbol*."""
        self._prices[symbol] = price

    def inject_error(self, method_name: str, error: Exception) -> None:
        """Make *method_name* raise *error* on its next call."""
        self._errors[method_name] = error

    def clear_error(self, method_name: str) -> None:
        """Remove any injected error for *method_name*."""
        self._errors.pop(method_name, None)

    def _check_error(self, method_name: str) -> None:
        if method_name in self._errors:
            raise self._errors.pop(method_name)

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"mock-order-{self._order_counter:06d}"

    def _price_for(self, symbol: str) -> float:
        return self._prices.get(symbol, 100.0)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self.call_counts["connect"] += 1
        self._check_error("connect")
        self._connected = True

    async def disconnect(self) -> None:
        self.call_counts["disconnect"] += 1
        self._connected = False

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        self.call_counts["get_balance"] += 1
        self._check_error("get_balance")
        return Balance(
            total={"USDT": self._usdt_balance},
            free={"USDT": self._usdt_balance},
            used={"USDT": 0.0},
            usdt_total=self._usdt_balance,
            usdt_free=self._usdt_balance,
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        self.call_counts["get_ticker"] += 1
        self._check_error("get_ticker")
        price = self._price_for(symbol)
        spread = price * 0.0001  # 1 bps synthetic spread
        return Ticker(
            symbol=symbol,
            bid=price - spread,
            ask=price + spread,
            last=price,
            high=price * 1.01,
            low=price * 0.99,
            volume=1_000.0,
            timestamp=int(time.time() * 1000),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        self.call_counts["get_orderbook"] += 1
        self._check_error("get_orderbook")
        price = self._price_for(symbol)
        tick = price * 0.0001
        bids = [[price - tick * (i + 1), 10.0] for i in range(limit)]
        asks = [[price + tick * (i + 1), 10.0] for i in range(limit)]
        return {"bids": bids, "asks": asks, "symbol": symbol}

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        self.call_counts["get_ohlcv"] += 1
        self._check_error("get_ohlcv")
        price = self._price_for(symbol)
        rows = []
        for i in range(limit):
            t = time.time() - (limit - i) * _SECONDS_PER_HOUR
            o = price * (1 + (i % 5 - 2) * 0.001)
            c = o * (1 + 0.002)
            h = max(o, c) * _CANDLE_HIGH_MULTIPLIER
            lo = min(o, c) * _CANDLE_LOW_MULTIPLIER
            rows.append(
                {"timestamp": int(t * 1000), "open": o, "high": h, "low": lo, "close": c, "volume": 500.0}
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        self.call_counts["create_market_order"] += 1
        self._check_error("create_market_order")
        price = self._price_for(symbol)
        order_id = self._next_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side=side,
            amount=amount,
            price=price,
            filled=amount,
            remaining=0.0,
            status=OrderStatus.CLOSED,
            timestamp=int(time.time() * 1000),
            fee=amount * price * self._taker_fee,
        )
        self._orders[order_id] = order
        self._apply_fill(symbol, side, amount, price)
        return order

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        self.call_counts["create_limit_order"] += 1
        self._check_error("create_limit_order")
        order_id = self._next_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.LIMIT,
            side=side,
            amount=amount,
            price=price,
            filled=amount,
            remaining=0.0,
            status=OrderStatus.CLOSED,
            timestamp=int(time.time() * 1000),
            fee=amount * price * self._maker_fee,
        )
        self._orders[order_id] = order
        self._apply_fill(symbol, side, amount, price)
        return order

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        self.call_counts["create_stop_loss_order"] += 1
        self._check_error("create_stop_loss_order")
        order_id = self._next_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.STOP_LOSS,
            side=side,
            amount=amount,
            price=stop_price,
            filled=0.0,
            remaining=amount,
            status=OrderStatus.OPEN,
            timestamp=int(time.time() * 1000),
        )
        self._orders[order_id] = order
        return order

    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        self.call_counts["create_take_profit_order"] += 1
        self._check_error("create_take_profit_order")
        order_id = self._next_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.TAKE_PROFIT,
            side=side,
            amount=amount,
            price=tp_price,
            filled=0.0,
            remaining=amount,
            status=OrderStatus.OPEN,
            timestamp=int(time.time() * 1000),
        )
        self._orders[order_id] = order
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        self.call_counts["cancel_order"] += 1
        self._check_error("cancel_order")
        if order_id in self._orders:
            order = self._orders[order_id]
            updated = order.model_copy(update={"status": OrderStatus.CANCELED})
            self._orders[order_id] = updated
            return {"id": order_id, "status": "canceled"}
        return {"id": order_id, "status": "not_found"}

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        self.call_counts["cancel_all_orders"] += 1
        self._check_error("cancel_all_orders")
        canceled = []
        for oid, order in list(self._orders.items()):
            if order.symbol == symbol and order.status == OrderStatus.OPEN:
                updated = order.model_copy(update={"status": OrderStatus.CANCELED})
                self._orders[oid] = updated
                canceled.append({"id": oid, "status": "canceled"})
        return canceled

    async def get_order(self, order_id: str, symbol: str) -> Order:
        self.call_counts["get_order"] += 1
        self._check_error("get_order")
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found in MockExchange")
        return self._orders[order_id]

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        self.call_counts["get_open_orders"] += 1
        self._check_error("get_open_orders")
        return [
            o for o in self._orders.values()
            if o.status == OrderStatus.OPEN and (symbol is None or o.symbol == symbol)
        ]

    # ------------------------------------------------------------------
    # Position & leverage
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        self.call_counts["set_leverage"] += 1
        self._check_error("set_leverage")
        self._leverages[symbol] = leverage
        return {"symbol": symbol, "leverage": leverage}

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        self.call_counts["set_margin_type"] += 1
        return {"symbol": symbol, "marginType": margin_type.value}

    async def get_positions(self) -> List[Position]:
        self.call_counts["get_positions"] += 1
        self._check_error("get_positions")
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Optional[Position]:
        self.call_counts["get_position"] += 1
        self._check_error("get_position")
        return self._positions.get(symbol)

    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        self.call_counts["close_position"] += 1
        self._check_error("close_position")
        position = self._positions.pop(symbol, None)
        price = self._price_for(symbol)
        close_side = OrderSide.SELL if (
            position and position.side == PositionSide.LONG
        ) else OrderSide.BUY
        close_amount = amount or (position.amount if position else 0.0)
        order_id = self._next_order_id()
        return Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side=close_side,
            amount=close_amount,
            price=price,
            filled=close_amount,
            remaining=0.0,
            status=OrderStatus.CLOSED,
            timestamp=int(time.time() * 1000),
        )

    # ------------------------------------------------------------------
    # Derivatives-specific data
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        self.call_counts["get_funding_rate"] += 1
        self._check_error("get_funding_rate")
        return 0.0001  # 0.01% default

    async def get_open_interest(self, symbol: str) -> float:
        self.call_counts["get_open_interest"] += 1
        return 1_000_000.0

    # ------------------------------------------------------------------
    # WebSocket subscriptions (no-op for sync tests)
    # ------------------------------------------------------------------

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        self.call_counts["subscribe_ticker"] += 1

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        self.call_counts["subscribe_orderbook"] += 1

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        self.call_counts["subscribe_trades"] += 1

    async def subscribe_user_data(self, callback: Callable) -> None:
        self.call_counts["subscribe_user_data"] += 1

    # ------------------------------------------------------------------
    # Internal fill logic
    # ------------------------------------------------------------------

    def _apply_fill(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
    ) -> None:
        """Update the in-memory position and balance after a fill."""
        lev = self._leverages.get(symbol, 1)
        notional = amount * price
        margin = notional / lev
        fee = notional * self._taker_fee

        if side == OrderSide.BUY:
            self._usdt_balance -= margin + fee
            existing = self._positions.get(symbol)
            if existing and existing.side == PositionSide.LONG:
                # Add to long
                new_amount = existing.amount + amount
                avg_entry = (
                    (existing.entry_price * existing.amount + price * amount) / new_amount
                )
                self._positions[symbol] = existing.model_copy(
                    update={"amount": new_amount, "entry_price": avg_entry}
                )
            elif existing and existing.side == PositionSide.SHORT:
                # Partial or full close of short
                remaining = existing.amount - amount
                if remaining > 0:
                    self._positions[symbol] = existing.model_copy(
                        update={"amount": remaining}
                    )
                else:
                    self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    amount=amount,
                    entry_price=price,
                    current_price=price,
                    leverage=lev,
                    margin=margin,
                    timestamp=int(time.time() * 1000),
                )
        else:  # SELL
            self._usdt_balance -= margin + fee
            existing = self._positions.get(symbol)
            if existing and existing.side == PositionSide.SHORT:
                new_amount = existing.amount + amount
                avg_entry = (
                    (existing.entry_price * existing.amount + price * amount) / new_amount
                )
                self._positions[symbol] = existing.model_copy(
                    update={"amount": new_amount, "entry_price": avg_entry}
                )
            elif existing and existing.side == PositionSide.LONG:
                remaining = existing.amount - amount
                if remaining > 0:
                    self._positions[symbol] = existing.model_copy(
                        update={"amount": remaining}
                    )
                else:
                    self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    amount=amount,
                    entry_price=price,
                    current_price=price,
                    leverage=lev,
                    margin=margin,
                    timestamp=int(time.time() * 1000),
                )

    @property
    def name(self) -> str:
        return "MockExchange"
