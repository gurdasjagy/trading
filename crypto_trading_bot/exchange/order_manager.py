"""Order lifecycle management — placement, monitoring, and cancellation."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from .base_exchange import BaseExchange, Order, OrderSide, OrderStatus, OrderType


class OrderTracker(BaseModel):
    """Wrapper around an :class:`Order` that carries strategy metadata."""

    order: Order
    created_at: datetime
    strategy: str
    position_id: Optional[str] = None
    is_stop_loss: bool = False
    is_take_profit: bool = False
    auto_cancel_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


class OrderManager:
    """Manages the full lifecycle of orders on a single exchange.

    Responsibilities:
    * Placement of market, limit, stop-loss and take-profit orders.
    * Background monitoring of open orders to detect fills / cancellations.
    * Cancellation helpers (single order or all orders).
    """

    # Interval between order-status polls in the background monitor loop.
    _MONITOR_INTERVAL_SECONDS = 5

    def __init__(self, exchange: BaseExchange) -> None:
        self._exchange = exchange
        # Mapping of order_id → OrderTracker for every *active* order.
        self._orders: Dict[str, OrderTracker] = {}
        # Completed / cancelled orders kept for history.
        self._history: List[OrderTracker] = []
        self._lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API — order placement
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        type_: OrderType,
        amount: float,
        price: Optional[float] = None,
        strategy: str = "",
        **kwargs,
    ) -> Order:
        """Generic order placement; dispatches to the appropriate helper."""
        if type_ == OrderType.MARKET:
            return await self.place_market_order(symbol, side, amount, strategy, **kwargs)
        if type_ == OrderType.LIMIT:
            if price is None:
                raise ValueError("price is required for limit orders")
            return await self.place_limit_order(symbol, side, amount, price, strategy, **kwargs)
        if type_ == OrderType.STOP_LOSS:
            if price is None:
                raise ValueError("stop_price is required for stop-loss orders")
            return await self.place_stop_loss(symbol, side, amount, price, strategy)
        if type_ == OrderType.TAKE_PROFIT:
            if price is None:
                raise ValueError("tp_price is required for take-profit orders")
            return await self.place_take_profit(symbol, side, amount, price, strategy)
        raise ValueError(f"Unsupported order type: {type_}")

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        strategy: str = "",
        **kwargs,
    ) -> Order:
        """Place a market order and register it for tracking.

        If the exchange raises an error that suggests the order may have been
        partially filled (e.g. "partial" or "insufficient" in the message), the
        error is re-raised without retrying so the caller can inspect the actual
        filled amount before deciding how to proceed.  Auto-retry on partial
        fills is intentionally avoided to prevent double-entry.
        """
        try:
            order = await self._exchange.create_market_order(symbol, side, amount, kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "partial" in exc_str or "insufficient" in exc_str:
                logger.warning(
                    "Market order for {} may have been partially filled — "
                    "check order status before retrying ({})",
                    symbol,
                    exc,
                )
            raise
        await self._register(order, strategy=strategy)
        logger.info(
            "[{}] Market order {} placed: {} {} {}", strategy, order.id, side.value, amount, symbol
        )
        return order

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        strategy: str = "",
        **kwargs,
    ) -> Order:
        """Place a limit order and register it for tracking."""
        order = await self._exchange.create_limit_order(symbol, side, amount, price, kwargs)
        await self._register(order, strategy=strategy)
        logger.info(
            "[{}] Limit order {} placed: {} {} {} @ {}",
            strategy,
            order.id,
            side.value,
            amount,
            symbol,
            price,
        )
        return order

    async def place_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        strategy: str = "",
    ) -> Order:
        """Place a stop-loss order and register it for tracking."""
        order = await self._exchange.create_stop_loss_order(symbol, side, amount, stop_price)
        await self._register(order, strategy=strategy, is_stop_loss=True)
        logger.info(
            "[{}] Stop-loss {} placed: {} {} {} trigger={}",
            strategy,
            order.id,
            side.value,
            amount,
            symbol,
            stop_price,
        )
        return order

    async def place_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        strategy: str = "",
    ) -> Order:
        """Place a take-profit order and register it for tracking."""
        order = await self._exchange.create_take_profit_order(symbol, side, amount, tp_price)
        await self._register(order, strategy=strategy, is_take_profit=True)
        logger.info(
            "[{}] Take-profit {} placed: {} {} {} trigger={}",
            strategy,
            order.id,
            side.value,
            amount,
            symbol,
            tp_price,
        )
        return order

    # ------------------------------------------------------------------
    # Public API — order cancellation
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a single order by *order_id*.

        Returns *True* if the cancellation was accepted, *False* otherwise.
        """
        try:
            await self._exchange.cancel_order(order_id, symbol)
            async with self._lock:
                tracker = self._orders.pop(order_id, None)
                if tracker:
                    tracker.order.status = OrderStatus.CANCELED
                    self._history.append(tracker)
            logger.info("Order {} cancelled", order_id)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel order {}: {}", order_id, exc)
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all tracked open orders, optionally filtered by *symbol*.

        Returns the number of orders successfully cancelled.
        """
        async with self._lock:
            targets = [
                t for t in self._orders.values() if symbol is None or t.order.symbol == symbol
            ]

        cancelled = 0
        for tracker in targets:
            success = await self.cancel_order(tracker.order.id, tracker.order.symbol)
            if success:
                cancelled += 1
        logger.info("Cancelled {} order(s) for symbol={}", cancelled, symbol or "all")
        return cancelled

    # ------------------------------------------------------------------
    # Public API — status & queries
    # ------------------------------------------------------------------

    async def update_order_status(self, order_id: str) -> Order:
        """Fetch the latest status of *order_id* from the exchange and update the tracker."""
        async with self._lock:
            tracker = self._orders.get(order_id)
            if tracker is None:
                raise KeyError(f"Order {order_id} not tracked by OrderManager")
            symbol = tracker.order.symbol

        updated_order = await self._exchange.get_order(order_id, symbol)

        async with self._lock:
            tracker = self._orders.get(order_id)
            if tracker:
                tracker.order = updated_order
                if updated_order.status in (
                    OrderStatus.CLOSED,
                    OrderStatus.CANCELED,
                    OrderStatus.REJECTED,
                ):
                    self._history.append(self._orders.pop(order_id))
        return updated_order

    async def get_open_orders(self) -> List[OrderTracker]:
        """Return all currently tracked open orders."""
        async with self._lock:
            return list(self._orders.values())

    async def get_order_history(self) -> List[OrderTracker]:
        """Return all completed / cancelled orders seen since startup."""
        async with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    async def monitor_orders(self) -> None:
        """Background task: periodically poll the exchange for order fills.

        Call this once via ``asyncio.create_task(manager.monitor_orders())``.
        The task runs indefinitely until cancelled.
        """
        logger.info("OrderManager monitor started")
        while True:
            try:
                await asyncio.sleep(self._MONITOR_INTERVAL_SECONDS)
                async with self._lock:
                    order_ids = list(self._orders.keys())

                for order_id in order_ids:
                    try:
                        updated = await self.update_order_status(order_id)
                        if updated.status == OrderStatus.CLOSED:
                            logger.info("Order {} filled: {}", order_id, updated.symbol)
                        elif updated.status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
                            logger.info(
                                "Order {} terminal status: {}", order_id, updated.status.value
                            )
                    except KeyError:
                        pass  # already removed by a concurrent cancel
                    except Exception as exc:
                        logger.warning("Error updating order {}: {}", order_id, exc)

                # Honour auto-cancel deadlines
                await self._check_auto_cancels()

            except asyncio.CancelledError:
                logger.info("OrderManager monitor stopped")
                break
            except Exception as exc:
                logger.error("Unexpected error in order monitor: {}", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _register(
        self,
        order: Order,
        strategy: str = "",
        is_stop_loss: bool = False,
        is_take_profit: bool = False,
        auto_cancel_at: Optional[datetime] = None,
    ) -> None:
        tracker = OrderTracker(
            order=order,
            created_at=datetime.now(tz=timezone.utc),
            strategy=strategy,
            is_stop_loss=is_stop_loss,
            is_take_profit=is_take_profit,
            auto_cancel_at=auto_cancel_at,
        )
        async with self._lock:
            self._orders[order.id] = tracker

    async def _check_auto_cancels(self) -> None:
        """Cancel any orders past their *auto_cancel_at* deadline."""
        now = datetime.now(tz=timezone.utc)
        async with self._lock:
            expired = [
                t for t in self._orders.values() if t.auto_cancel_at and now >= t.auto_cancel_at
            ]
        for tracker in expired:
            logger.info("Auto-cancelling order {} (deadline reached)", tracker.order.id)
            await self.cancel_order(tracker.order.id, tracker.order.symbol)
