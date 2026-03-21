"""Paper trading exchange — simulates order fills without placing real orders.

State (virtual balance, positions, trade history) is persisted to
``data/paper_state.json`` so it survives bot restarts.

Real-time market prices are fetched via a read-only CCXT exchange client
(no API key required for public endpoints).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from .base_exchange import (
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
from utils.crypto_utils import calculate_liquidation_price

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Market-order slippage applied to fill price (0.05 %)
_SLIPPAGE_PCT: float = 0.0005

#: Taker fee applied to market order fills (0.1 %)
_TAKER_FEE_PCT: float = 0.001

#: Maker fee applied to limit order fills (0.1 %)
_MAKER_FEE_PCT: float = 0.001

#: Seconds between automatic SL/TP check cycles when not driven externally
_SL_TP_CHECK_INTERVAL: float = 5.0
#: REST polling interval for subscribe_ticker / subscribe_orderbook / subscribe_trades
_SUBSCRIBE_POLL_INTERVAL: float = 2.0

#: Floating-point epsilon for residual fill-quantity comparisons
_FILL_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# PaperExchange
# ---------------------------------------------------------------------------


class PaperExchange(BaseExchange):
    """Simulated exchange for paper trading.

    * Tracks a virtual USDT balance starting at *starting_balance*.
    * Fills market orders immediately at the real-time mid-price ± slippage.
    * Applies realistic taker/maker fees.
    * Simulates stop-loss and take-profit by periodically checking live prices.
    * Persists all state to ``data/paper_state.json``.

    Args:
        starting_balance: Starting virtual USDT balance (default 10 000 USDT).
        state_file: Path to the JSON state file.
        price_exchange: A connected :class:`~.base_exchange.BaseExchange`
            used *only* for read-only price queries.  Pass ``None`` to disable
            live-price simulation (prices will default to 0).
    """

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        state_file: str = "data/paper_state.json",
        price_exchange: Optional[BaseExchange] = None,
        local_orderbook_manager: Optional[Any] = None,
    ) -> None:
        super().__init__(api_key="paper", secret_key="paper")
        self._starting_balance = starting_balance
        self._state_file = Path(state_file)
        self._price_exchange = price_exchange
        self._local_orderbook_manager = local_orderbook_manager

        # In-memory state (will be overwritten by _load_state if file exists)
        self._usdt_balance: float = starting_balance
        self._positions: Dict[str, dict] = {}  # symbol → position dict
        self._open_orders: Dict[str, dict] = {}  # order_id → order dict
        self._trade_history: List[dict] = []
        self._order_counter: int = 0
        self._lock = asyncio.Lock()
        # Pending leverage per symbol — set via set_leverage() before position open
        self._pending_leverage: Dict[str, int] = {}  # symbol → leverage

        # Optional state-change callback (set by RealtimeHub for immediate push)
        self._on_state_change: Optional[Callable] = None

        # In-memory ticker price cache (updated by RealtimeHub via a setter)
        self._latest_tickers: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseExchange properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "paper"

    # ------------------------------------------------------------------
    # State-change callback (used by RealtimeHub for immediate push)
    # ------------------------------------------------------------------

    def set_state_change_callback(self, callback: Callable) -> None:
        """Register a callback that is called after any position/order state change.

        Args:
            callback: A zero-argument callable (sync or async will both work;
                      the paper exchange calls it via ``asyncio.ensure_future``
                      if it is a coroutine).
        """
        self._on_state_change = callback

    def update_ticker_cache(self, tickers: Dict[str, float]) -> None:
        """Update the internal ticker price cache used for uPnL calculations.

        Called by :class:`~core.realtime_hub.RealtimeHub` after each price
        refresh so that :meth:`_calculate_position_upnl` and
        :meth:`get_balance` use up-to-date prices without extra REST calls.

        Args:
            tickers: Mapping of ``symbol → last_price``.
        """
        self._latest_tickers.update(tickers)

    def _notify_state_change(self) -> None:
        """Call the registered state-change callback if set."""
        if self._on_state_change is None:
            return
        try:
            result = self._on_state_change()
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
        except Exception as exc:
            logger.debug("PaperExchange: state-change callback error: {}", exc)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Load persisted state and (optionally) connect the price exchange."""
        self._load_state()
        if self._price_exchange is not None:
            try:
                await self._price_exchange.connect()
                logger.info("PaperExchange: price feed connected via {}", self._price_exchange.name)
            except Exception as exc:
                logger.warning(
                    "PaperExchange: price feed connect failed ({}); prices may be 0", exc
                )
        logger.info(
            "PaperExchange ready — balance={:.2f} USDT positions={}",
            self._usdt_balance,
            len(self._positions),
        )

    async def disconnect(self) -> None:
        """Persist state and disconnect the price exchange."""
        self._save_state()
        if self._price_exchange is not None:
            try:
                await self._price_exchange.disconnect()
            except Exception:
                pass
        logger.info("PaperExchange disconnected — state saved to {}", self._state_file)

    # ------------------------------------------------------------------
    # Market data — delegate to price exchange (read-only)
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Return the current virtual balance including margin in use and unrealized PnL."""
        async with self._lock:
            usdt_free = self._usdt_balance
            positions_snapshot = {k: dict(v) for k, v in self._positions.items()}

        margin_in_use = sum(
            (pos.get("entry_price", 0) * pos.get("amount", 0))
            / max(pos.get("leverage", 1), 1)
            for pos in positions_snapshot.values()
        )
        unrealized_pnl = sum(
            self._calculate_position_upnl(symbol, pos)
            for symbol, pos in positions_snapshot.items()
        )
        equity = usdt_free + margin_in_use + unrealized_pnl

        logger.debug(
            "PaperExchange.get_balance: free={:.2f} margin={:.2f} upnl={:.2f} equity={:.2f}",
            usdt_free, margin_in_use, unrealized_pnl, equity,
        )
        return Balance(
            total={"USDT": equity},
            free={"USDT": usdt_free},
            used={"USDT": margin_in_use},
            usdt_total=equity,
            usdt_free=usdt_free,
        )

    def _calculate_position_upnl(self, symbol: str, pos: dict) -> float:
        """Compute unrealized PnL for a position dict using cached ticker prices.

        Args:
            symbol: The trading symbol (e.g. ``"BTC/USDT"``), used to look up
                the latest price from the hub's ticker cache.
            pos: A position dict from ``self._positions``.

        Returns:
            Unrealized PnL in USDT (positive = profit, negative = loss).
            Returns 0.0 when entry price or amount is invalid.
        """
        entry = float(pos.get("entry_price", 0))
        amount = float(pos.get("amount", 0))

        # Cannot compute PnL without valid entry price or amount
        if entry <= 0 or amount <= 0:
            return 0.0

        side_str = pos.get("side", "long")

        # Use cached ticker price if available, otherwise fall back to entry price
        current_price = self._latest_tickers.get(symbol, entry)
        if current_price <= 0:
            current_price = entry

        if side_str == "long":
            return (current_price - entry) * amount
        else:
            return (entry - current_price) * amount

    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch live ticker via the price exchange."""
        if self._price_exchange is None:
            logger.warning(
                "PaperExchange: no price exchange — returning synthetic ticker for {}", symbol
            )
            ts = int(time.time() * 1000)
            return Ticker(
                symbol=symbol,
                bid=0.0,
                ask=0.0,
                last=0.0,
                high=0.0,
                low=0.0,
                volume=0.0,
                timestamp=ts,
            )
        return await self._price_exchange.get_ticker(symbol)

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch live order book via the price exchange."""
        if self._price_exchange is None:
            return {"bids": [], "asks": [], "symbol": symbol, "timestamp": int(time.time() * 1000)}
        return await self._price_exchange.get_orderbook(symbol, limit)

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles via the price exchange."""
        if self._price_exchange is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return await self._price_exchange.get_ohlcv(symbol, timeframe, limit)

    async def watch_order_book(self, symbol: str, limit: int = None, params={}):
        """Return a live order book snapshot, falling back to REST polling.

        Tries the price-exchange's WebSocket ``watch_order_book`` first; on
        failure (or when no price exchange is configured) falls back to a
        REST ``get_orderbook`` call.  Never raises ``NotImplementedError``.
        """
        if self._price_exchange is not None:
            try:
                return await self._price_exchange.watch_order_book(symbol, limit, params)
            except Exception:
                pass
            # WS unavailable — fall back to REST snapshot
            try:
                return await self._price_exchange.get_orderbook(symbol, limit or 20)
            except Exception as exc:
                logger.debug(
                    "PaperExchange.watch_order_book: REST fallback failed for {}: {}", symbol, exc
                )
        # No price exchange — return synthetic empty book
        return {"bids": [], "asks": [], "symbol": symbol, "timestamp": int(time.time() * 1000)}

    async def watch_ticker(self, symbol: str, params={}):
        """Return a live ticker snapshot, falling back to REST polling.

        Tries the price-exchange's WebSocket ``watch_ticker`` first; on
        failure falls back to a REST ``get_ticker`` call.  Never raises
        ``NotImplementedError``.
        """
        if self._price_exchange is not None:
            try:
                return await self._price_exchange.watch_ticker(symbol, params)
            except Exception:
                pass
            # WS unavailable — fall back to REST snapshot
            try:
                return await self._price_exchange.get_ticker(symbol)
            except Exception as exc:
                logger.debug(
                    "PaperExchange.watch_ticker: REST fallback failed for {}: {}", symbol, exc
                )
        # No price exchange — return synthetic ticker
        ts = int(time.time() * 1000)
        return Ticker(
            symbol=symbol, bid=0.0, ask=0.0, last=0.0,
            high=0.0, low=0.0, volume=0.0, timestamp=ts,
        )

    async def watch_ohlcv(self, symbol: str, timeframe: str = "1m"):
        """Return a live OHLCV snapshot, delegating to the price exchange.

        Falls back to ``get_ohlcv`` when the WebSocket method is unavailable.
        Never raises ``NotImplementedError``.
        """
        if self._price_exchange is not None:
            if hasattr(self._price_exchange, "watch_ohlcv"):
                try:
                    return await self._price_exchange.watch_ohlcv(symbol, timeframe)
                except Exception:
                    pass
            try:
                return await self._price_exchange.get_ohlcv(symbol, timeframe)
            except Exception as exc:
                logger.debug(
                    "PaperExchange.watch_ohlcv: REST fallback failed for {} {}: {}",
                    symbol, timeframe, exc,
                )
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # ------------------------------------------------------------------
    # Order management — simulated
    # ------------------------------------------------------------------

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Simulate a market order fill with realistic orderbook impact.

        When a local (or REST) order book is available the fill price is
        computed as the Volume-Weighted Average Price (VWAP) of the consumed
        levels — walking *asks* for buys and *bids* for sells.  When no book
        is available the method falls back to the previous mid-price ± slippage
        behaviour.
        """
        # Resolve order book: local cache → REST → None
        book = self._get_best_available_book_sync(symbol)
        if book is None and self._price_exchange is not None:
            try:
                book = await self._price_exchange.get_orderbook(symbol, limit=20)
            except Exception as exc:
                logger.debug("PaperExchange: REST orderbook fallback failed for {}: {}", symbol, exc)

        if book is not None:
            vwap_price = self._walk_book_vwap(book, side, amount)
        else:
            vwap_price = 0.0

        # Get the current price for sanity checks regardless of fill method
        price = await self._get_live_price(symbol)

        if vwap_price > 0:
            fill_price = vwap_price
        else:
            # Fallback: mid-price + fixed slippage
            fill_price = self._apply_slippage(price, side)

        fee = fill_price * amount * _TAKER_FEE_PCT
        cost = fill_price * amount

        reduce_only = params.get("reduceOnly", False)

        # Sanity check: reject orders whose notional value exceeds 50% of account balance.
        # This guards against unit-conversion bugs (e.g. passing USDT amount as base units).
        # We use the raw price (before slippage) so legitimate large orders aren't affected
        # by minor slippage pushing the cost fractionally over the threshold.
        async with self._lock:
            current_balance = self._usdt_balance
        if not reduce_only and current_balance > 0:
            notional_at_mid = price * amount
            max_notional = current_balance * 0.50
            if notional_at_mid > max_notional:
                logger.error(
                    "PaperExchange: order notional {:.2f} USDT exceeds 50% of balance "
                    "({:.2f} USDT) for {} — order rejected to prevent runaway loss",
                    notional_at_mid,
                    current_balance,
                    symbol,
                )
                raise ValueError(
                    f"Order notional {notional_at_mid:.2f} USDT exceeds sanity limit "
                    f"({max_notional:.2f} USDT = 50% of balance). "
                    "Ensure position sizes are in base currency units, not USDT."
                )

        async with self._lock:
            if reduce_only:
                order = self._close_position_internal(symbol, side, amount, fill_price, fee)
            else:
                order = await self._open_position_internal(
                    symbol, side, amount, fill_price, fee, cost
                )

        self._save_state()
        self._notify_state_change()
        logger.info(
            "PaperExchange market order filled: {} {} {} @ {:.4f} fee={:.4f}",
            side.value,
            amount,
            symbol,
            fill_price,
            fee,
        )
        return order

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Register a limit order with realistic queue-position simulation.

        **Immediate fill**: If the market is already past the limit price at
        the time of submission the order fills instantly (same as before).

        **Queue simulation**: Otherwise the order is stored as pending and
        will only fill when *either*:

        1. The real-time ticker trades *through* the limit price (price
           crosses the level), **or**
        2. The ticker is *at* the limit price (within 0.5 %) **and** the
           order has been sitting in the queue long enough.  The required
           queue wait is proportional to the order size — larger orders have
           lower queue priority (they sit behind accumulated depth).
        """
        current_price = await self._get_live_price(symbol)
        # Determine if immediately fillable
        if side == OrderSide.BUY and current_price <= price:
            fill_price: Optional[float] = price
        elif side == OrderSide.SELL and current_price >= price:
            fill_price = price
        else:
            fill_price = None  # Pending

        order_id = self._next_order_id()
        ts = int(time.time() * 1000)

        if fill_price is not None:
            fee = fill_price * amount * _MAKER_FEE_PCT
            cost = fill_price * amount
            reduce_only = params.get("reduceOnly", False)
            async with self._lock:
                if reduce_only:
                    order = self._close_position_internal(symbol, side, amount, fill_price, fee)
                else:
                    order = await self._open_position_internal(
                        symbol, side, amount, fill_price, fee, cost
                    )
                order.id = order_id
        else:
            # Store as a pending open order with queue-simulation metadata
            queue_wait = self._estimate_queue_wait(amount, price)
            pending: dict = {
                "id": order_id,
                "symbol": symbol,
                "type": "limit",
                "side": side.value,
                "amount": amount,
                "price": price,
                "filled": 0.0,
                "remaining": amount,
                "status": "open",
                "timestamp": ts,
                "fee": 0.0,
                "params": params,
                # Queue simulation: time at which the order may fill if price is at level
                "queue_fill_time": time.time() + queue_wait,
            }
            async with self._lock:
                self._open_orders[order_id] = pending
            order = self._dict_to_order(pending)
            logger.info(
                "PaperExchange limit order queued: {} {} {} @ {:.4f} (queue_wait={:.1f}s)",
                side.value,
                amount,
                symbol,
                price,
                queue_wait,
            )

        self._save_state()
        return order

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Register a stop-loss trigger order."""
        return await self._register_trigger_order(
            symbol, side, amount, stop_price, OrderType.STOP_LOSS, params
        )

    async def create_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        tp_price: float,
        params: Dict[str, Any] = {},
    ) -> Order:
        """Register a take-profit trigger order."""
        return await self._register_trigger_order(
            symbol, side, amount, tp_price, OrderType.TAKE_PROFIT, params
        )

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a pending (unfilled) order."""
        async with self._lock:
            order = self._open_orders.pop(order_id, None)
        if order is None:
            logger.warning("PaperExchange cancel_order: order {} not found", order_id)
            return {"id": order_id, "status": "not_found"}
        order["status"] = "canceled"
        self._save_state()
        self._notify_state_change()
        logger.info("PaperExchange order {} cancelled", order_id)
        return order

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all pending orders for *symbol*."""
        async with self._lock:
            to_cancel = [o for o in self._open_orders.values() if o["symbol"] == symbol]
            for o in to_cancel:
                o["status"] = "canceled"
                self._open_orders.pop(o["id"], None)
        self._save_state()
        self._notify_state_change()
        logger.info("PaperExchange cancelled {} orders for {}", len(to_cancel), symbol)
        return to_cancel

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Fetch the current state of a paper order."""
        async with self._lock:
            raw = self._open_orders.get(order_id)
        if raw is None:
            # Check trade history for filled orders
            for trade in self._trade_history:
                if trade.get("order_id") == order_id:
                    return Order(
                        id=order_id,
                        symbol=symbol,
                        type=OrderType.MARKET,
                        side=OrderSide(trade.get("side", "buy")),
                        amount=float(trade.get("amount", 0)),
                        price=float(trade.get("fill_price", 0)),
                        filled=float(trade.get("amount", 0)),
                        remaining=0.0,
                        status=OrderStatus.CLOSED,
                        timestamp=int(trade.get("timestamp", 0)),
                        fee=float(trade.get("fee", 0)),
                    )
            raise KeyError(f"PaperExchange: order {order_id} not found")
        return self._dict_to_order(raw)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return all pending open orders, optionally filtered by *symbol*."""
        async with self._lock:
            orders = list(self._open_orders.values())
        if symbol is not None:
            orders = [o for o in orders if o["symbol"] == symbol]
        return [self._dict_to_order(o) for o in orders]

    async def get_trade_history(self) -> List[dict]:
        """Return all simulated trade records."""
        async with self._lock:
            return list(self._trade_history)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for *symbol* (updates existing position metadata if any)."""
        async with self._lock:
            self._pending_leverage[symbol] = leverage
            if symbol in self._positions:
                self._positions[symbol]["leverage"] = leverage
        logger.info("PaperExchange leverage set to {}x for {}", leverage, symbol)
        return {"symbol": symbol, "leverage": leverage}

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> Dict[str, Any]:
        """No-op for paper trading — margin type is logged only."""
        logger.info("PaperExchange margin type set to {} for {} (no-op)", margin_type.value, symbol)
        return {"symbol": symbol, "margin_type": margin_type.value}

    async def get_positions(self) -> List[Position]:
        """Return all currently open virtual positions."""
        live_prices = {}
        async with self._lock:
            symbols = list(self._positions.keys())

        for sym in symbols:
            try:
                ticker = await self.get_ticker(sym)
                live_prices[sym] = ticker.last
            except Exception:
                live_prices[sym] = 0.0

        async with self._lock:
            positions = []
            for sym, pos in self._positions.items():
                current_price = live_prices.get(sym, pos.get("entry_price", 0.0))
                side = PositionSide(pos.get("side", "long"))
                amount = float(pos.get("amount", 0))
                entry = float(pos.get("entry_price", 0))
                leverage = int(pos.get("leverage", 1))

                if side == PositionSide.LONG:
                    unrealized_pnl = (current_price - entry) * amount
                else:
                    unrealized_pnl = (entry - current_price) * amount

                margin = entry * amount / leverage if leverage > 0 else 0.0
                position_value = current_price * amount
                roe_pct = (unrealized_pnl / margin * 100.0) if margin > 0 else 0.0

                liq_side = "buy" if side == PositionSide.LONG else "sell"
                liquidation_price = (
                    calculate_liquidation_price(entry, leverage, liq_side)
                    if leverage > 0 and entry > 0
                    else 0.0
                )

                positions.append(
                    Position(
                        symbol=sym,
                        side=side,
                        amount=amount,
                        entry_price=entry,
                        current_price=current_price,
                        mark_price=current_price,
                        unrealized_pnl=unrealized_pnl,
                        leverage=leverage,
                        margin=margin,
                        liquidation_price=liquidation_price,
                        position_value=position_value,
                        roe_pct=roe_pct,
                        timestamp=int(pos.get("opened_at", time.time() * 1000)),
                    )
                )
        return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open virtual position for *symbol*, or *None* if flat."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_positions_with_live_prices(self, tickers: Dict[str, float]) -> List[Position]:
        """Return all open positions using pre-fetched ticker prices (no REST calls).

        This avoids the N+1 query problem by using prices already cached in the
        RealtimeHub instead of fetching each symbol individually.

        Args:
            tickers: Mapping of symbol → latest last price from the hub's cache.

        Returns:
            List of :class:`Position` objects with updated P&L computed from
            the provided ticker prices.
        """
        positions: List[Position] = []
        # Take a snapshot of positions without holding the lock for the whole loop
        with_pos = {k: dict(v) for k, v in self._positions.items()}

        for sym, pos in with_pos.items():
            current_price = tickers.get(sym, float(pos.get("entry_price", 0)))
            side = PositionSide(pos.get("side", "long"))
            amount = float(pos.get("amount", 0))
            entry = float(pos.get("entry_price", 0))
            leverage = int(pos.get("leverage", 1))

            if side == PositionSide.LONG:
                unrealized_pnl = (current_price - entry) * amount
            else:
                unrealized_pnl = (entry - current_price) * amount

            margin = entry * amount / leverage if leverage > 0 else 0.0
            position_value = current_price * amount
            roe_pct = (unrealized_pnl / margin * 100.0) if margin > 0 else 0.0

            liq_side = "buy" if side == PositionSide.LONG else "sell"
            liquidation_price = (
                calculate_liquidation_price(entry, leverage, liq_side)
                if leverage > 0 and entry > 0
                else 0.0
            )

            positions.append(
                Position(
                    symbol=sym,
                    side=side,
                    amount=amount,
                    entry_price=entry,
                    current_price=current_price,
                    mark_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation_price,
                    position_value=position_value,
                    roe_pct=roe_pct,
                    timestamp=int(pos.get("opened_at", time.time() * 1000)),
                )
            )
        return positions

    async def update_stop_loss(self, symbol: str, new_price: float) -> Order:
        """Cancel existing SL order(s) for *symbol* and create a new one at *new_price*."""
        async with self._lock:
            pos = self._positions.get(symbol)
        if pos is None:
            raise ValueError(f"PaperExchange: no open position for {symbol}")
        # Cancel all existing stop-loss orders for this symbol
        async with self._lock:
            sl_ids = [
                oid
                for oid, o in self._open_orders.items()
                if o.get("symbol") == symbol and o.get("type") == OrderType.STOP_LOSS.value
            ]
            for oid in sl_ids:
                self._open_orders.pop(oid, None)
        if sl_ids:
            self._save_state()
            logger.info("PaperExchange: cancelled {} SL order(s) for {}", len(sl_ids), symbol)
        # Place new stop-loss
        pos_side = PositionSide(pos.get("side", "long"))
        close_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
        amount = float(pos.get("amount", 0))
        return await self.create_stop_loss_order(symbol, close_side, amount, new_price)

    async def update_take_profit(self, symbol: str, new_price: float) -> Order:
        """Cancel existing TP order(s) for *symbol* and create a new one at *new_price*."""
        async with self._lock:
            pos = self._positions.get(symbol)
        if pos is None:
            raise ValueError(f"PaperExchange: no open position for {symbol}")
        # Cancel all existing take-profit orders for this symbol
        async with self._lock:
            tp_ids = [
                oid
                for oid, o in self._open_orders.items()
                if o.get("symbol") == symbol and o.get("type") == OrderType.TAKE_PROFIT.value
            ]
            for oid in tp_ids:
                self._open_orders.pop(oid, None)
        if tp_ids:
            self._save_state()
            logger.info("PaperExchange: cancelled {} TP order(s) for {}", len(tp_ids), symbol)
        # Place new take-profit
        pos_side = PositionSide(pos.get("side", "long"))
        close_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
        amount = float(pos.get("amount", 0))
        return await self.create_take_profit_order(symbol, close_side, amount, new_price)

    async def reset_paper_state(self, starting_balance: Optional[float] = None) -> dict:
        """Close all positions, cancel all orders, and reset to a fresh state.

        Args:
            starting_balance: New starting balance.  Defaults to the original
                starting balance passed to ``__init__``.

        Returns:
            dict with ``balance`` and ``message``.
        """
        balance = starting_balance if starting_balance is not None else self._starting_balance
        async with self._lock:
            self._positions.clear()
            self._open_orders.clear()
            self._trade_history.clear()
            self._order_counter = 0
            self._usdt_balance = balance
        # Remove the state file so it starts fresh on next connect
        try:
            if self._state_file.exists():
                self._state_file.unlink()
        except Exception as exc:
            logger.warning("PaperExchange: could not delete state file: {}", exc)
        self._save_state()
        logger.info("PaperExchange: paper state reset — new balance={:.2f} USDT", balance)
        return {"balance": balance, "message": "Paper trading state reset successfully"}

    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Order:
        """Close the virtual position for *symbol* at the current live price."""
        async with self._lock:
            pos = self._positions.get(symbol)
        if pos is None:
            raise ValueError(f"PaperExchange: no open position for {symbol}")

        pos_side = PositionSide(pos.get("side", "long"))
        close_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
        close_amount = amount if amount is not None else float(pos.get("amount", 0))

        return await self.create_market_order(
            symbol, close_side, close_amount, {"reduceOnly": True}
        )

    async def modify_leverage(self, symbol: str, new_leverage: int) -> Dict[str, Any]:
        """Change the leverage for an existing position in paper trading.

        Recalculates the margin and liquidation price for the position.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            new_leverage: New leverage multiplier (must be >= 1).

        Returns:
            dict with updated position metadata.
        """
        if new_leverage < 1:
            raise ValueError(f"leverage must be >= 1, got {new_leverage}")
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                raise ValueError(f"PaperExchange: no open position for {symbol}")
            entry = float(pos.get("entry_price", 0))
            amount = float(pos.get("amount", 0))
            side_str = pos.get("side", "long")
            pos["leverage"] = new_leverage
            new_margin = entry * amount / new_leverage if new_leverage > 0 else 0.0
            liq_side = "buy" if side_str == "long" else "sell"
            new_liq = (
                calculate_liquidation_price(entry, new_leverage, liq_side)
                if new_leverage > 0 and entry > 0
                else 0.0
            )
        self._save_state()
        self._notify_state_change()
        logger.info("PaperExchange: leverage for {} updated to {}x", symbol, new_leverage)
        return {
            "symbol": symbol,
            "leverage": new_leverage,
            "margin": round(new_margin, 4),
            "liquidation_price": round(new_liq, 4),
        }

    async def add_margin(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Add extra margin to an isolated position in paper trading.

        Deducts *amount* USDT from the free balance and credits it to the
        position's margin, which lowers the liquidation price.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            amount: USDT amount to add as margin (must be > 0).

        Returns:
            dict with updated position metadata.
        """
        if amount <= 0:
            raise ValueError(f"add_margin amount must be > 0, got {amount}")
        async with self._lock:
            if self._usdt_balance < amount:
                raise ValueError(
                    f"Insufficient balance: have {self._usdt_balance:.2f} USDT, need {amount:.2f}"
                )
            pos = self._positions.get(symbol)
            if pos is None:
                raise ValueError(f"PaperExchange: no open position for {symbol}")
            self._usdt_balance -= amount
            current_margin = float(pos.get("extra_margin", 0.0))
            pos["extra_margin"] = current_margin + amount
            entry = float(pos.get("entry_price", 0))
            amt = float(pos.get("amount", 0))
            leverage = int(pos.get("leverage", 1))
            side_str = pos.get("side", "long")
            base_margin = entry * amt / leverage if leverage > 0 else 0.0
            total_margin = base_margin + pos["extra_margin"]
            # Effective leverage decreases as extra margin is added
            effective_leverage = (entry * amt / total_margin) if total_margin > 0 else 1
            liq_side = "buy" if side_str == "long" else "sell"
            new_liq = (
                calculate_liquidation_price(entry, effective_leverage, liq_side)
                if effective_leverage > 0 and entry > 0
                else 0.0
            )
        self._save_state()
        self._notify_state_change()
        logger.info("PaperExchange: added {:.4f} USDT margin to {}", amount, symbol)
        return {
            "symbol": symbol,
            "added_margin": round(amount, 4),
            "total_margin": round(total_margin, 4),
            "liquidation_price": round(new_liq, 4),
        }

    async def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return comprehensive position data for *symbol*.

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.

        Returns:
            dict with full position details, or ``None`` if no position exists.
        """
        async with self._lock:
            pos = self._positions.get(symbol)
        if pos is None:
            return None

        try:
            ticker = await self.get_ticker(symbol)
            current_price = ticker.last
        except Exception:
            current_price = float(pos.get("entry_price", 0.0))

        side_str = pos.get("side", "long")
        side = PositionSide(side_str)
        amount = float(pos.get("amount", 0))
        entry = float(pos.get("entry_price", 0))
        leverage = int(pos.get("leverage", 1))
        extra_margin = float(pos.get("extra_margin", 0.0))

        if side == PositionSide.LONG:
            unrealized_pnl = (current_price - entry) * amount
        else:
            unrealized_pnl = (entry - current_price) * amount

        base_margin = entry * amount / leverage if leverage > 0 else 0.0
        margin = base_margin + extra_margin
        position_value = current_price * amount
        roe_pct = (unrealized_pnl / margin * 100.0) if margin > 0 else 0.0
        margin_ratio = (margin / position_value * 100.0) if position_value > 0 else 0.0

        liq_side = "buy" if side == PositionSide.LONG else "sell"
        effective_leverage = (entry * amount / margin) if margin > 0 else leverage
        liquidation_price = (
            calculate_liquidation_price(entry, effective_leverage, liq_side)
            if effective_leverage > 0 and entry > 0
            else 0.0
        )

        funding_rate = await self.get_funding_rate(symbol)

        # Collect attached open orders (SL/TP) for this symbol
        async with self._lock:
            attached_orders = [
                {
                    "id": oid,
                    "type": o.get("type"),
                    "side": o.get("side"),
                    "amount": o.get("amount"),
                    "price": o.get("price"),
                    "status": o.get("status"),
                }
                for oid, o in self._open_orders.items()
                if o.get("symbol") == symbol
            ]

        opened_at = pos.get("opened_at", time.time() * 1000)
        time_open_secs = time.time() - (opened_at / 1000.0)

        return {
            "symbol": symbol,
            "side": side_str,
            "amount": round(amount, 8),
            "entry_price": round(entry, 4),
            "mark_price": round(current_price, 4),
            "current_price": round(current_price, 4),
            "liquidation_price": round(liquidation_price, 4),
            "margin": round(margin, 4),
            "margin_ratio": round(margin_ratio, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "roe_pct": round(roe_pct, 4),
            "position_value": round(position_value, 4),
            "leverage": leverage,
            "funding_rate": funding_rate,
            "open_orders": attached_orders,
            "time_open": round(time_open_secs),
            "strategy": pos.get("strategy", ""),
        }

    # ------------------------------------------------------------------
    # Derivatives-specific data — delegate or return defaults
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        if self._price_exchange is not None:
            try:
                return await self._price_exchange.get_funding_rate(symbol)
            except Exception:
                pass
        return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        if self._price_exchange is not None:
            try:
                return await self._price_exchange.get_open_interest(symbol)
            except Exception:
                pass
        return 0.0

    # ------------------------------------------------------------------
    # WebSocket subscriptions — no-op for paper trading
    # ------------------------------------------------------------------

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker updates via REST polling (paper mode).

        Polls :meth:`get_ticker` every 2 seconds and invokes *callback* with
        the resulting :class:`~.base_exchange.Ticker` object.  Runs until the
        enclosing task is cancelled.
        """
        if self._price_exchange is None:
            logger.warning("PaperExchange: subscribe_ticker not possible — no price exchange")
            return
        logger.info("PaperExchange: starting REST ticker polling for {}", symbol)
        while True:
            try:
                ticker = await self._price_exchange.get_ticker(symbol)
                await callback(ticker)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("PaperExchange: ticker poll error for {}: {}", symbol, exc)
            await asyncio.sleep(_SUBSCRIBE_POLL_INTERVAL)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to order book updates via REST polling (paper mode).

        Polls :meth:`get_orderbook` every 2 seconds and invokes *callback* with
        the raw order-book dict.  Runs until the enclosing task is cancelled.
        """
        if self._price_exchange is None:
            logger.warning("PaperExchange: subscribe_orderbook not possible — no price exchange")
            return
        logger.info("PaperExchange: starting REST order-book polling for {}", symbol)
        while True:
            try:
                book = await self._price_exchange.get_orderbook(symbol, limit=20)
                await callback(book)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("PaperExchange: order-book poll error for {}: {}", symbol, exc)
            await asyncio.sleep(_SUBSCRIBE_POLL_INTERVAL)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to public trade feed via REST polling (paper mode).

        Uses the latest ticker price as a proxy for recent trade data, polling
        every 2 seconds.  Runs until the enclosing task is cancelled.
        """
        if self._price_exchange is None:
            logger.warning("PaperExchange: subscribe_trades not possible — no price exchange")
            return
        logger.info("PaperExchange: starting REST trades polling for {}", symbol)
        while True:
            try:
                ticker = await self._price_exchange.get_ticker(symbol)
                trade = {
                    "symbol": symbol,
                    "price": ticker.last,
                    "amount": 0.0,
                    "side": "buy",
                    "timestamp": ticker.timestamp,
                }
                await callback(trade)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("PaperExchange: trades poll error for {}: {}", symbol, exc)
            await asyncio.sleep(_SUBSCRIBE_POLL_INTERVAL)

    async def subscribe_user_data(self, callback: Callable) -> None:
        logger.warning("PaperExchange: subscribe_user_data not implemented")

    # ------------------------------------------------------------------
    # SL/TP simulation loop
    # ------------------------------------------------------------------

    async def run_sl_tp_monitor(self) -> None:
        """Background task that checks SL/TP conditions and pending limit orders.

        Call via ``asyncio.create_task(paper_exchange.run_sl_tp_monitor())``.
        """
        logger.info("PaperExchange SL/TP monitor started")
        while True:
            try:
                await asyncio.sleep(_SL_TP_CHECK_INTERVAL)
                await self._check_trigger_orders()
                await self._check_pending_limit_orders()
            except asyncio.CancelledError:
                logger.info("PaperExchange SL/TP monitor stopped")
                break
            except Exception as exc:
                logger.error("PaperExchange SL/TP monitor error: {}", exc)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load paper state from JSON file if it exists."""
        if not self._state_file.exists():
            logger.info(
                "PaperExchange: no state file found at {}; starting fresh with {:.2f} USDT",
                self._state_file,
                self._starting_balance,
            )
            return
        try:
            with self._state_file.open("r") as fh:
                state = json.load(fh)
            self._usdt_balance = float(state.get("usdt_balance", self._starting_balance))
            self._positions = state.get("positions", {})
            self._open_orders = state.get("open_orders", {})
            self._trade_history = state.get("trade_history", [])
            self._order_counter = int(state.get("order_counter", 0))
            logger.info(
                "PaperExchange state loaded from {}: balance={:.2f} USDT positions={} trades={}",
                self._state_file,
                self._usdt_balance,
                len(self._positions),
                len(self._trade_history),
            )
        except Exception as exc:
            logger.error(
                "PaperExchange: failed to load state from {}: {} — starting fresh",
                self._state_file,
                exc,
            )

    def _save_state(self) -> None:
        """Persist current paper state to JSON file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "usdt_balance": self._usdt_balance,
                "positions": self._positions,
                "open_orders": self._open_orders,
                "trade_history": self._trade_history,
                "order_counter": self._order_counter,
                "saved_at": time.time(),
            }
            with self._state_file.open("w") as fh:
                json.dump(state, fh, indent=2, default=str)
            logger.debug("PaperExchange state saved to {}", self._state_file)
        except Exception as exc:
            logger.error("PaperExchange: failed to save state: {}", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"paper-{self._order_counter}-{uuid.uuid4().hex[:8]}"

    async def _get_live_price(self, symbol: str) -> float:
        """Return the current mid-price for *symbol* via the price exchange."""
        if self._price_exchange is None:
            return 0.0
        try:
            ticker = await self._price_exchange.get_ticker(symbol)
            mid = (ticker.bid + ticker.ask) / 2.0 if ticker.bid and ticker.ask else ticker.last
            return mid if mid else 0.0
        except Exception as exc:
            logger.warning("PaperExchange: failed to fetch price for {}: {}", symbol, exc)
            return 0.0

    @staticmethod
    def _apply_slippage(price: float, side: OrderSide) -> float:
        """Apply market-order slippage to *price*."""
        if side == OrderSide.BUY:
            return price * (1 + _SLIPPAGE_PCT)
        return price * (1 - _SLIPPAGE_PCT)

    async def _open_position_internal(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        fill_price: float,
        fee: float,
        cost: float,
    ) -> Order:
        """Open (or add to) a virtual position.  Must be called while *_lock* is held."""
        pos_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT

        # Deduct cost + fee from balance
        self._usdt_balance -= cost + fee

        # Resolve leverage: use any pending leverage set via set_leverage(), default to 1
        leverage = self._pending_leverage.get(symbol, 1)

        if symbol not in self._positions:
            margin = float(amount) * fill_price / leverage if leverage > 0 else 0.0
            self._positions[symbol] = {
                "side": pos_side.value,
                "amount": float(amount),
                "entry_price": fill_price,
                "leverage": leverage,
                "opened_at": int(time.time() * 1000),
                # CCXT-standard margin/position keys
                "contracts": float(amount),
                "entryPrice": fill_price,
                "initialMargin": margin,
                "collateral": margin,
            }
        else:
            # Average-in to existing position
            existing = self._positions[symbol]
            existing_amount = float(existing["amount"])
            existing_entry = float(existing["entry_price"])
            total_amount = existing_amount + float(amount)
            avg_entry = (existing_entry * existing_amount + fill_price * float(amount)) / total_amount
            existing["amount"] = total_amount
            existing["entry_price"] = avg_entry
            # Update CCXT-standard keys to reflect the averaged position
            new_margin = total_amount * avg_entry / leverage if leverage > 0 else 0.0
            existing["contracts"] = total_amount
            existing["entryPrice"] = avg_entry
            existing["initialMargin"] = new_margin
            existing["collateral"] = new_margin

        order_id = self._next_order_id()
        ts = int(time.time() * 1000)
        self._trade_history.append(
            {
                "order_id": order_id,
                "symbol": symbol,
                "side": side.value,
                "amount": amount,
                "fill_price": fill_price,
                "fee": fee,
                "timestamp": ts,
                "type": "open",
            }
        )
        return Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side=side,
            amount=amount,
            price=fill_price,
            filled=amount,
            remaining=0.0,
            status=OrderStatus.CLOSED,
            timestamp=ts,
            fee=fee,
        )

    def _close_position_internal(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        fill_price: float,
        fee: float,
    ) -> Order:
        """Close (or reduce) a virtual position.  Must be called while *_lock* is held."""
        pos = self._positions.get(symbol)
        order_id = self._next_order_id()
        ts = int(time.time() * 1000)

        if pos is None:
            logger.warning("PaperExchange: close_position called but no position for {}", symbol)
        else:
            pos_side = PositionSide(pos["side"])
            entry_price = float(pos["entry_price"])
            pos_amount = float(pos["amount"])
            close_amount = min(amount, pos_amount)

            if pos_side == PositionSide.LONG:
                pnl = (fill_price - entry_price) * close_amount
            else:
                pnl = (entry_price - fill_price) * close_amount

            # Realise PnL, return margin, deduct fee
            realised = entry_price * close_amount + pnl - fee
            self._usdt_balance += realised

            logger.info(
                "PaperExchange position closed: {} {} amount={} pnl={:.4f} balance={:.2f}",
                symbol,
                pos_side.value,
                close_amount,
                pnl,
                self._usdt_balance,
            )

            if pos_amount <= close_amount:
                del self._positions[symbol]
            else:
                pos["amount"] = pos_amount - close_amount

            self._trade_history.append(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side.value,
                    "amount": close_amount,
                    "fill_price": fill_price,
                    "fee": fee,
                    "pnl": pnl,
                    "timestamp": ts,
                    "type": "close",
                }
            )

        return Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side=side,
            amount=amount,
            price=fill_price,
            filled=amount,
            remaining=0.0,
            status=OrderStatus.CLOSED,
            timestamp=ts,
            fee=fee,
        )

    async def _register_trigger_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        trigger_price: float,
        order_type: OrderType,
        params: Dict[str, Any],
    ) -> Order:
        """Register a stop-loss or take-profit trigger order."""
        order_id = self._next_order_id()
        ts = int(time.time() * 1000)
        pending: dict = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type.value,
            "side": side.value,
            "amount": amount,
            "price": trigger_price,
            "filled": 0.0,
            "remaining": amount,
            "status": "open",
            "timestamp": ts,
            "fee": 0.0,
            "is_trigger": True,
            "trigger_price": trigger_price,
            "params": params,
        }
        async with self._lock:
            self._open_orders[order_id] = pending
        self._save_state()
        logger.info(
            "PaperExchange trigger order registered: {} {} {} {} trigger={}",
            order_type.value,
            side.value,
            amount,
            symbol,
            trigger_price,
        )
        return self._dict_to_order(pending)

    @staticmethod
    def _walk_book_vwap(
        book: Dict[str, Any],
        side: OrderSide,
        amount: float,
    ) -> float:
        """Walk the order book to fill *amount* and return the VWAP fill price.

        Iterates through asks (for buys) or bids (for sells), consuming
        volume at each price level until the full order size is filled.
        Returns the Volume-Weighted Average Price of the consumed levels.

        If the book does not have enough depth the remaining size is filled
        at the last available level price (market-impact assumption).

        Args:
            book: Order book dict with ``bids`` and ``asks`` lists.
            side: ``OrderSide.BUY`` or ``OrderSide.SELL``.
            amount: Order size in base currency units.

        Returns:
            VWAP fill price, or 0.0 when the book has no usable levels.
        """
        levels = book.get("asks", []) if side == OrderSide.BUY else book.get("bids", [])
        if not levels:
            return 0.0

        remaining = amount
        total_cost = 0.0
        total_filled = 0.0
        last_price = 0.0

        for level in levels:
            try:
                level_price = float(level[0])
                level_size = float(level[1])
            except (IndexError, ValueError, TypeError):
                continue
            if level_price <= 0 or level_size <= 0:
                continue

            fill_qty = min(remaining, level_size)
            total_cost += fill_qty * level_price
            total_filled += fill_qty
            last_price = level_price
            remaining -= fill_qty
            if remaining <= _FILL_EPSILON:
                break

        if total_filled <= 0:
            return 0.0

        # Fill any remaining size at the last level price (market impact)
        if remaining > _FILL_EPSILON and last_price > 0:
            total_cost += remaining * last_price
            total_filled += remaining

        return total_cost / total_filled

    @staticmethod
    def _estimate_queue_wait(amount: float, price: float) -> float:
        """Return the simulated queue-wait in seconds for a limit order.

        Larger orders have lower queue priority and must wait longer before
        they can be filled at the limit price.  The formula scales linearly
        with order *notional value* (amount × price) up to a cap.

        Args:
            amount: Order size in base currency units.
            price: Limit price.

        Returns:
            Estimated wait time in seconds (min 5 s, max 300 s).
        """
        notional = amount * price
        # 5 s baseline + 1 s per 100 USDT of notional, capped at 300 s
        wait = 5.0 + notional / 100.0
        return min(wait, 300.0)

    def _get_best_available_book_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the best synchronously available order book for *symbol*.

        Checks the local orderbook manager first (zero latency) and returns
        ``None`` when the cache is absent or stale so callers can fall back
        to an async REST fetch.
        """
        if self._local_orderbook_manager is not None:
            return self._local_orderbook_manager.get_book(symbol)
        return None

    async def _check_pending_limit_orders(self) -> None:
        """Fill pending limit orders that have been triggered or passed queue wait.

        An order is eligible to fill when *either*:
        1. The live market price has traded *through* the limit price, **or**
        2. The price is within 0.5 % of the limit price *and* the order has
           been waiting for at least its estimated queue-wait duration.
        """
        async with self._lock:
            pending = [
                o
                for o in self._open_orders.values()
                if o.get("type") == "limit" and o.get("status") == "open"
                and not o.get("is_trigger")
            ]

        for order in pending:
            symbol = order["symbol"]
            limit_price = float(order["price"])
            side = OrderSide(order["side"])
            order_amount = float(order["amount"])
            queue_fill_time = float(order.get("queue_fill_time", 0.0))

            try:
                ticker = await self.get_ticker(symbol)
                current_price = ticker.last
            except Exception:
                continue

            # Condition 1: price trades through the limit
            price_through = (
                (side == OrderSide.BUY and current_price <= limit_price)
                or (side == OrderSide.SELL and current_price >= limit_price)
            )

            # Condition 2: price is near the limit and queue wait has elapsed
            near_limit = limit_price > 0 and abs(current_price - limit_price) / limit_price <= 0.005
            queue_elapsed = time.time() >= queue_fill_time
            price_at_level_with_wait = near_limit and queue_elapsed

            if not (price_through or price_at_level_with_wait):
                continue

            fill_price = limit_price
            logger.info(
                "PaperExchange limit order filled ({}): {} {} {} @ {:.4f} price={:.4f}",
                "price_through" if price_through else "queue_elapsed",
                side.value,
                order_amount,
                symbol,
                fill_price,
                current_price,
            )

            async with self._lock:
                self._open_orders.pop(order["id"], None)

            fee = fill_price * order_amount * _MAKER_FEE_PCT
            cost = fill_price * order_amount
            reduce_only = order.get("params", {}).get("reduceOnly", False)
            async with self._lock:
                if reduce_only:
                    self._close_position_internal(symbol, side, order_amount, fill_price, fee)
                else:
                    await self._open_position_internal(
                        symbol, side, order_amount, fill_price, fee, cost
                    )

            self._save_state()
            self._notify_state_change()

    async def _check_trigger_orders(self) -> None:
        """Check all pending trigger orders against live prices and fill if hit."""
        async with self._lock:
            trigger_orders = [
                o
                for o in self._open_orders.values()
                if o.get("is_trigger") and o.get("status") == "open"
            ]

        for order in trigger_orders:
            symbol = order["symbol"]
            trigger_price = float(order.get("trigger_price", 0))
            side = OrderSide(order["side"])
            order_type = order.get("type", "")
            try:
                ticker = await self.get_ticker(symbol)
                price = ticker.last
            except Exception:
                continue

            triggered = False
            if order_type == OrderType.STOP_LOSS.value:
                # Stop-loss: BUY-side triggers when price rises to trigger, SELL when falls
                if side == OrderSide.SELL and price <= trigger_price:
                    triggered = True
                elif side == OrderSide.BUY and price >= trigger_price:
                    triggered = True
            elif order_type == OrderType.TAKE_PROFIT.value:
                if side == OrderSide.SELL and price >= trigger_price:
                    triggered = True
                elif side == OrderSide.BUY and price <= trigger_price:
                    triggered = True

            if triggered:
                logger.info(
                    "PaperExchange trigger fired: {} {} @ price={} trigger={}",
                    order_type,
                    symbol,
                    price,
                    trigger_price,
                )
                async with self._lock:
                    self._open_orders.pop(order["id"], None)
                fill_price = self._apply_slippage(price, side)
                fee = fill_price * float(order["amount"]) * _TAKER_FEE_PCT
                async with self._lock:
                    self._close_position_internal(
                        symbol, side, float(order["amount"]), fill_price, fee
                    )
                self._save_state()
                self._notify_state_change()

    @staticmethod
    def _dict_to_order(raw: dict) -> Order:
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_loss": OrderType.STOP_LOSS,
            "take_profit": OrderType.TAKE_PROFIT,
        }
        price_raw = raw.get("price")
        return Order(
            id=str(raw.get("id", "")),
            symbol=str(raw.get("symbol", "")),
            type=type_map.get(str(raw.get("type", "market")).lower(), OrderType.MARKET),
            side=OrderSide(str(raw.get("side", "buy")).lower()),
            amount=float(raw.get("amount", 0)),
            price=float(price_raw) if price_raw is not None else None,
            filled=float(raw.get("filled", 0)),
            remaining=float(raw.get("remaining", 0)),
            status=OrderStatus(str(raw.get("status", "open")).lower()),
            timestamp=int(raw.get("timestamp", 0)),
            fee=float(raw.get("fee", 0)),
        )
