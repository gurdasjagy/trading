"""Real-time market data feed aggregator over exchange WebSocket connections."""

import asyncio
from typing import Callable, Dict, List, Optional

from loguru import logger

from .base_exchange import BaseExchange, Ticker

try:
    from rust_trading_engine.ws_parser import (
        parse_ticker_message,
        parse_orderbook_message,
    )
    _USE_RUST_PARSER = True
except ImportError:
    _USE_RUST_PARSER = False


class MarketDataFeed:
    """Aggregates real-time market data from an exchange WebSocket.

    Maintains an in-memory cache of the latest :class:`Ticker` and order-book
    snapshots for each subscribed symbol.  External consumers can register
    callbacks for specific event types and read the latest data synchronously
    via :meth:`get_ticker` / :meth:`get_orderbook`.
    """

    # Supported event types for :meth:`subscribe`.
    EVENT_TICKER = "ticker"
    EVENT_ORDERBOOK = "orderbook"
    EVENT_TRADE = "trade"

    def __init__(self, exchange: BaseExchange, symbols: List[str]) -> None:
        self._exchange = exchange
        self._symbols = list(symbols)
        self._tickers: Dict[str, Ticker] = {}
        self._orderbooks: Dict[str, dict] = {}
        # event_type → list of callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            self.EVENT_TICKER: [],
            self.EVENT_ORDERBOOK: [],
            self.EVENT_TRADE: [],
        }
        self._running = False
        self._tasks: List[asyncio.Task] = []  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to all configured symbols across all data channels.

        This spawns background tasks for each symbol / channel combination.
        The feed keeps running until :meth:`stop` is called.
        """
        if self._running:
            logger.warning("MarketDataFeed is already running")
            return

        self._running = True
        logger.info(
            "MarketDataFeed starting for {} symbol(s): {}",
            len(self._symbols),
            self._symbols,
        )

        for symbol in self._symbols:
            self._tasks.append(
                asyncio.create_task(
                    self._exchange.subscribe_ticker(
                        symbol,
                        lambda data, s=symbol: asyncio.ensure_future(
                            self._handle_ticker_update(s, data)
                        ),
                    ),
                    name=f"feed_ticker_{symbol}",
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._exchange.subscribe_orderbook(
                        symbol,
                        lambda data, s=symbol: asyncio.ensure_future(
                            self._handle_orderbook_update(s, data)
                        ),
                    ),
                    name=f"feed_orderbook_{symbol}",
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._exchange.subscribe_trades(
                        symbol,
                        lambda data, s=symbol: asyncio.ensure_future(self._handle_trade(s, data)),
                    ),
                    name=f"feed_trades_{symbol}",
                )
            )

        logger.info("MarketDataFeed subscriptions initiated")

    async def stop(self) -> None:
        """Cancel all active subscription tasks."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("MarketDataFeed stopped")

    # ------------------------------------------------------------------
    # Synchronous data access
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Return the most recent :class:`Ticker` for *symbol*, or *None*."""
        return self._tickers.get(symbol)

    def get_orderbook(self, symbol: str) -> Optional[dict]:
        """Return the most recent order-book snapshot for *symbol*, or *None*."""
        return self._orderbooks.get(symbol)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register *callback* to be called for every *event_type* update.

        *event_type* must be one of :attr:`EVENT_TICKER`, :attr:`EVENT_ORDERBOOK`,
        or :attr:`EVENT_TRADE`.

        The callback signature depends on the event type:
        * ``ticker``: ``callback(symbol: str, ticker: Ticker)``
        * ``orderbook``: ``callback(symbol: str, orderbook: dict)``
        * ``trade``: ``callback(symbol: str, trade: dict)``

        Callbacks may be plain functions or coroutines.
        """
        if event_type not in self._callbacks:
            raise ValueError(
                f"Unknown event type {event_type!r}. "
                f"Valid types: {list(self._callbacks.keys())}"
            )
        self._callbacks[event_type].append(callback)
        logger.debug("Registered {} callback: {}", event_type, callback)

    # ------------------------------------------------------------------
    # Internal update handlers
    # ------------------------------------------------------------------

    async def _handle_ticker_update(self, symbol: str, data: dict) -> None:
        """Parse raw exchange ticker data, update cache, and notify subscribers."""
        try:
            ticker = self._parse_ticker(symbol, data)
            if ticker is None:
                return
            self._tickers[symbol] = ticker
            await self._dispatch(self.EVENT_TICKER, symbol, ticker)
        except Exception as exc:
            logger.warning("Error handling ticker update for {}: {}", symbol, exc)

    async def _handle_orderbook_update(self, symbol: str, data: dict) -> None:
        """Parse raw order-book data, update cache, and notify subscribers."""
        try:
            # Fast path: if data arrived as raw bytes, use the Rust parser.
            if _USE_RUST_PARSER and isinstance(data, (bytes, bytearray)):
                try:
                    book = parse_orderbook_message(data)
                    if book is not None:
                        self._orderbooks[symbol] = book
                        await self._dispatch(self.EVENT_ORDERBOOK, symbol, book)
                        return
                except Exception as exc:
                    logger.debug(
                        "Rust orderbook parser failed for {} ({}), falling back",
                        symbol,
                        exc,
                    )
            if not isinstance(data, dict):
                return
            # Normalise regardless of exchange-specific wrapper structure.
            book = data.get("data", data)
            if isinstance(book, list) and book:
                book = book[0]
            self._orderbooks[symbol] = book
            await self._dispatch(self.EVENT_ORDERBOOK, symbol, book)
        except Exception as exc:
            logger.warning("Error handling orderbook update for {}: {}", symbol, exc)

    async def _handle_trade(self, symbol: str, data: dict) -> None:
        """Forward a raw trade event to registered callbacks."""
        try:
            trade = data.get("data", data)
            await self._dispatch(self.EVENT_TRADE, symbol, trade)
        except Exception as exc:
            logger.warning("Error handling trade for {}: {}", symbol, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _dispatch(self, event_type: str, *args) -> None:
        """Call all registered callbacks for *event_type* with *args*."""
        for cb in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(*args)
                else:
                    cb(*args)
            except Exception as exc:
                logger.warning("Callback error ({} / {}): {}", event_type, cb, exc)

    @staticmethod
    def _parse_ticker(symbol: str, data: dict) -> Optional[Ticker]:
        """Attempt to extract a :class:`Ticker` from a raw WebSocket message.

        Returns *None* if the payload does not contain recognisable ticker
        fields (e.g. a subscription confirmation message).

        When the Rust parser is available and *data* is raw bytes the fast-path
        is taken; otherwise the Python implementation is used as a fallback.
        """
        if _USE_RUST_PARSER and isinstance(data, (bytes, bytearray)):
            try:
                result = parse_ticker_message(symbol, data)
                if result is not None:
                    return Ticker(
                        symbol=result.symbol,
                        bid=result.bid,
                        ask=result.ask,
                        last=result.last,
                        high=result.high,
                        low=result.low,
                        volume=result.volume,
                        timestamp=result.timestamp,
                    )
                return None
            except Exception as exc:
                logger.warning(
                    "Rust ticker parser failed for {} ({}), falling back to Python",
                    symbol,
                    exc,
                )

        # Original Python fallback — keep ALL existing code unchanged below
        # Handle both top-level data and wrapped data structures.
        payload = data.get("data", data)
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            return None

        # Common field names used across MEXC / Gate.io / BingX / Bitget
        last = (
            payload.get("last")
            or payload.get("c")  # close / last
            or payload.get("close")
            or payload.get("lastPr")  # Bitget
        )
        if last is None:
            return None

        bid = payload.get("bid") or payload.get("b") or payload.get("bestBid") or last
        ask = payload.get("ask") or payload.get("a") or payload.get("bestAsk") or last
        high = payload.get("high") or payload.get("h") or last
        low = payload.get("low") or payload.get("l") or last
        volume = payload.get("volume") or payload.get("v") or payload.get("baseVolume") or 0
        ts = payload.get("timestamp") or payload.get("t") or payload.get("ts") or 0

        try:
            return Ticker(
                symbol=symbol,
                bid=float(bid),
                ask=float(ask),
                last=float(last),
                high=float(high),
                low=float(low),
                volume=float(volume),
                timestamp=int(ts),
            )
        except (TypeError, ValueError):
            return None
