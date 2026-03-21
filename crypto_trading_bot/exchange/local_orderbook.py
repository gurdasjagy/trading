"""Zero-latency local order book cache backed by WebSocket subscriptions.

:class:`LocalOrderBookManager` spawns one background asyncio task per symbol
that calls :meth:`~exchange.base_exchange.BaseExchange.subscribe_orderbook`.
The exchange client uses ``ccxt.pro``'s ``watch_order_book`` when available
and falls back to REST polling otherwise.

Usage::

    manager = LocalOrderBookManager(exchange, ["BTC/USDT", "ETH/USDT"])
    await manager.start()

    book = manager.get_book("BTC/USDT")  # None if stale, dict otherwise
    if book is None:
        book = await exchange.get_orderbook("BTC/USDT")  # REST fallback
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange

try:
    from rust_trading_engine.orderbook import RustOrderBook
    _USE_RUST_BOOK = True
except ImportError:
    _USE_RUST_BOOK = False


class LocalOrderBookManager:
    """Maintains a real-time in-memory cache of L2 order books.

    Spawns one background asyncio task per symbol that calls
    ``exchange.subscribe_orderbook`` (backed by ccxt.pro's
    ``watch_order_book`` when available, or REST polling when not).

    The cached snapshot is considered *stale* after
    :attr:`_STALE_THRESHOLD_SECONDS` seconds; callers should fall back to a
    REST fetch when :meth:`get_book` returns ``None``.

    Args:
        exchange: An active exchange client used for subscriptions.
        symbols: List of trading pair symbols to maintain books for.
    """

    #: Maximum age of a cached book before it is considered stale (seconds)
    _STALE_THRESHOLD_SECONDS: float = 2.0
    #: REST polling interval used when WebSocket streaming is not available
    _REST_POLL_INTERVAL_SECONDS: float = 2.0

    def __init__(self, exchange: "BaseExchange", symbols: List[str]) -> None:
        self._exchange = exchange
        self._symbols = list(symbols)
        self._books: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self._tasks: List[asyncio.Task] = []
        # Rust-backed per-symbol order books (populated when the Rust engine is available)
        self._rust_books: Dict[str, "RustOrderBook"] = {} if _USE_RUST_BOOK else {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn background streaming tasks for all configured symbols."""
        for symbol in self._symbols:
            task = asyncio.create_task(
                self._stream_symbol(symbol),
                name=f"lob-{symbol}",
            )
            self._tasks.append(task)
        logger.info(
            "LocalOrderBookManager started — streaming {} symbol(s): {}",
            len(self._symbols),
            self._symbols,
        )

    async def stop(self) -> None:
        """Cancel all streaming tasks and clear the internal cache."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("LocalOrderBookManager stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the cached order book for *symbol* if it is still fresh.

        When the Rust engine is available, the :class:`RustOrderBook` snapshot
        is used as the primary source (sub-millisecond staleness check via
        ``Instant``).  The Python dict cache is used as a fallback.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.

        Returns:
            Dict with ``bids`` and ``asks`` lists (``[price, size]`` pairs),
            or ``None`` when the cache has no entry for *symbol* or the last
            update is older than :attr:`_STALE_THRESHOLD_SECONDS`.
        """
        # Fast path: use the Rust book when available
        if _USE_RUST_BOOK and symbol in self._rust_books:
            rust_book = self._rust_books[symbol]
            if not rust_book.is_stale(self._STALE_THRESHOLD_SECONDS * 1000.0):
                return rust_book.get_snapshot()
            logger.debug(
                "LocalOrderBook: stale Rust book for {} (age={:.2f}ms > {:.0f}ms threshold)",
                symbol,
                rust_book.get_age_ms(),
                self._STALE_THRESHOLD_SECONDS * 1000.0,
            )
            return None

        # Python fallback
        if symbol not in self._books:
            return None
        age = time.monotonic() - self._timestamps.get(symbol, 0.0)
        if age > self._STALE_THRESHOLD_SECONDS:
            logger.debug(
                "LocalOrderBook: stale book for {} (age={:.2f}s > {:.1f}s threshold)",
                symbol,
                age,
                self._STALE_THRESHOLD_SECONDS,
            )
            return None
        return self._books[symbol]

    def get_rust_book(self, symbol: str) -> Optional["RustOrderBook"]:
        """Return the :class:`RustOrderBook` for *symbol*, or ``None`` if unavailable.

        Callers that need direct access to the Rust object (e.g. for passing to
        :class:`~exchange.gateio_book_analyzer.GateioBookAnalyzer`) should use
        this method rather than :meth:`get_book`.
        """
        if not _USE_RUST_BOOK:
            return None
        return self._rust_books.get(symbol)

    def get_book_age(self, symbol: str) -> float:
        """Return the age of the cached book in seconds, or infinity if absent."""
        if symbol not in self._timestamps:
            return float("inf")
        return time.monotonic() - self._timestamps[symbol]

    # ------------------------------------------------------------------
    # Internal streaming
    # ------------------------------------------------------------------

    async def _stream_symbol(self, symbol: str) -> None:
        """Background task: continuously stream order book updates for *symbol*.

        Tries WebSocket streaming first; falls back to REST polling permanently
        when ``watch_order_book`` raises :exc:`NotImplementedError` (e.g. paper
        trading mode or an exchange that lacks ccxt.pro support).
        """
        _use_rest = False
        while True:
            try:
                if _use_rest:
                    # REST polling fallback — avoids tight-loop reconnections
                    book = await self._exchange.get_orderbook(symbol, limit=20)
                    self._books[symbol] = {
                        "bids": book.get("bids", []),
                        "asks": book.get("asks", []),
                        "timestamp": book.get("timestamp"),
                    }
                    self._timestamps[symbol] = time.monotonic()
                    self._update_rust_book(symbol, self._books[symbol])
                    logger.debug(
                        "LocalOrderBook: REST poll updated {} — bids={} asks={}",
                        symbol,
                        len(self._books[symbol]["bids"]),
                        len(self._books[symbol]["asks"]),
                    )
                    await asyncio.sleep(self._REST_POLL_INTERVAL_SECONDS)
                else:
                    logger.debug("LocalOrderBook: subscribing to {} order book", symbol)
                    raw = await self._exchange.watch_order_book(symbol)
                    self._books[symbol] = {
                        "bids": raw.get("bids", []),
                        "asks": raw.get("asks", []),
                        "timestamp": raw.get("timestamp"),
                    }
                    self._timestamps[symbol] = time.monotonic()
                    self._update_rust_book(symbol, self._books[symbol])
                    logger.debug(
                        "LocalOrderBook: updated {} — bids={} asks={}",
                        symbol,
                        len(self._books[symbol]["bids"]),
                        len(self._books[symbol]["asks"]),
                    )
            except asyncio.CancelledError:
                logger.debug("LocalOrderBook: stream task for {} cancelled", symbol)
                break
            except NotImplementedError:
                logger.info(
                    "LocalOrderBook: watch_order_book not supported for {} — "
                    "switching to REST polling permanently",
                    symbol,
                )
                _use_rest = True
            except Exception as exc:
                if "parse_frame" in str(exc):
                    logger.warning(
                        "Aiohttp compatibility error detected for {} ({}), forcing connection reset",
                        symbol,
                        exc,
                    )
                    try:
                        await self._exchange.close()
                    except Exception:
                        pass
                    await asyncio.sleep(5)
                else:
                    logger.warning(
                        "LocalOrderBook: stream error for {} — {} — reconnecting in 2 s",
                        symbol,
                        exc,
                    )
                    await asyncio.sleep(self._REST_POLL_INTERVAL_SECONDS)

    def _make_callback(self, symbol: str):
        """Return an async callback that updates the internal book for *symbol*."""

        async def _on_update(orderbook: Dict[str, Any]) -> None:
            self._books[symbol] = {
                "bids": orderbook.get("bids", []),
                "asks": orderbook.get("asks", []),
                "timestamp": orderbook.get("timestamp"),
            }
            self._timestamps[symbol] = time.monotonic()
            self._update_rust_book(symbol, self._books[symbol])
            logger.debug(
                "LocalOrderBook: updated {} — bids={} asks={}",
                symbol,
                len(self._books[symbol]["bids"]),
                len(self._books[symbol]["asks"]),
            )

        return _on_update

    def _update_rust_book(self, symbol: str, book: Dict[str, Any]) -> None:
        """Synchronise the :class:`RustOrderBook` for *symbol* with *book* data.

        Called after every Python dict update so the Rust book is always kept in
        sync.  Does nothing when the Rust engine is not available.
        """
        if not _USE_RUST_BOOK:
            return
        try:
            if symbol not in self._rust_books:
                self._rust_books[symbol] = RustOrderBook(symbol)
            rust_book = self._rust_books[symbol]
            # Convert list-of-pairs to list-of-tuples expected by Rust
            bids = [(float(b[0]), float(b[1])) for b in book.get("bids", []) if len(b) >= 2]
            asks = [(float(a[0]), float(a[1])) for a in book.get("asks", []) if len(a) >= 2]
            # Count any malformed levels that were silently skipped
            raw_bids = book.get("bids", [])
            raw_asks = book.get("asks", [])
            skipped = sum(1 for b in raw_bids if len(b) < 2) + sum(1 for a in raw_asks if len(a) < 2)
            if skipped:
                logger.debug(
                    "LocalOrderBook: skipped {} malformed level(s) for {}", skipped, symbol
                )
            rust_book.update_snapshot(bids, asks)
        except Exception as exc:
            logger.debug("LocalOrderBook: Rust book update failed for {}: {}", symbol, exc)
