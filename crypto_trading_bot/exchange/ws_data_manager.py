"""WebSocket Data Manager — real-time market data cache.

Maintains an in-memory cache of latest tickers and trade streams
for all configured trading pairs, updated via exchange WebSocket
streams.  Strategies and the fast cycle consume from the cache
(zero REST latency) rather than issuing individual REST calls.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


class WebSocketDataManager:
    """Subscribe to ticker and trade streams; maintain zero-latency in-memory cache.

    Args:
        exchange: CCXT Pro (or wrapper) exchange instance that exposes
            ``watch_ticker()`` / ``watch_tickers()`` and ``watch_trades()`` coroutines.
            When the exchange supports ``watchTickers`` (bulk), that is used for
            efficiency.  Otherwise (e.g. Gate.io) the manager falls back to
            per-symbol ``watch_ticker()`` streams.
        symbols: List of trading-pair symbols to subscribe to.
        significant_move_pct: Minimum price change (as a fraction, e.g. 0.001
            for 0.1 %) that triggers registered callbacks.
        max_trades_per_symbol: Number of recent trades to retain per symbol.
    """

    def __init__(
        self,
        exchange,
        symbols: List[str],
        significant_move_pct: float = 0.001,
        max_trades_per_symbol: int = 1000,
    ) -> None:
        self._exchange = exchange
        self._symbols = list(symbols)
        self._significant_move_pct = significant_move_pct
        self._max_trades = max_trades_per_symbol

        # Latest ticker snapshot: symbol → ticker dict
        self._latest_tickers: Dict[str, dict] = {}
        # Last price used for significant-move detection: symbol → float
        self._last_eval_prices: Dict[str, float] = {}
        # Recent trades ring-buffer: symbol → deque(maxlen)
        self._latest_trades: Dict[str, deque] = {
            sym: deque(maxlen=self._max_trades) for sym in self._symbols
        }

        # Registered callbacks for significant price moves
        self._move_callbacks: List[Callable[[str, float, float], None]] = []

        # Symbol map: original symbol (e.g. "BTC/USDT") → WS-compatible swap
        # symbol (e.g. "BTC/USDT:USDT").  Futures exchanges (Gate.io, Binance,
        # MEXC …) require the swap/perpetual format for their futures WS
        # channels.  The reverse map is used to key the cache back under the
        # original symbol so callers always use the same notation.
        self._ws_symbol_map: Dict[str, str] = {}
        self._ws_reverse_map: Dict[str, str] = {}

        # Background tasks
        self._ticker_task: Optional[asyncio.Task] = None
        self._trades_task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to all WebSocket streams and start background tasks."""
        if self._running:
            logger.debug("WebSocketDataManager already running — skipping start()")
            return
        self._running = True
        logger.info(
            "WebSocketDataManager starting streams for {} symbols: {}",
            len(self._symbols),
            self._symbols,
        )
        self._ticker_task = asyncio.create_task(
            self._ticker_loop(), name="ws_dm_tickers"
        )
        self._trades_task = asyncio.create_task(
            self._trades_loop(), name="ws_dm_trades"
        )
        logger.info("WebSocketDataManager streams started.")

    async def stop(self) -> None:
        """Cancel background tasks and clean up."""
        self._running = False
        for task in (self._ticker_task, self._trades_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("WebSocketDataManager stopped.")

    def get_price(self, symbol: str) -> float:
        """Return the latest mid-price from the cache (zero REST latency).

        Returns 0.0 if no data has arrived yet for the symbol.
        """
        ticker = self._latest_tickers.get(symbol)
        if ticker is None:
            return 0.0
        last = ticker.get("last") or ticker.get("close") or 0.0
        return float(last)

    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Return (bid, ask) from the cache.

        Returns (0.0, 0.0) if no data is available yet.
        """
        ticker = self._latest_tickers.get(symbol)
        if ticker is None:
            return 0.0, 0.0
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        return bid, ask

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """Return the latest raw ticker dict (or None)."""
        return self._latest_tickers.get(symbol)

    def get_recent_trades(self, symbol: str) -> deque:
        """Return the deque of recent trades for *symbol*."""
        return self._latest_trades.get(symbol, deque())

    def on_significant_move(
        self, callback: Callable[[str, float, float], None]
    ) -> None:
        """Register a callback invoked when any symbol moves > ``significant_move_pct``.

        The callback receives ``(symbol, old_price, new_price)``.
        """
        self._move_callbacks.append(callback)

    def is_ready(self, symbol: str) -> bool:
        """Return True if at least one ticker has been received for *symbol*."""
        return symbol in self._latest_tickers

    # ------------------------------------------------------------------
    # Background stream loops
    # ------------------------------------------------------------------

    def _build_symbol_map(self) -> None:
        """Build mapping from original symbols → WS-compatible (swap) symbols.

        Futures exchanges (Gate.io, Binance, MEXC …) require swap-format symbols
        such as ``BTC/USDT:USDT`` for their futures WebSocket channels.  Passing
        spot-format symbols (``BTC/USDT``) to a ccxt.pro client configured with
        ``defaultType='swap'`` raises a ``KeyError('spot')`` error at runtime.

        This method resolves each configured symbol via the exchange's own
        ``_resolve_swap_symbol`` helper (when available) and stores both the
        forward map (original → ws_sym) and the reverse map (ws_sym → original)
        so that tickers/trades received under the swap symbol are keyed back to
        the original symbol in the cache.

        Safe to call multiple times — returns immediately on the second call.
        """
        if self._ws_symbol_map:
            return  # Already built

        resolve = getattr(self._exchange, "_resolve_swap_symbol", None)
        for sym in self._symbols:
            try:
                ws_sym = resolve(sym) if callable(resolve) else sym
            except Exception as exc:
                logger.debug(
                    "WebSocketDataManager: symbol resolution failed for {}: {} — using as-is",
                    sym,
                    exc,
                )
                ws_sym = sym
            self._ws_symbol_map[sym] = ws_sym
            # If multiple originals map to the same ws_sym keep the first mapping
            if ws_sym not in self._ws_reverse_map:
                self._ws_reverse_map[ws_sym] = sym

        remapped = {k: v for k, v in self._ws_symbol_map.items() if k != v}
        if remapped:
            logger.info(
                "WebSocketDataManager: resolved spot→swap symbols for WS streams: {}",
                remapped,
            )

    async def _ticker_loop(self) -> None:
        """Continuously watch tickers and update the cache."""
        _use_bulk: Optional[bool] = None
        while self._running:
            try:
                client = self._get_ws_client()
                if client is None:
                    await asyncio.sleep(5)
                    continue

                # Resolve symbols to swap/perpetual format so Gate.io and other
                # futures exchanges route to the correct WS channel.
                self._build_symbol_map()

                # Lazily detect whether the exchange supports watchTickers (bulk).
                # Gate.io and some other exchanges only support watchTicker per-symbol.
                if _use_bulk is None:
                    has = getattr(client, "has", {}) or {}
                    _use_bulk = bool(has.get("watchTickers"))
                    if not _use_bulk:
                        logger.info(
                            "WebSocketDataManager: watchTickers not supported — "
                            "falling back to per-symbol watchTicker streams"
                        )

                if _use_bulk:
                    ws_syms = [self._ws_symbol_map.get(s, s) for s in self._symbols]
                    tickers = await client.watch_tickers(ws_syms)
                    if isinstance(tickers, dict):
                        for ws_sym, ticker in tickers.items():
                            original_sym = self._ws_reverse_map.get(ws_sym, ws_sym)
                            self._update_ticker(original_sym, ticker)
                    else:
                        # Some adapters return a single ticker object
                        ws_sym = getattr(tickers, "symbol", None) or (
                            tickers.get("symbol") if isinstance(tickers, dict) else None
                        )
                        if ws_sym:
                            original_sym = self._ws_reverse_map.get(ws_sym, ws_sym)
                            self._update_ticker(original_sym, tickers)
                else:
                    # Run one watch_ticker loop per symbol concurrently until stopped.
                    await asyncio.gather(
                        *(
                            self._watch_single_ticker_loop(client, sym)
                            for sym in self._symbols
                        )
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "WebSocketDataManager ticker stream error (reconnecting): {}", exc
                )
                await asyncio.sleep(2)

    async def _watch_single_ticker_loop(self, client: Any, sym: str) -> None:
        """Per-symbol ticker loop used when watchTickers is not supported."""
        # Use the swap-format symbol for the WS call; store under original symbol.
        ws_sym = self._ws_symbol_map.get(sym, sym)
        while self._running:
            try:
                ticker = await client.watch_ticker(ws_sym)
                self._update_ticker(sym, ticker)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(
                    "WebSocketDataManager ticker stream error for {} (reconnecting): {}",
                    sym,
                    exc,
                )
                await asyncio.sleep(2)

    async def _trades_loop(self) -> None:
        """Continuously watch trades and update per-symbol deques."""
        while self._running:
            try:
                client = self._get_ws_client()
                if client is None:
                    await asyncio.sleep(5)
                    continue

                # Resolve symbols to swap/perpetual format (same reason as ticker loop).
                self._build_symbol_map()

                # Watch all symbols; CCXT Pro returns a list of trades
                for sym in self._symbols:
                    ws_sym = self._ws_symbol_map.get(sym, sym)
                    try:
                        trades = await client.watch_trades(ws_sym)
                        # Store under the original symbol
                        buf = self._latest_trades.get(sym)
                        if buf is None:
                            buf = deque(maxlen=self._max_trades)
                            self._latest_trades[sym] = buf
                        if isinstance(trades, list):
                            buf.extend(trades)
                        else:
                            buf.append(trades)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.debug(
                            "Trade stream error for {}: {}", sym, exc
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "WebSocketDataManager trade stream error (reconnecting): {}", exc
                )
                await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ws_client(self):
        """Return the underlying CCXT Pro client from the exchange wrapper.

        Resolution order:
        1. Call the wrapper's own ``_get_ws_client()`` getter if it exists
           (e.g. ``CcxtExchange`` lazily initializes a ``ccxt.pro`` client).
        2. Look for a ``_ws_client`` attribute that supports WebSocket methods.
        3. Fall back to other attributes for simple/direct wrapper objects.
        4. Return the exchange itself if it is already a ``ccxt.pro`` instance.
        """
        # 1. Prefer the explicit getter from the wrapper if available.
        #    CcxtExchange._get_ws_client() lazily initializes and returns the
        #    ccxt.pro client stored in self._ws_client — this is the correct
        #    path to avoid accidentally returning the REST _client.
        if hasattr(self._exchange, "_get_ws_client") and callable(
            self._exchange._get_ws_client
        ):
            try:
                return self._exchange._get_ws_client()
            except Exception as exc:
                logger.debug(
                    "WebSocketDataManager: exchange._get_ws_client() failed: {}", exc
                )

        # 2. Look for a dedicated WS-client attribute that actually supports
        #    WebSocket calls.  Check _ws_client before _client to avoid
        #    returning the REST client.  Also require that the candidate's
        #    ``has`` dict advertises genuine watchTicker/watchTickers support
        #    (not just stub methods that raise NotSupported at runtime).
        for attr in ("_ws_client", "_client", "_exchange", "exchange"):
            candidate = getattr(self._exchange, attr, None)
            if candidate is None:
                continue
            has = getattr(candidate, "has", {}) or {}
            if has.get("watchTicker") or has.get("watchTickers"):
                return candidate

        # 3. If the exchange itself is a ccxt.pro instance (has dict confirms WS support).
        has = getattr(self._exchange, "has", {}) or {}
        if has.get("watchTicker") or has.get("watchTickers"):
            return self._exchange

        return None

    def _update_ticker(self, symbol: str, ticker: dict) -> None:
        """Store ticker and fire significant-move callbacks."""
        self._latest_tickers[symbol] = ticker
        new_price = float(ticker.get("last") or ticker.get("close") or 0.0)
        if new_price <= 0:
            return
        old_price = self._last_eval_prices.get(symbol, 0.0)
        if old_price > 0:
            move = abs(new_price - old_price) / old_price
            if move >= self._significant_move_pct:
                self._last_eval_prices[symbol] = new_price
                for cb in self._move_callbacks:
                    try:
                        cb(symbol, old_price, new_price)
                    except Exception as exc:
                        logger.debug(
                            "Significant-move callback error for {}: {}", symbol, exc
                        )
        else:
            self._last_eval_prices[symbol] = new_price
