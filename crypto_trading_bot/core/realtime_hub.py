"""Real-time data hub that bridges exchange feeds to dashboard WebSocket clients.

The hub maintains an in-memory cache of the latest tickers, balance, positions
and orders.  Dashboard WebSocket connections register with the hub and receive
pushed updates as market data changes instead of polling every 5 seconds.

Supported modes
---------------
* **Paper mode** — a background loop fetches tickers via REST every second and
  recomputes paper P&L in real time.  The paper exchange ``_on_state_change``
  callback fires the hub immediately whenever a position opens/closes.
* **Live mode** — the hub can subscribe to WebSocket feeds via the exchange's
  ``subscribe_ticker`` / ``subscribe_user_data`` methods (if the exchange
  supports them).  Falls back to the 1-second REST polling loop otherwise.

The ``_broadcast_loop`` runs every 1 second as a safety net and pushes the
full snapshot to all connected clients.  Individual price/position updates are
pushed immediately via the event callbacks above.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class RealtimeHub:
    """Central real-time data hub.

    Args:
        exchange: The active exchange instance (paper or live).
        position_manager: Optional position manager for richer summaries.
        order_manager: Optional order manager.
        paper_exchange: Convenience reference to the PaperExchange when in
            paper mode (may be the same object as *exchange*).
        broadcast_fn: An ``async`` callable that accepts a ``dict`` payload
            and sends it to all connected dashboard WebSocket clients.  Set
            by the dashboard after creation.
    """

    def __init__(
        self,
        exchange: Any,
        position_manager: Any = None,
        order_manager: Any = None,
        paper_exchange: Any = None,
        broadcast_fn: Any = None,
    ) -> None:
        self._exchange = exchange
        self._position_manager = position_manager
        self._order_manager = order_manager
        self._paper_exchange = paper_exchange
        self._broadcast_fn = broadcast_fn

        # Cached state
        self._latest_tickers: Dict[str, float] = {}  # symbol → last price
        self._latest_balance: Dict[str, float] = {"usdt_free": 0.0, "usdt_total": 0.0}
        self._latest_positions: List[dict] = []
        self._latest_orders: List[dict] = []
        self._latest_trades: List[dict] = []  # Recent closed trades
        self._latest_funding_rates: Dict[str, float] = {}  # symbol → funding rate

        # Session equity tracking for daily PnL computation
        self._session_start_equity: float = 0.0

        # OHLCV / K-line cache: (symbol, timeframe) → list of candle dicts
        self._kline_cache: Dict[Tuple[str, str], List[dict]] = {}

        self._lock = asyncio.Lock()
        self._running = False
        self._symbols: List[str] = []

        # Background tasks
        self._price_loop_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._position_update_task: Optional[asyncio.Task] = None
        self._ws_tasks: List[asyncio.Task] = []
        self._ohlcv_tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_broadcast_fn(self, fn: Any) -> None:
        """Register the async broadcast function used to push to all WS clients."""
        self._broadcast_fn = fn

    async def trigger_state_refresh(self) -> None:
        """Trigger an immediate recomputation and broadcast of positions/balance.

        A public convenience method for callers (e.g. dashboard API handlers)
        that want to force a state refresh after making a trade or other change.
        """
        await self._recompute_state()

    async def start(self, symbols: List[str]) -> None:
        """Start the hub for the given trading symbols.

        Args:
            symbols: List of trading pair symbols to monitor (e.g. ``["BTC/USDT"]``).
        """
        self._symbols = list(symbols)
        self._running = True

        # Record session start equity for daily PnL tracking
        try:
            initial_balance = await self._collect_balance()
            self._session_start_equity = initial_balance.get("usdt_total", 0.0)
            logger.info("RealtimeHub: session start equity = {:.2f} USDT", self._session_start_equity)
        except Exception as exc:
            logger.debug("RealtimeHub: could not record session start equity: {}", exc)

        # Register callback on paper exchange for immediate push on state changes
        if self._paper_exchange is not None:
            try:
                self._paper_exchange.set_state_change_callback(self._on_paper_state_change)
                logger.info("RealtimeHub: paper state-change callback registered.")
            except Exception as exc:
                logger.debug("RealtimeHub: could not register paper callback: {}", exc)

        # Try to use WebSocket streams if the exchange supports subscribe_ticker.
        # Fall back to the REST polling _price_loop when WS is unavailable.
        ws_started = False
        if self._exchange is not None and hasattr(self._exchange, "subscribe_ticker"):
            try:
                for sym in self._symbols:
                    task = asyncio.create_task(
                        self._ws_ticker_listener(sym),
                        name=f"realtime_hub_ws_{sym}",
                    )
                    self._ws_tasks.append(task)
                ws_started = bool(self._ws_tasks)
                logger.info(
                    "RealtimeHub: started {} WebSocket ticker stream(s).",
                    len(self._ws_tasks),
                )
            except Exception as exc:
                logger.warning(
                    "RealtimeHub: failed to start WebSocket streams, falling back to REST: {}",
                    exc,
                )
                for t in self._ws_tasks:
                    t.cancel()
                self._ws_tasks = []
                ws_started = False

        if not ws_started:
            self._price_loop_task = asyncio.create_task(
                self._price_loop(), name="realtime_hub_price_loop"
            )

        self._broadcast_task = asyncio.create_task(
            self._broadcast_loop(), name="realtime_hub_broadcast_loop"
        )

        # Start periodic position P&L update loop
        self._position_update_task = asyncio.create_task(
            self._position_update_loop(), name="position_update_loop"
        )

        # Start OHLCV polling tasks for 1m and 5m timeframes
        if self._exchange is not None:
            for sym in self._symbols:
                for tf in ("1m", "5m"):
                    task = asyncio.create_task(
                        self._ohlcv_poll_loop(sym, tf),
                        name=f"realtime_hub_ohlcv_{sym}_{tf}",
                    )
                    self._ohlcv_tasks.append(task)
            logger.info(
                "RealtimeHub: started {} OHLCV polling task(s).",
                len(self._ohlcv_tasks),
            )

        logger.info(
            "RealtimeHub started for {} symbol(s): {} (mode={})",
            len(self._symbols),
            ", ".join(self._symbols),
            "websocket" if ws_started else "rest_poll",
        )

    async def stop(self) -> None:
        """Stop the hub and cancel background tasks."""
        self._running = False
        all_tasks = (
            list(self._ws_tasks)
            + list(self._ohlcv_tasks)
            + [self._price_loop_task, self._broadcast_task, self._position_update_task]
        )
        for task in all_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._ws_tasks = []
        self._ohlcv_tasks = []
        logger.info("RealtimeHub stopped.")

    def get_snapshot(self) -> dict:
        """Return the current cached state for initial page load.

        Returns a dict in the same format as ``_collect_live_data()`` so that
        the dashboard's initial HTTP render can use hub data instead of making
        REST calls.
        """
        return {
            "type": "update",
            "portfolio": self._build_portfolio_dict(),
            "positions": list(self._latest_positions),
            "open_orders": list(self._latest_orders),
        }

    def get_kline_snapshot(self, symbol: str, timeframe: str) -> List[dict]:
        """Return cached OHLCV candles for *symbol* / *timeframe*, or empty list."""
        return list(self._kline_cache.get((symbol, timeframe), []))

    # ------------------------------------------------------------------
    # Internal — price loop (1-second tick for paper + REST fallback)
    # ------------------------------------------------------------------

    async def _price_loop(self) -> None:
        """Fetch tickers every 1 second and recompute paper P&L."""
        while self._running:
            try:
                await self._refresh_prices()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("RealtimeHub._price_loop error: {}", exc)
            await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Internal — WebSocket ticker listener
    # ------------------------------------------------------------------

    async def _ws_ticker_listener(self, symbol: str) -> None:
        """Subscribe to the exchange WebSocket ticker stream for *symbol*.

        ``subscribe_ticker`` is a long-running coroutine that accepts a
        callback; it is NOT an async generator.  This method registers an
        async callback that updates the hub cache and immediately pushes
        ``price_update`` events to connected clients.  On failure it falls
        back to the REST ``_price_loop``.
        """
        async def _ticker_cb(ticker: Any) -> None:
            if not self._running:
                return
            if ticker is None:
                return
            try:
                raw_price = getattr(ticker, "last", None)
                if raw_price is None:
                    return
                price = float(raw_price)
                async with self._lock:
                    self._latest_tickers[symbol] = price
                await self._on_ws_price_update(symbol, price)
            except Exception as exc:
                logger.debug(
                    "RealtimeHub._ws_ticker_listener parse error for {}: {}", symbol, exc
                )

        try:
            await self._exchange.subscribe_ticker(symbol, _ticker_cb)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(
                "RealtimeHub: WebSocket stream for {} failed ({}); falling back to REST poll.",
                symbol,
                exc,
            )
            # Fall back to REST polling for this symbol only
            if self._price_loop_task is None or self._price_loop_task.done():
                self._price_loop_task = asyncio.create_task(
                    self._price_loop(), name="realtime_hub_price_loop"
                )

    async def _on_ws_price_update(self, symbol: str, price: float) -> None:
        """Handle a fresh price tick from the WebSocket stream.

        Recomputes positions and balance using the updated cache and
        immediately pushes a ``price_update`` event to all connected clients.
        * **Paper mode**: fully re-collects positions and balance from the
          paper exchange (which tracks exact margin per position).
        * **Live mode**: updates PnL for every cached position whose price is
          now known without making a REST call.  A full REST refresh is done
          in the background every ``_LIVE_REST_REFRESH_INTERVAL`` seconds.

        Args:
            symbol: The symbol whose price was updated.
            price: The latest traded price for *symbol*.
        """
        try:
            if self._paper_exchange is not None:
                positions = await self._collect_positions()
                balance = await self._collect_balance()
                async with self._lock:
                    self._latest_positions = positions
                    self._latest_balance = balance
                await self._push_update({"type": "position_update"})
            else:
                # Live mode: apply price update to cached positions immediately
                # so PnL animates without waiting for the next REST poll.
                updated = self._apply_price_to_cached_positions(symbol, price)
                if updated:
                    await self._push_update({"type": "position_update"})
                # Schedule a background REST refresh if one isn't already running
                await self._maybe_schedule_live_rest_refresh()
        except Exception as exc:
            logger.debug("RealtimeHub._on_ws_price_update error for {}: {}", symbol, exc)

    # ------------------------------------------------------------------
    # Internal — live-mode PnL helpers (no REST calls on every tick)
    # ------------------------------------------------------------------

    #: How often (seconds) to do a full REST refresh of live positions/balance.
    _LIVE_REST_REFRESH_INTERVAL: float = 30.0

    def _apply_price_to_cached_positions(self, symbol: str, price: float) -> bool:
        """Update PnL for the matching cached position using *price*.

        Mutates ``_latest_positions`` in-place (no lock needed — called from a
        single asyncio task) and returns ``True`` when at least one position
        was updated.

        The unrealized PnL formula for both long and short:
            pnl = (mark_price - entry_price) * amount  (long)
            pnl = (entry_price - mark_price) * amount  (short)
        where *amount* is in base currency units (contracts × contract_size
        should already be stored in the position dict's ``amount`` field as set
        by ``_collect_positions``).
        """
        updated = False
        for pos in self._latest_positions:
            if pos.get("symbol") != symbol:
                continue
            entry = pos.get("entry_price", 0.0)
            amount = pos.get("amount", 0.0)
            leverage = pos.get("leverage", 1) or 1
            direction = str(pos.get("direction", "long")).lower()
            if entry <= 0 or amount <= 0:
                continue
            if direction in ("long", "buy"):
                pnl = (price - entry) * amount
                pnl_pct = (price - entry) / entry * 100.0
            else:
                pnl = (entry - price) * amount
                pnl_pct = (entry - price) / entry * 100.0
            roe_pct = pnl_pct * leverage
            pos["current_price"] = round(price, 8)
            pos["mark_price"] = round(price, 8)
            pos["pnl"] = round(pnl, 4)
            pos["pnl_pct"] = round(pnl_pct, 4)
            pos["roe_pct"] = round(roe_pct, 4)
            updated = True
        if updated:
            self._latest_tickers[symbol] = price
        return updated

    async def _maybe_schedule_live_rest_refresh(self) -> None:
        """Schedule a full REST refresh of live positions if the interval has elapsed.

        A background task is created so the WS tick handler returns immediately
        and the REST call does not block the ticker callback.
        """
        import time as _time

        if not hasattr(self, "_last_live_rest_refresh"):
            self._last_live_rest_refresh: float = 0.0
        now = _time.monotonic()
        if now - self._last_live_rest_refresh < self._LIVE_REST_REFRESH_INTERVAL:
            return
        self._last_live_rest_refresh = now
        asyncio.ensure_future(self._live_rest_refresh())

    async def _live_rest_refresh(self) -> None:
        """Full REST refresh: fetch actual position data and balance from exchange."""
        try:
            positions = await self._collect_positions()
            balance = await self._collect_balance()
            async with self._lock:
                self._latest_positions = positions
                self._latest_balance = balance
            await self._push_update({"type": "price_update"})
        except Exception as exc:
            logger.debug("RealtimeHub._live_rest_refresh error: {}", exc)

    async def _refresh_prices(self) -> None:
        """Fetch tickers for all symbols and recompute positions + balance."""
        if not self._exchange:
            return

        new_tickers: Dict[str, float] = {}
        for sym in self._symbols:
            try:
                ticker = await self._exchange.get_ticker(sym)
                new_tickers[sym] = float(ticker.last)
            except Exception as exc:
                logger.debug("RealtimeHub: ticker fetch error for {}: {}", sym, exc)
                # Keep previous price if available
                if sym in self._latest_tickers:
                    new_tickers[sym] = self._latest_tickers[sym]

        async with self._lock:
            self._latest_tickers.update(new_tickers)

        # Push updated prices to PaperExchange so balance/upnl calculations are current
        if self._paper_exchange is not None and hasattr(self._paper_exchange, "update_ticker_cache"):
            self._paper_exchange.update_ticker_cache(new_tickers)

        await self._recompute_state()

    async def _recompute_state(self) -> None:
        """Recompute positions and balance from latest tickers and push updates."""
        import time as _time

        positions = await self._collect_positions()
        balance = await self._collect_balance()

        # Periodically refresh recent trades and funding rates (every 10 seconds to avoid excessive calls)
        if not hasattr(self, "_last_trades_refresh"):
            self._last_trades_refresh: float = 0.0
        if _time.time() - self._last_trades_refresh > 10:
            try:
                trades = await self._collect_recent_trades(20)
                funding_rates = await self._collect_funding_rates()
                async with self._lock:
                    self._latest_trades = trades
                    self._latest_funding_rates = funding_rates
                self._last_trades_refresh = _time.time()
            except Exception as exc:
                logger.debug("RealtimeHub: trades/funding refresh error: {}", exc)

        changed = False
        async with self._lock:
            if positions != self._latest_positions or balance != self._latest_balance:
                self._latest_positions = positions
                self._latest_balance = balance
                changed = True

        if changed:
            await self._push_update({"type": "price_update"})

    # ------------------------------------------------------------------
    # Internal — OHLCV / K-line polling
    # ------------------------------------------------------------------

    async def _ohlcv_poll_loop(self, symbol: str, timeframe: str) -> None:
        """Fetch OHLCV candles periodically and broadcast ``kline_update`` messages.

        Uses the exchange's ``subscribe_ohlcv`` when available (ccxt.pro WebSocket),
        otherwise falls back to polling ``get_ohlcv`` via REST every 30/60 seconds.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            timeframe: Candle timeframe, e.g. ``"1m"`` or ``"5m"``.
        """
        poll_interval = 30 if timeframe == "1m" else 60

        # Try WebSocket subscription first (ccxt.pro)
        if hasattr(self._exchange, "subscribe_ohlcv"):
            try:

                async def _ohlcv_cb(sym: str, tf: str, raw: list) -> None:
                    candles = [
                        {
                            "time": int(c[0] / 1000),  # ms → s for Lightweight Charts
                            "open": float(c[1]),
                            "high": float(c[2]),
                            "low": float(c[3]),
                            "close": float(c[4]),
                            "volume": float(c[5]),
                        }
                        for c in raw
                        if c and len(c) >= 6
                    ]
                    if candles:
                        async with self._lock:
                            self._kline_cache[(sym, tf)] = candles
                        await self._broadcast(
                            {
                                "type": "kline_update",
                                "symbol": sym,
                                "timeframe": tf,
                                "candles": candles,
                            }
                        )

                await self._exchange.subscribe_ohlcv(symbol, timeframe, _ohlcv_cb)
                return  # subscribe_ohlcv runs indefinitely — only reaches here on error
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "RealtimeHub: OHLCV WS subscription for {} {} failed ({}); using REST poll.",
                    symbol,
                    timeframe,
                    exc,
                )

        # REST polling fallback
        while self._running:
            try:
                if self._exchange:
                    df = await self._exchange.get_ohlcv(symbol, timeframe=timeframe, limit=200)
                    candles = [
                        {
                            "time": int(ts.timestamp()),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        }
                        for ts, row in df.iterrows()
                    ]
                    if candles:
                        async with self._lock:
                            self._kline_cache[(symbol, timeframe)] = candles
                        await self._broadcast(
                            {
                                "type": "kline_update",
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "candles": candles,
                            }
                        )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "RealtimeHub._ohlcv_poll_loop error for {} {}: {}",
                    symbol,
                    timeframe,
                    exc,
                )
            await asyncio.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Internal — paper state-change callback
    # ------------------------------------------------------------------

    def _on_paper_state_change(self) -> None:
        """Called synchronously by PaperExchange after a state change.

        Schedules an immediate async refresh + push so that position opens,
        closes, and SL/TP triggers appear in the dashboard within ~100 ms.
        """
        asyncio.ensure_future(self._handle_paper_state_change())

    async def _handle_paper_state_change(self) -> None:
        """Async handler for paper exchange state changes."""
        try:
            positions = await self._collect_positions()
            balance = await self._collect_balance()
            async with self._lock:
                self._latest_positions = positions
                self._latest_balance = balance
            await self._push_update({"type": "position_update"})
        except Exception as exc:
            logger.debug("RealtimeHub._handle_paper_state_change error: {}", exc)

    # ------------------------------------------------------------------
    # Internal — broadcast loop (1-second fallback)
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Push a full state snapshot to all connected clients every second."""
        while self._running:
            try:
                snapshot = self.get_snapshot()
                await self._broadcast(snapshot)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("RealtimeHub._broadcast_loop error: {}", exc)
            await asyncio.sleep(1)

    async def _position_update_loop(self) -> None:
        """Push position P&L updates to all connected clients every 2 seconds."""
        while self._running:
            try:
                await self._broadcast_position_update()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("RealtimeHub._position_update_loop error: {}", exc)
            await asyncio.sleep(2.0)

    async def _broadcast_position_update(self) -> None:
        """Push position P&L updates to connected dashboard clients."""
        import time as _time

        if self._position_manager is None:
            return
        try:
            trackers = await self._position_manager.get_all_positions()
            positions_data = []
            for tracker in trackers:
                pos = tracker.position
                positions_data.append(
                    {
                        "symbol": pos.symbol,
                        "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
                        "amount": float(pos.amount or 0),
                        "entry_price": float(pos.entry_price or 0),
                        "current_price": float(pos.current_price or 0),
                        "unrealized_pnl": float(pos.unrealized_pnl or 0),
                        "roe_pct": float(getattr(pos, "roe_pct", 0.0)),
                        "leverage": int(pos.leverage or 1),
                        "stop_loss": getattr(tracker, "stop_loss", None),
                        "take_profit": getattr(tracker, "take_profit", None),
                    }
                )

            if self._broadcast_fn is not None:
                await self._broadcast_fn(
                    {
                        "type": "positions_update",
                        "data": positions_data,
                        "timestamp": _time.time(),
                    }
                )
        except Exception as exc:
            logger.debug("Position broadcast error: {}", exc)

    # ------------------------------------------------------------------
    # Internal — data collection helpers
    # ------------------------------------------------------------------

    async def _collect_positions(self) -> List[dict]:
        """Collect positions using cached ticker prices (avoids N+1 REST calls)."""
        try:
            # Use get_positions_with_live_prices if available (PaperExchange)
            if self._paper_exchange is not None and hasattr(
                self._paper_exchange, "get_positions_with_live_prices"
            ):
                async with self._lock:
                    tickers_snapshot = dict(self._latest_tickers)
                raw_positions = self._paper_exchange.get_positions_with_live_prices(tickers_snapshot)
            elif self._exchange:
                raw_positions = await self._exchange.get_positions()
            else:
                return []
        except Exception as exc:
            logger.debug("RealtimeHub._collect_positions error: {}", exc)
            return list(self._latest_positions)

        positions: List[dict] = []
        for p in raw_positions:
            entry = float(p.entry_price or 0)
            current = float(p.current_price or entry)
            pnl = float(p.unrealized_pnl or 0)
            margin = float(p.margin or 0)
            pnl_pct = ((current - entry) / entry * 100.0) if entry else 0.0
            # For short positions pnl_pct is inverse
            if hasattr(p, "side") and str(p.side).lower() in ("short", "positionside.short"):
                pnl_pct = ((entry - current) / entry * 100.0) if entry else 0.0
            roe_pct = float(getattr(p, "roe_pct", 0.0))
            positions.append(
                {
                    "symbol": p.symbol,
                    "direction": str(p.side.value) if hasattr(p.side, "value") else str(p.side),
                    "entry_price": round(entry, 8),
                    "current_price": round(current, 8),
                    "mark_price": round(float(getattr(p, "mark_price", current)), 8),
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "roe_pct": round(roe_pct, 4),
                    "leverage": int(p.leverage or 1),
                    "margin": round(margin, 4),
                    "liquidation_price": round(float(p.liquidation_price or 0), 8),
                    "position_value": round(float(getattr(p, "position_value", current * p.amount)), 4),
                    "stop_loss": None,
                    "take_profit": None,
                    "strategy": "",
                    "amount": round(float(p.amount or 0), 8),
                    "funding_rate": getattr(p, "funding_rate", None),
                    "timestamp": int(p.timestamp or 0),
                }
            )

        # Enrich with SL/TP from open orders
        try:
            orders = await self._collect_orders()
            orders_by_sym: Dict[str, List[dict]] = {}
            for o in orders:
                orders_by_sym.setdefault(o["symbol"], []).append(o)
            for pos in positions:
                sym_orders = orders_by_sym.get(pos["symbol"], [])
                for o in sym_orders:
                    if o["type"] == "stop_loss" and pos["stop_loss"] is None:
                        pos["stop_loss"] = o.get("price")
                    if o["type"] == "take_profit" and pos["take_profit"] is None:
                        pos["take_profit"] = o.get("price")
                pos["open_orders"] = sym_orders
        except Exception:
            pass

        return positions

    async def _collect_orders(self) -> List[dict]:
        """Collect open orders."""
        try:
            if not self._exchange:
                return []
            raw = await self._exchange.get_open_orders()
            orders = []
            for o in raw:
                orders.append(
                    {
                        "id": o.id,
                        "symbol": o.symbol,
                        "type": o.type.value,
                        "side": o.side.value,
                        "amount": o.amount,
                        "price": o.price,
                        "status": o.status.value,
                    }
                )
            async with self._lock:
                self._latest_orders = orders
            return orders
        except Exception as exc:
            logger.debug("RealtimeHub._collect_orders error: {}", exc)
            return list(self._latest_orders)

    async def _collect_balance(self) -> Dict[str, float]:
        """Collect balance, computing equity from positions for paper mode."""
        try:
            if not self._exchange:
                return dict(self._latest_balance)
            bal = await self._exchange.get_balance()
            usdt_free = float(bal.usdt_free or 0)
            usdt_total = float(bal.usdt_total or usdt_free)

            # For paper mode: equity = free_balance + sum(margin) + sum(unrealized_pnl)
            if self._paper_exchange is not None:
                async with self._lock:
                    positions = list(self._latest_positions)
                total_margin = sum(p.get("margin", 0) for p in positions)
                total_upnl = sum(p.get("pnl", 0) for p in positions)
                equity = usdt_free + total_margin + total_upnl
                return {"usdt_free": round(usdt_free, 4), "usdt_total": round(equity, 4)}

            return {"usdt_free": round(usdt_free, 4), "usdt_total": round(usdt_total, 4)}
        except Exception as exc:
            logger.debug("RealtimeHub._collect_balance error: {}", exc)
            return dict(self._latest_balance)

    async def _collect_recent_trades(self, limit: int = 20) -> List[dict]:
        """Collect recent closed trades from the exchange or paper exchange.

        Args:
            limit: Maximum number of trades to return (default 20).

        Returns:
            List of trade dicts with symbol, side, entry/exit prices, PnL, etc.
        """
        try:
            # Try paper exchange trade history first
            if self._paper_exchange is not None and hasattr(self._paper_exchange, "get_trade_history"):
                raw_trades = await self._paper_exchange.get_trade_history()
                # Get the last N closed positions/trades
                closed_trades = [
                    t for t in raw_trades
                    if t.get("status") == "closed" or t.get("exit_price")
                ]
                closed_trades = closed_trades[-limit:] if len(closed_trades) > limit else closed_trades

                trades = []
                for t in closed_trades:
                    trades.append({
                        "symbol": t.get("symbol", ""),
                        "side": t.get("side", ""),
                        "entry_price": round(float(t.get("entry_price", 0)), 4),
                        "exit_price": round(float(t.get("exit_price", 0)), 4),
                        "amount": round(float(t.get("amount", 0)), 8),
                        "pnl": round(float(t.get("pnl", 0)), 4),
                        "pnl_pct": round(float(t.get("pnl_pct", 0)), 2),
                        "duration_mins": int(t.get("duration_mins", 0)),
                        "opened_at": int(t.get("opened_at", 0)),
                        "closed_at": int(t.get("closed_at", 0)),
                        "strategy": t.get("strategy", ""),
                    })
                return trades
            return list(self._latest_trades)
        except Exception as exc:
            logger.debug("RealtimeHub._collect_recent_trades error: {}", exc)
            return list(self._latest_trades)

    async def _collect_funding_rates(self) -> Dict[str, float]:
        """Collect funding rates for all active symbols.

        Returns:
            Dict mapping symbol → funding rate (as a percentage, e.g. 0.01 for 0.01%)
        """
        rates: Dict[str, float] = {}
        try:
            for sym in self._symbols:
                try:
                    if self._exchange and hasattr(self._exchange, "get_funding_rate"):
                        rate = await self._exchange.get_funding_rate(sym)
                        if rate is not None:
                            rates[sym] = float(rate)
                except Exception:
                    # Funding rate unavailable for this symbol — skip
                    pass
            return rates
        except Exception as exc:
            logger.debug("RealtimeHub._collect_funding_rates error: {}", exc)
            return dict(self._latest_funding_rates)

    def _build_portfolio_dict(self) -> dict:
        """Build a portfolio summary dict from the cached state."""
        balance = self._latest_balance
        positions = self._latest_positions
        unrealized_pnl = sum(p.get("pnl", 0) for p in positions)
        current_equity = balance.get("usdt_total", 0.0)
        if self._session_start_equity and self._session_start_equity > 0:
            daily_pnl = current_equity - self._session_start_equity
            daily_pnl_pct = (daily_pnl / self._session_start_equity) * 100.0
        else:
            daily_pnl = 0.0
            daily_pnl_pct = 0.0
        return {
            "equity": current_equity,
            "balance": balance.get("usdt_free", 0.0),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "daily_pnl": round(daily_pnl, 4),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "open_positions": len(positions),
        }

    # ------------------------------------------------------------------
    # Internal — broadcast helpers
    # ------------------------------------------------------------------

    async def _push_update(self, extra: Optional[dict] = None) -> None:
        """Push a targeted update (with the latest state) to all clients."""
        payload: dict = {
            "type": "update",
            "portfolio": self._build_portfolio_dict(),
            "positions": list(self._latest_positions),
            "open_orders": list(self._latest_orders),
            "recent_trades": list(self._latest_trades),
            "funding_rates": dict(self._latest_funding_rates),
        }
        if extra:
            payload.update(extra)
        await self._broadcast(payload)

    async def _broadcast(self, data: dict) -> None:
        """Send *data* to all dashboard WebSocket clients via the registered fn."""
        if self._broadcast_fn is None:
            return
        try:
            await self._broadcast_fn(data)
        except Exception as exc:
            logger.debug("RealtimeHub._broadcast error: {}", exc)
