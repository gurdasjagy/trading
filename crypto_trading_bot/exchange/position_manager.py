"""Position tracking and lifecycle management."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from .base_exchange import BaseExchange, Order, OrderSide, Position, PositionSide


class PositionState(str, Enum):
    """Lifecycle states for a tracked position.

    Transitions::

        PENDING_ENTRY → OPEN_UNPROTECTED → OPEN_PROTECTED
                                        ↓
                               PARTIALLY_CLOSED → PENDING_CLOSE → CLOSED
    """

    PENDING_ENTRY = "pending_entry"        # Order placed, not yet filled
    OPEN_UNPROTECTED = "open_unprotected"  # Filled but SL/TP not yet confirmed
    OPEN_PROTECTED = "open_protected"      # Has both SL and TP active on exchange
    PARTIALLY_CLOSED = "partially_closed"  # At least one TP level hit
    PENDING_CLOSE = "pending_close"        # Close order placed, awaiting fill
    CLOSED = "closed"                      # Position fully closed


@dataclass
class PositionTracker:
    """Wraps a :class:`Position` with strategy-level metadata for risk management."""

    position: Position
    strategy: str
    opened_at: datetime
    stop_loss: Optional[float] = None
    take_profit: List[float] = field(default_factory=list)  # multiple TP levels
    trailing_stop: Optional[float] = None  # distance in price units
    trailing_stop_active: bool = False
    highest_price: float = 0.0  # tracks the peak for long trailing stops
    lowest_price: float = 0.0  # tracks the trough for short trailing stops
    # ── Trailing take-profit ──────────────────────────────────────────────
    trailing_tp_distance: Optional[float] = None  # distance as price units
    trailing_tp_active: bool = False
    peak_price_for_tp: float = 0.0
    # ── Break-even stop-loss ──────────────────────────────────────────────
    break_even_activated: bool = False
    # ── Partial close history ─────────────────────────────────────────────
    partial_closes: List[dict] = field(default_factory=list)
    # ── Funding & realised P&L tracking ──────────────────────────────────
    funding_costs: float = 0.0       # cumulative funding costs paid
    realized_pnl: float = 0.0        # realised P&L from partial closes


class PositionManager:
    """Tracks open positions and manages risk overlays (SL / TP / trailing stop).

    All state is maintained in memory and synchronised from the exchange
    periodically via :meth:`sync_positions`.
    """

    # Seconds between exchange sync cycles in the background monitor.
    _MONITOR_INTERVAL_SECONDS = 10
    #: Default maximum position hold time in hours before auto-close (when not in profit)
    _MAX_HOLD_HOURS: float = 24.0

    # Seconds after placing an SL order during which a duplicate will not be placed.
    _SL_PLACEMENT_COOLDOWN: float = 60.0

    def __init__(self, exchange: BaseExchange, max_hold_hours: float = 24.0) -> None:
        self._exchange = exchange
        self._positions: Dict[str, PositionTracker] = {}  # symbol → tracker
        self._lock = asyncio.Lock()
        # Second lock serialises concurrent sync_positions() calls to prevent
        # race conditions that could lead to duplicate SL orders.
        self._sync_lock = asyncio.Lock()
        self._MAX_HOLD_HOURS = max_hold_hours
        # PnL milestone tracking: symbol → set of milestone levels already notified
        self._notified_pnl_levels: Dict[str, set] = {}
        # Optional TelegramAlerter injected externally or lazily created
        self._telegram_alerter: Optional[Any] = None
        # Watchdog protection cache: symbols where an SL order has been placed.
        # Prevents the watchdog from repeatedly placing emergency SL orders.
        # Maps symbol → (sl_order_id, placed_at_epoch_seconds)
        self._protected_symbols: Dict[str, Tuple[str, float]] = {}
        # Per-symbol timestamp of the most recent SL placement (by watchdog or executor).
        # Used by _is_sl_recently_placed() to enforce _SL_PLACEMENT_COOLDOWN.
        self._sl_placement_timestamps: Dict[str, float] = {}
        # Symbols whose trailing stop has already been tightened after 6h in profit.
        # Prevents repeated tightening across monitor cycles.
        self._trail_tightened_symbols: Set[str] = set()
        # Symbols for which the trade executor has filled the entry order but has
        # not yet placed the SL order.  The watchdog skips these to avoid placing
        # a duplicate emergency SL during the brief SL-placement window.
        self._pending_sl_symbols: Set[str] = set()

    # ------------------------------------------------------------------
    # Sync & query
    # ------------------------------------------------------------------

    def mark_position_protected(self, symbol: str, sl_order_id: str) -> None:
        """Mark *symbol* as having an active stop-loss *sl_order_id*.

        Called by the trade executor immediately after placing an SL order so
        the watchdog does not treat the position as unprotected and try to place
        a duplicate emergency SL.  The protection entry persists until the
        position closes or the watchdog confirms the SL is no longer active.

        Both the plain symbol (e.g. ``SOL/USDT``) and the Gate.io swap-suffix
        format (e.g. ``SOL/USDT:USDT``) are registered so that the watchdog
        can find the entry regardless of how the exchange reports the symbol.
        """
        now = time.time()
        self._protected_symbols[symbol] = (sl_order_id, now)
        # Record timestamp for the simple per-symbol SL placement cooldown so
        # the watchdog will not place a duplicate emergency SL for at least
        # _SL_PLACEMENT_COOLDOWN seconds after this call.
        self._sl_placement_timestamps[symbol] = now
        # Gate.io reports futures positions with the swap suffix (e.g. :USDT).
        # Register the swap variant so the watchdog finds the protection entry.
        if ":" not in symbol:
            _base, _sep, _quote = symbol.partition("/")
            if _sep and _quote:
                swap_symbol = f"{symbol}:{_quote}"
                self._protected_symbols[swap_symbol] = (sl_order_id, now)
                self._sl_placement_timestamps[swap_symbol] = now
        # Remove from pending set now that the SL is confirmed (or pre-registered).
        self._pending_sl_symbols.discard(symbol)
        if ":" not in symbol:
            _base2, _sep2, _quote2 = symbol.partition("/")
            if _sep2 and _quote2:
                self._pending_sl_symbols.discard(f"{symbol}:{_quote2}")
        logger.debug("Watchdog: marked {} as protected (SL order={})", symbol, sl_order_id)

    def mark_sl_pending(self, symbol: str) -> None:
        """Register *symbol* as having a pending SL placement.

        Called by the trade executor immediately after an entry order fills but
        before the SL order is sent to the exchange.  The watchdog will skip
        symbols in this set to avoid placing a duplicate emergency SL during
        the brief window between fill confirmation and SL order submission.

        Both the plain symbol and the Gate.io swap-suffix variant are registered.
        """
        self._pending_sl_symbols.add(symbol)
        if ":" not in symbol:
            _base, _sep, _quote = symbol.partition("/")
            if _sep and _quote:
                self._pending_sl_symbols.add(f"{symbol}:{_quote}")
        logger.debug("Watchdog: {} flagged as pending-SL", symbol)

    def clear_sl_pending(self, symbol: str) -> None:
        """Remove *symbol* from the pending-SL set.

        Called by the trade executor when SL placement fails so that the
        watchdog can place an emergency SL if needed.
        """
        self._pending_sl_symbols.discard(symbol)
        if ":" not in symbol:
            _base, _sep, _quote = symbol.partition("/")
            if _sep and _quote:
                self._pending_sl_symbols.discard(f"{symbol}:{_quote}")
        logger.debug("Watchdog: {} pending-SL flag cleared", symbol)

    async def sync_positions(self) -> List[Position]:
        """Fetch all open positions from the exchange and update local state.

        Serialised by ``_sync_lock`` to prevent concurrent invocations (e.g.
        from the background monitor and a WebSocket fill event) from causing
        race conditions that could result in duplicate SL orders or stale state.
        """
        async with self._sync_lock:
            return await self._sync_positions_impl()

    async def _sync_positions_impl(self) -> List[Position]:
        """Internal sync implementation — must only be called under ``_sync_lock``."""
        raw_positions = await self._exchange.get_positions()
        live_symbols = {p.symbol for p in raw_positions}

        _symbols_needing_be: list = []

        async with self._lock:
            # Remove closed positions
            closed = [s for s in self._positions if s not in live_symbols]
            for sym in closed:
                logger.info("Position closed (synced from exchange): {}", sym)
                del self._positions[sym]
                # Clean up per-symbol state dicts to prevent unbounded growth
                self._notified_pnl_levels.pop(sym, None)
                self._protected_symbols.pop(sym, None)
                self._sl_placement_timestamps.pop(sym, None)
                self._trail_tightened_symbols.discard(sym)

            # Update existing / add new
            for pos in raw_positions:
                if pos.symbol in self._positions:
                    tracker = self._positions[pos.symbol]
                    prev_amount = tracker.position.amount
                    tracker.position = pos

                    # Detect partial close from native exchange TP order fill (Task 2)
                    if prev_amount > 0 and pos.amount < prev_amount and pos.amount > 0:
                        closed_amount = prev_amount - pos.amount
                        logger.info(
                            "Detected partial fill (likely TP) for {}: {} -> {} (closed {})",
                            pos.symbol, prev_amount, pos.amount, closed_amount,
                        )
                        # Pop the first TP level since it was likely hit
                        if tracker.take_profit:
                            hit_tp = tracker.take_profit.pop(0)
                            logger.info("TP level {} consumed for {}", hit_tp, pos.symbol)
                            # Activate break-even after first TP hit
                            if not tracker.break_even_activated:
                                _symbols_needing_be.append(pos.symbol)
                        # Record the partial close
                        tracker.partial_closes.append({
                            "ts": datetime.now(tz=timezone.utc).isoformat(),
                            "amount": closed_amount,
                            "price": pos.current_price,
                            "pnl": 0.0,  # Will be calculated from exchange data
                        })
                else:
                    tracker = PositionTracker(
                        position=pos,
                        strategy="unknown",
                        opened_at=datetime.now(tz=timezone.utc),
                        highest_price=pos.current_price,
                        lowest_price=pos.current_price,
                    )
                    self._positions[pos.symbol] = tracker
                    logger.info(
                        "New position detected (external): {} {}", pos.symbol, pos.side.value
                    )

        # Activate break-even outside the lock to avoid deadlocks
        for sym in _symbols_needing_be:
            await self.activate_break_even(sym)

        return raw_positions

    async def get_position(self, symbol: str) -> Optional[PositionTracker]:
        """Return the :class:`PositionTracker` for *symbol*, or *None* if flat."""
        async with self._lock:
            return self._positions.get(symbol)

    def get_position_sync(self, symbol: str) -> Optional[PositionTracker]:
        """Return the :class:`PositionTracker` for *symbol* synchronously (no lock).

        This is a non-blocking, non-async read of the in-memory cache.  It is
        safe to call from synchronous contexts (e.g. the engine's
        ``_has_conflicting_position`` method) where awaiting would not be
        possible.  The returned snapshot may be slightly stale between sync
        cycles; use :meth:`get_position` for authoritative reads.
        """
        return self._positions.get(symbol)

    async def get_all_positions(self) -> List[PositionTracker]:
        """Return all currently tracked positions."""
        async with self._lock:
            return list(self._positions.values())

    async def verify_protection(self) -> Dict[str, Dict[str, Any]]:
        """Verify that every tracked position has active SL and TP orders on the exchange.

        Queries the exchange for open orders and cross-references them against
        tracked positions.  Returns a snapshot of protection status for each
        symbol so callers can decide whether to re-place missing orders.

        Returns:
            Dict mapping symbol → ``{"has_sl": bool, "has_tp": bool,
            "sl_price": float | None, "tp_prices": list[float]}``.
        """
        result: Dict[str, Dict[str, Any]] = {}
        async with self._lock:
            symbols = list(self._positions.keys())

        for symbol in symbols:
            has_sl = False
            has_tp = False
            sl_price: Optional[float] = None
            tp_prices: List[float] = []
            try:
                open_orders = await self._exchange.get_open_orders(symbol)
                for order in open_orders:
                    order_type = str(getattr(order, "type", "")).lower()
                    stop_price = getattr(order, "stop_price", None) or getattr(order, "price", None)
                    # Detect stop-loss: stop-market / stop-limit reduce-only orders
                    if any(x in order_type for x in ("stop", "sl")):
                        has_sl = True
                        if stop_price:
                            sl_price = float(stop_price)
                    # Detect take-profit: take-profit or limit reduce-only orders
                    if any(x in order_type for x in ("take_profit", "tp", "take profit")):
                        has_tp = True
                        if stop_price:
                            tp_prices.append(float(stop_price))
                    # Fallback: plain limit orders flagged as reduce-only act as TP
                    if "limit" in order_type and getattr(order, "reduce_only", False):
                        has_tp = True
                        price = getattr(order, "price", None)
                        if price:
                            tp_prices.append(float(price))
            except Exception as exc:
                logger.debug("verify_protection fetch error for {}: {}", symbol, exc)

            result[symbol] = {
                "has_sl": has_sl,
                "has_tp": has_tp,
                "sl_price": sl_price,
                "tp_prices": tp_prices,
            }
            logger.debug(
                "verify_protection {}: has_sl={} has_tp={} sl_price={} tp_prices={}",
                symbol, has_sl, has_tp, sl_price, tp_prices,
            )

        return result



    async def open_position(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        strategy: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> PositionTracker:
        """Open a new position via a market order and register it.

        Optionally attach *sl* (stop-loss price) and *tp* (take-profit price)
        as metadata so the background monitor can enforce them.
        """
        await self._exchange.create_market_order(symbol, side, amount, {"reduceOnly": False})
        # Refresh the actual position from exchange
        position = await self._exchange.get_position(symbol)
        if position is None:
            raise RuntimeError(f"Position not found after opening {symbol}")

        tracker = PositionTracker(
            position=position,
            strategy=strategy,
            opened_at=datetime.now(tz=timezone.utc),
            stop_loss=sl,
            take_profit=[tp] if tp is not None else [],
            highest_price=position.current_price,
            lowest_price=position.current_price,
        )
        async with self._lock:
            self._positions[symbol] = tracker

        logger.info(
            "[{}] Position opened: {} {} amount={} sl={} tp={}",
            strategy,
            symbol,
            side.value,
            amount,
            sl,
            tp,
        )
        return tracker

    async def close_position(self, symbol: str, reason: str = "") -> dict:
        """Close the position for *symbol* entirely.

        Returns a summary dict with P&L information.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)

        if tracker is None:
            logger.warning("close_position: no tracked position for {}", symbol)
            return {"symbol": symbol, "closed": False, "reason": "not_found"}

        order: Order = await self._exchange.close_position(symbol)

        # Total PnL = realized from any prior partial closes + the final unrealized amount
        # that is now being closed.  Computed before lock release so values are consistent.
        total_pnl = tracker.realized_pnl + tracker.position.unrealized_pnl

        close_time = datetime.now(tz=timezone.utc)

        async with self._lock:
            self._positions.pop(symbol, None)
            self._notified_pnl_levels.pop(symbol, None)
            self._protected_symbols.pop(symbol, None)
            self._sl_placement_timestamps.pop(symbol, None)
            self._trail_tightened_symbols.discard(symbol)

        pnl = tracker.position.unrealized_pnl
        logger.info("Position closed: {} reason='{}' pnl={:.4f}", symbol, reason, pnl)

        # Persist trade record to the database (populates the Trades and Performance tabs)
        asyncio.create_task(
            self._persist_trade_history(tracker, close_time, total_pnl, reason, order)
        )

        # Fire-and-forget Telegram receipt
        alerter = self._get_telegram_alerter()
        if alerter is not None:
            asyncio.create_task(
                alerter.send_trade_closed(symbol, total_pnl, reason=reason)
            )

        return {
            "symbol": symbol,
            "closed": True,
            "reason": reason,
            "pnl": pnl,
            "close_order_id": order.id,
        }

    async def reduce_position(self, symbol: str, amount: float) -> dict:
        """Partially close a position by *amount* (in base currency units).

        Args:
            symbol: Trading symbol, e.g. ``"BTC/USDT"``.
            amount: Base-currency amount to close (must be > 0 and ≤ position size).

        Returns:
            dict with ``symbol``, ``reduced_amount``, ``remaining_amount``, ``order_id``.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)

        if tracker is None:
            logger.warning("reduce_position: no tracked position for {}", symbol)
            return {"symbol": symbol, "reduced": False, "reason": "not_found"}

        pos_amount = tracker.position.amount
        if amount <= 0 or amount > pos_amount:
            raise ValueError(
                f"reduce_position: amount {amount} out of range (position size={pos_amount})"
            )

        order: Order = await self._exchange.close_position(symbol, amount)
        remaining = pos_amount - amount

        # Update tracked position amount; remove if fully closed
        # Use epsilon comparison to handle floating-point arithmetic edge cases
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is not None:
                if remaining < 1e-8:
                    self._positions.pop(symbol, None)
                else:
                    tracker.position.amount = remaining

        logger.info(
            "Position reduced: {} amount={} remaining={} order_id={}",
            symbol,
            amount,
            remaining,
            order.id,
        )
        return {
            "symbol": symbol,
            "reduced": True,
            "reduced_amount": round(amount, 8),
            "remaining_amount": round(remaining, 8),
            "order_id": order.id,
        }

    async def close_all_positions(self, reason: str = "") -> List[dict]:
        """Close every tracked open position.

        Returns a list of close-result dicts (one per position).
        """
        async with self._lock:
            symbols = list(self._positions.keys())

        results = []
        for sym in symbols:
            try:
                result = await self.close_position(sym, reason)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to close {}: {}", sym, exc)
                results.append({"symbol": sym, "closed": False, "error": str(exc)})
        return results

    # ------------------------------------------------------------------
    # Trade history persistence
    # ------------------------------------------------------------------

    async def _persist_trade_history(
        self,
        tracker: "PositionTracker",
        close_time: datetime,
        total_pnl: float,
        exit_reason: str,
        close_order: "Order",
    ) -> None:
        """Persist a closed trade to the ``trade_history`` and ``daily_performance`` tables.

        Called as a fire-and-forget task from :meth:`close_position` so that
        the closing path is never delayed by database I/O.
        """
        try:
            from data.storage.models import (
                DailyPerformance,
                TradeHistory,
                create_tables,
                get_async_session,
            )
        except ImportError:
            logger.debug("DB models unavailable — skipping trade history persistence")
            return

        try:
            await create_tables()

            pos = tracker.position
            entry_time: datetime = tracker.opened_at
            if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)

            duration_seconds = int((close_time - entry_time).total_seconds())

            # Use the order price if available; fall back to current_price then entry_price
            exit_price: float = (
                float(close_order.price)
                if getattr(close_order, "price", None)
                else float(pos.current_price or pos.entry_price)
            )

            pnl_pct: float = 0.0
            if pos.entry_price > 0 and pos.margin > 0:
                pnl_pct = (total_pnl / pos.margin) * 100.0

            exchange_name: str = ""
            try:
                exchange_name = self._exchange.name  # type: ignore[attr-defined]
            except Exception:
                pass

            # ── TradeHistory record ───────────────────────────────────
            trade_record = TradeHistory(
                order_id=getattr(close_order, "id", None),
                symbol=pos.symbol,
                side=pos.side.value,
                order_type="market",
                size=float(pos.amount),
                price=float(pos.entry_price),
                filled_price=exit_price,
                leverage=int(pos.leverage),
                pnl=round(total_pnl, 8),
                pnl_pct=round(pnl_pct, 4),
                fees=float(getattr(close_order, "fee", 0.0)),
                entry_time=entry_time,
                close_time=close_time,
                duration=duration_seconds,
                strategy=tracker.strategy or "unknown",
                exit_reason=exit_reason or "manual",
                exchange=exchange_name,
                stop_loss=tracker.stop_loss,
                take_profit=tracker.take_profit[0] if tracker.take_profit else None,
                funding_fee=float(tracker.funding_costs),
                realized_pnl=round(total_pnl, 8),
                margin_used=float(pos.margin),
            )

            today_str = close_time.strftime("%Y-%m-%d")

            async with get_async_session() as session:
                session.add(trade_record)

                # ── DailyPerformance upsert ───────────────────────────
                from sqlalchemy import select
                dp_query = select(DailyPerformance).where(DailyPerformance.date == today_str)
                dp_result = await session.execute(dp_query)
                dp = dp_result.scalars().first()
                is_win = total_pnl >= 0
                if dp is None:
                    dp = DailyPerformance(
                        date=today_str,
                        starting_balance=0.0,
                        ending_balance=0.0,
                        pnl=round(total_pnl, 8),
                        trades_count=1,
                        wins=1 if is_win else 0,
                        losses=0 if is_win else 1,
                        win_rate=100.0 if is_win else 0.0,
                        fees_paid=float(getattr(close_order, "fee", 0.0)),
                    )
                    session.add(dp)
                else:
                    dp.pnl = round((dp.pnl or 0.0) + total_pnl, 8)
                    dp.trades_count = (dp.trades_count or 0) + 1
                    if is_win:
                        dp.wins = (dp.wins or 0) + 1
                    else:
                        dp.losses = (dp.losses or 0) + 1
                    total_finished = (dp.wins or 0) + (dp.losses or 0)
                    dp.win_rate = round((dp.wins or 0) / total_finished * 100, 2) if total_finished > 0 else 0.0
                    dp.fees_paid = round((dp.fees_paid or 0.0) + float(getattr(close_order, "fee", 0.0)), 8)

            logger.debug("Trade history persisted: {} pnl={:.4f}", pos.symbol, total_pnl)
        except Exception as exc:
            logger.warning("Failed to persist trade history for {}: {}", getattr(tracker.position, "symbol", "?"), exc)

    # ------------------------------------------------------------------
    # Risk overlay updates
    # ------------------------------------------------------------------

    async def update_stop_loss(self, symbol: str, new_sl: float) -> bool:
        """Update the in-memory stop-loss price for *symbol*.

        Returns *True* if the position exists and was updated.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None:
                return False
            tracker.stop_loss = new_sl
        logger.info("Stop-loss updated for {}: {}", symbol, new_sl)
        return True

    async def update_take_profit(self, symbol: str, new_tp: float) -> bool:
        """Append a take-profit level for *symbol*.

        Returns *True* if the position exists and was updated.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None:
                return False
            if new_tp not in tracker.take_profit:
                tracker.take_profit.append(new_tp)
                tracker.take_profit.sort()
        logger.info("Take-profit updated for {}: {}", symbol, new_tp)
        return True

    async def activate_break_even(self, symbol: str, buffer_pct: float = 0.001) -> bool:
        """Move the stop-loss to entry price ± *buffer_pct* for break-even protection.

        For a **long** position the new SL is ``entry * (1 + buffer_pct)``.
        For a **short** position the new SL is ``entry * (1 - buffer_pct)``.

        Should only be called after the first TP level has been hit.  Cancels
        any existing SL orders on the exchange and places a fresh one sized to
        the *current* position amount so the order is not over/under-sized after
        a partial close.

        Args:
            symbol: Trading pair symbol.
            buffer_pct: Fraction of entry price added as a buffer (default 0.1 %).

        Returns:
            ``True`` if the position exists and break-even was activated.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None:
                return False
            if tracker.break_even_activated:
                return True  # already active

            entry = tracker.position.entry_price
            if entry <= 0:
                return False

            current_amount = tracker.position.amount
            if current_amount <= 0:
                logger.warning(
                    "Cannot activate break-even for {} — position amount is 0", symbol
                )
                return False

            is_long = tracker.position.side == PositionSide.LONG
            if is_long:
                be_price = entry * (1.0 + buffer_pct)
            else:
                be_price = entry * (1.0 - buffer_pct)

        # Cancel existing SL orders and place a new one — done outside the lock
        # to avoid holding it over slow I/O.
        try:
            open_orders = await self._exchange.get_open_orders(symbol)
            for order in open_orders:
                order_type_str = str(getattr(order, "type", "")).lower()
                if "stop" in order_type_str:
                    try:
                        await self._exchange.cancel_order(order.id, symbol)
                        logger.debug(
                            "Cancelled old SL order {} for break-even activation", order.id
                        )
                    except Exception as cancel_exc:
                        logger.debug(
                            "Could not cancel SL order {} for break-even activation: {}",
                            order.id, cancel_exc,
                        )

            sl_side = OrderSide.SELL if is_long else OrderSide.BUY
            sl_order = await self._exchange.create_stop_loss_order(
                symbol, sl_side, current_amount, be_price
            )
            self.mark_position_protected(symbol, sl_order.id)
        except Exception as exc:
            logger.error("Failed to place break-even SL order for {}: {}", symbol, exc)
            # Still update in-memory SL so the software-side monitor can enforce it.

        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None:
                return False
            tracker.stop_loss = be_price
            tracker.break_even_activated = True

        logger.info(
            "Break-even SL activated for {}: entry={:.4f} be_sl={:.4f} amount={:.4f}",
            symbol,
            entry,
            be_price,
            current_amount,
        )
        return True

    async def update_trailing_take_profit(
        self, symbol: str, current_price: float
    ) -> Optional[float]:
        """Update the trailing take-profit and close if price retraces enough.

        When ``trailing_tp_active`` is ``True`` and the position makes a new
        high (long) / low (short), the ``peak_price_for_tp`` is updated.  If
        the price then retraces by ``trailing_tp_distance`` from the peak, the
        position is closed and the close price is returned.

        Args:
            symbol: Trading pair symbol.
            current_price: Latest market price.

        Returns:
            The close price if the trailing TP was triggered, otherwise ``None``.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None or not tracker.trailing_tp_active:
                return None
            if tracker.trailing_tp_distance is None or tracker.trailing_tp_distance <= 0:
                return None

            distance = tracker.trailing_tp_distance
            side = tracker.position.side

            if side == PositionSide.LONG:
                # Update peak if new high
                if current_price > tracker.peak_price_for_tp:
                    tracker.peak_price_for_tp = current_price
                # Close if price has retraced from peak
                if tracker.peak_price_for_tp > 0 and current_price <= (
                    tracker.peak_price_for_tp - distance
                ):
                    trigger_price = current_price
                else:
                    return None
            else:  # SHORT
                # Update peak (lowest price) if new low
                if tracker.peak_price_for_tp == 0.0 or current_price < tracker.peak_price_for_tp:
                    tracker.peak_price_for_tp = current_price
                # Close if price has retraced up from trough
                if tracker.peak_price_for_tp > 0 and current_price >= (
                    tracker.peak_price_for_tp + distance
                ):
                    trigger_price = current_price
                else:
                    return None

        logger.info(
            "Trailing TP triggered for {} at price={:.4f} peak={:.4f} distance={:.4f}",
            symbol,
            trigger_price,
            tracker.peak_price_for_tp,
            distance,
        )
        await self.close_position(symbol, reason="trailing_take_profit")
        return trigger_price

    async def record_partial_close(
        self, symbol: str, amount: float, price: float, pnl: float
    ) -> bool:
        """Record a partial position close in the tracker.

        Args:
            symbol: Trading pair symbol.
            amount: Amount closed (base currency units).
            price: Fill price.
            pnl: Realised P&L from this partial close.

        Returns:
            ``True`` if the position was found and the record was added.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None:
                return False
            tracker.partial_closes.append(
                {
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "amount": amount,
                    "price": price,
                    "pnl": pnl,
                }
            )
            tracker.realized_pnl += pnl
        logger.info(
            "Partial close recorded for {}: amount={} price={} pnl={:.4f}",
            symbol,
            amount,
            price,
            pnl,
        )
        return True

    async def record_funding_cost(self, symbol: str, cost: float) -> bool:
        """Accumulate a funding cost payment for *symbol*.

        Args:
            symbol: Trading pair symbol.
            cost: Funding cost amount.  Negative means the position paid funding.

        Returns:
            ``True`` if the position was found and the cost was recorded.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None:
                return False
            tracker.funding_costs += cost
        logger.debug("Funding cost recorded for {}: cost={:.4f}", symbol, cost)
        return True

    async def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """Ratchet the trailing stop for *symbol* given *current_price*.

        High-water mark (for longs) / low-water mark (for shorts) is updated
        whenever the price makes a new extreme.  The effective SL is then
        always recomputed as ``high_water - distance`` (long) or
        ``low_water + distance`` (short) — not from the current price — so
        that the trailing stop correctly re-activates after a break-even SL
        adjustment that may have moved the SL below the trailing value.

        Returns the new effective stop-loss price if the SL was updated,
        or *None* if no change occurred or the position is not tracked.
        """
        async with self._lock:
            tracker = self._positions.get(symbol)
            if tracker is None or tracker.trailing_stop is None:
                return None

            distance = tracker.trailing_stop
            side = tracker.position.side

            if side == PositionSide.LONG:
                # Always update high-water mark when a new high is seen.
                if current_price > tracker.highest_price:
                    tracker.highest_price = current_price
                # Trail SL from the highest price seen (not from current price).
                new_sl = tracker.highest_price - distance
                # Only move SL up; never let it slip back.
                # Default to 0.0 so any positive trailing SL always passes the > check.
                current_sl = tracker.stop_loss or 0.0
                if new_sl > current_sl:
                    tracker.stop_loss = new_sl
                    tracker.trailing_stop_active = True
                    logger.debug("Trailing stop raised for {}: sl={:.4f}", symbol, new_sl)
                    return new_sl
            else:
                # For shorts, update low-water mark when a new low is seen.
                if current_price < tracker.lowest_price or tracker.lowest_price == 0.0:
                    tracker.lowest_price = current_price
                # Trail SL from the lowest price seen.
                new_sl = tracker.lowest_price + distance
                # Only move SL down; never let it creep back up.
                # Default to +inf so any finite trailing SL always passes the < check.
                current_sl = tracker.stop_loss if tracker.stop_loss is not None else float("inf")
                if new_sl < current_sl:
                    tracker.stop_loss = new_sl
                    tracker.trailing_stop_active = True
                    logger.debug("Trailing stop lowered for {}: sl={:.4f}", symbol, new_sl)
                    return new_sl

        return None

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    def _is_sl_recently_placed(self, symbol: str) -> bool:
        """Return ``True`` if an SL order was placed for *symbol* within the cooldown window.

        Checks ``_sl_placement_timestamps`` which is written by both
        :meth:`mark_position_protected` (called by the trade executor) and the
        watchdog itself immediately after placing an emergency SL.  This
        prevents duplicate SL orders that can arise because newly placed
        exchange orders do not appear instantly in :meth:`get_open_orders`.
        """
        last_placed = self._sl_placement_timestamps.get(symbol, 0.0)
        return (time.time() - last_placed) < self._SL_PLACEMENT_COOLDOWN

    async def monitor_positions(self) -> None:
        """Background task: sync positions and enforce SL / TP / trailing stops.

        Launch once via ``asyncio.create_task(manager.monitor_positions())``.
        """
        logger.info("PositionManager monitor started")
        while True:
            try:
                await asyncio.sleep(self._MONITOR_INTERVAL_SECONDS)
                await self.sync_positions()
                await self._enforce_risk_overlays()
                await self._check_pnl_milestones()
            except asyncio.CancelledError:
                logger.info("PositionManager monitor stopped")
                break
            except Exception as exc:
                logger.error("Error in position monitor: {}", exc)

    async def watchdog_unprotected_positions(self) -> None:
        """Background task: detect positions without SL orders and place emergency SL.

        Runs every 5 seconds.  If a position exists on the exchange but has no
        corresponding stop-loss order, places an emergency SL at 3% from current price.

        Protection cache:
        - After successfully placing an SL (either via trade executor via
          :meth:`mark_position_protected` or by this watchdog as an emergency),
          the symbol is added to ``_protected_symbols`` with the order ID and
          timestamp.
        - The watchdog skips symbols that are already "protected" for the next
          ``PROTECTION_COOLDOWN`` seconds.  After the cooldown expires it
          re-checks whether the position still exists and whether an active SL
          order can be found (including Gate.io price-triggered orders).
        - Before placing a new emergency SL any existing orders for that symbol
          are cancelled first, preventing orphan SL accumulation.
        """
        PROTECTION_COOLDOWN = 300  # seconds (5 min) before re-verifying a protected symbol
        logger.info("Unprotected position watchdog started")
        while True:
            try:
                await asyncio.sleep(5)
                positions = await self._exchange.get_positions()
                if not positions:
                    # Remove entries for symbols that no longer have open positions
                    self._protected_symbols.clear()
                    continue

                live_symbols: Set[str] = {p.symbol for p in positions}

                # Expire protection entries for closed positions
                for sym in list(self._protected_symbols.keys()):
                    if sym not in live_symbols:
                        del self._protected_symbols[sym]
                        logger.debug("Watchdog: cleared protection for closed position {}", sym)

                open_orders = await self._exchange.get_open_orders()
                # Build set of symbols that have at least one SL-type order.
                # get_open_orders() now returns both regular orders AND Gate.io
                # price-triggered orders (SL/TP conditional orders), so this
                # detection is reliable.
                # Note: only stop/reduce-only orders count as SL protection;
                # take-profit-only orders are NOT counted since they leave the
                # downside unprotected.
                symbols_with_sl: set = set()
                for o in open_orders:
                    order_type_str = str(getattr(o, "type", "")).lower()
                    info = getattr(o, "info", {}) or {}
                    is_sl = (
                        "stop" in order_type_str
                        or info.get("reduceOnly", False)
                        or "trigger" in str(info.get("triggerPrice", "")).lower()
                    )
                    if is_sl:
                        symbols_with_sl.add(o.symbol)

                now = time.time()
                for pos in positions:
                    # Normalize symbol for display (strip swap suffix like :USDT)
                    display_symbol = pos.symbol.split(":")[0] if ":" in pos.symbol else pos.symbol

                    if pos.symbol in symbols_with_sl:
                        # Position has a live SL order — update protection entry
                        # so the cooldown is refreshed and we don't re-trigger.
                        self._protected_symbols[pos.symbol] = ("detected", now)
                        continue

                    # Skip positions where the trade executor has filled the entry
                    # but has not yet placed the SL order (pending SL window).
                    if pos.symbol in self._pending_sl_symbols:
                        logger.debug(
                            "Watchdog: {} has a pending SL placement — skipping",
                            display_symbol,
                        )
                        continue

                    # Check protection cooldown
                    if pos.symbol in self._protected_symbols:
                        _order_id, _placed_at = self._protected_symbols[pos.symbol]
                        elapsed = now - _placed_at
                        if elapsed < PROTECTION_COOLDOWN:
                            logger.debug(
                                "Watchdog: {} is protected (SL order={}, {:.0f}s ago) — skipping",
                                display_symbol, _order_id, elapsed,
                            )
                            continue
                        # Cooldown expired — re-verify by fetching orders for this symbol.
                        # This includes Gate.io price-triggered orders (stop/TP).
                        try:
                            emerg_orders = await self._exchange.get_open_orders(pos.symbol)
                            if emerg_orders:
                                # At least one order (SL/TP or regular) still exists
                                self._protected_symbols[pos.symbol] = (_order_id, now)
                                logger.debug(
                                    "Watchdog: orders still active for {} — refreshing cooldown",
                                    display_symbol,
                                )
                                continue
                        except Exception as verify_exc:
                            logger.debug(
                                "Watchdog: could not verify SL for {}: {}",
                                display_symbol, verify_exc,
                            )
                        # No active orders found — remove protection so emergency SL is placed
                        del self._protected_symbols[pos.symbol]

                    logger.critical(
                        "🚨 UNPROTECTED POSITION DETECTED: {} {} amount={} - placing emergency SL!",
                        display_symbol, pos.side.value, pos.amount,
                    )
                    # Guard against the ordering gap: an SL may have just been
                    # placed by the trade executor but not yet visible in
                    # get_open_orders().  Skip if within the cooldown window.
                    if self._is_sl_recently_placed(pos.symbol):
                        logger.debug(
                            "Watchdog: skipping emergency SL for {} — placed {:.0f}s ago",
                            display_symbol,
                            time.time() - self._sl_placement_timestamps.get(pos.symbol, 0.0),
                        )
                        continue
                    try:
                        # Cancel ALL existing trigger orders for this symbol first
                        # to avoid accumulating orphan SL orders on the exchange.
                        try:
                            await self._exchange.cancel_all_orders(pos.symbol)
                            logger.debug(
                                "Watchdog: cancelled existing orders for {} before placing emergency SL",
                                display_symbol,
                            )
                        except Exception as cancel_exc:
                            logger.debug(
                                "Watchdog: could not cancel orders for {}: {}",
                                display_symbol, cancel_exc,
                            )

                        is_long = pos.side == PositionSide.LONG
                        price = pos.current_price or pos.mark_price or pos.entry_price
                        emergency_sl = price * (0.97 if is_long else 1.03)
                        close_side = OrderSide.SELL if is_long else OrderSide.BUY
                        try:
                            sl_order = await self._exchange.create_stop_loss_order(
                                pos.symbol, close_side, pos.amount, emergency_sl
                            )
                        except Exception as sl_exc:
                            exc_str = str(sl_exc).lower()
                            if "trigger" in exc_str or "auto_trigger" in exc_str:
                                # Gate.io rejected the trigger price; retry with a
                                # tighter offset (2% instead of 3%) to ensure the
                                # price is on the correct side of the current market.
                                logger.warning(
                                    "Watchdog: trigger price error for {} — retrying with 2% offset: {}",
                                    display_symbol, sl_exc,
                                )
                                emergency_sl = price * (0.98 if is_long else 1.02)
                                sl_order = await self._exchange.create_stop_loss_order(
                                    pos.symbol, close_side, pos.amount, emergency_sl
                                )
                            else:
                                raise
                        # Mark symbol as protected with the new order ID and timestamp
                        self._protected_symbols[pos.symbol] = (sl_order.id, now)
                        # Record in placement-timestamp dict so _is_sl_recently_placed()
                        # blocks any further duplicate within the cooldown window.
                        self._sl_placement_timestamps[pos.symbol] = now
                        logger.info(
                            "Emergency SL placed for {} at {:.4f} (order={})",
                            display_symbol, emergency_sl, sl_order.id,
                        )
                    except Exception as exc:
                        logger.error(
                            "CRITICAL: Failed to place emergency SL for {}: {}",
                            display_symbol, exc,
                        )
                        # Last resort: close the position entirely
                        try:
                            await self._exchange.close_position(pos.symbol)
                            logger.warning("Emergency close executed for {}", display_symbol)
                        except Exception as close_exc:
                            logger.critical(
                                "FAILED TO CLOSE UNPROTECTED POSITION {}: {}",
                                display_symbol, close_exc,
                            )
            except asyncio.CancelledError:
                logger.info("Unprotected position watchdog stopped")
                break
            except Exception as exc:
                logger.error("Watchdog error: {}", exc)

    async def start_ws_monitor(self, exchange: "BaseExchange") -> None:
        """Start WebSocket-based position monitoring (preferred over polling).

        Falls back to polling-based monitor_positions() if WS is unavailable.
        """

        async def _handle_user_data(event: dict) -> None:
            event_type = event.get("type", "")

            if event_type == "order_update":
                order = event.get("order")
                if order and getattr(order, "status", None) is not None:
                    from .base_exchange import OrderStatus
                    if order.status == OrderStatus.CLOSED:
                        # An order filled - sync positions immediately
                        await self.sync_positions()
                        logger.info(
                            "Order filled (WS): {} {} {}",
                            order.symbol,
                            getattr(order.side, "value", order.side),
                            order.filled,
                        )

            elif event_type == "position_update":
                positions = event.get("positions", [])
                # Direct position update from exchange
                live_symbols = {p.symbol for p in positions}
                async with self._lock:
                    closed = [s for s in self._positions if s not in live_symbols]
                    for sym in closed:
                        logger.info("Position closed (WS): {}", sym)
                        del self._positions[sym]
                    for pos in positions:
                        if pos.symbol in self._positions:
                            self._positions[pos.symbol].position = pos

            elif event_type == "poll_update":
                # Fallback polling mode - no action needed here
                pass

        try:
            logger.info("Starting WebSocket position monitor")
            # Run WS subscription and periodic reconciliation in parallel
            await asyncio.gather(
                exchange.subscribe_user_data(_handle_user_data),
                self._periodic_reconciliation(),
            )
        except Exception as exc:
            logger.warning("WS monitor failed: {} - falling back to polling", exc)
            await self.monitor_positions()

    async def _periodic_reconciliation(self) -> None:
        """Backup reconciliation loop that runs alongside WebSocket monitoring."""
        while True:
            await asyncio.sleep(60)  # Every 60 seconds
            try:
                await self.sync_positions()
                await self._enforce_risk_overlays()
                await self._check_pnl_milestones()
            except Exception as exc:
                logger.debug("Periodic reconciliation error: {}", exc)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    async def calculate_total_pnl(self) -> float:
        """Return the sum of unrealised P&L across all open positions."""
        async with self._lock:
            return sum(t.position.unrealized_pnl for t in self._positions.values())

    async def get_position_summary(self) -> dict:
        """Return a high-level summary of all open positions."""
        from datetime import datetime, timezone

        async with self._lock:
            trackers = list(self._positions.values())

        total_pnl = sum(t.position.unrealized_pnl for t in trackers)
        now = datetime.now(tz=timezone.utc)
        positions = []
        for t in trackers:
            p = t.position
            time_open_secs = int((now - t.opened_at).total_seconds())
            positions.append(
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "amount": p.amount,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "mark_price": p.mark_price or p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": p.leverage,
                    "margin": p.margin,
                    "liquidation_price": p.liquidation_price,
                    "roe_pct": p.roe_pct,
                    "position_value": p.position_value,
                    "funding_rate": p.funding_rate,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "strategy": t.strategy,
                    "opened_at": t.opened_at.isoformat(),
                    "time_open": time_open_secs,
                    "trailing_tp_active": t.trailing_tp_active,
                    "break_even_activated": t.break_even_activated,
                    "funding_costs": t.funding_costs,
                    "realized_pnl": t.realized_pnl,
                }
            )
        return {
            "open_positions": len(trackers),
            "total_unrealized_pnl": total_pnl,
            "positions": positions,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_telegram_alerter(self) -> Optional[Any]:
        """Lazily initialise and return the :class:`~monitoring.alerting.TelegramAlerter`."""
        if self._telegram_alerter is None:
            try:
                from monitoring.alerting import TelegramAlerter
                self._telegram_alerter = TelegramAlerter()
            except Exception as exc:
                logger.debug("TelegramAlerter unavailable: {}", exc)
        return self._telegram_alerter

    async def _check_pnl_milestones(self) -> None:
        """Fire Telegram alerts when a position crosses a PnL milestone for the first time.

        Milestones are ±5 %, ±10 %, ±20 %, ±50 %.  Each milestone is reported
        at most once per position to prevent alert spam.
        """
        from monitoring.alerting import _PNL_MILESTONES

        alerter = self._get_telegram_alerter()
        if alerter is None:
            return

        async with self._lock:
            trackers = list(self._positions.values())

        for tracker in trackers:
            pos = tracker.position
            symbol = pos.symbol
            entry = pos.entry_price or 0.0
            if entry == 0.0:
                continue
            current = pos.current_price or entry
            side_multiplier = 1.0 if pos.side.value in ("long", "buy") else -1.0
            pnl_pct = (current - entry) / entry * 100.0 * side_multiplier

            notified = self._notified_pnl_levels.setdefault(symbol, set())

            for milestone in _PNL_MILESTONES:
                # Profit milestone
                if pnl_pct >= milestone and milestone not in notified:
                    notified.add(milestone)
                    position_dict = {
                        "symbol": symbol,
                        "pnl_pct": pnl_pct,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    asyncio.create_task(alerter.send_pnl_alert(position_dict, milestone))

                # Loss milestone (negative)
                neg_milestone = -milestone
                if pnl_pct <= neg_milestone and neg_milestone not in notified:
                    notified.add(neg_milestone)
                    position_dict = {
                        "symbol": symbol,
                        "pnl_pct": pnl_pct,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    asyncio.create_task(alerter.send_pnl_alert(position_dict, neg_milestone))

    async def _enforce_risk_overlays(self) -> None:
        """Check each position against its SL / TP / trailing-TP thresholds and close if hit.

        The lock is acquired in brief, non-overlapping windows:
        1. Once at start to snapshot the tracker list.
        2. Once after an awaited call to verify the position is still open.
        3. Once for the TP level removal to prevent concurrent modification.
        4. Once for trailing-TP activation.
        Each lock section releases before the next await to avoid deadlocks.
        """
        async with self._lock:
            trackers = list(self._positions.values())

        for tracker in trackers:
            symbol = tracker.position.symbol
            try:
                ticker = await self._exchange.get_ticker(symbol)
                price = ticker.last
            except Exception as exc:
                logger.warning("Could not fetch ticker for {}: {}", symbol, exc)
                continue

            # Update trailing stop with current price (acquires lock internally)
            if tracker.trailing_stop is not None:
                await self.update_trailing_stop(symbol, price)

            # Re-read tracker after the awaited update to verify position is still open
            async with self._lock:
                tracker = self._positions.get(symbol)
                if tracker is None:
                    continue

            side = tracker.position.side

            # Stop-loss check
            sl = tracker.stop_loss
            if sl is not None:
                triggered = (side == PositionSide.LONG and price <= sl) or (
                    side == PositionSide.SHORT and price >= sl
                )
                if triggered:
                    logger.warning(
                        "Stop-loss triggered for {} at price={} sl={}", symbol, price, sl
                    )
                    entry = tracker.position.entry_price
                    size = tracker.position.amount
                    leverage = tracker.position.leverage
                    strategy = getattr(tracker.position, "strategy", "—")
                    side_multiplier = 1.0 if side == PositionSide.LONG else -1.0
                    pnl = (price - entry) * size * side_multiplier
                    pnl_pct = (price - entry) / entry * 100 * leverage * side_multiplier if entry > 1e-10 else 0.0
                    alert_manager = getattr(self, "alert_manager", None)
                    if alert_manager is not None:
                        try:
                            trading_mode = getattr(self, "_trading_mode", "paper")
                            await alert_manager.send_sl_alert(
                                {
                                    "symbol": symbol,
                                    "side": side.value if hasattr(side, "value") else str(side),
                                    "entry_price": entry,
                                    "exit_price": price,
                                    "size": size,
                                    "leverage": leverage,
                                    "pnl": pnl,
                                    "pnl_pct": pnl_pct,
                                    "strategy": strategy,
                                    "stop_loss": sl,
                                },
                                mode=trading_mode,
                            )
                        except Exception as _alert_exc:
                            logger.debug("SL alert error: {}", _alert_exc)
                    await self.close_position(symbol, reason="stop_loss")
                    continue

            # Take-profit execution is handled by native exchange orders placed by TradeExecutor.
            # PositionManager only updates internal state when it detects position size changes
            # during sync_positions() (indicating a TP order filled on the exchange).
            # The block below is intentionally removed to prevent double-execution races.

            # Trailing take-profit check (activates once all TP levels are consumed)
            if tracker.trailing_tp_active:
                close_price = await self.update_trailing_take_profit(symbol, price)
                if close_price is not None:
                    continue

            # Activate trailing TP when price exceeds all defined TP levels
            # and trailing_tp_distance is configured but not yet active
            async with self._lock:
                tracker = self._positions.get(symbol)
                if tracker is None:
                    continue
                if (
                    not tracker.trailing_tp_active
                    and not tracker.take_profit
                    and tracker.trailing_tp_distance is not None
                ):
                    tracker.trailing_tp_active = True
                    tracker.peak_price_for_tp = price
                    logger.info(
                        "Trailing TP activated for {} at price={:.4f}", symbol, price
                    )

            # Break-even stop trigger: when unrealised profit >= 1.5× the initial
            # stop-loss distance, move the stop to entry price + small buffer.
            async with self._lock:
                tracker = self._positions.get(symbol)
                if tracker is None:
                    continue
                if (
                    not tracker.break_even_activated
                    and tracker.stop_loss is not None
                    and tracker.position.entry_price > 0
                ):
                    entry = tracker.position.entry_price
                    sl = tracker.stop_loss
                    stop_distance = abs(entry - sl)
                    profit = (
                        (price - entry) if side == PositionSide.LONG else (entry - price)
                    )
                    if stop_distance > 0 and profit >= stop_distance * 1.5:
                        logger.info(
                            "Break-even SL trigger: {} profit={:.4f} >= 1.5× stop_dist={:.4f}",
                            symbol,
                            profit,
                            stop_distance,
                        )
                        # Schedule the activation outside the lock
                        _should_activate_be = True
                    else:
                        _should_activate_be = False
                else:
                    _should_activate_be = False

            if _should_activate_be:
                await self.activate_break_even(symbol)

            # Stale position management: escalating actions based on position age.
            # After 12h losing >1%: warning alert
            # After 24h losing: close position
            # After 48h: close regardless of P&L (capital efficiency)
            # After 6h in profit: tighten trailing stop to 50% of original distance
            async with self._lock:
                tracker = self._positions.get(symbol)
                if tracker is None:
                    continue
                from datetime import datetime, timezone
                age_hours = (
                    datetime.now(tz=timezone.utc) - tracker.opened_at
                ).total_seconds() / 3600.0
                entry = tracker.position.entry_price
                unrealized_pnl = tracker.position.unrealized_pnl
                pnl_pct = (
                    unrealized_pnl / (entry * tracker.position.amount)
                    if entry > 0 and tracker.position.amount > 0
                    and (entry * tracker.position.amount) > 1e-8
                    else 0.0
                )
                _should_warn_stale = age_hours >= 12.0 and pnl_pct < -0.01
                _should_close_stale = (
                    (age_hours >= self._MAX_HOLD_HOURS and unrealized_pnl < 0)
                    or age_hours >= 48.0
                )
                _should_tighten_trail = (
                    age_hours >= 6.0
                    and unrealized_pnl > 0
                    and tracker.trailing_stop is not None
                    and symbol not in self._trail_tightened_symbols
                )

            if _should_warn_stale and not _should_close_stale:
                logger.warning(
                    "Stale position {} warning: age={:.1f}h pnl={:.2%} — "
                    "consider manual review",
                    symbol,
                    age_hours,
                    pnl_pct,
                )
                alerter = self._get_telegram_alerter()
                if alerter is not None:
                    asyncio.create_task(
                        alerter.send_alert(
                            f"⚠️ Stale position {symbol}: age={age_hours:.1f}h "
                            f"pnl={pnl_pct:+.2%} — consider manual review"
                        )
                        if hasattr(alerter, "send_alert")
                        else alerter.send_trade_closed(symbol, unrealized_pnl, reason="stale_warning")
                    )

            if _should_tighten_trail:
                async with self._lock:
                    tracker = self._positions.get(symbol)
                    if tracker is not None and tracker.trailing_stop is not None:
                        tracker.trailing_stop = tracker.trailing_stop * 0.5
                        self._trail_tightened_symbols.add(symbol)
                        logger.info(
                            "Tightened trailing stop for {} after 6h in profit: "
                            "new distance={:.4f}",
                            symbol,
                            tracker.trailing_stop,
                        )

            if _should_close_stale:
                close_reason = (
                    "stale_48h" if age_hours >= 48.0 else "stale_position"
                )
                logger.warning(
                    "Auto-closing stale position {} (age={:.1f}h, pnl={:.2f}, reason={})",
                    symbol,
                    age_hours,
                    unrealized_pnl,
                    close_reason,
                )
                await self.close_position(symbol, reason=close_reason)
