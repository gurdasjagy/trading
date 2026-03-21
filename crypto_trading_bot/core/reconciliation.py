"""Startup state reconciliation — re-syncs exchange positions with the database.

On bot startup (before generating new signals) :class:`StateReconciler` fetches
all open positions and orders from the exchange and reconciles them with the
persisted state in the ``active_positions`` and ``active_orders`` database
tables.  This guarantees that the bot's in-memory :class:`PositionManager` and
:class:`RiskManager` always reflect the true state of the exchange — even after
an unexpected restart or crash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from loguru import logger
from sqlalchemy import select

try:
    from data.storage.models import ActivePosition, get_async_session
except ImportError as _import_err:
    ActivePosition = None  # type: ignore[assignment]
    get_async_session = None  # type: ignore[assignment]
    logger.warning("Could not import ActivePosition/get_async_session: {}", _import_err)

if TYPE_CHECKING:
    from config.settings import Settings
    from exchange.base_exchange import BaseExchange, Position
    from exchange.position_manager import PositionManager
    from monitoring.alerting import AlertManager
    from risk.risk_manager import RiskManager


class StateReconciler:
    """Reconciles live exchange state with the database on startup.

    Usage::

        reconciler = StateReconciler(
            exchange=engine.exchange,
            position_manager=engine.position_manager,
            alert_manager=engine.alert_manager,
            settings=engine.settings,
        )
        await reconciler.reconcile_state()
    """

    def __init__(
        self,
        exchange: "BaseExchange",
        position_manager: "PositionManager",
        risk_manager: Optional["RiskManager"] = None,
        alert_manager: Optional["AlertManager"] = None,
        settings: Optional["Settings"] = None,
    ) -> None:
        self._exchange = exchange
        self._position_manager = position_manager
        self._risk_manager = risk_manager
        self._alert_manager = alert_manager
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reconcile_exchange_first(self) -> dict:
        """Exchange-first reconciliation: fetch ALL open positions from the exchange
        and compare against the database.  Called BEFORE reconcile_state().

        For any exchange position with no matching DB record (orphaned):
          - Place emergency SL/TP orders at default percentages (2% SL, 4% TP)
            from the current price, OR
          - Auto-close immediately if ``settings.risk.auto_close_orphaned_positions``.
        Sends a Telegram alert for every orphaned position found.

        Returns:
            Summary dict with keys ``live_positions``, ``orphaned``,
            ``emergency_sltp_placed``, ``auto_closed``, and ``errors``.
        """
        if ActivePosition is None or get_async_session is None:
            logger.warning("Database models not available — skipping exchange-first reconciliation.")
            return {
                "live_positions": 0,
                "orphaned": 0,
                "emergency_sltp_placed": 0,
                "auto_closed": 0,
                "errors": ["Database models not available"],
            }

        summary: dict = {
            "live_positions": 0,
            "orphaned": 0,
            "emergency_sltp_placed": 0,
            "auto_closed": 0,
            "errors": [],
        }

        logger.info("🔄 Starting exchange-first reconciliation…")

        try:
            live_positions: List["Position"] = await self._exchange.get_positions()
        except Exception as exc:
            msg = f"Exchange-first reconciliation: failed to fetch positions: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)
            return summary

        summary["live_positions"] = len(live_positions)

        # Load all open trigger orders to detect if SL/TP already exist
        try:
            open_orders = await self._exchange.get_open_orders()
        except Exception as exc:
            logger.warning("Could not fetch open orders for reconciliation: {}", exc)
            open_orders = []

        order_symbols = {o.symbol for o in open_orders}

        for pos in live_positions:
            try:
                # Check if DB has a record for this position
                async with get_async_session() as session:
                    result = await session.execute(
                        select(ActivePosition).where(
                            ActivePosition.symbol == pos.symbol
                        )
                    )
                    db_record = result.scalar_one_or_none()

                if db_record is not None:
                    logger.debug(
                        "Exchange-first reconciliation: DB record found for {}", pos.symbol
                    )
                    continue

                # Orphaned position: exists on exchange, not in DB
                summary["orphaned"] += 1
                side_str = (
                    pos.side.value if hasattr(pos.side, "value") else str(pos.side)
                )
                logger.warning(
                    "⚠️ ORPHANED POSITION (exchange-first): {} {} amount={} entry={}",
                    pos.symbol,
                    side_str,
                    pos.amount,
                    pos.entry_price,
                )

                # Send Telegram alert
                if self._alert_manager is not None:
                    try:
                        await self._alert_manager.send_alert(
                            f"⚠️ ORPHANED POSITION: {pos.symbol} {side_str.upper()} "
                            f"amount={pos.amount:.6f} entry_price={pos.entry_price:.4f}\n"
                            "Position exists on exchange but has no DB record. "
                            "Emergency SL/TP will be placed.",
                            level="warning",
                        )
                    except Exception as exc:
                        logger.debug("Failed to send orphaned alert: {}", exc)

                auto_close: bool = bool(
                    getattr(
                        getattr(self._settings, "risk", None),
                        "auto_close_orphaned_positions",
                        False,
                    )
                )

                if auto_close:
                    try:
                        await self._exchange.close_position(pos.symbol)
                        summary["auto_closed"] += 1
                        logger.info("Auto-closed orphaned position: {}", pos.symbol)
                        if self._alert_manager is not None:
                            try:
                                await self._alert_manager.send_alert(
                                    f"🔒 AUTO-CLOSED orphaned position: {pos.symbol}",
                                    level="warning",
                                )
                            except Exception:
                                pass
                    except Exception as exc:
                        msg = f"Failed to auto-close orphaned {pos.symbol}: {exc}"
                        logger.error(msg)
                        summary["errors"].append(msg)
                elif pos.symbol not in order_symbols:
                    # Place emergency SL/TP if none exist yet
                    try:
                        await self._place_emergency_sltp(pos)
                        summary["emergency_sltp_placed"] += 1
                    except Exception as exc:
                        msg = f"Failed to place emergency SL/TP for {pos.symbol}: {exc}"
                        logger.error(msg)
                        summary["errors"].append(msg)

            except Exception as exc:
                msg = f"Exchange-first reconciliation error for {pos.symbol}: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

        logger.info(
            "🔄 Exchange-first reconciliation complete — live={} orphaned={} "
            "emergency_sltp={} auto_closed={} errors={}",
            summary["live_positions"],
            summary["orphaned"],
            summary["emergency_sltp_placed"],
            summary["auto_closed"],
            len(summary["errors"]),
        )
        return summary

    async def reconcile_state(self) -> dict:
        """Reconcile exchange live state with the persisted database state.

        Steps:
        1. Fetch all open positions from the exchange.
        2. For each live position query ``active_positions`` by symbol.
           - **Found**: restore ``stop_loss`` / ``take_profit`` / ``strategy``
             into the :class:`~exchange.position_manager.PositionManager` and
             :class:`~risk.risk_manager.RiskManager` in-memory state.
           - **Not found (orphaned)**: send a warning alert via
             :class:`~monitoring.alerting.AlertManager` and optionally
             auto-close the position (controlled by
             ``settings.risk.auto_close_orphaned_positions``).
        3. Remove ``active_positions`` DB rows for positions no longer open on
           the exchange (stale records from normal trade closure).

        Returns:
            Summary dict with keys ``live_positions``, ``matched``,
            ``orphaned``, ``stale_db``, and ``errors``.
        """
        if ActivePosition is None or get_async_session is None:
            logger.warning("Database models not available — skipping reconciliation.")
            return {
                "live_positions": 0,
                "matched": 0,
                "orphaned": 0,
                "stale_db": 0,
                "errors": ["Database models not available"],
            }

        summary: dict = {
            "live_positions": 0,
            "matched": 0,
            "orphaned": 0,
            "stale_db": 0,
            "errors": [],
        }

        logger.info("🔄 Starting startup state reconciliation…")

        try:
            # 1. Fetch live positions from the exchange
            try:
                live_positions: List["Position"] = await self._exchange.get_positions()
            except Exception as exc:
                msg = f"Failed to fetch live positions from exchange: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)
                return summary

            summary["live_positions"] = len(live_positions)
            live_symbols = {p.symbol for p in live_positions}

            # Ensure in-memory position tracker is up-to-date before restoring metadata
            try:
                await self._position_manager.sync_positions()
            except Exception as exc:
                logger.warning("sync_positions failed during reconciliation: {}", exc)

            # 2. Process each live exchange position
            for pos in live_positions:
                try:
                    async with get_async_session() as session:
                        result = await session.execute(
                            select(ActivePosition).where(
                                ActivePosition.symbol == pos.symbol
                            )
                        )
                        db_record = result.scalar_one_or_none()

                    if db_record is not None:
                        await self._restore_position_metadata(pos.symbol, db_record)
                        summary["matched"] += 1
                        logger.info(
                            "✅ Reconciled position: {} (strategy={} sl={} tp={})",
                            pos.symbol,
                            db_record.strategy_name,
                            db_record.stop_loss,
                            db_record.take_profit,
                        )
                    else:
                        summary["orphaned"] += 1
                        await self._handle_orphaned_position(pos)

                except Exception as exc:
                    msg = f"Error reconciling position {pos.symbol}: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)

            # 3. Remove stale DB records (positions that closed while bot was down)
            try:
                async with get_async_session() as session:
                    result = await session.execute(select(ActivePosition))
                    all_db_positions = result.scalars().all()

                for db_pos in all_db_positions:
                    if db_pos.symbol not in live_symbols:
                        summary["stale_db"] += 1
                        logger.info(
                            "🗑️ Removing stale DB record for closed position: {}",
                            db_pos.symbol,
                        )
                        try:
                            async with get_async_session() as session:
                                fresh = await session.get(ActivePosition, db_pos.id)
                                if fresh is not None:
                                    await session.delete(fresh)
                        except Exception as exc:
                            logger.warning(
                                "Could not remove stale record for {}: {}", db_pos.symbol, exc
                            )
            except Exception as exc:
                msg = f"Failed to clean up stale DB positions: {exc}"
                logger.warning(msg)
                summary["errors"].append(msg)

        except Exception as exc:
            msg = f"State reconciliation failed: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

        logger.info(
            "🔄 Reconciliation complete — live={} matched={} orphaned={} stale_db={} errors={}",
            summary["live_positions"],
            summary["matched"],
            summary["orphaned"],
            summary["stale_db"],
            len(summary["errors"]),
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _restore_position_metadata(self, symbol: str, db_record) -> None:
        """Restore SL / TP and strategy metadata into the PositionManager."""
        try:
            tracker = await self._position_manager.get_position(symbol)

            if tracker is None:
                logger.warning(
                    "Could not find position tracker for {} after sync — "
                    "metadata not restored.",
                    symbol,
                )
                return

            if db_record.stop_loss is not None:
                tracker.stop_loss = db_record.stop_loss

            if db_record.take_profit is not None:
                raw = db_record.take_profit
                tp_levels = raw if isinstance(raw, list) else [raw]
                tracker.take_profit = sorted(float(v) for v in tp_levels if v is not None)

            if db_record.strategy_name:
                tracker.strategy = db_record.strategy_name

            logger.debug(
                "Restored metadata for {}: strategy={} sl={} tp={}",
                symbol,
                tracker.strategy,
                tracker.stop_loss,
                tracker.take_profit,
            )

        except Exception as exc:
            logger.error("Failed to restore metadata for {}: {}", symbol, exc)

    async def _handle_orphaned_position(self, position: "Position") -> None:
        """Handle an exchange position that has no database record."""
        symbol = position.symbol
        side = position.side.value if hasattr(position.side, "value") else str(position.side)
        amount = position.amount
        entry_price = position.entry_price

        logger.warning(
            "⚠️ ORPHANED POSITION detected: {} {} amount={} entry={}",
            symbol,
            side,
            amount,
            entry_price,
        )

        # Send alert via AlertManager
        if self._alert_manager is not None:
            try:
                await self._alert_manager.send_alert(
                    f"⚠️ ORPHANED POSITION: {symbol} {side.upper()} "
                    f"amount={amount:.6f} entry_price={entry_price:.4f}\n"
                    "This position exists on the exchange but has no database "
                    "record. It may have been opened outside this bot.",
                    level="warning",
                )
            except Exception as exc:
                logger.debug("Failed to send orphaned position alert: {}", exc)

        # Auto-close if configured
        auto_close: bool = False
        if self._settings is not None:
            auto_close = bool(
                getattr(
                    getattr(self._settings, "risk", None),
                    "auto_close_orphaned_positions",
                    False,
                )
            )

        if auto_close:
            logger.warning("Auto-closing orphaned position: {}", symbol)
            try:
                await self._exchange.close_position(symbol)
                logger.info("Orphaned position auto-closed: {}", symbol)
                if self._alert_manager is not None:
                    try:
                        await self._alert_manager.send_alert(
                            f"🔒 AUTO-CLOSED orphaned position: {symbol}",
                            level="warning",
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.error(
                    "Failed to auto-close orphaned position {}: {}", symbol, exc
                )

    async def _place_emergency_sltp(self, position: "Position") -> None:
        """Place emergency SL (2%) and TP (4%) orders for an orphaned position."""
        from exchange.base_exchange import OrderSide, PositionSide

        symbol = position.symbol
        side = position.side
        amount = position.amount

        # Fetch current price
        try:
            ticker = await self._exchange.get_ticker(symbol)
            current_price = ticker.last
        except Exception as exc:
            logger.warning("Could not fetch ticker for emergency SL/TP on {}: {}", symbol, exc)
            current_price = position.entry_price or position.current_price

        if current_price <= 0:
            logger.error("Cannot place emergency SL/TP for {}: invalid price", symbol)
            return

        is_long = side == PositionSide.LONG
        sl_price = current_price * (0.98 if is_long else 1.02)
        tp_price = current_price * (1.04 if is_long else 0.96)
        close_side = OrderSide.SELL if is_long else OrderSide.BUY

        try:
            sl_order = await self._exchange.create_stop_loss_order(
                symbol, close_side, amount, sl_price
            )
            logger.info(
                "🛡️ Emergency SL placed for orphaned {}: id={} trigger={}",
                symbol, sl_order.id, sl_price,
            )
        except Exception as exc:
            logger.error("Failed to place emergency SL for {}: {}", symbol, exc)

        try:
            tp_order = await self._exchange.create_take_profit_order(
                symbol, close_side, amount, tp_price
            )
            logger.info(
                "🎯 Emergency TP placed for orphaned {}: id={} trigger={}",
                symbol, tp_order.id, tp_price,
            )
        except Exception as exc:
            logger.error("Failed to place emergency TP for {}: {}", symbol, exc)
