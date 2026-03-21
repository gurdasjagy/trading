"""Forex position reconciliation — re-syncs exchange positions with the database.

On startup (before generating new signals) :class:`ForexReconciler` fetches
all open positions from the Gate.io TradFi exchange and reconciles them with
the persisted ``forex_active_positions`` table.  This ensures that:

* Positions opened before a crash are re-discovered and protected.
* Stale DB records for already-closed positions are cleaned up.
* SL / TP / trailing-stop state is restored for recovered positions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

try:
    from sqlalchemy import select

    from data.storage.models import ForexActivePosition, get_async_session

    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    ForexActivePosition = None  # type: ignore[assignment,misc]
    get_async_session = None  # type: ignore[assignment]
    logger.warning("ForexReconciler: DB models not available.")

if TYPE_CHECKING:
    from monitoring.alerting import AlertManager


class ForexReconciler:
    """Reconciles live Gate.io TradFi positions with the ``forex_active_positions`` table.

    Usage::

        reconciler = ForexReconciler(exchange=forex_exchange, alert_manager=alerting)
        report = await reconciler.reconcile()
    """

    def __init__(
        self,
        exchange: Any,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        self._exchange = exchange
        self._alert_manager = alert_manager

    async def reconcile(self) -> dict:
        """Perform full reconciliation between exchange and database positions.

        Steps:
        1. Fetch open positions from exchange.
        2. Load ``forex_active_positions`` records from DB.
        3. For each exchange position with no DB record → insert new DB record.
        4. For each DB record with no exchange position → remove orphaned record.
        5. Return reconciliation report.

        Returns:
            Dict with keys:
            ``live_positions``, ``db_positions``, ``added``, ``removed``, ``errors``.
        """
        report: dict = {
            "live_positions": [],
            "db_positions": [],
            "added": [],
            "removed": [],
            "errors": [],
        }

        if not _DB_AVAILABLE:
            report["errors"].append("DB models not available")
            return report

        # ── 1. Fetch exchange positions ────────────────────────────────
        live_by_symbol: Dict[str, dict] = {}
        try:
            positions = await self._exchange.get_positions()
            for pos in positions:
                sym = getattr(pos, "symbol", None) or pos.get("symbol", "")
                if not sym:
                    continue
                live_by_symbol[sym] = {
                    "symbol": sym,
                    "side": getattr(pos, "side", None) or pos.get("side", "long"),
                    "lot_size": float(
                        getattr(pos, "contracts", None) or pos.get("lot_size", 0.01) or 0.01
                    ),
                    "entry_price": float(
                        getattr(pos, "entry_price", None) or pos.get("entry_price", 0.0) or 0.0
                    ),
                    "leverage": int(
                        getattr(pos, "leverage", None) or pos.get("leverage", 20) or 20
                    ),
                }
            report["live_positions"] = list(live_by_symbol.values())
            logger.info("ForexReconciler: {} live positions on exchange", len(live_by_symbol))
        except Exception as exc:
            msg = f"Failed to fetch live positions: {exc}"
            logger.warning("ForexReconciler: {}", msg)
            report["errors"].append(msg)
            return report

        # ── 2. Load DB positions ────────────────────────────────────────
        db_by_symbol: Dict[str, "ForexActivePosition"] = {}  # type: ignore[type-arg]
        try:
            async with get_async_session() as session:
                result = await session.execute(select(ForexActivePosition))
                db_records = result.scalars().all()
            for rec in db_records:
                db_by_symbol[rec.symbol] = rec
            report["db_positions"] = [{"symbol": s} for s in db_by_symbol]
            logger.info("ForexReconciler: {} positions in database", len(db_by_symbol))
        except Exception as exc:
            msg = f"Failed to load DB positions: {exc}"
            logger.warning("ForexReconciler: {}", msg)
            report["errors"].append(msg)

        now = datetime.now(tz=timezone.utc)

        # ── 3. Exchange positions missing from DB (add) ─────────────────
        for symbol, live_pos in live_by_symbol.items():
            if symbol not in db_by_symbol:
                try:
                    async with get_async_session() as session:
                        rec = ForexActivePosition(
                            symbol=symbol,
                            side=live_pos["side"],
                            lot_size=live_pos["lot_size"],
                            entry_price=live_pos["entry_price"],
                            leverage=live_pos["leverage"],
                            opened_at=now,
                            updated_at=now,
                        )
                        session.add(rec)
                    report["added"].append(symbol)
                    logger.info(
                        "ForexReconciler: added missing DB record for {} {}@{}",
                        symbol,
                        live_pos["side"],
                        live_pos["entry_price"],
                    )
                    await self._send_alert(
                        f"⚠️ <b>Forex Reconciliation</b>: Found untracked position "
                        f"<b>{symbol}</b> ({live_pos['side']}) — added to DB."
                    )
                except Exception as exc:
                    msg = f"Error adding DB record for {symbol}: {exc}"
                    logger.warning("ForexReconciler: {}", msg)
                    report["errors"].append(msg)

        # ── 4. DB positions missing from exchange (remove orphans) ─────
        for symbol, db_rec in db_by_symbol.items():
            if symbol not in live_by_symbol:
                try:
                    async with get_async_session() as session:
                        result = await session.execute(
                            select(ForexActivePosition).where(
                                ForexActivePosition.symbol == symbol
                            )
                        )
                        existing = result.scalar_one_or_none()
                        if existing is not None:
                            await session.delete(existing)
                    report["removed"].append(symbol)
                    logger.info(
                        "ForexReconciler: removed orphaned DB record for {} (no longer on exchange)",
                        symbol,
                    )
                    await self._send_alert(
                        f"ℹ️ <b>Forex Reconciliation</b>: Removed orphaned DB record "
                        f"for <b>{symbol}</b> (position no longer on exchange)."
                    )
                except Exception as exc:
                    msg = f"Error removing orphaned DB record for {symbol}: {exc}"
                    logger.warning("ForexReconciler: {}", msg)
                    report["errors"].append(msg)

        logger.info(
            "ForexReconciler: reconciliation complete — "
            "live={} db={} added={} removed={} errors={}",
            len(report["live_positions"]),
            len(report["db_positions"]),
            len(report["added"]),
            len(report["removed"]),
            len(report["errors"]),
        )
        return report

    async def restore_position_state(self) -> List[dict]:
        """Load all persisted active positions from DB for state restoration.

        Call this on startup to re-arm SL/TP/trailing-stop logic for positions
        that survived a crash.

        Returns:
            List of position dicts (same format as ForexTradeJournal.get_active_positions).
        """
        if not _DB_AVAILABLE:
            return []
        try:
            async with get_async_session() as session:
                result = await session.execute(select(ForexActivePosition))
                rows = result.scalars().all()
            restored = []
            for r in rows:
                restored.append({
                    "symbol": r.symbol,
                    "side": r.side,
                    "lot_size": r.lot_size,
                    "entry_price": r.entry_price,
                    "stop_loss_price": r.stop_loss_price,
                    "take_profit_prices": r.take_profit_prices,
                    "leverage": r.leverage,
                    "strategy": r.strategy,
                    "session": r.session,
                    "opened_at": r.opened_at.isoformat() if r.opened_at else None,
                    "trailing_stop_active": r.trailing_stop_active,
                    "break_even_active": r.break_even_active,
                    "partial_closes": r.partial_closes,
                })
            logger.info("ForexReconciler: restored {} position states from DB", len(restored))
            return restored
        except Exception as exc:
            logger.warning("ForexReconciler.restore_position_state error: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _send_alert(self, message: str) -> None:
        """Send a Telegram alert if an alert manager is available."""
        if self._alert_manager is None:
            return
        try:
            await self._alert_manager.send_alert(message, level="warning")
        except Exception as exc:
            logger.debug("ForexReconciler._send_alert error: {}", exc)
