"""Forex-specific state persistence using the shared ``bot_state`` table.

Saves and restores forex-specific in-memory state to/from the database so that
a bot restart does not lose:

  * ForexRiskManager session PnL and trade counts
  * ForexRiskManager drawdown recovery mode state
  * ForexProfitCompounder compounding state
  * Session PnL tracking per session (london/new_york/asian/sydney)
  * Strategy performance metrics for forex strategies

State is stored with ``forex_`` prefixed keys to avoid collisions with the
crypto-side :class:`~core.state_persistence.StatePersistence`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.orm import DeclarativeBase

if TYPE_CHECKING:
    from core.forex_engine import ForexTradingEngine


class _Base(DeclarativeBase):
    pass


class _BotStateRecord(_Base):
    """Reuses the ``bot_state`` table schema (shared with crypto side)."""

    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)  # JSON payload
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
        onupdate=lambda: datetime.now(tz=timezone.utc),
        nullable=False,
    )


class ForexStatePersistence:
    """Serialise and restore all critical forex in-memory state.

    Usage::

        fsp = ForexStatePersistence(engine=forex_trading_engine)
        await fsp.restore_state()   # call on startup
        await fsp.save_state()      # call every 5 minutes and on shutdown
    """

    # State keys (``forex_`` prefix prevents collision with crypto keys)
    _KEY_SESSION_PNL = "forex_session_pnl"
    _KEY_RECOVERY_MODE = "forex_recovery_mode"
    _KEY_COMPOUNDER = "forex_profit_compounder"
    _KEY_STRATEGY_PERF = "forex_strategy_performance"
    _KEY_RISK_STATS = "forex_risk_stats"

    def __init__(self, engine: "ForexTradingEngine") -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_state(self) -> None:
        """Serialise all forex in-memory state to the database."""
        try:
            payloads: Dict[str, Any] = {}

            # ── ForexRiskManager state ─────────────────────────────────
            frm = getattr(self._engine, "forex_risk_manager", None)
            if frm is not None:
                try:
                    payloads[self._KEY_RISK_STATS] = {
                        "session_pnl": getattr(frm, "_session_pnl", 0.0),
                        "session_trades": getattr(frm, "_session_trades", 0),
                        "consecutive_losses": getattr(frm, "_consecutive_losses", 0),
                        "consecutive_wins": getattr(frm, "_consecutive_wins", 0),
                        "in_recovery_mode": getattr(frm, "_in_recovery_mode", False),
                        "recovery_mode_level": getattr(frm, "_recovery_mode_level", 0),
                        "peak_equity": getattr(frm, "_peak_equity", 0.0),
                    }
                except Exception as exc:
                    logger.debug("ForexStatePersistence: risk_stats snapshot failed — {}", exc)

                # ── Session PnL breakdown ──────────────────────────────
                try:
                    session_pnl = getattr(frm, "_session_pnl_breakdown", {})
                    payloads[self._KEY_SESSION_PNL] = dict(session_pnl)
                except Exception as exc:
                    logger.debug("ForexStatePersistence: session_pnl snapshot failed — {}", exc)

                # ── Recovery mode state ────────────────────────────────
                try:
                    recovery = {
                        "active": getattr(frm, "_in_recovery_mode", False),
                        "level": getattr(frm, "_recovery_mode_level", 0),
                        "max_drawdown_pct": getattr(frm, "_max_drawdown_pct_seen", 0.0),
                        "last_peak_equity": getattr(frm, "_peak_equity", 0.0),
                    }
                    payloads[self._KEY_RECOVERY_MODE] = recovery
                except Exception as exc:
                    logger.debug("ForexStatePersistence: recovery_mode snapshot failed — {}", exc)

            # ── ForexProfitCompounder state ────────────────────────────
            fpc = getattr(self._engine, "profit_compounder", None)
            if fpc is not None:
                try:
                    payloads[self._KEY_COMPOUNDER] = {
                        "base_lot_multiplier": getattr(fpc, "_base_lot_multiplier", 1.0),
                        "compound_growth": getattr(fpc, "_compound_growth", 0.0),
                        "original_base_lot": getattr(fpc, "_original_base_lot", 0.01),
                    }
                except Exception as exc:
                    logger.debug("ForexStatePersistence: compounder snapshot failed — {}", exc)

            # ── ForexStrategyManager performance ──────────────────────
            fsm = getattr(self._engine, "strategy_manager", None)
            if fsm is not None:
                try:
                    perf = getattr(fsm, "_performance", {})
                    payloads[self._KEY_STRATEGY_PERF] = {k: dict(v) for k, v in perf.items()}
                except Exception as exc:
                    logger.debug("ForexStatePersistence: strategy_perf snapshot failed — {}", exc)

            if not payloads:
                return

            await self._upsert_all(payloads)
            logger.debug("ForexStatePersistence: saved {} state keys", len(payloads))

        except Exception as exc:
            logger.warning("ForexStatePersistence.save_state failed: {}", exc)

    async def restore_state(self) -> None:
        """Load persisted state from the database and restore in-memory objects."""
        try:
            await self._ensure_table()
            rows = await self._load_all()
            if not rows:
                logger.debug("ForexStatePersistence: no persisted state found — starting fresh.")
                return

            restored = 0

            # ── ForexRiskManager risk stats ────────────────────────────
            frm = getattr(self._engine, "forex_risk_manager", None)
            if frm is not None and self._KEY_RISK_STATS in rows:
                try:
                    rs = rows[self._KEY_RISK_STATS]
                    frm._session_pnl = float(rs.get("session_pnl", 0.0))
                    frm._session_trades = int(rs.get("session_trades", 0))
                    frm._consecutive_losses = int(rs.get("consecutive_losses", 0))
                    frm._consecutive_wins = int(rs.get("consecutive_wins", 0))
                    frm._in_recovery_mode = bool(rs.get("in_recovery_mode", False))
                    frm._recovery_mode_level = int(rs.get("recovery_mode_level", 0))
                    frm._peak_equity = float(rs.get("peak_equity", 0.0))
                    restored += 1
                    logger.info(
                        "ForexStatePersistence: restored risk_stats "
                        "(consec_losses={} recovery={})",
                        frm._consecutive_losses,
                        frm._in_recovery_mode,
                    )
                except Exception as exc:
                    logger.warning("ForexStatePersistence: restore risk_stats failed — {}", exc)

            # ── Session PnL ────────────────────────────────────────────
            if frm is not None and self._KEY_SESSION_PNL in rows:
                try:
                    frm._session_pnl_breakdown = dict(rows[self._KEY_SESSION_PNL])
                    restored += 1
                    logger.info("ForexStatePersistence: restored session_pnl_breakdown")
                except Exception as exc:
                    logger.warning("ForexStatePersistence: restore session_pnl failed — {}", exc)

            # ── Recovery mode ──────────────────────────────────────────
            if frm is not None and self._KEY_RECOVERY_MODE in rows:
                try:
                    rd = rows[self._KEY_RECOVERY_MODE]
                    frm._in_recovery_mode = bool(rd.get("active", False))
                    frm._recovery_mode_level = int(rd.get("level", 0))
                    frm._max_drawdown_pct_seen = float(rd.get("max_drawdown_pct", 0.0))
                    frm._peak_equity = float(rd.get("last_peak_equity", 0.0))
                    restored += 1
                except Exception as exc:
                    logger.warning("ForexStatePersistence: restore recovery_mode failed — {}", exc)

            # ── ForexProfitCompounder ──────────────────────────────────
            fpc = getattr(self._engine, "profit_compounder", None)
            if fpc is not None and self._KEY_COMPOUNDER in rows:
                try:
                    cd = rows[self._KEY_COMPOUNDER]
                    fpc._base_lot_multiplier = float(cd.get("base_lot_multiplier", 1.0))
                    fpc._compound_growth = float(cd.get("compound_growth", 0.0))
                    fpc._original_base_lot = float(cd.get("original_base_lot", 0.01))
                    restored += 1
                    logger.info(
                        "ForexStatePersistence: restored compounder (multiplier={:.3f})",
                        fpc._base_lot_multiplier,
                    )
                except Exception as exc:
                    logger.warning("ForexStatePersistence: restore compounder failed — {}", exc)

            # ── Strategy performance ───────────────────────────────────
            fsm = getattr(self._engine, "strategy_manager", None)
            if fsm is not None and self._KEY_STRATEGY_PERF in rows:
                try:
                    perf_data = rows[self._KEY_STRATEGY_PERF]
                    for strat, stats in perf_data.items():
                        if hasattr(fsm._performance, "__setitem__"):
                            fsm._performance[strat].update(stats)
                    restored += 1
                    logger.info(
                        "ForexStatePersistence: restored strategy_performance ({} strategies)",
                        len(perf_data),
                    )
                except Exception as exc:
                    logger.warning("ForexStatePersistence: restore strategy_perf failed — {}", exc)

            logger.info("ForexStatePersistence: restored {} state blobs", restored)

        except Exception as exc:
            logger.warning("ForexStatePersistence.restore_state failed: {}", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _ensure_table(self) -> None:
        """Create the ``bot_state`` table if it does not exist."""
        try:
            from data.storage.models import get_async_engine

            db_engine = get_async_engine()
            async with db_engine.begin() as conn:
                await conn.run_sync(_Base.metadata.create_all)
        except Exception as exc:
            logger.debug("ForexStatePersistence._ensure_table: {}", exc)

    async def _upsert_all(self, payloads: Dict[str, Any]) -> None:
        """Upsert all key→JSON pairs in a single transaction."""
        try:
            from data.storage.models import get_async_session

            async with get_async_session() as session:
                for key, value in payloads.items():
                    json_value = json.dumps(value, default=str)
                    result = await session.execute(
                        select(_BotStateRecord).where(_BotStateRecord.key == key)
                    )
                    existing: Optional[_BotStateRecord] = result.scalar_one_or_none()
                    if existing is not None:
                        existing.value = json_value
                        existing.updated_at = datetime.now(tz=timezone.utc)
                    else:
                        session.add(
                            _BotStateRecord(
                                key=key,
                                value=json_value,
                                updated_at=datetime.now(tz=timezone.utc),
                            )
                        )
        except Exception as exc:
            logger.warning("ForexStatePersistence._upsert_all failed: {}", exc)

    async def _load_all(self) -> Dict[str, Any]:
        """Load all forex-prefixed state rows."""
        try:
            from data.storage.models import get_async_session

            async with get_async_session() as session:
                result = await session.execute(
                    select(_BotStateRecord).where(
                        _BotStateRecord.key.like("forex_%")
                    )
                )
                rows = result.scalars().all()
                return {row.key: json.loads(row.value) for row in rows}
        except Exception as exc:
            logger.warning("ForexStatePersistence._load_all failed: {}", exc)
            return {}
