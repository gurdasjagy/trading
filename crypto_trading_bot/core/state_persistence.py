"""Full state persistence for the trading bot.

Saves and restores all critical in-memory state to/from the database so that
a bot restart does not lose trade performance metrics, risk parameters, or
compounding accumulators.

State saved:
  * StrategyManager._performance (wins/losses/pnl per strategy)
  * StrategyManager._rolling_trades (recent trade outcomes per strategy/regime)
  * RiskManager._trade_wins, _trade_losses, _total_win_return, _total_loss_return
  * RiskManager._consecutive_losses
  * ProfitCompounder._extra_allocation_pct
  * DailyPnLManager (queried from DB; no extra persistence needed)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.orm import DeclarativeBase

if TYPE_CHECKING:
    from core.engine import TradingEngine


class _Base(DeclarativeBase):
    pass


class BotStateRecord(_Base):
    """Database table for serialised bot state snapshots."""

    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)  # JSON-serialised payload
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
        onupdate=lambda: datetime.now(tz=timezone.utc),
        nullable=False,
    )


class StatePersistence:
    """Serialise and restore all critical in-memory bot state.

    Usage::

        sp = StatePersistence(engine=trading_engine)
        await sp.restore_state()   # call on startup
        await sp.save_state()      # call every 5 minutes and on shutdown
    """

    # State keys stored in ``bot_state`` table
    _KEY_STRATEGY_PERF = "strategy_performance"
    _KEY_STRATEGY_ROLLING = "strategy_rolling_trades"
    _KEY_RISK_STATS = "risk_stats"
    _KEY_COMPOUNDER = "profit_compounder"

    def __init__(self, engine: "TradingEngine") -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_state(self) -> None:
        """Serialise all in-memory state and upsert into the database."""
        try:
            payloads: Dict[str, Any] = {}

            # Strategy performance metrics
            if self._engine.strategy_manager is not None:
                try:
                    perf = dict(self._engine.strategy_manager._performance)
                    # Convert to plain dicts (defaultdict → dict, deque → list omitted here)
                    payloads[self._KEY_STRATEGY_PERF] = {k: dict(v) for k, v in perf.items()}
                except Exception as exc:
                    logger.debug("StatePersistence: could not snapshot strategy_performance — {}", exc)

                try:
                    rolling: Dict[str, Any] = {}
                    for strat, regime_map in self._engine.strategy_manager._rolling_trades.items():
                        rolling[strat] = {}
                        for regime, deque_val in regime_map.items():
                            # Each item is (pnl, won) tuple
                            rolling[strat][regime] = list(deque_val)
                    payloads[self._KEY_STRATEGY_ROLLING] = rolling
                except Exception as exc:
                    logger.debug("StatePersistence: could not snapshot rolling_trades — {}", exc)

            # RiskManager state
            if self._engine.risk_manager is not None:
                try:
                    rm = self._engine.risk_manager
                    payloads[self._KEY_RISK_STATS] = {
                        "trade_wins": rm._trade_wins,
                        "trade_losses": rm._trade_losses,
                        "total_win_return": rm._total_win_return,
                        "total_loss_return": rm._total_loss_return,
                        "consecutive_losses": rm._consecutive_losses,
                    }
                except Exception as exc:
                    logger.debug("StatePersistence: could not snapshot risk_stats — {}", exc)

                # ProfitCompounder state
                try:
                    payloads[self._KEY_COMPOUNDER] = {
                        "extra_allocation_pct": rm._profit_compounder.extra_allocation_pct,
                    }
                except Exception as exc:
                    logger.debug("StatePersistence: could not snapshot profit_compounder — {}", exc)

            if not payloads:
                return

            await self._upsert_all(payloads)
            logger.debug("StatePersistence: state saved ({} keys)", len(payloads))

        except Exception as exc:
            logger.warning("StatePersistence.save_state failed: {}", exc)

    async def restore_state(self) -> None:
        """Load persisted state from the database and restore in-memory objects."""
        try:
            await self._ensure_table()
            rows = await self._load_all()
            if not rows:
                logger.debug("StatePersistence: no persisted state found — starting fresh.")
                return

            restored = 0

            # ── Strategy performance ────────────────────────────────────
            if self._KEY_STRATEGY_PERF in rows and self._engine.strategy_manager is not None:
                try:
                    perf_data: Dict[str, dict] = rows[self._KEY_STRATEGY_PERF]
                    for strat, stats in perf_data.items():
                        self._engine.strategy_manager._performance[strat].update(stats)
                    restored += 1
                    logger.info(
                        "StatePersistence: restored strategy_performance for {} strategies",
                        len(perf_data),
                    )
                except Exception as exc:
                    logger.warning("StatePersistence: restore strategy_performance failed — {}", exc)

            # ── Strategy rolling trades ─────────────────────────────────
            if self._KEY_STRATEGY_ROLLING in rows and self._engine.strategy_manager is not None:
                try:
                    rolling_data: Dict[str, dict] = rows[self._KEY_STRATEGY_ROLLING]
                    from collections import deque
                    for strat, regime_map in rolling_data.items():
                        for regime, trade_list in regime_map.items():
                            # Repopulate deque; honour maxlen from existing deque if present
                            existing = self._engine.strategy_manager._rolling_trades[strat][regime]
                            maxlen = existing.maxlen
                            new_deque: deque = deque(
                                [tuple(t) for t in trade_list],
                                maxlen=maxlen,
                            )
                            self._engine.strategy_manager._rolling_trades[strat][regime] = new_deque
                    restored += 1
                    logger.info(
                        "StatePersistence: restored rolling_trades for {} strategies",
                        len(rolling_data),
                    )
                except Exception as exc:
                    logger.warning("StatePersistence: restore rolling_trades failed — {}", exc)

            # ── RiskManager stats ───────────────────────────────────────
            if self._KEY_RISK_STATS in rows and self._engine.risk_manager is not None:
                try:
                    rs = rows[self._KEY_RISK_STATS]
                    rm = self._engine.risk_manager
                    rm._trade_wins = int(rs.get("trade_wins", 0))
                    rm._trade_losses = int(rs.get("trade_losses", 0))
                    rm._total_win_return = float(rs.get("total_win_return", 0.0))
                    rm._total_loss_return = float(rs.get("total_loss_return", 0.0))
                    rm._consecutive_losses = int(rs.get("consecutive_losses", 0))
                    restored += 1
                    logger.info(
                        "StatePersistence: restored risk_stats "
                        "(wins={} losses={} consec_losses={})",
                        rm._trade_wins,
                        rm._trade_losses,
                        rm._consecutive_losses,
                    )
                except Exception as exc:
                    logger.warning("StatePersistence: restore risk_stats failed — {}", exc)

            # ── ProfitCompounder ────────────────────────────────────────
            if self._KEY_COMPOUNDER in rows and self._engine.risk_manager is not None:
                try:
                    cd = rows[self._KEY_COMPOUNDER]
                    extra = float(cd.get("extra_allocation_pct", 0.0))
                    self._engine.risk_manager._profit_compounder._extra_allocation_pct = extra
                    restored += 1
                    logger.info(
                        "StatePersistence: restored profit_compounder (extra_allocation={:.1f}%%)",
                        extra,
                    )
                except Exception as exc:
                    logger.warning("StatePersistence: restore profit_compounder failed — {}", exc)

            logger.info("StatePersistence: restored {} state blobs from database.", restored)

        except Exception as exc:
            logger.warning("StatePersistence.restore_state failed: {}", exc)

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
            logger.debug("StatePersistence._ensure_table: {}", exc)

    async def _upsert_all(self, payloads: Dict[str, Any]) -> None:
        """Upsert all key→JSON pairs in a single transaction."""
        try:
            from data.storage.models import get_async_session
            async with get_async_session() as session:
                for key, value in payloads.items():
                    json_value = json.dumps(value, default=str)
                    result = await session.execute(
                        select(BotStateRecord).where(BotStateRecord.key == key)
                    )
                    existing: Optional[BotStateRecord] = result.scalar_one_or_none()
                    if existing is not None:
                        existing.value = json_value
                        existing.updated_at = datetime.now(tz=timezone.utc)
                    else:
                        session.add(
                            BotStateRecord(
                                key=key,
                                value=json_value,
                                updated_at=datetime.now(tz=timezone.utc),
                            )
                        )
        except Exception as exc:
            logger.warning("StatePersistence._upsert_all failed: {}", exc)

    async def _load_all(self) -> Dict[str, Any]:
        """Load all persisted state rows from the database."""
        try:
            from data.storage.models import get_async_session
            async with get_async_session() as session:
                result = await session.execute(select(BotStateRecord))
                rows = result.scalars().all()
                return {row.key: json.loads(row.value) for row in rows}
        except Exception as exc:
            logger.warning("StatePersistence._load_all failed: {}", exc)
            return {}
