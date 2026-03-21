"""Forex trade journal — records and queries forex trades with pip-based metrics."""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from sqlalchemy import desc, select

    from data.storage.models import (
        ForexActivePosition,
        ForexDailyPerformance,
        ForexTradeHistory,
        get_async_session,
    )

    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


class ForexTradeJournal:
    """Record and query forex trades with pip-based metrics.

    All reads/writes go through the async SQLAlchemy session so they are
    compatible with the same database used by the crypto trading engine.
    """

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    async def record_trade(self, trade: dict) -> None:
        """Persist a completed (or open) forex trade to ForexTradeHistory.

        Expected keys in *trade*:
        ``order_id``, ``symbol``, ``side``, ``lot_size``, ``entry_price``,
        ``exit_price``, ``stop_loss_price``, ``take_profit_price``,
        ``stop_loss_pips``, ``take_profit_pips``, ``pip_pnl``, ``usd_pnl``,
        ``spread_at_entry``, ``leverage``, ``margin_used``, ``swap_cost``,
        ``commission``, ``strategy``, ``session``, ``entry_time``,
        ``exit_time``, ``duration_seconds``, ``exit_reason``,
        ``max_favorable_pips``, ``max_adverse_pips``, ``notes``.
        """
        if not _DB_AVAILABLE:
            logger.debug("ForexTradeJournal: DB not available, skipping record_trade")
            return
        try:
            entry_time = trade.get("entry_time")
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            if entry_time is None:
                entry_time = datetime.now(tz=timezone.utc)

            exit_time = trade.get("exit_time")
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time)

            record = ForexTradeHistory(
                order_id=trade.get("order_id"),
                symbol=trade.get("symbol", ""),
                side=trade.get("side", "long"),
                lot_size=float(trade.get("lot_size", 0.01)),
                entry_price=float(trade.get("entry_price", 0.0)),
                exit_price=_optional_float(trade.get("exit_price")),
                stop_loss_price=_optional_float(trade.get("stop_loss_price")),
                take_profit_price=_optional_float(trade.get("take_profit_price")),
                stop_loss_pips=_optional_float(trade.get("stop_loss_pips")),
                take_profit_pips=_optional_float(trade.get("take_profit_pips")),
                pip_pnl=_optional_float(trade.get("pip_pnl")),
                usd_pnl=_optional_float(trade.get("usd_pnl")),
                spread_at_entry=_optional_float(trade.get("spread_at_entry")),
                leverage=int(trade.get("leverage", 20)),
                margin_used=_optional_float(trade.get("margin_used")),
                swap_cost=float(trade.get("swap_cost", 0.0)),
                commission=float(trade.get("commission", 0.0)),
                strategy=trade.get("strategy"),
                session=trade.get("session"),
                entry_time=entry_time,
                exit_time=exit_time,
                duration_seconds=trade.get("duration_seconds"),
                exit_reason=trade.get("exit_reason"),
                max_favorable_pips=_optional_float(trade.get("max_favorable_pips")),
                max_adverse_pips=_optional_float(trade.get("max_adverse_pips")),
                notes=trade.get("notes"),
            )
            async with get_async_session() as session:
                session.add(record)
            logger.debug("ForexTradeJournal: recorded trade {} {}", trade.get("symbol"), trade.get("side"))
        except Exception as exc:
            logger.warning("ForexTradeJournal.record_trade error: {}", exc)

    async def record_daily_performance(self, perf_date: date, metrics: dict) -> None:
        """Persist (or update) daily forex performance in ForexDailyPerformance.

        Expected keys in *metrics*: same as ForexDailyPerformance column names.
        """
        if not _DB_AVAILABLE:
            return
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexDailyPerformance).where(ForexDailyPerformance.date == perf_date)
                )
                existing = result.scalar_one_or_none()

                now = datetime.now(tz=timezone.utc)
                if existing is not None:
                    # Update in place
                    for key, val in metrics.items():
                        if hasattr(existing, key) and key not in ("id", "date", "created_at"):
                            setattr(existing, key, val)
                    existing.updated_at = now
                else:
                    record = ForexDailyPerformance(
                        date=perf_date,
                        starting_equity=float(metrics.get("starting_equity", 0.0)),
                        ending_equity=_optional_float(metrics.get("ending_equity")),
                        total_pnl_usd=_optional_float(metrics.get("total_pnl_usd")),
                        total_pnl_pips=_optional_float(metrics.get("total_pnl_pips")),
                        total_trades=int(metrics.get("total_trades", 0)),
                        wins=int(metrics.get("wins", 0)),
                        losses=int(metrics.get("losses", 0)),
                        win_rate=_optional_float(metrics.get("win_rate")),
                        profit_factor=_optional_float(metrics.get("profit_factor")),
                        max_drawdown_pct=_optional_float(metrics.get("max_drawdown_pct")),
                        best_trade_pips=_optional_float(metrics.get("best_trade_pips")),
                        worst_trade_pips=_optional_float(metrics.get("worst_trade_pips")),
                        avg_win_pips=_optional_float(metrics.get("avg_win_pips")),
                        avg_loss_pips=_optional_float(metrics.get("avg_loss_pips")),
                        total_lots_traded=_optional_float(metrics.get("total_lots_traded")),
                        total_commission=float(metrics.get("total_commission", 0.0)),
                        total_swap=float(metrics.get("total_swap", 0.0)),
                        london_trades=int(metrics.get("london_trades", 0)),
                        london_pnl_pips=float(metrics.get("london_pnl_pips", 0.0)),
                        ny_trades=int(metrics.get("ny_trades", 0)),
                        ny_pnl_pips=float(metrics.get("ny_pnl_pips", 0.0)),
                        asian_trades=int(metrics.get("asian_trades", 0)),
                        asian_pnl_pips=float(metrics.get("asian_pnl_pips", 0.0)),
                        sydney_trades=int(metrics.get("sydney_trades", 0)),
                        sydney_pnl_pips=float(metrics.get("sydney_pnl_pips", 0.0)),
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(record)
            logger.debug("ForexTradeJournal: recorded daily performance for {}", perf_date)
        except Exception as exc:
            logger.warning("ForexTradeJournal.record_daily_performance error: {}", exc)

    async def upsert_active_position(self, position: dict) -> None:
        """Insert or update an active forex position record."""
        if not _DB_AVAILABLE:
            return
        try:
            symbol = position.get("symbol", "")
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexActivePosition).where(ForexActivePosition.symbol == symbol)
                )
                existing = result.scalar_one_or_none()
                now = datetime.now(tz=timezone.utc)
                if existing is not None:
                    existing.side = position.get("side", existing.side)
                    existing.lot_size = float(position.get("lot_size", existing.lot_size))
                    existing.entry_price = float(position.get("entry_price", existing.entry_price))
                    existing.stop_loss_price = _optional_float(position.get("stop_loss_price"))
                    existing.take_profit_prices = position.get("take_profit_prices")
                    existing.trailing_stop_active = bool(position.get("trailing_stop_active", False))
                    existing.break_even_active = bool(position.get("break_even_active", False))
                    existing.partial_closes = position.get("partial_closes")
                    existing.updated_at = now
                else:
                    opened_at = position.get("opened_at")
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at)
                    record = ForexActivePosition(
                        order_id=position.get("order_id"),
                        symbol=symbol,
                        side=position.get("side", "long"),
                        lot_size=float(position.get("lot_size", 0.01)),
                        entry_price=float(position.get("entry_price", 0.0)),
                        stop_loss_price=_optional_float(position.get("stop_loss_price")),
                        take_profit_prices=position.get("take_profit_prices"),
                        leverage=int(position.get("leverage", 20)),
                        strategy=position.get("strategy"),
                        session=position.get("session"),
                        opened_at=opened_at or now,
                        updated_at=now,
                        trailing_stop_active=bool(position.get("trailing_stop_active", False)),
                        break_even_active=bool(position.get("break_even_active", False)),
                        partial_closes=position.get("partial_closes"),
                    )
                    session.add(record)
        except Exception as exc:
            logger.warning("ForexTradeJournal.upsert_active_position error: {}", exc)

    async def remove_active_position(self, symbol: str) -> None:
        """Remove an active position record when a position is closed."""
        if not _DB_AVAILABLE:
            return
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexActivePosition).where(ForexActivePosition.symbol == symbol)
                )
                existing = result.scalar_one_or_none()
                if existing is not None:
                    await session.delete(existing)
        except Exception as exc:
            logger.warning("ForexTradeJournal.remove_active_position error: {}", exc)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    async def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Return the most recent *limit* forex trades."""
        if not _DB_AVAILABLE:
            return []
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexTradeHistory)
                    .order_by(desc(ForexTradeHistory.entry_time))
                    .limit(limit)
                )
                rows = result.scalars().all()
            return [_trade_to_dict(r) for r in rows]
        except Exception as exc:
            logger.warning("ForexTradeJournal.get_recent_trades error: {}", exc)
            return []

    async def get_session_stats(self, session_name: str, days: int = 30) -> dict:
        """Return performance stats for a specific trading session.

        Args:
            session_name: One of ``london``, ``new_york``, ``asian``, ``sydney``.
            days: Lookback window in calendar days.
        """
        if not _DB_AVAILABLE:
            return {}
        try:
            from datetime import timedelta

            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexTradeHistory)
                    .where(
                        ForexTradeHistory.session == session_name,
                        ForexTradeHistory.entry_time >= cutoff,
                        ForexTradeHistory.exit_time.isnot(None),
                    )
                )
                trades = result.scalars().all()

            return _compute_stats(trades, label=session_name)
        except Exception as exc:
            logger.warning("ForexTradeJournal.get_session_stats error: {}", exc)
            return {}

    async def get_strategy_stats(self, strategy: str, days: int = 30) -> dict:
        """Return performance stats for a specific strategy."""
        if not _DB_AVAILABLE:
            return {}
        try:
            from datetime import timedelta

            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexTradeHistory)
                    .where(
                        ForexTradeHistory.strategy == strategy,
                        ForexTradeHistory.entry_time >= cutoff,
                        ForexTradeHistory.exit_time.isnot(None),
                    )
                )
                trades = result.scalars().all()

            return _compute_stats(trades, label=strategy)
        except Exception as exc:
            logger.warning("ForexTradeJournal.get_strategy_stats error: {}", exc)
            return {}

    async def get_daily_performance(self, days: int = 30) -> List[dict]:
        """Return daily performance records for the past *days* days."""
        if not _DB_AVAILABLE:
            return []
        try:
            from datetime import timedelta

            cutoff = date.today() - timedelta(days=days)
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexDailyPerformance)
                    .where(ForexDailyPerformance.date >= cutoff)
                    .order_by(ForexDailyPerformance.date)
                )
                rows = result.scalars().all()
            return [_daily_to_dict(r) for r in rows]
        except Exception as exc:
            logger.warning("ForexTradeJournal.get_daily_performance error: {}", exc)
            return []

    async def get_active_positions(self) -> List[dict]:
        """Return all open forex positions from the database."""
        if not _DB_AVAILABLE:
            return []
        try:
            async with get_async_session() as session:
                result = await session.execute(select(ForexActivePosition))
                rows = result.scalars().all()
            return [_active_pos_to_dict(r) for r in rows]
        except Exception as exc:
            logger.warning("ForexTradeJournal.get_active_positions error: {}", exc)
            return []

    async def export_trades_csv(self, filters: Optional[dict] = None) -> str:
        """Export filtered trades as a CSV string.

        *filters* may contain:
        ``symbol``, ``strategy``, ``session``, ``side``,
        ``date_from`` (ISO string), ``date_to`` (ISO string).
        """
        if not _DB_AVAILABLE:
            return ""
        filters = filters or {}
        try:
            from datetime import timedelta

            from sqlalchemy import and_

            conditions: list = []
            if filters.get("symbol"):
                conditions.append(ForexTradeHistory.symbol == filters["symbol"])
            if filters.get("strategy"):
                conditions.append(ForexTradeHistory.strategy == filters["strategy"])
            if filters.get("session"):
                conditions.append(ForexTradeHistory.session == filters["session"])
            if filters.get("side"):
                conditions.append(ForexTradeHistory.side == filters["side"])
            if filters.get("date_from"):
                dt = datetime.fromisoformat(filters["date_from"])
                conditions.append(ForexTradeHistory.entry_time >= dt)
            if filters.get("date_to"):
                dt = datetime.fromisoformat(filters["date_to"])
                conditions.append(ForexTradeHistory.entry_time <= dt)

            async with get_async_session() as session:
                stmt = select(ForexTradeHistory).order_by(desc(ForexTradeHistory.entry_time))
                if conditions:
                    stmt = stmt.where(and_(*conditions))
                result = await session.execute(stmt)
                trades = result.scalars().all()

            output = io.StringIO()
            fieldnames = [
                "id", "order_id", "symbol", "side", "lot_size",
                "entry_price", "exit_price", "pip_pnl", "usd_pnl",
                "stop_loss_pips", "take_profit_pips", "spread_at_entry",
                "leverage", "strategy", "session", "entry_time", "exit_time",
                "duration_seconds", "exit_reason", "max_favorable_pips",
                "max_adverse_pips", "swap_cost", "commission",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for t in trades:
                writer.writerow(_trade_to_dict(t))
            return output.getvalue()
        except Exception as exc:
            logger.warning("ForexTradeJournal.export_trades_csv error: {}", exc)
            return ""

    async def get_overall_stats(self, days: int = 30) -> Dict[str, Any]:
        """Return aggregate performance metrics across all forex trades."""
        if not _DB_AVAILABLE:
            return {}
        try:
            from datetime import timedelta

            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            async with get_async_session() as session:
                result = await session.execute(
                    select(ForexTradeHistory)
                    .where(
                        ForexTradeHistory.entry_time >= cutoff,
                        ForexTradeHistory.exit_time.isnot(None),
                    )
                )
                trades = result.scalars().all()
            return _compute_stats(trades, label="overall")
        except Exception as exc:
            logger.warning("ForexTradeJournal.get_overall_stats error: {}", exc)
            return {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _optional_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _trade_to_dict(t: "ForexTradeHistory") -> dict:
    return {
        "id": t.id,
        "order_id": t.order_id,
        "symbol": t.symbol,
        "side": t.side,
        "lot_size": t.lot_size,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "stop_loss_price": t.stop_loss_price,
        "take_profit_price": t.take_profit_price,
        "stop_loss_pips": t.stop_loss_pips,
        "take_profit_pips": t.take_profit_pips,
        "pip_pnl": t.pip_pnl,
        "usd_pnl": t.usd_pnl,
        "spread_at_entry": t.spread_at_entry,
        "leverage": t.leverage,
        "margin_used": t.margin_used,
        "swap_cost": t.swap_cost,
        "commission": t.commission,
        "strategy": t.strategy,
        "session": t.session,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "duration_seconds": t.duration_seconds,
        "exit_reason": t.exit_reason,
        "max_favorable_pips": t.max_favorable_pips,
        "max_adverse_pips": t.max_adverse_pips,
        "notes": t.notes,
    }


def _daily_to_dict(r: "ForexDailyPerformance") -> dict:
    return {
        "id": r.id,
        "date": r.date.isoformat() if r.date else None,
        "starting_equity": r.starting_equity,
        "ending_equity": r.ending_equity,
        "total_pnl_usd": r.total_pnl_usd,
        "total_pnl_pips": r.total_pnl_pips,
        "total_trades": r.total_trades,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": r.win_rate,
        "profit_factor": r.profit_factor,
        "max_drawdown_pct": r.max_drawdown_pct,
        "best_trade_pips": r.best_trade_pips,
        "worst_trade_pips": r.worst_trade_pips,
        "avg_win_pips": r.avg_win_pips,
        "avg_loss_pips": r.avg_loss_pips,
        "total_lots_traded": r.total_lots_traded,
        "total_commission": r.total_commission,
        "total_swap": r.total_swap,
        "london_trades": r.london_trades,
        "london_pnl_pips": r.london_pnl_pips,
        "ny_trades": r.ny_trades,
        "ny_pnl_pips": r.ny_pnl_pips,
        "asian_trades": r.asian_trades,
        "asian_pnl_pips": r.asian_pnl_pips,
        "sydney_trades": r.sydney_trades,
        "sydney_pnl_pips": r.sydney_pnl_pips,
    }


def _active_pos_to_dict(r: "ForexActivePosition") -> dict:
    return {
        "id": r.id,
        "order_id": r.order_id,
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
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        "trailing_stop_active": r.trailing_stop_active,
        "break_even_active": r.break_even_active,
        "partial_closes": r.partial_closes,
    }


def _compute_stats(trades: list, label: str = "") -> dict:
    """Compute aggregate pip/USD stats for a list of ForexTradeHistory rows."""
    if not trades:
        return {
            "label": label,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pip_pnl": 0.0,
            "total_usd_pnl": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
            "profit_factor": 0.0,
            "expectancy_pips": 0.0,
            "best_trade_pips": 0.0,
            "worst_trade_pips": 0.0,
            "avg_duration_sec": 0,
        }

    wins = [t for t in trades if (t.pip_pnl or 0.0) > 0]
    losses = [t for t in trades if (t.pip_pnl or 0.0) <= 0]
    win_pips = [t.pip_pnl for t in wins]
    loss_pips = [t.pip_pnl for t in losses]

    total_pip_pnl = sum(t.pip_pnl or 0.0 for t in trades)
    total_usd_pnl = sum(t.usd_pnl or 0.0 for t in trades)

    gross_profit = sum(p for p in win_pips if p)
    gross_loss = abs(sum(p for p in loss_pips if p))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = (sum(win_pips) / len(wins)) if wins else 0.0
    avg_loss = (sum(loss_pips) / len(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    durations = [t.duration_seconds for t in trades if t.duration_seconds is not None]
    avg_duration = int(sum(durations) / len(durations)) if durations else 0

    all_pips = [t.pip_pnl or 0.0 for t in trades]

    return {
        "label": label,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "total_pip_pnl": round(total_pip_pnl, 2),
        "total_usd_pnl": round(total_usd_pnl, 2),
        "avg_win_pips": round(avg_win, 2),
        "avg_loss_pips": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "expectancy_pips": round(expectancy, 2),
        "best_trade_pips": round(max(all_pips), 2) if all_pips else 0.0,
        "worst_trade_pips": round(min(all_pips), 2) if all_pips else 0.0,
        "avg_duration_sec": avg_duration,
    }
