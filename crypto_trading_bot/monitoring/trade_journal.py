"""Automated trade journal with entry/exit context and AI reasoning."""

from __future__ import annotations

import asyncio
import pathlib
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

_DB_PATH = pathlib.Path(__file__).parent.parent / "data" / "trading_bot.db"

_CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    symbol TEXT,
    direction TEXT,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    leverage INTEGER,
    strategy TEXT,
    opened_at TEXT,
    closed_at TEXT,
    status TEXT
)
"""


def _db_init(db_path: pathlib.Path) -> None:
    """Create the trades table if it does not already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TRADES_SQL)
        conn.commit()


def _db_upsert(db_path: pathlib.Path, record: dict) -> None:
    """Insert or replace a trade record in the database."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO trades
                (id, symbol, direction, entry_price, exit_price,
                 pnl, pnl_pct, leverage, strategy, opened_at, closed_at, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.get("trade_id"),
                record.get("symbol"),
                record.get("direction"),
                record.get("entry_price"),
                record.get("exit_price"),
                record.get("pnl"),
                record.get("pnl_pct"),
                record.get("leverage", 1),
                record.get("strategy"),
                record.get("entry_time"),
                record.get("exit_time"),
                "closed" if record.get("exit_price") is not None else "open",
            ),
        )
        conn.commit()


def _db_recent(db_path: pathlib.Path, limit: int) -> List[dict]:
    """Return the most recent *limit* trades from the database."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


class TradeJournal:
    """Records complete trade context for later analysis and reporting."""

    def __init__(self) -> None:
        # trade_id → trade record
        self._entries: Dict[str, dict] = {}
        self._db_ready = False
        # Best-effort sync init: create the DB directory and table if accessible.
        try:
            _db_init(_DB_PATH)
            self._db_ready = True
        except Exception as exc:
            logger.warning("Trade journal DB init failed (will retry on first write): {}", exc)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_entry(
        self,
        trade: dict,
        reasoning: str,
        market_context: dict,
    ) -> str:
        """Record a trade entry with AI reasoning and market context.

        Args:
            trade: Trade details (symbol, direction, size, entry_price, etc.).
            reasoning: Human/AI reasoning for taking the trade.
            market_context: Snapshot of market conditions at entry (regime,
                volatility, sentiment, indicators, etc.).

        Returns:
            Generated trade ID string.
        """
        trade_id = trade.get("trade_id") or str(uuid.uuid4())
        record = {
            "trade_id": trade_id,
            "symbol": trade.get("symbol"),
            "direction": trade.get("direction"),
            "size": trade.get("size"),
            "entry_price": trade.get("entry_price"),
            "stop_loss": trade.get("stop_loss"),
            "take_profit_levels": trade.get("take_profit_levels", []),
            "leverage": trade.get("leverage", 1),
            "strategy": trade.get("strategy"),
            "entry_time": datetime.now(tz=timezone.utc).isoformat(),
            "reasoning": reasoning,
            "market_context_entry": market_context,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "pnl": None,
            "market_context_exit": None,
        }
        self._entries[trade_id] = record
        logger.info(
            "Journal: entry recorded for trade_id={} {} {}",
            trade_id,
            trade.get("symbol"),
            trade.get("direction"),
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist(record))
        except RuntimeError:
            pass  # Not in async context; record is retained in memory
        return trade_id

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        reason: str,
        pnl: float,
        market_context: Optional[dict] = None,
    ) -> None:
        """Record the exit of a trade.

        Args:
            trade_id: The ID returned by :meth:`record_entry`.
            exit_price: Actual exit price.
            reason: Reason for exit (e.g. ``"stop_loss"``, ``"take_profit"``, ``"manual"``).
            pnl: Realised PnL for this trade.
            market_context: Optional market snapshot at exit.
        """
        record = self._entries.get(trade_id)
        if record is None:
            logger.warning("Journal: trade_id={} not found for exit recording", trade_id)
            return
        record["exit_price"] = exit_price
        record["exit_time"] = datetime.now(tz=timezone.utc).isoformat()
        record["exit_reason"] = reason
        record["pnl"] = pnl
        entry_price = record.get("entry_price") or 0.0
        record["pnl_pct"] = (pnl / entry_price * 100.0) if entry_price else 0.0
        record["market_context_exit"] = market_context or {}
        logger.info(
            "Journal: exit recorded for trade_id={} pnl={:.4f} reason={}", trade_id, pnl, reason
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist(record))
        except RuntimeError:
            pass  # Not in async context; record is retained in memory

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[dict]:
        """Return filtered trade history.

        Args:
            symbol: Filter by symbol (optional).
            start_date: ISO date string lower bound for entry_time (optional).
            end_date: ISO date string upper bound for entry_time (optional).

        Returns:
            List of trade records matching the filter criteria.
        """
        records = list(self._entries.values())
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol]
        if start_date:
            records = [r for r in records if r.get("entry_time", "") >= start_date]
        if end_date:
            records = [r for r in records if r.get("entry_time", "") <= end_date]
        return sorted(records, key=lambda r: r.get("entry_time", ""), reverse=True)

    def generate_trade_report(self, trade_id: str) -> dict:
        """Generate a full report for a single trade.

        Args:
            trade_id: Trade identifier.

        Returns:
            Complete trade record dict, or an error dict if not found.
        """
        record = self._entries.get(trade_id)
        if record is None:
            logger.warning("Journal: trade_id={} not found", trade_id)
            return {"error": f"Trade {trade_id} not found"}

        entry_price = record.get("entry_price") or 0.0
        exit_price = record.get("exit_price") or 0.0
        pnl = record.get("pnl") or 0.0

        report = dict(record)
        if entry_price and exit_price:
            report["price_change_pct"] = (exit_price - entry_price) / entry_price * 100.0
        report["is_winner"] = pnl > 0 if record.get("pnl") is not None else None
        return report

    # ------------------------------------------------------------------
    # New query helpers
    # ------------------------------------------------------------------

    def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Return up to *limit* most-recent trades, newest first.

        Prefers in-memory records for speed; falls back to the SQLite
        database when the in-memory store is empty (e.g. after a restart).

        Args:
            limit: Maximum number of trades to return.

        Returns:
            List of trade record dicts sorted newest-first.
        """
        records = sorted(
            self._entries.values(),
            key=lambda r: r.get("entry_time", ""),
            reverse=True,
        )[:limit]
        if not records:
            try:
                records = _db_recent(_DB_PATH, limit)
            except Exception as exc:
                logger.warning("Could not read trades from DB: {}", exc)
        return list(records)

    def get_trade_stats(self) -> dict:
        """Return aggregate statistics computed from all recorded trades.

        Returns:
            Dict with keys: total_trades, closed_trades, win_count, loss_count,
            win_rate (0–100), total_pnl, profit_factor.
        """
        all_trades = list(self._entries.values())
        closed = [t for t in all_trades if t.get("pnl") is not None]
        total = len(all_trades)
        wins = [t for t in closed if (t.get("pnl") or 0.0) > 0]
        losses = [t for t in closed if (t.get("pnl") or 0.0) < 0]
        gross_profit = sum(t.get("pnl", 0.0) for t in wins)
        gross_loss = abs(sum(t.get("pnl", 0.0) for t in losses))
        profit_factor = (
            (gross_profit / gross_loss)
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )
        win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
        total_pnl = sum(t.get("pnl", 0.0) for t in closed)
        return {
            "total_trades": total,
            "closed_trades": len(closed),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _persist(self, record: dict) -> None:
        """Persist *record* to the SQLite database asynchronously."""
        try:
            if not self._db_ready:
                await asyncio.to_thread(_db_init, _DB_PATH)
                self._db_ready = True
            await asyncio.to_thread(_db_upsert, _DB_PATH, record)
        except Exception as exc:
            logger.warning("Trade journal DB write failed: {}", exc)
