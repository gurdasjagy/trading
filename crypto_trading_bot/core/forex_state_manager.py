"""Forex-specific state management — session PnL, pip metrics, daily performance."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ForexDailyPerformance:
    """Per-day performance record for the forex engine."""

    date: str          # ISO date string "YYYY-MM-DD"
    session: str       # e.g. "London", "New York", "Tokyo"
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pips: float = 0.0     # total pips won/lost
    net_pips: float = 0.0       # after spread cost
    gross_pnl_usdt: float = 0.0
    net_pnl_usdt: float = 0.0
    lot_size_used: float = 0.0
    max_drawdown_pips: float = 0.0
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        return d


@dataclass
class ForexSessionStats:
    """Rolling stats for the current trading session."""

    session_name: str = "Unknown"
    session_start: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pips_pnl: float = 0.0
    usdt_pnl: float = 0.0
    lot_sizes: List[float] = field(default_factory=list)
    peak_pnl: float = 0.0
    trough_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades

    @property
    def avg_lot_size(self) -> float:
        if not self.lot_sizes:
            return 0.0
        return sum(self.lot_sizes) / len(self.lot_sizes)


# ---------------------------------------------------------------------------
# ForexStateManager
# ---------------------------------------------------------------------------


class ForexStateManager:
    """Tracks and persists forex-specific state.

    * Session PnL, daily PnL in pips.
    * Per-session performance (London/NY/Tokyo/Sydney win rate).
    * Persistent storage to ``data/forex_state.json``.
    """

    STATE_FILE = Path("data/forex_state.json")
    PERFORMANCE_FILE = Path("data/forex_performance.json")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._current_session = ForexSessionStats()
        self._daily_performance: Dict[str, List[ForexDailyPerformance]] = {}  # date → list
        self._session_history: List[ForexSessionStats] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load persisted state from disk."""
        async with self._lock:
            try:
                if self.STATE_FILE.exists():
                    with open(self.STATE_FILE) as f:
                        data = json.load(f)
                    sess = data.get("current_session", {})
                    self._current_session = ForexSessionStats(
                        session_name=sess.get("session_name", "Unknown"),
                        session_start=sess.get("session_start", ""),
                        trades=sess.get("trades", 0),
                        wins=sess.get("wins", 0),
                        losses=sess.get("losses", 0),
                        pips_pnl=sess.get("pips_pnl", 0.0),
                        usdt_pnl=sess.get("usdt_pnl", 0.0),
                        lot_sizes=sess.get("lot_sizes", []),
                        peak_pnl=sess.get("peak_pnl", 0.0),
                        trough_pnl=sess.get("trough_pnl", 0.0),
                    )
                    logger.info("ForexStateManager: loaded state (session={}, trades={})",
                                self._current_session.session_name,
                                self._current_session.trades)

                if self.PERFORMANCE_FILE.exists():
                    with open(self.PERFORMANCE_FILE) as f:
                        perf_data = json.load(f)
                    for date_str, records in perf_data.items():
                        self._daily_performance[date_str] = [
                            ForexDailyPerformance(**r) for r in records
                        ]
            except Exception as e:
                logger.warning("ForexStateManager: failed to load state — {}", e)
            self._loaded = True

    async def save(self) -> None:
        """Persist state to disk."""
        async with self._lock:
            try:
                self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "current_session": asdict(self._current_session),
                    "saved_at": datetime.now(tz=timezone.utc).isoformat(),
                }
                with open(self.STATE_FILE, "w") as f:
                    json.dump(state, f, indent=2)

                perf_data = {
                    d: [r.to_dict() for r in records]
                    for d, records in self._daily_performance.items()
                }
                with open(self.PERFORMANCE_FILE, "w") as f:
                    json.dump(perf_data, f, indent=2)
            except Exception as e:
                logger.warning("ForexStateManager: failed to save state — {}", e)

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    async def record_trade(
        self,
        symbol: str,
        direction: str,
        pips: float,
        pnl_usdt: float,
        lot_size: float,
        session_name: str,
    ) -> None:
        """Record a completed forex trade.

        Args:
            symbol: Trading symbol (e.g. ``"XAU/USD"``).
            direction: ``"long"`` or ``"short"``.
            pips: Pips gained (positive = win, negative = loss).
            pnl_usdt: P&L in USDT.
            lot_size: Lot size used.
            session_name: Current session name.
        """
        async with self._lock:
            is_win = pips > 0
            # Update current session
            self._current_session.trades += 1
            if is_win:
                self._current_session.wins += 1
            else:
                self._current_session.losses += 1
            self._current_session.pips_pnl += pips
            self._current_session.usdt_pnl += pnl_usdt
            self._current_session.lot_sizes.append(lot_size)
            if self._current_session.pips_pnl > self._current_session.peak_pnl:
                self._current_session.peak_pnl = self._current_session.pips_pnl
            if self._current_session.pips_pnl < self._current_session.trough_pnl:
                self._current_session.trough_pnl = self._current_session.pips_pnl

            # Update daily performance
            today = date.today().isoformat()
            if today not in self._daily_performance:
                self._daily_performance[today] = []

            # Find or create record for this symbol+session
            record: Optional[ForexDailyPerformance] = None
            for r in self._daily_performance[today]:
                if r.symbol == symbol and r.session == session_name:
                    record = r
                    break
            if record is None:
                record = ForexDailyPerformance(date=today, session=session_name, symbol=symbol)
                self._daily_performance[today].append(record)

            record.trades += 1
            if is_win:
                record.wins += 1
            else:
                record.losses += 1
            record.gross_pips += pips
            record.gross_pnl_usdt += pnl_usdt
            record.lot_size_used = lot_size
            if pips > record.best_trade_pips:
                record.best_trade_pips = pips
            if pips < record.worst_trade_pips:
                record.worst_trade_pips = pips

        await self.save()

    async def start_new_session(self, session_name: str) -> None:
        """Archive current session stats and start fresh."""
        async with self._lock:
            if self._current_session.trades > 0:
                self._session_history.append(self._current_session)
            self._current_session = ForexSessionStats(
                session_name=session_name,
                session_start=datetime.now(tz=timezone.utc).isoformat(),
            )
        await self.save()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_session_win_rate(self, session_name: str) -> float:
        """Return historical win rate for a named session."""
        total_wins = 0
        total_trades = 0
        # Check all daily performance records
        for records in self._daily_performance.values():
            for r in records:
                if session_name.lower() in r.session.lower():
                    total_wins += r.wins
                    total_trades += r.trades
        if total_trades == 0:
            return 0.5  # assume 50% if no history
        return total_wins / total_trades

    def get_daily_pnl_pips(self, target_date: Optional[str] = None) -> float:
        """Return total pip P&L for a specific date (defaults to today)."""
        d = target_date or date.today().isoformat()
        records = self._daily_performance.get(d, [])
        return sum(r.gross_pips for r in records)

    def get_daily_pnl_usdt(self, target_date: Optional[str] = None) -> float:
        """Return total USDT P&L for a specific date (defaults to today)."""
        d = target_date or date.today().isoformat()
        records = self._daily_performance.get(d, [])
        return sum(r.gross_pnl_usdt for r in records)

    def get_current_session_stats(self) -> ForexSessionStats:
        """Return stats for the current trading session."""
        return self._current_session

    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Return aggregated performance over the last *days* days."""
        from datetime import timedelta
        total_trades = total_wins = 0
        total_pips = total_usdt = 0.0
        today = date.today()
        for i in range(days):
            d = (today - timedelta(days=i)).isoformat()
            for r in self._daily_performance.get(d, []):
                total_trades += r.trades
                total_wins += r.wins
                total_pips += r.gross_pips
                total_usdt += r.gross_pnl_usdt
        return {
            "days": days,
            "total_trades": total_trades,
            "win_rate": total_wins / total_trades if total_trades > 0 else 0.0,
            "total_pips": round(total_pips, 1),
            "total_pnl_usdt": round(total_usdt, 2),
            "avg_pips_per_day": round(total_pips / days, 1) if days > 0 else 0.0,
        }
