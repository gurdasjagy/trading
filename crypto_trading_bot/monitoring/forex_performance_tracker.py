"""Forex performance tracker — aggregates pip/USD metrics across all forex trades.

Provides:
* Overall win rate, profit factor, expectancy, average pip P&L
* Per-session breakdown (London, New York, Asian, Sydney)
* Per-symbol breakdown (XAU/USD, XAG/USD, majors)
* Per-strategy breakdown
* Rolling drawdown and equity-curve tracking
* JSON-serialisable report generation for the dashboard API

The tracker is lightweight and maintains in-memory caches updated on every
trade.  For persistence, all trade records go through
:class:`~monitoring.forex_trade_journal.ForexTradeJournal`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


class ForexPerformanceTracker:
    """Real-time forex performance analytics.

    Usage::

        tracker = ForexPerformanceTracker(lookback_trades=500)
        tracker.record_trade(trade_dict)
        report = tracker.generate_report()
    """

    def __init__(self, lookback_trades: int = 500) -> None:
        self._lookback = lookback_trades
        # Rolling trade ring-buffer
        self._trades: Deque[dict] = deque(maxlen=lookback_trades)

        # Aggregate counters
        self._total_trades = 0
        self._total_wins = 0
        self._total_losses = 0
        self._total_pip_pnl = 0.0
        self._total_usd_pnl = 0.0
        self._gross_profit_pips = 0.0
        self._gross_loss_pips = 0.0

        # Per-dimension breakdowns: dimension → {wins, losses, pip_pnl, usd_pnl, trades}
        self._by_session: Dict[str, dict] = defaultdict(lambda: _empty_bucket())
        self._by_symbol: Dict[str, dict] = defaultdict(lambda: _empty_bucket())
        self._by_strategy: Dict[str, dict] = defaultdict(lambda: _empty_bucket())

        # Equity curve and drawdown (updated on each trade)
        self._equity_curve: Deque[float] = deque(maxlen=lookback_trades)
        self._peak_equity: float = 0.0
        self._max_drawdown_pct: float = 0.0
        self._current_equity: float = 0.0

        # Daily P&L tracking
        self._daily_pip_pnl: Dict[str, float] = {}  # date_str → pip_pnl
        self._daily_usd_pnl: Dict[str, float] = {}  # date_str → usd_pnl
        self._daily_trades: Dict[str, int] = {}      # date_str → trade count

        # Trade streak
        self._current_streak: int = 0   # positive = wins, negative = losses
        self._max_win_streak: int = 0
        self._max_loss_streak: int = 0

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict) -> None:
        """Record a completed forex trade and update all metrics.

        Expected keys: ``symbol``, ``side``, ``pip_pnl``, ``usd_pnl``,
        ``session``, ``strategy``, ``entry_time``, ``exit_time``,
        ``lot_size``, ``exit_reason``.
        """
        pip_pnl = float(trade.get("pip_pnl") or 0.0)
        usd_pnl = float(trade.get("usd_pnl") or 0.0)
        won = pip_pnl > 0
        symbol = trade.get("symbol", "UNKNOWN")
        session = (trade.get("session") or "unknown").lower()
        strategy = trade.get("strategy") or "unknown"

        # Aggregate
        self._total_trades += 1
        self._total_pip_pnl += pip_pnl
        self._total_usd_pnl += usd_pnl
        if won:
            self._total_wins += 1
            self._gross_profit_pips += pip_pnl
        else:
            self._total_losses += 1
            self._gross_loss_pips += abs(pip_pnl)

        # Per-dimension
        for bucket in (
            self._by_session[session],
            self._by_symbol[symbol],
            self._by_strategy[strategy],
        ):
            _update_bucket(bucket, won, pip_pnl, usd_pnl)

        # Streak
        if won:
            self._current_streak = max(1, self._current_streak + 1)
            self._max_win_streak = max(self._max_win_streak, self._current_streak)
        else:
            self._current_streak = min(-1, self._current_streak - 1)
            self._max_loss_streak = max(self._max_loss_streak, abs(self._current_streak))

        # Daily
        entry_time = trade.get("entry_time")
        day_key = (
            datetime.fromisoformat(str(entry_time)).strftime("%Y-%m-%d")
            if entry_time
            else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        )
        self._daily_pip_pnl[day_key] = self._daily_pip_pnl.get(day_key, 0.0) + pip_pnl
        self._daily_usd_pnl[day_key] = self._daily_usd_pnl.get(day_key, 0.0) + usd_pnl
        self._daily_trades[day_key] = self._daily_trades.get(day_key, 0) + 1

        # Rolling buffer
        self._trades.append(trade)

    def update_equity(self, equity: float) -> None:
        """Update current equity and drawdown tracking."""
        if equity <= 0:
            return
        self._current_equity = equity
        self._equity_curve.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
            self._max_drawdown_pct = max(self._max_drawdown_pct, dd_pct)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report.

        Returns:
            Dict suitable for direct JSON serialisation and dashboard display.
        """
        win_rate = (
            self._total_wins / self._total_trades if self._total_trades > 0 else 0.0
        )
        profit_factor = (
            self._gross_profit_pips / self._gross_loss_pips
            if self._gross_loss_pips > 0
            else 0.0
        )
        avg_win = (
            self._gross_profit_pips / self._total_wins if self._total_wins > 0 else 0.0
        )
        avg_loss = (
            -self._gross_loss_pips / self._total_losses if self._total_losses > 0 else 0.0
        )
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

        return {
            "summary": {
                "total_trades": self._total_trades,
                "wins": self._total_wins,
                "losses": self._total_losses,
                "win_rate": round(win_rate, 4),
                "win_rate_pct": round(win_rate * 100, 2),
                "total_pip_pnl": round(self._total_pip_pnl, 2),
                "total_usd_pnl": round(self._total_usd_pnl, 2),
                "profit_factor": round(profit_factor, 4),
                "avg_win_pips": round(avg_win, 2),
                "avg_loss_pips": round(avg_loss, 2),
                "expectancy_pips": round(expectancy, 2),
                "current_streak": self._current_streak,
                "max_win_streak": self._max_win_streak,
                "max_loss_streak": self._max_loss_streak,
                "max_drawdown_pct": round(self._max_drawdown_pct, 2),
                "current_equity": self._current_equity,
                "peak_equity": self._peak_equity,
            },
            "by_session": {
                k: _bucket_to_dict(v) for k, v in self._by_session.items()
            },
            "by_symbol": {
                k: _bucket_to_dict(v) for k, v in self._by_symbol.items()
            },
            "by_strategy": {
                k: _bucket_to_dict(v) for k, v in self._by_strategy.items()
            },
            "daily": self._build_daily_series(),
            "equity_curve": list(self._equity_curve)[-50:],  # Last 50 data points
            "recent_trades": list(self._trades)[-10:],
        }

    def get_session_breakdown(self) -> Dict[str, dict]:
        """Return performance breakdown by trading session."""
        return {k: _bucket_to_dict(v) for k, v in self._by_session.items()}

    def get_symbol_breakdown(self) -> Dict[str, dict]:
        """Return performance breakdown by symbol."""
        return {k: _bucket_to_dict(v) for k, v in self._by_symbol.items()}

    def get_strategy_breakdown(self) -> Dict[str, dict]:
        """Return performance breakdown by strategy."""
        return {k: _bucket_to_dict(v) for k, v in self._by_strategy.items()}

    def get_recent_trades(self, limit: int = 20) -> List[dict]:
        """Return the most recent *limit* completed trades."""
        return list(self._trades)[-limit:]

    def get_daily_summary(self, days: int = 30) -> List[dict]:
        """Return daily performance for the last *days* calendar days."""
        return self._build_daily_series(limit=days)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_daily_series(self, limit: Optional[int] = None) -> List[dict]:
        """Build an ordered list of daily performance dicts."""
        if not self._daily_pip_pnl:
            return []
        all_days = sorted(self._daily_pip_pnl.keys())
        if limit:
            all_days = all_days[-limit:]
        result = []
        running_pips = 0.0
        running_usd = 0.0
        for day in all_days:
            pip_pnl = self._daily_pip_pnl.get(day, 0.0)
            usd_pnl = self._daily_usd_pnl.get(day, 0.0)
            running_pips += pip_pnl
            running_usd += usd_pnl
            result.append({
                "date": day,
                "pip_pnl": round(pip_pnl, 2),
                "usd_pnl": round(usd_pnl, 2),
                "trades": self._daily_trades.get(day, 0),
                "cumulative_pip_pnl": round(running_pips, 2),
                "cumulative_usd_pnl": round(running_usd, 2),
            })
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict:
    return {"trades": 0, "wins": 0, "losses": 0, "pip_pnl": 0.0, "usd_pnl": 0.0}


def _update_bucket(bucket: dict, won: bool, pip_pnl: float, usd_pnl: float) -> None:
    bucket["trades"] += 1
    bucket["pip_pnl"] += pip_pnl
    bucket["usd_pnl"] += usd_pnl
    if won:
        bucket["wins"] += 1
    else:
        bucket["losses"] += 1


def _bucket_to_dict(b: dict) -> dict:
    trades = b["trades"]
    win_rate = b["wins"] / trades if trades > 0 else 0.0
    return {
        "trades": trades,
        "wins": b["wins"],
        "losses": b["losses"],
        "win_rate": round(win_rate, 4),
        "win_rate_pct": round(win_rate * 100, 2),
        "pip_pnl": round(b["pip_pnl"], 2),
        "usd_pnl": round(b["usd_pnl"], 2),
    }
