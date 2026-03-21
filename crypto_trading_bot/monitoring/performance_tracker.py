"""Performance tracker — Sharpe, Sortino, max drawdown, win rate, and more."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from loguru import logger


class PerformanceTracker:
    """Tracks and calculates comprehensive trading performance metrics."""

    def __init__(self) -> None:
        self._trades: List[dict] = []
        self._equity_curve: List[float] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade.

        Args:
            trade: Trade dict with at least ``pnl``, ``symbol``, ``entry_price``,
                ``exit_price``, and ``direction`` keys.
        """
        self._trades.append(trade)
        # Update equity curve: last equity + pnl
        last_equity = self._equity_curve[-1] if self._equity_curve else 10000.0
        self._equity_curve.append(last_equity + trade.get("pnl", 0.0))
        logger.debug(
            "Trade recorded: pnl={:.4f} equity={:.2f}", trade.get("pnl", 0), self._equity_curve[-1]
        )

    def set_initial_equity(self, equity: float) -> None:
        """Set the starting equity for the equity curve.

        Args:
            equity: Starting portfolio value.
        """
        if not self._equity_curve:
            self._equity_curve.append(equity)

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------

    def calculate_sharpe(
        self,
        returns: List[float],
        risk_free: float = 0.0,
    ) -> float:
        """Calculate the Sharpe ratio.

        Args:
            returns: List of period returns (e.g. daily PnL fractions).
            risk_free: Risk-free rate per period.

        Returns:
            Sharpe ratio, or 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        excess = [r - risk_free for r in returns]
        mean_excess = sum(excess) / len(excess)
        variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        sharpe = mean_excess / std
        logger.debug("Sharpe ratio: {:.4f}", sharpe)
        return sharpe

    def calculate_sortino(self, returns: List[float], risk_free: float = 0.0) -> float:
        """Calculate the Sortino ratio (penalises only downside volatility).

        Args:
            returns: List of period returns.
            risk_free: Risk-free rate per period.

        Returns:
            Sortino ratio, or 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        excess = [r - risk_free for r in returns]
        mean_excess = sum(excess) / len(excess)
        downside = [r for r in excess if r < 0]
        if not downside:
            return float("inf")
        downside_var = sum(r**2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return 0.0
        sortino = mean_excess / downside_std
        logger.debug("Sortino ratio: {:.4f}", sortino)
        return sortino

    def calculate_max_drawdown(self, equity_curve: Optional[List[float]] = None) -> float:
        """Calculate maximum drawdown as a percentage.

        Args:
            equity_curve: Sequence of equity values. Defaults to internal curve.

        Returns:
            Maximum drawdown percentage (0–100).
        """
        curve = equity_curve or self._equity_curve
        if len(curve) < 2:
            return 0.0
        peak = curve[0]
        max_dd = 0.0
        for value in curve:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak * 100.0
                max_dd = max(max_dd, dd)
        logger.debug("Max drawdown: {:.2f}%", max_dd)
        return max_dd

    def calculate_drawdown_series(self) -> List[float]:
        """Return the drawdown at each point in the equity curve (as a percentage).

        Returns:
            List of drawdown values (0–100) aligned with the equity curve.
        """
        curve = self._equity_curve
        if not curve:
            return []
        peak = curve[0]
        series: List[float] = []
        for value in curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100.0 if peak > 0 else 0.0
            series.append(round(dd, 2))
        return series

    def calculate_win_rate(self, trades: Optional[List[dict]] = None) -> float:
        """Calculate the win rate across all recorded trades.

        Args:
            trades: Trade list to use. Defaults to internal list.

        Returns:
            Win rate as a percentage (0–100).
        """
        t = trades or self._trades
        if not t:
            return 0.0
        wins = sum(1 for trade in t if trade.get("pnl", 0.0) > 0)
        rate = wins / len(t) * 100.0
        logger.debug("Win rate: {:.1f}% ({}/{} trades)", rate, wins, len(t))
        return rate

    def _profit_factor(self) -> float:
        """Return gross profit / gross loss ratio."""
        gross_profit = sum(t["pnl"] for t in self._trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self._trades if t.get("pnl", 0) < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _expectancy(self) -> float:
        """Return average expected gain per trade."""
        if not self._trades:
            return 0.0
        return sum(t.get("pnl", 0.0) for t in self._trades) / len(self._trades)

    def get_pair_pnl(self) -> Dict[str, float]:
        """Return total P&L grouped by trading pair symbol.

        Returns:
            Dict mapping symbol → total realised PnL.
        """
        pair_pnl: Dict[str, float] = defaultdict(float)
        for trade in self._trades:
            symbol = trade.get("symbol", "UNKNOWN")
            pair_pnl[symbol] += trade.get("pnl", 0.0)
        return dict(pair_pnl)

    def get_strategy_pnl(self) -> Dict[str, float]:
        """Return total P&L grouped by strategy name.

        Returns:
            Dict mapping strategy name → total realised PnL.
        """
        strategy_pnl: Dict[str, float] = defaultdict(float)
        for trade in self._trades:
            strategy = trade.get("strategy", trade.get("strategy_name", "unknown"))
            strategy_pnl[strategy] += trade.get("pnl", 0.0)
        return dict(strategy_pnl)

    def get_win_rate_by_hour(self) -> Dict[int, float]:
        """Return win rate (%) for each UTC hour of the day (0–23).

        Trades are bucketed by the UTC hour of their ``closed_at`` timestamp.
        Hours with no trades return 0.0.

        Returns:
            Dict mapping hour (int 0–23) → win rate percentage.
        """
        from datetime import timezone

        hourly: Dict[int, List[bool]] = defaultdict(list)
        for trade in self._trades:
            ts = trade.get("closed_at") or trade.get("exit_time") or trade.get("timestamp")
            if ts is None:
                continue
            try:
                if hasattr(ts, "hour"):
                    hour = ts.hour
                    # Convert to UTC if timezone-aware
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        from datetime import datetime
                        hour = ts.astimezone(timezone.utc).hour
                else:
                    from datetime import datetime
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(str(ts))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                    hour = dt.hour
            except Exception:
                continue
            hourly[hour].append(trade.get("pnl", 0.0) > 0)

        result: Dict[int, float] = {}
        for hour in range(24):
            trades_in_hour = hourly.get(hour, [])
            if trades_in_hour:
                result[hour] = sum(trades_in_hour) / len(trades_in_hour) * 100.0
            else:
                result[hour] = 0.0
        return result

    def _calculate_streak(self) -> dict:
        """Return current and maximum win/loss streak data.

        Returns:
            Dict with keys: ``current_streak`` (positive = wins, negative = losses),
            ``max_consecutive_wins``, ``max_consecutive_losses``.
        """
        if not self._trades:
            return {"current_streak": 0, "max_consecutive_wins": 0, "max_consecutive_losses": 0}
        max_wins = 0
        max_losses = 0
        current = 0
        cur_streak = 0
        for trade in self._trades:
            won = trade.get("pnl", 0.0) > 0
            if won:
                if current > 0:
                    current += 1
                else:
                    current = 1
                max_wins = max(max_wins, current)
            else:
                if current < 0:
                    current -= 1
                else:
                    current = -1
                max_losses = max(max_losses, abs(current))
        cur_streak = current
        return {
            "current_streak": cur_streak,
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
        }

    def _get_avg_trade_duration(self) -> float:
        """Return average trade duration in minutes.

        Uses ``closed_at``/``opened_at`` or ``exit_time``/``entry_time`` fields.
        Returns 0.0 if no duration data is available.
        """
        durations: List[float] = []
        for trade in self._trades:
            # Try numeric timestamps first (ms or seconds)
            opened = trade.get("opened_at") or trade.get("entry_time")
            closed = trade.get("closed_at") or trade.get("exit_time")
            duration_mins = trade.get("duration_mins")
            if duration_mins is not None:
                try:
                    durations.append(float(duration_mins))
                    continue
                except (TypeError, ValueError):
                    pass
            if opened is None or closed is None:
                continue
            try:
                # Handle datetime objects
                if hasattr(opened, "timestamp") and hasattr(closed, "timestamp"):
                    diff_secs = closed.timestamp() - opened.timestamp()
                else:
                    # Numeric: detect milliseconds vs seconds
                    o_ts = float(opened)
                    c_ts = float(closed)
                    if o_ts > 1e12:
                        o_ts /= 1000.0
                    if c_ts > 1e12:
                        c_ts /= 1000.0
                    diff_secs = c_ts - o_ts
                if diff_secs > 0:
                    durations.append(diff_secs / 60.0)
            except (TypeError, ValueError):
                continue
        return sum(durations) / len(durations) if durations else 0.0

    def get_performance_report(self) -> dict:
        """Return a comprehensive performance metrics report.

        Returns:
            Dict with keys: sharpe, sortino, max_drawdown, win_rate,
            profit_factor, avg_win, avg_loss, expectancy, total_trades,
            total_pnl, recovery_factor, calmar_ratio, pair_pnl,
            strategy_pnl, win_rate_by_hour, best_trade, worst_trade,
            avg_trade_duration_mins, max_consecutive_wins,
            max_consecutive_losses, current_streak, total_return_pct.
            Returns zeros when no trades have been recorded yet.
        """
        _empty = {
            "total_trades": 0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_trade_duration_mins": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "current_streak": 0,
            "recovery_factor": 0.0,
            "calmar_ratio": 0.0,
            "pair_pnl": {},
            "strategy_pnl": {},
            "win_rate_by_hour": {h: 0.0 for h in range(24)},
        }
        if not self._trades:
            return _empty

        pnls = [t.get("pnl", 0.0) for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_pnl = sum(pnls)

        # Use equity curve returns
        returns: List[float] = []
        for i in range(1, len(self._equity_curve)):
            prev = self._equity_curve[i - 1]
            returns.append((self._equity_curve[i] - prev) / prev if prev != 0 else 0.0)

        start_equity = self._equity_curve[0] if self._equity_curve else 0.0
        end_equity = self._equity_curve[-1] if self._equity_curve else 0.0
        total_return_pct = (
            (end_equity - start_equity) / start_equity * 100.0
            if start_equity > 0
            else 0.0
        )

        max_dd = self.calculate_max_drawdown()
        sharpe = self.calculate_sharpe(returns)
        sortino = self.calculate_sortino(returns)
        win_rate = self.calculate_win_rate()
        profit_factor = self._profit_factor()
        expectancy = self._expectancy()
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0
        avg_duration = self._get_avg_trade_duration()
        streak_data = self._calculate_streak()
        recovery_factor = (
            total_pnl / (max_dd / 100.0 * start_equity)
            if max_dd > 0 and start_equity > 0
            else 0.0
        )
        calmar = (
            (total_pnl / start_equity * 100.0) / max_dd
            if max_dd > 0 and start_equity > 0
            else 0.0
        )

        return {
            "total_trades": len(self._trades),
            "total_pnl": total_pnl,
            "total_return_pct": round(total_return_pct, 4),
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_trade_duration_mins": round(avg_duration, 1),
            "max_consecutive_wins": streak_data["max_consecutive_wins"],
            "max_consecutive_losses": streak_data["max_consecutive_losses"],
            "current_streak": streak_data["current_streak"],
            "recovery_factor": recovery_factor,
            "calmar_ratio": calmar,
            "pair_pnl": self.get_pair_pnl(),
            "strategy_pnl": self.get_strategy_pnl(),
            "win_rate_by_hour": self.get_win_rate_by_hour(),
        }

    def get_equity_curve(self) -> List[float]:
        """Return the equity curve as a list of equity values.

        Returns:
            List of portfolio equity values over time.
        """
        return list(self._equity_curve)
