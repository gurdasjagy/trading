"""Standard trading performance metrics for backtest results."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from loguru import logger


class PerformanceMetrics:
    """Calculates all standard trading performance metrics."""

    def calculate_all(self, trades: List[dict], equity_curve: List[float]) -> dict:
        """Calculate and return a complete performance metric dictionary.

        Args:
            trades: List of closed trade dicts (each must have a ``pnl_pct`` key
                and optionally ``entry_time``, ``exit_time``).
            equity_curve: List of portfolio value snapshots over time.

        Returns:
            Dict of metric names → values.
        """
        if not trades:
            logger.warning("No trades provided — returning zeroed metrics")
            return self._zero_metrics()

        returns = [t.get("pnl_pct", 0.0) / 100.0 for t in trades]
        max_dd = self.max_drawdown(equity_curve) if len(equity_curve) > 1 else 0.0
        total_return = (
            (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            if len(equity_curve) >= 2 and equity_curve[0] != 0
            else 0.0
        )
        avg_wl = self.average_win_loss(trades)

        metrics = {
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "max_drawdown_pct": max_dd * 100,
            "calmar_ratio": self.calmar_ratio(returns, max_dd),
            "win_rate": self.win_rate(trades),
            "profit_factor": self.profit_factor(trades),
            "expectancy": self.expectancy(trades),
            "avg_win_pct": avg_wl.get("avg_win_pct", 0.0),
            "avg_loss_pct": avg_wl.get("avg_loss_pct", 0.0),
            "win_loss_ratio": avg_wl.get("win_loss_ratio", 0.0),
            "recovery_factor": self.recovery_factor(total_return, max_dd),
            "total_return_pct": total_return * 100,
            "monthly_returns": self.monthly_returns(trades),
            "total_trades": len(trades),
            "winning_trades": sum(1 for t in trades if t.get("pnl", 0.0) > 0),
            "losing_trades": sum(1 for t in trades if t.get("pnl", 0.0) <= 0),
        }
        return metrics

    # ------------------------------------------------------------------
    # Risk-adjusted return metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(
        self,
        returns: List[float],
        risk_free: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Annualised Sharpe ratio.

        Args:
            returns: List of period returns (fractions, not percentages).
            risk_free: Risk-free rate per period.
            periods_per_year: Number of trading periods per year.

        Returns:
            Sharpe ratio, or 0.0 if standard deviation is zero.
        """
        if len(returns) < 2:
            return 0.0
        excess = [r - risk_free for r in returns]
        mean = sum(excess) / len(excess)
        variance = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(periods_per_year)

    def sortino_ratio(
        self,
        returns: List[float],
        risk_free: float = 0.0,
    ) -> float:
        """Annualised Sortino ratio (penalises only downside volatility).

        Args:
            returns: List of period returns (fractions).
            risk_free: Minimum acceptable return per period.

        Returns:
            Sortino ratio, or 0.0 if downside deviation is zero.
        """
        if len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        downside = [min(0.0, r - risk_free) for r in returns]
        downside_var = sum(d**2 for d in downside) / len(downside)
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return 0.0
        return ((mean_return - risk_free) / downside_std) * math.sqrt(252)

    def max_drawdown(self, equity_curve: List[float]) -> float:
        """Maximum peak-to-trough drawdown as a fraction.

        Args:
            equity_curve: Portfolio value over time.

        Returns:
            Max drawdown fraction (0.0 → 1.0).  Always non-negative.
        """
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
        return max_dd

    def calmar_ratio(self, returns: List[float], max_dd: float) -> float:
        """Calmar ratio: annualised return divided by max drawdown.

        Args:
            returns: List of period returns.
            max_dd: Maximum drawdown fraction.

        Returns:
            Calmar ratio, or 0.0 if max_dd is zero.
        """
        if not returns or max_dd == 0:
            return 0.0
        annual_return = (sum(returns) / len(returns)) * 252
        return annual_return / max_dd

    # ------------------------------------------------------------------
    # Trade-level statistics
    # ------------------------------------------------------------------

    def win_rate(self, trades: List[dict]) -> float:
        """Fraction of trades with positive PnL.

        Returns:
            Win rate in [0, 1], or 0.0 for empty trade list.
        """
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        return wins / len(trades)

    def profit_factor(self, trades: List[dict]) -> float:
        """Gross profit divided by gross loss.

        Returns:
            Profit factor, or 0.0 if there are no losses.
        """
        gross_profit = sum(t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) < 0))
        if gross_loss == 0:
            return 0.0 if gross_profit == 0 else float("inf")
        return gross_profit / gross_loss

    def expectancy(self, trades: List[dict]) -> float:
        """Average dollar PnL per trade.

        Returns:
            Expectancy in quote currency, or 0.0 for empty trade list.
        """
        if not trades:
            return 0.0
        return sum(t.get("pnl", 0.0) for t in trades) / len(trades)

    def average_win_loss(self, trades: List[dict]) -> dict:
        """Average win, average loss, and win/loss ratio.

        Returns:
            Dict with ``avg_win_pct``, ``avg_loss_pct``, ``win_loss_ratio``.
        """
        wins = [t.get("pnl_pct", 0.0) for t in trades if t.get("pnl", 0.0) > 0]
        losses = [abs(t.get("pnl_pct", 0.0)) for t in trades if t.get("pnl", 0.0) < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        ratio = avg_win / avg_loss if avg_loss != 0 else 0.0
        return {"avg_win_pct": avg_win, "avg_loss_pct": avg_loss, "win_loss_ratio": ratio}

    def recovery_factor(self, total_return: float, max_dd: float) -> float:
        """Net profit divided by max drawdown.

        Args:
            total_return: Total return as a fraction.
            max_dd: Maximum drawdown fraction.

        Returns:
            Recovery factor, or 0.0 if max_dd is zero.
        """
        if max_dd == 0:
            return 0.0
        return total_return / max_dd

    def monthly_returns(self, trades: List[dict]) -> Dict[str, float]:
        """Aggregate trade PnL by calendar month.

        Args:
            trades: List of closed trade dicts with ``exit_time`` (datetime)
                and ``pnl_pct`` fields.

        Returns:
            Dict mapping ``"YYYY-MM"`` to total return percentage for that month.
        """
        monthly: dict = defaultdict(float)
        for trade in trades:
            exit_time = trade.get("exit_time")
            if exit_time is None:
                continue
            if isinstance(exit_time, str):
                try:
                    exit_time = datetime.fromisoformat(exit_time)
                except ValueError:
                    continue
            key = exit_time.strftime("%Y-%m")
            monthly[key] += trade.get("pnl_pct", 0.0)
        return dict(sorted(monthly.items()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_metrics() -> dict:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "win_loss_ratio": 0.0,
            "recovery_factor": 0.0,
            "total_return_pct": 0.0,
            "monthly_returns": {},
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
