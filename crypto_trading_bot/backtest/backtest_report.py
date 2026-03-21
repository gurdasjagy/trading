"""Backtest report generator — stats, equity curve chart, and JSON output."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from backtest.backtest_engine import BacktestResult

_REPORTS_DIR = Path("data/reports")


class BacktestReport:
    """Generates a comprehensive backtest report from a :class:`BacktestResult`.

    Outputs:
    * Console summary (printed via loguru)
    * JSON file saved to ``data/reports/``
    * Equity-curve PNG chart saved to ``data/reports/`` (requires matplotlib)

    Usage::

        report = BacktestReport()
        paths = report.generate(result)
        # paths["json_path"] and paths["chart_path"] contain saved file paths
    """

    def __init__(self, reports_dir: Optional[Path] = None) -> None:
        self._reports_dir = reports_dir or _REPORTS_DIR
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, result: "BacktestResult") -> dict:
        """Generate and persist all report artefacts for *result*.

        Args:
            result: Completed :class:`BacktestResult` from :class:`BacktestEngine`.

        Returns:
            Dict with keys ``"stats"``, ``"json_path"``, and ``"chart_path"``
            (``chart_path`` is ``None`` if matplotlib is unavailable).
        """
        stats = self._compute_stats(result)
        self._print_summary(result, stats)
        json_path = self._save_json(result, stats)
        chart_path = self._save_equity_chart(result)
        return {"stats": stats, "json_path": json_path, "chart_path": chart_path}

    # ------------------------------------------------------------------
    # Stats computation
    # ------------------------------------------------------------------

    def _compute_stats(self, result: "BacktestResult") -> dict:
        """Derive all reportable statistics from *result*."""
        days = max(1, (result.end_date - result.start_date).days)
        annual_factor = 365.0 / days
        annual_return_pct = result.total_return_pct * annual_factor

        pnl_list = [t.get("pnl", 0.0) for t in result.trades]
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        avg_trade_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else 0.0

        win_pnls = [p for p in pnl_list if p > 0]
        loss_pnls = [abs(p) for p in pnl_list if p < 0]
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

        durations = [t.get("duration_hours", 0.0) for t in result.trades]
        max_duration = max(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0

        # Find best and worst trade details
        best_trade = _best_trade(result.trades)
        worst_trade = _worst_trade(result.trades)

        # Max drawdown calculation from equity curve
        max_dd_pct = result.max_drawdown_pct

        return {
            "strategy": result.strategy_name,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "period_days": days,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": round(result.final_capital, 4),
            "total_return_pct": round(result.total_return_pct, 4),
            "annual_return_pct": round(annual_return_pct, 4),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "sortino_ratio": round(result.sortino_ratio, 4),
            "calmar_ratio": round(result.calmar_ratio, 4),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "win_rate": round(result.win_rate, 4),
            "profit_factor": round(result.profit_factor, 4),
            "expectancy": round(result.expectancy, 4),
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "avg_trade_pnl": round(avg_trade_pnl, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "avg_trade_duration_hours": round(result.avg_trade_duration_hours, 2),
            "max_trade_duration_hours": round(max_duration, 2),
            "min_trade_duration_hours": round(min_duration, 2),
            "best_trade_pct": round(result.best_trade_pct, 4),
            "worst_trade_pct": round(result.worst_trade_pct, 4),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "monthly_returns": result.monthly_returns,
        }

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _print_summary(self, result: "BacktestResult", stats: dict) -> None:
        """Print a formatted summary to the console via loguru."""
        pf_str = f"{stats['profit_factor']:.2f}" if math.isfinite(stats["profit_factor"]) else "inf"
        logger.info(
            "\n"
            "╔══════════════════════ BACKTEST REPORT ══════════════════════╗\n"
            "║  Strategy    : {strategy:<42} ║\n"
            "║  Symbol      : {symbol:<42} ║\n"
            "║  Timeframe   : {timeframe:<42} ║\n"
            "║  Period      : {start} → {end}               ║\n"
            "║  Duration    : {days} days                                  ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║  RETURNS                                                    ║\n"
            "║    Initial Capital  : {initial:>10.2f} USDT                     ║\n"
            "║    Final Capital    : {final:>10.2f} USDT                     ║\n"
            "║    Total Return     : {ret:>+10.2f} %                         ║\n"
            "║    Annual Return    : {annual:>+10.2f} %                         ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║  RISK                                                       ║\n"
            "║    Sharpe Ratio     : {sharpe:>10.3f}                           ║\n"
            "║    Sortino Ratio    : {sortino:>10.3f}                           ║\n"
            "║    Calmar Ratio     : {calmar:>10.3f}                           ║\n"
            "║    Max Drawdown     : {dd:>10.2f} %                         ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║  TRADES                                                     ║\n"
            "║    Total Trades     : {total:>10}                           ║\n"
            "║    Win Rate         : {wr:>10.1%}                           ║\n"
            "║    Profit Factor    : {pf:>10}                           ║\n"
            "║    Expectancy       : {exp:>10.4f} USDT                     ║\n"
            "║    Avg Duration     : {dur:>10.1f} h                           ║\n"
            "║    Best Trade       : {best:>+10.2f} %                         ║\n"
            "║    Worst Trade      : {worst:>+10.2f} %                         ║\n"
            "╚═════════════════════════════════════════════════════════════╝",
            strategy=stats["strategy"],
            symbol=stats["symbol"],
            timeframe=stats["timeframe"],
            start=result.start_date.date(),
            end=result.end_date.date(),
            days=stats["period_days"],
            initial=stats["initial_capital"],
            final=stats["final_capital"],
            ret=stats["total_return_pct"],
            annual=stats["annual_return_pct"],
            sharpe=stats["sharpe_ratio"],
            sortino=stats["sortino_ratio"],
            calmar=stats["calmar_ratio"],
            dd=stats["max_drawdown_pct"],
            total=stats["total_trades"],
            wr=stats["win_rate"],
            pf=pf_str,
            exp=stats["expectancy"],
            dur=stats["avg_trade_duration_hours"],
            best=stats["best_trade_pct"],
            worst=stats["worst_trade_pct"],
        )

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def _save_json(self, result: "BacktestResult", stats: dict) -> Path:
        """Save full stats dict to a JSON file and return its path."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_strategy = result.strategy_name.replace(" ", "_").replace("/", "_")
        safe_symbol = result.symbol.replace("/", "_")
        filename = f"{safe_strategy}_{safe_symbol}_{timestamp}.json"
        path = self._reports_dir / filename

        payload = dict(stats)
        # Include trade log (convert non-serialisable datetimes to ISO strings)
        payload["trades"] = [_serialise_trade(t) for t in result.trades]

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            logger.info("Backtest JSON report saved to {}", path)
        except Exception as exc:
            logger.error("Failed to save JSON report: {}", exc)

        return path

    # ------------------------------------------------------------------
    # Equity curve chart
    # ------------------------------------------------------------------

    def _save_equity_chart(self, result: "BacktestResult") -> Optional[Path]:
        """Render and save an equity-curve PNG chart.

        Returns the saved :class:`Path`, or ``None`` if matplotlib is not
        installed.
        """
        try:
            import matplotlib  # noqa: F401

            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            logger.warning(
                "matplotlib not installed — skipping equity chart. "
                "Install with: pip install matplotlib"
            )
            return None

        equity = result.equity_curve
        if len(equity) < 2:
            logger.warning("Equity curve has fewer than 2 points — skipping chart")
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(
            f"{result.strategy_name} | {result.symbol} | {result.timeframe}\n"
            f"{result.start_date.date()} → {result.end_date.date()}",
            fontsize=13,
            fontweight="bold",
        )

        # ── Equity curve ─────────────────────────────────────────────
        ax_eq = axes[0]
        xs = list(range(len(equity)))
        ax_eq.plot(xs, equity, color="#2196F3", linewidth=1.5, label="Portfolio Value")
        ax_eq.axhline(result.initial_capital, color="#9E9E9E", linestyle="--", linewidth=0.8)
        ax_eq.fill_between(xs, result.initial_capital, equity, alpha=0.15, color="#2196F3")
        ax_eq.set_ylabel("Portfolio Value (USDT)")
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax_eq.legend(loc="upper left")
        ax_eq.grid(True, alpha=0.3)

        # ── Drawdown ─────────────────────────────────────────────────
        ax_dd = axes[1]
        drawdowns = _compute_drawdown_series(equity)
        ax_dd.fill_between(xs, drawdowns, color="#F44336", alpha=0.6, label="Drawdown")
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Trade / Update #")
        ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax_dd.legend(loc="lower left")
        ax_dd.grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_strategy = result.strategy_name.replace(" ", "_").replace("/", "_")
        safe_symbol = result.symbol.replace("/", "_")
        filename = f"{safe_strategy}_{safe_symbol}_{timestamp}_equity.png"
        path = self._reports_dir / filename

        try:
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            logger.info("Equity chart saved to {}", path)
        except Exception as exc:
            logger.error("Failed to save equity chart: {}", exc)
            path = None
        finally:
            plt.close(fig)

        return path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_drawdown_series(equity: list) -> list:
    """Return percentage drawdown at each point in the equity curve."""
    drawdowns = []
    peak = equity[0]
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100 if peak > 0 else 0.0
        drawdowns.append(-dd)  # negative so it plots below the axis
    return drawdowns


def _best_trade(trades: list) -> dict:
    """Return the trade with the highest pnl_pct, or empty dict."""
    if not trades:
        return {}
    return max(trades, key=lambda t: t.get("pnl_pct", float("-inf")))


def _worst_trade(trades: list) -> dict:
    """Return the trade with the lowest pnl_pct, or empty dict."""
    if not trades:
        return {}
    return min(trades, key=lambda t: t.get("pnl_pct", float("inf")))


def _serialise_trade(trade: dict) -> dict:
    """Convert datetime objects in *trade* dict to ISO strings for JSON serialisation."""
    out = {}
    for key, value in trade.items():
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out
