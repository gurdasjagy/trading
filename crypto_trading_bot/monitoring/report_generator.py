"""Report generator — daily, weekly, and monthly trading reports."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from loguru import logger

from .alerting import AlertManager, AlertType
from .performance_tracker import PerformanceTracker
from .trade_journal import TradeJournal


class ReportGenerator:
    """Generates periodic trading performance reports."""

    def __init__(
        self,
        journal: Optional[TradeJournal] = None,
        performance: Optional[PerformanceTracker] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self._journal = journal or TradeJournal()
        self._performance = performance or PerformanceTracker()
        self._alerts = alert_manager or AlertManager()

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_daily_report(self, report_date: Optional[date] = None) -> dict:
        """Generate a report for a single trading day.

        Args:
            report_date: Date to report on. Defaults to today (UTC).

        Returns:
            Report dict with performance metrics and trade list.
        """
        report_date = report_date or datetime.now(tz=timezone.utc).date()
        date_str = str(report_date)
        trades = self._journal.get_trade_history(start_date=date_str, end_date=date_str)
        return self._build_report(trades, label="daily", period=date_str)

    def generate_weekly_report(self, week_start: Optional[date] = None) -> dict:
        """Generate a report for an ISO calendar week.

        Args:
            week_start: Monday of the target week. Defaults to this week.

        Returns:
            Report dict covering all trades in the week.
        """
        if week_start is None:
            today = datetime.now(tz=timezone.utc).date()
            week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        trades = self._journal.get_trade_history(
            start_date=str(week_start),
            end_date=str(week_end),
        )
        return self._build_report(
            trades,
            label="weekly",
            period=f"{week_start} to {week_end}",
        )

    def generate_monthly_report(self, year: int, month: int) -> dict:
        """Generate a report for a calendar month.

        Args:
            year: Four-digit year.
            month: Month number (1–12).

        Returns:
            Report dict covering all trades in the month.
        """
        _, last_day = calendar.monthrange(year, month)
        start = date(year, month, 1)
        end = date(year, month, last_day)
        trades = self._journal.get_trade_history(
            start_date=str(start),
            end_date=str(end),
        )
        return self._build_report(
            trades,
            label="monthly",
            period=f"{year}-{month:02d}",
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_report_html(self, report: dict) -> str:
        """Format a report dict as a simple HTML document.

        Args:
            report: Report dict as returned by the generate_* methods.

        Returns:
            HTML string.
        """
        label = report.get("label", "Report").title()
        period = report.get("period", "")
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report.get("metrics", {}).items()
        )
        trade_rows = "".join(
            f"<tr><td>{t.get('symbol')}</td><td>{t.get('direction')}</td>"
            f"<td>{t.get('pnl', 0):.4f}</td></tr>"
            for t in report.get("trades", [])
        )
        return (
            f"<html><body>"
            f"<h1>{label} Report — {period}</h1>"
            f"<h2>Metrics</h2><table>{rows}</table>"
            f"<h2>Trades</h2>"
            f"<table><tr><th>Symbol</th><th>Dir</th><th>PnL</th></tr>{trade_rows}</table>"
            f"</body></html>"
        )

    def format_report_text(self, report: dict) -> str:
        """Format a report dict as plain text.

        Args:
            report: Report dict as returned by the generate_* methods.

        Returns:
            Plain-text string.
        """
        label = report.get("label", "Report").upper()
        period = report.get("period", "")
        lines = [f"=== {label} TRADING REPORT — {period} ===", ""]
        for k, v in report.get("metrics", {}).items():
            lines.append(f"  {k}: {v}")
        lines += ["", "--- Trades ---"]
        for t in report.get("trades", []):
            pnl = t.get("pnl", 0) or 0
            lines.append(f"  {t.get('symbol')} {t.get('direction')} PnL={pnl:+.4f}")
        return "\n".join(lines)

    async def send_report(
        self,
        report: dict,
        channels: List[str] | None = None,
    ) -> None:
        """Send a formatted report through alert channels.

        Args:
            report: Report dict to send.
            channels: List of channel names (``"telegram"``, ``"email"``).
                Defaults to both.
        """
        channels = channels or ["telegram", "email"]
        label = report.get("label", "")
        alert_type = AlertType.DAILY_SUMMARY

        data = {
            "date": report.get("period", ""),
            "total_pnl": report.get("metrics", {}).get("total_pnl", 0),
            "win_rate": report.get("metrics", {}).get("win_rate", 0),
            "trade_count": report.get("trade_count", 0),
        }
        try:
            await self._alerts.send_typed_alert(alert_type, data)
        except Exception as exc:
            logger.error("Failed to send {} report: {}", label, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_report(self, trades: list, label: str, period: str) -> dict:
        """Build a structured report from a list of trades.

        Args:
            trades: List of trade records from the journal.
            label: Period label (``"daily"``, ``"weekly"``, ``"monthly"``).
            period: Human-readable period string.

        Returns:
            Complete report dict.
        """
        pnls = [t.get("pnl", 0.0) or 0.0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0

        metrics = {
            "total_pnl": round(total_pnl, 4),
            "trade_count": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(sum(wins) / len(wins), 4) if wins else 0.0,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
            "largest_win": round(max(wins), 4) if wins else 0.0,
            "largest_loss": round(min(losses), 4) if losses else 0.0,
        }
        logger.info(
            "{} report built for {}: total_pnl={:.4f} trades={}",
            label,
            period,
            total_pnl,
            len(trades),
        )
        return {
            "label": label,
            "period": period,
            "trade_count": len(trades),
            "metrics": metrics,
            "trades": trades,
        }
