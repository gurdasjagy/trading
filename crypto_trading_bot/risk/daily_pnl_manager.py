"""Daily PnL manager — tracks intraday performance and enforces profit/loss limits."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from config.settings import Settings


class DailyPnLManager:
    """Manages daily profit targets and loss limits with adaptive sizing and compounding."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings.get_settings()
        self._lock = asyncio.Lock()

        # PnL records: date_str → list of {"trade_id": str, "amount": float, "ts": datetime}
        self._records: Dict[str, List[dict]] = defaultdict(list)

        # Equity at start of each trading day for compounding
        self._starting_equity: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_pnl(self, amount: float, trade_id: str) -> None:
        """Record a realised PnL entry for the current trading day.

        Args:
            amount: PnL amount (positive = profit, negative = loss).
            trade_id: Identifier for the trade.
        """
        today = self._today()
        async with self._lock:
            self._records[today].append(
                {
                    "trade_id": trade_id,
                    "amount": amount,
                    "ts": datetime.now(tz=timezone.utc),
                }
            )
        logger.info("PnL recorded: trade_id={} amount={:.4f} date={}", trade_id, amount, today)

    async def check_daily_status(self) -> dict:
        """Return the current daily PnL status.

        Returns:
            Dict with keys: ``date``, ``total_pnl``, ``trade_count``,
            ``daily_pnl_pct``, ``limit_reached``, ``target_reached``.
        """
        today = self._today()
        async with self._lock:
            records = list(self._records.get(today, []))

        total_pnl = sum(r["amount"] for r in records)
        starting = self._starting_equity.get(today, 1.0) or 1.0
        pnl_pct = total_pnl / starting * 100.0

        risk = self._settings.risk
        limit_reached = pnl_pct <= -risk.max_daily_loss_pct
        target_reached = pnl_pct >= risk.daily_profit_target_pct

        status = {
            "date": today,
            "total_pnl": total_pnl,
            "trade_count": len(records),
            "daily_pnl_pct": pnl_pct,
            "limit_reached": limit_reached,
            "target_reached": target_reached,
            "max_loss_pct": risk.max_daily_loss_pct,
            "profit_target_pct": risk.daily_profit_target_pct,
        }
        logger.debug("Daily status: {}", status)
        return status

    async def get_adaptive_target(self) -> float:
        """Return an adaptive daily profit target based on recent performance.

        If the last 5 trading days averaged above-target performance, the target
        is raised by 20 %.  If below average, it is lowered by 20 %.

        Returns:
            Adaptive daily profit target percentage.
        """
        base_target = self._settings.risk.daily_profit_target_pct
        recent_avg = await self._recent_daily_avg(days=5)
        if recent_avg > base_target * 1.2:
            adaptive = base_target * 1.2
        elif recent_avg < base_target * 0.5:
            adaptive = base_target * 0.8
        else:
            adaptive = base_target
        logger.debug(
            "Adaptive target: base={:.2f}% recent_avg={:.2f}% adaptive={:.2f}%",
            base_target,
            recent_avg,
            adaptive,
        )
        return adaptive

    async def should_stop_trading(self) -> bool:
        """Return True if trading should halt for the rest of the day.

        Trading is halted when the daily loss limit has been reached or the
        profit target has been met.

        Returns:
            ``True`` if trading should stop.
        """
        status = await self.check_daily_status()
        halt = status["limit_reached"] or status["target_reached"]
        if halt:
            logger.warning(
                "Trading halt: limit_reached={} target_reached={}",
                status["limit_reached"],
                status["target_reached"],
            )
        return halt

    async def get_daily_summary(self) -> dict:
        """Return a summary of today's trading activity."""
        return await self.check_daily_status()

    async def get_weekly_summary(self) -> dict:
        """Return a summary of the current ISO calendar week's PnL."""
        today = datetime.now(tz=timezone.utc).date()
        week_dates = [
            str(today.fromisocalendar(today.isocalendar().year, today.isocalendar().week, d))
            for d in range(1, 8)
        ]
        return await self._aggregate_summary(week_dates, label="weekly")

    async def get_monthly_summary(self) -> dict:
        """Return a summary of the current calendar month's PnL."""
        today = datetime.now(tz=timezone.utc).date()
        monthly_dates = [str(date(today.year, today.month, d)) for d in range(1, today.day + 1)]
        return await self._aggregate_summary(monthly_dates, label="monthly")

    def set_starting_equity(self, equity: float) -> None:
        """Record today's starting equity for percentage PnL calculations.

        Args:
            equity: Portfolio equity at the start of the trading day.
        """
        today = self._today()
        self._starting_equity[today] = equity
        logger.info("Starting equity set: {} = {:.2f}", today, equity)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _today(self) -> str:
        return str(datetime.now(tz=timezone.utc).date())

    async def _recent_daily_avg(self, days: int = 5) -> float:
        """Return average daily PnL percentage over the last *days* trading days."""
        async with self._lock:
            all_dates = sorted(self._records.keys(), reverse=True)
        recent = all_dates[:days]
        if not recent:
            return 0.0
        totals = []
        for day in recent:
            async with self._lock:
                recs = list(self._records.get(day, []))
            total = sum(r["amount"] for r in recs)
            starting = self._starting_equity.get(day, 1.0) or 1.0
            totals.append(total / starting * 100.0)
        return sum(totals) / len(totals) if totals else 0.0

    async def _aggregate_summary(self, dates: List[str], label: str) -> dict:
        """Aggregate PnL records across a list of date strings."""
        async with self._lock:
            records = [r for d in dates for r in self._records.get(d, [])]
        total_pnl = sum(r["amount"] for r in records)
        winning = [r for r in records if r["amount"] > 0]
        losing = [r for r in records if r["amount"] < 0]
        return {
            "label": label,
            "dates": dates,
            "total_pnl": total_pnl,
            "trade_count": len(records),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(records) * 100.0 if records else 0.0,
            "avg_win": sum(r["amount"] for r in winning) / len(winning) if winning else 0.0,
            "avg_loss": sum(r["amount"] for r in losing) / len(losing) if losing else 0.0,
        }
