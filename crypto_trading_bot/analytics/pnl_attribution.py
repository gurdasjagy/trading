"""PnL Attribution Engine — Institutional-Grade Performance Decomposition.

Decomposes realized PnL into granular attribution buckets:
  - By strategy (which strategies made/lost money)
  - By symbol (which instruments contributed to P&L)
  - By time-of-day (session analysis: Asian/London/NY)
  - By regime (trending vs ranging vs volatile)
  - By execution quality (slippage cost, fee impact)
  - By signal confidence bucket (how well does confidence predict outcomes)

This enables:
  1. Identifying which strategies to allocate more capital to
  2. Understanding time-based edge decay
  3. Measuring execution quality degradation
  4. Optimizing regime-dependent allocation
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class TradeRecord:
    """Single completed trade record for attribution analysis."""
    trade_id: str
    symbol: str
    strategy: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    fees_paid: float
    slippage_cost: float  # Estimated execution cost vs. signal price
    signal_confidence: float
    regime_at_entry: str
    leverage: int
    # Computed fields
    pnl_pct: float = 0.0
    hold_duration_seconds: float = 0.0


@dataclass
class AttributionBucket:
    """Aggregated P&L attribution for a single bucket."""
    label: str
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_volume: float = 0.0
    avg_hold_seconds: float = 0.0
    sharpe_approx: float = 0.0  # Simplified Sharpe using trade-level returns
    max_drawdown_pct: float = 0.0
    _returns: list = field(default_factory=list)

    def record_trade(self, trade: TradeRecord) -> None:
        """Add a trade to this attribution bucket."""
        self.total_pnl += trade.realized_pnl
        self.total_fees += trade.fees_paid
        self.total_slippage += trade.slippage_cost
        self.trade_count += 1
        self.total_volume += abs(trade.quantity * trade.entry_price)

        if trade.realized_pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        # Running average of hold duration
        duration = trade.hold_duration_seconds
        if self.trade_count > 0:
            self.avg_hold_seconds = (
                (self.avg_hold_seconds * (self.trade_count - 1) + duration)
                / self.trade_count
            )

        self._returns.append(trade.pnl_pct)

    def compute_sharpe(self) -> float:
        """Compute approximate Sharpe ratio from trade-level returns."""
        if len(self._returns) < 5:
            return 0.0
        import numpy as np
        returns = np.array(self._returns)
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret < 1e-10:
            return 0.0
        # Annualize assuming ~250 trading days, ~20 trades/day
        trades_per_year = 250 * 20
        self.sharpe_approx = (mean_ret / std_ret) * (trades_per_year ** 0.5)
        return self.sharpe_approx

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(r for r in self._returns if r > 0)
        gross_loss = abs(sum(r for r in self._returns if r < 0))
        if gross_loss < 1e-10:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "total_pnl": round(self.total_pnl, 4),
            "total_fees": round(self.total_fees, 4),
            "total_slippage": round(self.total_slippage, 4),
            "net_pnl": round(self.total_pnl - self.total_fees - self.total_slippage, 4),
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_approx": round(self.compute_sharpe(), 4),
            "avg_hold_seconds": round(self.avg_hold_seconds, 1),
            "total_volume": round(self.total_volume, 2),
        }


def _get_session(dt: datetime) -> str:
    """Classify a UTC datetime into trading session."""
    hour = dt.hour
    if 0 <= hour < 8:
        return "asian"
    elif 8 <= hour < 13:
        return "london"
    elif 13 <= hour < 21:
        return "newyork"
    else:
        return "late_session"


def _get_confidence_bucket(confidence: float) -> str:
    """Classify signal confidence into discrete buckets."""
    if confidence < 0.5:
        return "low_<0.50"
    elif confidence < 0.65:
        return "medium_0.50-0.65"
    elif confidence < 0.80:
        return "high_0.65-0.80"
    else:
        return "very_high_>0.80"


class PnLAttributionEngine:
    """Institutional-grade P&L attribution and decomposition engine.

    Maintains rolling attribution across multiple dimensions:
      - strategy, symbol, session, regime, confidence, execution_quality

    Thread-safe via asyncio lock. All attribution is computed incrementally
    (O(1) per trade) without requiring full trade history replay.
    """

    def __init__(self) -> None:
        # Attribution dimensions
        self._by_strategy: Dict[str, AttributionBucket] = {}
        self._by_symbol: Dict[str, AttributionBucket] = {}
        self._by_session: Dict[str, AttributionBucket] = {}
        self._by_regime: Dict[str, AttributionBucket] = {}
        self._by_confidence: Dict[str, AttributionBucket] = {}

        # Overall aggregates
        self._overall = AttributionBucket(label="overall")

        # Trade history (last 10,000 for detailed analysis)
        self._trade_history: List[TradeRecord] = []
        self._max_history: int = 10_000

        # Equity curve for drawdown computation
        self._equity_curve: List[float] = [0.0]
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0

        # Daily P&L tracking
        self._daily_pnl: Dict[str, float] = defaultdict(float)

        self._lock = asyncio.Lock()

        logger.info("PnLAttributionEngine initialized")

    async def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade and update all attribution dimensions."""
        async with self._lock:
            # Compute derived fields
            if trade.entry_price > 0:
                if trade.direction == "long":
                    trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                else:
                    trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price

            trade.hold_duration_seconds = (
                trade.exit_time - trade.entry_time
            ).total_seconds()

            # Update each attribution dimension
            self._get_or_create(self._by_strategy, trade.strategy).record_trade(trade)
            self._get_or_create(self._by_symbol, trade.symbol).record_trade(trade)

            session = _get_session(trade.entry_time)
            self._get_or_create(self._by_session, session).record_trade(trade)

            self._get_or_create(self._by_regime, trade.regime_at_entry).record_trade(trade)

            conf_bucket = _get_confidence_bucket(trade.signal_confidence)
            self._get_or_create(self._by_confidence, conf_bucket).record_trade(trade)

            self._overall.record_trade(trade)

            # Update equity curve and drawdown
            cum_pnl = self._equity_curve[-1] + trade.realized_pnl
            self._equity_curve.append(cum_pnl)
            if cum_pnl > self._peak_equity:
                self._peak_equity = cum_pnl
            dd = (self._peak_equity - cum_pnl) / max(self._peak_equity, 1.0)
            if dd > self._max_drawdown:
                self._max_drawdown = dd

            # Daily tracking
            day_key = trade.exit_time.strftime("%Y-%m-%d")
            self._daily_pnl[day_key] += trade.realized_pnl

            # History (ring buffer)
            self._trade_history.append(trade)
            if len(self._trade_history) > self._max_history:
                self._trade_history = self._trade_history[-self._max_history:]

    def _get_or_create(
        self, bucket_map: Dict[str, AttributionBucket], key: str
    ) -> AttributionBucket:
        if key not in bucket_map:
            bucket_map[key] = AttributionBucket(label=key)
        return bucket_map[key]

    async def get_full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive attribution report."""
        async with self._lock:
            return {
                "overall": self._overall.to_dict(),
                "max_drawdown_pct": round(self._max_drawdown * 100, 2),
                "by_strategy": {
                    k: v.to_dict() for k, v in sorted(
                        self._by_strategy.items(),
                        key=lambda x: x[1].total_pnl, reverse=True
                    )
                },
                "by_symbol": {
                    k: v.to_dict() for k, v in sorted(
                        self._by_symbol.items(),
                        key=lambda x: x[1].total_pnl, reverse=True
                    )
                },
                "by_session": {
                    k: v.to_dict() for k, v in self._by_session.items()
                },
                "by_regime": {
                    k: v.to_dict() for k, v in self._by_regime.items()
                },
                "by_confidence": {
                    k: v.to_dict() for k, v in sorted(
                        self._by_confidence.items()
                    )
                },
                "daily_pnl": dict(sorted(self._daily_pnl.items())[-30:]),
                "total_trades": self._overall.trade_count,
                "cumulative_pnl": round(self._equity_curve[-1], 4) if self._equity_curve else 0.0,
            }

    async def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by Sharpe ratio."""
        async with self._lock:
            rankings = []
            for name, bucket in self._by_strategy.items():
                data = bucket.to_dict()
                data["strategy"] = name
                rankings.append(data)
            return sorted(rankings, key=lambda x: x.get("sharpe_approx", 0), reverse=True)

    async def get_execution_quality_report(self) -> Dict[str, Any]:
        """Analyze execution quality across all trades."""
        async with self._lock:
            if not self._trade_history:
                return {"status": "no_data"}

            total_slippage = sum(t.slippage_cost for t in self._trade_history)
            total_fees = sum(t.fees_paid for t in self._trade_history)
            total_pnl = sum(t.realized_pnl for t in self._trade_history)
            maker_count = sum(
                1 for t in self._trade_history if t.slippage_cost <= 0
            )

            return {
                "total_slippage_cost": round(total_slippage, 4),
                "total_fees": round(total_fees, 4),
                "execution_drag_pct": round(
                    (total_slippage + total_fees) / max(abs(total_pnl), 1) * 100, 2
                ),
                "maker_ratio": round(maker_count / max(len(self._trade_history), 1), 4),
                "avg_slippage_per_trade": round(
                    total_slippage / max(len(self._trade_history), 1), 6
                ),
                "avg_fee_per_trade": round(
                    total_fees / max(len(self._trade_history), 1), 6
                ),
            }
