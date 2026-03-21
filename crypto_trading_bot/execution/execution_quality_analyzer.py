"""Execution quality analyzer - post-trade analysis and quality metrics."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger
import pandas as pd


@dataclass
class ExecutionReport:
    """Report for a single execution."""

    execution_id: str
    symbol: str
    side: str
    strategy: str
    timestamp: float

    # Order details
    order_amount: float
    filled_amount: float
    average_fill_price: float

    # Benchmark prices
    decision_price: float  # Mid price at signal generation
    mid_price_at_submission: float
    vwap_benchmark: Optional[float] = None
    twap_benchmark: Optional[float] = None

    # Quality metrics
    slippage_bps: float = 0.0  # Basis points
    implementation_shortfall_bps: float = 0.0
    market_impact_bps: float = 0.0
    fill_rate: float = 1.0  # Percentage filled
    timing_cost_bps: float = 0.0

    # Post-execution price movements
    price_5min_after: Optional[float] = None
    price_15min_after: Optional[float] = None
    price_30min_after: Optional[float] = None

    # Fees
    total_fees: float = 0.0

    @property
    def total_cost_bps(self) -> float:
        """Total cost of execution in basis points."""
        return self.slippage_bps + self.implementation_shortfall_bps + self.market_impact_bps

    @property
    def execution_quality_score(self) -> float:
        """Quality score from 0 (poor) to 100 (excellent)."""
        # Start with 100 and deduct for costs
        score = 100.0

        # Deduct for slippage (1 bps = -2 points)
        score -= abs(self.slippage_bps) * 2.0

        # Deduct for implementation shortfall (1 bps = -1.5 points)
        score -= abs(self.implementation_shortfall_bps) * 1.5

        # Deduct for poor fill rate
        score -= (1.0 - self.fill_rate) * 50.0

        # Bonus for beating VWAP
        if self.vwap_benchmark and self.side == "buy":
            if self.average_fill_price < self.vwap_benchmark:
                score += 5.0
        elif self.vwap_benchmark and self.side == "sell":
            if self.average_fill_price > self.vwap_benchmark:
                score += 5.0

        return max(0.0, min(100.0, score))


@dataclass
class StrategyQualityStats:
    """Quality statistics for a strategy."""

    strategy_name: str
    execution_count: int = 0
    avg_slippage_bps: float = 0.0
    avg_implementation_shortfall_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    avg_fill_rate: float = 1.0
    avg_quality_score: float = 0.0
    total_fees: float = 0.0


@dataclass
class ExchangeQualityStats:
    """Quality statistics for an exchange."""

    exchange_name: str
    execution_count: int = 0
    avg_slippage_bps: float = 0.0
    avg_fill_rate: float = 1.0
    avg_quality_score: float = 0.0
    reliability_score: float = 1.0
    total_fees: float = 0.0


@dataclass
class TimeOfDayStats:
    """Quality statistics by time of day."""

    hour: int
    execution_count: int = 0
    avg_slippage_bps: float = 0.0
    avg_quality_score: float = 0.0


class ExecutionQualityAnalyzer:
    """Analyzes execution quality with post-trade metrics.

    Features:
    - Slippage analysis vs mid-price at signal time
    - Implementation shortfall calculation
    - Market impact measurement
    - Fill rate tracking
    - Timing analysis vs VWAP/TWAP benchmarks
    - Per-strategy, per-exchange, per-time quality reports
    - Trend analysis and quality degradation detection
    - Feedback loop integration with routing and RL optimizer
    """

    # Rolling window size for reports
    REPORT_WINDOW_SIZE = 500

    def __init__(self):
        """Initialize execution quality analyzer."""
        # Store execution reports
        self._reports: deque = deque(maxlen=self.REPORT_WINDOW_SIZE)

        # Per-strategy statistics
        self._strategy_stats: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Per-exchange statistics
        self._exchange_stats: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Per-symbol statistics
        self._symbol_stats: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Time-of-day statistics (by hour)
        self._hourly_stats: Dict[int, List[ExecutionReport]] = defaultdict(list)

        # Trend tracking (for degradation detection)
        self._recent_scores: deque = deque(maxlen=50)

        logger.info("ExecutionQualityAnalyzer initialized")

    async def analyze_execution(
        self,
        execution_id: str,
        symbol: str,
        side: str,
        strategy: str,
        order_amount: float,
        filled_amount: float,
        average_fill_price: float,
        decision_price: float,
        mid_price_at_submission: float,
        total_fees: float = 0.0,
        vwap_benchmark: Optional[float] = None,
        twap_benchmark: Optional[float] = None
    ) -> ExecutionReport:
        """Analyze a completed execution.

        Args:
            execution_id: Unique execution identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            strategy: Strategy name
            order_amount: Total order amount
            filled_amount: Actually filled amount
            average_fill_price: Average fill price
            decision_price: Mid price at signal generation
            mid_price_at_submission: Mid price at order submission
            total_fees: Total trading fees
            vwap_benchmark: VWAP price for comparison (optional)
            twap_benchmark: TWAP price for comparison (optional)

        Returns:
            ExecutionReport with quality metrics
        """
        # Calculate slippage (vs mid price at signal)
        if side == "buy":
            slippage_bps = ((average_fill_price - decision_price) / decision_price) * 10000
        else:
            slippage_bps = ((decision_price - average_fill_price) / decision_price) * 10000

        # Calculate implementation shortfall (vs theoretical cost at decision price)
        theoretical_cost = decision_price * filled_amount
        actual_cost = average_fill_price * filled_amount
        if side == "buy":
            is_cost = actual_cost - theoretical_cost
        else:
            is_cost = theoretical_cost - actual_cost
        implementation_shortfall_bps = (is_cost / theoretical_cost) * 10000 if theoretical_cost > 0 else 0.0

        # Calculate market impact (difference between decision and submission)
        if side == "buy":
            market_impact_bps = ((mid_price_at_submission - decision_price) / decision_price) * 10000
        else:
            market_impact_bps = ((decision_price - mid_price_at_submission) / decision_price) * 10000

        # Calculate fill rate
        fill_rate = filled_amount / order_amount if order_amount > 0 else 1.0

        # Calculate timing cost (vs submission price)
        if side == "buy":
            timing_cost_bps = ((average_fill_price - mid_price_at_submission) / mid_price_at_submission) * 10000
        else:
            timing_cost_bps = ((mid_price_at_submission - average_fill_price) / mid_price_at_submission) * 10000

        report = ExecutionReport(
            execution_id=execution_id,
            symbol=symbol,
            side=side,
            strategy=strategy,
            timestamp=time.time(),
            order_amount=order_amount,
            filled_amount=filled_amount,
            average_fill_price=average_fill_price,
            decision_price=decision_price,
            mid_price_at_submission=mid_price_at_submission,
            vwap_benchmark=vwap_benchmark,
            twap_benchmark=twap_benchmark,
            slippage_bps=slippage_bps,
            implementation_shortfall_bps=implementation_shortfall_bps,
            market_impact_bps=market_impact_bps,
            fill_rate=fill_rate,
            timing_cost_bps=timing_cost_bps,
            total_fees=total_fees
        )

        # Store report
        self._reports.append(report)
        self._strategy_stats[strategy].append(report)
        self._symbol_stats[symbol].append(report)
        self._recent_scores.append(report.execution_quality_score)

        # Store by hour for time-of-day analysis
        from datetime import datetime
        hour = datetime.fromtimestamp(report.timestamp).hour
        self._hourly_stats[hour].append(report)

        logger.info(
            "Execution quality: {} {} - slippage={:.1f}bps IS={:.1f}bps impact={:.1f}bps fill={:.1%} score={:.1f}",
            symbol,
            execution_id,
            slippage_bps,
            implementation_shortfall_bps,
            market_impact_bps,
            fill_rate,
            report.execution_quality_score
        )

        return report

    async def record_post_execution_prices(
        self,
        execution_id: str,
        price_5min: Optional[float] = None,
        price_15min: Optional[float] = None,
        price_30min: Optional[float] = None
    ) -> None:
        """Record post-execution price movements for timing analysis.

        Args:
            execution_id: Execution to update
            price_5min: Price 5 minutes after execution
            price_15min: Price 15 minutes after execution
            price_30min: Price 30 minutes after execution
        """
        # Find the report in recent reports
        for report in reversed(self._reports):
            if report.execution_id == execution_id:
                if price_5min is not None:
                    report.price_5min_after = price_5min
                if price_15min is not None:
                    report.price_15min_after = price_15min
                if price_30min is not None:
                    report.price_30min_after = price_30min
                break

    def get_strategy_quality(self, strategy: str) -> StrategyQualityStats:
        """Get quality statistics for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            StrategyQualityStats with aggregated metrics
        """
        if strategy not in self._strategy_stats:
            return StrategyQualityStats(strategy_name=strategy)

        reports = list(self._strategy_stats[strategy])

        if not reports:
            return StrategyQualityStats(strategy_name=strategy)

        stats = StrategyQualityStats(
            strategy_name=strategy,
            execution_count=len(reports),
            avg_slippage_bps=sum(r.slippage_bps for r in reports) / len(reports),
            avg_implementation_shortfall_bps=sum(r.implementation_shortfall_bps for r in reports) / len(reports),
            avg_market_impact_bps=sum(r.market_impact_bps for r in reports) / len(reports),
            avg_fill_rate=sum(r.fill_rate for r in reports) / len(reports),
            avg_quality_score=sum(r.execution_quality_score for r in reports) / len(reports),
            total_fees=sum(r.total_fees for r in reports)
        )

        return stats

    def get_exchange_quality(self, exchange: str, reports: List[ExecutionReport]) -> ExchangeQualityStats:
        """Get quality statistics for an exchange.

        Args:
            exchange: Exchange name
            reports: List of execution reports for this exchange

        Returns:
            ExchangeQualityStats with aggregated metrics
        """
        if not reports:
            return ExchangeQualityStats(exchange_name=exchange)

        # Calculate reliability score based on fill rates
        fill_rates = [r.fill_rate for r in reports]
        avg_fill_rate = sum(fill_rates) / len(fill_rates)
        reliability_score = avg_fill_rate  # Simple: reliability = average fill rate

        stats = ExchangeQualityStats(
            exchange_name=exchange,
            execution_count=len(reports),
            avg_slippage_bps=sum(r.slippage_bps for r in reports) / len(reports),
            avg_fill_rate=avg_fill_rate,
            avg_quality_score=sum(r.execution_quality_score for r in reports) / len(reports),
            reliability_score=reliability_score,
            total_fees=sum(r.total_fees for r in reports)
        )

        return stats

    def get_time_of_day_quality(self) -> List[TimeOfDayStats]:
        """Get quality statistics by time of day (hourly).

        Returns:
            List of TimeOfDayStats for each hour (0-23)
        """
        stats_list = []

        for hour in range(24):
            if hour not in self._hourly_stats or not self._hourly_stats[hour]:
                stats_list.append(TimeOfDayStats(hour=hour))
                continue

            reports = self._hourly_stats[hour]
            stats = TimeOfDayStats(
                hour=hour,
                execution_count=len(reports),
                avg_slippage_bps=sum(r.slippage_bps for r in reports) / len(reports),
                avg_quality_score=sum(r.execution_quality_score for r in reports) / len(reports)
            )
            stats_list.append(stats)

        return stats_list

    def detect_quality_degradation(self, window_size: int = 20) -> Tuple[bool, Dict]:
        """Detect if execution quality is degrading over time.

        Args:
            window_size: Number of recent executions to analyze

        Returns:
            Tuple of (is_degrading, details_dict)
        """
        if len(self._recent_scores) < window_size * 2:
            return False, {"reason": "insufficient_data", "sample_count": len(self._recent_scores)}

        recent_scores = list(self._recent_scores)

        # Compare recent window to previous window
        recent_window = recent_scores[-window_size:]
        previous_window = recent_scores[-window_size * 2:-window_size]

        recent_avg = sum(recent_window) / len(recent_window)
        previous_avg = sum(previous_window) / len(previous_window)

        # Degradation if recent average is significantly lower
        degradation_threshold = 5.0  # 5 point drop in score
        is_degrading = (previous_avg - recent_avg) > degradation_threshold

        details = {
            "recent_avg_score": recent_avg,
            "previous_avg_score": previous_avg,
            "score_change": recent_avg - previous_avg,
            "is_degrading": is_degrading,
            "window_size": window_size
        }

        if is_degrading:
            logger.warning(
                "Execution quality degradation detected: recent={:.1f} previous={:.1f} (drop={:.1f})",
                recent_avg,
                previous_avg,
                previous_avg - recent_avg
            )

        return is_degrading, details

    def get_symbol_quality(self, symbol: str) -> Dict:
        """Get quality statistics for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with symbol quality metrics
        """
        if symbol not in self._symbol_stats or not self._symbol_stats[symbol]:
            return {
                "symbol": symbol,
                "execution_count": 0,
                "avg_slippage_bps": 0.0,
                "avg_quality_score": 0.0
            }

        reports = list(self._symbol_stats[symbol])

        return {
            "symbol": symbol,
            "execution_count": len(reports),
            "avg_slippage_bps": sum(r.slippage_bps for r in reports) / len(reports),
            "avg_implementation_shortfall_bps": sum(r.implementation_shortfall_bps for r in reports) / len(reports),
            "avg_fill_rate": sum(r.fill_rate for r in reports) / len(reports),
            "avg_quality_score": sum(r.execution_quality_score for r in reports) / len(reports),
            "total_fees": sum(r.total_fees for r in reports)
        }

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive execution quality report.

        Returns:
            Dict with all quality statistics and trends
        """
        if not self._reports:
            return {"status": "no_data", "message": "No executions recorded yet"}

        all_reports = list(self._reports)

        # Overall statistics
        overall_stats = {
            "total_executions": len(all_reports),
            "avg_slippage_bps": sum(r.slippage_bps for r in all_reports) / len(all_reports),
            "avg_implementation_shortfall_bps": sum(r.implementation_shortfall_bps for r in all_reports) / len(all_reports),
            "avg_market_impact_bps": sum(r.market_impact_bps for r in all_reports) / len(all_reports),
            "avg_fill_rate": sum(r.fill_rate for r in all_reports) / len(all_reports),
            "avg_quality_score": sum(r.execution_quality_score for r in all_reports) / len(all_reports),
            "total_fees": sum(r.total_fees for r in all_reports)
        }

        # Per-strategy statistics
        strategy_stats = {
            strategy: self.get_strategy_quality(strategy).__dict__
            for strategy in self._strategy_stats
        }

        # Time-of-day analysis
        time_of_day = [stats.__dict__ for stats in self.get_time_of_day_quality()]

        # Quality degradation check
        is_degrading, degradation_details = self.detect_quality_degradation()

        # Symbol statistics
        symbol_stats = {
            symbol: self.get_symbol_quality(symbol)
            for symbol in self._symbol_stats
        }

        return {
            "status": "success",
            "generated_at": time.time(),
            "overall": overall_stats,
            "by_strategy": strategy_stats,
            "by_symbol": symbol_stats,
            "by_time_of_day": time_of_day,
            "quality_trend": degradation_details,
            "recent_score_history": list(self._recent_scores)[-20:] if len(self._recent_scores) > 0 else []
        }

    def get_routing_feedback(self, exchange_reports: Dict[str, List[ExecutionReport]]) -> Dict[str, Dict]:
        """Generate feedback for SmartOrderRouter venue scoring.

        Args:
            exchange_reports: Dict mapping exchange name to list of execution reports

        Returns:
            Dict mapping exchange name to feedback metrics for routing
        """
        feedback = {}

        for exchange, reports in exchange_reports.items():
            if not reports:
                continue

            stats = self.get_exchange_quality(exchange, reports)

            # Convert quality metrics to routing feedback
            feedback[exchange] = {
                "reliability_score": stats.reliability_score,
                "historical_fill_rate": stats.avg_fill_rate,
                "avg_slippage_bps": stats.avg_slippage_bps,
                "execution_count": stats.execution_count,
                "avg_quality_score": stats.avg_quality_score,
                "recommendation": "prefer" if stats.avg_quality_score > 75 else (
                    "normal" if stats.avg_quality_score > 50 else "avoid"
                )
            }

        return feedback

    def get_rl_feedback(self) -> Dict:
        """Generate feedback for RL strategy optimizer.

        Returns:
            Dict with execution quality signals for reward calculation
        """
        if len(self._recent_scores) < 5:
            return {"execution_quality_signal": 0.0, "confidence": "low"}

        # Recent average score (0-100)
        recent_avg = sum(list(self._recent_scores)[-10:]) / min(10, len(self._recent_scores))

        # Normalize to -1 to +1 for RL reward
        # 50 is neutral, >75 is good (+1), <25 is bad (-1)
        signal = (recent_avg - 50.0) / 50.0  # Maps 0-100 to -1 to +1

        return {
            "execution_quality_signal": signal,
            "recent_avg_score": recent_avg,
            "confidence": "high" if len(self._recent_scores) >= 20 else "medium"
        }

    def get_risk_feedback(self) -> Dict:
        """Generate feedback for risk manager position sizing.

        Returns:
            Dict with expected slippage and cost metrics
        """
        if not self._reports:
            return {"expected_slippage_bps": 5.0, "confidence": "low"}

        recent_reports = list(self._reports)[-50:]

        avg_slippage = sum(abs(r.slippage_bps) for r in recent_reports) / len(recent_reports)
        avg_total_cost = sum(r.total_cost_bps for r in recent_reports) / len(recent_reports)

        # Add safety margin (90th percentile)
        slippages = sorted(abs(r.slippage_bps) for r in recent_reports)
        p90_slippage = slippages[int(len(slippages) * 0.9)] if slippages else avg_slippage

        return {
            "expected_slippage_bps": avg_slippage,
            "p90_slippage_bps": p90_slippage,
            "avg_total_cost_bps": avg_total_cost,
            "confidence": "high" if len(recent_reports) >= 30 else "medium"
        }
