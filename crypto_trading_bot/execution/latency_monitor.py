"""Latency monitoring for execution engine - tracks and analyzes execution latency."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class LatencyStats:
    """Statistical measures of latency."""

    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def update_percentiles(self, values: List[float]) -> None:
        """Update percentile values from sorted list."""
        if not values:
            return
        n = len(values)
        self.p50 = values[int(n * 0.50)] if n > 0 else 0.0
        self.p95 = values[int(n * 0.95)] if n > 0 else 0.0
        self.p99 = values[int(n * 0.99)] if n > 0 else 0.0


@dataclass
class ExecutionLatency:
    """Tracks latency for a single execution."""

    execution_id: str
    symbol: str
    signal_timestamp: float = field(default_factory=time.time)
    order_submission_timestamp: Optional[float] = None
    exchange_ack_timestamp: Optional[float] = None
    first_fill_timestamp: Optional[float] = None
    completion_timestamp: Optional[float] = None

    @property
    def internal_latency(self) -> Optional[float]:
        """Signal generation to order submission latency (ms)."""
        if self.order_submission_timestamp:
            return (self.order_submission_timestamp - self.signal_timestamp) * 1000
        return None

    @property
    def network_latency(self) -> Optional[float]:
        """Order submission to exchange acknowledgment latency (ms)."""
        if self.order_submission_timestamp and self.exchange_ack_timestamp:
            return (self.exchange_ack_timestamp - self.order_submission_timestamp) * 1000
        return None

    @property
    def exchange_matching_latency(self) -> Optional[float]:
        """Exchange acknowledgment to first fill latency (ms)."""
        if self.exchange_ack_timestamp and self.first_fill_timestamp:
            return (self.first_fill_timestamp - self.exchange_ack_timestamp) * 1000
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """Total round-trip time from signal to completion (ms)."""
        if self.completion_timestamp:
            return (self.completion_timestamp - self.signal_timestamp) * 1000
        return None


@dataclass
class ConnectionQuality:
    """Tracks connection quality metrics."""

    reconnection_count: int = 0
    last_reconnection_time: Optional[float] = None
    message_delays: deque = field(default_factory=lambda: deque(maxlen=100))
    jitter_events: int = 0
    quality_score: float = 1.0  # 0.0 (poor) to 1.0 (excellent)


class LatencyMonitor:
    """Monitors and tracks execution latency at every stage.

    Features:
    - Stage-by-stage latency tracking (signal → submission → ack → fill)
    - Rolling statistics: p50, p95, p99 latencies
    - Alerts when latency exceeds thresholds
    - Latency-aware execution recommendations
    - Connection quality scoring
    - WebSocket jitter detection
    """

    # Latency thresholds in milliseconds
    CRYPTO_LATENCY_THRESHOLD_MS = 500.0
    FOREX_LATENCY_THRESHOLD_MS = 100.0
    JITTER_THRESHOLD_MS = 50.0  # Variation in message delay

    # Rolling window size
    STATS_WINDOW_SIZE = 1000

    def __init__(self, market_type: str = "crypto"):
        """Initialize latency monitor.

        Args:
            market_type: "crypto" or "forex" for different thresholds
        """
        self._market_type = market_type
        self._threshold_ms = (
            self.FOREX_LATENCY_THRESHOLD_MS
            if market_type == "forex"
            else self.CRYPTO_LATENCY_THRESHOLD_MS
        )

        # Active executions being tracked
        self._active_executions: Dict[str, ExecutionLatency] = {}

        # Historical latency data (rolling window)
        self._internal_latencies: deque = deque(maxlen=self.STATS_WINDOW_SIZE)
        self._network_latencies: deque = deque(maxlen=self.STATS_WINDOW_SIZE)
        self._exchange_latencies: deque = deque(maxlen=self.STATS_WINDOW_SIZE)
        self._total_latencies: deque = deque(maxlen=self.STATS_WINDOW_SIZE)

        # Per-symbol latency tracking
        self._symbol_latencies: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Connection quality tracking
        self._connection_quality: Dict[str, ConnectionQuality] = {}

        # Alert callback
        self._alert_callback: Optional[callable] = None

        logger.info(
            "LatencyMonitor initialized for {} (threshold: {:.0f}ms)",
            market_type,
            self._threshold_ms
        )

    def start_tracking(
        self,
        execution_id: str,
        symbol: str,
        signal_timestamp: Optional[float] = None
    ) -> ExecutionLatency:
        """Start tracking latency for an execution.

        Args:
            execution_id: Unique identifier for this execution
            symbol: Trading symbol
            signal_timestamp: Signal generation timestamp (defaults to now)

        Returns:
            ExecutionLatency object for tracking
        """
        latency = ExecutionLatency(
            execution_id=execution_id,
            symbol=symbol,
            signal_timestamp=signal_timestamp or time.time()
        )
        self._active_executions[execution_id] = latency
        return latency

    def record_order_submission(self, execution_id: str) -> None:
        """Record order submission timestamp."""
        if execution_id in self._active_executions:
            self._active_executions[execution_id].order_submission_timestamp = time.time()

    def record_exchange_ack(self, execution_id: str) -> None:
        """Record exchange acknowledgment timestamp."""
        if execution_id in self._active_executions:
            self._active_executions[execution_id].exchange_ack_timestamp = time.time()

    def record_first_fill(self, execution_id: str) -> None:
        """Record first fill timestamp."""
        if execution_id in self._active_executions:
            self._active_executions[execution_id].first_fill_timestamp = time.time()

    def complete_tracking(self, execution_id: str) -> Optional[ExecutionLatency]:
        """Complete tracking and store statistics.

        Args:
            execution_id: Execution to complete

        Returns:
            Completed ExecutionLatency object with all metrics
        """
        if execution_id not in self._active_executions:
            return None

        latency = self._active_executions.pop(execution_id)
        latency.completion_timestamp = time.time()

        # Store latency metrics
        if latency.internal_latency is not None:
            self._internal_latencies.append(latency.internal_latency)

        if latency.network_latency is not None:
            self._network_latencies.append(latency.network_latency)

        if latency.exchange_matching_latency is not None:
            self._exchange_latencies.append(latency.exchange_matching_latency)

        if latency.total_latency is not None:
            self._total_latencies.append(latency.total_latency)
            self._symbol_latencies[latency.symbol].append(latency.total_latency)

            # Check threshold and alert
            if latency.total_latency > self._threshold_ms:
                self._trigger_alert(latency)

        logger.debug(
            "Execution {} completed: internal={:.1f}ms network={:.1f}ms exchange={:.1f}ms total={:.1f}ms",
            execution_id,
            latency.internal_latency or 0.0,
            latency.network_latency or 0.0,
            latency.exchange_matching_latency or 0.0,
            latency.total_latency or 0.0
        )

        return latency

    def get_stats(self, stat_type: str = "total") -> LatencyStats:
        """Get latency statistics.

        Args:
            stat_type: "internal", "network", "exchange", or "total"

        Returns:
            LatencyStats with current statistics
        """
        data_map = {
            "internal": self._internal_latencies,
            "network": self._network_latencies,
            "exchange": self._exchange_latencies,
            "total": self._total_latencies
        }

        data = data_map.get(stat_type, self._total_latencies)

        if not data:
            return LatencyStats()

        values = sorted(data)
        stats = LatencyStats(
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values)
        )
        stats.update_percentiles(values)

        return stats

    def get_symbol_stats(self, symbol: str) -> LatencyStats:
        """Get latency statistics for a specific symbol."""
        if symbol not in self._symbol_latencies:
            return LatencyStats()

        data = self._symbol_latencies[symbol]
        values = sorted(data)

        stats = LatencyStats(
            count=len(values),
            sum=sum(values),
            min=min(values) if values else 0.0,
            max=max(values) if values else 0.0
        )
        stats.update_percentiles(values)

        return stats

    def should_prefer_market_order(self) -> bool:
        """Determine if high latency suggests preferring market orders.

        Returns:
            True if current latency is high and market orders are recommended
        """
        stats = self.get_stats("total")

        if stats.count < 10:
            return False  # Not enough data

        # If p95 latency exceeds threshold, prefer market orders
        return stats.p95 > self._threshold_ms

    def get_execution_recommendation(self) -> Dict[str, any]:
        """Get latency-aware execution recommendations.

        Returns:
            Dict with execution recommendations based on current latency
        """
        total_stats = self.get_stats("total")
        network_stats = self.get_stats("network")

        if total_stats.count < 10:
            return {
                "confidence": "low",
                "prefer_market_orders": False,
                "reduce_chunk_count": False,
                "message": "Insufficient latency data"
            }

        high_latency = total_stats.p95 > self._threshold_ms
        high_network_latency = network_stats.p95 > (self._threshold_ms * 0.5)

        recommendation = {
            "confidence": "high" if total_stats.count > 50 else "medium",
            "prefer_market_orders": high_latency,
            "reduce_chunk_count": high_latency,
            "avg_latency_ms": total_stats.sum / total_stats.count if total_stats.count > 0 else 0,
            "p95_latency_ms": total_stats.p95,
            "high_latency_detected": high_latency,
            "high_network_latency": high_network_latency
        }

        if high_latency:
            recommendation["message"] = (
                f"High latency detected (p95={total_stats.p95:.1f}ms). "
                "Recommend market orders and reduced TWAP chunks."
            )
        else:
            recommendation["message"] = "Latency is acceptable. Normal execution strategies available."

        return recommendation

    def record_websocket_message(
        self,
        exchange: str,
        message_type: str,
        sent_timestamp: Optional[float] = None
    ) -> None:
        """Record WebSocket message for jitter detection.

        Args:
            exchange: Exchange identifier
            message_type: Type of message (ticker, orderbook, etc.)
            sent_timestamp: When message was sent (if available from exchange)
        """
        received_timestamp = time.time()

        if exchange not in self._connection_quality:
            self._connection_quality[exchange] = ConnectionQuality()

        quality = self._connection_quality[exchange]

        # Calculate delay if sent timestamp available
        if sent_timestamp:
            delay_ms = (received_timestamp - sent_timestamp) * 1000
            quality.message_delays.append(delay_ms)

            # Check for jitter (high variance in delays)
            if len(quality.message_delays) >= 10:
                recent_delays = list(quality.message_delays)[-10:]
                avg_delay = sum(recent_delays) / len(recent_delays)
                max_deviation = max(abs(d - avg_delay) for d in recent_delays)

                if max_deviation > self.JITTER_THRESHOLD_MS:
                    quality.jitter_events += 1
                    logger.warning(
                        "Jitter detected on {} {}: deviation={:.1f}ms",
                        exchange,
                        message_type,
                        max_deviation
                    )

        # Update connection quality score
        self._update_connection_quality(exchange)

    def record_reconnection(self, exchange: str) -> None:
        """Record a WebSocket reconnection event."""
        if exchange not in self._connection_quality:
            self._connection_quality[exchange] = ConnectionQuality()

        quality = self._connection_quality[exchange]
        quality.reconnection_count += 1
        quality.last_reconnection_time = time.time()

        logger.warning(
            "WebSocket reconnection recorded for {} (count: {})",
            exchange,
            quality.reconnection_count
        )

        # Update connection quality score
        self._update_connection_quality(exchange)

    def _update_connection_quality(self, exchange: str) -> None:
        """Update connection quality score for an exchange."""
        if exchange not in self._connection_quality:
            return

        quality = self._connection_quality[exchange]

        # Start with perfect score
        score = 1.0

        # Penalize for reconnections (exponential decay)
        if quality.reconnection_count > 0:
            recent_reconnections = quality.reconnection_count
            if quality.last_reconnection_time:
                # Decay old reconnections (half-life of 1 hour)
                hours_since = (time.time() - quality.last_reconnection_time) / 3600
                recent_reconnections = quality.reconnection_count * (0.5 ** hours_since)

            score *= 0.95 ** recent_reconnections

        # Penalize for jitter
        if quality.jitter_events > 0:
            score *= 0.98 ** min(quality.jitter_events, 10)

        # Penalize for high message delays
        if len(quality.message_delays) >= 10:
            avg_delay = sum(quality.message_delays) / len(quality.message_delays)
            if avg_delay > 100:  # Over 100ms average delay
                score *= 0.9

        quality.quality_score = max(0.0, min(1.0, score))

    def get_connection_quality(self, exchange: str) -> Tuple[float, Dict]:
        """Get connection quality score and details for an exchange.

        Args:
            exchange: Exchange identifier

        Returns:
            Tuple of (quality_score, details_dict)
        """
        if exchange not in self._connection_quality:
            return 1.0, {"status": "no_data"}

        quality = self._connection_quality[exchange]

        avg_delay = 0.0
        if quality.message_delays:
            avg_delay = sum(quality.message_delays) / len(quality.message_delays)

        details = {
            "quality_score": quality.quality_score,
            "reconnection_count": quality.reconnection_count,
            "jitter_events": quality.jitter_events,
            "avg_message_delay_ms": avg_delay,
            "status": self._get_quality_status(quality.quality_score)
        }

        return quality.quality_score, details

    def _get_quality_status(self, score: float) -> str:
        """Get human-readable status from quality score."""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        elif score >= 0.50:
            return "poor"
        else:
            return "unstable"

    def set_alert_callback(self, callback: callable) -> None:
        """Set callback for latency alerts.

        Args:
            callback: Function to call with ExecutionLatency when threshold exceeded
        """
        self._alert_callback = callback

    def _trigger_alert(self, latency: ExecutionLatency) -> None:
        """Trigger alert for high latency execution."""
        logger.warning(
            "HIGH LATENCY ALERT: {} {} - total={:.1f}ms (threshold={:.1f}ms)",
            latency.symbol,
            latency.execution_id,
            latency.total_latency or 0.0,
            self._threshold_ms
        )

        if self._alert_callback:
            try:
                if asyncio.iscoroutinefunction(self._alert_callback):
                    asyncio.create_task(self._alert_callback(latency))
                else:
                    self._alert_callback(latency)
            except Exception as e:
                logger.error("Failed to trigger alert callback: {}", e)

    def get_summary(self) -> Dict:
        """Get comprehensive latency summary.

        Returns:
            Dict with all latency statistics and recommendations
        """
        internal = self.get_stats("internal")
        network = self.get_stats("network")
        exchange = self.get_stats("exchange")
        total = self.get_stats("total")
        recommendation = self.get_execution_recommendation()

        return {
            "market_type": self._market_type,
            "threshold_ms": self._threshold_ms,
            "internal_latency": {
                "count": internal.count,
                "avg_ms": internal.sum / internal.count if internal.count > 0 else 0.0,
                "p50_ms": internal.p50,
                "p95_ms": internal.p95,
                "p99_ms": internal.p99,
                "min_ms": internal.min if internal.min != float('inf') else 0.0,
                "max_ms": internal.max
            },
            "network_latency": {
                "count": network.count,
                "avg_ms": network.sum / network.count if network.count > 0 else 0.0,
                "p50_ms": network.p50,
                "p95_ms": network.p95,
                "p99_ms": network.p99,
                "min_ms": network.min if network.min != float('inf') else 0.0,
                "max_ms": network.max
            },
            "exchange_latency": {
                "count": exchange.count,
                "avg_ms": exchange.sum / exchange.count if exchange.count > 0 else 0.0,
                "p50_ms": exchange.p50,
                "p95_ms": exchange.p95,
                "p99_ms": exchange.p99,
                "min_ms": exchange.min if exchange.min != float('inf') else 0.0,
                "max_ms": exchange.max
            },
            "total_latency": {
                "count": total.count,
                "avg_ms": total.sum / total.count if total.count > 0 else 0.0,
                "p50_ms": total.p50,
                "p95_ms": total.p95,
                "p99_ms": total.p99,
                "min_ms": total.min if total.min != float('inf') else 0.0,
                "max_ms": total.max
            },
            "recommendation": recommendation,
            "active_tracking_count": len(self._active_executions),
            "connection_quality": {
                exchange: self.get_connection_quality(exchange)[1]
                for exchange in self._connection_quality
            }
        }
