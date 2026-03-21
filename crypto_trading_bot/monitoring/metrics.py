"""Prometheus-compatible metrics collector for the trading bot."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

from loguru import logger

# Standard Prometheus histogram buckets for trade execution latency (seconds)
_TRADE_LATENCY_BUCKETS: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# ---------------------------------------------------------------------------
# Optional prometheus_client integration
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (  # type: ignore[import]
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed — using built-in text export only")


def _make_registry() -> Optional[object]:
    """Return a fresh CollectorRegistry if prometheus_client is available."""
    if _PROMETHEUS_AVAILABLE:
        from prometheus_client import CollectorRegistry
        return CollectorRegistry()
    return None


class MetricsCollector:
    """Collects and exports trading metrics in Prometheus text format.

    When ``prometheus_client`` is installed (it is listed in requirements.txt)
    this class creates proper labeled Prometheus metrics:

    - ``trades_total``           Counter  (labels: symbol, direction, strategy, result)
    - ``trade_pnl``              Histogram (labels: symbol, strategy)
    - ``order_latency_seconds``  Histogram (labels: exchange, order_type)
    - ``open_positions_count``   Gauge
    - ``portfolio_equity``       Gauge
    - ``daily_pnl_pct``          Gauge
    - ``circuit_breaker_active`` Gauge
    - ``exchange_errors_total``  Counter  (labels: error_type)
    - ``funding_costs_total``    Counter  (labels: symbol)

    When ``prometheus_client`` is **not** available it falls back to a
    hand-rolled Prometheus text exposition (no labels on counters/histograms).
    """

    def __init__(self) -> None:
        # -------------------------------------------------------------------
        # Legacy / fallback counters (always maintained for backward compat)
        # -------------------------------------------------------------------
        self._active_positions: int = 0
        self._total_trades: int = 0
        self._winning_trades: int = 0
        self._daily_pnl: float = 0.0
        self._daily_pnl_pct: float = 0.0
        self._portfolio_value: float = 0.0
        self._signal_count: int = 0
        self._error_count: int = 0
        self._circuit_breaker_active: bool = False

        # Latency tracking: exchange → list of latencies (seconds)
        self._api_latencies: Dict[str, List[float]] = defaultdict(list)
        self._latency_window: int = 100

        # Trade execution latency histogram (signal creation → fill confirmation)
        self._trade_latencies: List[float] = []
        self._trade_latency_sum: float = 0.0
        self._trade_latency_count: int = 0

        # -------------------------------------------------------------------
        # prometheus_client labeled metrics
        # -------------------------------------------------------------------
        self._registry = _make_registry()
        self._prom_trades_total: Optional[object] = None
        self._prom_trade_pnl: Optional[object] = None
        self._prom_order_latency: Optional[object] = None
        self._prom_open_positions: Optional[object] = None
        self._prom_portfolio_equity: Optional[object] = None
        self._prom_daily_pnl_pct: Optional[object] = None
        self._prom_circuit_breaker: Optional[object] = None
        self._prom_exchange_errors: Optional[object] = None
        self._prom_funding_costs: Optional[object] = None

        if _PROMETHEUS_AVAILABLE and self._registry is not None:
            self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> None:
        """Initialise all prometheus_client metric objects."""
        reg = self._registry
        self._prom_trades_total = Counter(
            "trades_total",
            "Total number of completed trades",
            ["symbol", "direction", "strategy", "result"],
            registry=reg,
        )
        self._prom_trade_pnl = Histogram(
            "trade_pnl",
            "Realised P&L per trade in USDT",
            ["symbol", "strategy"],
            buckets=[-500, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500],
            registry=reg,
        )
        self._prom_order_latency = Histogram(
            "order_latency_seconds",
            "Time from order placement to fill confirmation (seconds)",
            ["exchange", "order_type"],
            buckets=_TRADE_LATENCY_BUCKETS,
            registry=reg,
        )
        self._prom_open_positions = Gauge(
            "open_positions_count",
            "Number of currently open positions",
            registry=reg,
        )
        self._prom_portfolio_equity = Gauge(
            "portfolio_equity",
            "Total portfolio equity in USDT",
            registry=reg,
        )
        self._prom_daily_pnl_pct = Gauge(
            "daily_pnl_pct",
            "Today's realised P&L as a percentage of starting equity",
            registry=reg,
        )
        self._prom_circuit_breaker = Gauge(
            "circuit_breaker_active",
            "1 when the circuit breaker is active, 0 otherwise",
            registry=reg,
        )
        self._prom_exchange_errors = Counter(
            "exchange_errors_total",
            "Total number of exchange errors by type",
            ["error_type"],
            registry=reg,
        )
        self._prom_funding_costs = Counter(
            "funding_costs_total",
            "Cumulative funding costs paid in USDT",
            ["symbol"],
            registry=reg,
        )
        logger.debug("prometheus_client metrics initialised with dedicated registry")

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade.

        Args:
            trade: Trade dict with keys: ``pnl``, ``symbol``, ``direction``,
                ``strategy``, ``result`` (``"win"`` / ``"loss"``), and
                optionally ``latency_seconds``.
        """
        self._total_trades += 1
        pnl = trade.get("pnl", 0.0)
        if pnl > 0:
            self._winning_trades += 1
        self._daily_pnl += pnl

        symbol = trade.get("symbol", "unknown")
        direction = trade.get("direction", "unknown")
        strategy = trade.get("strategy", "unknown")
        result = "win" if pnl > 0 else "loss"

        # Update prometheus_client labeled counter / histogram
        if self._prom_trades_total is not None:
            self._prom_trades_total.labels(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                result=result,
            ).inc()
        if self._prom_trade_pnl is not None:
            self._prom_trade_pnl.labels(symbol=symbol, strategy=strategy).observe(pnl)

        latency = trade.get("latency_seconds")
        if latency is not None:
            self.record_trade_latency(float(latency))

        logger.debug(
            "Metrics: trade recorded — total={} daily_pnl={:.4f}",
            self._total_trades,
            self._daily_pnl,
        )

    def record_trade_latency(
        self,
        latency_seconds: float,
        *,
        exchange: str = "unknown",
        order_type: str = "market",
    ) -> None:
        """Record a trade / order execution latency sample.

        Args:
            latency_seconds: Time from order placement to fill (seconds).
            exchange: Exchange identifier for labeled histogram (keyword-only).
            order_type: Order type, e.g. ``"market"`` or ``"limit"`` (keyword-only).
        """
        self._trade_latencies.append(latency_seconds)
        self._trade_latency_sum += latency_seconds
        self._trade_latency_count += 1
        if len(self._trade_latencies) > 10_000:
            self._trade_latencies = self._trade_latencies[-10_000:]

        if self._prom_order_latency is not None:
            self._prom_order_latency.labels(
                exchange=exchange, order_type=order_type
            ).observe(latency_seconds)

    def record_exchange_error(self, error_type: str = "unknown") -> None:
        """Increment the exchange error counter with a specific error type label.

        Args:
            error_type: One of ``"network"``, ``"auth"``, ``"insufficient_balance"``,
                ``"maintenance"``, ``"order_rejected"``, ``"rate_limit"``, or ``"unknown"``.
        """
        self._error_count += 1
        if self._prom_exchange_errors is not None:
            self._prom_exchange_errors.labels(error_type=error_type).inc()

    def record_funding_cost(self, symbol: str, cost_usdt: float) -> None:
        """Record funding cost paid for *symbol*.

        Args:
            symbol: Trading symbol (e.g. ``"BTC/USDT"``).
            cost_usdt: Funding cost in USDT (positive = paid, negative = received).
        """
        if cost_usdt > 0 and self._prom_funding_costs is not None:
            self._prom_funding_costs.labels(symbol=symbol).inc(cost_usdt)

    def update_portfolio(self, portfolio: dict) -> None:
        """Update portfolio-level metrics.

        Args:
            portfolio: Dict with ``equity``, ``open_positions``,
                and optionally ``daily_pnl_pct``, ``circuit_breaker_active``.
        """
        self._portfolio_value = portfolio.get("equity", 0.0)
        self._active_positions = portfolio.get("open_positions", 0)
        self._daily_pnl_pct = portfolio.get("daily_pnl_pct", 0.0)
        self._circuit_breaker_active = bool(portfolio.get("circuit_breaker_active", False))

        if self._prom_open_positions is not None:
            self._prom_open_positions.set(self._active_positions)
        if self._prom_portfolio_equity is not None:
            self._prom_portfolio_equity.set(self._portfolio_value)
        if self._prom_daily_pnl_pct is not None:
            self._prom_daily_pnl_pct.set(self._daily_pnl_pct)
        if self._prom_circuit_breaker is not None:
            self._prom_circuit_breaker.set(1.0 if self._circuit_breaker_active else 0.0)

        logger.debug(
            "Metrics: portfolio updated — equity={:.2f} positions={}",
            self._portfolio_value,
            self._active_positions,
        )

    def record_api_call(self, exchange: str, latency: float) -> None:
        """Record an API call latency.

        Args:
            exchange: Exchange identifier.
            latency: Call duration in seconds.
        """
        samples = self._api_latencies[exchange]
        samples.append(latency)
        if len(samples) > self._latency_window:
            self._api_latencies[exchange] = samples[-self._latency_window:]

    def record_signal(self) -> None:
        """Increment the signal counter."""
        self._signal_count += 1

    def record_error(self) -> None:
        """Increment the generic error counter (prefer record_exchange_error with a type)."""
        self._error_count += 1

    def set_circuit_breaker(self, active: bool) -> None:
        """Update the circuit breaker gauge.

        Args:
            active: True when the circuit breaker is tripped.
        """
        self._circuit_breaker_active = active
        if self._prom_circuit_breaker is not None:
            self._prom_circuit_breaker.set(1.0 if active else 0.0)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    @property
    def win_rate(self) -> float:
        """Return the overall win rate (0–1)."""
        if self._total_trades == 0:
            return 0.0
        return self._winning_trades / self._total_trades

    def get_metrics(self) -> dict:
        """Return all metrics as a plain dict."""
        avg_latencies = {
            ex: (sum(lats) / len(lats) if lats else 0.0)
            for ex, lats in self._api_latencies.items()
        }
        avg_trade_latency = (
            self._trade_latency_sum / self._trade_latency_count
            if self._trade_latency_count > 0
            else 0.0
        )
        return {
            "active_positions": self._active_positions,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": self.win_rate,
            "daily_pnl": self._daily_pnl,
            "daily_pnl_pct": self._daily_pnl_pct,
            "portfolio_value": self._portfolio_value,
            "signal_count": self._signal_count,
            "error_count": self._error_count,
            "circuit_breaker_active": self._circuit_breaker_active,
            "api_latency_avg": avg_latencies,
            "trade_latency_avg": avg_trade_latency,
            "trade_latency_count": self._trade_latency_count,
            "prometheus_available": _PROMETHEUS_AVAILABLE,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Prometheus text export
    # ------------------------------------------------------------------

    def _build_histogram_lines(
        self, name: str, help_text: str, samples: List[float]
    ) -> List[str]:
        """Build Prometheus histogram text lines for *samples* (fallback mode)."""
        lines = [
            f"# HELP {name} {help_text}",
            f"# TYPE {name} histogram",
        ]
        total_sum = sum(samples)
        total_count = len(samples)
        for bucket in _TRADE_LATENCY_BUCKETS:
            bucket_count = sum(1 for s in samples if s <= bucket)
            lines.append(f'{name}_bucket{{le="{bucket}"}} {bucket_count}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {total_count}')
        lines.append(f"{name}_sum {total_sum:.6f}")
        lines.append(f"{name}_count {total_count}")
        return lines

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format.

        When ``prometheus_client`` is available, delegates to
        ``generate_latest()`` which includes all labeled metrics.
        Falls back to a hand-rolled format when the library is absent.
        """
        if _PROMETHEUS_AVAILABLE and self._registry is not None:
            try:
                from prometheus_client import generate_latest
                return generate_latest(self._registry).decode("utf-8")
            except Exception as exc:
                logger.debug("prometheus_client generate_latest failed ({}); using fallback", exc)

        # Fallback: hand-rolled Prometheus text format (no labels)
        m = self.get_metrics()
        lines: List[str] = [
            "# HELP trading_active_positions Number of open positions",
            "# TYPE trading_active_positions gauge",
            f"trading_active_positions {m['active_positions']}",
            "# HELP trades_total Total number of trades executed",
            "# TYPE trades_total counter",
            f"trades_total {m['total_trades']}",
            "# HELP trading_win_rate Win rate as a fraction (0–1)",
            "# TYPE trading_win_rate gauge",
            f"trading_win_rate {m['win_rate']:.6f}",
            "# HELP trading_daily_pnl Daily realised P&L in USDT",
            "# TYPE trading_daily_pnl gauge",
            f"trading_daily_pnl {m['daily_pnl']:.6f}",
            "# HELP daily_pnl_pct Daily P&L as percentage of equity",
            "# TYPE daily_pnl_pct gauge",
            f"daily_pnl_pct {m['daily_pnl_pct']:.6f}",
            "# HELP portfolio_equity Total portfolio equity in USDT",
            "# TYPE portfolio_equity gauge",
            f"portfolio_equity {m['portfolio_value']:.6f}",
            "# HELP open_positions_count Number of currently open positions",
            "# TYPE open_positions_count gauge",
            f"open_positions_count {m['active_positions']}",
            "# HELP circuit_breaker_active 1 when circuit breaker is active",
            "# TYPE circuit_breaker_active gauge",
            f"circuit_breaker_active {1 if m['circuit_breaker_active'] else 0}",
            "# HELP signals_generated Total number of trading signals generated",
            "# TYPE signals_generated counter",
            f"signals_generated {m['signal_count']}",
            "# HELP exchange_errors_total Total number of exchange errors",
            "# TYPE exchange_errors_total counter",
            f"exchange_errors_total {m['error_count']}",
        ]

        for exchange, avg_lat in m["api_latency_avg"].items():
            lines += [
                f"# HELP order_latency_seconds Average REST API latency for {exchange}",
                "# TYPE order_latency_seconds gauge",
                f'order_latency_seconds{{exchange="{exchange}"}} {avg_lat:.6f}',
            ]

        if self._trade_latencies:
            lines += self._build_histogram_lines(
                "trade_latency_seconds",
                "Time from signal creation to order fill (seconds)",
                self._trade_latencies,
            )

        return "\n".join(lines) + "\n"

    def reset_daily(self) -> None:
        """Reset daily counters at the start of a new trading day."""
        self._daily_pnl = 0.0
        self._daily_pnl_pct = 0.0
        self._winning_trades = 0
        self._total_trades = 0
        if self._prom_daily_pnl_pct is not None:
            self._prom_daily_pnl_pct.set(0.0)
        logger.info("Daily metrics reset")

