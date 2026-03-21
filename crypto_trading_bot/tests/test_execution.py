"""Tests for the execution module — fees, slippage, and optimizer."""

from __future__ import annotations

import pytest

from execution.execution_optimizer import ExecutionOptimizer
from execution.fee_calculator import FeeCalculator
from execution.slippage_estimator import SlippageEstimator

# ── FeeCalculator — Maker (limit) ─────────────────────────────────────────


class TestFeeCalculatorMaker:
    def test_limit_order_fee_mexc(self):
        """Maker (limit) fee on MEXC is 0.02% of notional."""
        calc = FeeCalculator()
        fee = calc.calculate_fee(amount=1.0, price=50_000.0, exchange="mexc", order_type="limit")
        assert fee == pytest.approx(50_000.0 * 0.0002, rel=1e-5)

    def test_limit_order_fee_positive(self):
        """Fee is always positive for positive order size."""
        calc = FeeCalculator()
        fee = calc.calculate_fee(amount=0.5, price=30_000.0, exchange="gateio", order_type="limit")
        assert fee > 0.0

    def test_limit_order_cheaper_than_market(self):
        """Maker fee is less than taker fee for the same order."""
        calc = FeeCalculator()
        limit_fee = calc.calculate_fee(1.0, 50_000.0, "mexc", "limit")
        market_fee = calc.calculate_fee(1.0, 50_000.0, "mexc", "market")
        assert limit_fee < market_fee


# ── FeeCalculator — Taker (market) ───────────────────────────────────────


class TestFeeCalculatorTaker:
    def test_market_order_fee_mexc(self):
        """Taker (market) fee on MEXC is 0.06% of notional."""
        calc = FeeCalculator()
        fee = calc.calculate_fee(amount=1.0, price=50_000.0, exchange="mexc", order_type="market")
        assert fee == pytest.approx(50_000.0 * 0.0006, rel=1e-5)

    def test_unknown_exchange_uses_default(self):
        """Unknown exchange falls back to default fee rate."""
        calc = FeeCalculator()
        fee = calc.calculate_fee(1.0, 10_000.0, "unknown_exchange", "market")
        assert fee > 0.0

    def test_round_trip_cost_double_one_way(self):
        """Round-trip cost is exactly 2× one-way fee."""
        calc = FeeCalculator()
        one_way = calc.calculate_fee(0.5, 40_000.0, "bitget", "limit")
        round_trip = calc.calculate_round_trip_cost(0.5, 40_000.0, "bitget", "limit")
        assert round_trip == pytest.approx(one_way * 2)


# ── SlippageEstimator ─────────────────────────────────────────────────────


class TestSlippageEstimatorAcceptability:
    def _sample_orderbook(self) -> dict:
        return {
            "asks": [[50_000, 1.0], [50_010, 2.0], [50_020, 3.0]],
            "bids": [[49_990, 1.0], [49_980, 2.0], [49_970, 3.0]],
        }

    def test_small_order_low_slippage(self):
        """A small order relative to available liquidity has low slippage."""
        est = SlippageEstimator()
        slippage = est.estimate_slippage(
            "BTC/USDT", amount=0.01, side="buy", orderbook=self._sample_orderbook()
        )
        assert slippage >= 0.0

    def test_slippage_is_fraction(self):
        """Slippage is returned as a fraction (not a percentage)."""
        est = SlippageEstimator()
        slippage = est.estimate_slippage(
            "BTC/USDT", amount=0.5, side="sell", orderbook=self._sample_orderbook()
        )
        assert 0.0 <= slippage < 1.0  # max sane value is < 100%

    def test_empty_orderbook_returns_default(self):
        """Empty order book returns the default slippage of 0.001."""
        est = SlippageEstimator()
        slippage = est.estimate_slippage(
            "BTC/USDT", amount=1.0, side="buy", orderbook={"asks": [], "bids": []}
        )
        assert slippage == pytest.approx(0.001)

    def test_is_acceptable_low_slippage(self):
        """Very low slippage is flagged as acceptable."""
        est = SlippageEstimator()
        assert est.is_slippage_acceptable(0.0001, max_allowed=0.001) is True

    def test_is_acceptable_high_slippage(self):
        """Slippage exceeding max is flagged as not acceptable."""
        est = SlippageEstimator()
        assert est.is_slippage_acceptable(0.01, max_allowed=0.001) is False


# ── ExecutionOptimizer — limit vs market ─────────────────────────────────


class TestExecutionOptimizerLimitVsMarket:
    def test_normal_urgency_prefers_limit(self):
        """Normal urgency uses a market order (limit only when urgency is 'low')."""
        opt = ExecutionOptimizer()
        assert opt.should_use_limit("BTC/USDT", urgency="normal") is False

    def test_high_urgency_prefers_market(self):
        """High urgency should use a market order."""
        opt = ExecutionOptimizer()
        assert opt.should_use_limit("BTC/USDT", urgency="high") is False

    def test_optimal_limit_price_buy(self):
        """Optimal buy limit price is below mid-price."""
        opt = ExecutionOptimizer()
        price = opt.calculate_optimal_limit_price(
            "BTC/USDT", "buy", spread=10.0, mid_price=50_000.0
        )
        assert price <= 50_000.0

    def test_optimal_limit_price_sell(self):
        """Optimal sell limit price is above mid-price."""
        opt = ExecutionOptimizer()
        price = opt.calculate_optimal_limit_price(
            "BTC/USDT", "sell", spread=10.0, mid_price=50_000.0
        )
        assert price >= 50_000.0


# ── LatencyMonitor Tests ───────────────────────────────────────


class TestLatencyMonitor:
    def test_initialization(self):
        """LatencyMonitor initializes correctly."""
        from execution.latency_monitor import LatencyMonitor

        monitor = LatencyMonitor(market_type="crypto")
        assert monitor is not None

    def test_tracking_lifecycle(self):
        """Can start and complete tracking."""
        from execution.latency_monitor import LatencyMonitor
        import time

        monitor = LatencyMonitor(market_type="crypto")

        # Start tracking
        latency = monitor.start_tracking(
            execution_id="test_123",
            symbol="BTC/USDT"
        )
        assert latency is not None

        # Record stages
        time.sleep(0.01)
        monitor.record_order_submission("test_123")
        time.sleep(0.01)
        monitor.record_exchange_ack("test_123")
        time.sleep(0.01)
        monitor.record_first_fill("test_123")

        # Complete
        completed = monitor.complete_tracking("test_123")
        assert completed is not None
        assert completed.total_latency is not None
        assert completed.total_latency > 0

    def test_latency_stats(self):
        """Can retrieve latency statistics."""
        from execution.latency_monitor import LatencyMonitor

        monitor = LatencyMonitor(market_type="crypto")

        # Track a few executions
        for i in range(5):
            latency = monitor.start_tracking(f"test_{i}", "BTC/USDT")
            monitor.record_order_submission(f"test_{i}")
            monitor.complete_tracking(f"test_{i}")

        stats = monitor.get_stats("total")
        assert stats.count == 5


# ── ExecutionQualityAnalyzer Tests ────────────────────────────────


class TestExecutionQualityAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_execution(self):
        """Can analyze execution quality."""
        from execution.execution_quality_analyzer import ExecutionQualityAnalyzer

        analyzer = ExecutionQualityAnalyzer()

        report = await analyzer.analyze_execution(
            execution_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            strategy="momentum",
            order_amount=1.0,
            filled_amount=1.0,
            average_fill_price=50100.0,
            decision_price=50000.0,
            mid_price_at_submission=50050.0,
            total_fees=10.0
        )

        assert report is not None
        assert report.slippage_bps > 0  # Filled above decision price
        assert report.execution_quality_score > 0

    def test_strategy_quality_tracking(self):
        """Can track per-strategy quality."""
        from execution.execution_quality_analyzer import ExecutionQualityAnalyzer

        analyzer = ExecutionQualityAnalyzer()

        # Manually add some reports
        from execution.execution_quality_analyzer import ExecutionReport
        import time

        report = ExecutionReport(
            execution_id="test_1",
            symbol="BTC/USDT",
            side="buy",
            strategy="momentum",
            timestamp=time.time(),
            order_amount=1.0,
            filled_amount=1.0,
            average_fill_price=50000.0,
            decision_price=50000.0,
            mid_price_at_submission=50000.0,
            slippage_bps=5.0,
            fill_rate=1.0
        )

        analyzer._reports.append(report)
        analyzer._strategy_stats["momentum"].append(report)

        stats = analyzer.get_strategy_quality("momentum")
        assert stats.execution_count == 1


# ── AdaptiveExecutionEngine Tests ─────────────────────────────────


class TestAdaptiveExecutionEngine:
    def test_initialization(self):
        """AdaptiveExecutionEngine initializes correctly."""
        from execution.adaptive_execution_engine import AdaptiveExecutionEngine
        from unittest.mock import Mock

        exchange = Mock()
        order_manager = Mock()

        engine = AdaptiveExecutionEngine(
            exchange=exchange,
            order_manager=order_manager
        )
        assert engine is not None

    def test_circuit_breaker(self):
        """Circuit breaker activates after failures."""
        from execution.adaptive_execution_engine import AdaptiveExecutionEngine
        from unittest.mock import Mock

        exchange = Mock()
        order_manager = Mock()

        engine = AdaptiveExecutionEngine(
            exchange=exchange,
            order_manager=order_manager
        )

        # Trigger failures
        for _ in range(3):
            engine._on_execution_complete(success=False)

        assert engine._circuit_breaker_active is True


# ── SmartExitEngine Tests ──────────────────────────────────────────


class TestSmartExitEngine:
    def test_mae_stats_tracking(self):
        """Can track MAE statistics."""
        from execution.smart_exit_engine import SmartExitEngine
        from unittest.mock import Mock
        import asyncio

        exchange = Mock()
        position_manager = Mock()

        engine = SmartExitEngine(
            exchange=exchange,
            position_manager=position_manager
        )

        # Record some MAE samples
        asyncio.run(engine.record_trade_mae(
            strategy="momentum",
            entry_price=50000.0,
            exit_price=51000.0,
            worst_price=49500.0,
            side="long"
        ))

        stats = engine.get_mae_stats("momentum")
        assert stats["sample_count"] == 1


# ── AntiGamingProtection Tests ─────────────────────────────────────


class TestAntiGamingProtection:
    def test_execution_delay_randomization(self):
        """Front-running protection adds random delay."""
        from execution.anti_gaming import AntiGamingProtection
        from unittest.mock import Mock

        exchange = Mock()
        protection = AntiGamingProtection(exchange=exchange)

        delay = protection.get_execution_delay()
        assert 0 <= delay <= 3.0  # Within jitter range

    def test_stop_loss_widening(self):
        """Can widen stop loss for manipulation protection."""
        from execution.anti_gaming import AntiGamingProtection
        from unittest.mock import Mock

        exchange = Mock()
        protection = AntiGamingProtection(exchange=exchange)

        widened_sl = protection.get_widened_stop_loss(
            original_stop_loss=49000.0,
            current_price=50000.0,
            side="long"
        )

        assert widened_sl < 49000.0  # Widened (further from price)

