"""Tests for enhanced fee optimization (Phase 3: System 3)."""

import pytest

from crypto_trading_bot.execution.gateio_fee_optimizer import GateioFeeOptimizer


class TestFeeOptimization:
    """Test suite for dynamic fee optimization and smart routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = GateioFeeOptimizer(
            maker_fee=-0.00025,  # -0.025% rebate
            taker_fee=0.00075,   # 0.075% fee
        )

    def test_tight_spread_high_confidence_uses_market(self):
        """Test that tight spread + high confidence routes to market order."""
        route = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.9,
            spread_bps=2.0,  # Tight spread
            book_state={"bid_depth_usdt": 10000, "ask_depth_usdt": 10000},
            position_notional=1000.0,
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        assert route["order_type"] == "market"
        assert route["use_iceberg"] is False
        assert route["chunk_count"] == 1
        assert "High confidence" in route["reasoning"]
        assert "tight spread" in route["reasoning"]

    def test_wide_spread_uses_maker(self):
        """Test that wide spread routes to post_only limit order."""
        # Fee cost = 2.5 + 7.5 = 10 bps
        # Wide spread = 25 bps (> 2× fee cost)
        route = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.6,
            spread_bps=25.0,  # Wide spread
            book_state={"bid_depth_usdt": 10000, "ask_depth_usdt": 10000},
            position_notional=1000.0,
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        assert route["order_type"] == "limit_passive"
        assert route["use_iceberg"] is False
        assert "Wide spread" in route["reasoning"]

    def test_large_order_uses_twap(self):
        """Test that large orders (> 5% depth) route to TWAP with iceberg."""
        visible_depth = 10000.0
        large_position = visible_depth * 0.1  # 10% of depth

        route = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.7,
            spread_bps=5.0,
            book_state={"bid_depth_usdt": visible_depth, "ask_depth_usdt": visible_depth},
            position_notional=large_position,
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        assert route["order_type"] == "twap"
        assert route["use_iceberg"] is True
        assert route["chunk_count"] >= 2
        assert "Large order" in route["reasoning"]

    def test_fee_aware_breakeven_calculation(self):
        """Test that fee cost breakdown is calculated correctly."""
        cost = self.optimizer.calculate_trade_cost(
            entry_price=100.0,
            exit_price=102.0,
            amount=10.0,  # 10 BTC
            leverage=1.0,
            direction="long",
            use_maker_entry=True,
        )

        # Entry: 10 × 100 × -0.00025 = -0.25 USDT (rebate)
        # Exit: 10 × 102 × 0.00075 = 0.765 USDT (fee)
        # Funding: 10 × 100 × 0.0001 × 3 = 0.3 USDT
        # Total: -0.25 + 0.765 + 0.3 = 0.815 USDT

        assert cost["entry_fee"] < 0  # Rebate
        assert cost["exit_fee"] > 0
        assert cost["funding_estimate"] > 0
        assert cost["total_cost"] > 0
        assert cost["break_even_pct"] > 0

        # Verify break-even percentage
        notional = 10.0 * 100.0
        expected_breakeven = (cost["total_cost"] / notional) * 100.0
        assert abs(cost["break_even_pct"] - expected_breakeven) < 0.001

    def test_viability_check_passes(self):
        """Test that viable trades pass the fee filter."""
        cost = self.optimizer.calculate_trade_cost(
            entry_price=100.0,
            exit_price=102.0,
            amount=10.0,
            leverage=1.0,
            direction="long",
            use_maker_entry=True,
        )

        # Expected profit: 2% (102 - 100) / 100
        # Break-even: ~0.08% (from calculation above)
        # 2% >> 2 × 0.08% → should pass
        is_viable = self.optimizer.trade_is_viable(
            cost_breakdown=cost,
            expected_profit_pct=2.0,
            min_profit_to_cost_ratio=2.0,
        )

        assert is_viable is True

    def test_viability_check_fails(self):
        """Test that unprofitable trades fail the fee filter."""
        cost = self.optimizer.calculate_trade_cost(
            entry_price=100.0,
            exit_price=100.1,  # Only 0.1% profit
            amount=10.0,
            leverage=1.0,
            direction="long",
            use_maker_entry=False,  # Use taker fees
        )

        # Expected profit: 0.1%
        # Break-even with taker fees: ~0.15%
        # 0.1% < 2 × 0.15% → should fail
        is_viable = self.optimizer.trade_is_viable(
            cost_breakdown=cost,
            expected_profit_pct=0.1,
            min_profit_to_cost_ratio=2.0,
        )

        assert is_viable is False

    def test_default_routing_fallback(self):
        """Test default routing when no special conditions apply."""
        route = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.6,
            spread_bps=8.0,  # Moderate spread
            book_state={"bid_depth_usdt": 10000, "ask_depth_usdt": 10000},
            position_notional=100.0,  # Small position
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        assert route["order_type"] == "limit_passive"
        assert route["use_iceberg"] is False
        assert "Default routing" in route["reasoning"]

    def test_chunk_count_scaling(self):
        """Test that chunk count scales appropriately with order size."""
        visible_depth = 10000.0

        # Small order relative to depth
        route_small = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.7,
            spread_bps=5.0,
            book_state={"bid_depth_usdt": visible_depth, "ask_depth_usdt": visible_depth},
            position_notional=visible_depth * 0.06,  # 6% of depth
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        # Large order relative to depth
        route_large = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.7,
            spread_bps=5.0,
            book_state={"bid_depth_usdt": visible_depth, "ask_depth_usdt": visible_depth},
            position_notional=visible_depth * 0.2,  # 20% of depth
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        # Larger order should have more chunks
        if route_small["order_type"] == "twap" and route_large["order_type"] == "twap":
            assert route_large["chunk_count"] >= route_small["chunk_count"]

    def test_zero_depth_handling(self):
        """Test handling of zero or missing depth data."""
        route = self.optimizer.calculate_optimal_execution_route(
            signal_confidence=0.7,
            spread_bps=5.0,
            book_state={"bid_depth_usdt": 0, "ask_depth_usdt": 0},
            position_notional=1000.0,
            maker_fee_bps=-2.5,
            taker_fee_bps=7.5,
        )

        # Should fall back to default routing
        assert route["order_type"] in ["market", "limit_passive"]
        assert route["use_iceberg"] is False
