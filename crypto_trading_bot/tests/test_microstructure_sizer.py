"""Tests for MicrostructureSizer."""

import pytest

from crypto_trading_bot.risk.microstructure_sizer import MicrostructureSizer


class TestMicrostructureSizer:
    """Test suite for microstructure-aware position sizing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = MicrostructureSizer()

    def test_vpin_multiplier_ranges(self):
        """Test VPIN multiplier returns correct values for different VPIN levels."""
        # Low VPIN (< 0.5) → no reduction
        assert self.sizer.get_vpin_multiplier(0.3) == 1.0
        assert self.sizer.get_vpin_multiplier(0.4) == 1.0

        # Moderate VPIN (0.5-0.7) → 0.85x
        assert self.sizer.get_vpin_multiplier(0.6) == 0.85
        assert self.sizer.get_vpin_multiplier(0.65) == 0.85

        # High VPIN (> 0.7) → 0.7x
        assert self.sizer.get_vpin_multiplier(0.8) == 0.7
        assert self.sizer.get_vpin_multiplier(0.95) == 0.7

        # Boundary cases
        assert self.sizer.get_vpin_multiplier(0.5) == 0.85
        assert self.sizer.get_vpin_multiplier(0.7) == 0.7

    def test_depth_multiplier_calculation(self):
        """Test depth multiplier calculation for different depth scenarios."""
        # Sufficient depth (2x position) → 1.0
        mult = self.sizer.get_depth_multiplier(depth_usdt=2000.0, position_notional=1000.0)
        assert mult == 1.0

        # Exactly 2x depth → 1.0
        mult = self.sizer.get_depth_multiplier(depth_usdt=2000.0, position_notional=1000.0)
        assert mult == 1.0

        # 1x depth (half of required) → 0.5
        mult = self.sizer.get_depth_multiplier(depth_usdt=1000.0, position_notional=1000.0)
        assert mult == 0.5

        # 0.5x depth (quarter of required) → 0.25
        mult = self.sizer.get_depth_multiplier(depth_usdt=500.0, position_notional=1000.0)
        assert mult == 0.25

        # Zero depth → 1.0 (safety fallback)
        mult = self.sizer.get_depth_multiplier(depth_usdt=0.0, position_notional=1000.0)
        assert mult == 1.0

        # Zero position → 1.0 (safety fallback)
        mult = self.sizer.get_depth_multiplier(depth_usdt=1000.0, position_notional=0.0)
        assert mult == 1.0

    def test_spread_multiplier_percentile(self):
        """Test spread multiplier based on percentile calculation."""
        # Create spread history with known distribution
        spread_history = [5.0] * 50 + [10.0] * 25 + [15.0] * 15 + [20.0] * 10

        # Spread at 50th percentile (5.0) → 1.0
        mult = self.sizer.get_spread_multiplier(5.0, spread_history)
        assert mult == 1.0

        # Spread at 80th percentile (10.0) → 0.9
        mult = self.sizer.get_spread_multiplier(12.0, spread_history)
        assert mult == 0.9

        # Spread at 95th percentile (20.0) → 0.8
        mult = self.sizer.get_spread_multiplier(20.0, spread_history)
        assert mult == 0.8

        # Insufficient history → conservative default
        mult = self.sizer.get_spread_multiplier(5.0, [])
        assert mult == 1.0

        mult = self.sizer.get_spread_multiplier(5.0, [3.0, 4.0])
        assert mult == 1.0

        # Wide spread with insufficient history → 0.9
        mult = self.sizer.get_spread_multiplier(15.0, [])
        assert mult == 0.9

    def test_combined_multipliers(self):
        """Test that multipliers can be combined correctly."""
        vpin_mult = self.sizer.get_vpin_multiplier(0.6)  # 0.85
        depth_mult = self.sizer.get_depth_multiplier(1000.0, 1000.0)  # 0.5
        spread_mult = self.sizer.get_spread_multiplier(20.0, [5.0] * 100)  # 0.8

        # Combined effect
        base_size = 1000.0
        adjusted_size = base_size * vpin_mult * depth_mult * spread_mult
        expected = 1000.0 * 0.85 * 0.5 * 0.8  # 340.0

        assert abs(adjusted_size - expected) < 0.01

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # VPIN at exact boundaries
        assert self.sizer.get_vpin_multiplier(0.0) == 1.0
        assert self.sizer.get_vpin_multiplier(1.0) == 0.7

        # Depth with very large position
        mult = self.sizer.get_depth_multiplier(100.0, 10000.0)
        assert 0.0 <= mult <= 1.0

        # Spread with single-value history
        mult = self.sizer.get_spread_multiplier(10.0, [10.0] * 100)
        assert mult == 1.0  # All values equal → 0th percentile
