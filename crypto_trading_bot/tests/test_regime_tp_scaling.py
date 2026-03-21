"""Tests for regime-aware take profit scaling (Phase 3: System 4)."""

import pytest

from crypto_trading_bot.risk.dynamic_take_profit import DynamicTakeProfitEngine


class TestRegimeTPScaling:
    """Test suite for regime-aware take profit adjustments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DynamicTakeProfitEngine()

    def _create_sample_tp_levels(self) -> list:
        """Create sample TP levels for testing."""
        return [
            {"price": 110.0, "percentage": 0.30, "type": "fixed"},
            {"price": 120.0, "percentage": 0.30, "type": "fixed"},
            {"price": 130.0, "percentage": 0.20, "type": "fixed"},
            {"price": 0.0, "percentage": 0.20, "type": "trailing"},
        ]

    def test_trending_regime_widens_tps(self):
        """Test that trending regime uses 2x wider TPs."""
        # Calculate TPs in trending regime
        tp_levels_trending = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="long",
            atr=2.0,
            market_regime="trending_bullish",
        )

        # Calculate TPs in normal regime
        tp_levels_normal = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="long",
            atr=2.0,
            market_regime="unknown",
        )

        # Extract fixed TP prices
        trending_prices = [lvl["price"] for lvl in tp_levels_trending if lvl["type"] == "fixed"]
        normal_prices = [lvl["price"] for lvl in tp_levels_normal if lvl["type"] == "fixed"]

        # Trending TPs should be wider (higher prices for long)
        for trending_tp, normal_tp in zip(trending_prices, normal_prices):
            assert trending_tp > normal_tp, f"Trending TP {trending_tp} should be > normal TP {normal_tp}"

        # Verify approximately 2x multiplier (trending uses 2.0 vs 1.0)
        # TP1: 1.5 ATR × 2.0 = 3.0 ATR vs 1.5 ATR × 1.0 = 1.5 ATR
        assert trending_prices[0] > normal_prices[0] * 1.5

    def test_ranging_regime_tightens_tps(self):
        """Test that ranging regime uses 0.7x tighter TPs."""
        # Calculate TPs in ranging regime
        tp_levels_ranging = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="long",
            atr=2.0,
            market_regime="ranging",
        )

        # Calculate TPs in normal regime
        tp_levels_normal = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="long",
            atr=2.0,
            market_regime="unknown",
        )

        # Extract fixed TP prices
        ranging_prices = [lvl["price"] for lvl in tp_levels_ranging if lvl["type"] == "fixed"]
        normal_prices = [lvl["price"] for lvl in tp_levels_normal if lvl["type"] == "fixed"]

        # Ranging TPs should be tighter (lower prices for long)
        for ranging_tp, normal_tp in zip(ranging_prices, normal_prices):
            assert ranging_tp < normal_tp, f"Ranging TP {ranging_tp} should be < normal TP {normal_tp}"

        # Verify approximately 0.7x multiplier
        assert ranging_prices[0] < normal_prices[0] * 0.8

    def test_crash_regime_minimal_tps(self):
        """Test that crash regime uses 0.5x minimal TPs."""
        # Calculate TPs in crash regime
        tp_levels_crash = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="long",
            atr=2.0,
            market_regime="crash",
        )

        # Calculate TPs in normal regime
        tp_levels_normal = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="long",
            atr=2.0,
            market_regime="unknown",
        )

        # Extract fixed TP prices
        crash_prices = [lvl["price"] for lvl in tp_levels_crash if lvl["type"] == "fixed"]
        normal_prices = [lvl["price"] for lvl in tp_levels_normal if lvl["type"] == "fixed"]

        # Crash TPs should be much tighter (grab any profit)
        for crash_tp, normal_tp in zip(crash_prices, normal_prices):
            assert crash_tp < normal_tp, f"Crash TP {crash_tp} should be < normal TP {normal_tp}"

        # Verify approximately 0.5x multiplier
        assert crash_prices[0] < normal_prices[0] * 0.6

    def test_high_volatility_increases_trailing(self):
        """Test that high volatility increases trailing TP from 20% to 30%."""
        sample_levels = self._create_sample_tp_levels()

        # Adjust for high volatility
        adjusted = self.engine.adjust_tp_for_regime(
            tp_levels=sample_levels,
            market_regime="unknown",
            volatility_regime="high_volatility",
        )

        # Find trailing level
        trailing_original = next(lvl for lvl in sample_levels if lvl["type"] == "trailing")
        trailing_adjusted = next(lvl for lvl in adjusted if lvl["type"] == "trailing")

        assert trailing_original["percentage"] == 0.20
        assert trailing_adjusted["percentage"] == 0.30

    def test_crash_regime_adjustment(self):
        """Test crash regime adjustment reduces TP distances."""
        sample_levels = self._create_sample_tp_levels()

        # Adjust for crash regime
        adjusted = self.engine.adjust_tp_for_regime(
            tp_levels=sample_levels,
            market_regime="crash",
            volatility_regime="normal",
        )

        # Fixed TPs should be reduced
        for i, (original, adj) in enumerate(zip(sample_levels, adjusted)):
            if original["type"] == "fixed":
                assert adj["price"] < original["price"], f"TP{i+1} should be reduced in crash regime"

    def test_ranging_regime_tightens_tp1(self):
        """Test ranging regime tightens TP1 specifically."""
        sample_levels = self._create_sample_tp_levels()

        # Adjust for ranging regime
        adjusted = self.engine.adjust_tp_for_regime(
            tp_levels=sample_levels,
            market_regime="ranging",
            volatility_regime="normal",
        )

        # TP1 (first fixed level) should be tightened
        tp1_original = sample_levels[0]
        tp1_adjusted = adjusted[0]

        assert tp1_adjusted["price"] < tp1_original["price"]
        # Should be approximately 80% of original (1.2/1.5 = 0.8)
        assert tp1_adjusted["price"] < tp1_original["price"] * 0.85

    def test_combined_regime_adjustments(self):
        """Test that multiple regime adjustments can be applied together."""
        sample_levels = self._create_sample_tp_levels()

        # Adjust for crash + high volatility
        adjusted = self.engine.adjust_tp_for_regime(
            tp_levels=sample_levels,
            market_regime="crash",
            volatility_regime="high_volatility",
        )

        # Fixed TPs should be reduced (crash)
        for original, adj in zip(sample_levels, adjusted):
            if original["type"] == "fixed":
                assert adj["price"] < original["price"]

        # Trailing should be increased (high vol)
        trailing_adj = next(lvl for lvl in adjusted if lvl["type"] == "trailing")
        assert trailing_adj["percentage"] == 0.30

    def test_no_adjustment_for_normal_regime(self):
        """Test that normal regimes don't modify TPs."""
        sample_levels = self._create_sample_tp_levels()

        # Adjust for normal regimes
        adjusted = self.engine.adjust_tp_for_regime(
            tp_levels=sample_levels,
            market_regime="unknown",
            volatility_regime="normal",
        )

        # Should be unchanged
        for original, adj in zip(sample_levels, adjusted):
            if original["type"] == "fixed":
                assert adj["price"] == original["price"]
            if original["type"] == "trailing":
                assert adj["percentage"] == original["percentage"]

    def test_empty_tp_levels_handling(self):
        """Test handling of empty TP levels list."""
        adjusted = self.engine.adjust_tp_for_regime(
            tp_levels=[],
            market_regime="crash",
            volatility_regime="high_volatility",
        )

        assert adjusted == []

    def test_short_position_regime_scaling(self):
        """Test that regime scaling works correctly for short positions."""
        # Calculate TPs for short in trending regime
        tp_levels_trending = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="short",
            atr=2.0,
            market_regime="trending_bearish",
        )

        # Calculate TPs for short in normal regime
        tp_levels_normal = self.engine.calculate_tp_levels(
            entry_price=100.0,
            direction="short",
            atr=2.0,
            market_regime="unknown",
        )

        # Extract fixed TP prices
        trending_prices = [lvl["price"] for lvl in tp_levels_trending if lvl["type"] == "fixed"]
        normal_prices = [lvl["price"] for lvl in tp_levels_normal if lvl["type"] == "fixed"]

        # For shorts, trending TPs should be lower (wider distance down)
        for trending_tp, normal_tp in zip(trending_prices, normal_prices):
            assert trending_tp < normal_tp, f"Trending short TP {trending_tp} should be < normal TP {normal_tp}"
