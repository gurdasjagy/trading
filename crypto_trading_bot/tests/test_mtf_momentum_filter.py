"""Tests for MTFMomentumFilter."""

import numpy as np
import pandas as pd
import pytest

from crypto_trading_bot.strategy.signals.mtf_momentum_filter import MTFMomentumFilter


class TestMTFMomentumFilter:
    """Test suite for multi-timeframe momentum filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.filter = MTFMomentumFilter()

    def _create_trending_data(self, length: int, trend: str) -> pd.DataFrame:
        """Create synthetic OHLCV data with a specific trend.

        Args:
            length: Number of candles
            trend: "bullish", "bearish", or "neutral"

        Returns:
            DataFrame with OHLCV columns
        """
        if trend == "bullish":
            # Upward trending prices
            close = np.linspace(100, 150, length)
        elif trend == "bearish":
            # Downward trending prices
            close = np.linspace(150, 100, length)
        else:
            # Sideways/neutral
            close = np.ones(length) * 125 + np.random.randn(length) * 2

        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(length) * 1000,
        })

    def test_all_timeframes_agree_bullish(self):
        """Test 3/3 bullish agreement returns 1.0 multiplier."""
        market_data = {
            "1h": self._create_trending_data(100, "bullish"),
            "4h": self._create_trending_data(100, "bullish"),
            "1d": self._create_trending_data(100, "bullish"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is True
        assert multiplier == 1.0
        assert "3/3" in reasoning
        assert "bullish" in reasoning.lower()

    def test_all_timeframes_agree_bearish(self):
        """Test 3/3 bearish agreement returns 1.0 multiplier."""
        market_data = {
            "1h": self._create_trending_data(100, "bearish"),
            "4h": self._create_trending_data(100, "bearish"),
            "1d": self._create_trending_data(100, "bearish"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is True
        assert multiplier == 1.0
        assert "3/3" in reasoning
        assert "bearish" in reasoning.lower()

    def test_two_of_three_agree(self):
        """Test 2/3 agreement returns 0.8 multiplier."""
        market_data = {
            "1h": self._create_trending_data(100, "bullish"),
            "4h": self._create_trending_data(100, "bullish"),
            "1d": self._create_trending_data(100, "neutral"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is True
        assert multiplier == 0.8
        assert "2/3" in reasoning
        assert "bullish" in reasoning.lower()

    def test_one_of_three_agree(self):
        """Test 1/3 agreement returns rejection."""
        market_data = {
            "1h": self._create_trending_data(100, "bullish"),
            "4h": self._create_trending_data(100, "bearish"),
            "1d": self._create_trending_data(100, "neutral"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is False
        assert multiplier == 0.0
        assert "consensus" in reasoning.lower()

    def test_insufficient_data(self):
        """Test handling of missing or insufficient timeframe data."""
        # Only one timeframe provided
        market_data = {
            "1h": self._create_trending_data(100, "bullish"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is False
        assert multiplier == 0.0
        assert "Insufficient timeframes" in reasoning

        # Empty data
        market_data = {}
        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is False
        assert multiplier == 0.0
        assert "No market data" in reasoning

        # Insufficient candles
        market_data = {
            "1h": self._create_trending_data(10, "bullish"),  # Only 10 candles
            "4h": self._create_trending_data(100, "bullish"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        assert should_proceed is False
        assert multiplier == 0.0

    def test_mixed_signals_no_consensus(self):
        """Test that mixed signals (1B/1Be/1N) are rejected."""
        market_data = {
            "1h": self._create_trending_data(100, "bullish"),
            "4h": self._create_trending_data(100, "bearish"),
            "1d": self._create_trending_data(100, "bearish"),
        }

        should_proceed, multiplier, reasoning = self.filter.check_momentum_alignment(market_data)

        # 2 bearish should win
        assert should_proceed is True
        assert multiplier == 0.8

    def test_ema_crossover_detection(self):
        """Test that EMA crossover is correctly detected."""
        # Create data where EMA20 crosses above EMA50
        length = 100
        close = np.concatenate([
            np.linspace(100, 95, 50),  # Downtrend first
            np.linspace(95, 110, 50),  # Then uptrend
        ])

        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(length) * 1000,
        })

        direction = self.filter._analyze_timeframe_momentum(df, "test")

        # After the crossover, EMA20 should be above EMA50
        assert direction == "bullish"

    def test_edge_case_exact_ema_equality(self):
        """Test handling when EMAs are exactly equal (neutral)."""
        # Flat price → EMAs converge
        close = np.ones(100) * 100.0

        df = pd.DataFrame({
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(100) * 1000,
        })

        direction = self.filter._analyze_timeframe_momentum(df, "test")

        assert direction == "neutral"
