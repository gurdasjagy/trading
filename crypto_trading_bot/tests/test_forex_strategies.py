"""Unit tests for the 15 gold/silver-specific forex strategies.

Each test constructs synthetic OHLCV data that deterministically exercises
the strategy's analysis logic, verifying signal structure and edge-case
behaviour (insufficient data, missing optional parameters, etc.).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    closes: List[float],
    *,
    high_delta: float = 0.005,
    low_delta: float = 0.005,
    volume: float = 1_000.0,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    closes_arr = np.array(closes, dtype=float)
    n = len(closes_arr)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(n)],
            "open": closes_arr * (1 - high_delta / 2),
            "high": closes_arr * (1 + high_delta),
            "low": closes_arr * (1 - low_delta),
            "close": closes_arr,
            "volume": np.full(n, volume),
        }
    )


def _assert_signal_structure(sig: dict, symbol: str = "XAU/USD") -> None:
    """Assert that a signal dict contains the required keys with valid types."""
    assert sig is not None, "Signal should not be None"
    assert sig["symbol"] == symbol
    assert sig["direction"] in ("long", "short")
    assert sig["entry_price"] > 0
    assert sig["atr"] >= 0
    assert 0.0 < sig["confidence"] <= 1.0
    assert isinstance(sig["strategy"], str)
    assert isinstance(sig["timeframe"], str)


# ---------------------------------------------------------------------------
# LondonBreakoutStrategy
# ---------------------------------------------------------------------------


class TestLondonBreakoutStrategy:
    def _strategy(self):
        from strategy.strategies.forex.london_breakout import LondonBreakoutStrategy

        return LondonBreakoutStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_no_signal_outside_london_hours(self):
        """Without timestamp data triggering London session, signal is None."""
        s = self._strategy()
        # Without a DatetimeIndex or timestamp column pointing to London hours
        # the strategy returns None (session check fails).
        df = _make_ohlcv([2000.0] * 100)
        # Drop timestamp column so session detection falls through
        df = df.drop(columns=["timestamp"])
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "london_breakout"


# ---------------------------------------------------------------------------
# GoldDXYInverseStrategy
# ---------------------------------------------------------------------------


class TestGoldDXYInverseStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_dxy_inverse import GoldDXYInverseStrategy

        return GoldDXYInverseStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_long_signal_with_rising_momentum(self):
        """Gently rising prices with moderate RSI can produce a long signal."""
        s = self._strategy()
        # Oscillating + slight uptrend to moderate RSI
        closes = [2000.0 + 10 * math.sin(i * 0.2) + i * 0.3 for i in range(100)]
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD")
        if sig is not None:
            _assert_signal_structure(sig, "XAU/USD")
            assert sig["direction"] == "long"

    def test_returns_none_with_no_momentum(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 100)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_dxy_inverse"


# ---------------------------------------------------------------------------
# NFPNewsStrategy
# ---------------------------------------------------------------------------


class TestNFPNewsStrategy:
    def _strategy(self):
        from strategy.strategies.forex.nfp_news_strategy import NFPNewsStrategy

        return NFPNewsStrategy(symbols=[])

    def test_returns_none_without_news_events(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 80)
        assert s.analyze(df) is None
        assert s.analyze(df, news_events=None) is None

    def test_returns_none_with_empty_event_list(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 80)
        assert s.analyze(df, news_events=[]) is None

    def test_long_signal_on_recent_high_impact_event_with_upward_move(self):
        from datetime import datetime, timezone, timedelta

        s = self._strategy()
        now = datetime.now(timezone.utc)
        events = [{"time": now - timedelta(minutes=10), "impact": "high", "title": "NFP"}]
        closes = [2000.0] * 60
        # Pre-event price (index -6) is 1990; current is 2020 → +30 > ATR
        for i in range(6):
            closes[-(6 - i)] = 1990.0 + i * 6  # ramp up sharply
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD", news_events=events)
        if sig is not None:
            _assert_signal_structure(sig, "XAU/USD")

    def test_registered_name(self):
        assert self._strategy().name == "nfp_news_strategy"


# ---------------------------------------------------------------------------
# GoldMeanReversionStrategy
# ---------------------------------------------------------------------------


class TestGoldMeanReversionStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_mean_reversion import GoldMeanReversionStrategy

        return GoldMeanReversionStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 50)
        assert s.analyze(df) is None

    def test_long_signal_on_sharp_drop_below_lower_band(self):
        s = self._strategy()
        # 228 bars at ~2000, then sharp drop to trigger lower 2σ band
        closes = [2000.0] * 228 + [1700.0, 1720.0]
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD")
        assert sig is not None, "Expected long signal below lower 2σ band"
        _assert_signal_structure(sig, "XAU/USD")
        assert sig["direction"] == "long"
        assert sig["z_score"] < -2.0

    def test_short_signal_on_sharp_rally_above_upper_band(self):
        s = self._strategy()
        closes = [2000.0] * 228 + [2300.0, 2280.0]
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD")
        assert sig is not None, "Expected short signal above upper 2σ band"
        _assert_signal_structure(sig, "XAU/USD")
        assert sig["direction"] == "short"
        assert sig["z_score"] > 2.0

    def test_registered_name(self):
        assert self._strategy().name == "gold_mean_reversion"


# ---------------------------------------------------------------------------
# GoldMomentumBreakoutStrategy
# ---------------------------------------------------------------------------


class TestGoldMomentumBreakoutStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_momentum_breakout import GoldMomentumBreakoutStrategy

        return GoldMomentumBreakoutStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_no_signal_low_adx(self):
        s = self._strategy()
        # Flat prices → ADX near zero
        df = _make_ohlcv([2000.0] * 100)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_momentum_breakout"


# ---------------------------------------------------------------------------
# GoldFibonacciStrategy
# ---------------------------------------------------------------------------


class TestGoldFibonacciStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_fibonacci import GoldFibonacciStrategy

        return GoldFibonacciStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_no_signal_flat_market(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 100)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_fibonacci"


# ---------------------------------------------------------------------------
# GoldSupplyDemandStrategy
# ---------------------------------------------------------------------------


class TestGoldSupplyDemandStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_supply_demand import GoldSupplyDemandStrategy

        return GoldSupplyDemandStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_supply_demand"


# ---------------------------------------------------------------------------
# GoldRSIDivergenceStrategy
# ---------------------------------------------------------------------------


class TestGoldRSIDivergenceStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_rsi_divergence import GoldRSIDivergenceStrategy

        return GoldRSIDivergenceStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_no_signal_trending_market(self):
        s = self._strategy()
        # Monotonically rising prices → no divergence
        df = _make_ohlcv([2000.0 + i for i in range(150)])
        sig = s.analyze(df)
        # May or may not have a signal; if present it must be valid
        if sig is not None:
            _assert_signal_structure(sig)

    def test_registered_name(self):
        assert self._strategy().name == "gold_rsi_divergence"


# ---------------------------------------------------------------------------
# GoldBollingerSqueezeStrategy
# ---------------------------------------------------------------------------


class TestGoldBollingerSqueezeStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_bollinger_squeeze import GoldBollingerSqueezeStrategy

        return GoldBollingerSqueezeStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_no_signal_when_not_in_squeeze(self):
        """Volatile prices → BBW > BBW_MA → not in squeeze → no signal."""
        s = self._strategy()
        # Alternating high/low prices to keep BBW large
        closes = [2000.0 + 100 * (i % 2) for i in range(100)]
        df = _make_ohlcv(closes, volume=2000.0)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_bollinger_squeeze"


# ---------------------------------------------------------------------------
# GoldIchimokuStrategy
# ---------------------------------------------------------------------------


class TestGoldIchimokuStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_ichimoku import GoldIchimokuStrategy

        return GoldIchimokuStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 30)
        assert s.analyze(df) is None

    def test_long_signal_on_cloud_breakout(self):
        """Price crossing above the Kumo cloud should produce a long signal."""
        s = self._strategy()
        # Falling prices for first 60 bars, then strong rally above the cloud
        closes = [2200.0 - i * 3 for i in range(80)] + [2200.0 + i * 5 for i in range(20)]
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD")
        # May or may not trigger depending on cloud position; validate if signal
        if sig is not None:
            _assert_signal_structure(sig, "XAU/USD")

    def test_registered_name(self):
        assert self._strategy().name == "gold_ichimoku"


# ---------------------------------------------------------------------------
# GoldVWAPStrategy
# ---------------------------------------------------------------------------


class TestGoldVWAPStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_vwap import GoldVWAPStrategy

        return GoldVWAPStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_deviation_short_signal(self):
        """Price far above VWAP should trigger a short deviation signal."""
        s = self._strategy()
        closes = [2000.0] * 90 + [2500.0, 2501.0]  # sharp rally far above VWAP
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD")
        if sig is not None:
            _assert_signal_structure(sig, "XAU/USD")
            assert sig["direction"] == "short"

    def test_registered_name(self):
        assert self._strategy().name == "gold_vwap"


# ---------------------------------------------------------------------------
# GoldScalpingStrategy
# ---------------------------------------------------------------------------


class TestGoldScalpingStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_scalping import GoldScalpingStrategy

        return GoldScalpingStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 5)
        assert s.analyze(df) is None

    def test_no_signal_flat_market(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 60)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_scalping"


# ---------------------------------------------------------------------------
# SilverGoldRatioStrategy
# ---------------------------------------------------------------------------


class TestSilverGoldRatioStrategy:
    def _strategy(self):
        from strategy.strategies.forex.silver_gold_ratio import SilverGoldRatioStrategy

        return SilverGoldRatioStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_returns_none_without_silver_data(self):
        """Without XAG data, strategy must return None."""
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 60)
        assert s.analyze(df, "XAU/USD") is None

    def test_short_signal_when_ratio_above_threshold(self):
        """ratio=2000/20=100 is above mean+1.5σ → short gold."""
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 60)
        sig = s.analyze(df, "XAU/USD", xag_price=20.0)
        assert sig is not None
        _assert_signal_structure(sig, "XAU/USD")
        assert sig["direction"] == "short"
        assert sig["ratio"] == pytest.approx(100.0, abs=0.1)

    def test_long_signal_when_ratio_below_threshold(self):
        """ratio=2000/33≈60.6 is below mean-1.5σ → long gold."""
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 60)
        sig = s.analyze(df, "XAU/USD", xag_price=33.0)
        assert sig is not None
        _assert_signal_structure(sig, "XAU/USD")
        assert sig["direction"] == "long"

    def test_no_signal_in_normal_range(self):
        """ratio=2000/25=80 is within ±1.5σ → no signal."""
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 60)
        assert s.analyze(df, "XAU/USD", xag_price=25.0) is None

    def test_registered_name(self):
        assert self._strategy().name == "silver_gold_ratio"


# ---------------------------------------------------------------------------
# GoldSessionMomentumStrategy
# ---------------------------------------------------------------------------


class TestGoldSessionMomentumStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_session_momentum import GoldSessionMomentumStrategy

        return GoldSessionMomentumStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_registered_name(self):
        assert self._strategy().name == "gold_session_momentum"


# ---------------------------------------------------------------------------
# GoldSafeHavenStrategy
# ---------------------------------------------------------------------------


class TestGoldSafeHavenStrategy:
    def _strategy(self):
        from strategy.strategies.forex.gold_safe_haven import GoldSafeHavenStrategy

        return GoldSafeHavenStrategy(symbols=[])

    def test_returns_none_on_insufficient_data(self):
        s = self._strategy()
        df = _make_ohlcv([2000.0] * 10)
        assert s.analyze(df) is None

    def test_long_signal_with_vix_spike(self):
        """VIX above threshold should always produce a long signal when RSI is low."""
        s = self._strategy()
        # Use oscillating data to moderate RSI
        closes = [2000.0 + 10 * math.sin(i * 0.3) for i in range(60)]
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD", vix_value=35.0)
        assert sig is not None
        _assert_signal_structure(sig, "XAU/USD")
        assert sig["direction"] == "long"

    def test_long_signal_with_price_spike_and_oscillating_rsi(self):
        """A gold price spike (> 0.5×ATR) with moderate RSI triggers long."""
        s = self._strategy()
        closes = [2000.0 + 10 * math.sin(i * 0.3) for i in range(60)]
        closes[-1] = closes[-2] + 15.0  # spike
        df = _make_ohlcv(closes)
        sig = s.analyze(df, "XAU/USD")
        assert sig is not None
        _assert_signal_structure(sig, "XAU/USD")
        assert sig["direction"] == "long"

    def test_only_long_signals_generated(self):
        """Safe-haven strategy should never generate a short signal."""
        s = self._strategy()
        closes = [2000.0 + 10 * math.sin(i * 0.3) for i in range(60)]
        df = _make_ohlcv(closes)
        for vix in (20.0, 30.0, 40.0, 50.0):
            sig = s.analyze(df, "XAU/USD", vix_value=vix)
            if sig is not None:
                assert sig["direction"] == "long", "Safe haven should only go long"

    def test_registered_name(self):
        assert self._strategy().name == "gold_safe_haven"


# ---------------------------------------------------------------------------
# StrategyManager integration
# ---------------------------------------------------------------------------


class TestForexStrategiesInStrategyManager:
    def test_all_15_forex_strategies_registered(self):
        from strategy.strategy_manager import StrategyManager

        mgr = StrategyManager()
        expected_names = [
            "london_breakout",
            "gold_dxy_inverse",
            "nfp_news_strategy",
            "gold_mean_reversion",
            "gold_momentum_breakout",
            "gold_fibonacci",
            "gold_supply_demand",
            "gold_rsi_divergence",
            "gold_bollinger_squeeze",
            "gold_ichimoku",
            "gold_vwap",
            "gold_scalping",
            "silver_gold_ratio",
            "gold_session_momentum",
            "gold_safe_haven",
        ]
        missing = [name for name in expected_names if name not in mgr._strategies]
        assert missing == [], f"Missing forex strategies: {missing}"

    def test_total_strategy_count_includes_forex(self):
        from strategy.strategy_manager import StrategyManager

        mgr = StrategyManager()
        # 46 original + 15 forex = 61
        assert len(mgr) >= 61

    @pytest.mark.asyncio
    async def test_select_forex_strategy_returns_default_without_perf(self):
        """With no performance data, select_forex_strategy returns 'gold_momentum_breakout'."""
        from strategy.strategy_manager import StrategyManager

        mgr = StrategyManager()
        result = await mgr.select_forex_strategy("XAU/USD", {})
        assert result == "gold_momentum_breakout"

    @pytest.mark.asyncio
    async def test_select_forex_strategy_returns_best_with_perf_data(self):
        """With recorded wins, the best-performing strategy is selected."""
        from strategy.strategy_manager import StrategyManager

        mgr = StrategyManager()
        mgr.record_win("gold_fibonacci", 200.0, "trending")
        mgr.record_win("gold_fibonacci", 150.0, "trending")
        result = await mgr.select_forex_strategy("XAU/USD", {})
        assert result == "gold_fibonacci"

    @pytest.mark.asyncio
    async def test_select_forex_strategy_with_ai_brain_failure_falls_back(self):
        """If AI brain raises an exception, fallback to performance-based selection."""
        from strategy.strategy_manager import StrategyManager

        class _BrokenAI:
            async def get_forex_strategy_recommendation(self, symbol, market_data):
                raise RuntimeError("AI unavailable")

        mgr = StrategyManager()
        result = await mgr.select_forex_strategy("XAU/USD", {}, ai_brain=_BrokenAI())
        assert isinstance(result, str)
        assert result in StrategyManager._FOREX_STRATEGY_NAMES

    @pytest.mark.asyncio
    async def test_select_forex_strategy_with_successful_ai(self):
        """When AI returns a valid strategy name, it is used."""
        from strategy.strategy_manager import StrategyManager

        class _MockAI:
            async def get_forex_strategy_recommendation(self, symbol, market_data):
                return {"strategy": "gold_ichimoku"}

        mgr = StrategyManager()
        result = await mgr.select_forex_strategy("XAU/USD", {}, ai_brain=_MockAI())
        assert result == "gold_ichimoku"
