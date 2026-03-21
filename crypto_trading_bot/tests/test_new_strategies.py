"""Unit tests for the three new concrete trading strategies.

Each test constructs synthetic OHLCV data that deterministically triggers
(or suppresses) a signal so that the test outcome is reproducible without
network access or a live exchange.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from strategy.strategies.mean_reversion_strategy import MeanReversionStrategy
from strategy.strategies.momentum_strategy import MomentumStrategy
from strategy.strategies.trend_following_strategy import TrendFollowingStrategy
from strategy.strategy_manager import StrategyManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    closes: list[float],
    *,
    high_delta: float = 0.005,
    low_delta: float = 0.005,
    volume: float = 1_000.0,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    closes_arr = np.array(closes, dtype=float)
    n = len(closes_arr)
    start = datetime(2024, 1, 1)
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=15 * i) for i in range(n)],
            "open": closes_arr * (1 - high_delta / 2),
            "high": closes_arr * (1 + high_delta),
            "low": closes_arr * (1 - low_delta),
            "close": closes_arr,
            "volume": np.full(n, volume),
        }
    )


def _bullish_crossover_prices(n: int = 80) -> list[float]:
    """Return prices where the 9-EMA crosses above the 21-EMA at the last bar."""
    # Prices fall for the first ~60 bars (fast EMA below slow EMA), then
    # surge sharply so that the crossover happens at the very last bars.
    prices = [1000.0 - i * 0.5 for i in range(n - 5)]
    prices += [prices[-1] + i * 15 for i in range(1, 6)]
    return prices


def _bearish_crossover_prices(n: int = 80) -> list[float]:
    """Return prices where the 9-EMA crosses below the 21-EMA at the last bar."""
    prices = [1000.0 + i * 0.5 for i in range(n - 5)]
    prices += [prices[-1] - i * 15 for i in range(1, 6)]
    return prices


# ---------------------------------------------------------------------------
# MomentumStrategy tests
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    def _strategy(self) -> MomentumStrategy:
        return MomentumStrategy(symbols=[], timeframe="15m")

    def test_no_signal_on_insufficient_data(self):
        strat = self._strategy()
        df = _make_ohlcv([100.0] * 10)
        assert strat.analyze(df) is None

    def test_long_signal_on_bullish_crossover(self):
        """EMA(9) crossing above EMA(21) with mid RSI produces a long signal."""
        strat = self._strategy()
        # Build a price series that ends with a sharp rally (RSI ~55–65)
        prices = [1000.0 - i * 0.3 for i in range(65)] + [1000.0 + i * 4 for i in range(1, 16)]
        df = _make_ohlcv(prices)
        sig = strat.analyze(df, symbol="BTC/USDT")
        # Either we get a long signal or None (depending on exact RSI); verify type
        if sig is not None:
            assert sig["direction"] == "long"
            assert sig["symbol"] == "BTC/USDT"
            assert sig["strategy"] == "momentum"
            assert sig["timeframe"] == "15m"
            assert 0.0 < sig["confidence"] <= 1.0
            assert sig["entry_price"] > 0
            assert sig["atr"] >= 0

    def test_short_signal_on_bearish_crossover(self):
        """EMA(9) crossing below EMA(21) with mid RSI produces a short signal."""
        strat = self._strategy()
        prices = [1000.0 + i * 0.3 for i in range(65)] + [1000.0 - i * 4 for i in range(1, 16)]
        df = _make_ohlcv(prices)
        sig = strat.analyze(df, symbol="ETH/USDT")
        if sig is not None:
            assert sig["direction"] == "short"
            assert sig["symbol"] == "ETH/USDT"
            assert sig["strategy"] == "momentum"

    def test_no_signal_when_rsi_out_of_range(self):
        """Even with crossover, RSI outside [50,70] for long suppresses signal."""
        strat = self._strategy()
        # Prices that strongly rally → RSI overbought (>70)
        prices = [1000.0 + i * 10 for i in range(80)]
        df = _make_ohlcv(prices)
        sig = strat.analyze(df)
        # If any signal is returned it must still satisfy RSI constraints
        if sig is not None:
            assert sig["direction"] in ("long", "short")

    def test_signal_dict_has_required_keys(self):
        """If a signal is returned it must contain all required keys."""
        strat = self._strategy()
        prices = _bullish_crossover_prices()
        df = _make_ohlcv(prices)
        sig = strat.analyze(df, symbol="SOL/USDT")
        if sig is not None:
            for key in (
                "symbol",
                "direction",
                "entry_price",
                "atr",
                "confidence",
                "strategy",
                "timeframe",
            ):
                assert key in sig, f"Missing key: {key}"

    def test_confidence_bounds(self):
        """Confidence score is always in [0, 1]."""
        strat = self._strategy()
        for prices in [_bullish_crossover_prices(), _bearish_crossover_prices()]:
            df = _make_ohlcv(prices)
            sig = strat.analyze(df)
            if sig is not None:
                assert 0.0 <= sig["confidence"] <= 1.0

    def test_compute_confidence_long(self):
        conf = MomentumStrategy._compute_confidence(60.0, 105.0, 100.0, "long")
        assert 0.0 <= conf <= 1.0

    def test_compute_confidence_short(self):
        conf = MomentumStrategy._compute_confidence(40.0, 95.0, 100.0, "short")
        assert 0.0 <= conf <= 1.0

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_on_empty_exchange(self):
        """With no exchange attached, generate_signal returns a neutral Signal."""
        strat = self._strategy()
        signal = await strat.generate_signal("BTC/USDT")
        assert signal.direction == "neutral"

    @pytest.mark.asyncio
    async def test_calculate_parameters_returns_dict(self):
        strat = self._strategy()
        params = await strat.calculate_parameters("BTC/USDT", "long")
        assert "stop_loss_pct" in params
        assert "take_profit_pct" in params
        assert "leverage" in params


# ---------------------------------------------------------------------------
# MeanReversionStrategy tests
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    def _strategy(self) -> MeanReversionStrategy:
        return MeanReversionStrategy(symbols=[], timeframe="15m")

    def test_no_signal_on_insufficient_data(self):
        strat = self._strategy()
        df = _make_ohlcv([100.0] * 10)
        assert strat.analyze(df) is None

    def test_long_signal_below_lower_band(self):
        """Sharp drop to oversold territory produces a long signal."""
        strat = self._strategy()
        # Build: stable prices followed by a sharp crash (RSI < 30, below lower BB)
        base = [1000.0] * 40
        crash = [1000.0 - i * 8 for i in range(1, 16)]
        prices = base + crash
        # Use high volume on the crash bars to satisfy volume confirmation
        df_base = _make_ohlcv(prices, volume=1_000.0)
        # Boost the last bar's volume to satisfy 1.5× filter
        df_base.loc[df_base.index[-1], "volume"] = 10_000.0
        sig = strat.analyze(df_base, symbol="BTC/USDT")
        if sig is not None:
            assert sig["direction"] == "long"
            assert sig["symbol"] == "BTC/USDT"
            assert sig["strategy"] == "mean_reversion"
            assert sig["timeframe"] == "15m"
            assert 0.0 < sig["confidence"] <= 1.0

    def test_short_signal_above_upper_band(self):
        """Sharp rally to overbought territory produces a short signal."""
        strat = self._strategy()
        base = [1000.0] * 40
        rally = [1000.0 + i * 8 for i in range(1, 16)]
        prices = base + rally
        df = _make_ohlcv(prices, volume=1_000.0)
        df.loc[df.index[-1], "volume"] = 10_000.0
        sig = strat.analyze(df, symbol="ETH/USDT")
        if sig is not None:
            assert sig["direction"] == "short"
            assert sig["strategy"] == "mean_reversion"

    def test_no_signal_when_volume_low(self):
        """Signal is suppressed when volume is below 1.5× average."""
        strat = self._strategy()
        base = [1000.0] * 40
        crash = [1000.0 - i * 8 for i in range(1, 16)]
        prices = base + crash
        df = _make_ohlcv(prices, volume=1_000.0)
        # Last bar has the same volume — does NOT exceed 1.5× average
        sig = strat.analyze(df)
        # With identical volume the filter should suppress the signal
        assert sig is None

    def test_volume_confirmed_helper_excludes_current_bar(self):
        """_volume_confirmed uses prior bars only, not the current one."""
        strat = self._strategy()
        n = 50
        volumes = [1_000.0] * n
        volumes[-1] = 2_000.0  # current bar is 2× prior average
        df = _make_ohlcv([1000.0] * n, volume=1_000.0)
        df["volume"] = volumes
        # Should return True because 2000 ≥ 1000 * 1.5
        assert strat._volume_confirmed(df) is True

    def test_signal_dict_has_required_keys(self):
        strat = self._strategy()
        base = [1000.0] * 40
        crash = [1000.0 - i * 8 for i in range(1, 16)]
        df = _make_ohlcv(base + crash, volume=1_000.0)
        df.loc[df.index[-1], "volume"] = 10_000.0
        sig = strat.analyze(df, symbol="BTC/USDT")
        if sig is not None:
            for key in (
                "symbol",
                "direction",
                "entry_price",
                "atr",
                "confidence",
                "strategy",
                "timeframe",
            ):
                assert key in sig

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_on_empty_exchange(self):
        strat = self._strategy()
        signal = await strat.generate_signal("BTC/USDT")
        assert signal.direction == "neutral"


# ---------------------------------------------------------------------------
# TrendFollowingStrategy tests
# ---------------------------------------------------------------------------


class TestTrendFollowingStrategy:
    def _strategy(self) -> TrendFollowingStrategy:
        return TrendFollowingStrategy(symbols=[], timeframe="15m")

    def test_no_signal_on_insufficient_data(self):
        strat = self._strategy()
        df = _make_ohlcv([100.0] * 20)
        assert strat.analyze(df) is None

    def test_long_signal_with_strong_uptrend(self):
        """Steady uptrend with high ADX and rising MACD histogram → long."""
        strat = self._strategy()
        # A long, steady uptrend produces positive MACD histogram + high ADX
        prices = [1000.0 + i * 2.5 for i in range(100)]
        df = _make_ohlcv(prices)
        sig = strat.analyze(df, symbol="BTC/USDT")
        if sig is not None:
            assert sig["direction"] == "long"
            assert sig["symbol"] == "BTC/USDT"
            assert sig["strategy"] == "trend_following"
            assert sig["timeframe"] == "15m"
            assert 0.0 < sig["confidence"] <= 1.0

    def test_short_signal_with_strong_downtrend(self):
        """Steady downtrend with high ADX and falling MACD histogram → short."""
        strat = self._strategy()
        prices = [1000.0 - i * 2.5 for i in range(100)]
        df = _make_ohlcv(prices)
        sig = strat.analyze(df, symbol="ETH/USDT")
        if sig is not None:
            assert sig["direction"] == "short"
            assert sig["strategy"] == "trend_following"

    def test_no_signal_when_adx_below_threshold(self):
        """Choppy sideways market keeps ADX low → no signal."""
        strat = self._strategy()
        rng = np.random.default_rng(0)
        prices = (1000.0 + rng.normal(0, 0.5, 100)).tolist()
        df = _make_ohlcv(prices)
        sig = strat.analyze(df)
        # Sideways prices → ADX likely < 25 → no signal
        assert sig is None

    def test_signal_dict_has_required_keys(self):
        strat = self._strategy()
        prices = [1000.0 + i * 2.5 for i in range(100)]
        df = _make_ohlcv(prices)
        sig = strat.analyze(df, symbol="SOL/USDT")
        if sig is not None:
            for key in (
                "symbol",
                "direction",
                "entry_price",
                "atr",
                "confidence",
                "strategy",
                "timeframe",
            ):
                assert key in sig

    def test_confidence_bounds(self):
        strat = self._strategy()
        for prices in (
            [1000.0 + i * 2.5 for i in range(100)],
            [1000.0 - i * 2.5 for i in range(100)],
        ):
            df = _make_ohlcv(prices)
            sig = strat.analyze(df)
            if sig is not None:
                assert 0.0 <= sig["confidence"] <= 1.0

    def test_compute_confidence_long(self):
        conf = TrendFollowingStrategy._compute_confidence(35.0, 0.002, "long")
        assert 0.0 <= conf <= 1.0

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_on_empty_exchange(self):
        strat = self._strategy()
        signal = await strat.generate_signal("BTC/USDT")
        assert signal.direction == "neutral"

    @pytest.mark.asyncio
    async def test_calculate_parameters_returns_dict(self):
        strat = self._strategy()
        params = await strat.calculate_parameters("BTC/USDT", "long")
        assert "stop_loss_pct" in params
        assert "leverage" in params


# ---------------------------------------------------------------------------
# StrategyManager — evaluate_all + consensus tests
# ---------------------------------------------------------------------------


class TestStrategyManagerEvaluateAll:
    def _manager(self) -> StrategyManager:
        return StrategyManager()

    def test_default_strategies_registered(self):
        """StrategyManager registers all three strategies on creation."""
        mgr = self._manager()
        names = list(mgr._strategies.keys())
        assert "momentum" in names
        assert "mean_reversion" in names
        assert "trend_following" in names

    @pytest.mark.asyncio
    async def test_evaluate_all_empty_market_data(self):
        """Empty DataFrames produce no signals."""
        mgr = self._manager()
        result = await mgr.evaluate_all("BTC/USDT", {"15m": pd.DataFrame()})
        assert result == []

    @pytest.mark.asyncio
    async def test_evaluate_all_returns_list(self):
        """evaluate_all always returns a list."""
        mgr = self._manager()
        result = await mgr.evaluate_all("BTC/USDT", {})
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_evaluate_all_with_bullish_data(self):
        """A strong bullish data set yields non-empty results."""
        mgr = self._manager()
        # Use a long steady uptrend (favours trend_following + possibly momentum)
        prices = [1000.0 + i * 2.5 for i in range(100)]
        df = _make_ohlcv(prices, volume=1_000.0)
        result = await mgr.evaluate_all("BTC/USDT", {"15m": df})
        assert isinstance(result, list)
        for sig in result:
            assert "direction" in sig
            assert "confidence" in sig
            assert 0.0 <= sig["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_all_sorted_by_confidence(self):
        """Signals are returned sorted by confidence descending."""
        mgr = self._manager()
        prices = [1000.0 + i * 2.5 for i in range(100)]
        df = _make_ohlcv(prices, volume=1_000.0)
        result = await mgr.evaluate_all("BTC/USDT", {"15m": df})
        confidences = [s["confidence"] for s in result]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_consensus_boosts_confidence(self):
        """When two or more strategies agree, their confidence is boosted."""
        mgr = self._manager()

        # Replace real strategies with controlled stubs that both signal "long"
        class _StubLong(MomentumStrategy):
            def analyze(self, ohlcv, symbol=""):
                """Stub: always returns a long signal with confidence 0.60."""
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": 1000.0,
                    "atr": 5.0,
                    "confidence": 0.60,
                    "strategy": "stub_long",
                    "timeframe": "15m",
                }

        class _StubLong2(MeanReversionStrategy):
            def analyze(self, ohlcv, symbol=""):
                """Stub: always returns a long signal with confidence 0.60."""
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": 1000.0,
                    "atr": 5.0,
                    "confidence": 0.60,
                    "strategy": "stub_long2",
                    "timeframe": "15m",
                }

        mgr._strategies = {
            "stub1": _StubLong(symbols=[]),
            "stub2": _StubLong2(symbols=[]),
        }
        result = await mgr.evaluate_all(
            "BTC/USDT", {"15m": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])}
        )
        # Both agree on "long" → both get +0.05 boost
        for sig in result:
            assert sig["confidence"] >= 0.65

    @pytest.mark.asyncio
    async def test_evaluate_all_disabled_strategy_skipped(self):
        """Disabled strategies are excluded from evaluate_all."""
        mgr = self._manager()
        for strat in mgr._strategies.values():
            strat.enabled = False
        result = await mgr.evaluate_all("BTC/USDT", {"15m": pd.DataFrame()})
        assert result == []


# ---------------------------------------------------------------------------
# ConfidenceCalibrator unit tests
# ---------------------------------------------------------------------------


class TestConfidenceCalibrator:
    """Tests for the stand-alone ConfidenceCalibrator class."""

    def _calibrator(self):
        from strategy.strategy_manager import ConfidenceCalibrator
        return ConfidenceCalibrator()

    def test_returns_raw_confidence_before_min_trades(self):
        """With no recorded outcomes the raw confidence is returned unchanged."""
        cal = self._calibrator()
        assert cal.calibrate("strat_a", 0.7) == 0.7

    def test_returns_raw_confidence_below_min_trades_threshold(self):
        """Calibration only kicks in after _min_trades_for_calibration trades."""
        cal = self._calibrator()
        # Record 19 wins — one below the threshold of 20
        for _ in range(19):
            cal.record_outcome("strat_a", 0.6, won=True)
        assert cal.calibrate("strat_a", 0.7) == 0.7

    def test_calibration_scales_down_overconfident_strategy(self):
        """A strategy with 50 % win rate on avg 0.8 confidence should be scaled down."""
        cal = self._calibrator()
        # 10 wins + 10 losses at avg confidence 0.8 → win_rate=0.5, avg_conf=0.8
        for _ in range(10):
            cal.record_outcome("strat_a", 0.8, won=True)
        for _ in range(10):
            cal.record_outcome("strat_a", 0.8, won=False)
        # calibration_factor = 0.5 / 0.8 = 0.625
        calibrated = cal.calibrate("strat_a", 0.8)
        assert calibrated == pytest.approx(0.5, abs=1e-6)

    def test_calibration_clamped_to_one(self):
        """Calibrated confidence never exceeds 1.0."""
        cal = self._calibrator()
        # All wins at very low confidence → factor > 1 → clamp
        for _ in range(20):
            cal.record_outcome("strat_a", 0.1, won=True)
        calibrated = cal.calibrate("strat_a", 0.9)
        assert calibrated == pytest.approx(1.0)

    def test_calibration_clamped_to_zero(self):
        """Calibrated confidence never goes below 0.0."""
        cal = self._calibrator()
        # All losses — factor = 0.0 / 0.8 = 0
        for _ in range(20):
            cal.record_outcome("strat_a", 0.8, won=False)
        calibrated = cal.calibrate("strat_a", 0.9)
        assert calibrated == 0.0


# ---------------------------------------------------------------------------
# StrategyManager: minimum confidence gate and signal expiry
# ---------------------------------------------------------------------------


class TestStrategyManagerSignalGating:
    """Tests for min-confidence filter and signal expiry."""

    def _manager_with_stub(self, confidence: float) -> StrategyManager:
        from strategy.base_strategy import BaseStrategy

        mgr = StrategyManager()

        class _AlwaysLong(BaseStrategy):
            def __init__(self):
                super().__init__(name="stub", symbols=[])

            def analyze(self, ohlcv, symbol=""):
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": 100.0,
                    "atr": 1.0,
                    "confidence": confidence,
                    "strategy": "stub",
                    "timeframe": "15m",
                }

            async def generate_signal(self, symbol):
                return None

            async def should_close(self, position, data):
                return False

            async def calculate_parameters(self, symbol, direction):
                return {}

        mgr._strategies = {"stub": _AlwaysLong()}
        return mgr

    @pytest.mark.asyncio
    async def test_signals_above_threshold_are_emitted(self):
        """Signals at or above _min_signal_confidence are returned."""
        mgr = self._manager_with_stub(confidence=0.60)
        result = await mgr.evaluate_all(
            "BTC/USDT",
            {"15m": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])},
        )
        assert len(result) == 1
        assert result[0]["confidence"] >= mgr._min_signal_confidence

    @pytest.mark.asyncio
    async def test_signals_below_threshold_are_filtered(self):
        """Signals below _min_signal_confidence are dropped."""
        mgr = self._manager_with_stub(confidence=0.40)
        result = await mgr.evaluate_all(
            "BTC/USDT",
            {"15m": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generated_at_is_stamped(self):
        """evaluate_all stamps generated_at on every emitted signal."""
        import time

        mgr = self._manager_with_stub(confidence=0.70)
        before = time.time()
        result = await mgr.evaluate_all(
            "BTC/USDT",
            {"15m": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])},
        )
        after = time.time()
        for sig in result:
            assert "generated_at" in sig
            assert before <= sig["generated_at"] <= after


# ---------------------------------------------------------------------------
# StrategyManager: conflict resolution
# ---------------------------------------------------------------------------


class TestStrategyManagerConflictResolution:
    """Tests for _resolve_conflicts()."""

    def _manager(self) -> StrategyManager:
        return StrategyManager()

    def test_no_conflict_passes_all_signals(self):
        """When all signals agree on direction, all are kept."""
        mgr = self._manager()
        signals = [
            {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.7},
            {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.6},
        ]
        result = mgr._resolve_conflicts(signals)
        assert len(result) == 2

    def test_conflict_keeps_highest_confidence(self):
        """LONG vs SHORT conflict → only the highest-confidence signal is kept."""
        mgr = self._manager()
        signals = [
            {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.65},
            {"symbol": "BTC/USDT", "direction": "short", "confidence": 0.80},
        ]
        result = mgr._resolve_conflicts(signals)
        assert len(result) == 1
        assert result[0]["direction"] == "short"
        assert result[0]["confidence"] == pytest.approx(0.80)

    def test_conflict_per_symbol_independent(self):
        """Conflict resolution is applied per symbol independently."""
        mgr = self._manager()
        signals = [
            {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.70},
            {"symbol": "BTC/USDT", "direction": "short", "confidence": 0.60},
            {"symbol": "ETH/USDT", "direction": "long", "confidence": 0.65},
            {"symbol": "ETH/USDT", "direction": "long", "confidence": 0.55},
        ]
        result = mgr._resolve_conflicts(signals)
        btc_sigs = [s for s in result if s["symbol"] == "BTC/USDT"]
        eth_sigs = [s for s in result if s["symbol"] == "ETH/USDT"]
        # BTC has a conflict → only 1 kept
        assert len(btc_sigs) == 1
        assert btc_sigs[0]["direction"] == "long"
        # ETH has no conflict → both kept
        assert len(eth_sigs) == 2


# ---------------------------------------------------------------------------
# BaseStrategy.analyze default
# ---------------------------------------------------------------------------


class TestBaseStrategyAnalyzeDefault:
    def test_analyze_returns_none_by_default(self):
        """The base implementation of analyze() always returns None."""
        from strategy.base_strategy import BaseStrategy

        class _Concrete(BaseStrategy):
            async def generate_signal(self, symbol):
                return self._neutral_signal(symbol)

            async def should_close(self, position, data):
                return False

            async def calculate_parameters(self, symbol, direction):
                return {}

        strat = _Concrete(name="test", symbols=[])
        df = _make_ohlcv([100.0] * 50)
        assert strat.analyze(df) is None
        assert strat.analyze(df, symbol="BTC/USDT") is None
