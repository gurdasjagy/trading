"""Tests for the strategy module — signals, indicators, and filters."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from strategy.base_strategy import BaseStrategy, Signal
from strategy.signals.confluence_checker import ConfluenceChecker
from strategy.signals.signal_filter import SignalFilter
from strategy.signals.signal_scorer import SignalScorer

# ── Signal dataclass ──────────────────────────────────────────────────────


class TestBaseStrategySignal:
    def test_signal_creation(self):
        """Signal dataclass can be created with required fields."""
        sig = Signal(
            symbol="BTC/USDT",
            direction="long",
            strength=0.8,
            confidence=0.9,
            strategy_name="test_strategy",
        )
        assert sig.symbol == "BTC/USDT"
        assert sig.direction == "long"
        assert 0.0 <= sig.strength <= 1.0
        assert 0.0 <= sig.confidence <= 1.0

    def test_strength_clamped(self):
        """Signal strength is clamped to [0, 1]."""
        sig = Signal(
            symbol="ETH/USDT",
            direction="short",
            strength=5.0,
            confidence=0.5,
            strategy_name="test",
        )
        assert sig.strength == 1.0

    def test_confidence_clamped(self):
        """Signal confidence is clamped to [0, 1]."""
        sig = Signal(
            symbol="SOL/USDT",
            direction="neutral",
            strength=0.5,
            confidence=-2.0,
            strategy_name="test",
        )
        assert sig.confidence == 0.0

    def test_invalid_direction_raises(self):
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError):
            Signal(
                symbol="BTC/USDT",
                direction="sideways",
                strength=0.5,
                confidence=0.5,
                strategy_name="test",
            )

    def test_default_timestamp(self):
        """Signal gets a default timestamp close to now."""
        before = datetime.now(timezone.utc)
        sig = Signal(
            symbol="BTC/USDT", direction="long", strength=0.5, confidence=0.5, strategy_name="test"
        )
        after = datetime.now(timezone.utc)
        assert before <= sig.timestamp <= after


# ── RSI calculation ───────────────────────────────────────────────────────


class TestRSICalculation:
    def test_rsi_range(self):
        """RSI result is always in [0, 100]."""
        prices = [100 + i * 0.5 for i in range(30)]
        rsi = BaseStrategy._calculate_rsi(prices, period=14)
        assert 0.0 <= rsi <= 100.0

    def test_rsi_overbought_trend(self):
        """Consistently rising prices produce RSI > 70 (overbought)."""
        prices = [100.0 + i * 2.0 for i in range(30)]
        rsi = BaseStrategy._calculate_rsi(prices, period=14)
        assert rsi > 70

    def test_rsi_oversold_trend(self):
        """Consistently falling prices produce RSI < 30 (oversold)."""
        prices = [100.0 - i * 2.0 for i in range(30)]
        rsi = BaseStrategy._calculate_rsi(prices, period=14)
        assert rsi < 30

    def test_rsi_insufficient_data(self):
        """Returns 50.0 when not enough data is available."""
        prices = [100.0, 101.0, 99.0]
        rsi = BaseStrategy._calculate_rsi(prices, period=14)
        assert rsi == 50.0


# ── MACD calculation ──────────────────────────────────────────────────────


class TestMACDCalculation:
    def test_macd_returns_dict_keys(self):
        """MACD returns a dict with macd, signal, histogram keys."""
        prices = [100.0 + i * 0.1 for i in range(50)]
        result = BaseStrategy._calculate_macd(prices)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_macd_insufficient_data(self):
        """Returns zeroed dict when not enough prices are provided."""
        prices = [100.0, 101.0]
        result = BaseStrategy._calculate_macd(prices)
        assert result == {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    def test_macd_histogram_sign(self):
        """Histogram = macd_line - signal_line."""
        prices = [100.0 + i * 0.5 for i in range(60)]
        result = BaseStrategy._calculate_macd(prices)
        assert pytest.approx(result["histogram"], abs=1e-9) == result["macd"] - result["signal"]


# ── ConfluenceChecker ─────────────────────────────────────────────────────


class TestConfluenceCheckerScore:
    def test_score_range(self):
        """Confluence score is in [0.0, max_score]."""
        checker = ConfluenceChecker()
        signals = {
            "technical": "long",
            "sentiment": "long",
        }
        result = checker.check_confluence(signals)
        assert 0.0 <= result.score <= result.max_score

    def test_all_agreeing_high_score(self):
        """All signals pointing the same direction yields a higher score."""
        checker = ConfluenceChecker()
        agreeing = {k: "long" for k in ConfluenceChecker.FACTOR_WEIGHTS}
        conflicting = {
            k: ("short" if i % 2 else "long")
            for i, k in enumerate(ConfluenceChecker.FACTOR_WEIGHTS)
        }
        score_agree = checker.check_confluence(agreeing).score
        score_conflict = checker.check_confluence(conflicting).score
        assert score_agree > score_conflict

    def test_empty_signals_returns_zero(self):
        """Empty signal dict returns 0.0 score."""
        checker = ConfluenceChecker()
        result = checker.check_confluence({})
        assert result.score == 0.0


# ── SignalFilter ──────────────────────────────────────────────────────────


class TestSignalFilter:
    def _make_signal(self, strength: float, confidence: float, direction: str = "long") -> Signal:
        return Signal(
            symbol="BTC/USDT",
            direction=direction,
            strength=strength,
            confidence=confidence,
            strategy_name="test",
        )

    def test_strong_signal_passes(self):
        """A high-strength, high-confidence signal passes the filter."""
        f = SignalFilter()
        sig = self._make_signal(strength=0.8, confidence=0.8)
        result = f.filter_signals([sig], min_strength=0.5, min_confidence=0.5)
        assert sig in result

    def test_weak_signal_rejected(self):
        """A low-strength signal is rejected."""
        f = SignalFilter()
        sig = self._make_signal(strength=0.2, confidence=0.8)
        result = f.filter_signals([sig], min_strength=0.5, min_confidence=0.5)
        assert sig not in result

    def test_neutral_signal_rejected(self):
        """Neutral signals are filtered out."""
        f = SignalFilter()
        sig = self._make_signal(strength=0.5, confidence=0.5, direction="neutral")
        result = f.filter_signals([sig], min_strength=0.0, min_confidence=0.0)
        assert sig not in result


# ── SignalScorer ──────────────────────────────────────────────────────────


class TestSignalScorerRanking:
    def _make_signal(self, symbol: str, strength: float, confidence: float) -> Signal:
        return Signal(
            symbol=symbol,
            direction="long",
            strength=strength,
            confidence=confidence,
            strategy_name="test",
        )

    def test_higher_score_ranked_first(self):
        """Signal with higher composite score is ranked first."""
        scorer = SignalScorer()
        signals = [
            self._make_signal("ETH/USDT", strength=0.4, confidence=0.5),
            self._make_signal("BTC/USDT", strength=0.9, confidence=0.95),
        ]
        ranked = scorer.rank_signals(signals)
        assert ranked[0].symbol == "BTC/USDT"

    def test_empty_list_returns_empty(self):
        """Ranking an empty list returns an empty list."""
        scorer = SignalScorer()
        assert scorer.rank_signals([]) == []
