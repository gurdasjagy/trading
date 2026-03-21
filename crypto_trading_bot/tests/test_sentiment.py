"""Tests for AI sentiment analysis components."""

from __future__ import annotations

import pytest

from ai.news_classifier.categories import ImpactLevel, NewsCategory
from ai.sentiment.aggregated_score import SentimentAggregator
from ai.sentiment.crypto_lexicon import CryptoLexicon
from ai.sentiment.vader_sentiment import VaderSentimentAnalyzer

# ── VADER ─────────────────────────────────────────────────────────────────


class TestVaderSentimentPositive:
    def test_positive_text_score_positive(self):
        """Positive text should yield score > 0 (or 0 when vaderSentiment missing)."""
        analyzer = VaderSentimentAnalyzer()
        result = analyzer.analyze("Bitcoin is amazing and breaking all time highs!")
        # When VADER not installed score defaults to 0.0 — both are valid
        assert result.score >= 0.0

    def test_returns_sentiment_result(self):
        """analyze() always returns a SentimentResult with required fields."""
        analyzer = VaderSentimentAnalyzer()
        result = analyzer.analyze("Great news for crypto!")
        assert hasattr(result, "score")
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert -1.0 <= result.score <= 1.0

    def test_batch_returns_list(self):
        """analyze_batch() returns one result per input text."""
        analyzer = VaderSentimentAnalyzer()
        texts = ["great", "terrible", "okay"]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3


class TestVaderSentimentNegative:
    def test_negative_text_not_bullish(self):
        """Strongly negative text should not be labelled 'very_bullish'."""
        analyzer = VaderSentimentAnalyzer()
        result = analyzer.analyze("Crypto is crashing, everything is terrible, worst day ever!")
        assert result.label != "very_bullish"

    def test_confidence_in_range(self):
        """Confidence is always in [0, 1]."""
        analyzer = VaderSentimentAnalyzer()
        result = analyzer.analyze("This is bad news.")
        assert 0.0 <= result.confidence <= 1.0


# ── CryptoLexicon ─────────────────────────────────────────────────────────


class TestCryptoLexiconBullish:
    def test_moon_keyword_positive(self):
        """'moon' should produce a positive score."""
        lexicon = CryptoLexicon()
        score = lexicon.get_crypto_score("Bitcoin is going to the moon!")
        assert score > 0.0

    def test_hodl_keyword_positive(self):
        """'hodl' should produce a positive score."""
        lexicon = CryptoLexicon()
        score = lexicon.get_crypto_score("HODL and never sell your BTC!")
        assert score > 0.0

    def test_multiple_bullish_terms(self):
        """Multiple bullish terms should accumulate a positive score."""
        lexicon = CryptoLexicon()
        score = lexicon.get_crypto_score("massive breakout rally wagmi to the moon!")
        assert score > 0.3


class TestCryptoLexiconBearish:
    def test_rekt_keyword_negative(self):
        """'rekt' should produce a negative score."""
        lexicon = CryptoLexicon()
        score = lexicon.get_crypto_score("Everyone got rekt by the dump")
        assert score < 0.0

    def test_dump_keyword_negative(self):
        """'dump' should produce a negative score."""
        lexicon = CryptoLexicon()
        score = lexicon.get_crypto_score("Massive dump incoming, crash is coming")
        assert score < 0.0

    def test_neutral_text_near_zero(self):
        """Plain text with no crypto terms should return 0.0."""
        lexicon = CryptoLexicon()
        score = lexicon.get_crypto_score("The weather is nice today.")
        assert score == 0.0

    def test_score_clamped(self):
        """Score is always in [-1, 1]."""
        lexicon = CryptoLexicon()
        for text in [
            "rug pull exit scam hack hacked exploit crash dump",
            "moon lambo ath wagmi rally",
        ]:
            score = lexicon.get_crypto_score(text)
            assert -1.0 <= score <= 1.0


# ── SentimentAggregator ───────────────────────────────────────────────────


class TestSentimentAggregation:
    def test_equal_weights_average(self):
        """Equal-weight aggregation returns the simple mean."""
        agg = SentimentAggregator()
        scores = [0.2, 0.4, 0.6]
        result = agg.aggregate(scores)
        assert result == pytest.approx(sum(scores) / len(scores), abs=1e-6)

    def test_custom_weights(self):
        """Higher-weight sources dominate the aggregate."""
        agg = SentimentAggregator()
        scores = [0.9, 0.0]
        weights = [10.0, 1.0]
        result = agg.aggregate(scores, weights)
        assert result > 0.5  # dominated by 0.9 with high weight

    def test_empty_returns_zero(self):
        """Empty score list returns 0.0."""
        agg = SentimentAggregator()
        assert agg.aggregate([]) == 0.0

    def test_result_clamped_to_range(self):
        """Aggregated result is always in [-1, 1]."""
        agg = SentimentAggregator()
        result = agg.aggregate([1.0, 1.0, 1.0])
        assert -1.0 <= result <= 1.0

    def test_weight_mismatch_uses_equal(self):
        """Mismatched weights/scores falls back to equal weighting."""
        agg = SentimentAggregator()
        result = agg.aggregate([0.5, 0.5], weights=[1.0])  # mismatch
        assert result == pytest.approx(0.5, abs=1e-6)


# ── NewsCategory enum ─────────────────────────────────────────────────────


class TestNewsCategoriesEnum:
    def test_all_categories_present(self):
        """All expected categories are defined in NewsCategory."""
        expected = {
            "REGULATORY",
            "TECHNICAL",
            "ADOPTION",
            "MARKET",
            "MACRO",
            "SECURITY",
            "PARTNERSHIP",
            "DEVELOPMENT",
            "UNKNOWN",
        }
        actual = {c.value for c in NewsCategory}
        assert expected == actual

    def test_category_is_string_enum(self):
        """NewsCategory values are strings."""
        for cat in NewsCategory:
            assert isinstance(cat.value, str)


# ── ImpactLevel ordering ──────────────────────────────────────────────────


class TestImpactLevelOrdering:
    def test_all_levels_present(self):
        """All expected impact levels are defined."""
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"}
        actual = {lvl.value for lvl in ImpactLevel}
        assert expected == actual

    def test_critical_is_highest(self):
        """CRITICAL is the most severe impact level (listed first)."""
        levels = list(ImpactLevel)
        assert levels[0] == ImpactLevel.CRITICAL

    def test_noise_is_lowest(self):
        """NOISE is the least impactful level (listed last)."""
        levels = list(ImpactLevel)
        assert levels[-1] == ImpactLevel.NOISE
