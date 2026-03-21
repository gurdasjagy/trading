"""Tests for the AI brain sub-modules."""

from __future__ import annotations

import numpy as np
import pytest

from ai.market_analyzer.anomaly_detector import AnomalyDetector
from ai.market_analyzer.volatility_analyzer import VolatilityAnalyzer
from ai.news_classifier.categories import Direction, ImpactLevel, NewsCategory, TimeHorizon
from ai.news_classifier.fake_news_detector import FakeNewsDetector
from ai.news_classifier.impact_scorer import ImpactScorer

# ── NewsClassifier — categories ───────────────────────────────────────────


class TestNewsClassifierCategories:
    def test_all_categories_accessible(self):
        """All NewsCategory enum members can be iterated."""
        cats = list(NewsCategory)
        assert len(cats) >= 5

    def test_regulatory_category_value(self):
        """REGULATORY category value is a string."""
        assert NewsCategory.REGULATORY.value == "REGULATORY"

    def test_security_category_exists(self):
        """SECURITY category is present for hack/exploit news."""
        assert NewsCategory.SECURITY in NewsCategory.__members__.values()

    def test_direction_enum_values(self):
        """Direction enum has BULLISH, BEARISH, NEUTRAL."""
        values = {d.value for d in Direction}
        assert "BULLISH" in values
        assert "BEARISH" in values
        assert "NEUTRAL" in values

    def test_time_horizon_enum_values(self):
        """TimeHorizon enum has expected members."""
        values = {h.value for h in TimeHorizon}
        assert "IMMEDIATE" in values
        assert "SHORT" in values
        assert "LONG" in values


# ── FakeNewsDetector — credibility ────────────────────────────────────────


class TestFakeNewsDetectorCredibility:
    def test_returns_score_in_range(self):
        """Credibility score is always in [0, 1]."""
        detector = FakeNewsDetector()
        score = detector.calculate_source_credibility("cointelegraph.com")
        assert 0.0 <= score <= 1.0

    def test_known_source_higher_than_unknown(self):
        """Known reputable source should have higher credibility than unknown."""
        detector = FakeNewsDetector()
        known = detector.calculate_source_credibility("coindesk.com")
        unknown = detector.calculate_source_credibility("random-crypto-blog-xyz.io")
        assert known >= unknown

    def test_empty_source_returns_default(self):
        """Empty source string returns a default credibility score."""
        detector = FakeNewsDetector()
        score = detector.calculate_source_credibility("")
        assert 0.0 <= score <= 1.0


# ── ImpactScorer — category ───────────────────────────────────────────────


class TestImpactScorerCategory:
    def _make_classification(
        self, category: NewsCategory, direction: Direction, impact: ImpactLevel = ImpactLevel.MEDIUM
    ):
        """Create a NewsClassification object."""
        from ai.news_classifier.categories import NewsClassification, TimeHorizon

        return NewsClassification(
            category=category,
            impact=impact,
            direction=direction,
            time_horizon=TimeHorizon.SHORT,
            confidence=0.8,
        )

    def test_score_in_range(self):
        """Impact score is always in [0, 1]."""
        scorer = ImpactScorer()
        classification = self._make_classification(
            NewsCategory.REGULATORY, Direction.BEARISH, ImpactLevel.HIGH
        )
        score = scorer.score_impact({}, classification)
        assert 0.0 <= score <= 1.0

    def test_critical_higher_than_noise(self):
        """CRITICAL impact scores higher than NOISE for the same category."""
        scorer = ImpactScorer()
        critical = scorer.score_impact(
            {},
            self._make_classification(
                NewsCategory.SECURITY, Direction.BEARISH, ImpactLevel.CRITICAL
            ),
        )
        noise = scorer.score_impact(
            {},
            self._make_classification(NewsCategory.SECURITY, Direction.BEARISH, ImpactLevel.NOISE),
        )
        assert critical >= noise

    def test_security_breach_high_impact(self):
        """A SECURITY / BEARISH / CRITICAL item should have high impact score."""
        scorer = ImpactScorer()
        score = scorer.score_impact(
            {},
            self._make_classification(
                NewsCategory.SECURITY, Direction.BEARISH, ImpactLevel.CRITICAL
            ),
        )
        assert score > 0.1


# ── AnomalyDetector ───────────────────────────────────────────────────────


class TestAnomalyDetectorInit:
    def test_initializes_without_error(self):
        """AnomalyDetector can be instantiated without raising."""
        detector = AnomalyDetector()
        assert detector is not None

    def test_detect_with_normal_data(self):
        """detect_price_anomaly returns a list for normal price data."""
        rng = np.random.default_rng(0)
        prices = list(50_000 + rng.normal(0, 200, 100))
        detector = AnomalyDetector()
        result = detector.detect_price_anomaly(prices)
        assert isinstance(result, list)

    def test_detect_obvious_spike(self):
        """An obvious price spike is flagged as an anomaly."""
        prices = [50_000.0] * 50 + [100_000.0] + [50_000.0] * 49
        detector = AnomalyDetector()
        result = detector.detect_price_anomaly(prices)
        assert len(result) > 0


# ── VolatilityAnalyzer — regime ───────────────────────────────────────────


class TestVolatilityAnalyzerRegime:
    def test_classify_returns_string(self):
        """detect_volatility_regime returns a non-empty string."""
        analyzer = VolatilityAnalyzer()
        regime = analyzer.detect_volatility_regime("BTC/USDT", realized_vol=0.5)
        assert isinstance(regime, str)
        assert len(regime) > 0

    def test_low_volatility_regime(self):
        """Very low annualised vol → 'low' regime."""
        analyzer = VolatilityAnalyzer()
        regime = analyzer.detect_volatility_regime("BTC/USDT", realized_vol=0.1)
        assert regime == "low"

    def test_extreme_volatility_regime(self):
        """Very high annualised vol → 'extreme' regime."""
        analyzer = VolatilityAnalyzer()
        regime = analyzer.detect_volatility_regime("BTC/USDT", realized_vol=2.0)
        assert regime == "extreme"

    def test_normal_volatility_regime(self):
        """Mid-range vol → 'medium' or 'high' regime."""
        analyzer = VolatilityAnalyzer()
        regime = analyzer.detect_volatility_regime("BTC/USDT", realized_vol=0.5)
        assert regime in ("medium", "high")  # 50% annual is borderline

    def test_realized_volatility_non_negative(self):
        """Realized volatility is always non-negative."""
        analyzer = VolatilityAnalyzer()
        prices = [100.0 + i * 0.5 for i in range(50)]
        rv = analyzer.calculate_realized_volatility(prices)
        assert rv >= 0.0


# ── LLMClient — analyze_market caching & retry ───────────────────────────────


class TestLLMClientAnalyzeMarket:
    """Tests for LLMClient.analyze_market (retry + cache)."""

    def _make_client(self):
        from ai.llm_client import LLMClient

        return LLMClient(openai_api_key="", anthropic_api_key="")

    @pytest.mark.asyncio
    async def test_returns_dict_on_success(self):
        """analyze_market returns a dict when the underlying query succeeds."""
        from unittest.mock import AsyncMock, patch

        client = self._make_client()
        good_result = {"direction": "bullish", "confidence": 0.7}
        with patch.object(client, "query_json", new=AsyncMock(return_value=good_result)):
            result = await client.analyze_market("test prompt")
        assert result == good_result

    @pytest.mark.asyncio
    async def test_cache_hit_skips_query(self):
        """A second call with the same prompt returns the cached value without calling query_json."""
        from unittest.mock import patch

        client = self._make_client()
        good_result = {"direction": "bearish", "confidence": 0.6}
        call_count = 0

        async def mock_query_json(prompt):
            nonlocal call_count
            call_count += 1
            return good_result

        with patch.object(client, "query_json", side_effect=mock_query_json):
            r1 = await client.analyze_market("same prompt")
            r2 = await client.analyze_market("same prompt")

        assert r1 == good_result
        assert r2 == good_result
        assert call_count == 1  # second call served from cache

    @pytest.mark.asyncio
    async def test_different_prompts_not_shared_in_cache(self):
        """Two different prompts are cached independently."""
        from unittest.mock import patch

        client = self._make_client()
        results = {
            "prompt A": {"direction": "bullish", "confidence": 0.8},
            "prompt B": {"direction": "bearish", "confidence": 0.5},
        }

        async def mock_query_json(prompt):
            return results[prompt]

        with patch.object(client, "query_json", side_effect=mock_query_json):
            r_a = await client.analyze_market("prompt A")
            r_b = await client.analyze_market("prompt B")

        assert r_a["direction"] == "bullish"
        assert r_b["direction"] == "bearish"

    @pytest.mark.asyncio
    async def test_retry_on_error_then_succeeds(self):
        """Retries up to 3 times; succeeds on the third attempt."""
        from unittest.mock import AsyncMock, patch

        client = self._make_client()
        attempt = 0
        good_result = {"direction": "neutral", "confidence": 0.5}

        async def flaky_query(prompt):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise RuntimeError("temporary failure")
            return good_result

        with patch.object(client, "query_json", side_effect=flaky_query):
            with patch("asyncio.sleep", new=AsyncMock()):
                result = await client.analyze_market("retry prompt")

        assert result == good_result
        assert attempt == 3

    @pytest.mark.asyncio
    async def test_returns_error_dict_after_all_retries_fail(self):
        """Returns {'error': ...} when all 3 retries are exhausted."""
        from unittest.mock import AsyncMock, patch

        client = self._make_client()

        async def always_fails(prompt):
            raise RuntimeError("always broken")

        with patch.object(client, "query_json", side_effect=always_fails):
            with patch("asyncio.sleep", new=AsyncMock()):
                result = await client.analyze_market("doomed prompt")

        assert "error" in result


# ── AIBrain — get_confidence_modifier ─────────────────────────────────────────


class TestAIBrainGetConfidenceModifier:
    """Tests for AIBrain.get_confidence_modifier."""

    @pytest.mark.asyncio
    async def test_returns_none_without_llm(self):
        """Returns None silently when no LLM client is wired."""
        from ai.brain import AIBrain

        brain = AIBrain()  # no llm_client
        result = await brain.get_confidence_modifier({"symbols": ["BTC/USDT"]})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_on_success(self):
        """Returns the LLM result dict when analyze_market succeeds."""
        from unittest.mock import AsyncMock, MagicMock

        from ai.brain import AIBrain

        llm = MagicMock()
        ai_result = {"direction": "bullish", "confidence": 0.8}
        llm.analyze_market = AsyncMock(return_value=ai_result)
        brain = AIBrain(llm_client=llm)
        result = await brain.get_confidence_modifier({"symbols": ["BTC/USDT"]})
        assert result == ai_result

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self):
        """Returns None when the LLM call raises an exception."""
        from unittest.mock import AsyncMock, MagicMock

        from ai.brain import AIBrain

        llm = MagicMock()
        llm.analyze_market = AsyncMock(side_effect=RuntimeError("network error"))
        brain = AIBrain(llm_client=llm)
        result = await brain.get_confidence_modifier({"symbols": ["BTC/USDT"]})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_error_in_result(self):
        """Returns None when the LLM returns an error dict."""
        from unittest.mock import AsyncMock, MagicMock

        from ai.brain import AIBrain

        llm = MagicMock()
        llm.analyze_market = AsyncMock(return_value={"error": "All providers failed"})
        brain = AIBrain(llm_client=llm)
        result = await brain.get_confidence_modifier({"symbols": ["BTC/USDT"]})
        assert result is None


# ── TradingEngine — AI confidence modifier ────────────────────────────────────


class TestEngineAIConfidenceModifier:
    """Tests for the AI confidence modifier logic (via ai.brain.apply_ai_confidence_modifier)."""

    def _modifier(self, signal: dict, ai_signal: dict) -> float:
        from ai.brain import apply_ai_confidence_modifier

        return apply_ai_confidence_modifier(signal, ai_signal)

    def test_bullish_ai_boosts_long_signal(self):
        """AI bullish + signal long → +15% confidence."""
        signal = {"direction": "long", "confidence": 0.5}
        ai = {"direction": "bullish"}
        assert self._modifier(signal, ai) == pytest.approx(0.65, abs=1e-3)

    def test_bearish_ai_boosts_short_signal(self):
        """AI bearish + signal short → +15% confidence."""
        signal = {"direction": "short", "confidence": 0.5}
        ai = {"direction": "bearish"}
        assert self._modifier(signal, ai) == pytest.approx(0.65, abs=1e-3)

    def test_bearish_ai_reduces_long_signal(self):
        """AI bearish + signal long → −15% confidence."""
        signal = {"direction": "long", "confidence": 0.5}
        ai = {"direction": "bearish"}
        assert self._modifier(signal, ai) == pytest.approx(0.35, abs=1e-3)

    def test_bullish_ai_reduces_short_signal(self):
        """AI bullish + signal short → −15% confidence."""
        signal = {"direction": "short", "confidence": 0.5}
        ai = {"direction": "bullish"}
        assert self._modifier(signal, ai) == pytest.approx(0.35, abs=1e-3)

    def test_neutral_ai_no_change(self):
        """AI neutral → confidence unchanged."""
        signal = {"direction": "long", "confidence": 0.7}
        ai = {"direction": "neutral"}
        assert self._modifier(signal, ai) == pytest.approx(0.7, abs=1e-3)

    def test_confidence_clamped_at_max(self):
        """Result never exceeds 1.0."""
        signal = {"direction": "long", "confidence": 0.95}
        ai = {"direction": "bullish"}
        assert self._modifier(signal, ai) <= 1.0

    def test_confidence_clamped_at_zero(self):
        """Result never goes below 0.0."""
        signal = {"direction": "long", "confidence": 0.05}
        ai = {"direction": "bearish"}
        assert self._modifier(signal, ai) >= 0.0

    def test_missing_direction_treated_as_neutral(self):
        """Missing 'direction' in ai_signal treated as neutral (no change)."""
        signal = {"direction": "long", "confidence": 0.6}
        ai = {}  # no direction key
        assert self._modifier(signal, ai) == pytest.approx(0.6, abs=1e-3)


# ── PromptEngine — build_market_confidence_prompt ─────────────────────────────


class TestPromptEngineMarketConfidence:
    """Tests for PromptEngine.build_market_confidence_prompt."""

    def test_returns_string(self):
        """build_market_confidence_prompt returns a non-empty string."""
        from ai.prompt_engine import PromptEngine

        prompt = PromptEngine().build_market_confidence_prompt(
            market_overview={"symbols": ["BTC/USDT", "ETH/USDT"]},
            sentiment_score=0.3,
            news_summary="BTC hits new ATH",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_contains_json_keys(self):
        """Prompt instructs the LLM to return direction, confidence, key_levels, risk_assessment."""
        from ai.prompt_engine import PromptEngine

        prompt = PromptEngine().build_market_confidence_prompt(
            market_overview={"symbols": ["BTC/USDT"]},
        )
        assert "direction" in prompt
        assert "confidence" in prompt
        assert "key_levels" in prompt
        assert "risk_assessment" in prompt

    def test_symbols_included_in_prompt(self):
        """Active symbols appear in the prompt."""
        from ai.prompt_engine import PromptEngine

        prompt = PromptEngine().build_market_confidence_prompt(
            market_overview={"symbols": ["SOL/USDT", "XRP/USDT"]},
        )
        assert "SOL/USDT" in prompt
        assert "XRP/USDT" in prompt

    def test_news_summary_truncated(self):
        """Long news summary is present but truncated to 500 chars."""
        from ai.prompt_engine import PromptEngine

        long_news = "X" * 2000
        prompt = PromptEngine().build_market_confidence_prompt(
            market_overview={"symbols": []},
            news_summary=long_news,
        )
        # The 500-char slice appears in the prompt; the full 2000 chars do not
        assert "X" * 500 in prompt
        assert "X" * 501 not in prompt
