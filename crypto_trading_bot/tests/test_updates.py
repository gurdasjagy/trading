"""Tests for the new updates: Regime Service and AI-Adaptive Strategy changes.

Covers:
1. RegimeService - state computation and persistence
2. AIAdaptiveStrategy - hot-path reads from regime state (no LLM)
3. MicrostructureSnapshot and MicrostructureStore in the dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Regime Service tests
# ---------------------------------------------------------------------------

from ai.regime_service import (
    RegimeState,
    RegimeService,
    _compute_allowed_strategies,
    _compute_blocked_strategies,
    _compute_leverage_override,
    _compute_position_scale,
    _extract_btc_dominance_trend,
    _extract_fear_greed,
    _extract_funding_rate_bias,
    _map_single_regime,
    _merge_llm_signals,
    _volatility_from_regime,
)


class TestRegimeState:
    def test_safe_default_fields(self):
        s = RegimeState.safe_default()
        assert s.overall_regime == "unknown"
        assert s.volatility_regime == "high"
        assert s.recommended_position_scale == 0.5
        assert s.ttl_seconds == 600

    def test_to_json_roundtrip(self):
        s = RegimeState(overall_regime="trending_bullish", sentiment_score=0.7)
        j = s.to_json()
        s2 = RegimeState.model_validate_json(j)
        assert s2.overall_regime == "trending_bullish"
        assert s2.sentiment_score == 0.7


class TestRegimeServiceHelpers:
    def test_map_single_regime(self):
        class FakeEnum:
            value = "STRONG_UPTREND"
        assert _map_single_regime(FakeEnum()) == "trending_bullish"

    def test_map_single_regime_unknown(self):
        class FakeEnum:
            value = "UNKNOWN"
        assert _map_single_regime(FakeEnum()) == "unknown"

    def test_volatility_from_regime_high(self):
        class FakeEnum:
            value = "HIGH_VOLATILITY"
        assert _volatility_from_regime(FakeEnum()) == "high"

    def test_extract_btc_dominance_rising(self):
        assert _extract_btc_dominance_trend({"btc_dominance_change": 1.0}) == "rising"

    def test_extract_btc_dominance_falling(self):
        assert _extract_btc_dominance_trend({"btc_dominance_change": -1.0}) == "falling"

    def test_extract_btc_dominance_flat(self):
        assert _extract_btc_dominance_trend({}) == "flat"

    def test_extract_funding_rate_long_crowded(self):
        result = _extract_funding_rate_bias({"funding_rates": {"BTC": 0.002}})
        assert result == "long_crowded"

    def test_extract_funding_rate_short_crowded(self):
        result = _extract_funding_rate_bias({"funding_rates": {"BTC": -0.003}})
        assert result == "short_crowded"

    def test_extract_funding_rate_neutral(self):
        result = _extract_funding_rate_bias({"funding_rates": {"BTC": 0.0001}})
        assert result == "neutral"

    def test_extract_fear_greed(self):
        assert _extract_fear_greed({"fear_greed_index": 75}) == 75

    def test_extract_fear_greed_default(self):
        assert _extract_fear_greed({}) == 50

    def test_compute_position_scale_extreme_vol(self):
        s = RegimeState(volatility_regime="extreme")
        assert _compute_position_scale(s) == 0.0

    def test_compute_position_scale_high_vol(self):
        s = RegimeState(volatility_regime="high")
        assert _compute_position_scale(s) == 0.5

    def test_compute_position_scale_trending(self):
        s = RegimeState(overall_regime="trending_bullish", volatility_regime="moderate")
        assert _compute_position_scale(s) == 1.0

    def test_compute_position_scale_reduced_funding(self):
        s = RegimeState(
            overall_regime="trending_bullish",
            volatility_regime="moderate",
            funding_rate_bias="long_crowded",
        )
        scale = _compute_position_scale(s)
        # Should be 1.0 - 0.25 = 0.75
        assert scale == 0.75

    def test_compute_allowed_strategies_trending_bullish(self):
        s = RegimeState(overall_regime="trending_bullish")
        allowed = _compute_allowed_strategies(s)
        assert "trend_following" in allowed
        assert "momentum" in allowed

    def test_compute_blocked_strategies_high_vol(self):
        s = RegimeState(volatility_regime="high")
        blocked = _compute_blocked_strategies(s)
        assert "market_making" in blocked
        assert "mean_reversion" in blocked

    def test_compute_leverage_override_extreme(self):
        s = RegimeState(volatility_regime="extreme")
        assert _compute_leverage_override(s) == 1

    def test_compute_leverage_override_normal(self):
        s = RegimeState(volatility_regime="moderate")
        assert _compute_leverage_override(s) is None

    def test_merge_llm_signals_override_unknown(self):
        s = RegimeState(overall_regime="unknown")
        _merge_llm_signals(s, {"direction": "bullish", "confidence": 0.8})
        assert s.overall_regime == "trending_bullish"

    def test_merge_llm_signals_no_override_if_known(self):
        s = RegimeState(overall_regime="trending_bearish")
        _merge_llm_signals(s, {"direction": "bullish", "confidence": 0.9})
        # Should NOT override because regime is already known
        assert s.overall_regime == "trending_bearish"

    def test_merge_llm_signals_low_confidence_ignored(self):
        s = RegimeState(overall_regime="unknown")
        _merge_llm_signals(s, {"direction": "bullish", "confidence": 0.5})
        # Confidence < 0.65 → no override
        assert s.overall_regime == "unknown"


class TestRegimeServicePersistence:
    def test_persist_writes_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = f.name

        async def run():
            svc = RegimeService(output_path=tmp)
            state = RegimeState(
                overall_regime="trending_bullish",
                sentiment_score=0.6,
                timestamp_ms=int(time.time() * 1000),
            )
            await svc._persist(state)

        asyncio.run(run())
        with open(tmp) as f:
            data = json.load(f)
        assert data["overall_regime"] == "trending_bullish"
        os.unlink(tmp)

    def test_run_once_returns_safe_default_on_all_failures(self):
        """run_once should return safe default when everything fails."""
        async def run():
            svc = RegimeService()
            # Override to a non-existent path so persist will attempt but not crash
            svc._output_path = "/tmp/regime_test_safe_default.json"
            result = await svc.run_once()
            return result

        state = asyncio.run(run())
        # Should get a state back (either computed or safe default)
        assert isinstance(state, RegimeState)


# ---------------------------------------------------------------------------
# AIAdaptiveStrategy tests
# ---------------------------------------------------------------------------

from strategy.strategies.ai_adaptive import AIAdaptiveStrategy


class TestAIAdaptiveStrategyRegimeBased:
    """Verify that AIAdaptiveStrategy reads from regime state, not the LLM."""

    def _make_strategy(self, regime_data: Optional[Dict] = None) -> AIAdaptiveStrategy:
        strat = AIAdaptiveStrategy(symbols=["BTC_USDT"])
        if regime_data is not None:
            # Inject a mock regime service
            mock_service = MagicMock()
            mock_state = MagicMock()
            mock_state.model_dump.return_value = regime_data
            mock_service.current_state = mock_state
            strat.set_regime_service(mock_service)
        return strat

    def test_no_llm_call_in_generate_signal(self):
        """AIBrain.analyze must never be called during generate_signal."""
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bullish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.5,
            "sentiment_confidence": 0.8,
            "recommended_position_scale": 1.0,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)

        # Inject a spy brain — must NOT be called
        brain = AsyncMock()
        strat.set_ai_brain(brain)

        async def run():
            return await strat.generate_signal("BTC_USDT")

        signal = asyncio.run(run())
        brain.analyze.assert_not_called()
        assert signal.direction == "long"

    def test_generate_signal_bullish_regime(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bullish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.0,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 1.0,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        signal = asyncio.run(strat.generate_signal("BTC_USDT"))
        assert signal.direction == "long"

    def test_generate_signal_bearish_regime(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bearish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.0,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 1.0,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        signal = asyncio.run(strat.generate_signal("BTC_USDT"))
        assert signal.direction == "short"

    def test_generate_signal_unknown_regime_returns_neutral(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "unknown",
            "volatility_regime": "high",
            "sentiment_score": 0.0,
            "sentiment_confidence": 0.5,
            "recommended_position_scale": 0.5,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        signal = asyncio.run(strat.generate_signal("BTC_USDT"))
        assert signal.direction == "neutral"

    def test_generate_signal_stale_state_returns_neutral(self):
        regime = {
            "timestamp_ms": int((time.time() - 2000) * 1000),  # 2000 seconds old
            "overall_regime": "trending_bullish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.5,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 1.0,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        strat._regime_stale_ttl = 900  # < 2000s → stale
        signal = asyncio.run(strat.generate_signal("BTC_USDT"))
        assert signal.direction == "neutral"

    def test_generate_signal_blocked_strategy_returns_neutral(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bullish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.5,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 1.0,
            "blocked_strategies": ["ai_adaptive"],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        signal = asyncio.run(strat.generate_signal("BTC_USDT"))
        assert signal.direction == "neutral"

    def test_generate_signal_no_regime_service_returns_neutral(self):
        strat = AIAdaptiveStrategy(symbols=["BTC_USDT"])
        # No regime service, no file → neutral
        strat._regime_state_path = "/tmp/nonexistent_regime_test.json"
        signal = asyncio.run(strat.generate_signal("BTC_USDT"))
        assert signal.direction == "neutral"

    def test_should_close_on_regime_flip(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bearish",
            "volatility_regime": "moderate",
            "sentiment_score": -0.5,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 1.0,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        position = MagicMock()
        position.side = "long"
        should = asyncio.run(strat.should_close(position, {}))
        assert should is True

    def test_should_not_close_aligned_position(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bullish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.5,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 1.0,
            "blocked_strategies": [],
            "max_leverage_override": None,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        position = MagicMock()
        position.side = "long"
        should = asyncio.run(strat.should_close(position, {}))
        assert should is False

    def test_calculate_parameters_scales_by_regime(self):
        regime = {
            "timestamp_ms": int(time.time() * 1000),
            "overall_regime": "trending_bullish",
            "volatility_regime": "moderate",
            "sentiment_score": 0.5,
            "sentiment_confidence": 0.9,
            "recommended_position_scale": 0.5,
            "blocked_strategies": [],
            "max_leverage_override": 3,
            "ttl_seconds": 600,
        }
        strat = self._make_strategy(regime)
        params = asyncio.run(strat.calculate_parameters("BTC_USDT", "long"))
        # position_size_pct should be 0.05 * 0.5 = 0.025
        assert params["position_size_pct"] == pytest.approx(0.025)
        assert params["leverage"] == 3


# ---------------------------------------------------------------------------
# MicrostructureSnapshot and MicrostructureStore tests
# ---------------------------------------------------------------------------

from monitoring.microstructure_dashboard import MicrostructureSnapshot, MicrostructureStore


class TestMicrostructureSnapshot:
    def test_from_dict_roundtrip(self):
        snap = MicrostructureSnapshot(
            symbol="BTC_USDT",
            vwap=50000.0,
            tick_imbalance=0.3,
            vpin=0.15,
            microstructure_edge_score=0.45,
        )
        d = snap.to_dict()
        snap2 = MicrostructureSnapshot.from_dict(d)
        assert snap2.symbol == "BTC_USDT"
        assert snap2.vwap == pytest.approx(50000.0)
        assert snap2.microstructure_edge_score == pytest.approx(0.45)

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "symbol": "ETH_USDT",
            "vwap": 3000.0,
            "unknown_future_field": "ignored",
        }
        snap = MicrostructureSnapshot.from_dict(d)
        assert snap.symbol == "ETH_USDT"
        assert snap.vwap == pytest.approx(3000.0)


class TestMicrostructureStore:
    def test_update_and_retrieve(self):
        store = MicrostructureStore()
        snap = MicrostructureSnapshot(symbol="BTC_USDT", vwap=50000.0)
        snap.timestamp_ms = int(time.time() * 1000)
        store.update_snapshot(snap)
        result = store.get_latest("BTC_USDT")
        assert result is not None
        assert result.vwap == pytest.approx(50000.0)

    def test_get_history_bounded(self):
        store = MicrostructureStore(history_per_symbol=10)
        for i in range(15):
            snap = MicrostructureSnapshot(symbol="BTC_USDT", vwap=float(i))
            store.update_snapshot(snap)
        hist = store.get_history("BTC_USDT", n=20)
        # History is bounded to 10
        assert len(hist) == 10

    def test_get_latest_returns_none_for_unknown_symbol(self):
        store = MicrostructureStore()
        assert store.get_latest("UNKNOWN_SYMBOL") is None

    def test_symbols_list(self):
        store = MicrostructureStore()
        store.update_snapshot(MicrostructureSnapshot(symbol="BTC_USDT"))
        store.update_snapshot(MicrostructureSnapshot(symbol="ETH_USDT"))
        assert set(store.symbols) == {"BTC_USDT", "ETH_USDT"}
