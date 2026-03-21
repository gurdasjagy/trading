"""Regime computation engine — extracted from :mod:`ai.regime_service`.

**Issue 4**: Pure computation class that takes market state as input and
produces regime parameters as output.  No I/O, no persistence, no LLM calls.

The :class:`RegimeComputer` reads market data provided by the
:class:`~core.shared_state_reader.SharedStateReader` (or from a dict) and
returns a :class:`~ai.shared_regime_writer.RegimeData` suitable for writing
to shared memory via :class:`~ai.shared_regime_writer.SharedRegimeWriter`.

Composition
-----------
- Uses ``MarketRegimeDetector`` for single-asset regime classification.
- Uses ``VolatilityAnalyzer`` for volatility regime classification (optional).
- Uses ``CrossAssetRegimeDetector`` for multi-asset correlation (optional).
- Accepts sentiment input from :class:`~ai.sentiment_service.SentimentResult`.

All sub-detectors are optional — if unavailable, conservative defaults are
returned.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from ai.shared_regime_writer import RegimeData


# ═══════════════════════════════════════════════════════════════════════════
# Sentiment result placeholder (mirrors ai.sentiment_service.SentimentResult)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentInput:
    """Lightweight sentiment data used by RegimeComputer.

    Can be populated from :class:`~ai.sentiment_service.SentimentResult`
    or manually for testing.
    """
    score: float = 0.0           # -1.0 (bearish) to +1.0 (bullish)
    confidence: float = 0.0      # 0.0 to 1.0
    fear_greed_index: int = 50   # 0 (extreme fear) to 100 (extreme greed)
    news_impact_score: float = 0.0  # 0.0 to 1.0 (high = disruptive news)


# ═══════════════════════════════════════════════════════════════════════════
# RegimeComputer
# ═══════════════════════════════════════════════════════════════════════════


class RegimeComputer:
    """Pure computation engine for market regime classification.

    Produces :class:`~ai.shared_regime_writer.RegimeData` from market state
    and (optional) sentiment inputs.  Does not perform any I/O — all data
    is passed in via method arguments.

    Parameters
    ----------
    regime_detector :
        ``MarketRegimeDetector`` instance (optional).
    cross_asset_detector :
        ``CrossAssetRegimeDetector`` instance (optional).
    ttl_seconds :
        TTL value embedded in the output :class:`RegimeData`.
    """

    def __init__(
        self,
        regime_detector: Optional[Any] = None,
        cross_asset_detector: Optional[Any] = None,
        ttl_seconds: int = 600,
    ) -> None:
        self._regime_detector = regime_detector
        self._cross_asset_detector = cross_asset_detector
        self._ttl_seconds = ttl_seconds
        self._last_sentiment: SentimentInput = SentimentInput()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_sentiment(self, sentiment: SentimentInput) -> None:
        """Update the cached sentiment data used during regime computation."""
        self._last_sentiment = sentiment
        logger.debug(
            "RegimeComputer: sentiment updated — score={:.2f} "
            "confidence={:.2f} fear_greed={}",
            sentiment.score,
            sentiment.confidence,
            sentiment.fear_greed_index,
        )

    def compute(
        self,
        market_state: Dict[str, Any],
        ohlcv_data: Optional[Any] = None,
        exchange: Optional[Any] = None,
    ) -> RegimeData:
        """Compute regime parameters from the given market state.

        Parameters
        ----------
        market_state :
            Dict as returned by :meth:`SharedStateReader.get_market_summary`.
            Expected keys: ``symbols``, ``total_book_updates``, ``uptime_seconds``, etc.
        ohlcv_data :
            Optional pandas DataFrame for single-asset regime detection
            (columns: ``open``, ``high``, ``low``, ``close``, ``volume``).
        exchange :
            Optional exchange client passed to the cross-asset detector.

        Returns
        -------
        RegimeData
            Fully populated regime data ready for :meth:`SharedRegimeWriter.update`.
        """
        try:
            result = RegimeData(
                ttl_seconds=self._ttl_seconds,
            )

            # 1. Single-asset regime detection (from OHLCV)
            overall, volatility = self._detect_regime(ohlcv_data)
            result.overall_regime = overall
            result.volatility_regime = volatility

            # 2. Cross-asset correlation (optional, async not supported here)
            # Cross-asset detection requires async, so we skip it in pure compute.
            # The ColdPathOrchestrator can call it separately and merge results.

            # 3. Analyse market microstructure from shared memory data
            self._apply_microstructure_signals(result, market_state)

            # 4. Apply sentiment
            result.sentiment_score = self._last_sentiment.score
            result.sentiment_confidence = self._last_sentiment.confidence
            result.fear_greed_index = self._last_sentiment.fear_greed_index
            result.news_impact_score = self._last_sentiment.news_impact_score

            # 5. Compute derived parameters
            result.recommended_position_scale = _compute_position_scale(result)
            result.max_leverage_override = _compute_leverage_override(result) or 0

            # 6. Compute strategy masks from allowed/blocked lists
            allowed = _compute_allowed_strategies(result)
            blocked = _compute_blocked_strategies(result)
            result.allowed_strategies_mask = _strategies_to_mask(allowed)
            result.blocked_strategies_mask = _strategies_to_mask(blocked)

            logger.debug(
                "RegimeComputer: regime={} vol={} scale={:.2f}",
                result.overall_regime,
                result.volatility_regime,
                result.recommended_position_scale,
            )

            return result

        except Exception as exc:
            logger.error("RegimeComputer.compute() failed: {}", exc, exc_info=True)
            return self._safe_default()

    def compute_from_shared_state(
        self,
        market_summary: Dict[str, Any],
    ) -> RegimeData:
        """Convenience method: compute regime from SharedStateReader output.

        Equivalent to ``compute(market_summary)`` but with a clearer name
        for use in the ColdPathOrchestrator.
        """
        return self.compute(market_summary)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_regime(self, ohlcv_data: Optional[Any]) -> tuple[str, str]:
        """Run the MarketRegimeDetector on OHLCV data."""
        if self._regime_detector is None or ohlcv_data is None:
            return "unknown", "moderate"

        try:
            regime_result = self._regime_detector.detect(ohlcv_data)
            if regime_result is not None:
                overall = _map_single_regime(regime_result)
                volatility = _volatility_from_regime(regime_result)
                return overall, volatility
        except Exception as exc:
            logger.warning("RegimeComputer: regime detection failed: {}", exc)

        return "unknown", "moderate"

    def _apply_microstructure_signals(
        self, result: RegimeData, market_state: Dict[str, Any]
    ) -> None:
        """Apply microstructure signals from shared memory to the regime.

        Uses VPIN, imbalance, and Kyle's lambda from the live orderbook
        state to adjust the regime classification.
        """
        symbols = market_state.get("symbols", [])
        if not symbols:
            return

        # Aggregate microstructure signals across all symbols
        vpins = [s.get("vpin", 0.0) for s in symbols if s.get("vpin", 0.0) != 0.0]
        imbalances = [s.get("imbalance", 0.0) for s in symbols]
        spreads = [s.get("spread_bps", 0.0) for s in symbols if s.get("spread_bps", 0.0) > 0]

        # High average VPIN indicates informed trading / potential volatility
        if vpins:
            avg_vpin = statistics.mean(vpins)
            if avg_vpin > 0.7:
                # Elevated toxicity across market
                if result.volatility_regime in ("low", "moderate"):
                    result.volatility_regime = "high"

        # Wide spreads indicate market stress
        if spreads:
            avg_spread = statistics.mean(spreads)
            if avg_spread > 50.0:  # > 50 bps average spread = stressed market
                result.volatility_regime = "high"

        # Strong directional imbalance can confirm trend
        if imbalances:
            avg_imbalance = statistics.mean(imbalances)
            if abs(avg_imbalance) > 0.3 and result.overall_regime == "unknown":
                result.overall_regime = (
                    "trending_bullish" if avg_imbalance > 0 else "trending_bearish"
                )

    @staticmethod
    def _safe_default() -> RegimeData:
        """Return conservative safe-default regime data."""
        return RegimeData(
            overall_regime="unknown",
            volatility_regime="high",
            sentiment_score=0.0,
            sentiment_confidence=0.0,
            fear_greed_index=50,
            btc_dominance_trend="flat",
            funding_rate_bias="neutral",
            cross_asset_correlation=0.0,
            news_impact_score=0.0,
            recommended_position_scale=0.5,
            max_leverage_override=0,
            ttl_seconds=600,
            allowed_strategies_mask=0xFFFFFFFFFFFFFFFF,
            blocked_strategies_mask=0,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pure helper functions (extracted from ai.regime_service)
# ═══════════════════════════════════════════════════════════════════════════


def _map_single_regime(regime_enum: Any) -> str:
    """Convert a ``MarketRegime`` enum value to a simplified regime string."""
    name = getattr(regime_enum, "value", str(regime_enum))
    mapping = {
        "STRONG_UPTREND": "trending_bullish",
        "WEAK_UPTREND": "trending_bullish",
        "STRONG_DOWNTREND": "trending_bearish",
        "WEAK_DOWNTREND": "trending_bearish",
        "RANGING": "ranging",
        "HIGH_VOLATILITY": "high_volatility",
        "LOW_VOLATILITY": "ranging",
        "CRASH": "trending_bearish",
        "UNKNOWN": "unknown",
    }
    return mapping.get(name, "unknown")


def _volatility_from_regime(regime_enum: Any) -> str:
    """Extract the volatility sub-regime string from a ``MarketRegime``."""
    name = getattr(regime_enum, "value", str(regime_enum))
    if name in ("HIGH_VOLATILITY", "CRASH"):
        return "high"
    if name == "LOW_VOLATILITY":
        return "low"
    if name == "UNKNOWN":
        return "moderate"
    return "moderate"


def _compute_position_scale(state: RegimeData) -> float:
    """Compute the recommended position size multiplier.

    Rules (conservative bias):
    - extreme volatility         → 0.0 (no new positions)
    - high volatility / news     → 0.5
    - ranging / low volatility   → 0.75
    - trending                   → 1.0
    - crowded funding            → -0.25
    """
    scale = 1.0

    if state.volatility_regime == "extreme":
        return 0.0
    if state.volatility_regime == "high" or state.news_impact_score > 0.7:
        scale = 0.5
    elif state.volatility_regime == "low" or state.overall_regime == "ranging":
        scale = 0.75
    elif state.overall_regime in ("trending_bullish", "trending_bearish"):
        scale = 1.0

    # Reduce if market is crowded on one side
    if state.funding_rate_bias != "neutral":
        scale = max(0.25, scale - 0.25)

    return round(scale, 2)


def _compute_leverage_override(state: RegimeData) -> Optional[int]:
    """Return a hard leverage cap override if the regime warrants one."""
    if state.volatility_regime == "extreme":
        return 1
    if state.volatility_regime == "high":
        return 3
    if state.news_impact_score > 0.7:
        return 5
    return None


def _compute_allowed_strategies(state: RegimeData) -> List[str]:
    """Return the list of strategy tags that work well in the current regime."""
    if state.overall_regime == "trending_bullish":
        return ["trend_following", "momentum", "breakout", "ema_ribbon"]
    if state.overall_regime == "trending_bearish":
        return ["trend_following", "momentum", "breakout", "short_strategies"]
    if state.overall_regime == "ranging":
        return ["mean_reversion", "market_making", "range_trading", "imbalance_maker"]
    if state.volatility_regime == "high":
        return ["volatility_breakout", "gamma_scalping"]
    return []


def _compute_blocked_strategies(state: RegimeData) -> List[str]:
    """Return the list of strategy tags that should **not** be used now."""
    blocked: list[str] = []
    if state.volatility_regime in ("high", "extreme"):
        blocked.extend(["mean_reversion", "market_making", "imbalance_maker"])
    if state.overall_regime in ("trending_bullish", "trending_bearish"):
        blocked.extend(["mean_reversion", "range_trading"])
    if state.news_impact_score > 0.8:
        blocked.extend(["market_making", "imbalance_maker"])
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for s in blocked:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


# ── Strategy name → bitmask mapping ─────────────────────────────────────

_STRATEGY_BIT_MAP: Dict[str, int] = {
    "trend_following": 1 << 0,
    "momentum": 1 << 1,
    "breakout": 1 << 2,
    "ema_ribbon": 1 << 3,
    "short_strategies": 1 << 4,
    "mean_reversion": 1 << 5,
    "market_making": 1 << 6,
    "range_trading": 1 << 7,
    "imbalance_maker": 1 << 8,
    "volatility_breakout": 1 << 9,
    "gamma_scalping": 1 << 10,
}


def _strategies_to_mask(strategies: List[str]) -> int:
    """Convert a list of strategy tag names to a bitmask."""
    mask = 0
    for name in strategies:
        mask |= _STRATEGY_BIT_MAP.get(name, 0)
    return mask

