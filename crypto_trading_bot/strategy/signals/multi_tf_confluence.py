"""Multi-timeframe confluence scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# Timeframe weights (must sum to 1.0)
_TF_WEIGHTS: Dict[str, float] = {
    "1d": 0.30,
    "4h": 0.25,
    "1h": 0.20,
    "15m": 0.15,
    "5m": 0.10,
}

_SIGNAL_THRESHOLD = 0.6  # minimum weighted score to generate a signal


@dataclass
class TFAnalysis:
    """Per-timeframe trend analysis result."""

    timeframe: str
    direction: int  # +1 bullish, -1 bearish, 0 neutral
    score: float    # weighted contribution
    reason: str = ""


@dataclass
class ConfluenceSignal:
    """Result of multi-timeframe confluence analysis."""

    weighted_score: float
    direction: str          # "long", "short", or "neutral"
    should_trade: bool
    confidence: float
    tf_analyses: List[TFAnalysis] = field(default_factory=list)
    divergences: List[str] = field(default_factory=list)
    reasoning: str = ""


class MultiTimeframeConfluence:
    """Analyses trend direction across multiple timeframes simultaneously.

    Timeframes analysed (with their weights):
        * 1d  → 0.30
        * 4h  → 0.25
        * 1h  → 0.20
        * 15m → 0.15
        * 5m  → 0.10

    A signal is only generated when the absolute weighted score exceeds
    ``signal_threshold`` (default 0.6), ensuring strong confluence.

    Divergences between higher and lower timeframes reduce the final
    confidence score.
    """

    TIMEFRAMES: Tuple[str, ...] = ("1d", "4h", "1h", "15m", "5m")

    def __init__(self, signal_threshold: float = _SIGNAL_THRESHOLD) -> None:
        self._threshold = signal_threshold
        self._tf_weights: Dict[str, float] = dict(_TF_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        market_data: Dict[str, pd.DataFrame],
    ) -> ConfluenceSignal:
        """Run multi-timeframe confluence analysis.

        Args:
            market_data: Mapping of timeframe label to OHLCV DataFrame.
                At least two timeframes should be present for meaningful results.

        Returns:
            :class:`ConfluenceSignal` describing the overall direction and
            confidence, together with per-timeframe breakdowns and any
            divergence warnings.
        """
        tf_analyses: List[TFAnalysis] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for tf in self.TIMEFRAMES:
            df = market_data.get(tf)
            if df is None or df.empty or len(df) < 20:
                logger.debug(f"[MultiTF] No/insufficient data for timeframe {tf!r} — skipping")
                continue

            weight = self._tf_weights.get(tf, 0.0)
            direction, reason = self._score_timeframe(df)
            analysis = TFAnalysis(
                timeframe=tf,
                direction=direction,
                score=direction * weight,
                reason=reason,
            )
            tf_analyses.append(analysis)
            weighted_sum += direction * weight
            total_weight += weight

        if total_weight == 0:
            return ConfluenceSignal(
                weighted_score=0.0,
                direction="neutral",
                should_trade=False,
                confidence=0.0,
                tf_analyses=tf_analyses,
                reasoning="Insufficient timeframe data",
            )

        # Normalise to [-1, +1] range
        normalised_score = weighted_sum / total_weight

        # Detect divergences
        divergences = self._detect_divergences(tf_analyses)

        # Confidence penalty for divergences
        divergence_penalty = len(divergences) * 0.05
        raw_confidence = min(1.0, abs(normalised_score))
        confidence = max(0.0, raw_confidence - divergence_penalty)

        # Determine direction
        if normalised_score > 0:
            direction_label = "long"
        elif normalised_score < 0:
            direction_label = "short"
        else:
            direction_label = "neutral"

        should_trade = abs(normalised_score) >= self._threshold and confidence > 0

        reasoning_parts = [
            f"Weighted score: {normalised_score:+.3f} (threshold ±{self._threshold})"
        ]
        for a in tf_analyses:
            symbol = "↑" if a.direction > 0 else ("↓" if a.direction < 0 else "→")
            reasoning_parts.append(f"  {a.timeframe}: {symbol} ({a.reason})")
        if divergences:
            reasoning_parts.append(f"Divergences: {'; '.join(divergences)}")

        logger.debug(
            "[MultiTF] score={:.3f} dir={} trade={} conf={:.3f} divs={}",
            normalised_score,
            direction_label,
            should_trade,
            confidence,
            len(divergences),
        )

        return ConfluenceSignal(
            weighted_score=normalised_score,
            direction=direction_label,
            should_trade=should_trade,
            confidence=round(confidence, 3),
            tf_analyses=tf_analyses,
            divergences=divergences,
            reasoning="\n".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    # Timeframe scoring
    # ------------------------------------------------------------------

    def _score_timeframe(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Determine the trend direction for a single timeframe.

        Uses a combination of EMA crossover and price-relative-to-EMA
        signals to produce a +1 / 0 / -1 score.

        Returns:
            Tuple of (direction_int, reason_string).
        """
        closes = df["close"].astype(float)

        if len(closes) < 50:
            return 0, "insufficient data"

        ema_fast = closes.ewm(span=20, adjust=False).mean()
        ema_slow = closes.ewm(span=50, adjust=False).mean()

        last_fast = float(ema_fast.iloc[-1])
        last_slow = float(ema_slow.iloc[-1])
        last_close = float(closes.iloc[-1])

        # EMA crossover
        ema_cross = 1 if last_fast > last_slow else (-1 if last_fast < last_slow else 0)

        # Price vs. EMA
        price_vs_ema = 1 if last_close > last_fast else (-1 if last_close < last_fast else 0)

        # Recent momentum: last 5 candles
        momentum = 0
        if len(closes) >= 5:
            delta = closes.iloc[-1] - closes.iloc[-5]
            if delta > 0:
                momentum = 1
            elif delta < 0:
                momentum = -1

        # Aggregate: majority vote
        votes = ema_cross + price_vs_ema + momentum
        if votes >= 2:
            direction = 1
            reason = f"EMA_cross={ema_cross} price_vs_ema={price_vs_ema} mom={momentum}"
        elif votes <= -2:
            direction = -1
            reason = f"EMA_cross={ema_cross} price_vs_ema={price_vs_ema} mom={momentum}"
        else:
            direction = 0
            reason = f"mixed (votes={votes})"

        return direction, reason

    # ------------------------------------------------------------------
    # Divergence detection
    # ------------------------------------------------------------------

    def _detect_divergences(self, analyses: List[TFAnalysis]) -> List[str]:
        """Identify conflicting trend directions between adjacent timeframes.

        A divergence is flagged when a lower timeframe is bullish while the
        adjacent higher timeframe is bearish (or vice versa).
        """
        divergences: List[str] = []
        by_tf: Dict[str, TFAnalysis] = {a.timeframe: a for a in analyses}

        pairs: List[Tuple[str, str]] = [
            ("1d", "4h"),
            ("4h", "1h"),
            ("1h", "15m"),
            ("15m", "5m"),
        ]
        for higher, lower in pairs:
            h = by_tf.get(higher)
            lo = by_tf.get(lower)
            if h is None or lo is None:
                continue
            if h.direction != 0 and lo.direction != 0 and h.direction != lo.direction:
                divergences.append(
                    f"{higher} {'bullish' if h.direction > 0 else 'bearish'} "
                    f"vs {lower} {'bullish' if lo.direction > 0 else 'bearish'}"
                )

        return divergences

    # ------------------------------------------------------------------
    # Strategy manager integration helper
    # ------------------------------------------------------------------

    def evaluate_for_signal(
        self,
        market_data: Dict[str, pd.DataFrame],
        base_confidence: float = 0.5,
    ) -> Optional[Dict[str, object]]:
        """Return a signal dict compatible with StrategyManager.evaluate_all output.

        Args:
            market_data: Timeframe → OHLCV mapping.
            base_confidence: Base confidence to blend with the confluence result.

        Returns:
            A signal dict or ``None`` if confluence is insufficient.
        """
        result = self.analyze(market_data)
        if not result.should_trade or result.direction == "neutral":
            return None

        blended_confidence = round(
            (base_confidence + result.confidence) / 2.0, 3
        )
        return {
            "direction": result.direction,
            "confidence": blended_confidence,
            "strategy": "multi_tf_confluence",
            "reasoning": result.reasoning,
            "weighted_score": result.weighted_score,
            "divergences": result.divergences,
        }
