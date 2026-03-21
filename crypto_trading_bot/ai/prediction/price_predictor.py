"""Short-term price predictor combining technical, sentiment, and on-chain signals."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import BaseModel


class Prediction(BaseModel):
    """Result of a price prediction."""

    symbol: str
    timeframe: str
    predicted_price: float
    predicted_change_pct: float
    direction: str  # up / down / sideways
    confidence: float  # 0.0 – 1.0
    signal_breakdown: Dict[str, float] = {}
    timestamp: str = ""

    def __init__(self, **data) -> None:  # type: ignore[override]
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)


class PricePredictor:
    """Short-term price predictor that combines multiple signal sources.

    Signal sources (all optional; falls back gracefully when absent):
    - Technical indicators supplied by the caller
    - Sentiment score from the sentiment analyzer
    - On-chain metrics supplied by the caller

    When no signals are available the predictor returns a neutral ``Prediction``
    with near-zero confidence.
    """

    def __init__(
        self,
        sentiment_analyzer=None,
        regime_detector=None,
    ) -> None:
        self._sentiment = sentiment_analyzer
        self._regime_detector = regime_detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict(
        self,
        symbol: str,
        timeframe: str = "1h",
        current_price: float = 0.0,
        technical_signals: Optional[Dict[str, Any]] = None,
        sentiment_score: Optional[float] = None,
        onchain_signals: Optional[Dict[str, Any]] = None,
    ) -> Prediction:
        """Predict the price direction for *symbol* over *timeframe*.

        Args:
            symbol:            Trading pair (e.g. ``"BTC/USDT"``).
            timeframe:         Prediction horizon (e.g. ``"1h"``, ``"4h"``).
            current_price:     Latest close price.
            technical_signals: Dict of technical indicator values (e.g. RSI, MACD).
            sentiment_score:   Pre-computed sentiment score in [-1, 1].
            onchain_signals:   Dict of on-chain metrics (e.g. exchange_inflows).

        Returns:
            :class:`Prediction` object.
        """
        try:
            tech = technical_signals or {}
            onchain = onchain_signals or {}

            # Get sentiment if not provided
            if sentiment_score is None:
                sentiment_score = 0.0

            combined = self._combine_signals(tech, sentiment_score, onchain)

            direction = (
                "up"
                if combined["combined_score"] > 0.1
                else "down" if combined["combined_score"] < -0.1 else "sideways"
            )

            # Simple price move estimate from combined score
            estimated_move_pct = combined["combined_score"] * 2.0  # scale to ±2%

            predicted_price = (
                current_price * (1 + estimated_move_pct / 100) if current_price > 0 else 0.0
            )

            return Prediction(
                symbol=symbol,
                timeframe=timeframe,
                predicted_price=round(predicted_price, 8),
                predicted_change_pct=round(estimated_move_pct, 4),
                direction=direction,
                confidence=round(combined["confidence"], 4),
                signal_breakdown=combined["breakdown"],
            )
        except Exception as exc:
            logger.warning(f"PricePredictor.predict error for {symbol}: {exc}")
            return Prediction(
                symbol=symbol,
                timeframe=timeframe,
                predicted_price=current_price,
                predicted_change_pct=0.0,
                direction="sideways",
                confidence=0.0,
            )

    async def predict_range(
        self,
        symbol: str,
        horizon: str = "24h",
        current_price: float = 0.0,
        technical_signals: Optional[Dict[str, Any]] = None,
        sentiment_score: Optional[float] = None,
    ) -> Dict:
        """Predict a price range over a longer *horizon*.

        Args:
            symbol:            Trading pair.
            horizon:           Horizon string (e.g. ``"24h"``, ``"7d"``).
            current_price:     Latest close price.
            technical_signals: Optional technical indicator dict.
            sentiment_score:   Optional sentiment score.

        Returns:
            Dict with keys ``"low"``, ``"mid"``, ``"high"``, ``"confidence"``.
        """
        try:
            prediction = await self.predict(
                symbol=symbol,
                timeframe=horizon,
                current_price=current_price,
                technical_signals=technical_signals,
                sentiment_score=sentiment_score,
            )

            move = prediction.predicted_change_pct / 100.0
            # Widen the range proportionally for longer horizons
            horizon_mult = 2.0 if "24h" in horizon else (4.0 if "7d" in horizon else 1.5)

            low = current_price * (1 + move - abs(move) * horizon_mult) if current_price else 0.0
            high = current_price * (1 + move + abs(move) * horizon_mult) if current_price else 0.0

            return {
                "low": round(low, 8),
                "mid": round(prediction.predicted_price, 8),
                "high": round(high, 8),
                "confidence": prediction.confidence,
                "direction": prediction.direction,
            }
        except Exception as exc:
            logger.warning(f"PricePredictor.predict_range error for {symbol}: {exc}")
            return {"low": 0.0, "mid": current_price, "high": 0.0, "confidence": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine_signals(
        self,
        technical: Dict[str, Any],
        sentiment: float,
        onchain: Dict[str, Any],
    ) -> Dict:
        """Combine technical, sentiment, and on-chain signals into a single score.

        Returns a dict with ``"combined_score"`` (in [-1, 1]),
        ``"confidence"``, and ``"breakdown"``.
        """
        scores: Dict[str, float] = {}

        # --- Technical signals ---
        rsi = float(technical.get("rsi", 50))
        # RSI: oversold → bullish, overbought → bearish
        rsi_score = -(rsi - 50) / 50  # maps 0→+1, 100→-1
        scores["rsi"] = max(-1.0, min(1.0, rsi_score))

        macd_hist = float(technical.get("macd_histogram", 0.0))
        if macd_hist != 0.0:
            scores["macd"] = max(-1.0, min(1.0, macd_hist / max(abs(macd_hist), 1e-8)))

        bb_position = float(technical.get("bb_position", 0.5))  # 0=lower band, 1=upper band
        scores["bb"] = (0.5 - bb_position) * 2  # lower band → bullish

        # --- Sentiment ---
        if sentiment != 0.0:
            scores["sentiment"] = max(-1.0, min(1.0, sentiment))

        # --- On-chain ---
        exchange_inflow_pct = float(onchain.get("exchange_inflow_pct_change", 0.0))
        if exchange_inflow_pct != 0.0:
            # Rising inflows → bearish (coins being sent to sell)
            scores["onchain_inflow"] = max(-1.0, min(1.0, -exchange_inflow_pct / 20.0))

        if not scores:
            return {"combined_score": 0.0, "confidence": 0.1, "breakdown": {}}

        # Weighted average (technical slightly higher weight)
        weights = {
            "rsi": 1.2,
            "macd": 1.0,
            "bb": 0.8,
            "sentiment": 1.0,
            "onchain_inflow": 0.9,
        }
        total_w = sum(weights.get(k, 1.0) for k in scores)
        combined = sum(scores[k] * weights.get(k, 1.0) for k in scores) / total_w
        combined = max(-1.0, min(1.0, combined))

        confidence = min(1.0, len(scores) * 0.15 + abs(combined) * 0.5)

        return {
            "combined_score": round(combined, 4),
            "confidence": round(confidence, 4),
            "breakdown": {k: round(v, 4) for k, v in scores.items()},
        }
