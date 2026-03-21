"""Trend predictor: direction (up/down/sideways) and reversal detection."""

from datetime import datetime, timezone
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel


class TrendPrediction(BaseModel):
    """Result of a trend direction prediction."""

    symbol: str
    timeframe: str
    trend: str  # up / down / sideways
    strength: float  # 0.0 – 1.0
    reversal_likely: bool
    confidence: float  # 0.0 – 1.0
    ema_score: float = 0.0
    momentum_score: float = 0.0
    timestamp: str = ""

    def __init__(self, **data) -> None:  # type: ignore[override]
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)


class TrendPredictor:
    """Predicts trend direction and detects potential reversals.

    Uses EMA crossovers, momentum (ROC), and RSI divergence to assess the
    current trend and estimate the likelihood of a near-term reversal.
    """

    # Minimum bars required for a meaningful trend assessment
    _MIN_BARS: int = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict_trend(
        self,
        symbol: str,
        timeframe: str = "4h",
        prices: Optional[List[float]] = None,
        ohlcv: Optional[Any] = None,
    ) -> TrendPrediction:
        """Predict the trend direction for *symbol*.

        Args:
            symbol:    Trading pair (e.g. ``"BTC/USDT"``).
            timeframe: Chart timeframe string (informational).
            prices:    Optional list of close prices (oldest first).
            ohlcv:     Optional ``pd.DataFrame`` with a ``"close"`` column
                       (takes precedence over *prices* when provided).

        Returns:
            :class:`TrendPrediction`.
        """
        try:
            close_series = self._resolve_prices(prices, ohlcv)
            if close_series is None or len(close_series) < self._MIN_BARS:
                logger.debug(f"Insufficient data for trend prediction: {symbol}")
                return TrendPrediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    trend="sideways",
                    strength=0.0,
                    reversal_likely=False,
                    confidence=0.1,
                )

            ema_score = self._ema_trend_score(close_series)
            momentum = self._momentum_score(close_series)
            combined = ema_score * 0.6 + momentum * 0.4
            combined = max(-1.0, min(1.0, combined))

            trend = "up" if combined > 0.1 else "down" if combined < -0.1 else "sideways"
            strength = abs(combined)
            reversal = self.detect_trend_reversal(symbol, close_series)
            confidence = min(1.0, strength * 0.7 + 0.2)

            return TrendPrediction(
                symbol=symbol,
                timeframe=timeframe,
                trend=trend,
                strength=round(strength, 4),
                reversal_likely=reversal,
                confidence=round(confidence, 4),
                ema_score=round(ema_score, 4),
                momentum_score=round(momentum, 4),
            )
        except Exception as exc:
            logger.warning(f"TrendPredictor.predict_trend error for {symbol}: {exc}")
            return TrendPrediction(
                symbol=symbol,
                timeframe=timeframe,
                trend="sideways",
                strength=0.0,
                reversal_likely=False,
                confidence=0.0,
            )

    def calculate_trend_strength(
        self,
        symbol: str,
        prices: Optional[List[float]] = None,
        ohlcv: Optional[Any] = None,
    ) -> float:
        """Return the trend strength in [0, 1].

        Args:
            symbol: Trading pair (used for logging).
            prices: Optional list of close prices.
            ohlcv:  Optional OHLCV DataFrame.

        Returns:
            Float in [0.0, 1.0]; 0 = no trend, 1 = very strong trend.
        """
        try:
            close = self._resolve_prices(prices, ohlcv)
            if close is None or len(close) < self._MIN_BARS:
                return 0.0
            ema_score = self._ema_trend_score(close)
            momentum = self._momentum_score(close)
            return abs(max(-1.0, min(1.0, ema_score * 0.6 + momentum * 0.4)))
        except Exception as exc:
            logger.warning(f"TrendPredictor.calculate_trend_strength error: {exc}")
            return 0.0

    def detect_trend_reversal(
        self,
        symbol: str,
        prices: Optional[Any] = None,
    ) -> bool:
        """Return ``True`` if a trend reversal appears likely.

        Uses RSI extremes + momentum divergence as reversal signals.

        Args:
            symbol: Trading pair (used for logging).
            prices: Either a list of floats or a pd.Series of close prices.

        Returns:
            ``True`` if a reversal signal is detected.
        """
        try:
            if prices is None:
                return False
            close = pd.Series(prices, dtype=float) if not isinstance(prices, pd.Series) else prices
            if len(close) < 20:
                return False

            # RSI-based reversal: RSI > 75 or < 25 in last 3 bars
            rsi = self._calculate_rsi(close)
            if rsi is not None and (rsi > 75 or rsi < 25):
                return True

            # Momentum divergence: price makes new extreme but momentum doesn't
            last_n = close.tail(20)
            returns = last_n.pct_change().dropna()
            price_trend = float(last_n.iloc[-1]) - float(last_n.iloc[0])
            momentum_trend = float(returns.tail(5).mean()) - float(returns.head(5).mean())

            if (price_trend > 0 and momentum_trend < -0.001) or (
                price_trend < 0 and momentum_trend > 0.001
            ):
                return True

            return False
        except Exception as exc:
            logger.warning(f"TrendPredictor.detect_trend_reversal error: {exc}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_prices(
        prices: Optional[List[float]],
        ohlcv: Optional[Any],
    ) -> Optional[pd.Series]:
        """Return a pd.Series of close prices from whichever source is available."""
        if ohlcv is not None:
            try:
                return ohlcv["close"].astype(float)
            except Exception:
                pass
        if prices is not None:
            return pd.Series(prices, dtype=float)
        return None

    @staticmethod
    def _ema_trend_score(close: pd.Series) -> float:
        """Score the trend via EMA alignment: +1 strong uptrend, -1 strong downtrend."""
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        price = float(close.iloc[-1])
        e9 = float(ema_9.iloc[-1])
        e21 = float(ema_21.iloc[-1])
        e50 = float(ema_50.iloc[-1])

        score = 0.0
        # Each condition contributes ±0.33
        score += 0.33 if price > e9 else -0.33
        score += 0.33 if e9 > e21 else -0.33
        score += 0.34 if e21 > e50 else -0.34
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _momentum_score(close: pd.Series, period: int = 14) -> float:
        """Rate-of-change momentum in [-1, 1]."""
        if len(close) < period + 1:
            return 0.0
        roc = (float(close.iloc[-1]) - float(close.iloc[-period - 1])) / float(
            close.iloc[-period - 1]
        )
        # Normalise: ±10% move → ±1.0
        return max(-1.0, min(1.0, roc * 10))

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
        """Return the most recent RSI value."""
        try:
            delta = close.diff().dropna()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            val = float(rsi.iloc[-1])
            return val if not np.isnan(val) else None
        except Exception:
            return None
