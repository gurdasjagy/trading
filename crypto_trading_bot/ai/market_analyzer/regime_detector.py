"""Market regime detector using ADX, volatility, and EMA trend analysis."""

from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(str, Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGING = "RANGING"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRASH = "CRASH"
    UNKNOWN = "UNKNOWN"


class MarketRegimeDetector:
    """Classifies the current market regime to guide strategy selection.

    Thresholds
    ----------
    - ADX < 15  → ranging / low-volatility
    - ADX 15-25 → weak trend
    - ADX >= 25 → strong trend
    - Hourly σ > 5 % → high volatility
    - 24-bar return < -10 % → crash
    """

    # ADX thresholds
    _ADX_WEAK_THRESHOLD: float = 15.0
    _ADX_STRONG_THRESHOLD: float = 25.0

    # Volatility (σ of hourly returns)
    _VOL_HIGH_THRESHOLD: float = 0.05
    _VOL_LOW_THRESHOLD: float = 0.01

    # Minimum bars required
    _MIN_BARS: int = 50

    def __init__(self) -> None:
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_history: List[MarketRegime] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_regime(self, ohlcv: pd.DataFrame) -> MarketRegime:
        """Detect the market regime from an OHLCV DataFrame.

        The DataFrame must have columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` and at least :attr:`_MIN_BARS` rows.

        Returns :attr:`MarketRegime.UNKNOWN` on insufficient data or errors.
        """
        if ohlcv is None or len(ohlcv) < self._MIN_BARS:
            logger.debug("Insufficient OHLCV data for regime detection")
            return MarketRegime.UNKNOWN

        try:
            adx = self._calculate_adx(ohlcv)
            volatility = self._calculate_volatility(ohlcv)
            trend_score = self._analyze_trend(ohlcv)

            # 24-bar return (crash check)
            lookback = min(24, len(ohlcv) - 1)
            recent_change = (ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-lookback]) / ohlcv[
                "close"
            ].iloc[-lookback]

            if recent_change < -0.10:
                regime = MarketRegime.CRASH
            elif volatility > self._VOL_HIGH_THRESHOLD:
                regime = MarketRegime.HIGH_VOLATILITY
            elif adx < self._ADX_WEAK_THRESHOLD:
                regime = (
                    MarketRegime.LOW_VOLATILITY
                    if volatility < self._VOL_LOW_THRESHOLD
                    else MarketRegime.RANGING
                )
            elif adx >= self._ADX_STRONG_THRESHOLD:
                regime = (
                    MarketRegime.STRONG_UPTREND
                    if trend_score > 0
                    else MarketRegime.STRONG_DOWNTREND
                )
            else:  # weak trend (15 ≤ ADX < 25)
                regime = (
                    MarketRegime.WEAK_UPTREND if trend_score > 0 else MarketRegime.WEAK_DOWNTREND
                )

            self._current_regime = regime
            self._regime_history.append(regime)
            if len(self._regime_history) > 100:
                self._regime_history = self._regime_history[-100:]

            logger.debug(f"Regime detected: {regime.value} (ADX={adx:.1f}, vol={volatility:.3f})")
            return regime

        except Exception as exc:
            logger.warning(f"Regime detection error: {exc}")
            return MarketRegime.UNKNOWN

    # ------------------------------------------------------------------
    # Technical calculations
    # ------------------------------------------------------------------

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Return the ADX value for the last bar.

        Falls back to 20 (neutral) on calculation errors.
        """
        try:
            high: pd.Series = df["high"]
            low: pd.Series = df["low"]
            close: pd.Series = df["close"]

            # Directional Movement
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)

            # Zero out whichever DM is smaller on each bar
            mask = plus_dm >= minus_dm
            plus_dm = plus_dm.where(mask, 0.0)
            minus_dm = minus_dm.where(~mask, 0.0)

            # True Range
            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)

            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

            di_sum = plus_di + minus_di
            dx = (100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)).fillna(0)
            adx = dx.rolling(period).mean()

            return float(adx.iloc[-1])
        except Exception:
            return 20.0  # neutral default

    def _calculate_volatility(self, df: pd.DataFrame, window: int = 24) -> float:
        """Return the rolling standard deviation of hourly returns over *window* bars."""
        try:
            returns = df["close"].pct_change().dropna()
            return float(returns.tail(window).std())
        except Exception:
            return 0.02

    def _analyze_trend(self, df: pd.DataFrame) -> float:
        """Return a trend score: positive = uptrend, negative = downtrend.

        Scores three binary signals (price vs EMA-20, price vs EMA-50,
        EMA-20 vs EMA-50) and centres around zero.
        """
        try:
            close: pd.Series = df["close"]
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()

            current = float(close.iloc[-1])
            e20 = float(ema_20.iloc[-1])
            e50 = float(ema_50.iloc[-1])

            score = 0.0
            if current > e20:
                score += 1.0
            if current > e50:
                score += 1.0
            if e20 > e50:
                score += 1.0

            return score - 1.5  # centred: >0 = uptrend, <0 = downtrend
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> MarketRegime:
        """Most recently detected regime."""
        return self._current_regime

    @property
    def regime_history(self) -> List[MarketRegime]:
        """List of up to 100 most recent regime detections."""
        return list(self._regime_history)
