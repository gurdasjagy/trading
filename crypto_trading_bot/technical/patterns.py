"""Chart pattern recognizer — candlestick and chart pattern detection."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger


class PatternRecognizer:
    """Detects common candlestick and chart patterns.

    Candlestick patterns operate on individual candles (dicts with
    ``open``, ``high``, ``low``, ``close`` keys) or the last rows of a
    :class:`pandas.DataFrame`.

    Chart patterns (head & shoulders, double top/bottom) operate on a
    full OHLCV DataFrame.
    """

    # ------------------------------------------------------------------
    # Convenience: detect all patterns in one call
    # ------------------------------------------------------------------

    def detect_patterns(self, ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect all supported patterns and return a list of findings."""
        patterns: List[Dict[str, Any]] = []
        if len(ohlcv) < 2:
            return patterns

        # Last candle
        last = ohlcv.iloc[-1].to_dict()
        prev = ohlcv.iloc[-2].to_dict()

        if self.detect_doji(last):
            patterns.append({"pattern": "doji", "candle": -1, "signal": "neutral"})

        if self.detect_hammer(last):
            patterns.append({"pattern": "hammer", "candle": -1, "signal": "bullish"})

        engulf = self.detect_engulfing(prev, last)
        if engulf != "none":
            patterns.append({"pattern": f"engulfing_{engulf}", "candle": -1, "signal": engulf})

        try:
            hs = self.detect_head_shoulders(ohlcv)
            if hs.get("detected"):
                patterns.append(hs)
        except Exception as exc:
            logger.debug(f"Head & shoulders detection skipped: {exc}")

        try:
            dtb = self.detect_double_top_bottom(ohlcv)
            if dtb.get("detected"):
                patterns.append(dtb)
        except Exception as exc:
            logger.debug(f"Double top/bottom detection skipped: {exc}")

        return patterns

    # ------------------------------------------------------------------
    # Candlestick patterns
    # ------------------------------------------------------------------

    def detect_doji(self, candle: Dict[str, float]) -> bool:
        """Return True if the candle is a doji (tiny body)."""
        body = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        if total_range == 0:
            return False
        return (body / total_range) < 0.1

    def detect_hammer(self, candle: Dict[str, float]) -> bool:
        """Return True if the candle is a hammer (long lower wick, small body)."""
        body = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        total_range = candle["high"] - candle["low"]
        if total_range == 0 or body == 0:
            return False
        return lower_wick >= body * 2 and upper_wick <= body * 0.5

    def detect_engulfing(self, prev: Dict[str, float], curr: Dict[str, float]) -> str:
        """Return ``"bullish"``, ``"bearish"``, or ``"none"``."""
        prev_body = prev["close"] - prev["open"]
        curr_body = curr["close"] - curr["open"]

        # Bullish engulfing: previous bearish candle fully engulfed by bullish
        if (
            prev_body < 0
            and curr_body > 0
            and curr["open"] <= prev["close"]
            and curr["close"] >= prev["open"]
        ):
            return "bullish"

        # Bearish engulfing: previous bullish candle fully engulfed by bearish
        if (
            prev_body > 0
            and curr_body < 0
            and curr["open"] >= prev["close"]
            and curr["close"] <= prev["open"]
        ):
            return "bearish"

        return "none"

    # ------------------------------------------------------------------
    # Chart patterns
    # ------------------------------------------------------------------

    def detect_head_shoulders(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Detect a head and shoulders (or inverse) pattern.

        A simplified detection: finds three consecutive swing highs where
        the middle one is the tallest (or three swing lows where the
        middle is the lowest for inverse H&S).
        """
        if len(ohlcv) < 20:
            return {"detected": False}
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values

        # Find local maxima positions
        peaks = [
            i
            for i in range(1, len(highs) - 1)
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]
        ]
        if len(peaks) >= 3:
            left_peak, mid_peak, right_peak = peaks[-3], peaks[-2], peaks[-1]
            if highs[mid_peak] > highs[left_peak] and highs[mid_peak] > highs[right_peak]:
                return {
                    "detected": True,
                    "pattern": "head_and_shoulders",
                    "signal": "bearish",
                    "head_index": mid_peak,
                }

        # Inverse H&S
        troughs = [
            i for i in range(1, len(lows) - 1) if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]
        ]
        if len(troughs) >= 3:
            left_trough, mid_trough, right_trough = troughs[-3], troughs[-2], troughs[-1]
            if lows[mid_trough] < lows[left_trough] and lows[mid_trough] < lows[right_trough]:
                return {
                    "detected": True,
                    "pattern": "inverse_head_and_shoulders",
                    "signal": "bullish",
                    "head_index": mid_trough,
                }

        return {"detected": False}

    def detect_double_top_bottom(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Detect double top or double bottom pattern."""
        if len(ohlcv) < 20:
            return {"detected": False}
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        tolerance = 0.015  # 1.5 % price tolerance

        peaks = [
            i
            for i in range(1, len(highs) - 1)
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]
        ]
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(highs[p1] - highs[p2]) / highs[p1] <= tolerance:
                return {
                    "detected": True,
                    "pattern": "double_top",
                    "signal": "bearish",
                    "level": (highs[p1] + highs[p2]) / 2,
                }

        troughs = [
            i for i in range(1, len(lows) - 1) if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]
        ]
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(lows[t1] - lows[t2]) / lows[t1] <= tolerance:
                return {
                    "detected": True,
                    "pattern": "double_bottom",
                    "signal": "bullish",
                    "level": (lows[t1] + lows[t2]) / 2,
                }

        return {"detected": False}
