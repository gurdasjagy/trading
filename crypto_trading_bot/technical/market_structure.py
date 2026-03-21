"""Market structure analyzer — BOS, CHoCH, swing highs/lows."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class MarketStructureAnalyzer:
    """Analyses market structure using Smart Money Concepts.

    Detects
    -------
    * **Break of Structure (BOS)** — price breaks a prior swing high/low
      *in the direction of the trend*.
    * **Change of Character (CHoCH)** — price breaks a prior swing high/low
      *against the prevailing trend* (first sign of reversal).
    * **Swing Highs / Swing Lows** — local price extremes.
    * **Overall structure** — classified as ``"bullish"``, ``"bearish"``,
      or ``"ranging"``.
    """

    def __init__(self, swing_window: int = 5) -> None:
        self._swing_window = swing_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_bos(self, ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return a list of Break of Structure events.

        Each event dict contains ``index``, ``type`` (``"bullish_bos"`` /
        ``"bearish_bos"``), and ``level``.
        """
        events: List[Dict[str, Any]] = []
        swing_highs = self.find_swing_highs(ohlcv)
        swing_lows = self.find_swing_lows(ohlcv)
        closes = ohlcv["close"].values

        if len(swing_highs) >= 2 and len(closes) > 0:
            prev_high = swing_highs[-2]
            if closes[-1] > prev_high:
                events.append(
                    {
                        "index": len(ohlcv) - 1,
                        "type": "bullish_bos",
                        "level": prev_high,
                    }
                )

        if len(swing_lows) >= 2 and len(closes) > 0:
            prev_low = swing_lows[-2]
            if closes[-1] < prev_low:
                events.append(
                    {
                        "index": len(ohlcv) - 1,
                        "type": "bearish_bos",
                        "level": prev_low,
                    }
                )

        return events

    def detect_choch(self, ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return a list of Change of Character events.

        A CHoCH occurs when price breaks the *opposite* structural extreme —
        a bullish CHoCH breaks a prior swing low in a bearish trend (reversal
        upward), and vice versa.
        """
        events: List[Dict[str, Any]] = []
        structure = self.classify_structure(ohlcv)
        swing_highs = self.find_swing_highs(ohlcv)
        swing_lows = self.find_swing_lows(ohlcv)
        closes = ohlcv["close"].values
        if len(closes) == 0:
            return events

        if structure == "bearish" and swing_highs:
            last_high = swing_highs[-1]
            if closes[-1] > last_high:
                events.append(
                    {
                        "index": len(ohlcv) - 1,
                        "type": "bullish_choch",
                        "level": last_high,
                    }
                )

        if structure == "bullish" and swing_lows:
            last_low = swing_lows[-1]
            if closes[-1] < last_low:
                events.append(
                    {
                        "index": len(ohlcv) - 1,
                        "type": "bearish_choch",
                        "level": last_low,
                    }
                )

        return events

    def find_swing_highs(self, ohlcv: pd.DataFrame) -> List[float]:
        """Return a list of recent swing high prices."""
        highs = ohlcv["high"].values
        w = self._swing_window
        return [
            float(highs[i])
            for i in range(w, len(highs) - w)
            if highs[i] == max(highs[i - w : i + w + 1])
        ]

    def find_swing_lows(self, ohlcv: pd.DataFrame) -> List[float]:
        """Return a list of recent swing low prices."""
        lows = ohlcv["low"].values
        w = self._swing_window
        return [
            float(lows[i])
            for i in range(w, len(lows) - w)
            if lows[i] == min(lows[i - w : i + w + 1])
        ]

    def classify_structure(self, ohlcv: pd.DataFrame) -> str:
        """Return ``"bullish"``, ``"bearish"``, or ``"ranging"``.

        Classification is based on the sequence of swing highs and lows:
        * Bullish: higher highs **and** higher lows.
        * Bearish: lower highs **and** lower lows.
        * Otherwise: ranging.
        """
        swing_highs = self.find_swing_highs(ohlcv)
        swing_lows = self.find_swing_lows(ohlcv)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "ranging"

        hh = swing_highs[-1] > swing_highs[-2]  # higher high
        hl = swing_lows[-1] > swing_lows[-2]  # higher low
        lh = swing_highs[-1] < swing_highs[-2]  # lower high
        ll = swing_lows[-1] < swing_lows[-2]  # lower low

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return "ranging"
