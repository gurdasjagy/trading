"""Harmonic Pattern strategy — simplified Gartley pattern detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# Gartley ratios with tolerance
_GARTLEY_RATIOS = {
    "XAB": (0.618, 0.05),  # B retraces 61.8% of XA
    "ABC": (0.382, 0.10),  # C retraces 38.2%–88.6% of AB (use mid)
    "BCD": (1.272, 0.10),  # D extends 127.2% of BC
    "XAD": (0.786, 0.05),  # D retraces 78.6% of XA
}


class HarmonicPatternStrategy(BaseStrategy):
    """Simplified Gartley harmonic pattern strategy.

    Detects the bullish/bearish Gartley pattern from swing points:
    X → A → B → C → D

    Entry conditions
    ----------------
    * **Long (bullish Gartley)**: D is near the Potential Reversal Zone (PRZ)
      at ~78.6% retracement of XA, with proper internal ratios.
    * **Short (bearish Gartley)**: inverse pattern.
    """

    _STRATEGY_NAME = "harmonic_pattern"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        enabled: bool = True,
        swing_window: int = 10,
        ratio_tolerance: float = 0.05,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._swing_window = swing_window
        self._ratio_tolerance = ratio_tolerance
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Swing-point helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_alternating_swings(
        ohlcv: pd.DataFrame, swing_window: int, n: int = 5
    ) -> Optional[List[Tuple[int, float, str]]]:
        """Return last *n* alternating swing highs/lows as (index, price, type)."""
        highs = ohlcv["high"]
        lows = ohlcv["low"]
        swings: List[Tuple[int, float, str]] = []

        for i in range(swing_window, len(ohlcv) - swing_window):
            h = float(highs.iloc[i])
            l = float(lows.iloc[i])
            window_h = highs.iloc[i - swing_window: i + swing_window + 1]
            window_l = lows.iloc[i - swing_window: i + swing_window + 1]
            if h == float(window_h.max()):
                if not swings or swings[-1][2] != "high":
                    swings.append((i, h, "high"))
                elif h > swings[-1][1]:
                    swings[-1] = (i, h, "high")
            elif l == float(window_l.min()):
                if not swings or swings[-1][2] != "low":
                    swings.append((i, l, "low"))
                elif l < swings[-1][1]:
                    swings[-1] = (i, l, "low")

        return swings[-n:] if len(swings) >= n else None

    @staticmethod
    def _ratio_match(actual: float, target: float, tol: float) -> bool:
        return abs(actual - target) <= tol

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._swing_window * 6 + self._atr_period + 10
        if len(ohlcv) < min_rows:
            return None

        swings = self._find_alternating_swings(ohlcv, self._swing_window, n=5)
        if swings is None or len(swings) < 5:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(ohlcv["close"].iloc[-1])

        # Try both bullish (X=low, A=high, B=low, C=high, D≈current) and bearish
        for bullish in (True, False):
            pts = swings[-5:]
            if bullish:
                # Expect: low, high, low, high, low
                expected_types = ["low", "high", "low", "high", "low"]
            else:
                expected_types = ["high", "low", "high", "low", "high"]

            actual_types = [p[2] for p in pts]
            if actual_types != expected_types:
                continue

            x, a, b, c, d = [p[1] for p in pts]

            xa = abs(a - x)
            ab = abs(b - a)
            bc = abs(c - b)
            cd = abs(d - c)

            if xa == 0 or ab == 0 or bc == 0:
                continue

            xab_ratio = ab / xa
            abc_ratio = bc / ab
            bcd_ratio = cd / bc
            xad_ratio = abs(d - x) / xa

            tol = self._ratio_tolerance
            if not (
                self._ratio_match(xab_ratio, 0.618, tol)
                and self._ratio_match(abc_ratio, 0.618, tol + 0.15)
                and self._ratio_match(bcd_ratio, 1.272, tol + 0.05)
                and self._ratio_match(xad_ratio, 0.786, tol)
            ):
                continue

            # Check if current price is near D (within 1 ATR)
            if abs(curr_price - d) > curr_atr:
                continue

            direction = "long" if bullish else "short"
            # Confidence based on ratio precision
            ratio_error = (
                abs(xab_ratio - 0.618) + abs(xad_ratio - 0.786)
            ) / 2.0
            confidence = round(min(0.85, 0.7 - ratio_error * 2.0), 3)

            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": curr_price,
                "atr": curr_atr,
                "confidence": confidence,
                "strategy": self._STRATEGY_NAME,
                "timeframe": self._timeframe,
                "pattern": "gartley",
                "x": x, "a": a, "b": b, "c": c, "d": d,
            }

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Gartley pattern detected")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["d"] - atr * 1.5
                take_profit = sig["a"]
            else:
                stop_loss = sig["d"] + atr * 1.5
                take_profit = sig["a"]

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gartley {direction}: D={sig['d']:.4f}, "
                    f"target A={sig['a']:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 65:
            return True
        if side == "short" and rsi < 35:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.03,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 3.0),
            "leverage": 2,
        }
