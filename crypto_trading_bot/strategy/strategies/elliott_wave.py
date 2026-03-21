"""Elliott Wave strategy — simplified 5-wave impulse detection with zigzag."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class ElliottWaveStrategy(BaseStrategy):
    """Simplified Elliott Wave impulse strategy.

    Uses a zigzag filter to identify alternating swing points and attempts
    to recognise a completed Wave 2 (in a 5-wave impulse), providing an
    entry at the start of Wave 3 — typically the strongest wave.

    Rules applied
    -------------
    1. Wave 1 is an up-move from the origin.
    2. Wave 2 retraces 38.2%–61.8% of Wave 1.
    3. Entry at the end of Wave 2 (start of Wave 3).
    4. Wave 3 target = Wave 1 length × 1.618 above Wave 1 end.

    Bearish version uses an inverse structure.
    """

    _STRATEGY_NAME = "elliott_wave"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        enabled: bool = True,
        swing_window: int = 8,
        w2_min_retrace: float = 0.382,
        w2_max_retrace: float = 0.618,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._swing_window = swing_window
        self._w2_min = w2_min_retrace
        self._w2_max = w2_max_retrace
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Zigzag swing detection
    # ------------------------------------------------------------------

    @staticmethod
    def _zigzag_swings(
        ohlcv: pd.DataFrame, window: int
    ) -> List[Tuple[int, float, str]]:
        """Return alternating swing high/low points."""
        highs = ohlcv["high"]
        lows = ohlcv["low"]
        swings: List[Tuple[int, float, str]] = []

        for i in range(window, len(ohlcv) - window):
            h = float(highs.iloc[i])
            l = float(lows.iloc[i])
            wh = highs.iloc[i - window: i + window + 1]
            wl = lows.iloc[i - window: i + window + 1]

            is_high = h == float(wh.max())
            is_low = l == float(wl.min())

            if is_high and (not swings or swings[-1][2] != "high"):
                swings.append((i, h, "high"))
            elif is_high and swings[-1][2] == "high" and h > swings[-1][1]:
                swings[-1] = (i, h, "high")
            elif is_low and (not swings or swings[-1][2] != "low"):
                swings.append((i, l, "low"))
            elif is_low and swings[-1][2] == "low" and l < swings[-1][1]:
                swings[-1] = (i, l, "low")

        return swings

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._swing_window * 6 + self._atr_period + 10
        if len(ohlcv) < min_rows:
            return None

        swings = self._zigzag_swings(ohlcv, self._swing_window)
        if len(swings) < 4:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(ohlcv["close"].iloc[-1])

        # Try bullish impulse: look for low → high → low (W0, W1_top, W2_bottom)
        for i in range(len(swings) - 3, -1, -1):
            w0 = swings[i]
            w1 = swings[i + 1] if i + 1 < len(swings) else None
            w2 = swings[i + 2] if i + 2 < len(swings) else None
            if w1 is None or w2 is None:
                break

            # Bullish: W0=low, W1=high, W2=low
            if w0[2] == "low" and w1[2] == "high" and w2[2] == "low":
                w1_len = w1[1] - w0[1]
                if w1_len <= 0:
                    continue
                w2_retrace = (w1[1] - w2[1]) / w1_len
                if self._w2_min <= w2_retrace <= self._w2_max:
                    # W2 end should be close to current price
                    if abs(curr_price - w2[1]) > curr_atr * 2:
                        continue
                    w3_target = w1[1] + w1_len * 1.618
                    retrace_quality = 1.0 - abs(w2_retrace - 0.5) * 2.0
                    confidence = round(min(0.85, 0.6 + retrace_quality * 0.25), 3)
                    return {
                        "symbol": symbol,
                        "direction": "long",
                        "entry_price": curr_price,
                        "atr": curr_atr,
                        "confidence": confidence,
                        "strategy": self._STRATEGY_NAME,
                        "timeframe": self._timeframe,
                        "wave1_start": w0[1],
                        "wave1_end": w1[1],
                        "wave2_end": w2[1],
                        "wave3_target": w3_target,
                        "w2_retrace": w2_retrace,
                    }

            # Bearish: W0=high, W1=low, W2=high
            elif w0[2] == "high" and w1[2] == "low" and w2[2] == "high":
                w1_len = w0[1] - w1[1]
                if w1_len <= 0:
                    continue
                w2_retrace = (w2[1] - w1[1]) / w1_len
                if self._w2_min <= w2_retrace <= self._w2_max:
                    if abs(curr_price - w2[1]) > curr_atr * 2:
                        continue
                    w3_target = w1[1] - w1_len * 1.618
                    retrace_quality = 1.0 - abs(w2_retrace - 0.5) * 2.0
                    confidence = round(min(0.85, 0.6 + retrace_quality * 0.25), 3)
                    return {
                        "symbol": symbol,
                        "direction": "short",
                        "entry_price": curr_price,
                        "atr": curr_atr,
                        "confidence": confidence,
                        "strategy": self._STRATEGY_NAME,
                        "timeframe": self._timeframe,
                        "wave1_start": w0[1],
                        "wave1_end": w1[1],
                        "wave2_end": w2[1],
                        "wave3_target": w3_target,
                        "w2_retrace": w2_retrace,
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
                return self._neutral_signal(symbol, "No Elliott Wave 2 completion")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            w3_target = sig["wave3_target"]
            w2_end = sig["wave2_end"]

            if direction == "long":
                stop_loss = w2_end - atr * 1.0
                take_profit = w3_target
            else:
                stop_loss = w2_end + atr * 1.0
                take_profit = w3_target

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Elliott Wave 3 entry {direction}: "
                    f"W2_retrace={sig['w2_retrace']:.1%}, "
                    f"W3_target={w3_target:.4f}, ATR={atr:.6f}"
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
        if side == "long" and rsi > 75:
            return True
        if side == "short" and rsi < 25:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 3.5),
            "leverage": 2,
        }
