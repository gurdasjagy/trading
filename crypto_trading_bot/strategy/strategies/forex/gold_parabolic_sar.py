"""Gold Parabolic SAR trend strategy."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldParabolicSarStrategy(BaseStrategy):
    """Parabolic SAR trend strategy for gold.

    Long when SAR is below price (uptrend).
    Short when SAR is above price (downtrend).
    """

    _STRATEGY_NAME = "gold_parabolic_sar"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._accel = acceleration
        self._max_accel = maximum

    def _calculate_sar(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Parabolic SAR."""
        highs = ohlcv["high"].values.tolist()
        lows = ohlcv["low"].values.tolist()
        n = len(highs)
        if n < 3:
            return {"sar": 0.0, "trend": "neutral"}

        af = self._accel
        ep = highs[0]
        sar = lows[0]
        bull = True

        for i in range(2, n):
            if bull:
                new_sar = sar + af * (ep - sar)
                new_sar = min(new_sar, lows[i - 1], lows[i - 2])
                if lows[i] < new_sar:
                    bull = False
                    new_sar = ep
                    ep = lows[i]
                    af = self._accel
                else:
                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + self._accel, self._max_accel)
            else:
                new_sar = sar + af * (ep - sar)
                new_sar = max(new_sar, highs[i - 1], highs[i - 2])
                if highs[i] > new_sar:
                    bull = True
                    new_sar = ep
                    ep = highs[i]
                    af = self._accel
                else:
                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + self._accel, self._max_accel)
            sar = new_sar

        close = ohlcv["close"].values[-1]
        return {
            "sar": round(sar, 4),
            "trend": "up" if bull else "down",
            "distance_pct": abs(close - sar) / close * 100 if close > 0 else 0.0,
        }

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < 10:
            return self._neutral_signal(symbol)

        sar_data = self._calculate_sar(ohlcv)
        rsi = self._calculate_rsi(ohlcv["close"].tolist())
        distance_pct = sar_data.get("distance_pct", 0.0)

        if distance_pct < 0.05:
            return self._neutral_signal(symbol)

        strength = min(0.5 + distance_pct / 2, 0.9)

        if sar_data["trend"] == "up" and rsi < 72:
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(strength, 3),
                confidence=0.72,
                strategy_name=self.name,
                reasoning=f"SAR below price ({sar_data['sar']:.2f}), dist={distance_pct:.2f}%",
            )
        elif sar_data["trend"] == "down" and rsi > 28:
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(strength, 3),
                confidence=0.72,
                strategy_name=self.name,
                reasoning=f"SAR above price ({sar_data['sar']:.2f}), dist={distance_pct:.2f}%",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < 10:
            return False
        sar_data = self._calculate_sar(ohlcv)
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and sar_data["trend"] == "down":
                return True
            if side_val == "short" and sar_data["trend"] == "up":
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
