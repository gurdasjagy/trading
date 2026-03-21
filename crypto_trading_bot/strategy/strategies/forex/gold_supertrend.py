"""Gold Supertrend strategy — trend following with dynamic ATR-based bands."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldSupertrendStrategy(BaseStrategy):
    """Supertrend indicator strategy for gold.

    Uses ATR-based upper/lower bands to identify trend direction.
    Long when price above Supertrend line; short when below.
    """

    _STRATEGY_NAME = "gold_supertrend"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        atr_period: int = 10,
        multiplier: float = 3.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_period = atr_period
        self._multiplier = multiplier

    def _calculate_supertrend(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Supertrend indicator."""
        n = self._atr_period
        mult = self._multiplier
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        length = len(closes)
        if length < n + 2:
            return {"trend": "neutral", "value": closes[-1] if length > 0 else 0}

        trs = [max(highs[i] - lows[i],
                   abs(highs[i] - closes[i - 1]),
                   abs(lows[i] - closes[i - 1])) for i in range(1, length)]
        atr = sum(trs[:n]) / n
        for tr in trs[n:]:
            atr = (atr * (n - 1) + tr) / n

        hl2 = (highs[-1] + lows[-1]) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        prev_close = closes[-2]
        curr_close = closes[-1]

        if curr_close > upper:
            trend = "up"
            value = lower
        elif curr_close < lower:
            trend = "down"
            value = upper
        elif prev_close >= hl2:
            trend = "up"
            value = lower
        else:
            trend = "down"
            value = upper

        return {"trend": trend, "value": round(value, 2)}

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < self._atr_period + 5:
            return self._neutral_signal(symbol)

        st = self._calculate_supertrend(ohlcv)
        rsi = self._calculate_rsi(ohlcv["close"].tolist())

        if st["trend"] == "up" and rsi < 70:
            strength = min(0.5 + (rsi - 50) / 100, 1.0) if rsi > 50 else 0.5
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(max(0.4, strength), 3),
                confidence=0.75,
                strategy_name=self.name,
                reasoning=f"Supertrend UP @ {st['value']:.2f}, RSI={rsi:.1f}",
            )
        elif st["trend"] == "down" and rsi > 30:
            strength = min(0.5 + (50 - rsi) / 100, 1.0) if rsi < 50 else 0.5
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(max(0.4, strength), 3),
                confidence=0.75,
                strategy_name=self.name,
                reasoning=f"Supertrend DOWN @ {st['value']:.2f}, RSI={rsi:.1f}",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        if len(ohlcv) < self._atr_period + 2:
            return False
        st = self._calculate_supertrend(ohlcv)
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and st["trend"] == "down":
                return True
            if side_val == "short" and st["trend"] == "up":
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {
            "atr": round(atr, 4),
            "stop_loss_pips": round(atr / 0.01 * 1.5, 1),
            "take_profit_pips": round(atr / 0.01 * 2.5, 1),
            "leverage": 20,
        }
