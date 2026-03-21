"""Gold Stochastic Oversold/Overbought reversal strategy."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldStochasticOversoldStrategy(BaseStrategy):
    """Stochastic oscillator reversal strategy for gold.

    Buys when %K crosses above %D below 20 (oversold).
    Sells when %K crosses below %D above 80 (overbought).
    """

    _STRATEGY_NAME = "gold_stochastic_oversold"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._k_period = k_period
        self._d_period = d_period
        self._oversold = oversold
        self._overbought = overbought

    def _calculate_stochastic(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Calculate Stochastic %K and %D."""
        n = self._k_period
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        length = len(closes)
        if length < n + self._d_period:
            return {"k": 50.0, "d": 50.0, "k_prev": 50.0, "d_prev": 50.0}

        k_values: List[float] = []
        for i in range(n - 1, length):
            hh = max(highs[i - n + 1: i + 1])
            ll = min(lows[i - n + 1: i + 1])
            denom = hh - ll
            k_values.append(100 * (closes[i] - ll) / denom if denom > 0 else 50.0)

        def _sma(data: List[float], p: int) -> List[float]:
            return [sum(data[i: i + p]) / p for i in range(len(data) - p + 1)]

        d_values = _sma(k_values, self._d_period)
        if len(k_values) < 2 or len(d_values) < 2:
            return {
                "k": k_values[-1] if k_values else 50.0,
                "d": 50.0,
                "k_prev": 50.0,
                "d_prev": 50.0,
            }

        return {
            "k": round(k_values[-1], 2),
            "d": round(d_values[-1], 2),
            "k_prev": round(k_values[-2], 2),
            "d_prev": round(d_values[-2], 2),
        }

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < self._k_period + self._d_period + 5:
            return self._neutral_signal(symbol)

        st = self._calculate_stochastic(ohlcv)
        k, d = st["k"], st["d"]
        k_prev, d_prev = st["k_prev"], st["d_prev"]

        bullish_cross = k_prev <= d_prev and k > d and k < self._oversold + 10
        bearish_cross = k_prev >= d_prev and k < d and k > self._overbought - 10

        if bullish_cross:
            strength = (self._oversold + 10 - k) / 30
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(max(0.4, min(strength, 0.85)), 3),
                confidence=0.72,
                strategy_name=self.name,
                reasoning=f"Stochastic bullish cross K={k:.1f} D={d:.1f} (oversold)",
            )
        elif bearish_cross:
            strength = (k - (self._overbought - 10)) / 30
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(max(0.4, min(strength, 0.85)), 3),
                confidence=0.72,
                strategy_name=self.name,
                reasoning=f"Stochastic bearish cross K={k:.1f} D={d:.1f} (overbought)",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < self._k_period + self._d_period + 5:
            return False
        st = self._calculate_stochastic(ohlcv)
        k, d = st["k"], st["d"]
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and k > self._overbought and k < d:
                return True
            if side_val == "short" and k < self._oversold and k > d:
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
