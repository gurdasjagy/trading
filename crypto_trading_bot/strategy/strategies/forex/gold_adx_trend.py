"""Gold ADX Trend Filter strategy — only trades when ADX confirms strong trend."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldAdxTrendStrategy(BaseStrategy):
    """ADX trend filter strategy for gold.

    Requires ADX > 25 to confirm a strong trend.
    Uses +DI/-DI crossover for entry direction.
    """

    _STRATEGY_NAME = "gold_adx_trend"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        adx_period: int = 14,
        adx_strong: float = 25.0,
        adx_very_strong: float = 40.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._adx_period = adx_period
        self._adx_strong = adx_strong
        self._adx_very_strong = adx_very_strong

    def _calculate_adx_full(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Calculate ADX, +DI, -DI."""
        n = self._adx_period
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        length = len(closes)
        if length < n * 2 + 1:
            return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

        plus_dm: List[float] = []
        minus_dm: List[float] = []
        trs: List[float] = []
        for i in range(1, length):
            h_diff = highs[i] - highs[i - 1]
            l_diff = lows[i - 1] - lows[i]
            plus_dm.append(h_diff if h_diff > l_diff and h_diff > 0 else 0.0)
            minus_dm.append(l_diff if l_diff > h_diff and l_diff > 0 else 0.0)
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

        def _wilder(data: List[float], period: int) -> List[float]:
            if len(data) < period:
                return [0.0]
            result = [sum(data[:period])]
            for v in data[period:]:
                result.append(result[-1] - result[-1] / period + v)
            return result

        atr_s = _wilder(trs, n)
        pdi_s = _wilder(plus_dm, n)
        mdi_s = _wilder(minus_dm, n)

        dx_vals: List[float] = []
        for a, p, m in zip(atr_s, pdi_s, mdi_s):
            if a == 0:
                dx_vals.append(0.0)
                continue
            pdi = 100 * p / a
            mdi = 100 * m / a
            denom = pdi + mdi
            dx_vals.append(100 * abs(pdi - mdi) / denom if denom > 0 else 0.0)

        adx = sum(dx_vals[:n]) / n if len(dx_vals) >= n else 0.0
        for v in dx_vals[n:]:
            adx = (adx * (n - 1) + v) / n

        last_idx = -1
        a = atr_s[last_idx]
        plus_di = 100 * pdi_s[last_idx] / a if a > 0 else 0.0
        minus_di = 100 * mdi_s[last_idx] / a if a > 0 else 0.0

        return {"adx": round(adx, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2)}

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < self._adx_period * 2 + 5:
            return self._neutral_signal(symbol)

        adx_data = self._calculate_adx_full(ohlcv)
        adx = adx_data["adx"]
        plus_di = adx_data["plus_di"]
        minus_di = adx_data["minus_di"]

        if adx < self._adx_strong:
            return self._neutral_signal(symbol)

        strength = min(0.5 + (adx - self._adx_strong) / 50, 1.0)
        if adx >= self._adx_very_strong:
            strength = min(strength + 0.1, 1.0)

        if plus_di > minus_di:
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(strength, 3),
                confidence=0.80,
                strategy_name=self.name,
                reasoning=f"ADX={adx:.1f} +DI={plus_di:.1f} -DI={minus_di:.1f}",
            )
        elif minus_di > plus_di:
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(strength, 3),
                confidence=0.80,
                strategy_name=self.name,
                reasoning=f"ADX={adx:.1f} +DI={plus_di:.1f} -DI={minus_di:.1f}",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < self._adx_period * 2 + 5:
            return False
        adx_data = self._calculate_adx_full(ohlcv)
        if adx_data["adx"] < 15:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
