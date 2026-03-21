"""Gold Range Expansion strategy — NR4/NR7 narrow range breakouts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldRangeExpansionStrategy(BaseStrategy):
    """Narrow Range (NR4/NR7) breakout strategy for gold.

    After a period of compression (NR4 or NR7), expect strong expansion.
    Entry when price breaks above/below the narrow-range high/low.
    """

    _STRATEGY_NAME = "gold_range_expansion"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        narrow_period: int = 7,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._nr_period = narrow_period

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=self._nr_period + 20)
        if len(ohlcv) < self._nr_period + 3:
            return self._neutral_signal(symbol)

        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values

        ranges = [highs[i] - lows[i] for i in range(-self._nr_period - 1, -1)]
        current_range = highs[-1] - lows[-1]

        is_nr = current_range <= min(ranges)
        if not is_nr:
            return self._neutral_signal(symbol)

        nr_high = highs[-1]
        nr_low = lows[-1]
        current_close = closes[-1]

        if current_close > nr_high:
            atr = self._calculate_atr(ohlcv)
            breakout_strength = (current_close - nr_high) / atr if atr > 0 else 0.3
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(min(0.5 + breakout_strength, 0.85), 3),
                confidence=0.68,
                strategy_name=self.name,
                reasoning=f"NR{self._nr_period} breakout above {nr_high:.2f}",
            )
        elif current_close < nr_low:
            atr = self._calculate_atr(ohlcv)
            breakout_strength = (nr_low - current_close) / atr if atr > 0 else 0.3
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(min(0.5 + breakout_strength, 0.85), 3),
                confidence=0.68,
                strategy_name=self.name,
                reasoning=f"NR{self._nr_period} breakdown below {nr_low:.2f}",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
