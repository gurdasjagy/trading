"""Gold Donchian Channel breakout strategy."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldDonchianChannelStrategy(BaseStrategy):
    """Donchian Channel breakout strategy for gold.

    Upper band = highest high over N periods.
    Lower band = lowest low over N periods.
    Entry on close above upper (long) or close below lower (short).
    """

    _STRATEGY_NAME = "gold_donchian_channel"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        period: int = 20,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._period = period
        self._atr_period = atr_period

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=self._period + 10)
        if len(ohlcv) < self._period + 2:
            return self._neutral_signal(symbol)

        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values

        upper = max(highs[-self._period - 1:-1])
        lower = min(lows[-self._period - 1:-1])
        current = closes[-1]

        if current > upper:
            channel_width = upper - lower
            if channel_width > 0:
                strength = min(0.5 + (current - upper) / channel_width * 0.5, 1.0)
            else:
                strength = 0.6
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(strength, 3),
                confidence=0.70,
                strategy_name=self.name,
                reasoning=f"Donchian breakout above upper {upper:.2f} (price={current:.2f})",
            )
        elif current < lower:
            channel_width = upper - lower
            if channel_width > 0:
                strength = min(0.5 + (lower - current) / channel_width * 0.5, 1.0)
            else:
                strength = 0.6
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(strength, 3),
                confidence=0.70,
                strategy_name=self.name,
                reasoning=f"Donchian breakdown below lower {lower:.2f} (price={current:.2f})",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=self._period + 10)
        if len(ohlcv) < self._period + 2:
            return False
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        middle = (max(highs[-self._period:]) + min(lows[-self._period:])) / 2
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and closes[-1] < middle:
                return True
            if side_val == "short" and closes[-1] > middle:
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
