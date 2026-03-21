"""Gold Asian Range strategy — trade the Asian session range established by gold."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldAsianRangeStrategy(BaseStrategy):
    """Asian Range breakout/fade strategy for gold.

    Identifies the Asian session (00:00–08:00 UTC) range high/low.
    When the London session opens (08:00 UTC), trade breakouts above/below the Asian range.
    Also fades range extremes during the Asian session itself.
    """

    _STRATEGY_NAME = "gold_asian_range"

    ASIAN_START = 0
    ASIAN_END = 8
    LONDON_START = 8
    LONDON_END = 16
    RANGE_FADE_TOLERANCE = 0.002  # tolerance for range fade entries at extremes

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        atr_min_range: float = 0.3,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_min_range = atr_min_range

    async def generate_signal(self, symbol: str) -> Signal:
        now = datetime.now(tz=timezone.utc)
        hour = now.hour
        in_london = self.LONDON_START <= hour < self.LONDON_END
        in_asian = self.ASIAN_START <= hour < self.ASIAN_END

        if not in_london and not in_asian:
            return self._neutral_signal(symbol)

        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < 30:
            return self._neutral_signal(symbol)

        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        atr = self._calculate_atr(ohlcv)

        asian_bars = 32  # 8h × 4 bars/h at 15m
        if len(highs) < asian_bars:
            return self._neutral_signal(symbol)

        asian_high = max(highs[-asian_bars:-1])
        asian_low = min(lows[-asian_bars:-1])
        range_size = asian_high - asian_low

        if atr > 0 and range_size < atr * self._atr_min_range:
            return self._neutral_signal(symbol)

        current = closes[-1]

        if in_london:
            if current > asian_high:
                strength = (
                    min(0.5 + (current - asian_high) / range_size, 0.9) if range_size > 0 else 0.6
                )
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.72,
                    strategy_name=self.name,
                    reasoning=f"London breakout above Asian range high {asian_high:.2f}",
                )
            elif current < asian_low:
                strength = (
                    min(0.5 + (asian_low - current) / range_size, 0.9) if range_size > 0 else 0.6
                )
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.72,
                    strategy_name=self.name,
                    reasoning=f"London breakdown below Asian range low {asian_low:.2f}",
                )
        elif in_asian:
            if current >= asian_high * (1 - self.RANGE_FADE_TOLERANCE):
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=0.50,
                    confidence=0.60,
                    strategy_name=self.name,
                    reasoning=f"Asian range fade at top {asian_high:.2f}",
                )
            elif current <= asian_low * (1 + self.RANGE_FADE_TOLERANCE):
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=0.50,
                    confidence=0.60,
                    strategy_name=self.name,
                    reasoning=f"Asian range fade at bottom {asian_low:.2f}",
                )

        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
