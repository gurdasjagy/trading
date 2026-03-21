"""Gold London Fix strategy — trade around the AM/PM gold fix times."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldLondonFixStrategy(BaseStrategy):
    """London Gold Fix trading strategy.

    The London Bullion Market Association (LBMA) gold fix occurs at:
    - AM Fix: 10:30 London time (09:30 UTC)
    - PM Fix: 15:00 London time (14:00 UTC in summer / 15:00 in winter)

    Positions often move strongly towards the fix and then reverse.
    This strategy trades the 30-minute window BEFORE the fix.
    """

    _STRATEGY_NAME = "gold_london_fix"

    AM_FIX_HOUR = 9
    AM_FIX_MINUTE = 30
    PM_FIX_HOUR = 14
    PM_FIX_MINUTE = 0

    ENTRY_WINDOW_MINUTES = 30
    HOLD_MINUTES = 45

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "5m",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )

    def _minutes_to_fix(self) -> Optional[int]:
        """Return minutes until the next fix, or None if not near a fix."""
        now = datetime.now(tz=timezone.utc)
        current_min = now.hour * 60 + now.minute
        fix_times = [(self.AM_FIX_HOUR, self.AM_FIX_MINUTE), (self.PM_FIX_HOUR, self.PM_FIX_MINUTE)]
        for fix_h, fix_m in fix_times:
            fix_min = fix_h * 60 + fix_m
            delta = fix_min - current_min
            if 0 <= delta <= self.ENTRY_WINDOW_MINUTES:
                return delta
        return None

    async def generate_signal(self, symbol: str) -> Signal:
        minutes_to_fix = self._minutes_to_fix()
        if minutes_to_fix is None:
            return self._neutral_signal(symbol)

        ohlcv = await self._get_ohlcv(symbol, limit=50)
        if len(ohlcv) < 20:
            return self._neutral_signal(symbol)

        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)

        ema_fast = self._calculate_ema(closes, 5)
        ema_slow = self._calculate_ema(closes, 20)

        time_confidence = 1.0 - (minutes_to_fix / self.ENTRY_WINDOW_MINUTES) * 0.3

        if ema_fast > ema_slow and rsi > 50:
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(0.55 * time_confidence, 3),
                confidence=round(0.65 * time_confidence, 3),
                strategy_name=self.name,
                reasoning=f"Pre-London fix setup: {minutes_to_fix}min to fix, uptrend",
            )
        elif ema_fast < ema_slow and rsi < 50:
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(0.55 * time_confidence, 3),
                confidence=round(0.65 * time_confidence, 3),
                strategy_name=self.name,
                reasoning=f"Pre-London fix setup: {minutes_to_fix}min to fix, downtrend",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 10}
