"""Gold New York Open Reversal strategy — fade the initial NY session move."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldNyOpenReversalStrategy(BaseStrategy):
    """New York Open Reversal strategy for gold.

    The New York session opens at 13:30 UTC (09:30 ET).
    Gold often makes a sharp initial move at the NY open that reverses
    within the first 30-60 minutes. This strategy fades the initial spike.

    Entry: if there is a sharp move in the first 30min of NY session,
    take a position in the opposite direction.
    """

    _STRATEGY_NAME = "gold_ny_open_reversal"

    NY_OPEN_HOUR = 13
    NY_OPEN_MINUTE = 30
    ENTRY_WINDOW_MINUTES = 45
    SPIKE_ATR_MULTIPLE = 1.2

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

    def _minutes_since_ny_open(self) -> Optional[int]:
        """Return minutes since NY open, or None if not within entry window."""
        now = datetime.now(tz=timezone.utc)
        ny_open_min = self.NY_OPEN_HOUR * 60 + self.NY_OPEN_MINUTE
        current_min = now.hour * 60 + now.minute
        delta = current_min - ny_open_min
        if 0 <= delta <= self.ENTRY_WINDOW_MINUTES:
            return delta
        return None

    async def generate_signal(self, symbol: str) -> Signal:
        mins_since_open = self._minutes_since_ny_open()
        if mins_since_open is None:
            return self._neutral_signal(symbol)

        ohlcv = await self._get_ohlcv(symbol, limit=50)
        if len(ohlcv) < 20:
            return self._neutral_signal(symbol)

        closes = ohlcv["close"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr == 0:
            return self._neutral_signal(symbol)

        recent_move = closes[-1] - closes[-4]
        spike_ratio = abs(recent_move) / atr

        if spike_ratio < self.SPIKE_ATR_MULTIPLE:
            return self._neutral_signal(symbol)

        direction = "short" if recent_move > 0 else "long"
        strength = min(0.5 + spike_ratio / 5, 0.85)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=round(strength, 3),
            confidence=0.65,
            strategy_name=self.name,
            reasoning=(
                f"NY open reversal: {mins_since_open}min after open, "
                f"spike={spike_ratio:.1f}×ATR → fade {direction}"
            ),
        )

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 15}
