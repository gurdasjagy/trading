"""Gold Williams %R oscillator strategy — overbought/oversold reversals."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldWilliamsRStrategy(BaseStrategy):
    """Williams %R mean-reversion strategy for gold.

    Williams %R < -80 (oversold) → long; > -20 (overbought) → short.
    """

    _STRATEGY_NAME = "gold_williams_r"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        period: int = 14,
        oversold: float = -80.0,
        overbought: float = -20.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    def _calculate_williams_r(self, ohlcv: pd.DataFrame) -> float:
        """Calculate most recent Williams %R value."""
        n = self._period
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        if len(closes) < n:
            return -50.0
        hh = max(highs[-n:])
        ll = min(lows[-n:])
        denom = hh - ll
        if denom == 0:
            return -50.0
        return -100 * (hh - closes[-1]) / denom

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        if len(ohlcv) < self._period + 3:
            return self._neutral_signal(symbol)

        wr = self._calculate_williams_r(ohlcv)
        rsi = self._calculate_rsi(ohlcv["close"].tolist())

        if wr < self._oversold and rsi < 40:
            strength = min(0.5 + abs(wr + 80) / 40, 0.9)
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(strength, 3),
                confidence=0.68,
                strategy_name=self.name,
                reasoning=f"Williams %R={wr:.1f} (oversold) RSI={rsi:.1f}",
            )
        elif wr > self._overbought and rsi > 60:
            strength = min(0.5 + abs(wr + 20) / 40, 0.9)
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(strength, 3),
                confidence=0.68,
                strategy_name=self.name,
                reasoning=f"Williams %R={wr:.1f} (overbought) RSI={rsi:.1f}",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        if len(ohlcv) < self._period + 3:
            return False
        wr = self._calculate_williams_r(ohlcv)
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and wr > -30:
                return True
            if side_val == "short" and wr < -70:
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
