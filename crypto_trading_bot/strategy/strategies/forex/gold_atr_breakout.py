"""Gold ATR Breakout strategy — price breakout measured in ATR units."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldAtrBreakoutStrategy(BaseStrategy):
    """ATR-based breakout strategy for gold.

    Detects breakouts where price moves more than N×ATR from the prior close
    within the current candle. Strong breakout = trade in breakout direction.
    """

    _STRATEGY_NAME = "gold_atr_breakout"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        atr_period: int = 14,
        breakout_multiplier: float = 1.5,
        volume_confirm: bool = True,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_period = atr_period
        self._breakout_mult = breakout_multiplier
        self._volume_confirm = volume_confirm

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < self._atr_period + 5:
            return self._neutral_signal(symbol)

        closes = ohlcv["close"].values
        volumes = ohlcv["volume"].values
        atr = self._calculate_atr(ohlcv, self._atr_period)

        if atr == 0:
            return self._neutral_signal(symbol)

        current_move = closes[-1] - closes[-2]
        move_in_atr = abs(current_move) / atr

        if move_in_atr < self._breakout_mult:
            return self._neutral_signal(symbol)

        if self._volume_confirm and len(volumes) > 10:
            avg_vol = sum(volumes[-11:-1]) / 10
            if avg_vol > 0 and volumes[-1] < avg_vol * 1.2:
                return self._neutral_signal(symbol)

        strength = min(0.5 + (move_in_atr - self._breakout_mult) / 3, 0.95)

        if current_move > 0:
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(strength, 3),
                confidence=0.70,
                strategy_name=self.name,
                reasoning=f"ATR breakout UP {move_in_atr:.2f}×ATR (ATR={atr:.2f})",
            )
        else:
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(strength, 3),
                confidence=0.70,
                strategy_name=self.name,
                reasoning=f"ATR breakout DOWN {move_in_atr:.2f}×ATR (ATR={atr:.2f})",
            )

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
