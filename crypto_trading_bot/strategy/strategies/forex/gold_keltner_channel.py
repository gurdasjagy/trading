"""Gold Keltner Channel strategy — volatility-based channel breakout."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldKeltnerChannelStrategy(BaseStrategy):
    """Keltner Channel strategy for gold.

    Upper/lower bands = EMA ± ATR × multiplier.
    Long on close above upper (breakout), short below lower.
    """

    _STRATEGY_NAME = "gold_keltner_channel"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        ema_period: int = 20,
        atr_period: int = 14,
        multiplier: float = 2.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._ema_period = ema_period
        self._atr_period = atr_period
        self._multiplier = multiplier

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < max(self._ema_period, self._atr_period) + 5:
            return self._neutral_signal(symbol)

        closes = ohlcv["close"].tolist()
        ema = self._calculate_ema(closes, self._ema_period)
        atr = self._calculate_atr(ohlcv, self._atr_period)

        upper = ema + self._multiplier * atr
        lower = ema - self._multiplier * atr
        current = closes[-1]

        channel_width = upper - lower

        if current > upper:
            penetration = (current - upper) / channel_width if channel_width > 0 else 0.0
            strength = min(0.55 + penetration, 0.9)
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(strength, 3),
                confidence=0.72,
                strategy_name=self.name,
                reasoning=f"Price {current:.2f} above Keltner upper {upper:.2f}",
            )
        elif current < lower:
            penetration = (lower - current) / channel_width if channel_width > 0 else 0.0
            strength = min(0.55 + penetration, 0.9)
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(strength, 3),
                confidence=0.72,
                strategy_name=self.name,
                reasoning=f"Price {current:.2f} below Keltner lower {lower:.2f}",
            )
        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if len(ohlcv) < max(self._ema_period, self._atr_period) + 5:
            return False
        closes = ohlcv["close"].tolist()
        ema = self._calculate_ema(closes, self._ema_period)
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and closes[-1] < ema:
                return True
            if side_val == "short" and closes[-1] > ema:
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
