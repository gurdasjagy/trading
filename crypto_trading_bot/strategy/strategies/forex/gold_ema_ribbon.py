"""Gold EMA Ribbon strategy — multiple EMA alignment for trend confirmation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldEmaRibbonStrategy(BaseStrategy):
    """EMA Ribbon strategy for gold.

    Uses 8 EMAs (5,8,13,21,34,55,89,144) to detect ribbon expansion/compression.
    All EMAs aligned and expanding = strong trend. Crossing = potential reversal.
    """

    _STRATEGY_NAME = "gold_ema_ribbon"
    EMA_PERIODS = [5, 8, 13, 21, 34, 55, 89, 144]

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=200)
        if len(ohlcv) < self.EMA_PERIODS[-1] + 5:
            return self._neutral_signal(symbol)

        closes = ohlcv["close"].tolist()
        emas = [self._calculate_ema(closes, p) for p in self.EMA_PERIODS]

        bullish_aligned = all(emas[i] > emas[i + 1] for i in range(len(emas) - 1))
        bearish_aligned = all(emas[i] < emas[i + 1] for i in range(len(emas) - 1))

        if not bullish_aligned and not bearish_aligned:
            return self._neutral_signal(symbol)

        emas_prev = [
            self._calculate_ema(closes[:-5], p) for p in self.EMA_PERIODS if len(closes) >= p + 5
        ]
        spread_current = emas[0] - emas[-1]
        spread_prev = emas_prev[0] - emas_prev[-1] if emas_prev else spread_current

        expanding = abs(spread_current) > abs(spread_prev)
        strength = 0.7 if expanding else 0.5

        if bullish_aligned:
            return Signal(
                symbol=symbol,
                direction="long",
                strength=strength,
                confidence=0.80,
                strategy_name=self.name,
                reasoning=f"EMA ribbon bullish aligned, expanding={expanding}",
            )
        else:
            return Signal(
                symbol=symbol,
                direction="short",
                strength=strength,
                confidence=0.80,
                strategy_name=self.name,
                reasoning=f"EMA ribbon bearish aligned, expanding={expanding}",
            )

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=200)
        if len(ohlcv) < self.EMA_PERIODS[-1] + 5:
            return False
        closes = ohlcv["close"].tolist()
        emas = [self._calculate_ema(closes, p) for p in self.EMA_PERIODS]
        bullish = all(emas[i] > emas[i + 1] for i in range(len(emas) - 1))
        bearish = all(emas[i] < emas[i + 1] for i in range(len(emas) - 1))
        side = getattr(position, "side", None)
        if side is not None:
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and not bullish:
                return True
            if side_val == "short" and not bearish:
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
