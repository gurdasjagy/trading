"""Gold Volatility Squeeze strategy — Bollinger + Keltner band compression."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldVolatilitySqueezeStrategy(BaseStrategy):
    """Volatility Squeeze strategy for gold.

    A squeeze occurs when Bollinger Bands are inside Keltner Channels.
    When the squeeze fires (BB expands outside KC), trade the breakout direction.
    """

    _STRATEGY_NAME = "gold_volatility_squeeze"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_mult: float = 1.5,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._kc_period = kc_period
        self._kc_atr_mult = kc_atr_mult

    def _squeeze_analysis(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Detect squeeze and breakout direction."""
        closes = ohlcv["close"].tolist()
        n = max(self._bb_period, self._kc_period)
        if len(closes) < n + 5:
            return {"squeeze": False, "fired": False, "direction": "neutral"}

        window = closes[-self._bb_period:]
        bb_mid = sum(window) / len(window)
        variance = sum((x - bb_mid) ** 2 for x in window) / len(window)
        bb_std = variance ** 0.5
        bb_upper = bb_mid + self._bb_std * bb_std
        bb_lower = bb_mid - self._bb_std * bb_std

        kc_ema = self._calculate_ema(closes, self._kc_period)
        atr = self._calculate_atr(ohlcv, self._kc_period)
        kc_upper = kc_ema + self._kc_atr_mult * atr
        kc_lower = kc_ema - self._kc_atr_mult * atr

        in_squeeze_now = bb_upper < kc_upper and bb_lower > kc_lower

        window_prev = closes[-self._bb_period - 1:-1]
        if len(window_prev) < self._bb_period:
            return {"squeeze": in_squeeze_now, "fired": False, "direction": "neutral"}
        bb_mid_prev = sum(window_prev) / len(window_prev)
        var_prev = sum((x - bb_mid_prev) ** 2 for x in window_prev) / len(window_prev)
        bb_upper_prev = bb_mid_prev + self._bb_std * (var_prev ** 0.5)
        bb_lower_prev = bb_mid_prev - self._bb_std * (var_prev ** 0.5)
        in_squeeze_prev = bb_upper_prev < kc_upper and bb_lower_prev > kc_lower

        fired = in_squeeze_prev and not in_squeeze_now
        direction = "long" if closes[-1] > bb_mid else "short"

        return {
            "squeeze": in_squeeze_now,
            "fired": fired,
            "direction": direction,
            "bb_width": round(bb_upper - bb_lower, 4),
            "kc_width": round(kc_upper - kc_lower, 4),
        }

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        result = self._squeeze_analysis(ohlcv)

        if not result["fired"]:
            return self._neutral_signal(symbol)

        direction = result["direction"]

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=0.70,
            confidence=0.78,
            strategy_name=self.name,
            reasoning=f"Volatility squeeze fired → {direction}",
        )

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
