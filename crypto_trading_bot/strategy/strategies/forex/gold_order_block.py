"""Gold Order Block strategy — identify institutional supply/demand zones."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from strategy.base_strategy import BaseStrategy, Signal


class GoldOrderBlockStrategy(BaseStrategy):
    """Order Block strategy for gold (Smart Money Concepts).

    An order block is the last bearish candle before a strong bullish move (bullish OB)
    or the last bullish candle before a strong bearish move (bearish OB).

    Entry: When price returns to an order block zone.
    """

    _STRATEGY_NAME = "gold_order_block"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        impulse_atr_mult: float = 1.5,
        ob_lookback: int = 50,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._impulse_mult = impulse_atr_mult
        self._ob_lookback = ob_lookback

    def _find_order_blocks(
        self, ohlcv: pd.DataFrame, atr: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Find recent bullish and bearish order blocks."""
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        opens = ohlcv["open"].values
        closes = ohlcv["close"].values
        n = min(len(closes), self._ob_lookback)

        bullish_obs: List[Dict] = []
        bearish_obs: List[Dict] = []

        for i in range(1, n - 2):
            idx = -n + i
            is_bearish = closes[idx] < opens[idx]
            next_move = closes[idx + 1] - opens[idx + 1]
            if is_bearish and next_move > atr * self._impulse_mult:
                bullish_obs.append({
                    "high": highs[idx],
                    "low": lows[idx],
                    "mid": (highs[idx] + lows[idx]) / 2,
                })
            is_bullish = closes[idx] > opens[idx]
            next_drop = opens[idx + 1] - closes[idx + 1]
            if is_bullish and next_drop > atr * self._impulse_mult:
                bearish_obs.append({
                    "high": highs[idx],
                    "low": lows[idx],
                    "mid": (highs[idx] + lows[idx]) / 2,
                })

        return bullish_obs[-5:], bearish_obs[-5:]

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=self._ob_lookback + 10)
        if len(ohlcv) < 20:
            return self._neutral_signal(symbol)

        atr = self._calculate_atr(ohlcv)
        if atr == 0:
            return self._neutral_signal(symbol)

        current = ohlcv["close"].values[-1]
        bullish_obs, bearish_obs = self._find_order_blocks(ohlcv, atr)

        for ob in bullish_obs:
            if ob["low"] <= current <= ob["high"]:
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=0.72,
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Price {current:.2f} inside bullish OB "
                        f"[{ob['low']:.2f}–{ob['high']:.2f}]"
                    ),
                )

        for ob in bearish_obs:
            if ob["low"] <= current <= ob["high"]:
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=0.72,
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Price {current:.2f} inside bearish OB "
                        f"[{ob['low']:.2f}–{ob['high']:.2f}]"
                    ),
                )

        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
