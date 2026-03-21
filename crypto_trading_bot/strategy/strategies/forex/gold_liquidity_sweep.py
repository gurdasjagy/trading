"""Gold Liquidity Sweep strategy — trade after stop-hunts and false breakouts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldLiquiditySweepStrategy(BaseStrategy):
    """Liquidity Sweep / Stop Hunt strategy for gold (Smart Money Concepts).

    Identifies when price sweeps beyond a recent high/low (takes out stops)
    and then reverses sharply. Enters in the reversal direction after the sweep.

    Signal logic:
    * Price briefly breaks above N-bar high (sweeps buy-side liquidity) then closes below → short
    * Price briefly breaks below N-bar low (sweeps sell-side liquidity) then closes above → long
    """

    _STRATEGY_NAME = "gold_liquidity_sweep"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        lookback: int = 20,
        min_wick_pct: float = 0.3,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._lookback = lookback
        self._min_wick_pct = min_wick_pct

    async def generate_signal(self, symbol: str) -> Signal:
        ohlcv = await self._get_ohlcv(symbol, limit=self._lookback + 10)
        if len(ohlcv) < self._lookback + 2:
            return self._neutral_signal(symbol)

        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values

        swing_high = max(highs[-self._lookback - 2:-2])
        swing_low = min(lows[-self._lookback - 2:-2])

        c_close = closes[-1]
        c_high = highs[-1]
        c_low = lows[-1]
        c_range = c_high - c_low

        if c_range == 0:
            return self._neutral_signal(symbol)

        if c_low < swing_low and c_close > swing_low:
            wick_below = swing_low - c_low
            if wick_below / c_range >= self._min_wick_pct:
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=0.78,
                    confidence=0.78,
                    strategy_name=self.name,
                    reasoning=(
                        f"Liquidity sweep below {swing_low:.2f} "
                        f"(low={c_low:.2f}), close={c_close:.2f} reclaimed"
                    ),
                )

        if c_high > swing_high and c_close < swing_high:
            wick_above = c_high - swing_high
            if wick_above / c_range >= self._min_wick_pct:
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=0.78,
                    confidence=0.78,
                    strategy_name=self.name,
                    reasoning=(
                        f"Liquidity sweep above {swing_high:.2f} "
                        f"(high={c_high:.2f}), close={c_close:.2f} rejected"
                    ),
                )

        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
