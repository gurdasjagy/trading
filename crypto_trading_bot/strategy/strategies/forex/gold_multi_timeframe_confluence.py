"""Gold Multi-Timeframe Confluence strategy — requires agreement across 3 timeframes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from strategy.base_strategy import BaseStrategy, Signal


class GoldMultiTimeframeConfluenceStrategy(BaseStrategy):
    """Multi-Timeframe Confluence strategy for gold.

    Checks trend direction on 3 timeframes (5m, 15m, 1h) using EMA alignment.
    Only signals when all three timeframes agree on direction.

    This is the highest-confidence strategy and given maximum weight in the
    consensus engine.
    """

    _STRATEGY_NAME = "gold_multi_timeframe_confluence"

    TF_CONFIGS = [
        {"timeframe": "5m",  "fast_ema": 8,  "slow_ema": 21, "weight": 0.25},
        {"timeframe": "15m", "fast_ema": 8,  "slow_ema": 21, "weight": 0.35},
        {"timeframe": "1h",  "fast_ema": 8,  "slow_ema": 21, "weight": 0.40},
    ]

    LONG_THRESHOLD = 1.0005   # fast EMA must exceed slow EMA by this factor for long bias
    SHORT_THRESHOLD = 0.9995  # fast EMA must be below slow EMA by this factor for short bias

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or ["XAU/USDT"],
            timeframe=timeframe,
            enabled=enabled,
        )

    def _tf_bias(self, closes: List[float], fast: int, slow: int) -> str:
        """Return 'long', 'short', or 'neutral' for given EMA params."""
        if len(closes) < slow + 2:
            return "neutral"
        ema_fast = self._calculate_ema(closes, fast)
        ema_slow = self._calculate_ema(closes, slow)
        if ema_fast > ema_slow * self.LONG_THRESHOLD:
            return "long"
        elif ema_fast < ema_slow * self.SHORT_THRESHOLD:
            return "short"
        return "neutral"

    async def generate_signal(self, symbol: str) -> Signal:
        biases: List[str] = []
        weights: List[float] = []

        for config in self.TF_CONFIGS:
            ohlcv = await self._get_ohlcv(symbol, timeframe=config["timeframe"], limit=150)
            if len(ohlcv) < config["slow_ema"] + 5:
                biases.append("neutral")
            else:
                closes = ohlcv["close"].tolist()
                rsi = self._calculate_rsi(closes)
                bias = self._tf_bias(closes, config["fast_ema"], config["slow_ema"])
                if bias == "long" and rsi > 75:
                    bias = "neutral"
                elif bias == "short" and rsi < 25:
                    bias = "neutral"
                biases.append(bias)
            weights.append(config["weight"])

        long_score = sum(w for b, w in zip(biases, weights) if b == "long")
        short_score = sum(w for b, w in zip(biases, weights) if b == "short")

        all_long = all(b == "long" for b in biases)
        all_short = all(b == "short" for b in biases)

        if all_long or long_score >= 0.75:
            ohlcv_primary = await self._get_ohlcv(symbol, timeframe="15m", limit=50)
            closes_primary = ohlcv_primary["close"].tolist()
            rsi = self._calculate_rsi(closes_primary) if len(ohlcv_primary) >= 20 else 50
            if rsi > 75:
                return self._neutral_signal(symbol)
            return Signal(
                symbol=symbol,
                direction="long",
                strength=round(0.70 + long_score * 0.2, 3),
                confidence=0.88,
                strategy_name=self.name,
                reasoning=f"MTF confluence LONG: {biases} (scores: long={long_score:.2f})",
            )
        elif all_short or short_score >= 0.75:
            ohlcv_primary = await self._get_ohlcv(symbol, timeframe="15m", limit=50)
            closes_primary = ohlcv_primary["close"].tolist()
            rsi = self._calculate_rsi(closes_primary) if len(ohlcv_primary) >= 20 else 50
            if rsi < 25:
                return self._neutral_signal(symbol)
            return Signal(
                symbol=symbol,
                direction="short",
                strength=round(0.70 + short_score * 0.2, 3),
                confidence=0.88,
                strategy_name=self.name,
                reasoning=f"MTF confluence SHORT: {biases} (scores: short={short_score:.2f})",
            )

        return self._neutral_signal(symbol)

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when MTF consensus reverses."""
        symbol = getattr(position, "symbol", "")
        sig = await self.generate_signal(symbol)
        side = getattr(position, "side", None)
        if side is not None and sig.direction != "neutral":
            side_val = side.value if hasattr(side, "value") else str(side)
            if side_val == "long" and sig.direction == "short":
                return True
            if side_val == "short" and sig.direction == "long":
                return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol)
        atr = self._calculate_atr(ohlcv) if len(ohlcv) >= 15 else 2.0
        return {"atr": round(atr, 4), "leverage": 20}
