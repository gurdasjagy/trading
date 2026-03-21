"""Scalping strategy — high-frequency EMA crossover + RSI on short timeframes."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class ScalpingStrategy(BaseStrategy):
    """Short-term scalping on 1m / 5m timeframes.

    Entry conditions
    ----------------
    * **Long**: fast EMA crosses above slow EMA + RSI between 40–60 (momentum).
    * **Short**: fast EMA crosses below slow EMA + RSI between 40–60.
    * Only on high-liquidity markets (volume filter).
    * Very tight stop-loss (0.3–0.5 %), small take-profit (0.5 %).
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "5m",
        enabled: bool = True,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_low: float = 40.0,
        rsi_high: float = 60.0,
    ) -> None:
        super().__init__(
            name="scalping",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._rsi_low = rsi_low
        self._rsi_high = rsi_high

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=60)
            if len(ohlcv) < self._slow_ema + 5:
                return self._neutral_signal(symbol, "Insufficient data for scalp")

            closes = ohlcv["close"].tolist()
            rsi = self._calculate_rsi(closes[-30:])

            # EMA crossover detection (current vs previous bar)
            fast_now = self._calculate_ema(closes, self._fast_ema)
            slow_now = self._calculate_ema(closes, self._slow_ema)
            fast_prev = self._calculate_ema(closes[:-1], self._fast_ema)
            slow_prev = self._calculate_ema(closes[:-1], self._slow_ema)

            volume_ok = self._high_liquidity(ohlcv)
            if not volume_ok:
                return self._neutral_signal(symbol, "Insufficient liquidity for scalp")

            bullish_cross = fast_prev < slow_prev and fast_now > slow_now
            bearish_cross = fast_prev > slow_prev and fast_now < slow_now

            if bullish_cross and self._rsi_low <= rsi <= self._rsi_high:
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=0.6,
                    confidence=0.65,
                    strategy_name=self.name,
                    reasoning=(
                        f"Bullish EMA{self._fast_ema}/{self._slow_ema} cross; RSI={rsi:.1f}"
                    ),
                    leverage=5,
                )

            if bearish_cross and self._rsi_low <= rsi <= self._rsi_high:
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=0.6,
                    confidence=0.65,
                    strategy_name=self.name,
                    reasoning=(
                        f"Bearish EMA{self._fast_ema}/{self._slow_ema} cross; RSI={rsi:.1f}"
                    ),
                    leverage=5,
                )

            return self._neutral_signal(
                symbol,
                f"No EMA cross — fast={fast_now:.4f}, slow={slow_now:.4f}, RSI={rsi:.1f}",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close on opposite EMA cross or RSI reaching extreme."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=30)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        fast = self._calculate_ema(closes, self._fast_ema)
        slow = self._calculate_ema(closes, self._slow_ema)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and (fast < slow or rsi > 70):
            return True
        if side == "short" and (fast > slow or rsi < 30):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        return {
            "position_size_pct": 0.03,
            "stop_loss_pct": 0.004,  # 0.4 % stop
            "take_profit_pct": 0.008,  # 0.8 % target (2:1 R:R)
            "leverage": 5,
        }

    @staticmethod
    def _high_liquidity(ohlcv: Any, min_volume_percentile: float = 0.5) -> bool:
        """Return True if current volume exceeds the median."""
        if len(ohlcv) < 5:
            return False
        volumes = ohlcv["volume"].values
        median_vol = float(sorted(volumes)[len(volumes) // 2])
        return float(volumes[-1]) >= median_vol * min_volume_percentile
