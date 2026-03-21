"""Mean reversion strategy — fades extreme moves using Bollinger Bands + RSI."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Bollinger-Band mean-reversion strategy.

    Entry conditions
    ----------------
    * **Long**: price closes below lower Bollinger Band AND RSI ≤ 30.
    * **Short**: price closes above upper Bollinger Band AND RSI ≥ 70.

    Exit conditions
    ---------------
    * Close when price returns to the middle band (SMA) or RSI normalises.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ) -> None:
        super().__init__(
            name="mean_reversion",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=self._bb_period + 30)
            if len(ohlcv) < self._bb_period + 5:
                return self._neutral_signal(symbol, "Insufficient data for mean reversion")

            closes = ohlcv["close"].tolist()
            rsi = self._calculate_rsi(closes)
            upper, middle, lower = self._bollinger_bands(closes, self._bb_period, self._bb_std)
            last_close = closes[-1]

            # Long: price below lower band + oversold RSI
            if last_close < lower and rsi <= self._rsi_oversold:
                distance = (lower - last_close) / lower if lower else 0.0
                strength = min(1.0, 0.5 + distance * 10 + (self._rsi_oversold - rsi) / 100)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.72,
                    strategy_name=self.name,
                    reasoning=(
                        f"Price ({last_close:.4f}) below lower BB ({lower:.4f}), "
                        f"RSI={rsi:.1f} — mean reversion long"
                    ),
                    leverage=2,
                )

            # Short: price above upper band + overbought RSI
            if last_close > upper and rsi >= self._rsi_overbought:
                distance = (last_close - upper) / upper if upper else 0.0
                strength = min(1.0, 0.5 + distance * 10 + (rsi - self._rsi_overbought) / 100)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.72,
                    strategy_name=self.name,
                    reasoning=(
                        f"Price ({last_close:.4f}) above upper BB ({upper:.4f}), "
                        f"RSI={rsi:.1f} — mean reversion short"
                    ),
                    leverage=2,
                )

            return self._neutral_signal(
                symbol,
                f"Price within bands (lower={lower:.4f}, upper={upper:.4f}), RSI={rsi:.1f}",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when price reverts to the middle band or RSI normalises."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=self._bb_period + 10)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        _, middle, _ = self._bollinger_bands(closes, self._bb_period, self._bb_std)
        last_close = closes[-1]
        side = str(getattr(position, "side", "long")).lower()

        if side == "long" and (last_close >= middle or rsi >= 55):
            return True
        if side == "short" and (last_close <= middle or rsi <= 45):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=self._bb_period + 30)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.06, sl_pct * 1.5),
            "leverage": 2,
        }

    @staticmethod
    def _bollinger_bands(
        prices: List[float], period: int = 20, num_std: float = 2.0
    ) -> tuple[float, float, float]:
        """Return (upper, middle, lower) Bollinger Band values for the last bar."""
        if len(prices) < period:
            mid = prices[-1] if prices else 0.0
            return mid, mid, mid
        window = prices[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower
