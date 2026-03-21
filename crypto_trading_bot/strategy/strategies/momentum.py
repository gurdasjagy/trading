"""Momentum strategy — trades in the direction of strong price momentum."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Trend-continuation strategy based on MACD, RSI, and volume.

    Entry conditions
    ----------------
    * **Long**: MACD histogram > 0 and rising, RSI between 50–70, and
      volume above its 20-bar SMA.
    * **Short**: MACD histogram < 0 and falling, RSI between 30–50, and
      volume above its 20-bar SMA.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        rsi_long_low: float = 50.0,
        rsi_long_high: float = 70.0,
        rsi_short_low: float = 30.0,
        rsi_short_high: float = 50.0,
        volume_sma_period: int = 20,
    ) -> None:
        super().__init__(
            name="momentum",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._rsi_long_low = rsi_long_low
        self._rsi_long_high = rsi_long_high
        self._rsi_short_low = rsi_short_low
        self._rsi_short_high = rsi_short_high
        self._volume_sma_period = volume_sma_period

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=60)
            if len(ohlcv) < 30:
                return self._neutral_signal(symbol, "Insufficient data for momentum")

            closes = ohlcv["close"].tolist()
            rsi = self._calculate_rsi(closes)
            macd = self._calculate_macd(closes)
            histogram = macd["histogram"]

            # Compute previous histogram for direction
            macd_prev = self._calculate_macd(closes[:-1])
            histogram_prev = macd_prev["histogram"]

            volume_ok = self._volume_above_sma(ohlcv, self._volume_sma_period)
            if not volume_ok:
                return self._neutral_signal(symbol, "Volume below SMA — no momentum")

            # Bullish momentum
            if (
                histogram > 0
                and histogram > histogram_prev
                and self._rsi_long_low <= rsi <= self._rsi_long_high
            ):
                strength = min(1.0, 0.5 + abs(histogram) * 100)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.70,
                    strategy_name=self.name,
                    reasoning=(
                        f"Bullish momentum: MACD hist={histogram:.6f} rising, RSI={rsi:.1f}"
                    ),
                    leverage=3,
                )

            # Bearish momentum
            if (
                histogram < 0
                and histogram < histogram_prev
                and self._rsi_short_low <= rsi <= self._rsi_short_high
            ):
                strength = min(1.0, 0.5 + abs(histogram) * 100)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.70,
                    strategy_name=self.name,
                    reasoning=(
                        f"Bearish momentum: MACD hist={histogram:.6f} falling, RSI={rsi:.1f}"
                    ),
                    leverage=3,
                )

            return self._neutral_signal(
                symbol,
                f"No clear momentum — hist={histogram:.6f}, RSI={rsi:.1f}",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when momentum fades (MACD histogram reverses or RSI at extremes)."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=30)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        macd = self._calculate_macd(closes)
        histogram = macd["histogram"]
        side = str(getattr(position, "side", "long")).lower()

        if side == "long" and (histogram < 0 or rsi > 80):
            return True
        if side == "short" and (histogram > 0 or rsi < 20):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 3,
        }

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """Analyse an accumulated OHLCV DataFrame and return a backtest signal dict.

        This method enables the strategy to be used by :class:`BacktestEngine`
        without requiring a live exchange connection.

        Returns:
            A dict with keys ``side``, ``size``, ``stop_loss``, ``take_profit``
            when a signal is detected, or ``None`` for no signal.
        """
        if len(ohlcv) < 30:
            return None

        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        macd = self._calculate_macd(closes)
        histogram = macd["histogram"]

        macd_prev = self._calculate_macd(closes[:-1])
        histogram_prev = macd_prev["histogram"]

        if not self._volume_above_sma(ohlcv, self._volume_sma_period):
            return None

        last_close = float(closes[-1])
        atr = self._calculate_atr(ohlcv)

        if (
            histogram > 0
            and histogram > histogram_prev
            and self._rsi_long_low <= rsi <= self._rsi_long_high
        ):
            sl = last_close - 1.5 * atr if atr > 0 else None
            tp = last_close + 3.0 * atr if atr > 0 else None
            return {"side": "long", "size": 0.1, "stop_loss": sl, "take_profit": tp}

        if (
            histogram < 0
            and histogram < histogram_prev
            and self._rsi_short_low <= rsi <= self._rsi_short_high
        ):
            sl = last_close + 1.5 * atr if atr > 0 else None
            tp = last_close - 3.0 * atr if atr > 0 else None
            return {"side": "short", "size": 0.1, "stop_loss": sl, "take_profit": tp}

        return None

    @staticmethod
    def _volume_above_sma(ohlcv: Any, period: int = 20) -> bool:
        """Return True if the latest volume exceeds the *period*-bar SMA."""
        if len(ohlcv) < period:
            return False
        volumes = ohlcv["volume"].values
        sma = float(volumes[:-1][-period:].mean())
        return sma > 0 and float(volumes[-1]) > sma
