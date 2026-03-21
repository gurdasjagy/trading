"""Technical breakout strategy — resistance/support breakouts with confirmation."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class TechnicalBreakoutStrategy(BaseStrategy):
    """Detects and trades technical breakouts.

    Detection logic
    ---------------
    * Identifies recent resistance / support levels using rolling highs/lows.
    * Requires price to close *above* resistance (long) or *below* support (short).
    * Volume must be above average on the breakout candle.
    * Stop-loss placed at 1× ATR below breakout level.
    * Multi-timeframe confirmation: higher timeframe must agree with direction.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        lookback: int = 20,
    ) -> None:
        super().__init__(
            name="technical_breakout",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._lookback = lookback

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=self._lookback + 10)
            if len(ohlcv) < self._lookback:
                return self._neutral_signal(symbol, "Insufficient OHLCV data")

            resistance = float(ohlcv["high"].iloc[:-1].rolling(self._lookback).max().iloc[-1])
            support = float(ohlcv["low"].iloc[:-1].rolling(self._lookback).min().iloc[-1])
            last_close = float(ohlcv["close"].iloc[-1])
            atr = self._calculate_atr(ohlcv)
            volume_ok = self._volume_confirmed(ohlcv)

            if not volume_ok:
                return self._neutral_signal(symbol, "Volume not confirming breakout")

            # Higher timeframe bias check
            htf_bias = await self._get_htf_bias(symbol)

            if last_close > resistance and (htf_bias in ("bullish", "neutral")):
                strength = min(1.0, (last_close - resistance) / (atr or 1) * 0.5 + 0.5)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.72,
                    strategy_name=self.name,
                    reasoning=(
                        f"Breakout above resistance {resistance:.4f}; "
                        f"close {last_close:.4f}; HTF={htf_bias}"
                    ),
                    stop_loss=resistance - atr,
                    take_profit=last_close + atr * 2.0,
                    leverage=3,
                )

            if last_close < support and (htf_bias in ("bearish", "neutral")):
                strength = min(1.0, (support - last_close) / (atr or 1) * 0.5 + 0.5)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.72,
                    strategy_name=self.name,
                    reasoning=(
                        f"Breakdown below support {support:.4f}; "
                        f"close {last_close:.4f}; HTF={htf_bias}"
                    ),
                    stop_loss=support + atr,
                    take_profit=last_close - atr * 2.0,
                    leverage=3,
                )

            return self._neutral_signal(
                symbol,
                f"No breakout — price {last_close:.4f} between "
                f"support {support:.4f} and resistance {resistance:.4f}",
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if price re-enters the broken level (false breakout)."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=self._lookback + 5)
        if ohlcv.empty:
            return False
        resistance = float(ohlcv["high"].iloc[:-1].rolling(self._lookback).max().iloc[-1])
        support = float(ohlcv["low"].iloc[:-1].rolling(self._lookback).min().iloc[-1])
        last_close = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and last_close < resistance:
            return True
        if side == "short" and last_close > support:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _volume_confirmed(ohlcv: pd.DataFrame, factor: float = 1.3) -> bool:
        if len(ohlcv) < 5:
            return False
        avg_vol = float(ohlcv["volume"].iloc[:-1].mean())
        return float(ohlcv["volume"].iloc[-1]) > avg_vol * factor

    async def _get_htf_bias(self, symbol: str) -> str:
        """Fetch a higher timeframe bias (simple EMA cross)."""
        try:
            htf_ohlcv = await self._get_ohlcv(symbol, timeframe="4h", limit=50)
            if htf_ohlcv.empty:
                return "neutral"
            closes = htf_ohlcv["close"].tolist()
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)
            if ema20 > ema50:
                return "bullish"
            if ema20 < ema50:
                return "bearish"
        except Exception:
            pass
        return "neutral"
