"""Volatility Breakout strategy — ATR expansion after compression."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy based on ATR expansion.

    Detects a period of volatility compression (ATR below its EMA) followed
    by expansion (ATR crosses above threshold × EMA of ATR).  The direction
    of the first large candle after expansion indicates bias.

    Entry conditions
    ----------------
    * **Long**: ATR/EMA(ATR) ratio crosses above *expansion_threshold* AND the
      trigger candle is bullish (close > open).
    * **Short**: same threshold crossed AND trigger candle is bearish.
    """

    _STRATEGY_NAME = "volatility_breakout"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        atr_period: int = 14,
        atr_ema_period: int = 20,
        expansion_threshold: float = 1.3,
        ema_period: int = 50,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_period = atr_period
        self._atr_ema_period = atr_ema_period
        self._expansion_threshold = expansion_threshold
        self._ema_period = ema_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = max(self._atr_period, self._atr_ema_period, self._ema_period) + 10
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        highs = ohlcv["high"]
        lows = ohlcv["low"]
        opens = ohlcv["open"]

        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)
        ema_series = ta.ema(closes, length=self._ema_period)

        if atr_series is None or ema_series is None:
            return None

        # EMA of ATR to detect compression/expansion
        atr_ema = ta.ema(atr_series, length=self._atr_ema_period)
        if atr_ema is None:
            return None

        curr_atr = float(atr_series.iloc[-1])
        prev_atr = float(atr_series.iloc[-2])
        curr_atr_ema = float(atr_ema.iloc[-1])
        prev_atr_ema = float(atr_ema.iloc[-2])
        curr_ema = float(ema_series.iloc[-1])
        curr_price = float(closes.iloc[-1])
        curr_open = float(opens.iloc[-1])

        for val in (curr_atr, prev_atr, curr_atr_ema, prev_atr_ema, curr_ema):
            if pd.isna(val) or val == 0:
                return None

        curr_ratio = curr_atr / curr_atr_ema
        prev_ratio = prev_atr / prev_atr_ema

        # Detect expansion: ratio just crossed above threshold
        expanding = prev_ratio < self._expansion_threshold <= curr_ratio

        if not expanding:
            return None

        bullish_candle = curr_price > curr_open
        bearish_candle = curr_price < curr_open

        direction: Optional[str] = None
        if bullish_candle and curr_price > curr_ema:
            direction = "long"
        elif bearish_candle and curr_price < curr_ema:
            direction = "short"

        if direction is None:
            return None

        expansion_strength = curr_ratio - self._expansion_threshold
        confidence = round(min(0.88, 0.55 + expansion_strength * 0.3), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "atr_ratio": curr_ratio,
            "atr_ema": curr_atr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=120)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No ATR expansion detected")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 3.5
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 3.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Volatility breakout {direction}: "
                    f"ATR/ATR_EMA={sig['atr_ratio']:.2f} > {self._expansion_threshold}, "
                    f"ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        atr_s = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_s is None:
            return False
        atr_ema = ta.ema(atr_s, length=self._atr_ema_period)
        if atr_ema is None:
            return False
        curr_atr = float(atr_s.iloc[-1])
        curr_atr_ema = float(atr_ema.iloc[-1])
        if pd.isna(curr_atr) or pd.isna(curr_atr_ema) or curr_atr_ema == 0:
            return False
        # Close when volatility contracts back
        if curr_atr / curr_atr_ema < 1.0:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 3.5),
            "leverage": 3,
        }
