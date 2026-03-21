"""Williams %R strategy — overbought/oversold with EMA trend filter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class WilliamsRStrategy(BaseStrategy):
    """Williams %R oscillator strategy.

    Williams %R oscillates from -100 (most oversold) to 0 (most overbought).

    Entry conditions
    ----------------
    * **Long**: %R rises from below *oversold* threshold (−80) back above it,
      AND price is above the long-term EMA (trend filter).
    * **Short**: %R drops from above *overbought* threshold (−20) back below it,
      AND price is below the long-term EMA.
    """

    _STRATEGY_NAME = "williams_r"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        wr_period: int = 14,
        oversold: float = -80.0,
        overbought: float = -20.0,
        ema_period: int = 50,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._wr_period = wr_period
        self._oversold = oversold
        self._overbought = overbought
        self._ema_period = ema_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._wr_period + self._ema_period + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        highs = ohlcv["high"]
        lows = ohlcv["low"]

        wr_series = ta.willr(highs, lows, closes, length=self._wr_period)
        ema_series = ta.ema(closes, length=self._ema_period)
        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)

        if wr_series is None or ema_series is None or atr_series is None:
            return None

        curr_wr = float(wr_series.iloc[-1])
        prev_wr = float(wr_series.iloc[-2])
        curr_ema = float(ema_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        for val in (curr_wr, prev_wr, curr_ema, curr_atr):
            if pd.isna(val):
                return None

        # Cross from oversold back up
        cross_out_oversold = prev_wr <= self._oversold and curr_wr > self._oversold
        # Cross from overbought back down
        cross_out_overbought = prev_wr >= self._overbought and curr_wr < self._overbought

        direction: Optional[str] = None
        if cross_out_oversold and curr_price > curr_ema:
            direction = "long"
        elif cross_out_overbought and curr_price < curr_ema:
            direction = "short"

        if direction is None:
            return None

        if direction == "long":
            depth = abs(self._oversold - prev_wr) / abs(self._oversold)
        else:
            depth = abs(prev_wr - self._overbought) / abs(self._overbought)

        confidence = round(min(0.85, 0.5 + depth * 0.35), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "williams_r": curr_wr,
            "ema": curr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Williams %R signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 2.5
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 2.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Williams %R {direction}: %R={sig['williams_r']:.1f}, "
                    f"EMA={sig['ema']:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        wr = sig["williams_r"]
        # Close near overbought/oversold extremes
        if side == "long" and wr >= self._overbought:
            return True
        if side == "short" and wr <= self._oversold:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2,
        }
