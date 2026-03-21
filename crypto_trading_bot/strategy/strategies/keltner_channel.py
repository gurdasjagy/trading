"""Keltner Channel strategy — breakout from EMA ± 2*ATR channels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class KeltnerChannelStrategy(BaseStrategy):
    """Keltner Channel breakout strategy.

    Keltner Channels are built as EMA ± (multiplier × ATR).  A breakout
    above the upper channel signals a long entry; a break below the lower
    channel signals a short entry.  Mean reversion inside the channel is
    NOT traded — only genuine breakouts.

    Entry conditions
    ----------------
    * **Long**: close > upper Keltner Channel (previous close was inside).
    * **Short**: close < lower Keltner Channel (previous close was inside).
    """

    _STRATEGY_NAME = "keltner_channel"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._ema_period = ema_period
        self._atr_period = atr_period
        self._multiplier = multiplier

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = max(self._ema_period, self._atr_period) + 10
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        highs = ohlcv["high"]
        lows = ohlcv["low"]

        ema_series = ta.ema(closes, length=self._ema_period)
        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)

        if ema_series is None or atr_series is None:
            return None

        curr_ema = float(ema_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])

        if pd.isna(curr_ema) or pd.isna(curr_atr):
            return None

        upper = curr_ema + self._multiplier * curr_atr
        lower = curr_ema - self._multiplier * curr_atr

        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])

        # Breakout: current candle closes outside, previous was inside
        direction: Optional[str] = None
        if curr_price > upper and prev_price <= upper:
            direction = "long"
        elif curr_price < lower and prev_price >= lower:
            direction = "short"

        if direction is None:
            return None

        # Distance of breakout as fraction of channel width
        channel_width = upper - lower
        if channel_width <= 0:
            return None

        if direction == "long":
            breakout_ext = (curr_price - upper) / channel_width
        else:
            breakout_ext = (lower - curr_price) / channel_width

        confidence = round(min(0.85, 0.55 + breakout_ext * 0.5), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "upper": upper,
            "lower": lower,
            "ema": curr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Keltner Channel breakout")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["ema"]  # Stop at midline
                take_profit = entry + atr * 2.5
            else:
                stop_loss = sig["ema"]
                take_profit = entry - atr * 2.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Keltner breakout {direction}: upper={sig['upper']:.4f}, "
                    f"lower={sig['lower']:.4f}, price={entry:.4f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close when price reverts back inside the channel
        if side == "long" and curr_price < sig["upper"]:
            return True
        if side == "short" and curr_price > sig["lower"]:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }
