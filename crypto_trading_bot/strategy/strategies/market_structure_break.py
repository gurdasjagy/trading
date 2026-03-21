"""Market Structure Break strategy — BOS and CHoCH detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MarketStructureBreakStrategy(BaseStrategy):
    """Market Structure Break (BOS) and Change of Character (CHoCH) strategy.

    Identifies the prevailing market structure by tracking swing highs and
    lows.  A Break of Structure (BOS) confirms trend continuation; a Change
    of Character (CHoCH) signals a potential reversal.

    Entry conditions
    ----------------
    * **Long BOS**: in an uptrend, price breaks above the last swing high →
      continuation long.
    * **Long CHoCH**: in a downtrend, price breaks above the last significant
      swing high → reversal long (CHoCH).
    * **Short BOS/CHoCH**: inverse of the above.
    """

    _STRATEGY_NAME = "market_structure_break"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        swing_window: int = 10,
        lookback: int = 60,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._swing_window = swing_window
        self._lookback = lookback
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Swing-point helpers
    # ------------------------------------------------------------------

    def _get_swings(
        self, ohlcv: pd.DataFrame
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Return (swing_highs, swing_lows) as (index, price) lists."""
        sw = self._swing_window
        highs: List[Tuple[int, float]] = []
        lows: List[Tuple[int, float]] = []

        for i in range(sw, len(ohlcv) - sw):
            h = float(ohlcv["high"].iloc[i])
            l = float(ohlcv["low"].iloc[i])
            if h == float(ohlcv["high"].iloc[i - sw: i + sw + 1].max()):
                highs.append((i, h))
            if l == float(ohlcv["low"].iloc[i - sw: i + sw + 1].min()):
                lows.append((i, l))

        return highs, lows

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._swing_window * 2 + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        recent = ohlcv.iloc[-self._lookback:]
        swing_highs, swing_lows = self._get_swings(recent)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        prev_price = float(ohlcv["close"].iloc[-2])

        last_sh = swing_highs[-1][1]  # Most recent swing high
        prev_sh = swing_highs[-2][1]
        last_sl = swing_lows[-1][1]
        prev_sl = swing_lows[-2][1]

        # Determine trend: higher highs/lows = uptrend
        uptrend = last_sh > prev_sh and last_sl > prev_sl
        downtrend = last_sh < prev_sh and last_sl < prev_sl

        direction: Optional[str] = None
        signal_type: str = ""

        if uptrend and curr_price > last_sh and prev_price <= last_sh:
            direction = "long"
            signal_type = "BOS"
        elif downtrend and curr_price < last_sl and prev_price >= last_sl:
            direction = "short"
            signal_type = "BOS"
        elif downtrend and curr_price > last_sh and prev_price <= last_sh:
            direction = "long"
            signal_type = "CHoCH"
        elif uptrend and curr_price < last_sl and prev_price >= last_sl:
            direction = "short"
            signal_type = "CHoCH"

        if direction is None:
            return None

        # CHoCH is higher confidence as a reversal; BOS is trend continuation
        confidence = 0.75 if signal_type == "CHoCH" else 0.65

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "signal_type": signal_type,
            "last_swing_high": last_sh,
            "last_swing_low": last_sl,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No market structure break")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["last_swing_low"] - atr * 0.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = sig["last_swing_high"] + atr * 0.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Market structure {sig['signal_type']} {direction}: "
                    f"SH={sig['last_swing_high']:.4f}, SL={sig['last_swing_low']:.4f}, "
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
        ohlcv = await self._get_ohlcv(symbol, limit=self._lookback + 30)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        curr_price = float(ohlcv["close"].iloc[-1])
        if side == "long" and curr_price < sig["last_swing_low"]:
            return True
        if side == "short" and curr_price > sig["last_swing_high"]:
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
            "take_profit_pct": min(0.10, sl_pct * 3.0),
            "leverage": 3,
        }
