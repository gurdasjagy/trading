"""Parabolic SAR strategy — trend-following with dynamic trailing stop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class ParabolicSARStrategy(BaseStrategy):
    """Parabolic SAR trend-following strategy.

    Entry conditions
    ----------------
    * **Long**: SAR value flips from above price to below price (SAR < close).
    * **Short**: SAR value flips from below price to above price (SAR > close).

    The SAR level itself serves as the trailing stop-loss.
    """

    _STRATEGY_NAME = "parabolic_sar"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        step: float = 0.02,
        max_step: float = 0.2,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._step = step
        self._max_step = max_step
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._atr_period + 10
        if len(ohlcv) < min_rows:
            return None

        highs = ohlcv["high"]
        lows = ohlcv["low"]
        closes = ohlcv["close"]

        psar_df = ta.psar(highs, lows, closes, af0=self._step, af=self._step, max_af=self._max_step)
        if psar_df is None:
            return None

        # pandas_ta psar returns: PSARl (long), PSARs (short), PSARaf, PSARr
        long_cols = [c for c in psar_df.columns if "PSARl" in c]
        short_cols = [c for c in psar_df.columns if "PSARs" in c]

        if not long_cols or not short_cols:
            # Fallback: first column is the combined SAR
            sar_col = psar_df.columns[0]
            curr_sar = float(psar_df[sar_col].iloc[-1])
            prev_sar = float(psar_df[sar_col].iloc[-2])
            if pd.isna(curr_sar) or pd.isna(prev_sar):
                return None
            curr_price = float(closes.iloc[-1])
            prev_price = float(closes.iloc[-2])
            prev_above = prev_sar > prev_price
            curr_above = curr_sar > curr_price
            if prev_above and not curr_above:
                direction = "long"
                sar_level = curr_sar
            elif not prev_above and curr_above:
                direction = "short"
                sar_level = curr_sar
            else:
                return None
        else:
            long_col = long_cols[0]
            short_col = short_cols[0]
            curr_long_sar = psar_df[long_col].iloc[-1]
            curr_short_sar = psar_df[short_col].iloc[-1]
            prev_long_sar = psar_df[long_col].iloc[-2]
            prev_short_sar = psar_df[short_col].iloc[-2]

            curr_price = float(closes.iloc[-1])
            # Flip from short to long: long SAR appears (was NaN, now has value)
            if pd.isna(prev_long_sar) and not pd.isna(curr_long_sar):
                direction = "long"
                sar_level = float(curr_long_sar)
            elif pd.isna(prev_short_sar) and not pd.isna(curr_short_sar):
                direction = "short"
                sar_level = float(curr_short_sar)
            else:
                return None

        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        distance = abs(curr_price - sar_level) / curr_price
        confidence = round(min(0.85, 0.5 + distance * 30), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "sar_level": sar_level,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Parabolic SAR flip")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            sar = sig["sar_level"]

            take_profit = entry + atr * 3.0 if direction == "long" else entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Parabolic SAR flipped {direction}: "
                    f"SAR={sar:.6f}, price={entry:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(sar, 6),
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
        psar_df = ta.psar(
            ohlcv["high"], ohlcv["low"], ohlcv["close"],
            af0=self._step, af=self._step, max_af=self._max_step
        )
        if psar_df is None:
            return False
        long_cols = [c for c in psar_df.columns if "PSARl" in c]
        short_cols = [c for c in psar_df.columns if "PSARs" in c]
        side = str(getattr(position, "side", "long")).lower()
        curr_price = float(ohlcv["close"].iloc[-1])
        if long_cols and short_cols:
            long_sar = psar_df[long_cols[0]].iloc[-1]
            short_sar = psar_df[short_cols[0]].iloc[-1]
            if side == "long" and pd.isna(long_sar):
                return True
            if side == "short" and pd.isna(short_sar):
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
