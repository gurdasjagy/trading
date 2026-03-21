"""Supertrend strategy — ATR-based trend-following."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class SupertrendStrategy(BaseStrategy):
    """Supertrend indicator strategy.

    Uses pandas_ta's supertrend indicator.  When price crosses above the
    supertrend line → long; when it crosses below → short.

    Entry conditions
    ----------------
    * **Long**: supertrend direction flips to +1 (price above supertrend).
    * **Short**: supertrend direction flips to -1 (price below supertrend).
    """

    _STRATEGY_NAME = "supertrend"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        atr_period: int = 10,
        multiplier: float = 3.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_period = atr_period
        self._multiplier = multiplier

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

        st_df = ta.supertrend(highs, lows, closes, length=self._atr_period, multiplier=self._multiplier)
        if st_df is None:
            return None

        # pandas_ta supertrend returns columns: SUPERT_n_m, SUPERTd_n_m, SUPERTl_n_m, SUPERTs_n_m
        dir_cols = [c for c in st_df.columns if c.startswith("SUPERTd")]
        st_cols = [c for c in st_df.columns if c.startswith("SUPERT_")]
        if not dir_cols or not st_cols:
            return None

        dir_col = dir_cols[0]
        st_col = st_cols[0]

        curr_dir = int(st_df[dir_col].iloc[-1]) if not pd.isna(st_df[dir_col].iloc[-1]) else 0
        prev_dir = int(st_df[dir_col].iloc[-2]) if not pd.isna(st_df[dir_col].iloc[-2]) else 0
        curr_st = float(st_df[st_col].iloc[-1])
        curr_price = float(closes.iloc[-1])

        if pd.isna(curr_st):
            return None

        # Compute ATR for position sizing
        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        # Only signal on direction change
        direction: Optional[str] = None
        if prev_dir == -1 and curr_dir == 1:
            direction = "long"
        elif prev_dir == 1 and curr_dir == -1:
            direction = "short"

        if direction is None:
            return None

        distance = abs(curr_price - curr_st) / curr_price
        confidence = round(min(0.85, 0.55 + distance * 20), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "supertrend": curr_st,
            "st_direction": curr_dir,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Supertrend direction change")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            st_level = sig["supertrend"]

            if direction == "long":
                stop_loss = st_level - atr * 0.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = st_level + atr * 0.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Supertrend flip to {direction}: "
                    f"ST={st_level:.4f}, price={entry:.4f}, ATR={atr:.6f}"
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
        st_df = ta.supertrend(
            ohlcv["high"], ohlcv["low"], ohlcv["close"],
            length=self._atr_period, multiplier=self._multiplier
        )
        if st_df is None:
            return False
        dir_cols = [c for c in st_df.columns if c.startswith("SUPERTd")]
        if not dir_cols:
            return False
        curr_dir = int(st_df[dir_cols[0]].iloc[-1]) if not pd.isna(st_df[dir_cols[0]].iloc[-1]) else 0
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and curr_dir == -1:
            return True
        if side == "short" and curr_dir == 1:
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
