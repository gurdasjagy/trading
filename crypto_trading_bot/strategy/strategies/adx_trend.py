"""ADX Trend strategy — only trade in strong trends using DI+/DI- direction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class ADXTrendStrategy(BaseStrategy):
    """ADX (Average Directional Index) trend-strength strategy.

    Entry conditions
    ----------------
    * ADX > *adx_threshold* (default 25) indicates a strong trend.
    * **Long**: ADX > threshold AND DI+ crosses above DI-.
    * **Short**: ADX > threshold AND DI- crosses above DI+.

    No trades are taken when ADX is below the threshold (ranging market).
    """

    _STRATEGY_NAME = "adx_trend"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._adx_period * 2 + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        highs = ohlcv["high"]
        lows = ohlcv["low"]
        closes = ohlcv["close"]

        adx_df = ta.adx(highs, lows, closes, length=self._adx_period)
        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)

        if adx_df is None or atr_series is None:
            return None

        adx_cols = [c for c in adx_df.columns if c.startswith(f"ADX_{self._adx_period}")]
        dmp_cols = [c for c in adx_df.columns if c.startswith(f"DMP_{self._adx_period}")]
        dmn_cols = [c for c in adx_df.columns if c.startswith(f"DMN_{self._adx_period}")]

        if not adx_cols or not dmp_cols or not dmn_cols:
            # Try generic column names
            cols = adx_df.columns.tolist()
            if len(cols) < 3:
                return None
            adx_cols = [cols[0]]
            dmp_cols = [cols[1]]
            dmn_cols = [cols[2]]

        curr_adx = float(adx_df[adx_cols[0]].iloc[-1])
        curr_dip = float(adx_df[dmp_cols[0]].iloc[-1])
        prev_dip = float(adx_df[dmp_cols[0]].iloc[-2])
        curr_din = float(adx_df[dmn_cols[0]].iloc[-1])
        prev_din = float(adx_df[dmn_cols[0]].iloc[-2])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        for val in (curr_adx, curr_dip, prev_dip, curr_din, prev_din, curr_atr):
            if pd.isna(val):
                return None

        if curr_adx < self._adx_threshold:
            return None

        cross_dip_up = prev_dip <= prev_din and curr_dip > curr_din
        cross_din_up = prev_din <= prev_dip and curr_din > curr_dip

        direction: Optional[str] = None
        if cross_dip_up:
            direction = "long"
        elif cross_din_up:
            direction = "short"

        if direction is None:
            return None

        adx_strength = (curr_adx - self._adx_threshold) / (100 - self._adx_threshold)
        confidence = round(min(0.9, 0.5 + adx_strength * 0.4), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "adx": curr_adx,
            "di_plus": curr_dip,
            "di_minus": curr_din,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "ADX below threshold or no DI cross")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 2.0
                take_profit = entry + atr * 4.0
            else:
                stop_loss = entry + atr * 2.0
                take_profit = entry - atr * 4.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"ADX trend {direction}: ADX={sig['adx']:.1f}, "
                    f"DI+={sig['di_plus']:.1f}, DI-={sig['di_minus']:.1f}, ATR={atr:.6f}"
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
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            # ADX weakened
            return True
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and sig["di_minus"] > sig["di_plus"]:
            return True
        if side == "short" and sig["di_plus"] > sig["di_minus"]:
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
            "take_profit_pct": min(0.12, sl_pct * 3.0),
            "leverage": 3,
        }
