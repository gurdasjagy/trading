"""VWAP Deviation strategy — mean-reversion when price diverges 2σ from VWAP."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class VWAPDeviationStrategy(BaseStrategy):
    """Mean-reversion strategy based on VWAP standard-deviation bands.

    Entry conditions
    ----------------
    * **Long**: price is more than *threshold* standard deviations below VWAP.
    * **Short**: price is more than *threshold* standard deviations above VWAP.

    VWAP and its rolling standard deviation are computed over the last
    *vwap_period* candles so the strategy works on any timeframe.
    """

    _STRATEGY_NAME = "vwap_deviation"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        vwap_period: int = 48,
        std_threshold: float = 2.0,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._vwap_period = vwap_period
        self._std_threshold = std_threshold
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(ohlcv) < self._vwap_period + self._atr_period:
            return None

        df = ohlcv.copy()
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_vp = (typical * df["volume"]).rolling(self._vwap_period).sum()
        cum_vol = df["volume"].rolling(self._vwap_period).sum()

        vwap = cum_vp / cum_vol.replace(0, np.nan)
        vwap_std = (typical - vwap).rolling(self._vwap_period).std()

        curr_price = float(df["close"].iloc[-1])
        curr_vwap = float(vwap.iloc[-1])
        curr_std = float(vwap_std.iloc[-1])

        if pd.isna(curr_vwap) or pd.isna(curr_std) or curr_std == 0:
            return None

        atr_series = ta.atr(df["high"], df["low"], df["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        deviation = (curr_price - curr_vwap) / curr_std

        direction: Optional[str] = None
        if deviation <= -self._std_threshold:
            direction = "long"
        elif deviation >= self._std_threshold:
            direction = "short"

        if direction is None:
            return None

        # Confidence scales with how far beyond the threshold we are
        excess = abs(deviation) - self._std_threshold
        confidence = round(min(0.9, 0.5 + excess * 0.15), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "vwap": curr_vwap,
            "vwap_std": curr_std,
            "deviation": deviation,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "Price within VWAP deviation bands")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            vwap = sig["vwap"]

            # Take profit targets VWAP; stop-loss is 1.5 ATR beyond entry
            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = vwap
            else:
                stop_loss = entry + atr * 1.5
                take_profit = vwap

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"VWAP deviation {sig['deviation']:.2f}σ → {direction}, "
                    f"VWAP={vwap:.4f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=self._vwap_period + 20)
        if ohlcv.empty:
            return False

        df = ohlcv.copy()
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_vp = (typical * df["volume"]).rolling(self._vwap_period).sum()
        cum_vol = df["volume"].rolling(self._vwap_period).sum()
        vwap = (cum_vp / cum_vol.replace(0, float("nan"))).iloc[-1]

        if pd.isna(vwap):
            return False

        curr_price = float(df["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()

        # Close when price reaches VWAP (mean-reversion target)
        if side == "long" and curr_price >= float(vwap):
            return True
        if side == "short" and curr_price <= float(vwap):
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
            "take_profit_pct": min(0.06, sl_pct * 1.5),
            "leverage": 2,
        }
