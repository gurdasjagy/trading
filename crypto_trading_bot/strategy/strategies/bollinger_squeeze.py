"""Bollinger Squeeze strategy — trade breakouts when BBW compresses below its MA."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class BollingerSqueezeStrategy(BaseStrategy):
    """Bollinger Band Width squeeze breakout strategy.

    A squeeze is detected when the current BBW (Band Width) is below the
    20-period moving average of BBW.  When the squeeze fires and price
    breaks above the upper band a long signal is generated; below the
    lower band → short.
    """

    _STRATEGY_NAME = "bollinger_squeeze"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_ma: int = 20,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._squeeze_ma = squeeze_ma
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._bb_period + self._squeeze_ma + 10
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        highs = ohlcv["high"]
        lows = ohlcv["low"]

        bbands = ta.bbands(closes, length=self._bb_period, std=self._bb_std)
        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)

        if bbands is None or atr_series is None:
            return None

        # Column names produced by pandas_ta bbands
        lower_col = f"BBL_{self._bb_period}_{self._bb_std}"
        mid_col = f"BBM_{self._bb_period}_{self._bb_std}"
        upper_col = f"BBU_{self._bb_period}_{self._bb_std}"
        bw_col = f"BBB_{self._bb_period}_{self._bb_std}"

        # Fallback to first/second/third columns if names differ
        cols = bbands.columns.tolist()
        if lower_col not in cols:
            lower_col, mid_col, upper_col = cols[0], cols[1], cols[2]
            bw_col = cols[3] if len(cols) > 3 else None

        upper = float(bbands[upper_col].iloc[-1])
        lower = float(bbands[lower_col].iloc[-1])

        if pd.isna(upper) or pd.isna(lower):
            return None

        mid = float(bbands[mid_col].iloc[-1])
        bbw = upper - lower

        if bw_col and bw_col in bbands.columns:
            bbw_series = bbands[bw_col]
        else:
            bbw_series = bbands[upper_col] - bbands[lower_col]

        bbw_ma = float(bbw_series.rolling(self._squeeze_ma).mean().iloc[-1])

        if pd.isna(bbw_ma) or bbw_ma == 0:
            return None

        in_squeeze = bbw < bbw_ma
        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])
        curr_atr = float(atr_series.iloc[-1])

        if pd.isna(curr_atr):
            return None

        if not in_squeeze:
            return None

        direction: Optional[str] = None
        if curr_price > upper and prev_price <= upper:
            direction = "long"
        elif curr_price < lower and prev_price >= lower:
            direction = "short"

        if direction is None:
            return None

        squeeze_ratio = max(0.0, 1.0 - bbw / bbw_ma)
        confidence = round(min(0.9, 0.5 + squeeze_ratio * 0.4), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "upper_band": upper,
            "lower_band": lower,
            "bbw": bbw,
            "bbw_ma": bbw_ma,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Bollinger squeeze breakout")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"BB squeeze breakout {direction}: BBW={sig['bbw']:.6f} "
                    f"< BBW_MA={sig['bbw_ma']:.6f}, ATR={atr:.6f}"
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
        side = str(getattr(position, "side", "long")).lower()
        # Close if price reverts back inside the bands
        curr_price = float(ohlcv["close"].iloc[-1])
        if side == "long" and curr_price < sig.get("upper_band", curr_price):
            return True
        if side == "short" and curr_price > sig.get("lower_band", curr_price):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 3,
        }
