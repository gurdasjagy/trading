"""Accumulation/Distribution strategy — Wyckoff A/D line divergence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class AccDistStrategy(BaseStrategy):
    """Accumulation/Distribution (A/D) line divergence strategy.

    The A/D line tracks whether a security is being accumulated (bought)
    or distributed (sold) based on price position within its candle range
    and volume.

    Divergence conditions
    ---------------------
    * **Long (bullish divergence)**: price is making a lower low over *lookback*
      bars but the A/D line is making a higher low → institutional accumulation.
    * **Short (bearish divergence)**: price is making a higher high but the
      A/D line is making a lower high → institutional distribution.
    """

    _STRATEGY_NAME = "accumulation_distribution"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        ad_smooth: int = 5,
        lookback: int = 30,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._ad_smooth = ad_smooth
        self._lookback = lookback
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # A/D line computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ad_line(ohlcv: pd.DataFrame) -> pd.Series:
        """Compute the raw Accumulation/Distribution line."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        volume = ohlcv["volume"]
        hl_range = (high - low).replace(0, float("nan"))
        clv = ((close - low) - (high - close)) / hl_range
        ad = (clv * volume).cumsum()
        return ad

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._ad_smooth + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        ad_line = self._compute_ad_line(ohlcv)
        ad_smooth = ad_line.rolling(self._ad_smooth).mean()

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        # Compare current period vs lookback period
        curr_price = float(ohlcv["close"].iloc[-1])
        lb_price = float(ohlcv["close"].iloc[-self._lookback])
        curr_ad = float(ad_smooth.iloc[-1])
        lb_ad = float(ad_smooth.iloc[-self._lookback])

        if pd.isna(curr_ad) or pd.isna(lb_ad):
            return None

        price_down = curr_price < lb_price * 0.999
        price_up = curr_price > lb_price * 1.001
        ad_up = curr_ad > lb_ad * 1.001
        ad_down = curr_ad < lb_ad * 0.999

        direction: Optional[str] = None
        if price_down and ad_up:
            direction = "long"
        elif price_up and ad_down:
            direction = "short"

        if direction is None:
            return None

        price_change = abs(curr_price - lb_price) / lb_price
        ad_change = abs(curr_ad - lb_ad) / (abs(lb_ad) + 1e-9)
        divergence_strength = min(1.0, (price_change + ad_change) / 2.0 * 20)
        confidence = round(min(0.82, 0.5 + divergence_strength * 0.32), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "ad_line": curr_ad,
            "ad_change": ad_change,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No A/D line divergence")

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
                    f"A/D {'bullish' if direction=='long' else 'bearish'} divergence: "
                    f"AD_change={sig['ad_change']:.3f}, ATR={atr:.6f}"
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
        if side == "long" and sig["direction"] == "short":
            return True
        if side == "short" and sig["direction"] == "long":
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
            "take_profit_pct": min(0.09, sl_pct * 2.5),
            "leverage": 2,
        }
