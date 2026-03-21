"""Ichimoku Cloud strategy — full Ichimoku system with Tenkan/Kijun confirmation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class IchimokuCloudStrategy(BaseStrategy):
    """Full Ichimoku Kinko Hyo system.

    Entry conditions
    ----------------
    * **Long**: price above the cloud (Senkou A > Senkou B, price > both) AND
      Tenkan crosses above Kijun.
    * **Short**: price below the cloud (Senkou A < Senkou B, price < both) AND
      Tenkan crosses below Kijun.

    The cloud thickness relative to price provides confidence weighting.
    """

    _STRATEGY_NAME = "ichimoku_cloud"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._tenkan = tenkan
        self._kijun = kijun
        self._senkou_b = senkou_b
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._senkou_b + self._kijun + 10
        if len(ohlcv) < min_rows:
            return None

        highs = ohlcv["high"]
        lows = ohlcv["low"]
        closes = ohlcv["close"]

        # Tenkan-sen (conversion line)
        tenkan = (highs.rolling(self._tenkan).max() + lows.rolling(self._tenkan).min()) / 2.0
        # Kijun-sen (base line)
        kijun = (highs.rolling(self._kijun).max() + lows.rolling(self._kijun).min()) / 2.0
        # Senkou A (leading span A) — shifted forward 26 by convention; compare current with 26 bars ago
        senkou_a = ((tenkan + kijun) / 2.0).shift(self._kijun)
        # Senkou B (leading span B) — shifted forward 26
        senkou_b_line = (
            (highs.rolling(self._senkou_b).max() + lows.rolling(self._senkou_b).min()) / 2.0
        ).shift(self._kijun)

        curr_tenkan = float(tenkan.iloc[-1])
        prev_tenkan = float(tenkan.iloc[-2])
        curr_kijun = float(kijun.iloc[-1])
        prev_kijun = float(kijun.iloc[-2])
        curr_senkou_a = float(senkou_a.iloc[-1])
        curr_senkou_b = float(senkou_b_line.iloc[-1])
        curr_price = float(closes.iloc[-1])

        atr_series = ta.atr(highs, lows, closes, length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])

        for val in (curr_tenkan, prev_tenkan, curr_kijun, prev_kijun,
                    curr_senkou_a, curr_senkou_b, curr_atr):
            if pd.isna(val):
                return None

        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        cloud_thickness = cloud_top - cloud_bottom

        tenkan_cross_up = prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun
        tenkan_cross_down = prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun
        price_above_cloud = curr_price > cloud_top
        price_below_cloud = curr_price < cloud_bottom
        bullish_cloud = curr_senkou_a > curr_senkou_b
        bearish_cloud = curr_senkou_a < curr_senkou_b

        direction: Optional[str] = None
        if price_above_cloud and bullish_cloud and tenkan_cross_up:
            direction = "long"
        elif price_below_cloud and bearish_cloud and tenkan_cross_down:
            direction = "short"

        if direction is None:
            return None

        price_distance = abs(curr_price - cloud_top if direction == "long" else curr_price - cloud_bottom)
        confidence = round(min(0.9, 0.5 + (cloud_thickness / curr_price) * 100 * 0.2), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "tenkan": curr_tenkan,
            "kijun": curr_kijun,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Ichimoku signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["cloud_bottom"] - atr * 0.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = sig["cloud_top"] + atr * 0.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Ichimoku {direction}: price={'above' if direction=='long' else 'below'} cloud, "
                    f"Tenkan/Kijun cross confirmed, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=200)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        side = str(getattr(position, "side", "long")).lower()
        curr_price = float(ohlcv["close"].iloc[-1])
        if sig is None:
            # If no signal, check if price entered the cloud
            return False
        # Close if price moves back into or through cloud
        if side == "long" and curr_price < sig.get("cloud_top", curr_price):
            return True
        if side == "short" and curr_price > sig.get("cloud_bottom", curr_price):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.03
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 3.0),
            "leverage": 3,
        }
