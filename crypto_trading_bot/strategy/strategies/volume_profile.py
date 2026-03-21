"""Volume Profile strategy — Point of Control and high-volume node mean-reversion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class VolumeProfileStrategy(BaseStrategy):
    """Volume-at-price (Volume Profile) strategy.

    Builds a simplified volume profile over the last *profile_bars* candles
    by bucketing typical prices into *num_bins* price bins.  The bin with
    the highest volume is the Point of Control (POC).

    Entry conditions
    ----------------
    * **Long**: current price is within *tolerance* of POC from below (price
      just touched the POC and is bouncing up) AND price is below short EMA
      (mean-reversion).
    * **Short**: current price is within *tolerance* of POC from above AND
      price is above short EMA.

    Additional signal: if price is in a low-volume node (LVN) it tends to
    move quickly — this increases confidence.
    """

    _STRATEGY_NAME = "volume_profile"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        profile_bars: int = 100,
        num_bins: int = 30,
        tolerance: float = 0.005,
        ema_period: int = 20,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._profile_bars = profile_bars
        self._num_bins = num_bins
        self._tolerance = tolerance
        self._ema_period = ema_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Volume profile helpers
    # ------------------------------------------------------------------

    def _build_profile(self, ohlcv: pd.DataFrame) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return (poc_price, bin_edges, bin_volumes)."""
        typical = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
        price_min = float(ohlcv["low"].min())
        price_max = float(ohlcv["high"].max())
        if price_max <= price_min:
            return float(typical.iloc[-1]), np.array([]), np.array([])

        bin_edges = np.linspace(price_min, price_max, self._num_bins + 1)
        bin_volumes = np.zeros(self._num_bins)

        for i in range(len(ohlcv)):
            t = float(typical.iloc[i])
            v = float(ohlcv["volume"].iloc[i])
            idx = int((t - price_min) / (price_max - price_min) * self._num_bins)
            idx = min(idx, self._num_bins - 1)
            bin_volumes[idx] += v

        poc_bin = int(np.argmax(bin_volumes))
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0
        return poc_price, bin_edges, bin_volumes

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = max(self._profile_bars, self._ema_period + self._atr_period + 5)
        if len(ohlcv) < min_rows:
            return None

        profile_data = ohlcv.iloc[-self._profile_bars:]
        poc_price, bin_edges, bin_volumes = self._build_profile(profile_data)

        if len(bin_edges) == 0:
            return None

        ema_series = ta.ema(ohlcv["close"], length=self._ema_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)

        if ema_series is None or atr_series is None:
            return None

        curr_ema = float(ema_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(ohlcv["close"].iloc[-1])
        prev_price = float(ohlcv["close"].iloc[-2])

        if pd.isna(curr_ema) or pd.isna(curr_atr):
            return None

        price_range = float(ohlcv["high"].max() - ohlcv["low"].min())
        tol = poc_price * self._tolerance

        at_poc_from_below = abs(curr_price - poc_price) <= tol and prev_price < poc_price
        at_poc_from_above = abs(curr_price - poc_price) <= tol and prev_price > poc_price

        direction: Optional[str] = None
        if at_poc_from_below and curr_price < curr_ema:
            direction = "long"
        elif at_poc_from_above and curr_price > curr_ema:
            direction = "short"

        if direction is None:
            return None

        # Volume at current bin vs average — low volume = LVN = faster move
        curr_bin = min(
            int((curr_price - float(bin_edges[0])) / (float(bin_edges[-1]) - float(bin_edges[0])) * self._num_bins),
            self._num_bins - 1,
        )
        avg_vol = float(bin_volumes.mean())
        curr_vol = float(bin_volumes[curr_bin]) if len(bin_volumes) > curr_bin else avg_vol
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
        # High volume at POC = stronger magnet, higher confidence
        confidence = round(min(0.85, 0.5 + min(1.0, vol_ratio) * 0.35), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "poc_price": poc_price,
            "ema": curr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=self._profile_bars + 50)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No volume profile signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            poc = sig["poc_price"]

            if direction == "long":
                stop_loss = poc - atr * 1.5
                take_profit = poc + atr * 2.0
            else:
                stop_loss = poc + atr * 1.5
                take_profit = poc - atr * 2.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Volume Profile {direction}: POC={poc:.4f}, "
                    f"price={entry:.4f}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=self._profile_bars + 20)
        if ohlcv.empty:
            return False
        profile_data = ohlcv.iloc[-self._profile_bars:]
        poc_price, _, _ = self._build_profile(profile_data)
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close when price reaches/passes POC
        if side == "long" and curr_price >= poc_price:
            return True
        if side == "short" and curr_price <= poc_price:
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
