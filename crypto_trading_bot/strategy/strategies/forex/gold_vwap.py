"""Gold VWAP strategy — trade bounces and deviations from VWAP."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldVWAPStrategy(BaseStrategy):
    """Gold VWAP Strategy.

    Institutional traders use VWAP heavily for gold execution.  This
    strategy trades:

    * **VWAP bounce**: price touches VWAP from above (in uptrend) → long,
      or from below (in downtrend) → short.
    * **VWAP deviation breakout**: price deviates more than ``std_bands``
      standard deviations from VWAP and snaps back.

    VWAP is computed as a running (intraday) VWAP over the full OHLCV
    window, or a rolling VWAP when there is no intraday timestamp.
    """

    _STRATEGY_NAME = "gold_vwap"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        std_bands: float = 2.0,
        bounce_tolerance: float = 0.001,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._std_bands = std_bands
        self._bounce_tolerance = bounce_tolerance
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vwap(ohlcv: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Return VWAP and ±2σ standard deviation bands."""
        typical = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
        volumes = ohlcv["volume"]

        cum_tp_vol = (typical * volumes).cumsum()
        cum_vol = volumes.cumsum()

        if float(cum_vol.iloc[-1]) == 0:
            return None

        vwap_series = cum_tp_vol / cum_vol
        vwap = float(vwap_series.iloc[-1])

        # Rolling variance of typical price weighted by volume
        # Use simple std of (typical - vwap) as deviation estimate
        deviations = typical - vwap_series
        dev_std = float(deviations.std())

        return {
            "vwap": vwap,
            "upper1": vwap + dev_std,
            "lower1": vwap - dev_std,
            "upper2": vwap + 2.0 * dev_std,
            "lower2": vwap - 2.0 * dev_std,
            "dev_std": dev_std,
        }

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if ohlcv is None or len(ohlcv) < 50:
            return None

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        vwap_data = self._compute_vwap(ohlcv)
        if vwap_data is None:
            return None

        closes = ohlcv["close"].tolist()
        curr_price = closes[-1]
        prev_price = closes[-2]
        vwap = vwap_data["vwap"]
        upper2 = vwap_data["upper2"]
        lower2 = vwap_data["lower2"]
        dev_std = vwap_data["dev_std"]

        if dev_std <= 0:
            return None

        z_score = (curr_price - vwap) / dev_std
        tol = curr_price * self._bounce_tolerance

        direction: Optional[str] = None
        signal_type: str = ""

        # Deviation mean-reversion signal
        if curr_price >= upper2 and prev_price < upper2:
            direction = "short"
            signal_type = "deviation_short"
        elif curr_price <= lower2 and prev_price > lower2:
            direction = "long"
            signal_type = "deviation_long"

        # VWAP bounce signal: price crosses VWAP from the correct side
        if direction is None:
            if abs(curr_price - vwap) <= tol:
                ema_short = self._calculate_ema(closes, 9)
                # Long bounce: EMA is above VWAP and price just crossed above VWAP
                if ema_short > vwap and prev_price <= vwap <= curr_price:
                    direction = "long"
                    signal_type = "vwap_bounce_long"
                # Short bounce: EMA is below VWAP and price just crossed below VWAP
                elif ema_short < vwap and prev_price >= vwap >= curr_price:
                    direction = "short"
                    signal_type = "vwap_bounce_short"

        if direction is None:
            return None

        abs_z = abs(z_score)
        if signal_type.startswith("deviation"):
            confidence = round(min(0.88, 0.55 + min(abs_z - 2.0, 2.0) * 0.17), 3)
        else:
            confidence = round(0.60, 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "vwap": round(vwap, 4),
            "upper2": round(upper2, 4),
            "lower2": round(lower2, 4),
            "z_score": round(z_score, 3),
            "signal_type": signal_type,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No VWAP signal on gold")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = sig["vwap"] if entry < sig["vwap"] else entry + atr * 2.0
            else:
                stop_loss = entry + atr * 1.5
                take_profit = sig["vwap"] if entry > sig["vwap"] else entry - atr * 2.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold VWAP {sig['signal_type']}: z={sig['z_score']:.2f}, "
                    f"VWAP={sig['vwap']:.4f}, ATR={atr:.4f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if ohlcv.empty:
            return False
        vwap_data = self._compute_vwap(ohlcv)
        if vwap_data is None:
            return False
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        vwap = vwap_data["vwap"]
        # Close when price crosses back through VWAP
        if side == "long" and curr_price < vwap:
            return True
        if side == "short" and curr_price > vwap:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 3,
        }
