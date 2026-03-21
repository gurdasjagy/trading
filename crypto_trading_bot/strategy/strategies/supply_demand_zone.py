"""Supply & Demand Zone strategy — institutional order-flow zones."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class SupplyDemandZoneStrategy(BaseStrategy):
    """Supply and Demand zone strategy.

    Identifies strong supply zones (sharp drops after consolidation) and
    demand zones (sharp rallies after consolidation), then trades when
    price returns to these zones.

    Zone detection
    --------------
    * **Demand zone**: a candle whose close is ≥ *impulse_factor* × ATR above
      its open, preceded by a ranging/consolidating base.  The base low
      defines the zone.
    * **Supply zone**: a candle whose close is ≤ −*impulse_factor* × ATR below
      its open (large red candle), preceded by a base.  The base high
      defines the zone.
    """

    _STRATEGY_NAME = "supply_demand_zone"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        impulse_factor: float = 2.0,
        zone_tolerance: float = 0.003,
        lookback: int = 50,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._impulse_factor = impulse_factor
        self._zone_tolerance = zone_tolerance
        self._lookback = lookback
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Zone detection
    # ------------------------------------------------------------------

    def _find_zones(
        self, ohlcv: pd.DataFrame, atr: float
    ) -> Tuple[List[float], List[float]]:
        """Return (demand_zones, supply_zones) as lists of zone-base prices."""
        demand_zones: List[float] = []
        supply_zones: List[float] = []
        impulse_min = self._impulse_factor * atr

        for i in range(2, len(ohlcv) - 1):
            body = float(ohlcv["close"].iloc[i]) - float(ohlcv["open"].iloc[i])
            # Bullish impulse → demand zone at base low
            if body >= impulse_min:
                base_low = float(ohlcv["low"].iloc[i - 1])
                demand_zones.append(base_low)
            # Bearish impulse → supply zone at base high
            elif body <= -impulse_min:
                base_high = float(ohlcv["high"].iloc[i - 1])
                supply_zones.append(base_high)

        return demand_zones, supply_zones

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr) or curr_atr == 0:
            return None

        recent = ohlcv.iloc[-self._lookback:]
        demand_zones, supply_zones = self._find_zones(recent, curr_atr)

        curr_price = float(ohlcv["close"].iloc[-1])
        tol = curr_price * self._zone_tolerance

        # Check demand zones (long entries)
        for zone_price in reversed(demand_zones):
            if abs(curr_price - zone_price) <= tol and curr_price >= zone_price:
                confidence = round(min(0.8, 0.55 + curr_atr / curr_price * 50), 3)
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "zone_price": zone_price,
                    "zone_type": "demand",
                }

        # Check supply zones (short entries)
        for zone_price in reversed(supply_zones):
            if abs(curr_price - zone_price) <= tol and curr_price <= zone_price:
                confidence = round(min(0.8, 0.55 + curr_atr / curr_price * 50), 3)
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": curr_price,
                    "atr": curr_atr,
                    "confidence": confidence,
                    "strategy": self._STRATEGY_NAME,
                    "timeframe": self._timeframe,
                    "zone_price": zone_price,
                    "zone_type": "supply",
                }

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No supply/demand zone interaction")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            zone = sig["zone_price"]

            if direction == "long":
                stop_loss = zone - atr * 1.0
                take_profit = entry + atr * 3.0
            else:
                stop_loss = zone + atr * 1.0
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"S/D zone {sig['zone_type']} {direction}: "
                    f"zone={zone:.4f}, price={entry:.4f}, ATR={atr:.6f}"
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
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 70:
            return True
        if side == "short" and rsi < 30:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.015
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.03, sl_pct),
            "take_profit_pct": min(0.09, sl_pct * 3.0),
            "leverage": 3,
        }
