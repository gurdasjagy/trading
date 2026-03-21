"""Gold Supply/Demand Zones strategy — identify institutional order zones."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldSupplyDemandStrategy(BaseStrategy):
    """Gold Supply/Demand Zone strategy.

    Identifies institutional supply (overhead resistance) and demand
    (underlying support) zones where large orders were placed.  Gold has
    well-defined zones due to central bank and institutional activity.

    Zone detection
    --------------
    * **Demand zone**: a candle with a long lower wick (wick ≥ ``wick_ratio``
      × body) followed by a strong up-move signals an area where buyers
      stepped in — this is a demand zone.
    * **Supply zone**: opposite pattern (long upper wick + down-move).
    * Price re-entering a demand zone → long.
    * Price re-entering a supply zone → short.
    """

    _STRATEGY_NAME = "gold_supply_demand"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        lookback: int = 100,
        zone_tolerance: float = 0.003,
        wick_ratio: float = 2.0,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._lookback = lookback
        self._zone_tolerance = zone_tolerance
        self._wick_ratio = wick_ratio
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_zones(
        self, ohlcv: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """Return lists of demand zone prices and supply zone prices."""
        demand_zones: List[float] = []
        supply_zones: List[float] = []

        df = ohlcv.iloc[-self._lookback:]

        for i in range(1, len(df) - 1):
            row = df.iloc[i]
            o, h, lo, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            body = abs(c - o)
            if body == 0:
                continue
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - lo

            # Demand zone: long lower wick candle followed by up-move
            if lower_wick >= self._wick_ratio * body:
                next_close = float(df.iloc[i + 1]["close"])
                if next_close > c:
                    demand_zones.append(lo)

            # Supply zone: long upper wick candle followed by down-move
            if upper_wick >= self._wick_ratio * body:
                next_close = float(df.iloc[i + 1]["close"])
                if next_close < c:
                    supply_zones.append(h)

        return demand_zones, supply_zones

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._lookback + self._atr_period + 5
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        demand_zones, supply_zones = self._find_zones(ohlcv)

        if not demand_zones and not supply_zones:
            return None

        tol = curr_price * self._zone_tolerance

        hit_demand: Optional[float] = None
        for zone in demand_zones:
            if abs(curr_price - zone) <= tol:
                hit_demand = zone
                break

        hit_supply: Optional[float] = None
        for zone in supply_zones:
            if abs(curr_price - zone) <= tol:
                hit_supply = zone
                break

        if hit_demand is None and hit_supply is None:
            return None

        if hit_demand is not None and hit_supply is not None:
            # Prefer the zone that's closest
            if abs(curr_price - hit_demand) <= abs(curr_price - hit_supply):
                hit_supply = None
            else:
                hit_demand = None

        if hit_demand is not None:
            direction = "long"
            zone_price = hit_demand
            zone_count = sum(1 for z in demand_zones if abs(z - zone_price) <= tol * 2)
        else:
            direction = "short"
            zone_price = hit_supply  # type: ignore[assignment]
            zone_count = sum(1 for z in supply_zones if abs(z - zone_price) <= tol * 2)

        # More touches = stronger zone
        confidence = round(min(0.9, 0.55 + min(zone_count - 1, 3) * 0.1), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "zone_price": round(zone_price, 4),
            "zone_count": zone_count,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No supply/demand zone hit")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["zone_price"] - atr * 1.0
                take_profit = entry + atr * 3.0
            else:
                stop_loss = sig["zone_price"] + atr * 1.0
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold S/D zone {direction}: zone={sig['zone_price']:.4f}, "
                    f"touches={sig['zone_count']}, ATR={atr:.4f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.015
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 3.0),
            "leverage": 3,
        }
