from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class DeFiTVLFlowStrategy(BaseStrategy):
    """DeFi TVL flow proxy via cumulative volume trend."""

    _STRATEGY_NAME = "defi_tvl_flow"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.6, "ranging": 0.4}
        self._min_rows = 20

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        cum_vol = volume.cumsum()
        cum_vol_ma = cum_vol.diff(10)
        trend = float(cum_vol_ma.iloc[-1])
        trend_prev = float(cum_vol_ma.iloc[-5])
        if np.isnan(trend) or np.isnan(trend_prev):
            return None
        price_mom = float(close.pct_change(5).iloc[-1])
        if np.isnan(price_mom):
            return None
        if trend > 0 and trend > trend_prev:
            direction = "long" if price_mom >= 0 else "short"
            confidence = round(min(0.9, 0.5 + (trend - trend_prev) / (abs(trend_prev) + 1e-9) * 0.1), 3)
        elif trend < 0 and trend < trend_prev:
            direction = "short"
            confidence = round(min(0.9, 0.5 + abs(trend - trend_prev) / (abs(trend_prev) + 1e-9) * 0.1), 3)
        else:
            return None
        confidence = round(min(0.9, max(0.5, confidence)), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"DeFi TVL proxy: cum_vol_trend={trend:.2f}, price_mom={price_mom:.4f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=30)
            sig = self.analyze(df, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No signal")
            return Signal(
                symbol=symbol,
                direction=sig["direction"],
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=sig["reasoning"],
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        df = await self._get_ohlcv(symbol, limit=30)
        if df.empty:
            return False
        sig = self.analyze(df, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        return (side == "long" and sig["direction"] == "short") or (
            side == "short" and sig["direction"] == "long"
        )

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        df = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(df)
        last = float(df["close"].iloc[-1]) if not df.empty else 1.0
        sl = (atr / last * 1.5) if last > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl),
            "take_profit_pct": min(0.08, sl * 2),
            "leverage": 3,
        }
