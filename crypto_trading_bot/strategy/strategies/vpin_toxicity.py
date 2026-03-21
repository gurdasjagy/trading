from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class VPINToxicityStrategy(BaseStrategy):
    """VPIN-inspired order flow toxicity measure."""

    _STRATEGY_NAME = "vpin_toxicity"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.6, "ranging": 0.4}
        self._min_rows = 30

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        open_ = df["open"]
        volume = df["volume"]
        buy_vol = volume.where(close >= open_, 0.0)
        sell_vol = volume.where(close < open_, 0.0)
        window = 20
        buy_roll = buy_vol.rolling(window).sum()
        sell_roll = sell_vol.rolling(window).sum()
        total_roll = (buy_roll + sell_roll + 1e-9)
        toxicity = abs(buy_roll - sell_roll) / total_roll
        t = float(toxicity.iloc[-1])
        if np.isnan(t):
            return None
        if t > 0.7:
            return None
        net_flow = float((buy_roll - sell_roll).iloc[-1])
        direction = "long" if net_flow > 0 else "short"
        confidence = round(min(0.9, 0.4 + (1.0 - t) * 0.5), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"VPIN toxicity={t:.2f}, net_flow={net_flow:.2f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=40)
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
        df = await self._get_ohlcv(symbol, limit=40)
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
