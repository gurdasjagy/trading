from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class DeltaDivergenceStrategy(BaseStrategy):
    """Cumulative delta vs price divergence signal."""

    _STRATEGY_NAME = "delta_divergence"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.5, "ranging": 0.5}
        self._min_rows = 40

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        open_ = df["open"]
        volume = df["volume"]
        delta = (close - open_) * volume
        cum_delta = delta.rolling(20).sum()
        cd = float(cum_delta.iloc[-1])
        cd_prev = float(cum_delta.iloc[-5])
        price_chg = float(close.iloc[-1]) - float(close.iloc[-5])
        if np.isnan(cd) or np.isnan(cd_prev):
            return None
        delta_chg = cd - cd_prev
        if delta_chg > 0 and abs(price_chg) < float(close.rolling(20).std().iloc[-1]) * 0.5:
            direction = "long"
            confidence = round(min(0.9, 0.5 + abs(delta_chg) / (abs(cd_prev) + 1e-9) * 0.3), 3)
        elif delta_chg < 0 and abs(price_chg) < float(close.rolling(20).std().iloc[-1]) * 0.5:
            direction = "short"
            confidence = round(min(0.9, 0.5 + abs(delta_chg) / (abs(cd_prev) + 1e-9) * 0.3), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Delta divergence: delta_chg={delta_chg:.2f}, price_chg={price_chg:.4f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=50)
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
        df = await self._get_ohlcv(symbol, limit=50)
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
