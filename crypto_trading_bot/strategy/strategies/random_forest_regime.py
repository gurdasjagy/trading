from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class RandomForestRegimeStrategy(BaseStrategy):
    """Decision-tree proxy regime classification."""

    _STRATEGY_NAME = "random_forest_regime"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.6, "ranging": 0.4}
        self._min_rows = 40

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        log_ret = np.log(close / close.shift(1))
        vol_level = float(log_ret.rolling(10).std().iloc[-1])
        high_vol = vol_level > float(log_ret.rolling(30).std().quantile(0.7) if len(log_ret.dropna()) >= 30 else log_ret.std())
        ema9 = self._calculate_ema(close.tolist(), 9)
        ema21 = self._calculate_ema(close.tolist(), 21)
        trend_up = ema9 > ema21
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        high_volume = float(volume.iloc[-1]) > avg_vol * 1.2
        if not high_vol and trend_up and high_volume:
            direction = "long"
            confidence = 0.75
        elif not high_vol and not trend_up and high_volume:
            direction = "short"
            confidence = 0.75
        elif high_vol and trend_up:
            direction = "long"
            confidence = 0.6
        elif high_vol and not trend_up:
            direction = "short"
            confidence = 0.6
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"RF regime: high_vol={high_vol}, trend_up={trend_up}, high_vol_bars={high_volume}",
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
