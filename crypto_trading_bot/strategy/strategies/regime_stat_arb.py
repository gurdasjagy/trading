from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class RegimeStatArbStrategy(BaseStrategy):
    """Hidden regime detection via volatility percentile with adaptive signal."""

    _STRATEGY_NAME = "regime_stat_arb"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.5, "ranging": 0.5}
        self._min_rows = 60

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        roll_vol = log_ret.rolling(10).std().fillna(0)
        vol_pct = float(roll_vol.rank(pct=True).iloc[-1])
        high_vol_regime = vol_pct > 0.7
        close_list = close.tolist()
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        if high_vol_regime:
            ema9 = self._calculate_ema(close_list, 9)
            ema21 = self._calculate_ema(close_list, 21)
            direction = "long" if ema9 > ema21 else "short"
            confidence = round(0.5 + (vol_pct - 0.7) * 0.8, 3)
        else:
            roll_mean = close.rolling(20).mean()
            roll_std = close.rolling(20).std()
            z = float(((close - roll_mean) / (roll_std + 1e-9)).iloc[-1])
            if np.isnan(z):
                return None
            if z < -2.0:
                direction = "long"
            elif z > 2.0:
                direction = "short"
            else:
                return None
            confidence = round(min(0.9, 0.5 + (abs(z) - 2.0) * 0.2), 3)
        confidence = round(min(0.9, confidence), 3)
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Regime vol_pct={vol_pct:.2f}, high_vol={high_vol_regime}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=70)
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
        df = await self._get_ohlcv(symbol, limit=70)
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
