from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class VolClusteringStrategy(BaseStrategy):
    """GARCH(1,1) proxy for volatility clustering position sizing."""

    _STRATEGY_NAME = "vol_clustering"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.5, "ranging": 0.5}
        self._min_rows = 30
        self._alpha = 0.1
        self._beta = 0.85

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        log_ret = np.log(close / close.shift(1)).fillna(0)
        sq_ret = log_ret ** 2
        vol_long = float(sq_ret.rolling(20).mean().iloc[-1])
        last_sq = float(sq_ret.iloc[-1])
        pred_var = self._alpha * last_sq + self._beta * vol_long
        pred_vol = float(np.sqrt(pred_var + 1e-9))
        vol_pct = float(sq_ret.rolling(20).mean().rank(pct=True).iloc[-1])
        if np.isnan(vol_pct):
            return None
        if vol_pct > 0.8:
            return None
        close_list = close.tolist()
        ema9 = self._calculate_ema(close_list, 9)
        ema21 = self._calculate_ema(close_list, 21)
        direction = "long" if ema9 > ema21 else "short"
        confidence = round(min(0.9, 0.4 + (1.0 - vol_pct) * 0.5), 3)
        atr = self._calculate_atr(df)
        entry = float(close.iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"GARCH pred_vol={pred_vol:.6f}, vol_pct={vol_pct:.2f}",
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
