from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class XGBoostClassifierStrategy(BaseStrategy):
    """Rule-ensemble classifier proxy using 5 technical indicators."""

    _STRATEGY_NAME = "xgboost_classifier"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.6, "ranging": 0.4}
        self._min_rows = 50

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        close_list = close.tolist()
        rsi = self._calculate_rsi(close_list, 14)
        ema12 = self._calculate_ema(close_list, 12)
        ema26 = self._calculate_ema(close_list, 26)
        macd_val = ema12 - ema26
        bb_mean = float(close.rolling(20).mean().iloc[-1])
        bb_std = float(close.rolling(20).std().iloc[-1])
        cur = float(close.iloc[-1])
        atr = self._calculate_atr(df)
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1]) / (avg_vol + 1e-9)
        if any(np.isnan(v) for v in [rsi, ema12, ema26, bb_mean, bb_std]):
            return None
        rsi_vote = 1 if rsi < 40 else (-1 if rsi > 60 else 0)
        macd_vote = 1 if macd_val > 0 else -1
        bb_z = (cur - bb_mean) / (bb_std + 1e-9)
        bb_vote = 1 if bb_z < -1 else (-1 if bb_z > 1 else 0)
        atr_vote = 1 if atr > 0 else 0
        vol_vote = 1 if vol_ratio > 1.2 else (-1 if vol_ratio < 0.8 else 0)
        weighted = rsi_vote * 0.3 + macd_vote * 0.3 + bb_vote * 0.2 + atr_vote * 0.1 + vol_vote * 0.1
        if weighted > 0.15:
            direction = "long"
        elif weighted < -0.15:
            direction = "short"
        else:
            return None
        confidence = round(min(0.9, 0.5 + abs(weighted) * 0.5), 3)
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": cur, "atr": atr,
            "reasoning": f"XGB ensemble weighted={weighted:.2f}",
        }

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            df = await self._get_ohlcv(symbol, self._timeframe, limit=60)
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
        df = await self._get_ohlcv(symbol, limit=60)
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
