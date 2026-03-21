from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class CopulaDependenceStrategy(BaseStrategy):
    """Tail dependence between volume and range via rank correlation."""

    _STRATEGY_NAME = "copula_dependence"

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
        hl_range = (df["high"] - df["low"]).values
        volume = df["volume"].values
        n = len(hl_range)
        rank_range = np.argsort(np.argsort(hl_range)) / (n - 1)
        rank_vol = np.argsort(np.argsort(volume)) / (n - 1)
        spearman_corr = float(np.corrcoef(rank_range, rank_vol)[0, 1])
        last_range_pct = float(rank_range[-1])
        last_vol_pct = float(rank_vol[-1])
        if np.isnan(spearman_corr):
            return None
        if last_range_pct > 0.8 and last_vol_pct > 0.8:
            close_chg = float(df["close"].iloc[-1]) - float(df["close"].iloc[-2])
            direction = "long" if close_chg > 0 else "short"
            confidence = round(0.5 + spearman_corr * 0.3 + (last_vol_pct - 0.8) * 0.5, 3)
            confidence = round(min(0.9, max(0.5, confidence)), 3)
        else:
            return None
        atr = self._calculate_atr(df)
        entry = float(df["close"].iloc[-1])
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": entry, "atr": atr,
            "reasoning": f"Tail dependence rank_vol={last_vol_pct:.2f}, rank_range={last_range_pct:.2f}",
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
