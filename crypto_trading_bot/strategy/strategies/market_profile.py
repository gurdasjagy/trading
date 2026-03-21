from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MarketProfileStrategy(BaseStrategy):
    """TPO Market Profile value area low/high strategy."""

    _STRATEGY_NAME = "market_profile"

    def __init__(self, symbols: Optional[List[str]] = None, timeframe: str = "1h", enabled: bool = True) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self.regime_affinity = {"trending": 0.3, "ranging": 0.7}
        self._min_rows = 40

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        if len(df) < self._min_rows:
            return None
        close = df["close"]
        volume = df["volume"]
        prices = close.values[-20:]
        vols = volume.values[-20:]
        p_min = float(np.min(prices))
        p_max = float(np.max(prices))
        if p_max == p_min:
            return None
        bins = 20
        edges = np.linspace(p_min, p_max, bins + 1)
        vol_by_bin = np.zeros(bins)
        for p, v in zip(prices, vols):
            idx = min(int((p - p_min) / (p_max - p_min) * bins), bins - 1)
            vol_by_bin[idx] += v
        total_vol = vol_by_bin.sum()
        sorted_idx = np.argsort(vol_by_bin)[::-1]
        cum_vol = 0.0
        value_area_bins = set()
        for i in sorted_idx:
            value_area_bins.add(i)
            cum_vol += vol_by_bin[i]
            if cum_vol >= total_vol * 0.7:
                break
        va_bins = sorted(value_area_bins)
        val = float(edges[min(va_bins)])
        vah = float(edges[max(va_bins) + 1])
        cur = float(close.iloc[-1])
        atr = self._calculate_atr(df)
        if cur <= val + atr * 0.3:
            direction = "long"
            confidence = round(min(0.9, 0.5 + (val - cur + atr * 0.3) / (atr + 1e-9) * 0.2), 3)
        elif cur >= vah - atr * 0.3:
            direction = "short"
            confidence = round(min(0.9, 0.5 + (cur - vah + atr * 0.3) / (atr + 1e-9) * 0.2), 3)
        else:
            return None
        return {
            "symbol": symbol, "direction": direction, "confidence": confidence,
            "strategy": self.name, "entry_price": cur, "atr": atr,
            "reasoning": f"Market profile VAL={val:.4f}, VAH={vah:.4f}, cur={cur:.4f}",
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
