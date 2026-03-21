"""Gold Ichimoku Cloud strategy — cloud breakouts and TK-crosses for gold."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldIchimokuStrategy(BaseStrategy):
    """Gold Ichimoku Cloud strategy.

    Ichimoku works exceptionally well on gold due to its trending nature.
    This strategy trades:

    * **Cloud breakout**: price crossing above/below the Kumo (cloud).
    * **TK cross**: Tenkan-sen (conversion line) crossing Kijun-sen (base line)
      while price is on the correct side of the cloud.

    Ichimoku parameters
    -------------------
    * Tenkan-sen (conversion):  9-period midpoint
    * Kijun-sen (base):        26-period midpoint
    * Senkou Span A:           (Tenkan + Kijun) / 2, displaced +26
    * Senkou Span B:           52-period midpoint, displaced +26
    * Chikou Span:             Close displaced -26
    """

    _STRATEGY_NAME = "gold_ichimoku"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
        displacement: int = 26,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._tenkan = tenkan
        self._kijun = kijun
        self._senkou_b = senkou_b
        self._displacement = displacement
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _midpoint(highs: pd.Series, lows: pd.Series, period: int) -> pd.Series:
        return (highs.rolling(period).max() + lows.rolling(period).min()) / 2.0

    def _compute_ichimoku(
        self, ohlcv: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        highs = ohlcv["high"]
        lows = ohlcv["low"]

        tenkan = self._midpoint(highs, lows, self._tenkan)
        kijun = self._midpoint(highs, lows, self._kijun)
        span_a = (tenkan + kijun) / 2.0
        span_b = self._midpoint(highs, lows, self._senkou_b)

        idx = -1
        disp = self._displacement
        if len(ohlcv) < self._senkou_b + disp:
            return None

        t = float(tenkan.iloc[idx])
        k = float(kijun.iloc[idx])

        # Cloud values are the span at the displaced index
        cloud_idx = -1 - disp if len(span_a) > disp else -1
        sa = float(span_a.iloc[cloud_idx])
        sb = float(span_b.iloc[cloud_idx])

        if any(pd.isna(v) for v in (t, k, sa, sb)):
            return None

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)

        prev_t = float(tenkan.iloc[-2]) if len(tenkan) >= 2 else t
        prev_k = float(kijun.iloc[-2]) if len(kijun) >= 2 else k

        return {
            "tenkan": t,
            "kijun": k,
            "prev_tenkan": prev_t,
            "prev_kijun": prev_k,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "span_a": sa,
            "span_b": sb,
        }

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._senkou_b + self._displacement + 5
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        ichi = self._compute_ichimoku(ohlcv)
        if ichi is None:
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        prev_price = float(ohlcv["close"].iloc[-2])

        cloud_top = ichi["cloud_top"]
        cloud_bottom = ichi["cloud_bottom"]

        direction: Optional[str] = None
        signal_type: str = ""

        # Cloud breakout
        if prev_price <= cloud_top < curr_price:
            direction = "long"
            signal_type = "cloud_breakout"
        elif prev_price >= cloud_bottom > curr_price:
            direction = "short"
            signal_type = "cloud_breakout"

        # TK cross (only if price is already above/below cloud)
        if direction is None:
            tk_bull = ichi["prev_tenkan"] <= ichi["prev_kijun"] and ichi["tenkan"] > ichi["kijun"]
            tk_bear = ichi["prev_tenkan"] >= ichi["prev_kijun"] and ichi["tenkan"] < ichi["kijun"]
            if tk_bull and curr_price > cloud_top:
                direction = "long"
                signal_type = "tk_cross"
            elif tk_bear and curr_price < cloud_bottom:
                direction = "short"
                signal_type = "tk_cross"

        if direction is None:
            return None

        # Confidence: cloud breakout is stronger; TK cross slightly weaker
        base_conf = 0.75 if signal_type == "cloud_breakout" else 0.62
        cloud_size = cloud_top - cloud_bottom
        size_factor = min(1.0, cloud_size / atr / 2.0)
        confidence = round(min(0.9, base_conf + size_factor * 0.15), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "signal_type": signal_type,
            "tenkan": round(ichi["tenkan"], 4),
            "kijun": round(ichi["kijun"], 4),
            "cloud_top": round(cloud_top, 4),
            "cloud_bottom": round(cloud_bottom, 4),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=300)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Ichimoku signal on gold")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = sig["cloud_bottom"] - atr * 0.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = sig["cloud_top"] + atr * 0.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold Ichimoku {sig['signal_type']} {direction}: "
                    f"T={sig['tenkan']:.4f}, K={sig['kijun']:.4f}, "
                    f"cloud=[{sig['cloud_bottom']:.4f},{sig['cloud_top']:.4f}]"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=200)
        if ohlcv.empty:
            return False
        ichi = self._compute_ichimoku(ohlcv)
        if ichi is None:
            return False
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        # Close if price re-enters the cloud
        if side == "long" and curr_price < ichi["cloud_top"]:
            return True
        if side == "short" and curr_price > ichi["cloud_bottom"]:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=200)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 2.5),
            "leverage": 3,
        }
