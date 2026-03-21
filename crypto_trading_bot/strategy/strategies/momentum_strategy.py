"""Momentum strategy — EMA(9/21) crossover + RSI(14) on 15-minute candles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """EMA crossover (9/21) + RSI(14) momentum strategy.

    Entry conditions
    ----------------
    * **Long**: fast EMA crosses above slow EMA AND ``50 < RSI < 70``.
    * **Short**: fast EMA crosses below slow EMA AND ``30 < RSI < 50``.

    Stop-loss / take-profit are sized dynamically using ATR(14).
    Primary timeframe is 15 m; a 1-hour EMA alignment is required as
    confirmation before a signal is accepted in :meth:`generate_signal`.
    """

    _STRATEGY_NAME = "momentum"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        confirmation_timeframe: str = "1h",
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._rsi_period = rsi_period
        self._atr_period = atr_period
        self._confirmation_timeframe = confirmation_timeframe

    # ------------------------------------------------------------------
    # Core analysis (accepts a plain DataFrame — no exchange required)
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """Detect an EMA crossover + RSI signal from *ohlcv* data.

        Returns a signal dict or ``None`` when no signal is found.
        """
        min_rows = self._slow_ema + self._rsi_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]

        ema_fast = ta.ema(closes, length=self._fast_ema)
        ema_slow = ta.ema(closes, length=self._slow_ema)
        rsi_series = ta.rsi(closes, length=self._rsi_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if ema_fast is None or ema_slow is None or rsi_series is None or atr_series is None:
            return None

        curr_fast = float(ema_fast.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        curr_slow = float(ema_slow.iloc[-1])
        prev_slow = float(ema_slow.iloc[-2])
        curr_rsi = float(rsi_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        entry_price = float(closes.iloc[-1])

        if pd.isna(curr_fast) or pd.isna(prev_fast) or pd.isna(curr_slow) or pd.isna(prev_slow):
            return None
        if pd.isna(curr_rsi) or pd.isna(curr_atr):
            return None

        cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if cross_up and 50.0 < curr_rsi < 70.0:
            confidence = self._compute_confidence(curr_rsi, curr_fast, curr_slow, "long")
            return {
                "symbol": symbol,
                "direction": "long",
                "entry_price": entry_price,
                "atr": curr_atr,
                "confidence": confidence,
                "strategy": self._STRATEGY_NAME,
                "timeframe": self._timeframe,
            }

        if cross_down and 30.0 < curr_rsi < 50.0:
            confidence = self._compute_confidence(curr_rsi, curr_fast, curr_slow, "short")
            return {
                "symbol": symbol,
                "direction": "short",
                "entry_price": entry_price,
                "atr": curr_atr,
                "confidence": confidence,
                "strategy": self._STRATEGY_NAME,
                "timeframe": self._timeframe,
            }

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        """Fetch OHLCV data, apply :meth:`analyze`, and validate with 1-hour confirmation."""
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No EMA crossover + RSI signal on 15m")

            # 1-hour trend confirmation — fast EMA must align with direction
            ohlcv_1h = await self._get_ohlcv(symbol, self._confirmation_timeframe, limit=60)
            if not ohlcv_1h.empty and len(ohlcv_1h) >= self._slow_ema + 5:
                closes_1h = ohlcv_1h["close"]
                ema_fast_1h = ta.ema(closes_1h, length=self._fast_ema)
                ema_slow_1h = ta.ema(closes_1h, length=self._slow_ema)
                if ema_fast_1h is not None and ema_slow_1h is not None:
                    f1h = float(ema_fast_1h.iloc[-1])
                    s1h = float(ema_slow_1h.iloc[-1])
                    if not (pd.isna(f1h) or pd.isna(s1h)):
                        if sig["direction"] == "long" and f1h < s1h:
                            return self._neutral_signal(symbol, "1h trend opposes long signal")
                        if sig["direction"] == "short" and f1h > s1h:
                            return self._neutral_signal(symbol, "1h trend opposes short signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry_price = sig["entry_price"]
            sl_mult, tp_mult = 1.5, 3.0
            if direction == "long":
                stop_loss = entry_price - atr * sl_mult
                take_profit = entry_price + atr * tp_mult
            else:
                stop_loss = entry_price + atr * sl_mult
                take_profit = entry_price - atr * tp_mult

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"EMA{self._fast_ema}/{self._slow_ema} crossover "
                    f"({sig['direction']}), "
                    f"ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close when the EMA alignment reverses or RSI reaches an extreme."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        fast = self._calculate_ema(closes, self._fast_ema)
        slow = self._calculate_ema(closes, self._slow_ema)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and (fast < slow or rsi > 75):
            return True
        if side == "short" and (fast > slow or rsi < 25):
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 3,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(rsi: float, fast: float, slow: float, direction: str) -> float:
        """Derive a [0, 1] confidence score from RSI value and EMA spread."""
        ema_spread = abs(fast - slow) / slow if slow > 0 else 0.0
        if direction == "long":
            rsi_score = (rsi - 50.0) / 20.0  # 0 → 1 as RSI goes 50 → 70
        else:
            rsi_score = (50.0 - rsi) / 20.0  # 0 → 1 as RSI goes 50 → 30
        confidence = 0.5 + 0.3 * min(1.0, rsi_score) + 0.2 * min(1.0, ema_spread * 100)
        return round(min(1.0, max(0.0, confidence)), 3)
