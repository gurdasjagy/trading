"""Time-Based strategy — session-aware trading (Asian, London, NY sessions)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# Session windows in UTC hours (start_inclusive, end_exclusive)
_SESSIONS = {
    "asian": (0, 8),
    "london": (8, 16),
    "new_york": (13, 21),
}


class TimeBasedStrategy(BaseStrategy):
    """Time-of-day / trading session strategy.

    Each major session (Asian 0–8 UTC, London 8–16 UTC, New York 13–21 UTC)
    has different volatility characteristics.  This strategy:

    * During **Asian** session: favours range-bound mean-reversion.
    * During **London** and **New York** sessions: favours breakout momentum.

    Entry conditions
    ----------------
    * Only generates signals if the current UTC time falls within an active
      session.
    * Uses RSI and EMA to confirm direction within each session's bias.
    """

    _STRATEGY_NAME = "time_based"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        active_sessions: Optional[List[str]] = None,
        ema_period: int = 20,
        rsi_period: int = 14,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._active_sessions = active_sessions or list(_SESSIONS.keys())
        self._ema_period = ema_period
        self._rsi_period = rsi_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Session detection
    # ------------------------------------------------------------------

    @staticmethod
    def _current_session() -> Optional[str]:
        hour = datetime.now(timezone.utc).hour
        active: List[str] = []
        for name, (start, end) in _SESSIONS.items():
            if start <= hour < end:
                active.append(name)
        return active[0] if active else None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = max(self._ema_period, self._rsi_period) + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        session = self._current_session()
        if session is None or session not in self._active_sessions:
            return None

        closes = ohlcv["close"]
        ema_series = ta.ema(closes, length=self._ema_period)
        rsi_series = ta.rsi(closes, length=self._rsi_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if ema_series is None or rsi_series is None or atr_series is None:
            return None

        curr_ema = float(ema_series.iloc[-1])
        curr_rsi = float(rsi_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        for val in (curr_ema, curr_rsi, curr_atr):
            if pd.isna(val):
                return None

        direction: Optional[str] = None
        confidence_base = 0.0

        if session == "asian":
            # Mean-reversion: oversold → long, overbought → short
            if curr_rsi < 35 and curr_price < curr_ema:
                direction = "long"
                confidence_base = 0.58
            elif curr_rsi > 65 and curr_price > curr_ema:
                direction = "short"
                confidence_base = 0.58

        else:
            # London / NY: breakout momentum
            prev_ema = float(ema_series.iloc[-2])
            if pd.isna(prev_ema):
                return None
            if curr_price > curr_ema and curr_rsi > 50:
                direction = "long"
                confidence_base = 0.62
            elif curr_price < curr_ema and curr_rsi < 50:
                direction = "short"
                confidence_base = 0.62

        if direction is None:
            return None

        rsi_extreme = abs(curr_rsi - 50) / 50.0
        confidence = round(min(0.85, confidence_base + rsi_extreme * 0.2), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "session": session,
            "rsi": curr_rsi,
            "ema": curr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "Outside active session or no signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]
            session = sig["session"]

            leverage = 2 if session == "asian" else 3
            sl_mult = 1.2 if session == "asian" else 1.5
            tp_mult = 1.5 if session == "asian" else 3.0

            if direction == "long":
                stop_loss = entry - atr * sl_mult
                take_profit = entry + atr * tp_mult
            else:
                stop_loss = entry + atr * sl_mult
                take_profit = entry - atr * tp_mult

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Session {session} {direction}: RSI={sig['rsi']:.1f}, "
                    f"EMA={sig['ema']:.4f}, ATR={atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=leverage,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        session = self._current_session()
        # Always close Asian session trades at session end
        if session != "asian" and data.get("session_was") == "asian":
            return True
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 70:
            return True
        if side == "short" and rsi < 30:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        session = self._current_session() or "london"
        size = 0.03 if session == "asian" else 0.05
        return {
            "position_size_pct": size,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2 if session == "asian" else 3,
        }
