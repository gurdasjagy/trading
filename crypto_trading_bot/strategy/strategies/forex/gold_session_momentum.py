"""Gold Session Momentum strategy — adapt to Asian/London/NY session characteristics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# GMT session boundaries (hour, inclusive start / exclusive end)
_SESSIONS = {
    "asian":  (0, 8),
    "london": (8, 16),
    "ny":     (13, 22),
}


class GoldSessionMomentumStrategy(BaseStrategy):
    """Session-Based Momentum strategy for Gold.

    Different sessions have different gold characteristics:

    * **Asian session** (00:00–08:00 GMT): range-bound; no directional
      signal — return ``None`` (strategy sits out).
    * **London session** (08:00–16:00 GMT): breakout momentum; trade EMA
      crossover in the first 2 hours.
    * **New York session** (13:00–22:00 GMT): continuation or reversal;
      use RSI divergence from London direction.

    Signals are only generated during London and NY sessions.
    """

    _STRATEGY_NAME = "gold_session_momentum"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        enabled: bool = True,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._rsi_period = rsi_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _current_session() -> Optional[str]:
        now_h = datetime.now(timezone.utc).hour
        # NY overlaps with London; check London first
        for session, (start, end) in _SESSIONS.items():
            if start <= now_h < end:
                return session
        return None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._slow_ema + self._rsi_period + 10
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        session = self._current_session()
        if session == "asian" or session is None:
            return None  # Sit out Asian session

        closes = ohlcv["close"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        fast = self._calculate_ema(closes, self._fast_ema)
        slow = self._calculate_ema(closes, self._slow_ema)
        prev_fast = self._calculate_ema(closes[:-1], self._fast_ema)
        prev_slow = self._calculate_ema(closes[:-1], self._slow_ema)
        rsi = self._calculate_rsi(closes, self._rsi_period)
        curr_price = closes[-1]

        direction: Optional[str] = None

        if session == "london":
            # London: trade EMA breakout momentum
            if prev_fast <= prev_slow and fast > slow and rsi < 70:
                direction = "long"
            elif prev_fast >= prev_slow and fast < slow and rsi > 30:
                direction = "short"

        elif session == "ny":
            # NY: continuation only if RSI confirms trend; reversal if over-extended
            if fast > slow and rsi > 55 and rsi < 75:
                direction = "long"
            elif fast < slow and rsi < 45 and rsi > 25:
                direction = "short"
            # NY reversal: RSI over-extended
            elif rsi >= 75 and fast > slow:
                direction = "short"
            elif rsi <= 25 and fast < slow:
                direction = "long"

        if direction is None:
            return None

        ema_sep = abs(fast - slow) / atr
        session_weight = 0.80 if session == "london" else 0.70
        confidence = round(min(0.9, session_weight * (0.6 + min(ema_sep, 0.5) * 0.4)), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "session": session,
            "fast_ema": round(fast, 4),
            "slow_ema": round(slow, 4),
            "rsi": round(rsi, 2),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No session momentum signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold {sig['session']} session {direction}: "
                    f"EMA_fast={sig['fast_ema']:.4f}, RSI={sig['rsi']:.1f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        session = self._current_session()
        if session is None:
            return True  # Close outside known sessions
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 3,
        }
