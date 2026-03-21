"""NFP / News Trading strategy for Gold — trade reactions to major economic events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class NFPNewsStrategy(BaseStrategy):
    """Non-Farm Payrolls / News Trading strategy for XAU/USD.

    Trades gold around major economic events (NFP, CPI, FOMC).  In
    practice, event timing is supplied via an external ``news_events``
    list; when no news API key is configured, the strategy returns no
    signal and is effectively skipped.

    Signal logic
    ------------
    * If a high-impact event occurred in the last 30 minutes and gold has
      moved more than 1 × ATR in one direction, trade momentum continuation.
    * Confidence scales with the magnitude of the move relative to ATR.
    * If no event data is provided, return ``None`` (strategy disabled).
    """

    _STRATEGY_NAME = "nfp_news_strategy"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "5m",
        enabled: bool = True,
        atr_period: int = 14,
        move_atr_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._atr_period = atr_period
        self._move_atr_multiplier = move_atr_multiplier

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        symbol: str = "",
        news_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyse OHLCV data with optional news event context.

        Args:
            ohlcv: OHLCV DataFrame.
            symbol: Trading pair symbol.
            news_events: List of dicts with keys ``"time"`` (datetime),
                ``"impact"`` (``"high"``/``"medium"``/``"low"``), and
                ``"title"`` (str).  Pass ``None`` or an empty list to
                skip the strategy (no news API configured).
        """
        if news_events is None:
            return None  # No news API — skip strategy

        if ohlcv is None or len(ohlcv) < 50:
            return None

        closes = ohlcv["close"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        curr_price = closes[-1]
        now = datetime.now(timezone.utc)

        # Find high-impact event in last 30 minutes
        recent_event: Optional[Dict[str, Any]] = None
        for event in news_events:
            ev_time = event.get("time")
            if ev_time is None:
                continue
            if not isinstance(ev_time, datetime):
                try:
                    ev_time = pd.Timestamp(ev_time).to_pydatetime()
                except Exception:
                    continue
            if ev_time.tzinfo is None:
                ev_time = ev_time.replace(tzinfo=timezone.utc)
            minutes_ago = (now - ev_time).total_seconds() / 60
            if 0 <= minutes_ago <= 30 and event.get("impact", "low") == "high":
                recent_event = event
                break

        if recent_event is None:
            return None

        # Measure the post-event move: compare current price with the open
        # of the candle that was current when the event fired
        pre_event_price = closes[-6] if len(closes) >= 6 else closes[0]
        move = curr_price - pre_event_price
        move_in_atr = move / atr

        if abs(move_in_atr) < self._move_atr_multiplier:
            return None

        direction = "long" if move > 0 else "short"
        confidence = round(min(0.9, 0.5 + min(abs(move_in_atr) - 1.0, 2.0) * 0.2), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "event_title": recent_event.get("title", "high-impact"),
            "move_atr": round(move_in_atr, 3),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            # Without live news feed, no signal can be generated
            sig = self.analyze(ohlcv, symbol, news_events=None)
            if sig is None:
                return self._neutral_signal(symbol, "No news event data available")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 2.5
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 2.5

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"NFP/News {direction}: {sig['event_title']}, "
                    f"move={sig['move_atr']:.2f}×ATR"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.03,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 3,
        }
