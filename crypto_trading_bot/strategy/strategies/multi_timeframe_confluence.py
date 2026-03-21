"""Multi-Timeframe Confluence strategy — only trade when 3+ TFs align."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

_DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]


class MTFConfluenceStrategy(BaseStrategy):
    """Multi-Timeframe Confluence (MTF) strategy.

    Runs the same trend analysis across *timeframes* and only fires a
    signal when at least *min_confluence* timeframes show the same
    direction.

    Per-timeframe analysis uses:
    * EMA crossover (fast/slow) for direction.
    * RSI to confirm trend strength.
    """

    _STRATEGY_NAME = "mtf_confluence"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        timeframes: Optional[List[str]] = None,
        min_confluence: int = 3,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._timeframes = timeframes or _DEFAULT_TIMEFRAMES
        self._min_confluence = min_confluence
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._rsi_period = rsi_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Single-timeframe analysis
    # ------------------------------------------------------------------

    def _analyze_single_tf(self, ohlcv: pd.DataFrame) -> Optional[str]:
        """Return 'long', 'short', or None for a single OHLCV DataFrame."""
        if len(ohlcv) < self._slow_ema + self._rsi_period + 5:
            return None

        closes = ohlcv["close"]
        fast = ta.ema(closes, length=self._fast_ema)
        slow = ta.ema(closes, length=self._slow_ema)
        rsi_s = ta.rsi(closes, length=self._rsi_period)

        if fast is None or slow is None or rsi_s is None:
            return None

        curr_fast = float(fast.iloc[-1])
        curr_slow = float(slow.iloc[-1])
        curr_rsi = float(rsi_s.iloc[-1])

        if pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(curr_rsi):
            return None

        if curr_fast > curr_slow and curr_rsi > 50:
            return "long"
        if curr_fast < curr_slow and curr_rsi < 50:
            return "short"
        return None

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """Analyse just the primary OHLCV (used when exchange is unavailable)."""
        direction = self._analyze_single_tf(ohlcv)
        if direction is None:
            return None

        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
        if atr_series is None:
            return None
        curr_atr = float(atr_series.iloc[-1])
        if pd.isna(curr_atr):
            return None

        curr_price = float(ohlcv["close"].iloc[-1])
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": 0.6,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "confluence_count": 1,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            signals: List[str] = []
            atrs: List[float] = []
            prices: List[float] = []

            for tf in self._timeframes:
                ohlcv = await self._get_ohlcv(symbol, tf, limit=100)
                if ohlcv.empty:
                    continue
                direction = self._analyze_single_tf(ohlcv)
                if direction is not None:
                    signals.append(direction)
                    atr_s = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=self._atr_period)
                    if atr_s is not None:
                        v = float(atr_s.iloc[-1])
                        if not pd.isna(v):
                            atrs.append(v)
                    prices.append(float(ohlcv["close"].iloc[-1]))

            if not signals:
                return self._neutral_signal(symbol, "No MTF signals generated")

            long_count = signals.count("long")
            short_count = signals.count("short")
            total_required = min(self._min_confluence, len(self._timeframes))

            if long_count >= total_required:
                direction = "long"
                confluence = long_count
            elif short_count >= total_required:
                direction = "short"
                confluence = short_count
            else:
                return self._neutral_signal(
                    symbol, f"Insufficient confluence: long={long_count}, short={short_count}"
                )

            avg_atr = sum(atrs) / len(atrs) if atrs else 0.0
            entry = prices[-1] if prices else 0.0
            confidence = round(min(0.92, 0.5 + confluence / len(self._timeframes) * 0.42), 3)

            if direction == "long":
                stop_loss = entry - avg_atr * 1.5
                take_profit = entry + avg_atr * 3.0
            else:
                stop_loss = entry + avg_atr * 1.5
                take_profit = entry - avg_atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=confidence,
                confidence=confidence,
                strategy_name=self.name,
                reasoning=(
                    f"MTF confluence {direction} ({confluence}/{len(self._timeframes)} TFs), "
                    f"TFs={self._timeframes}, ATR={avg_atr:.6f}"
                ),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        side = str(getattr(position, "side", "long")).lower()
        # Close if majority of TFs now disagree
        opposing = 0
        total = 0
        for tf in self._timeframes:
            ohlcv = await self._get_ohlcv(symbol, tf, limit=60)
            if ohlcv.empty:
                continue
            d = self._analyze_single_tf(ohlcv)
            if d is not None:
                total += 1
                if d != side:
                    opposing += 1
        if total > 0 and opposing >= total / 2:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.06,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }
