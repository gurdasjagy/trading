"""Gold Scalping strategy — quick scalps on 1m/5m using EMA crossovers + RSI + spread filter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# Maximum spread in ATR multiples to allow a trade (proxy for 3-pip limit)
_MAX_SPREAD_ATR = 0.15


class GoldScalpingStrategy(BaseStrategy):
    """Gold Scalping strategy (1m / 5m timeframes).

    Quick scalps using EMA crossovers + RSI + spread filter.  Only
    enters when the implied spread is below ``max_spread_atr`` × ATR
    (acting as proxy for the "< 3 pips" spread rule).

    Signal logic
    ------------
    * Fast EMA (9) crosses above slow EMA (21) + RSI 40–65 range → long.
    * Fast EMA (9) crosses below slow EMA (21) + RSI 35–60 range → short.
    * Spread filter: (high - low) on last bar < ``max_spread_atr`` × ATR.
    * Confidence scales with EMA separation and RSI position.
    """

    _STRATEGY_NAME = "gold_scalping"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "5m",
        enabled: bool = True,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        max_spread_atr: float = _MAX_SPREAD_ATR,
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
        self._max_spread_atr = max_spread_atr
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._slow_ema + self._rsi_period + 10
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        # Spread filter: use last bar's high-low as spread proxy
        last_spread = float(ohlcv["high"].iloc[-1]) - float(ohlcv["low"].iloc[-1])
        if last_spread > self._max_spread_atr * atr:
            return None

        fast = self._calculate_ema(closes, self._fast_ema)
        slow = self._calculate_ema(closes, self._slow_ema)
        prev_fast = self._calculate_ema(closes[:-1], self._fast_ema)
        prev_slow = self._calculate_ema(closes[:-1], self._slow_ema)

        rsi = self._calculate_rsi(closes, self._rsi_period)
        curr_price = closes[-1]

        direction: Optional[str] = None

        if prev_fast <= prev_slow and fast > slow and 40 <= rsi <= 65:
            direction = "long"
        elif prev_fast >= prev_slow and fast < slow and 35 <= rsi <= 60:
            direction = "short"

        if direction is None:
            return None

        ema_sep = abs(fast - slow) / atr
        rsi_mid_dist = abs(rsi - 50) / 50.0
        confidence = round(min(0.85, 0.52 + min(ema_sep, 0.5) * 0.4 + rsi_mid_dist * 0.1), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "fast_ema": round(fast, 4),
            "slow_ema": round(slow, 4),
            "rsi": round(rsi, 2),
            "spread": round(last_spread, 4),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No gold scalp signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            # Tight scalp targets: 1 × ATR SL, 2 × ATR TP
            if direction == "long":
                stop_loss = entry - atr * 1.0
                take_profit = entry + atr * 2.0
            else:
                stop_loss = entry + atr * 1.0
                take_profit = entry - atr * 2.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Gold scalp {direction}: EMA({self._fast_ema})={sig['fast_ema']:.4f}, "
                    f"EMA({self._slow_ema})={sig['slow_ema']:.4f}, RSI={sig['rsi']:.1f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=5,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        # Close on opposite crossover
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and sig["direction"] == "short":
            return True
        if side == "short" and sig["direction"] == "long":
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.0) if last_price > 0 else 0.01
        return {
            "position_size_pct": 0.03,
            "stop_loss_pct": min(0.02, sl_pct),
            "take_profit_pct": min(0.04, sl_pct * 2.0),
            "leverage": 5,
        }
