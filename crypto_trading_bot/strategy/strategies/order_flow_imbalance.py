"""Order Flow Imbalance strategy — buy/sell pressure detected from volume analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class OrderFlowImbalanceStrategy(BaseStrategy):
    """Volume-based order-flow imbalance strategy.

    Each candle's volume is classified as buy-dominated when close > open
    (up-volume) or sell-dominated when close < open (down-volume).  An
    imbalance ratio is computed over a rolling window; a strong buy
    imbalance with upward price momentum → long, and vice-versa.

    Entry conditions
    ----------------
    * **Long**: buy_vol / total_vol > *threshold* over last *window* bars AND
      price above short-term EMA.
    * **Short**: sell_vol / total_vol > *threshold* AND price below short-term EMA.
    """

    _STRATEGY_NAME = "order_flow_imbalance"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
        window: int = 20,
        imbalance_threshold: float = 0.65,
        ema_period: int = 20,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._window = window
        self._imbalance_threshold = imbalance_threshold
        self._ema_period = ema_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._window + self._ema_period + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        df = ohlcv.copy()
        df["up_vol"] = df["volume"].where(df["close"] >= df["open"], 0.0)
        df["down_vol"] = df["volume"].where(df["close"] < df["open"], 0.0)

        roll_up = df["up_vol"].rolling(self._window).sum()
        roll_down = df["down_vol"].rolling(self._window).sum()
        roll_total = df["volume"].rolling(self._window).sum()

        buy_ratio = (roll_up / roll_total.replace(0, float("nan"))).iloc[-1]
        sell_ratio = (roll_down / roll_total.replace(0, float("nan"))).iloc[-1]

        if pd.isna(buy_ratio) or pd.isna(sell_ratio):
            return None

        ema_series = ta.ema(df["close"], length=self._ema_period)
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=self._atr_period)

        if ema_series is None or atr_series is None:
            return None

        curr_ema = float(ema_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(df["close"].iloc[-1])

        if pd.isna(curr_ema) or pd.isna(curr_atr):
            return None

        direction: Optional[str] = None
        ratio: float = 0.0

        if float(buy_ratio) >= self._imbalance_threshold and curr_price > curr_ema:
            direction = "long"
            ratio = float(buy_ratio)
        elif float(sell_ratio) >= self._imbalance_threshold and curr_price < curr_ema:
            direction = "short"
            ratio = float(sell_ratio)

        if direction is None:
            return None

        confidence = round(min(0.9, 0.4 + (ratio - self._imbalance_threshold) * 2.0), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "buy_ratio": float(buy_ratio),
            "sell_ratio": float(sell_ratio),
            "ema": curr_ema,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No order-flow imbalance detected")

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
                    f"Order flow {direction}: buy_ratio={sig['buy_ratio']:.2%}, "
                    f"sell_ratio={sig['sell_ratio']:.2%}, ATR={atr:.6f}"
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
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        if sig is None:
            return False
        side = str(getattr(position, "side", "long")).lower()
        # Close when flow reverses
        if side == "long" and sig["direction"] == "short":
            return True
        if side == "short" and sig["direction"] == "long":
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 3,
        }
