"""MACD Crossover strategy — signal-line crossovers with zero-line filter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class MACDCrossoverStrategy(BaseStrategy):
    """MACD signal-line crossover strategy with zero-line bias filter.

    Entry conditions
    ----------------
    * **Long**: MACD line crosses above signal line AND MACD is above zero.
    * **Short**: MACD line crosses below signal line AND MACD is below zero.

    Histogram momentum is used to weight confidence.
    """

    _STRATEGY_NAME = "macd_crossover"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._fast = fast
        self._slow = slow
        self._signal = signal
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._slow + self._signal + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        macd_df = ta.macd(closes, fast=self._fast, slow=self._slow, signal=self._signal)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if macd_df is None or atr_series is None:
            return None

        macd_col = f"MACD_{self._fast}_{self._slow}_{self._signal}"
        sig_col = f"MACDs_{self._fast}_{self._slow}_{self._signal}"
        hist_col = f"MACDh_{self._fast}_{self._slow}_{self._signal}"

        cols = macd_df.columns.tolist()
        if macd_col not in cols:
            macd_col, sig_col, hist_col = cols[0], cols[1], cols[2]

        curr_macd = float(macd_df[macd_col].iloc[-1])
        prev_macd = float(macd_df[macd_col].iloc[-2])
        curr_sig = float(macd_df[sig_col].iloc[-1])
        prev_sig = float(macd_df[sig_col].iloc[-2])
        curr_hist = float(macd_df[hist_col].iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])

        for val in (curr_macd, prev_macd, curr_sig, prev_sig, curr_hist, curr_atr):
            if pd.isna(val):
                return None

        cross_up = prev_macd <= prev_sig and curr_macd > curr_sig
        cross_down = prev_macd >= prev_sig and curr_macd < curr_sig

        direction: Optional[str] = None
        if cross_up and curr_macd > 0:
            direction = "long"
        elif cross_down and curr_macd < 0:
            direction = "short"

        if direction is None:
            return None

        hist_abs = abs(curr_hist)
        # Normalise histogram relative to ATR
        confidence = round(min(0.9, 0.5 + min(1.0, hist_abs / (curr_atr + 1e-9)) * 0.4), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "macd": curr_macd,
            "macd_signal": curr_sig,
            "histogram": curr_hist,
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No MACD crossover signal")

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
                    f"MACD crossover {direction}: MACD={sig['macd']:.6f}, "
                    f"signal={sig['macd_signal']:.6f}, hist={sig['histogram']:.6f}"
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
        closes = ohlcv["close"].tolist()
        macd_data = self._calculate_macd(closes, self._fast, self._slow, self._signal)
        side = str(getattr(position, "side", "long")).lower()
        # Close on MACD/signal reversal
        if side == "long" and macd_data["macd"] < macd_data["signal"]:
            return True
        if side == "short" and macd_data["macd"] > macd_data["signal"]:
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
            "take_profit_pct": min(0.10, sl_pct * 2.5),
            "leverage": 3,
        }
