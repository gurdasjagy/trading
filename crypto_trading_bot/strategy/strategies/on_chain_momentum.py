"""On-Chain Momentum strategy — unusual volume surge as on-chain activity proxy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class OnChainMomentumStrategy(BaseStrategy):
    """On-chain momentum strategy using volume as an activity proxy.

    A volume surge (volume significantly above its rolling average) is treated
    as a proxy for elevated on-chain activity.  The direction of the
    accompanying price move determines long vs short.

    Entry conditions
    ----------------
    * **Long**: volume > *surge_factor* × rolling average volume AND price
      closes up (close > open) AND price is above EMA.
    * **Short**: volume surge AND price closes down AND price is below EMA.
    """

    _STRATEGY_NAME = "on_chain_momentum"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        vol_period: int = 20,
        surge_factor: float = 2.0,
        ema_period: int = 50,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._vol_period = vol_period
        self._surge_factor = surge_factor
        self._ema_period = ema_period
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = max(self._vol_period, self._ema_period) + self._atr_period + 5
        if len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        opens = ohlcv["open"]
        volume = ohlcv["volume"]

        avg_vol = volume.rolling(self._vol_period).mean()
        ema_series = ta.ema(closes, length=self._ema_period)
        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], closes, length=self._atr_period)

        if ema_series is None or atr_series is None:
            return None

        curr_vol = float(volume.iloc[-1])
        curr_avg_vol = float(avg_vol.iloc[-1])
        curr_ema = float(ema_series.iloc[-1])
        curr_atr = float(atr_series.iloc[-1])
        curr_price = float(closes.iloc[-1])
        curr_open = float(opens.iloc[-1])

        if pd.isna(curr_avg_vol) or curr_avg_vol == 0 or pd.isna(curr_ema) or pd.isna(curr_atr):
            return None

        vol_ratio = curr_vol / curr_avg_vol
        if vol_ratio < self._surge_factor:
            return None

        bullish_candle = curr_price > curr_open
        bearish_candle = curr_price < curr_open

        direction: Optional[str] = None
        if bullish_candle and curr_price > curr_ema:
            direction = "long"
        elif bearish_candle and curr_price < curr_ema:
            direction = "short"

        if direction is None:
            return None

        # Confidence scales with volume ratio above surge threshold
        excess_ratio = (vol_ratio - self._surge_factor) / self._surge_factor
        confidence = round(min(0.9, 0.55 + excess_ratio * 0.2), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": curr_atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "volume_ratio": vol_ratio,
            "avg_volume": curr_avg_vol,
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
                return self._neutral_signal(symbol, "No volume surge detected")

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
                    f"Volume surge {direction}: ratio={sig['volume_ratio']:.1f}x avg, "
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
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=80)
        if ohlcv.empty:
            return False
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and rsi > 72:
            return True
        if side == "short" and rsi < 28:
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
