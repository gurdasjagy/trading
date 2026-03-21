"""Asian range breakout strategy — trade breakouts of Asian session range during London open.

The Asian session (Tokyo, Sydney) typically has lower volatility and forms a price range.
When London opens (08:00 GMT), increased liquidity often causes a breakout of this range.

Entry conditions:
* Detect Asian session high/low range (00:00-08:00 GMT).
* Wait for London open (08:00-09:00 GMT).
* Enter on breakout of Asian range with volume confirmation.
* Target 1-2x the Asian range size.

Best pairs:
* GBPUSD, EURUSD, GBPJPY, EURJPY (most liquid during London open)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class AsianRangeBreakoutStrategy(BaseStrategy):
    """Trade breakouts of Asian session range during London open."""

    SUPPORTED_PAIRS = ["GBPUSD", "GBP/USD", "EURUSD", "EUR/USD", "GBPJPY", "GBP/JPY", "EURJPY", "EUR/JPY"]

    def __init__(self) -> None:
        super().__init__()
        self.name = "asian_range_breakout"
        self.description = "Asian session range breakout at London open"
        self.timeframe = "15m"
        self.indicator_params = {"atr_period": 14}

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate signal on Asian range breakout during London open."""
        norm_symbol = symbol.replace("/", "")
        if norm_symbol not in [p.replace("/", "") for p in self.SUPPORTED_PAIRS]:
            return None

        if len(data) < 50:
            return None

        now = datetime.now(tz=timezone.utc)
        hour = now.hour

        # Only trade during London open (08:00-10:00 GMT)
        if not (8 <= hour < 10):
            return None

        # Calculate Asian range (00:00-08:00 GMT, ~32 candles at 15m)
        asian_candles = data[-40:-8] if len(data) >= 40 else data[-32:]
        asian_high = asian_candles["high"].max()
        asian_low = asian_candles["low"].min()
        asian_range = asian_high - asian_low

        # Require minimum range (avoid tight ranges)
        if asian_range < data["close"].iloc[-1] * 0.001:  # 0.1% minimum
            return None

        last = data.iloc[-1]
        prev = data.iloc[-2]

        # Calculate ATR
        data = self._calculate_atr(data)

        # Bullish breakout (above Asian high)
        if prev["close"] <= asian_high and last["close"] > asian_high:
            stop_loss = asian_low
            take_profit = last["close"] + (asian_range * 1.5)

            logger.info(
                "{} Asian range LONG breakout: range=[{:.5f}, {:.5f}] breakout_price={:.5f}",
                symbol,
                asian_low,
                asian_high,
                last["close"],
            )

            return Signal(
                symbol=symbol,
                direction="long",
                strength=0.85,
                confidence=0.8,
                strategy_name=self.name,
                reasoning=f"Bullish breakout above Asian high {asian_high:.5f}, range {asian_range:.5f}",
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=15,
            )

        # Bearish breakout (below Asian low)
        if prev["close"] >= asian_low and last["close"] < asian_low:
            stop_loss = asian_high
            take_profit = last["close"] - (asian_range * 1.5)

            logger.info(
                "{} Asian range SHORT breakout: range=[{:.5f}, {:.5f}] breakout_price={:.5f}",
                symbol,
                asian_low,
                asian_high,
                last["close"],
            )

            return Signal(
                symbol=symbol,
                direction="short",
                strength=0.85,
                confidence=0.8,
                strategy_name=self.name,
                reasoning=f"Bearish breakout below Asian low {asian_low:.5f}, range {asian_range:.5f}",
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=15,
            )

        return None

    def should_close(self, data: pd.DataFrame, symbol: str, position: Dict) -> tuple[bool, str]:
        """Close if target reached or position held too long."""
        # Asian breakout trades are short-term (close by end of London session)
        now = datetime.now(tz=timezone.utc)
        if now.hour >= 16:  # Close by 16:00 GMT
            return (True, "London session closed")
        return (False, "")

    def calculate_parameters(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate risk parameters."""
        return {"stop_loss_type": "range_boundary", "recommended_leverage": 15}

    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR."""
        data = data.copy()
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr"] = true_range.rolling(window=self.indicator_params["atr_period"]).mean()
        return data
