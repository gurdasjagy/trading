"""Session-based mean reversion strategy — trade mean reversion within session boundaries.

Mean reversion works best during range-bound sessions when price oscillates around a mean.
This strategy uses Bollinger Bands and RSI to identify oversold/overbought conditions within
each trading session (London, New York, Asian).

Entry conditions:
* Price touches lower Bollinger Band + RSI < 30 → LONG
* Price touches upper Bollinger Band + RSI > 70 → SHORT
* Session boundaries act as support/resistance

Exit conditions:
* Price returns to middle Bollinger Band (mean)
* Session closes
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class ForexMeanReversionSessionStrategy(BaseStrategy):
    """Mean reversion trading within session boundaries using Bollinger Bands + RSI."""

    SUPPORTED_PAIRS = [
        "EURUSD", "EUR/USD",
        "GBPUSD", "GBP/USD",
        "USDJPY", "USD/JPY",
        "AUDUSD", "AUD/USD",
        "USDCAD", "USD/CAD",
        "EURGBP", "EUR/GBP",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.name = "forex_mean_reversion_session"
        self.description = "Session-based Bollinger Band mean reversion"
        self.timeframe = "15m"
        self.indicator_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate mean reversion signal at session extremes."""
        norm_symbol = symbol.replace("/", "")
        if norm_symbol not in [p.replace("/", "") for p in self.SUPPORTED_PAIRS]:
            return None

        if len(data) < 50:
            return None

        # Only trade during active sessions (not during quiet periods)
        now = datetime.now(tz=timezone.utc)
        hour = now.hour
        session = self._get_current_session(hour)

        if session == "Closed":
            return None

        # Calculate indicators
        data = self._calculate_indicators(data)
        last = data.iloc[-1]

        bb_upper = last["bb_upper"]
        bb_middle = last["bb_middle"]
        bb_lower = last["bb_lower"]
        rsi = last["rsi"]

        # Long signal (oversold at lower BB)
        if last["close"] <= bb_lower and rsi < 30:
            stop_loss = last["close"] - (bb_middle - bb_lower)
            take_profit = bb_middle  # Target mean reversion

            logger.info(
                "{} Mean reversion LONG: price={:.5f} BB_lower={:.5f} RSI={:.1f} session={}",
                symbol,
                last["close"],
                bb_lower,
                rsi,
                session,
            )

            return Signal(
                symbol=symbol,
                direction="long",
                strength=0.8,
                confidence=0.75,
                strategy_name=self.name,
                reasoning=f"Oversold at lower BB ({bb_lower:.5f}), RSI {rsi:.1f}, {session} session",
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=15,
            )

        # Short signal (overbought at upper BB)
        if last["close"] >= bb_upper and rsi > 70:
            stop_loss = last["close"] + (bb_upper - bb_middle)
            take_profit = bb_middle

            logger.info(
                "{} Mean reversion SHORT: price={:.5f} BB_upper={:.5f} RSI={:.1f} session={}",
                symbol,
                last["close"],
                bb_upper,
                rsi,
                session,
            )

            return Signal(
                symbol=symbol,
                direction="short",
                strength=0.8,
                confidence=0.75,
                strategy_name=self.name,
                reasoning=f"Overbought at upper BB ({bb_upper:.5f}), RSI {rsi:.1f}, {session} session",
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=15,
            )

        return None

    def should_close(self, data: pd.DataFrame, symbol: str, position: Dict) -> tuple[bool, str]:
        """Close when price returns to mean or session closes."""
        if len(data) < 30:
            return (False, "")

        data = self._calculate_indicators(data)
        last = data.iloc[-1]

        bb_middle = last["bb_middle"]

        # Close when price reverts to mean
        side = position.get("side", "long")
        if side == "long" and last["close"] >= bb_middle:
            return (True, "Price reverted to mean (BB middle)")
        elif side == "short" and last["close"] <= bb_middle:
            return (True, "Price reverted to mean (BB middle)")

        # Close at session end
        now = datetime.now(tz=timezone.utc)
        hour = now.hour
        # Close London positions at 16:00, NY positions at 21:00
        if hour in [16, 21]:
            return (True, "Session closing")

        return (False, "")

    def calculate_parameters(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate risk parameters."""
        return {"stop_loss_type": "bb_based", "recommended_leverage": 15}

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and RSI."""
        data = data.copy()

        # Bollinger Bands
        bb_period = self.indicator_params["bb_period"]
        bb_std = self.indicator_params["bb_std"]
        data["bb_middle"] = data["close"].rolling(window=bb_period).mean()
        bb_std_dev = data["close"].rolling(window=bb_period).std()
        data["bb_upper"] = data["bb_middle"] + (bb_std_dev * bb_std)
        data["bb_lower"] = data["bb_middle"] - (bb_std_dev * bb_std)

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.indicator_params["rsi_period"]).mean()
        avg_loss = loss.rolling(window=self.indicator_params["rsi_period"]).mean()
        rs = avg_gain / avg_loss
        data["rsi"] = 100 - (100 / (1 + rs))

        return data

    def _get_current_session(self, hour: int) -> str:
        """Return the name of the current trading session."""
        if 8 <= hour < 16:
            return "London"
        elif 13 <= hour < 21:
            return "New York"
        elif 0 <= hour < 9:
            return "Tokyo"
        elif hour >= 22 or hour < 7:
            return "Sydney"
        return "Closed"
