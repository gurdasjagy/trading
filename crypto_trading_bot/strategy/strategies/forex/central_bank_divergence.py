"""Central bank divergence strategy — trade based on monetary policy differences.

Trade currency pairs based on diverging central bank policies:
* Hawkish (raising rates) vs Dovish (cutting rates)
* Quantitative tightening vs Quantitative easing

Key events to monitor:
* FOMC (Federal Reserve) - USD
* ECB (European Central Bank) - EUR
* BOE (Bank of England) - GBP
* BOJ (Bank of Japan) - JPY
* RBA (Reserve Bank of Australia) - AUD

Entry conditions:
* Clear policy divergence (one hawkish, one dovish).
* Go long the currency with tightening policy.
* Go short the currency with easing policy.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class CentralBankDivergenceStrategy(BaseStrategy):
    """Trade based on central bank monetary policy divergence."""

    # Central bank policy stance (update after policy meetings)
    CENTRAL_BANK_STANCE = {
        "USD": "hawkish",   # Fed raising rates
        "EUR": "neutral",   # ECB on hold
        "GBP": "hawkish",   # BOE raising rates
        "JPY": "dovish",    # BOJ ultra-dovish (negative rates)
        "AUD": "neutral",   # RBA neutral
        "NZD": "neutral",   # RBNZ neutral
        "CHF": "dovish",    # SNB dovish
        "CAD": "neutral",   # BOC neutral
    }

    SUPPORTED_PAIRS = [
        "USDJPY", "USD/JPY",    # Strong divergence (hawkish vs dovish)
        "GBPJPY", "GBP/JPY",    # Strong divergence
        "EURUSD", "EUR/USD",    # Moderate divergence
        "GBPUSD", "GBP/USD",    # Moderate divergence
        "USDCHF", "USD/CHF",    # Strong divergence
    ]

    def __init__(self) -> None:
        super().__init__()
        self.name = "central_bank_divergence"
        self.description = "Monetary policy divergence trading"
        self.timeframe = "4h"
        self.indicator_params = {"ema_period": 50, "rsi_period": 14}

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate signal based on central bank policy divergence."""
        norm_symbol = symbol.replace("/", "")
        if norm_symbol not in [p.replace("/", "") for p in self.SUPPORTED_PAIRS]:
            return None

        if len(data) < 60:
            return None

        # Extract base and quote currencies
        if len(norm_symbol) == 6:
            base_ccy = norm_symbol[:3]
            quote_ccy = norm_symbol[3:]
        else:
            return None

        base_stance = self.CENTRAL_BANK_STANCE.get(base_ccy, "neutral")
        quote_stance = self.CENTRAL_BANK_STANCE.get(quote_ccy, "neutral")

        # Calculate divergence score
        stance_map = {"hawkish": 1, "neutral": 0, "dovish": -1}
        divergence = stance_map[base_stance] - stance_map[quote_stance]

        # Only trade pairs with strong divergence
        if abs(divergence) < 2:
            logger.debug(
                "{} insufficient policy divergence: {} ({}) vs {} ({})",
                symbol,
                base_ccy,
                base_stance,
                quote_ccy,
                quote_stance,
            )
            return None

        # Calculate indicators
        data = self._calculate_indicators(data)
        last = data.iloc[-1]

        ema = last["ema"]
        rsi = last["rsi"]
        atr = last["atr"]

        # Long signal (base currency hawkish, quote currency dovish)
        if divergence >= 2:
            # Confirm with trend and momentum
            if last["close"] > ema and rsi < 70:
                stop_loss = last["close"] - (atr * 2.5)
                take_profit = last["close"] + (atr * 5.0)

                logger.info(
                    "{} Central bank divergence LONG: {} hawkish, {} dovish (divergence={:.0f})",
                    symbol,
                    base_ccy,
                    quote_ccy,
                    divergence,
                )

                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=0.8,
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Policy divergence: {base_ccy} ({base_stance}) vs {quote_ccy} ({quote_stance}), "
                        f"trend confirmed (price > EMA), RSI {rsi:.1f}"
                    ),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=12,
                )

        # Short signal (base currency dovish, quote currency hawkish)
        elif divergence <= -2:
            if last["close"] < ema and rsi > 30:
                stop_loss = last["close"] + (atr * 2.5)
                take_profit = last["close"] - (atr * 5.0)

                logger.info(
                    "{} Central bank divergence SHORT: {} dovish, {} hawkish (divergence={:.0f})",
                    symbol,
                    base_ccy,
                    quote_ccy,
                    divergence,
                )

                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=0.8,
                    confidence=0.75,
                    strategy_name=self.name,
                    reasoning=(
                        f"Policy divergence: {base_ccy} ({base_stance}) vs {quote_ccy} ({quote_stance}), "
                        f"trend confirmed (price < EMA), RSI {rsi:.1f}"
                    ),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=12,
                )

        return None

    def should_close(self, data: pd.DataFrame, symbol: str, position: Dict) -> tuple[bool, str]:
        """Close if policy stance changes or trend reverses."""
        if len(data) < 30:
            return (False, "")

        data = self._calculate_indicators(data)
        last = data.iloc[-1]

        # Exit if trend reverses (price crosses EMA)
        side = position.get("side", "long")
        if side == "long" and last["close"] < last["ema"]:
            return (True, "Trend reversal (price < EMA)")
        elif side == "short" and last["close"] > last["ema"]:
            return (True, "Trend reversal (price > EMA)")

        return (False, "")

    def calculate_parameters(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate risk parameters."""
        return {"stop_loss_type": "atr_based", "recommended_leverage": 12}

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA, RSI, ATR."""
        data = data.copy()

        # EMA
        data["ema"] = data["close"].ewm(span=self.indicator_params["ema_period"], adjust=False).mean()

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.indicator_params["rsi_period"]).mean()
        avg_loss = loss.rolling(window=self.indicator_params["rsi_period"]).mean()
        rs = avg_gain / avg_loss
        data["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr"] = true_range.rolling(window=14).mean()

        return data
