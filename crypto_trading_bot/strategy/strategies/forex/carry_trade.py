"""Carry trade strategy — profit from interest rate differentials between currencies.

The carry trade strategy involves:
1. Going long on high-yield currencies (e.g., AUD, NZD).
2. Going short on low-yield currencies (e.g., JPY, CHF).
3. Holding positions to earn positive swap/rollover interest.

This strategy works best in:
* Low-volatility markets (Asian session).
* Stable trending environments.
* When central bank policy divergence is clear.

Avoid during:
* High-impact news events (NFP, FOMC, BOJ).
* Sudden risk-off moves (market crashes).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class CarryTradeStrategy(BaseStrategy):
    """Carry trade strategy based on interest rate differentials.

    This strategy targets currency pairs with high interest rate differentials,
    going long the high-yielder and short the low-yielder.

    SUPPORTED_PAIRS:
    * AUDJPY, NZDJPY — high yield (AUD/NZD) vs low yield (JPY)
    * AUDCHF, NZDCHF — high yield (AUD/NZD) vs low yield (CHF)
    * GBPJPY — moderate yield differential

    Entry conditions:
    * Positive swap rate for the direction (long high-yield).
    * Trend alignment (higher highs on 4H chart).
    * RSI not overbought (< 70).
    * Low volatility (ATR below 20-period average).

    Exit conditions:
    * Swap rate turns negative.
    * Trend reversal (price crosses below 20 EMA).
    * RSI oversold (< 30).
    """

    SUPPORTED_PAIRS = [
        "AUDJPY",
        "AUD/JPY",
        "NZDJPY",
        "NZD/JPY",
        "AUDCHF",
        "AUD/CHF",
        "NZDCHF",
        "NZD/CHF",
        "GBPJPY",
        "GBP/JPY",
    ]

    # Interest rate differentials (approximate, update quarterly)
    INTEREST_RATE_DIFF = {
        "AUDJPY": 4.0,   # AUD ~4.35%, JPY ~-0.1%
        "NZDJPY": 4.5,   # NZD ~5.5%, JPY ~-0.1%
        "AUDCHF": 3.8,   # AUD ~4.35%, CHF ~1.75%
        "NZDCHF": 4.3,   # NZD ~5.5%, CHF ~1.75%
        "GBPJPY": 5.0,   # GBP ~5.25%, JPY ~-0.1%
    }

    def __init__(self) -> None:
        super().__init__()
        self.name = "carry_trade"
        self.description = "Interest rate differential carry trade"
        self.timeframe = "4h"
        self.indicator_params = {
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_period": 14,
            "atr_period": 20,
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate carry trade signal based on trend and swap rates.

        Args:
            data: OHLCV DataFrame with indicators.
            symbol: Trading pair symbol.

        Returns:
            Signal object if entry conditions met, else None.
        """
        # Normalize symbol
        norm_symbol = symbol.replace("/", "")
        if norm_symbol not in [p.replace("/", "") for p in self.SUPPORTED_PAIRS]:
            return None

        if len(data) < 60:
            return None

        # Calculate indicators
        data = self._calculate_indicators(data)

        last = data.iloc[-1]
        prev = data.iloc[-2]

        # Check interest rate differential (must be positive for carry)
        rate_diff = self.INTEREST_RATE_DIFF.get(norm_symbol, 0.0)
        if rate_diff < 2.0:
            # Not worth it if differential < 2%
            return None

        # Entry conditions for LONG (carry trade)
        ema_fast = last["ema_fast"]
        ema_slow = last["ema_slow"]
        rsi = last["rsi"]
        atr = last["atr"]
        atr_avg = data["atr"].rolling(20).mean().iloc[-1]

        # 1. Uptrend: Fast EMA > Slow EMA
        if ema_fast <= ema_slow:
            return None

        # 2. Price above both EMAs
        if last["close"] <= ema_fast:
            return None

        # 3. RSI not overbought
        if rsi >= 70:
            return None

        # 4. Low volatility (ATR below average)
        if atr > atr_avg * 1.2:
            logger.debug("{} volatility too high for carry trade (ATR={:.2f})", symbol, atr)
            return None

        # 5. Recent bullish momentum (higher high)
        if last["high"] <= prev["high"]:
            return None

        # Calculate SL/TP
        stop_loss = last["close"] - (atr * 2.0)
        take_profit = last["close"] + (atr * 4.0)

        logger.info(
            "{} Carry trade LONG signal: rate_diff={:.1f}% RSI={:.1f} EMA_fast={:.5f} EMA_slow={:.5f}",
            symbol,
            rate_diff,
            rsi,
            ema_fast,
            ema_slow,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            strength=0.8,
            confidence=0.75,
            strategy_name=self.name,
            reasoning=(
                f"Carry trade: {rate_diff:.1f}% rate differential, "
                f"uptrend (EMA {ema_fast:.5f} > {ema_slow:.5f}), "
                f"RSI {rsi:.1f}, low vol (ATR {atr:.5f})"
            ),
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=10,  # Lower leverage for carry trades (long-term hold)
        )

    def should_close(self, data: pd.DataFrame, symbol: str, position: Dict) -> tuple[bool, str]:
        """Check if carry trade position should be closed.

        Exit conditions:
        * Price crosses below fast EMA (trend reversal).
        * RSI oversold (< 30) — potential reversal.
        * ATR spike (volatility surge).

        Args:
            data: OHLCV DataFrame.
            symbol: Trading pair.
            position: Current position dict.

        Returns:
            (should_close, reason): Tuple of bool and reason string.
        """
        if len(data) < 30:
            return (False, "")

        data = self._calculate_indicators(data)
        last = data.iloc[-1]

        # Exit if price crosses below fast EMA
        if last["close"] < last["ema_fast"]:
            return (True, "Price crossed below EMA (trend reversal)")

        # Exit if RSI oversold
        if last["rsi"] < 30:
            return (True, "RSI oversold (potential reversal)")

        # Exit if ATR spikes (volatility surge)
        atr_avg = data["atr"].rolling(20).mean().iloc[-1]
        if last["atr"] > atr_avg * 2.0:
            return (True, f"ATR spike (volatility surge: {last['atr']:.2f})")

        return (False, "")

    def calculate_parameters(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate risk parameters for carry trade.

        Returns:
            Dict with SL/TP and position size recommendations.
        """
        if len(data) < 30:
            return {}

        data = self._calculate_indicators(data)
        last = data.iloc[-1]

        atr = last["atr"]
        stop_loss_pips = atr * 2.0
        take_profit_pips = atr * 4.0

        return {
            "stop_loss_pips": stop_loss_pips,
            "take_profit_pips": take_profit_pips,
            "recommended_leverage": 10,
            "holding_period": "long_term",  # Carry trades are multi-day
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs, RSI, and ATR."""
        data = data.copy()

        # EMAs
        data["ema_fast"] = data["close"].ewm(span=self.indicator_params["ema_fast"], adjust=False).mean()
        data["ema_slow"] = data["close"].ewm(span=self.indicator_params["ema_slow"], adjust=False).mean()

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
        data["atr"] = true_range.rolling(window=self.indicator_params["atr_period"]).mean()

        return data
