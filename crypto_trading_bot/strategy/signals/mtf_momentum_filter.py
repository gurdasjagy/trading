"""Multi-timeframe momentum filter for trade validation.

Analyzes momentum alignment across 1h, 4h, and 1d timeframes using EMA crossovers
to validate trade signals. Only allows trades when momentum is aligned across
multiple timeframes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from loguru import logger


class MTFMomentumFilter:
    """Filter trades based on multi-timeframe momentum alignment.

    Uses EMA(20) vs EMA(50) crossovers on 1h, 4h, and 1d timeframes to
    determine momentum direction. Requires at least 2/3 timeframes to agree
    before allowing a trade.
    """

    TIMEFRAMES = ("1h", "4h", "1d")

    def __init__(self) -> None:
        pass

    def check_momentum_alignment(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, float, str]:
        """Check if momentum is aligned across multiple timeframes.

        Args:
            market_data: Dict mapping timeframe labels to OHLCV DataFrames.
                Expected keys: "1h", "4h", "1d"

        Returns:
            Tuple of (should_proceed, size_multiplier, reasoning):
                - should_proceed: True if trade should be allowed
                - size_multiplier: 1.0 for 3/3 agreement, 0.8 for 2/3, 0.0 for 1/3
                - reasoning: Human-readable explanation
        """
        if not market_data:
            return False, 0.0, "No market data provided"

        momentum_signals = {}
        available_tfs = []

        for tf in self.TIMEFRAMES:
            df = market_data.get(tf)
            if df is None or df.empty or len(df) < 50:
                logger.debug(f"[MTFMomentum] Insufficient data for {tf} timeframe")
                continue

            direction = self._analyze_timeframe_momentum(df, tf)
            momentum_signals[tf] = direction
            available_tfs.append(tf)

        if len(available_tfs) < 2:
            return False, 0.0, f"Insufficient timeframes (need 2+, have {len(available_tfs)})"

        # Count bullish and bearish signals
        bullish_count = sum(1 for d in momentum_signals.values() if d == "bullish")
        bearish_count = sum(1 for d in momentum_signals.values() if d == "bearish")
        neutral_count = sum(1 for d in momentum_signals.values() if d == "neutral")

        total_signals = len(momentum_signals)

        # Determine overall direction and agreement level
        if bullish_count >= 2:
            direction = "bullish"
            agreement = bullish_count
        elif bearish_count >= 2:
            direction = "bearish"
            agreement = bearish_count
        else:
            # No clear consensus
            return False, 0.0, f"No momentum consensus: {bullish_count}B/{bearish_count}B/{neutral_count}N"

        # Calculate size multiplier based on agreement level
        if agreement == 3:
            size_multiplier = 1.0
            should_proceed = True
            reasoning = f"Strong {direction} momentum: 3/3 timeframes agree ({', '.join(available_tfs)})"
        elif agreement == 2:
            size_multiplier = 0.8
            should_proceed = True
            reasoning = f"Moderate {direction} momentum: 2/3 timeframes agree ({', '.join(available_tfs)})"
        else:
            size_multiplier = 0.0
            should_proceed = False
            reasoning = f"Weak {direction} momentum: only 1/3 timeframes agree"

        logger.debug(
            "[MTFMomentum] {} - proceed={} mult={:.1f}",
            reasoning,
            should_proceed,
            size_multiplier,
        )

        return should_proceed, size_multiplier, reasoning

    def _analyze_timeframe_momentum(self, df: pd.DataFrame, timeframe: str) -> str:
        """Analyze momentum direction for a single timeframe.

        Uses EMA(20) vs EMA(50) crossover:
        - EMA(20) > EMA(50) → bullish
        - EMA(20) < EMA(50) → bearish
        - Otherwise → neutral

        Args:
            df: OHLCV DataFrame with at least 50 candles
            timeframe: Timeframe label (for logging)

        Returns:
            "bullish", "bearish", or "neutral"
        """
        try:
            closes = df["close"].astype(float)

            if len(closes) < 50:
                return "neutral"

            # Calculate EMAs
            ema_fast = closes.ewm(span=20, adjust=False).mean()
            ema_slow = closes.ewm(span=50, adjust=False).mean()

            last_fast = float(ema_fast.iloc[-1])
            last_slow = float(ema_slow.iloc[-1])

            # Determine direction
            if last_fast > last_slow * 1.001:  # 0.1% buffer to avoid noise
                direction = "bullish"
            elif last_fast < last_slow * 0.999:
                direction = "bearish"
            else:
                direction = "neutral"

            logger.debug(
                "[MTFMomentum] {}: EMA20={:.4f} EMA50={:.4f} → {}",
                timeframe,
                last_fast,
                last_slow,
                direction,
            )

            return direction

        except Exception as exc:
            logger.warning(
                "[MTFMomentum] Failed to analyze {} momentum: {}", timeframe, exc
            )
            return "neutral"
