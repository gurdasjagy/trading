"""Volatility analyzer: realized, implied, regime, and prediction."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Annualisation factor for crypto (365 days × 24 hours)
_ANNUALISE_HOURLY: float = (365 * 24) ** 0.5

# Volatility regime thresholds (annualised)
_REGIME_LOW: float = 0.30  # < 30% annualised → low
_REGIME_HIGH: float = 0.80  # > 80% annualised → high
_REGIME_EXTREME: float = 1.50  # > 150% annualised → extreme


class VolatilityAnalyzer:
    """Analyzes and predicts market volatility across multiple regimes.

    All volatility values are expressed as **annualised** standard deviations
    unless noted otherwise.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_realized_volatility(
        self,
        prices: List[float],
        window: int = 24,
    ) -> float:
        """Calculate realized volatility from a price series.

        Args:
            prices: Ordered list of close prices (oldest first).
            window: Number of periods to use (rolling window).

        Returns:
            Annualised realized volatility (float ≥ 0).
        """
        if len(prices) < 2:
            logger.debug("Insufficient price data for realized volatility")
            return 0.0
        try:
            s = pd.Series(prices, dtype=float)
            log_returns = np.log(s / s.shift(1)).dropna()
            if len(log_returns) < 2:
                return 0.0
            tail = log_returns.tail(window)
            rv = float(tail.std()) * _ANNUALISE_HOURLY
            return max(0.0, rv)
        except Exception as exc:
            logger.warning(f"VolatilityAnalyzer.calculate_realized_volatility error: {exc}")
            return 0.0

    def calculate_implied_volatility(self, symbol: str) -> float:
        """Return an implied volatility estimate for *symbol*.

        In the absence of live options data this method returns a heuristic
        estimate based on historical-volatility norms for major crypto assets.

        Args:
            symbol: Asset ticker (e.g. ``"BTC/USDT"``).

        Returns:
            Annualised implied volatility estimate (float ≥ 0).
        """
        # Heuristic IV estimates sourced from crypto options markets
        _IV_DEFAULTS: Dict[str, float] = {
            "BTC": 0.65,
            "ETH": 0.75,
            "SOL": 0.95,
            "BNB": 0.80,
        }
        base = symbol.split("/")[0].upper()
        iv = _IV_DEFAULTS.get(base, 0.90)  # default: 90% for alts
        logger.debug(f"Implied volatility for {symbol}: {iv:.2f} (heuristic)")
        return iv

    def detect_volatility_regime(
        self,
        symbol: str,
        prices: Optional[List[float]] = None,
        realized_vol: Optional[float] = None,
    ) -> str:
        """Classify the current volatility regime.

        Args:
            symbol:       Asset ticker (used for logging).
            prices:       Optional price list; realized vol computed from this
                          when *realized_vol* is not provided.
            realized_vol: Pre-computed annualised realized volatility.

        Returns:
            One of ``"low"``, ``"medium"``, ``"high"``, or ``"extreme"``.
        """
        try:
            if realized_vol is None:
                if prices:
                    realized_vol = self.calculate_realized_volatility(prices)
                else:
                    return "medium"

            if realized_vol >= _REGIME_EXTREME:
                regime = "extreme"
            elif realized_vol >= _REGIME_HIGH:
                regime = "high"
            elif realized_vol <= _REGIME_LOW:
                regime = "low"
            else:
                regime = "medium"

            logger.debug(f"Volatility regime for {symbol}: {regime} (rv={realized_vol:.2f})")
            return regime
        except Exception as exc:
            logger.warning(f"VolatilityAnalyzer.detect_volatility_regime error: {exc}")
            return "medium"

    def predict_volatility(
        self,
        symbol: str,
        horizon: str = "24h",
        prices: Optional[List[float]] = None,
    ) -> float:
        """Predict forward volatility using a simple EWMA (GARCH-like) model.

        Args:
            symbol:  Asset ticker (used for logging).
            horizon: Prediction horizon string — currently informational only.
            prices:  Historical price list to calibrate the model.

        Returns:
            Annualised predicted volatility estimate (float ≥ 0).
        """
        try:
            if not prices or len(prices) < 5:
                return self.calculate_implied_volatility(symbol)

            s = pd.Series(prices, dtype=float)
            log_returns = np.log(s / s.shift(1)).dropna()
            if len(log_returns) < 2:
                return self.calculate_implied_volatility(symbol)

            # EWMA variance with λ = 0.94 (RiskMetrics standard)
            lambda_: float = 0.94
            variance = float(log_returns.iloc[0] ** 2)
            for r in log_returns.iloc[1:]:
                variance = lambda_ * variance + (1 - lambda_) * float(r) ** 2

            predicted = (variance**0.5) * _ANNUALISE_HOURLY
            logger.debug(f"Predicted volatility for {symbol} ({horizon}): {predicted:.3f}")
            return max(0.0, predicted)
        except Exception as exc:
            logger.warning(f"VolatilityAnalyzer.predict_volatility error: {exc}")
            return self.calculate_implied_volatility(symbol)
