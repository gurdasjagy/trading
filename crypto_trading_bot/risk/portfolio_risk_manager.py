"""Portfolio-level risk management with correlation-aware position sizing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Default correlation matrix for major crypto pairs (empirically observed)
DEFAULT_CRYPTO_CORRELATIONS: Dict[str, Dict[str, float]] = {
    "BTC/USDT": {"ETH/USDT": 0.85, "SOL/USDT": 0.80, "BNB/USDT": 0.75, "XRP/USDT": 0.70},
    "ETH/USDT": {"BTC/USDT": 0.85, "SOL/USDT": 0.82, "BNB/USDT": 0.72, "XRP/USDT": 0.68},
    "SOL/USDT": {"BTC/USDT": 0.80, "ETH/USDT": 0.82, "BNB/USDT": 0.70, "XRP/USDT": 0.65},
    "BNB/USDT": {"BTC/USDT": 0.75, "ETH/USDT": 0.72, "SOL/USDT": 0.70, "XRP/USDT": 0.60},
    "XRP/USDT": {"BTC/USDT": 0.70, "ETH/USDT": 0.68, "SOL/USDT": 0.65, "BNB/USDT": 0.60},
}


class PortfolioRiskManager:
    """Manages portfolio-level risk including correlation-adjusted VaR."""

    def __init__(
        self,
        max_portfolio_risk_pct: float = 5.0,
        max_correlated_exposure_pct: float = 3.0,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._max_portfolio_risk = max_portfolio_risk_pct / 100.0
        self._max_correlated_exposure = max_correlated_exposure_pct / 100.0
        self._correlations = correlation_matrix or DEFAULT_CRYPTO_CORRELATIONS

    def calculate_portfolio_var(
        self,
        positions: List[dict],
        equity: float,
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate portfolio Value at Risk accounting for correlations.

        Args:
            positions: List of position dicts with 'symbol', 'amount', 'entry_price', 'side'.
            equity: Total portfolio equity.
            confidence_level: VaR confidence level (default 95%).

        Returns:
            Portfolio VaR as a fraction of equity.
        """
        if not positions or equity <= 0:
            return 0.0

        n = len(positions)
        # Build weight vector (position value / equity)
        weights = []
        symbols = []
        for pos in positions:
            value = abs(float(pos.get("amount", 0)) * float(pos.get("entry_price", 0)))
            weight = value / equity
            # Negative weight for shorts
            if pos.get("side", "long") in ("short", "sell"):
                weight = -weight
            weights.append(weight)
            symbols.append(pos.get("symbol", ""))

        weights_arr = np.array(weights)

        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._get_correlation(symbols[i], symbols[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Assume 2% daily vol per position (simplified)
        daily_vol = 0.02
        vol_vector = np.full(n, daily_vol)

        # Portfolio variance = w' * Sigma * w
        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix
        portfolio_variance = float(weights_arr @ cov_matrix @ weights_arr)
        portfolio_vol = float(np.sqrt(max(portfolio_variance, 0.0)))

        # VaR at confidence level (using normal approximation)
        try:
            from scipy.stats import norm

            z_score = float(norm.ppf(confidence_level))
        except ImportError:
            z_score = 1.645  # 95% confidence fallback

        var = portfolio_vol * z_score
        return float(var)

    def should_reduce_new_position(
        self,
        new_symbol: str,
        new_size_usdt: float,
        existing_positions: List[dict],
        equity: float,
    ) -> Tuple[bool, float, str]:
        """Check if a new position should be reduced due to portfolio correlation risk.

        Returns:
            Tuple of (should_reduce, adjusted_size, reason)
        """
        if not existing_positions:
            return False, new_size_usdt, ""

        # Calculate max correlation with existing positions
        max_corr = 0.0
        most_correlated_symbol = ""
        same_direction_count = 0

        for pos in existing_positions:
            sym = pos.get("symbol", "")
            corr = self._get_correlation(new_symbol, sym)
            if corr > max_corr:
                max_corr = corr
                most_correlated_symbol = sym
            same_direction_count += 1

        # If highly correlated (>0.7) with existing positions, reduce size
        if max_corr > 0.7:
            reduction_factor = 1.0 - (max_corr - 0.7) / 0.3  # Linear reduction from 0.7 to 1.0
            reduction_factor = max(0.3, reduction_factor)  # Never reduce below 30%
            adjusted_size = new_size_usdt * reduction_factor
            reason = (
                f"Correlation risk: {new_symbol} has {max_corr:.0%} correlation with "
                f"{most_correlated_symbol}. Size reduced from {new_size_usdt:.2f} to {adjusted_size:.2f} USDT"
            )
            logger.warning(reason)
            return True, adjusted_size, reason

        # If too many same-direction positions, reduce
        if same_direction_count >= 3:
            reduction_factor = max(0.5, 1.0 - (same_direction_count - 2) * 0.15)
            adjusted_size = new_size_usdt * reduction_factor
            reason = f"Portfolio concentration: {same_direction_count} existing positions. Size reduced."
            logger.warning(reason)
            return True, adjusted_size, reason

        return False, new_size_usdt, ""

    def _get_correlation(self, sym1: str, sym2: str) -> float:
        if sym1 == sym2:
            return 1.0
        return (
            self._correlations.get(sym1, {}).get(sym2)
            or self._correlations.get(sym2, {}).get(sym1)
            or 0.5  # Default moderate correlation for unknown pairs
        )
