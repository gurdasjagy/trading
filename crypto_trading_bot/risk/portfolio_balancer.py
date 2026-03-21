"""Portfolio rebalancer — maintains target asset allocations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from loguru import logger


class PortfolioBalancer:
    """Rebalances the portfolio to maintain target allocations."""

    def calculate_current_allocation(self, positions: List[dict]) -> Dict[str, float]:
        """Return the current portfolio allocation by symbol as fractions (0–1).

        Args:
            positions: List of position dicts, each containing ``symbol`` and
                ``value`` (notional value in quote currency).

        Returns:
            Dict mapping symbol → fractional allocation.
        """
        total_value = sum(p.get("value", 0.0) for p in positions)
        if total_value <= 0:
            return {}
        allocation = {
            p["symbol"]: p.get("value", 0.0) / total_value for p in positions if p.get("symbol")
        }
        logger.debug("Current allocation: {}", allocation)
        return allocation

    def calculate_target_allocation(
        self,
        symbols: List[str],
        method: str = "equal",
        weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Calculate target allocation fractions for *symbols*.

        Args:
            symbols: List of trading symbols.
            method: Allocation method — ``"equal"`` or ``"custom"``.
            weights: Required when *method* is ``"custom"``; must sum to 1.

        Returns:
            Dict mapping symbol → target fractional allocation.
        """
        if not symbols:
            return {}
        if method == "equal":
            equal_share = 1.0 / len(symbols)
            target = {s: equal_share for s in symbols}
        elif method == "custom" and weights:
            total = sum(weights.values())
            target = {s: weights.get(s, 0.0) / total for s in symbols}
        else:
            equal_share = 1.0 / len(symbols)
            target = {s: equal_share for s in symbols}
        logger.debug("Target allocation ({}): {}", method, target)
        return target

    def get_rebalancing_trades(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        capital: float,
    ) -> List[dict]:
        """Calculate the trades needed to move from *current* to *target* allocation.

        Args:
            current: Current allocation fractions by symbol.
            target: Target allocation fractions by symbol.
            capital: Total portfolio capital in quote currency.

        Returns:
            List of trade dicts with keys ``symbol``, ``side``, and ``amount``.
        """
        trades: List[dict] = []
        all_symbols = set(current) | set(target)
        for symbol in all_symbols:
            current_frac = current.get(symbol, 0.0)
            target_frac = target.get(symbol, 0.0)
            delta_frac = target_frac - current_frac
            amount = abs(delta_frac * capital)
            if amount < 1.0:
                continue
            side = "buy" if delta_frac > 0 else "sell"
            trades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "delta_pct": delta_frac * 100.0,
                }
            )
        logger.info("Rebalancing trades needed: {}", len(trades))
        return trades

    def should_rebalance(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        threshold: float = 0.05,
    ) -> bool:
        """Return True if any allocation deviation exceeds *threshold*.

        Args:
            current: Current allocation fractions.
            target: Target allocation fractions.
            threshold: Maximum allowed absolute deviation (default 5 %).

        Returns:
            ``True`` if rebalancing is needed.
        """
        all_symbols = set(current) | set(target)
        for symbol in all_symbols:
            deviation = abs(current.get(symbol, 0.0) - target.get(symbol, 0.0))
            if deviation >= threshold:
                logger.info(
                    "Rebalance needed for {}: current={:.2%} target={:.2%} deviation={:.2%}",
                    symbol,
                    current.get(symbol, 0.0),
                    target.get(symbol, 0.0),
                    deviation,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Correlation-aware position management
    # ------------------------------------------------------------------

    def reduce_correlated_positions(
        self,
        positions: List[dict],
        correlation_matrix: Dict[str, Dict[str, float]],
        max_correlation: float = 0.75,
        max_correlated_exposure: float = 0.40,
    ) -> List[dict]:
        """Identify positions that should be reduced due to high correlation.

        When the combined allocation of two or more correlated symbols
        (correlation ≥ *max_correlation*) exceeds *max_correlated_exposure*,
        a proportional reduction is suggested for each symbol in the cluster.

        Args:
            positions: List of position dicts with ``symbol`` and ``value``.
            correlation_matrix: Nested dict ``symbol → symbol → correlation``
                where correlation is in [-1, 1].
            max_correlation: Threshold above which two symbols are considered
                correlated (default 0.75).
            max_correlated_exposure: Maximum combined exposure allowed for a
                correlated cluster as a fraction of total portfolio (default
                40 %).

        Returns:
            List of position dicts with a ``reduce_by`` key added when a
            reduction is recommended (fraction 0–1).  Unaffected positions
            are returned unchanged.
        """
        total_value = sum(p.get("value", 0.0) for p in positions)
        if total_value <= 0:
            return positions

        # Build symbol → position map
        pos_map: Dict[str, dict] = {p["symbol"]: p for p in positions if p.get("symbol")}
        symbols = list(pos_map.keys())

        # Find correlated clusters using union-find
        parent: Dict[str, str] = {s: s for s in symbols}

        def find(s: str) -> str:
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                corr = correlation_matrix.get(s1, {}).get(s2, 0.0)
                if abs(corr) >= max_correlation:
                    r1, r2 = find(s1), find(s2)
                    if r1 != r2:
                        parent[r1] = r2

        # Group symbols into clusters
        clusters: Dict[str, List[str]] = {}
        for s in symbols:
            root = find(s)
            clusters.setdefault(root, []).append(s)

        result = list(positions)
        for root, cluster in clusters.items():
            if len(cluster) < 2:
                continue
            cluster_value = sum(pos_map[s].get("value", 0.0) for s in cluster)
            cluster_frac = cluster_value / total_value
            if cluster_frac <= max_correlated_exposure:
                continue

            # Suggest proportional reduction
            reduce_factor = max_correlated_exposure / cluster_frac
            logger.info(
                "Correlated cluster {}: combined_exposure={:.2%} max={:.2%} reduce_factor={:.2f}",
                cluster,
                cluster_frac,
                max_correlated_exposure,
                reduce_factor,
            )
            for i, pos in enumerate(result):
                if pos.get("symbol") in cluster:
                    result[i] = dict(pos, reduce_by=round(1.0 - reduce_factor, 4))

        return result

    # ------------------------------------------------------------------
    # Drawdown-based rebalancing
    # ------------------------------------------------------------------

    def drawdown_rebalance(
        self,
        current: Dict[str, float],
        portfolio_drawdown_pct: float,
        drawdown_threshold: float = 0.10,
        reduction_factor: float = 0.50,
    ) -> Dict[str, float]:
        """Reduce all target allocations when portfolio drawdown exceeds a threshold.

        When the portfolio is in a significant drawdown, risk is reduced by
        scaling down all non-cash target allocations by *reduction_factor*.

        Args:
            current: Current allocation fractions by symbol (0–1).
            portfolio_drawdown_pct: Current portfolio drawdown as a positive
                percentage (e.g. 15.0 means 15 % drawdown from peak).
            drawdown_threshold: Drawdown level that triggers a reduction
                (default 10 %).
            reduction_factor: Fraction by which to scale down allocations
                (default 0.5 = cut exposure in half).

        Returns:
            Adjusted target allocation dict.  If drawdown is below the
            threshold, the *current* allocation is returned unchanged.
        """
        if portfolio_drawdown_pct < drawdown_threshold:
            return dict(current)

        logger.warning(
            "Portfolio drawdown {:.1f}% exceeds threshold {:.1f}% — reducing exposure by {:.0%}",
            portfolio_drawdown_pct,
            drawdown_threshold,
            1.0 - reduction_factor,
        )
        adjusted = {symbol: frac * reduction_factor for symbol, frac in current.items()}
        return adjusted

    def sector_rotation_weights(
        self,
        sector_momentum: Dict[str, float],
        symbols: List[str],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Calculate allocation weights based on sector momentum.

        Symbols in high-momentum sectors receive a larger allocation; those in
        low-momentum sectors receive a smaller one.  The weights are normalised
        to sum to 1.

        Args:
            sector_momentum: Dict mapping sector name → momentum score (any
                positive float; higher = more momentum).
            symbols: List of trading pair symbols to allocate across.
            sector_map: Optional dict mapping symbol → sector name.  If a
                symbol has no mapping, it falls into the ``"other"`` sector.

        Returns:
            Dict mapping symbol → normalised allocation weight.
        """
        if not symbols:
            return {}

        _sector_map: Dict[str, str] = sector_map or {}

        # Default sector categorisation for common crypto pairs
        _DEFAULT_SECTOR: Dict[str, str] = {
            "BTC/USDT": "l1",
            "ETH/USDT": "l1",
            "SOL/USDT": "l1",
            "BNB/USDT": "l1",
            "ADA/USDT": "l1",
        }

        weights: Dict[str, float] = {}
        for symbol in symbols:
            sector = _sector_map.get(symbol) or _DEFAULT_SECTOR.get(symbol, "other")
            momentum = sector_momentum.get(sector, 1.0)
            weights[symbol] = max(momentum, 0.01)  # floor to avoid zero weights

        total = sum(weights.values())
        normalised = {s: w / total for s, w in weights.items()}
        logger.debug("Sector rotation weights: {}", normalised)
        return normalised
