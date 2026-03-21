"""Anti-liquidation manager — monitors margin usage and prevents forced liquidations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class LiquidationRisk:
    """Liquidation risk assessment for a single position."""

    symbol: str
    entry_price: float
    current_price: float
    liquidation_price: float
    leverage: float
    side: str           # "long" or "short"
    notional_value: float
    distance_pct: float  # % distance from current price to liquidation price
    risk_level: str      # "safe", "warning", "danger", "critical"


class AntiLiquidationManager:
    """Monitors open positions and takes automatic action to prevent liquidation.

    Action thresholds (distance from liquidation price):
    * ≤ 40 % → WARNING alert
    * ≤ 15 % → Reduce position by 50 %
    * ≤  5 % → Close position entirely

    Portfolio-level constraints:
    * Total margin usage > 70 % → start reducing positions
    * Total portfolio leverage > 3× equity → reduce positions
    """

    # Distance thresholds
    _WARNING_DISTANCE_PCT: float = 0.40   # 40 %
    _REDUCE_DISTANCE_PCT: float = 0.15    # 15 %
    _CLOSE_DISTANCE_PCT: float = 0.05     # 5 %

    # Portfolio constraints
    _MAX_MARGIN_USAGE_PCT: float = 0.70   # 70 %
    _MAX_PORTFOLIO_LEVERAGE: float = 3.0  # 3× equity

    def _get_maintenance_margin(self, notional_value: float) -> float:
        """Return the maintenance margin rate based on position notional value.

        Gate.io uses tiered maintenance margins based on position size.
        """
        # Gate.io USDT-M futures maintenance margin tiers (approximate)
        tiers = [
            (50_000, 0.005),       # ≤50K: 0.5%
            (200_000, 0.01),       # ≤200K: 1.0%
            (1_000_000, 0.02),     # ≤1M: 2.0%
            (5_000_000, 0.025),    # ≤5M: 2.5%
            (float("inf"), 0.05),  # >5M: 5.0%
        ]
        for threshold, rate in tiers:
            if notional_value <= threshold:
                return rate
        return 0.05

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        side: str,
        margin_pct: float = 1.0,
        position_size: float = 0.0,
    ) -> float:
        """Calculate the liquidation price for a leveraged position.

        Uses a simplified cross-margin formula with tiered maintenance margins.
        The actual liquidation price depends on the exchange's maintenance margin
        requirements.

        For **longs**: liq_price = entry × (1 - 1/leverage + maintenance_margin)
        For **shorts**: liq_price = entry × (1 + 1/leverage - maintenance_margin)

        Args:
            entry_price: Trade entry price.
            leverage: Position leverage multiplier.
            side: ``"long"`` or ``"short"``.
            margin_pct: Initial margin percentage (1/leverage by default).
            position_size: Position size in base units (used for accurate notional
                tier lookup).  Falls back to a 100-unit estimate when 0.

        Returns:
            Estimated liquidation price.
        """
        if leverage <= 0 or entry_price <= 0:
            return 0.0

        # Use actual position size for notional when available; fall back to a
        # rough ~100-unit estimate so the maintenance margin tier is accurate.
        if position_size > 0:
            notional_estimate = entry_price * position_size
        else:
            notional_estimate = entry_price * 100  # rough estimate
        maintenance_margin = self._get_maintenance_margin(notional_estimate)
        initial_margin = 1.0 / leverage

        if side == "long":
            liq_price = entry_price * (1.0 - initial_margin + maintenance_margin)
        else:
            liq_price = entry_price * (1.0 + initial_margin - maintenance_margin)

        return max(0.0, round(liq_price, 8))

    def assess_position_risk(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        leverage: float,
        side: str,
        position_size: float,
        liquidation_price: float = 0.0,
    ) -> LiquidationRisk:
        """Assess the liquidation risk for a single position.

        Args:
            symbol: Trading symbol.
            entry_price: Position entry price.
            current_price: Current market price.
            leverage: Position leverage.
            side: ``"long"`` or ``"short"``.
            position_size: Position size in base units.
            liquidation_price: Exchange-reported liquidation price.  When
                non-zero this value is used directly instead of the calculated
                estimate (the exchange knows the exact maintenance margin
                tiers).

        Returns:
            :class:`LiquidationRisk` assessment.
        """
        if liquidation_price > 0:
            # Prefer the exchange-supplied liquidation price — it accounts for
            # the exact maintenance margin tier and any cross/isolated margin
            # adjustments that our simplified formula cannot replicate.
            liq_price = liquidation_price
        else:
            liq_price = self.calculate_liquidation_price(
                entry_price, leverage, side, position_size=position_size
            )
        notional = position_size * current_price

        if liq_price <= 0 or current_price <= 0:
            distance_pct = 1.0  # assume safe if we can't calculate
        elif side == "long":
            distance_pct = (current_price - liq_price) / current_price
        else:
            distance_pct = (liq_price - current_price) / current_price

        distance_pct = max(0.0, distance_pct)

        if distance_pct <= self._CLOSE_DISTANCE_PCT:
            risk_level = "critical"
        elif distance_pct <= self._REDUCE_DISTANCE_PCT:
            risk_level = "danger"
        elif distance_pct <= self._WARNING_DISTANCE_PCT:
            risk_level = "warning"
        else:
            risk_level = "safe"

        logger.debug(
            "[AntiLiq] {} {} liq={:.4f} dist={:.1%} risk={}",
            symbol,
            side,
            liq_price,
            distance_pct,
            risk_level,
        )
        return LiquidationRisk(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            liquidation_price=liq_price,
            leverage=leverage,
            side=side,
            notional_value=notional,
            distance_pct=distance_pct,
            risk_level=risk_level,
        )

    def get_action(self, risk: LiquidationRisk) -> Optional[str]:
        """Return the required action for a given liquidation risk assessment.

        Args:
            risk: :class:`LiquidationRisk` for a position.

        Returns:
            One of ``"alert"``, ``"reduce_50pct"``, ``"close_position"``, or
            ``None`` when the position is safe.
        """
        if risk.risk_level == "critical":
            logger.warning(
                "[AntiLiq] CRITICAL: {} within {:.1%} of liquidation — closing",
                risk.symbol,
                risk.distance_pct,
            )
            return "close_position"
        if risk.risk_level == "danger":
            logger.warning(
                "[AntiLiq] DANGER: {} within {:.1%} of liquidation — reducing by 50 %",
                risk.symbol,
                risk.distance_pct,
            )
            return "reduce_50pct"
        if risk.risk_level == "warning":
            logger.info(
                "[AntiLiq] WARNING: {} within {:.1%} of liquidation",
                risk.symbol,
                risk.distance_pct,
            )
            return "alert"
        return None

    def assess_portfolio(
        self,
        positions: List[Dict[str, Any]],
        equity: float,
    ) -> Dict[str, Any]:
        """Assess portfolio-level margin and leverage constraints.

        Args:
            positions: List of position dicts.  Each should contain:
                ``symbol``, ``entry_price``, ``current_price``, ``leverage``,
                ``side``, ``size``, ``margin`` (optional).
            equity: Total account equity.

        Returns:
            Dict with keys ``margin_usage_pct``, ``portfolio_leverage``,
            ``exceeds_margin_limit``, ``exceeds_leverage_limit``, and
            ``recommended_action``.
        """
        if equity <= 0:
            return {
                "margin_usage_pct": 1.0,
                "portfolio_leverage": 0.0,
                "exceeds_margin_limit": True,
                "exceeds_leverage_limit": False,
                "recommended_action": "close_all",
            }

        total_notional = 0.0
        total_margin = 0.0
        for pos in positions:
            size = float(pos.get("size", 0.0))
            price = float(pos.get("current_price", pos.get("entry_price", 0.0)))
            lev = float(pos.get("leverage", 1.0))
            notional = size * price
            margin = notional / lev if lev > 0 else notional
            total_notional += notional
            total_margin += margin

        margin_usage_pct = total_margin / equity if equity > 0 else 1.0
        portfolio_leverage = total_notional / equity if equity > 0 else 0.0

        exceeds_margin = margin_usage_pct > self._MAX_MARGIN_USAGE_PCT
        exceeds_leverage = portfolio_leverage > self._MAX_PORTFOLIO_LEVERAGE

        action = None
        if exceeds_leverage or exceeds_margin:
            action = "reduce_positions"

        logger.debug(
            "[AntiLiq] Portfolio: margin={:.1%} leverage={:.2f}x margins_ok={} lev_ok={}",
            margin_usage_pct,
            portfolio_leverage,
            not exceeds_margin,
            not exceeds_leverage,
        )
        return {
            "margin_usage_pct": round(margin_usage_pct, 4),
            "portfolio_leverage": round(portfolio_leverage, 4),
            "exceeds_margin_limit": exceeds_margin,
            "exceeds_leverage_limit": exceeds_leverage,
            "recommended_action": action,
        }

    async def auto_topup_margin(
        self,
        exchange: Any,
        symbol: str,
        position: dict,
        equity: float,
    ) -> bool:
        """Auto top-up margin when within liquidation_margin_topup_pct of liquidation.

        Adds a small margin top-up for isolated-margin positions that are
        approaching their liquidation price.  The top-up is capped at the
        lesser of 50 USDT, 5 % of equity, and 50 % of current margin.

        Args:
            exchange: Exchange client with an ``add_margin`` method.
            symbol: Trading symbol.
            position: Position dict with at least ``distance_pct`` and
                ``margin`` keys.
            equity: Current account equity in USDT.

        Returns:
            ``True`` if the top-up was successfully submitted, ``False``
            otherwise.
        """
        dist_pct = position.get("distance_pct", 1.0)
        if dist_pct > 0.03:  # Only top up when within 3%
            return False
        margin = position.get("margin", 0)
        topup_amount = min(margin * 0.5, 50.0, equity * 0.05)  # Max 50 USDT or 5% equity
        if topup_amount < 5.0:
            return False
        try:
            await exchange.add_margin(symbol, topup_amount)
            logger.info(
                "[AntiLiq] Auto top-up margin for {}: +{:.2f} USDT (dist={:.1%})",
                symbol,
                topup_amount,
                dist_pct,
            )
            return True
        except Exception as exc:
            logger.warning("[AntiLiq] auto_topup_margin failed for {}: {}", symbol, exc)
            return False

    def screen_all_positions(
        self,
        positions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assess liquidation risk for all open positions.

        Args:
            positions: List of position dicts (see :meth:`assess_portfolio`).

        Returns:
            List of dicts, each containing the original position data augmented
            with a ``liquidation_risk`` key and ``recommended_action`` key.
        """
        results = []
        for pos in positions:
            risk = self.assess_position_risk(
                symbol=str(pos.get("symbol", "")),
                entry_price=float(pos.get("entry_price", 0.0)),
                current_price=float(pos.get("current_price", pos.get("entry_price", 0.0))),
                leverage=float(pos.get("leverage", 1.0)),
                side=str(pos.get("side", "long")).lower(),
                position_size=float(pos.get("size", 0.0)),
                liquidation_price=float(pos.get("liquidation_price", 0.0)),
            )
            action = self.get_action(risk)
            results.append({
                **pos,
                "liquidation_risk": {
                    "liquidation_price": risk.liquidation_price,
                    "distance_pct": risk.distance_pct,
                    "risk_level": risk.risk_level,
                },
                "recommended_action": action,
            })
        return results
