"""Reward shaping for reinforcement learning strategy optimization."""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger


class RewardShaper:
    """Shape raw PnL into better learning signals for RL optimization.

    Transforms raw profit/loss into normalized, risk-adjusted rewards that
    enable better strategy learning across different market conditions and
    trading pairs.
    """

    def __init__(
        self,
        atr_weight: float = 0.3,
        risk_adjusted_weight: float = 0.25,
        time_penalty_weight: float = 0.15,
        spread_penalty_weight: float = 0.15,
        regime_bonus_weight: float = 0.15,
    ) -> None:
        """Initialize RewardShaper with component weights.

        Args:
            atr_weight: Weight for ATR-normalized reward component.
            risk_adjusted_weight: Weight for risk-adjusted reward component.
            time_penalty_weight: Weight for time penalty component.
            spread_penalty_weight: Weight for spread cost component.
            regime_bonus_weight: Weight for regime consistency bonus.
        """
        self.atr_weight = atr_weight
        self.risk_adjusted_weight = risk_adjusted_weight
        self.time_penalty_weight = time_penalty_weight
        self.spread_penalty_weight = spread_penalty_weight
        self.regime_bonus_weight = regime_bonus_weight

        # Validate weights sum to 1.0
        total = sum([
            atr_weight,
            risk_adjusted_weight,
            time_penalty_weight,
            spread_penalty_weight,
            regime_bonus_weight,
        ])
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"RewardShaper weights sum to {total:.3f}, not 1.0. "
                "Reward components may not be balanced."
            )

    def shape_reward(
        self,
        pnl: float,
        position_size: float,
        atr: float,
        max_drawdown: float,
        holding_time: float,
        expected_holding_time: float,
        spread_cost: float,
        strategy_regime: str,
        actual_regime: str,
    ) -> float:
        """Compute shaped reward from trade outcome.

        Args:
            pnl: Raw profit/loss in quote currency (e.g., USDT).
            position_size: Position size in quote currency.
            atr: Average True Range at trade entry.
            max_drawdown: Maximum unrealized loss during the trade.
            holding_time: Actual time trade was held (seconds).
            expected_holding_time: Expected holding time for this strategy (seconds).
            spread_cost: Estimated spread cost paid on entry/exit.
            strategy_regime: Regime the strategy was designed for.
            actual_regime: Actual market regime during the trade.

        Returns:
            Composite shaped reward value (typically in range [-5, 5]).
        """
        # 1. ATR-normalized reward: makes rewards comparable across pairs
        atr_normalized = 0.0
        if atr > 0 and position_size > 0:
            atr_normalized = pnl / (atr * position_size)
        else:
            atr_normalized = pnl / max(position_size, 1.0)

        # 2. Risk-adjusted reward: penalize trades with high drawdown
        risk_adjusted = 0.0
        if max_drawdown > 0:
            risk_adjusted = pnl / max_drawdown
        else:
            risk_adjusted = atr_normalized  # fallback to ATR-normalized

        # 3. Time-penalized reward: discourage trades that hold too long
        time_penalty = 0.0
        if expected_holding_time > 0:
            time_ratio = holding_time / expected_holding_time
            # Penalty grows quadratically beyond expected time
            if time_ratio > 1.0:
                time_penalty = -0.5 * (time_ratio - 1.0) ** 2
            else:
                # Small bonus for quick trades
                time_penalty = 0.1 * (1.0 - time_ratio)

        # 4. Spread-adjusted reward: subtract spread cost
        spread_adjusted = -abs(spread_cost) / max(position_size, 1.0)

        # 5. Regime-consistency bonus: reward strategies used in correct regimes
        regime_bonus = 0.0
        if strategy_regime == actual_regime and pnl > 0:
            regime_bonus = 0.2
        elif strategy_regime != actual_regime and pnl > 0:
            regime_bonus = -0.1  # slight penalty for winning in wrong regime

        # Composite reward
        composite = (
            self.atr_weight * atr_normalized
            + self.risk_adjusted_weight * risk_adjusted
            + self.time_penalty_weight * time_penalty
            + self.spread_penalty_weight * spread_adjusted
            + self.regime_bonus_weight * regime_bonus
        )

        # Clip to reasonable range
        composite = max(-5.0, min(5.0, composite))

        logger.debug(
            f"RewardShaper: pnl={pnl:.2f} → composite={composite:.3f} "
            f"(atr={atr_normalized:.3f}, risk={risk_adjusted:.3f}, "
            f"time={time_penalty:.3f}, spread={spread_adjusted:.3f}, "
            f"regime={regime_bonus:.3f})"
        )

        return composite

    def shape_batch(
        self, trade_outcomes: list[Dict]
    ) -> list[float]:
        """Shape rewards for a batch of trade outcomes.

        Args:
            trade_outcomes: List of dicts with keys: pnl, position_size, atr,
                max_drawdown, holding_time, expected_holding_time, spread_cost,
                strategy_regime, actual_regime.

        Returns:
            List of shaped reward values.
        """
        shaped = []
        for outcome in trade_outcomes:
            reward = self.shape_reward(
                pnl=outcome.get("pnl", 0.0),
                position_size=outcome.get("position_size", 1.0),
                atr=outcome.get("atr", 1.0),
                max_drawdown=outcome.get("max_drawdown", 1.0),
                holding_time=outcome.get("holding_time", 0.0),
                expected_holding_time=outcome.get("expected_holding_time", 3600.0),
                spread_cost=outcome.get("spread_cost", 0.0),
                strategy_regime=outcome.get("strategy_regime", "unknown"),
                actual_regime=outcome.get("actual_regime", "unknown"),
            )
            shaped.append(reward)
        return shaped
