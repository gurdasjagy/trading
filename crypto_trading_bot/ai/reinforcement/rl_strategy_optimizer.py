"""Contextual multi-armed bandit strategy optimizer using Thompson Sampling."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats

from ai.reinforcement.experience_buffer import ExperienceBuffer
from ai.reinforcement.reward_shaper import RewardShaper


class RLStrategyOptimizer:
    """Contextual multi-armed bandit for strategy selection with Thompson Sampling.

    Each strategy is treated as an arm. The optimizer maintains Beta distributions
    (for binary rewards) or Normal-Inverse-Gamma distributions (for continuous rewards)
    as posterior beliefs about each strategy's performance in different contexts.

    Features:
    - Thompson Sampling for exploration-exploitation balance
    - Contextual features (regime, volatility, time, etc.)
    - Exponential decay on old observations for adaptation
    - UCB1 fallback when insufficient data
    - Top-K selection for portfolio diversification
    - Warm-up period using heuristic selection
    """

    def __init__(
        self,
        num_strategies: int,
        context_dim: int = 25,
        decay_factor: float = 0.995,
        epsilon: float = 0.1,
        warm_up_trades: int = 100,
        top_k: int = 7,
        min_trades_for_thompson: int = 10,
        state_dir: Path = Path("data"),
    ) -> None:
        """Initialize RL optimizer.

        Args:
            num_strategies: Total number of strategies (arms).
            context_dim: Dimension of context feature vector.
            decay_factor: Exponential decay for old observations (0.995 = ~200 trades half-life).
            epsilon: Epsilon-greedy exploration rate (fallback).
            warm_up_trades: Number of trades to use heuristic before enabling RL.
            top_k: Number of strategies to select per cycle.
            min_trades_for_thompson: Minimum trades per arm before using Thompson Sampling.
            state_dir: Directory for saving/loading model state.
        """
        self.num_strategies = num_strategies
        self.context_dim = context_dim
        self.decay_factor = decay_factor
        self.epsilon = epsilon
        self.warm_up_trades = warm_up_trades
        self.top_k = top_k
        self.min_trades_for_thompson = min_trades_for_thompson
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Posterior parameters for each arm (strategy)
        # Using Normal-Inverse-Gamma for continuous rewards
        # Each arm has: (mean, precision, alpha, beta) parameters
        self._arm_params: List[Dict[str, float]] = []
        for _ in range(num_strategies):
            self._arm_params.append({
                "mean": 0.0,       # posterior mean
                "kappa": 1.0,      # precision parameter
                "alpha": 1.0,      # inverse-gamma shape
                "beta": 1.0,       # inverse-gamma scale
                "n": 0,            # effective sample count
            })

        # Per-arm statistics
        self._arm_pulls: List[int] = [0] * num_strategies
        self._arm_rewards: List[deque] = [deque(maxlen=500) for _ in range(num_strategies)]

        # Experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=10000)

        # Reward shaper
        self.reward_shaper = RewardShaper()

        # Total trades processed
        self._total_trades = 0

        # NOTE: Do NOT call _load_state here; await initialize() instead.

        logger.info(
            f"RLStrategyOptimizer initialized: {num_strategies} arms, "
            f"warm_up={warm_up_trades}, top_k={top_k}"
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def select_strategies(
        self,
        context: Dict[str, float],
        strategy_names: List[str],
        heuristic_fallback: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Select top-K strategies using Thompson Sampling.

        Args:
            context: Context feature dict (regime, volatility, hour, etc.).
            strategy_names: List of all strategy names (length = num_strategies).
            heuristic_fallback: Heuristic selection to use during warm-up.

        Returns:
            Tuple of (selected_strategy_names, selection_metadata).
        """
        # Warm-up: use heuristic
        if self._total_trades < self.warm_up_trades:
            if heuristic_fallback:
                logger.debug(
                    f"RL warm-up: {self._total_trades}/{self.warm_up_trades} trades, "
                    f"using heuristic selection"
                )
                return heuristic_fallback[:self.top_k], {"method": "heuristic_warmup"}

        # Check if we have enough data for Thompson Sampling
        arms_with_data = sum(1 for n in self._arm_pulls if n >= self.min_trades_for_thompson)

        if arms_with_data < self.top_k:
            # Not enough data: use UCB1
            selected_indices = self._select_ucb1(self.top_k)
            method = "ucb1_coldstart"
        else:
            # Thompson Sampling
            if np.random.random() < self.epsilon:
                # Epsilon-greedy exploration
                selected_indices = np.random.choice(
                    self.num_strategies, size=self.top_k, replace=False
                ).tolist()
                method = "epsilon_greedy"
            else:
                selected_indices = self._select_thompson_sampling(context, self.top_k)
                method = "thompson_sampling"

        selected_names = [strategy_names[i] for i in selected_indices]

        metadata = {
            "method": method,
            "selected_indices": selected_indices,
            "total_trades": self._total_trades,
            "arms_with_data": arms_with_data,
        }

        logger.debug(
            f"RL selection ({method}): {selected_names[:3]}... "
            f"(total_trades={self._total_trades})"
        )

        return selected_names, metadata

    def _select_thompson_sampling(self, context: Dict[str, float], k: int) -> List[int]:
        """Select top-K arms using Thompson Sampling.

        Samples from each arm's posterior distribution and selects the K arms
        with highest sampled values.
        """
        samples = []
        for i in range(self.num_strategies):
            params = self._arm_params[i]

            # Sample from Normal-Inverse-Gamma posterior
            # First sample variance from Inverse-Gamma
            variance = stats.invgamma.rvs(
                a=params["alpha"],
                scale=params["beta"],
            )

            # Then sample mean from Normal given variance
            mean_sample = np.random.normal(
                loc=params["mean"],
                scale=math.sqrt(variance / params["kappa"]),
            )

            samples.append(mean_sample)

        # Select top-K arms
        top_k_indices = np.argsort(samples)[-k:].tolist()
        top_k_indices.reverse()  # Highest first

        return top_k_indices

    def _select_ucb1(self, k: int) -> List[int]:
        """Select top-K arms using UCB1 (Upper Confidence Bound).

        Used as fallback when Thompson Sampling has insufficient data.
        """
        ucb_values = []
        total_pulls = sum(self._arm_pulls) or 1

        for i in range(self.num_strategies):
            if self._arm_pulls[i] == 0:
                ucb = float('inf')  # Unplayed arms have infinite UCB
            else:
                mean_reward = self._arm_params[i]["mean"]
                exploration_bonus = math.sqrt(2 * math.log(total_pulls) / self._arm_pulls[i])
                ucb = mean_reward + exploration_bonus

            ucb_values.append(ucb)

        # Select top-K
        top_k_indices = np.argsort(ucb_values)[-k:].tolist()
        top_k_indices.reverse()

        return top_k_indices

    # ------------------------------------------------------------------
    # Learning updates
    # ------------------------------------------------------------------

    async def update(
        self,
        strategy_index: int,
        trade_outcome: Dict[str, Any],
        context: Dict[str, float],
        next_context: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update posterior after observing a trade outcome.

        Args:
            strategy_index: Index of strategy that generated the trade.
            trade_outcome: Dict with keys: pnl, position_size, atr, max_drawdown,
                holding_time, expected_holding_time, spread_cost, strategy_regime,
                actual_regime.
            context: Context features at trade entry.
            next_context: Context features at trade exit (optional).
        """
        # Shape the reward
        reward = self.reward_shaper.shape_reward(
            pnl=trade_outcome.get("pnl", 0.0),
            position_size=trade_outcome.get("position_size", 1.0),
            atr=trade_outcome.get("atr", 1.0),
            max_drawdown=trade_outcome.get("max_drawdown", 1.0),
            holding_time=trade_outcome.get("holding_time", 0.0),
            expected_holding_time=trade_outcome.get("expected_holding_time", 3600.0),
            spread_cost=trade_outcome.get("spread_cost", 0.0),
            strategy_regime=trade_outcome.get("strategy_regime", "unknown"),
            actual_regime=trade_outcome.get("actual_regime", "unknown"),
        )

        # Store in experience buffer
        self.experience_buffer.add(
            state=context,
            action=strategy_index,
            reward=reward,
            next_state=next_context or context,
            done=False,
        )

        # Update posterior for this arm
        self._update_posterior(strategy_index, reward)

        # Increment counters
        self._arm_pulls[strategy_index] += 1
        self._arm_rewards[strategy_index].append(reward)
        self._total_trades += 1

        # Periodic save
        if self._total_trades % 10 == 0:
            await self._save_state()

        logger.debug(
            f"RL update: strategy_idx={strategy_index}, reward={reward:.3f}, "
            f"pulls={self._arm_pulls[strategy_index]}, total={self._total_trades}"
        )

    def _update_posterior(self, arm_idx: int, reward: float) -> None:
        """Update Normal-Inverse-Gamma posterior for an arm.

        Uses Bayesian update with exponential decay on old observations.
        """
        params = self._arm_params[arm_idx]

        # Apply decay to effective sample count
        params["n"] = params["n"] * self.decay_factor + 1.0

        # Update sufficient statistics
        old_mean = params["mean"]
        new_n = params["n"]

        # Incremental mean update
        params["mean"] = old_mean + (reward - old_mean) / new_n

        # Update precision parameter (kappa)
        params["kappa"] = params["kappa"] * self.decay_factor + 1.0

        # Update Inverse-Gamma parameters (alpha, beta)
        # This approximates tracking variance
        params["alpha"] = params["alpha"] * self.decay_factor + 0.5
        squared_error = (reward - old_mean) ** 2
        params["beta"] = params["beta"] * self.decay_factor + 0.5 * squared_error

        logger.trace(
            f"Posterior update arm {arm_idx}: mean={params['mean']:.3f}, "
            f"kappa={params['kappa']:.2f}, n={params['n']:.1f}"
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    async def _save_state(self) -> None:
        """Save model state to disk."""
        state = {
            "arm_params": self._arm_params,
            "arm_pulls": self._arm_pulls,
            "total_trades": self._total_trades,
            "timestamp": time.time(),
        }

        state_file = self.state_dir / "rl_optimizer_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"RLStrategyOptimizer: saved state to {state_file}")
        except Exception as exc:
            logger.error(f"Failed to save RL state: {exc}")

        # Save experience buffer
        try:
            await self.experience_buffer.save()
        except Exception as exc:
            logger.error(f"Failed to save experience buffer: {exc}")

    async def _load_state(self) -> None:
        """Load model state from disk."""
        state_file = self.state_dir / "rl_optimizer_state.json"
        if not state_file.exists():
            logger.debug("No RL state file found, starting fresh")
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self._arm_params = state.get("arm_params", self._arm_params)
            self._arm_pulls = state.get("arm_pulls", self._arm_pulls)
            self._total_trades = state.get("total_trades", 0)

            # Reconstruct arm_rewards from experience buffer
            try:
                await self.experience_buffer.load()
                # Rebuild arm_rewards from buffer
                self._arm_rewards = [deque(maxlen=500) for _ in range(self.num_strategies)]
                for exp in self.experience_buffer._buffer:
                    action = exp["action"]
                    reward = exp["reward"]
                    if 0 <= action < self.num_strategies:
                        self._arm_rewards[action].append(reward)
            except Exception as exc:
                logger.warning(f"Could not load experience buffer: {exc}")

            logger.info(
                f"RLStrategyOptimizer: loaded state from {state_file} "
                f"(total_trades={self._total_trades})"
            )
        except Exception as exc:
            logger.error(f"Failed to load RL state: {exc}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the optimizer by loading persisted state from disk.

        Must be awaited after construction before using update().
        """
        await self.experience_buffer.initialize()
        await self._load_state()

    def get_arm_stats(self, arm_idx: int) -> Dict[str, float]:
        """Get statistics for a specific arm.

        Args:
            arm_idx: Index of the arm.

        Returns:
            Dict with keys: pulls, mean_reward, std_reward, posterior_mean.
        """
        if not (0 <= arm_idx < self.num_strategies):
            return {}

        rewards = self._arm_rewards[arm_idx]
        return {
            "pulls": self._arm_pulls[arm_idx],
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "std_reward": np.std(rewards) if len(rewards) > 1 else 0.0,
            "posterior_mean": self._arm_params[arm_idx]["mean"],
            "posterior_n": self._arm_params[arm_idx]["n"],
        }

    def get_all_arm_stats(self) -> List[Dict[str, float]]:
        """Get statistics for all arms."""
        return [self.get_arm_stats(i) for i in range(self.num_strategies)]

    def reset(self) -> None:
        """Reset all learning state (for testing or fresh start)."""
        self._arm_params = []
        for _ in range(self.num_strategies):
            self._arm_params.append({
                "mean": 0.0,
                "kappa": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "n": 0,
            })
        self._arm_pulls = [0] * self.num_strategies
        self._arm_rewards = [deque(maxlen=500) for _ in range(self.num_strategies)]
        self._total_trades = 0
        self.experience_buffer.clear()
        logger.info("RLStrategyOptimizer: reset all state")

    def __repr__(self) -> str:
        return (
            f"RLStrategyOptimizer(strategies={self.num_strategies}, "
            f"trades={self._total_trades}, top_k={self.top_k})"
        )
