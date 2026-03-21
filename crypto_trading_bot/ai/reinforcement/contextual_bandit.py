"""LinUCB contextual bandit for strategy selection.

Replaces Thompson Sampling with a Linear Upper Confidence Bound (LinUCB)
algorithm.  Each strategy is an "arm" with its own linear reward model.
Context features drive the selection decision; the exploration bonus decays
as confidence in reward estimates grows.

Non-stationary rewards are handled via a sliding window of the last 200
observations per arm.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger


class LinUCBBandit:
    """Linear Upper Confidence Bound contextual bandit.

    Maintains per-arm ridge-regression models: for arm *a*, ``A_a`` (the
    design matrix) and ``b_a`` (the response vector).  At selection time the
    UCB score for arm *a* is:

        p_a = theta_a @ context + alpha * sqrt(context @ A_a^-1 @ context)

    where ``theta_a = A_a^-1 @ b_a``.

    The exploration coefficient *alpha* decays from ``alpha_init`` towards
    ``alpha_min`` as the number of pulls grows, ensuring reduced exploration
    once estimates are reliable.

    Args:
        num_arms: Total number of strategy arms.
        context_dim: Dimension of the context feature vector.
        alpha: Initial exploration coefficient.
        alpha_min: Floor for the decaying exploration coefficient.
        alpha_decay: Multiplicative decay applied to alpha after each update.
        window_size: Maximum number of observations kept per arm (sliding
            window for non-stationary rewards).
    """

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        alpha_min: float = 0.1,
        alpha_decay: float = 0.9995,
        window_size: int = 200,
    ) -> None:
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.window_size = window_size

        # Per-arm ridge matrices and response vectors
        self._A: List[np.ndarray] = [
            np.identity(context_dim, dtype=float) for _ in range(num_arms)
        ]
        self._b: List[np.ndarray] = [
            np.zeros(context_dim, dtype=float) for _ in range(num_arms)
        ]

        # Sliding window of (context, reward) pairs for windowed updates
        self._window: List[deque] = [
            deque(maxlen=window_size) for _ in range(num_arms)
        ]

        # Pull counts for diagnostic purposes
        self._pulls: List[int] = [0] * num_arms

        logger.info(
            "LinUCBBandit initialised: arms={}, context_dim={}, alpha={:.3f}",
            num_arms,
            context_dim,
            alpha,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_arm(self, context: np.ndarray) -> int:
        """Select the arm with the highest UCB score.

        Args:
            context: 1-D context feature vector of length ``context_dim``.

        Returns:
            Index of the selected arm.
        """
        context = np.asarray(context, dtype=float).flatten()
        if len(context) != self.context_dim:
            raise ValueError(
                f"Context dim mismatch: expected {self.context_dim}, got {len(context)}"
            )

        scores = self._compute_scores(context)
        chosen = int(np.argmax(scores))

        logger.trace(
            "LinUCB select_arm: chosen={}, top_score={:.4f}, alpha={:.4f}",
            chosen,
            scores[chosen],
            self.alpha,
        )
        return chosen

    def select_top_k(self, context: np.ndarray, k: int) -> List[int]:
        """Return the indices of the top-*k* arms by UCB score.

        Args:
            context: Context feature vector.
            k: Number of arms to return.

        Returns:
            List of arm indices (highest score first).
        """
        context = np.asarray(context, dtype=float).flatten()
        scores = self._compute_scores(context)
        top_k = np.argsort(scores)[::-1][:k].tolist()
        return top_k

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update the model for *arm* given observed *reward*.

        Args:
            arm: Arm index that was pulled.
            context: Context vector used when the arm was selected.
            reward: Observed reward (e.g. shaped Sharpe contribution).
        """
        if not (0 <= arm < self.num_arms):
            raise IndexError(f"Arm {arm} out of range [0, {self.num_arms})")

        ctx = np.asarray(context, dtype=float).flatten()

        # Sliding-window: rebuild A / b from the last window_size observations
        self._window[arm].append((ctx.copy(), float(reward)))
        self._rebuild_arm(arm)

        # Decay exploration coefficient
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self._pulls[arm] += 1

        logger.debug(
            "LinUCB update: arm={}, reward={:.4f}, pulls={}, alpha={:.4f}",
            arm,
            reward,
            self._pulls[arm],
            self.alpha,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_scores(self, context: np.ndarray) -> List[float]:
        """Compute UCB scores for all arms given *context*."""
        scores: List[float] = []
        for a in range(self.num_arms):
            A_inv = np.linalg.solve(self._A[a], np.identity(self.context_dim))
            theta = A_inv @ self._b[a]
            exploitation = float(theta @ context)
            exploration = float(
                self.alpha * math.sqrt(max(0.0, float(context @ A_inv @ context)))
            )
            scores.append(exploitation + exploration)
        return scores

    def _rebuild_arm(self, arm: int) -> None:
        """Rebuild A_arm and b_arm from the sliding window."""
        A = np.identity(self.context_dim, dtype=float)
        b = np.zeros(self.context_dim, dtype=float)
        for ctx, rew in self._window[arm]:
            A += np.outer(ctx, ctx)
            b += rew * ctx
        self._A[arm] = A
        self._b[arm] = b

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_arm_stats(self) -> List[Dict[str, Any]]:
        """Return diagnostic stats for every arm."""
        stats = []
        for a in range(self.num_arms):
            A_inv = np.linalg.solve(self._A[a], np.identity(self.context_dim))
            theta = A_inv @ self._b[a]
            stats.append(
                {
                    "arm": a,
                    "pulls": self._pulls[a],
                    "theta_norm": float(np.linalg.norm(theta)),
                    "window_size": len(self._window[a]),
                    "alpha": self.alpha,
                }
            )
        return stats

    @staticmethod
    def get_context_features() -> Tuple[str, ...]:
        """Return the canonical ordering of context feature names.

        Callers should build the context vector in this order:
        market_regime_idx, volatility_norm, hour_norm, win_rate,
        funding_rate, correlation_regime_idx, drawdown_norm,
        volume_trend.
        """
        return (
            "market_regime_idx",
            "volatility_norm",
            "hour_norm",
            "win_rate",
            "funding_rate",
            "correlation_regime_idx",
            "drawdown_norm",
            "volume_trend",
        )

    def __repr__(self) -> str:
        return (
            f"LinUCBBandit(arms={self.num_arms}, "
            f"context_dim={self.context_dim}, alpha={self.alpha:.4f})"
        )
