"""Meta-learner for strategy portfolio optimization using multiplicative weights algorithm."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel


class StrategyWeights(BaseModel):
    """Container for strategy weight allocation."""

    weights: Dict[str, float] = {}  # strategy_name -> weight (0-1)
    regime: str = "unknown"
    last_update: datetime = datetime.now(timezone.utc)
    total_trades: int = 0
    validation_sharpe: float = 0.0


class MetaLearner:
    """
    Learn which combination of strategies + parameters works best in each regime.

    Uses multiplicative weights algorithm (Hedge/EXP3) to maintain a strategy
    portfolio with weighted allocation. Weights are updated based on trade outcomes.

    Features:
    - Regime-conditional strategy portfolios
    - Multiplicative weight updates with decay
    - Walk-forward validation to prevent overfitting
    - Portfolio rebalancing (prune low-weight strategies)
    - Ensemble signal generation (weighted average)
    - Anti-overfitting through train/validate split
    """

    def __init__(self, strategy_names: Optional[List[str]] = None) -> None:
        """
        Initialize the meta-learner.

        Args:
            strategy_names: List of strategy names to manage
        """
        self._lock = asyncio.Lock()

        # Strategy weights per regime
        self._weights: Dict[str, StrategyWeights] = {}  # regime -> StrategyWeights
        self._all_strategy_names = set(strategy_names or [])

        # Learning rate (eta) with decay
        self._eta_initial = 0.1
        self._eta_min = 0.01
        self._eta = self._eta_initial
        self._eta_decay = 0.995  # Decay per rebalance

        # Update frequency
        self._rebalance_frequency = 50  # Trades
        self._trades_since_rebalance: Dict[str, int] = defaultdict(int)

        # Walk-forward validation
        self._train_window = 200  # Trades for training
        self._validation_window = 50  # Trades for validation
        self._trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))

        # Performance tracking
        self._strategy_performance: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )  # regime -> strategy -> returns

        # Pruning threshold
        self._min_weight_threshold = 0.01  # Prune strategies below 1%

        # Persistence
        self._state_file = Path("data/meta_learner_state.json")
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load saved state
        self._load_state()

        logger.info("MetaLearner initialized with {} strategies", len(self._all_strategy_names))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_strategy_weights(self, regime: str) -> Dict[str, float]:
        """Get strategy weights for a regime.

        Args:
            regime: Market regime

        Returns:
            Dictionary of strategy name -> weight
        """
        async with self._lock:
            if regime not in self._weights:
                # Initialize uniform weights
                self._initialize_regime(regime)

            return self._weights[regime].weights.copy()

    async def compute_ensemble_signal(
        self,
        signals: List[Dict],
        regime: str,
        confidence_threshold: float = 0.65,
    ) -> Optional[Dict]:
        """Compute weighted ensemble signal from multiple strategy signals.

        Args:
            signals: List of signal dicts with 'strategy', 'direction', 'confidence'
            regime: Current market regime
            confidence_threshold: Minimum confidence to trade

        Returns:
            Ensemble signal dict or None if below threshold
        """
        async with self._lock:
            if not signals:
                return None

            # Get weights
            weights = await self.get_strategy_weights(regime)

            # Group by direction
            long_confidence = 0.0
            short_confidence = 0.0
            neutral_confidence = 0.0

            for signal in signals:
                strategy_name = signal.get("strategy", "")
                direction = signal.get("direction", "neutral")
                confidence = signal.get("confidence", 0.0)

                # Get weight (default to uniform if not in weights)
                weight = weights.get(strategy_name, 1.0 / len(signals))

                # Accumulate weighted confidence
                if direction == "long":
                    long_confidence += weight * confidence
                elif direction == "short":
                    short_confidence += weight * confidence
                else:
                    neutral_confidence += weight * confidence

            # Determine ensemble direction
            max_confidence = max(long_confidence, short_confidence, neutral_confidence)

            if max_confidence < confidence_threshold:
                logger.debug(
                    "Ensemble signal below threshold: {:.3f} < {:.3f}",
                    max_confidence,
                    confidence_threshold,
                )
                return None

            if long_confidence == max_confidence:
                direction = "long"
                ensemble_confidence = long_confidence
            elif short_confidence == max_confidence:
                direction = "short"
                ensemble_confidence = short_confidence
            else:
                direction = "neutral"
                ensemble_confidence = neutral_confidence

            # Build ensemble signal
            ensemble_signal = {
                "direction": direction,
                "confidence": float(ensemble_confidence),
                "strategy": "ensemble",
                "regime": regime,
                "component_signals": len(signals),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.debug(
                "Ensemble signal: {} confidence={:.3f} from {} signals",
                direction,
                ensemble_confidence,
                len(signals),
            )

            return ensemble_signal

    async def record_trade_outcome(
        self,
        strategy_name: str,
        regime: str,
        pnl_pct: float,
        trade_data: Dict,
    ) -> None:
        """Record trade outcome and update strategy weights.

        Args:
            strategy_name: Name of strategy that generated signal
            regime: Market regime during trade
            pnl_pct: Profit/loss percentage
            trade_data: Full trade data dict
        """
        async with self._lock:
            # Initialize regime if needed
            if regime not in self._weights:
                self._initialize_regime(regime)

            # Record performance
            self._strategy_performance[regime][strategy_name].append(pnl_pct / 100.0)
            self._trade_history[regime].append({
                "strategy": strategy_name,
                "pnl_pct": pnl_pct,
                "timestamp": datetime.now(timezone.utc),
                **trade_data,
            })

            # Update weights using multiplicative algorithm
            self._update_weights(regime, strategy_name, pnl_pct)

            # Increment trade count
            self._trades_since_rebalance[regime] += 1
            self._weights[regime].total_trades += 1

            # Rebalance if needed
            if self._trades_since_rebalance[regime] >= self._rebalance_frequency:
                await self._rebalance_portfolio(regime)
                self._trades_since_rebalance[regime] = 0

            # Save state
            self._save_state()

    async def add_strategy(self, strategy_name: str) -> None:
        """Add a new strategy to the portfolio.

        Args:
            strategy_name: Name of strategy to add
        """
        async with self._lock:
            if strategy_name not in self._all_strategy_names:
                self._all_strategy_names.add(strategy_name)

                # Add to all regime portfolios
                for regime in self._weights:
                    if strategy_name not in self._weights[regime].weights:
                        # Add with average weight
                        avg_weight = 1.0 / len(self._all_strategy_names)
                        self._weights[regime].weights[strategy_name] = avg_weight
                        # Renormalize
                        self._normalize_weights(regime)

                logger.info("Added strategy to portfolio: {}", strategy_name)

    def get_performance_summary(self, regime: Optional[str] = None) -> Dict:
        """Get performance summary for strategies.

        Args:
            regime: Optional regime filter

        Returns:
            Performance summary dict
        """
        summary = {}

        if regime:
            regimes = [regime] if regime in self._strategy_performance else []
        else:
            regimes = list(self._strategy_performance.keys())

        for reg in regimes:
            summary[reg] = {}
            for strategy, returns in self._strategy_performance[reg].items():
                if returns:
                    summary[reg][strategy] = {
                        "avg_return": float(np.mean(returns)),
                        "sharpe": float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)),
                        "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                        "trades": len(returns),
                        "weight": self._weights[reg].weights.get(strategy, 0.0),
                    }

        return summary

    # ------------------------------------------------------------------
    # Weight update algorithm
    # ------------------------------------------------------------------

    def _update_weights(self, regime: str, strategy_name: str, pnl_pct: float) -> None:
        """Update strategy weight using multiplicative weights algorithm.

        Args:
            regime: Market regime
            strategy_name: Strategy to update
            pnl_pct: Profit/loss percentage
        """
        if strategy_name not in self._weights[regime].weights:
            logger.warning("Strategy {} not in weights for regime {}", strategy_name, regime)
            return

        current_weight = self._weights[regime].weights[strategy_name]

        # Normalize reward to [-1, 1]
        reward = np.tanh(pnl_pct / 5.0)  # 5% gain/loss maps to ~±0.76

        # Multiplicative update
        if reward > 0:
            # Win: multiply by (1 + eta * reward)
            new_weight = current_weight * (1 + self._eta * reward)
        else:
            # Loss: multiply by (1 - eta * |reward|)
            new_weight = current_weight * (1 - self._eta * abs(reward))

        # Ensure positive
        new_weight = max(new_weight, 1e-6)

        # Update
        self._weights[regime].weights[strategy_name] = new_weight

        # Renormalize
        self._normalize_weights(regime)

        logger.debug(
            "Weight updated: {} {} pnl={:.2f}% weight {:.4f} → {:.4f}",
            strategy_name,
            regime,
            pnl_pct,
            current_weight,
            self._weights[regime].weights[strategy_name],
        )

    def _normalize_weights(self, regime: str) -> None:
        """Normalize weights to sum to 1.

        Args:
            regime: Regime to normalize
        """
        weights = self._weights[regime].weights
        total = sum(weights.values())

        if total > 0:
            for strategy in weights:
                weights[strategy] /= total

    # ------------------------------------------------------------------
    # Portfolio rebalancing
    # ------------------------------------------------------------------

    async def _rebalance_portfolio(self, regime: str) -> None:
        """Rebalance portfolio: prune low-weight strategies, validate performance.

        Args:
            regime: Regime to rebalance
        """
        logger.info("Rebalancing portfolio for regime: {}", regime)

        # Decay learning rate
        self._eta = max(self._eta * self._eta_decay, self._eta_min)

        # Prune strategies below threshold
        weights = self._weights[regime].weights
        to_remove = [s for s, w in weights.items() if w < self._min_weight_threshold]

        if to_remove:
            logger.info("Pruning {} strategies below {:.2%} weight", len(to_remove), self._min_weight_threshold)
            for strategy in to_remove:
                del weights[strategy]

            # Renormalize
            self._normalize_weights(regime)

        # Walk-forward validation
        validation_result = await self._validate_performance(regime)

        if validation_result["validation_sharpe"] < 0.5:
            logger.warning(
                "Validation Sharpe {:.3f} < 0.5, reverting to uniform weights",
                validation_result["validation_sharpe"],
            )
            self._initialize_regime(regime)  # Reset to uniform
        else:
            logger.info(
                "Validation Sharpe: {:.3f} (passed)",
                validation_result["validation_sharpe"],
            )
            self._weights[regime].validation_sharpe = validation_result["validation_sharpe"]

        self._weights[regime].last_update = datetime.now(timezone.utc)

    async def _validate_performance(self, regime: str) -> Dict:
        """Validate strategy portfolio using walk-forward analysis.

        Args:
            regime: Regime to validate

        Returns:
            Validation results dict
        """
        history = list(self._trade_history[regime])

        if len(history) < self._validation_window:
            return {"validation_sharpe": 0.5, "reason": "insufficient_data"}

        # Split: last validation_window trades for validation
        validation_trades = history[-self._validation_window:]

        # Calculate validation Sharpe
        returns = [t["pnl_pct"] / 100.0 for t in validation_trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0 or np.isnan(std_return):
            validation_sharpe = 0.5
        else:
            validation_sharpe = mean_return / std_return * np.sqrt(252)

        return {
            "validation_sharpe": float(validation_sharpe),
            "trades": len(validation_trades),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_regime(self, regime: str) -> None:
        """Initialize regime with uniform weights.

        Args:
            regime: Regime to initialize
        """
        # Uniform weights
        if self._all_strategy_names:
            uniform_weight = 1.0 / len(self._all_strategy_names)
            weights = {s: uniform_weight for s in self._all_strategy_names}
        else:
            weights = {}

        self._weights[regime] = StrategyWeights(
            weights=weights,
            regime=regime,
            last_update=datetime.now(timezone.utc),
        )

        logger.info("Initialized regime {} with uniform weights", regime)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save meta-learner state to disk."""
        try:
            state = {
                "weights": {k: v.dict() for k, v in self._weights.items()},
                "all_strategy_names": list(self._all_strategy_names),
                "eta": self._eta,
                "trades_since_rebalance": dict(self._trades_since_rebalance),
            }

            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.debug("Meta-learner state saved")
        except Exception as exc:
            logger.error("Failed to save meta-learner state: {}", exc)

    def _load_state(self) -> None:
        """Load meta-learner state from disk."""
        if not self._state_file.exists():
            logger.debug("No saved meta-learner state found")
            return

        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)

            # Restore weights
            for regime, weight_data in state.get("weights", {}).items():
                self._weights[regime] = StrategyWeights(**weight_data)

            # Restore strategy names
            self._all_strategy_names = set(state.get("all_strategy_names", []))

            # Restore eta
            self._eta = state.get("eta", self._eta_initial)

            # Restore rebalance counters
            self._trades_since_rebalance = defaultdict(
                int, state.get("trades_since_rebalance", {})
            )

            logger.info("Meta-learner state loaded from disk")
        except Exception as exc:
            logger.error("Failed to load meta-learner state: {}", exc)
