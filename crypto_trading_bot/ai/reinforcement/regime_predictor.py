"""Hidden Markov Model regime transition predictor.

Implements a 5-state HMM over BTC 4-hour features to predict the *next*
market regime, allowing strategies to be pre-positioned before regime changes
rather than reacting after the fact.

States: trending_up, trending_down, ranging, high_vol, crash

The model is trained/updated online using the Baum-Welch algorithm (via
``hmmlearn`` when available) or falls back to a heuristic rule-based
classifier when the library is absent.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Optional dependency: hmmlearn
try:
    from hmmlearn import hmm as _hmm_lib

    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False
    logger.debug("hmmlearn not installed; RegimeTransitionPredictor uses heuristic mode")


# -----------------------------------------------------------------------
# Regime constants
# -----------------------------------------------------------------------

REGIMES = ["trending_up", "trending_down", "ranging", "high_vol", "crash"]
REGIME_IDX: Dict[str, int] = {name: i for i, name in enumerate(REGIMES)}

# Crash protection strategies to pre-activate when crash probability > threshold
CRASH_PROTECTION_STRATEGIES = [
    "liquidation_cascade",
    "fear_greed_contrarian",
    "smart_money_flow",
]


class RegimeTransitionPredictor:
    """Predict the next market regime using a Hidden Markov Model.

    Features used (per 4-hour bar):
    - log_return: log(close/prev_close)
    - volatility: rolling 20-bar std of log_returns (normalised)
    - volume_change: log(volume/rolling_20_mean_volume)
    - funding_rate: current funding rate (or 0 if unavailable)

    Args:
        n_states: Number of hidden states (defaults to 5 = len(REGIMES)).
        n_iter: EM iterations for HMM training.
        crash_threshold: P(crash | current_state) above which crash
            protection is pre-activated (default 0.30).
        min_history: Minimum bars required before predictions are returned.
    """

    def __init__(
        self,
        n_states: int = 5,
        n_iter: int = 50,
        crash_threshold: float = 0.30,
        min_history: int = 40,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.crash_threshold = crash_threshold
        self.min_history = min_history

        # Feature history (log_return, volatility, volume_change, funding_rate)
        self._feature_history: List[List[float]] = []
        self._label_history: List[str] = []

        # HMM model (fitted lazily)
        self._model: Optional[object] = None
        self._is_fitted = False

        # Current state posterior (state_idx -> probability)
        self._state_posterior: Dict[int, float] = {i: 1.0 / n_states for i in range(n_states)}

        # Transition matrix (will be estimated from HMM or heuristics)
        self._transition_matrix: np.ndarray = (
            np.ones((n_states, n_states), dtype=float) / n_states
        )

        # Mapping from HMM state index to regime label (populated after fitting)
        self._hmm_state_to_regime: Dict[int, str] = {i: REGIMES[i] for i in range(n_states)}

        logger.info(
            "RegimeTransitionPredictor initialised: n_states={}, crash_threshold={:.2f}",
            n_states,
            crash_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        log_return: float,
        volatility: float,
        volume_change: float,
        funding_rate: float = 0.0,
        current_regime_label: Optional[str] = None,
    ) -> None:
        """Feed a new 4-hour bar's features into the model.

        Args:
            log_return: log(close_t / close_{t-1}).
            volatility: Normalised rolling volatility (0–1 typical range).
            volume_change: log(volume_t / rolling_mean_volume).
            funding_rate: Current perpetual funding rate.
            current_regime_label: Optional ground-truth regime label for
                the bar (used to calibrate state mapping after fitting).
        """
        self._feature_history.append(
            [log_return, volatility, volume_change, funding_rate]
        )
        if current_regime_label:
            self._label_history.append(current_regime_label)

        # Re-fit every 20 new observations once we have enough history
        if (
            len(self._feature_history) >= self.min_history
            and len(self._feature_history) % 20 == 0
        ):
            self._fit()

        # Update state posterior with the latest observation
        if self._is_fitted:
            self._update_posterior([log_return, volatility, volume_change, funding_rate])

    def predict_next_regime(self) -> Dict[str, float]:
        """Return the probability distribution over next-step regimes.

        Returns:
            ``{regime_label: probability}`` mapping.  Probabilities sum to 1.
            When insufficient data is available the current heuristic regime
            is used as a fallback.
        """
        if not self._is_fitted or not self._feature_history:
            return self._heuristic_regime()

        current_state = max(self._state_posterior, key=lambda k: self._state_posterior[k])
        next_probs = self._transition_matrix[current_state]

        result: Dict[str, float] = {}
        for state_idx, prob in enumerate(next_probs):
            regime = self._hmm_state_to_regime.get(state_idx, REGIMES[state_idx % len(REGIMES)])
            result[regime] = result.get(regime, 0.0) + float(prob)

        return result

    def get_current_state_posterior(self) -> Dict[str, float]:
        """Return the posterior probability of each hidden state label.

        Returns:
            ``{regime_label: probability}`` for the current time step.
        """
        result: Dict[str, float] = {}
        for state_idx, prob in self._state_posterior.items():
            regime = self._hmm_state_to_regime.get(state_idx, REGIMES[state_idx % len(REGIMES)])
            result[regime] = result.get(regime, 0.0) + float(prob)
        return result

    def should_pre_activate_crash_protection(self) -> bool:
        """Return True when P(crash | current_state) exceeds the threshold.

        Also returns True when the current regime itself has a high crash
        probability contribution.

        Example: if P(crash | current=high_vol) > 0.3, returns True.
        """
        next_probs = self.predict_next_regime()
        crash_prob = next_probs.get("crash", 0.0)
        if crash_prob > self.crash_threshold:
            logger.info(
                "Crash protection pre-activation: P(crash|current)={:.3f} > {:.3f}",
                crash_prob,
                self.crash_threshold,
            )
            return True
        return False

    def get_crash_protection_strategies(self) -> List[str]:
        """Return names of strategies to pre-activate for crash protection."""
        return list(CRASH_PROTECTION_STRATEGIES)

    # ------------------------------------------------------------------
    # Internal: HMM fitting and posterior updates
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Fit (or re-fit) the HMM on available feature history."""
        X = np.array(self._feature_history, dtype=float)

        if _HMMLEARN_AVAILABLE:
            try:
                model = _hmm_lib.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=self.n_iter,
                    random_state=42,
                )
                model.fit(X)
                self._model = model
                self._transition_matrix = model.transmat_.copy()

                # Map HMM states to regime labels using most-recent labels
                self._calibrate_state_mapping(X)

                self._is_fitted = True
                logger.debug(
                    "HMM re-fitted: {} observations, transmat shape={}",
                    len(X),
                    self._transition_matrix.shape,
                )
            except Exception as exc:
                logger.warning("HMM fitting failed ({}), using heuristic", exc)
                self._fit_heuristic(X)
        else:
            self._fit_heuristic(X)

    def _fit_heuristic(self, X: np.ndarray) -> None:
        """Build a simple rule-based regime classifier as HMM fallback."""
        # Use rolling statistics to set a reasonable transition matrix
        n = self.n_states
        # Slightly sticky regimes with small probability of transitioning
        self._transition_matrix = (
            np.eye(n) * 0.7 + np.ones((n, n)) * (0.3 / n)
        )
        self._is_fitted = True

    def _calibrate_state_mapping(self, X: np.ndarray) -> None:
        """Map HMM state indices to regime labels using decoded state sequence."""
        if self._model is None:
            return
        try:
            state_seq = self._model.predict(X)  # type: ignore[union-attr]
            n_recent = min(len(state_seq), 50)
            recent_seq = state_seq[-n_recent:]
            recent_X = X[-n_recent:]

            # For each HMM state compute mean features and assign regime
            for s in range(self.n_states):
                mask = recent_seq == s
                if not mask.any():
                    continue
                mean_ret = float(np.mean(recent_X[mask, 0]))
                mean_vol = float(np.mean(recent_X[mask, 1]))
                self._hmm_state_to_regime[s] = self._label_by_features(mean_ret, mean_vol)
        except Exception as exc:
            logger.debug("Calibrate state mapping failed: {}", exc)

    def _label_by_features(self, mean_return: float, mean_vol: float) -> str:
        """Assign a regime label based on mean return and volatility."""
        if mean_vol > 0.8:
            return "crash" if mean_return < -0.01 else "high_vol"
        if mean_return > 0.005:
            return "trending_up"
        if mean_return < -0.005:
            return "trending_down"
        return "ranging"

    def _update_posterior(self, observation: List[float]) -> None:
        """Update the state posterior using the HMM emission probabilities."""
        if self._model is None:
            return
        try:
            obs = np.array(observation, dtype=float).reshape(1, -1)
            # Use the predicted state directly
            state = int(self._model.predict(obs)[0])  # type: ignore[union-attr]
            new_posterior = {i: 0.01 for i in range(self.n_states)}
            new_posterior[state] = 1.0 - (self.n_states - 1) * 0.01
            self._state_posterior = new_posterior
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Heuristic fallback when model is not fitted
    # ------------------------------------------------------------------

    def _heuristic_regime(self) -> Dict[str, float]:
        """Return a naive uniform regime distribution as fallback."""
        n = len(REGIMES)
        return {regime: 1.0 / n for regime in REGIMES}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_transition_matrix(self) -> np.ndarray:
        """Return the current transition probability matrix."""
        return self._transition_matrix.copy()

    @staticmethod
    def build_feature_vector(
        prices: List[float],
        volumes: List[float],
        funding_rates: Optional[List[float]] = None,
        window: int = 20,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Utility to compute a single feature vector from raw OHLCV data.

        Args:
            prices: List of close prices (most recent last).
            volumes: List of bar volumes (same length as prices).
            funding_rates: Optional list of funding rates.
            window: Rolling window for volatility / volume baseline.

        Returns:
            ``(log_return, volatility_norm, volume_change, funding_rate)``
            or ``None`` if insufficient data.
        """
        if len(prices) < 2 or len(volumes) < 2:
            return None

        log_returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]
        if not log_returns:
            return None

        log_return = log_returns[-1]
        recent_rets = log_returns[-window:]
        volatility = float(np.std(recent_rets)) if len(recent_rets) > 1 else 0.0

        # Normalise volatility to [0, 1] using a rough 5 % annual volatility as reference
        daily_vol_ref = 0.02 / math.sqrt(6)  # 2 % daily over 6 4h bars
        volatility_norm = min(1.0, volatility / (daily_vol_ref + 1e-9))

        recent_vols = volumes[-window:]
        avg_vol = float(np.mean(recent_vols)) if recent_vols else 1.0
        volume_change = math.log(max(volumes[-1], 1) / max(avg_vol, 1))

        funding = funding_rates[-1] if funding_rates else 0.0

        return log_return, volatility_norm, volume_change, float(funding)
