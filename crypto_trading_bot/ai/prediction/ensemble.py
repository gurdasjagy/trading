"""Ensemble prediction engine: combines multiple models with dynamic weight adjustment."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from ai.prediction.lstm_model import LSTMPredictor
from ai.prediction.price_predictor import Prediction
from ai.prediction.transformer_model import TransformerPredictor


class EnsemblePredictionEngine:
    """Combines predictions from multiple models using weighted averaging.

    Weights are adjusted dynamically based on each model's recent prediction
    accuracy so that well-performing models receive higher influence over time.
    """

    def __init__(self, models: Optional[List[Any]] = None) -> None:
        """
        Args:
            models: Optional list of model instances.  Each must implement
                    a ``predict(sequence) -> float`` method and have a
                    ``__class__.__name__`` for identification.
        """
        self._models: List[Any] = models or []
        self._weights: List[float] = [1.0] * len(self._models)
        # Track recent accuracy per model: list of abs errors
        self._error_history: List[List[float]] = [[] for _ in self._models]
        self._max_history: int = 20

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_model(self, model: Any, weight: float = 1.0) -> None:
        """Add *model* to the ensemble with an initial *weight*.

        Args:
            model:  Model instance with a ``predict(sequence) -> float`` method.
            weight: Initial weight (positive float).
        """
        self._models.append(model)
        self._weights.append(max(0.0, weight))
        self._error_history.append([])
        logger.debug(f"EnsemblePredictionEngine: added model {model.__class__.__name__}")

    async def predict(
        self,
        symbol: str,
        timeframe: str = "1h",
        sequence: Optional[List[float]] = None,
        current_price: float = 0.0,
    ) -> Prediction:
        """Predict the next price by combining all registered models.

        Args:
            symbol:        Trading pair (e.g. ``"BTC/USDT"``).
            timeframe:     Prediction horizon string (informational).
            sequence:      Recent price sequence (oldest first).
            current_price: Current close price used as fallback.

        Returns:
            :class:`~ai.prediction.price_predictor.Prediction`.
        """
        if not self._models or not sequence:
            logger.debug("EnsemblePredictionEngine: no models or sequence — returning neutral")
            return _neutral_prediction(symbol, timeframe, current_price)

        predictions: List[float] = []
        valid_weights: List[float] = []

        for i, model in enumerate(self._models):
            try:
                pred = model.predict(sequence)
                predictions.append(float(pred))
                valid_weights.append(self._weights[i])
            except Exception as exc:
                logger.warning(
                    f"EnsemblePredictionEngine: model {model.__class__.__name__} "
                    f"predict error: {exc}"
                )

        if not predictions:
            return _neutral_prediction(symbol, timeframe, current_price)

        combined = self._combine_predictions_raw(predictions, valid_weights)
        return self._build_prediction(symbol, timeframe, current_price, combined, predictions)

    def update_weights(self, actual_price: Optional[float] = None) -> None:
        """Recompute model weights based on recent prediction errors.

        Should be called after the actual price is observed so that each
        model's error history can be updated before the next weight update.

        Args:
            actual_price: The observed price that predictions were targeting.
                          When ``None`` the method recomputes weights from the
                          existing error history without recording a new error.
        """
        if not self._models:
            return
        try:
            # Convert error history to weight adjustments (lower error → higher weight)
            new_weights: List[float] = []
            for errors in self._error_history:
                if not errors:
                    new_weights.append(1.0)
                else:
                    mean_err = sum(errors) / len(errors)
                    # Invert the mean error to get weight; add small constant for stability
                    new_weights.append(1.0 / (mean_err + 1e-6))

            # Normalise to sum to len(models) so magnitudes stay comparable
            total = sum(new_weights) or 1.0
            n = len(self._models)
            self._weights = [w / total * n for w in new_weights]
            logger.debug(
                f"EnsemblePredictionEngine: weights updated — {[round(w, 3) for w in self._weights]}"
            )
        except Exception as exc:
            logger.warning(f"EnsemblePredictionEngine.update_weights error: {exc}")

    def record_accuracy(self, model_index: int, predicted: float, actual: float) -> None:
        """Record a prediction error for *model_index* and maintain the rolling window.

        Args:
            model_index: Index of the model in :attr:`_models`.
            predicted:   The model's predicted price.
            actual:      The observed price.
        """
        if model_index >= len(self._error_history):
            return
        err = abs(predicted - actual) / max(abs(actual), 1e-8)
        self._error_history[model_index].append(err)
        if len(self._error_history[model_index]) > self._max_history:
            self._error_history[model_index] = self._error_history[model_index][
                -self._max_history :
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_predictions_raw(predictions: List[float], weights: List[float]) -> float:
        """Return the weighted average of *predictions*."""
        total_w = sum(weights) or 1.0
        return sum(p * w for p, w in zip(predictions, weights)) / total_w

    def _combine_predictions(self, predictions_with_meta: List[Prediction]) -> Prediction:
        """Combine a list of :class:`Prediction` objects (for external callers).

        Args:
            predictions_with_meta: List of :class:`Prediction` objects.

        Returns:
            A merged :class:`Prediction` representing the ensemble opinion.
        """
        if not predictions_with_meta:
            return _neutral_prediction("UNKNOWN", "1h", 0.0)

        symbol = predictions_with_meta[0].symbol
        timeframe = predictions_with_meta[0].timeframe
        current = predictions_with_meta[0].predicted_price

        prices = [p.predicted_price for p in predictions_with_meta]
        confs = [p.confidence for p in predictions_with_meta]
        combined_price = sum(p * c for p, c in zip(prices, confs)) / (sum(confs) or 1.0)

        return self._build_prediction(symbol, timeframe, current, combined_price, prices)

    @staticmethod
    def _build_prediction(
        symbol: str,
        timeframe: str,
        current_price: float,
        combined_price: float,
        all_predictions: List[float],
    ) -> Prediction:
        """Build a :class:`Prediction` from a combined price estimate."""
        if current_price > 0:
            change_pct = (combined_price - current_price) / current_price * 100
        else:
            change_pct = 0.0

        direction = "up" if change_pct > 0.05 else "down" if change_pct < -0.05 else "sideways"

        # Confidence is higher when models agree
        if len(all_predictions) > 1:
            spread = max(all_predictions) - min(all_predictions)
            agreement = 1.0 - min(1.0, spread / max(abs(combined_price), 1e-8) * 10)
        else:
            agreement = 0.5

        confidence = min(1.0, max(0.1, agreement))

        return Prediction(
            symbol=symbol,
            timeframe=timeframe,
            predicted_price=round(combined_price, 8),
            predicted_change_pct=round(change_pct, 4),
            direction=direction,
            confidence=round(confidence, 4),
            signal_breakdown={"ensemble_models": len(all_predictions)},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _neutral_prediction(symbol: str, timeframe: str, current_price: float) -> Prediction:
    """Return a no-signal neutral prediction."""
    return Prediction(
        symbol=symbol,
        timeframe=timeframe,
        predicted_price=current_price,
        predicted_change_pct=0.0,
        direction="sideways",
        confidence=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# EnsemblePredictor — combines LSTM + Transformer with dynamic weight tuning
# ---------------------------------------------------------------------------


class EnsemblePredictor:
    """Ensemble of LSTMPredictor + TransformerPredictor with dynamic weighting.

    Features:
    - Dynamic weight adjustment via EMA of recent hit-rate per model.
    - Disagreement detection: confidence → 0 when models strongly disagree.
    - Platt-scaling calibration layer for better-calibrated probabilities.

    Usage::

        predictor = EnsemblePredictor()
        result = predictor.predict(symbol="BTC/USDT", market_data=features_array)
        # result: {"direction", "confidence", "price_target", "uncertainty", ...}
    """

    # Class-level default constants (used as initial values for instance variables below)
    _DEFAULT_DISAGREEMENT_THRESHOLD: float = 0.3
    _DEFAULT_EMA_ALPHA: float = 0.05

    def __init__(
        self,
        lstm_weights_path: Optional[str] = None,
        transformer_weights_path: Optional[str] = None,
        input_size: int = 120,
    ) -> None:
        self._lstm = LSTMPredictor(input_size=input_size, weights_path=lstm_weights_path)
        self._transformer = TransformerPredictor(input_size=input_size, weights_path=transformer_weights_path)

        # EMA hit-rate: initialised to 0.5 (no prior knowledge)
        self._lstm_hit_rate: float = 0.5
        self._transformer_hit_rate: float = 0.5

        # History for weight calculation (last 100 binary correct/incorrect)
        self._lstm_history: List[int] = []
        self._transformer_history: List[int] = []
        self._max_history: int = 100

        # Tunable thresholds (instance-level so calibration can update them)
        self._disagreement_threshold: float = self._DEFAULT_DISAGREEMENT_THRESHOLD
        self._ema_alpha: float = self._DEFAULT_EMA_ALPHA

        # Platt scaling parameters (instance-level for online calibration)
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0

        logger.debug("EnsemblePredictor: initialised with LSTM + Transformer sub-models")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        symbol: str,  # noqa: ARG002
        market_data: Any,
        current_price: float = 0.0,
    ) -> Dict[str, Any]:
        """Combine LSTM and Transformer predictions into a single signal.

        Args:
            symbol:        Trading pair (informational).
            market_data:   Numpy feature array ``(seq_len, num_features)`` or
                           compatible structure.
            current_price: Current close price for price-target calculation.

        Returns:
            Dict with keys ``direction``, ``confidence``, ``price_target``,
            ``uncertainty``, ``lstm_direction``, ``transformer_direction``,
            ``disagreement``.
        """
        import numpy as np

        try:
            feat = np.array(market_data, dtype=np.float32) if not isinstance(market_data, np.ndarray) else market_data
        except Exception:
            feat = np.zeros((1, 120), dtype=np.float32)

        lstm_result = self._lstm.predict(feat)
        tf_result = self._transformer.predict(feat)

        lstm_dir = lstm_result.get("direction", "sideways")
        tf_dir = tf_result.get("direction", "sideways")
        lstm_conf = float(lstm_result.get("confidence", 0.33))
        tf_conf = float(tf_result.get("confidence", 0.0))
        tf_uncertainty = float(tf_result.get("uncertainty", 0.0))

        # Disagreement: penalise when models pick different directions
        disagreement = 0.0 if lstm_dir == tf_dir else min(1.0, abs(lstm_conf - tf_conf) + 0.3)

        # Weighted combination (by hit-rate)
        w_lstm = self._lstm_hit_rate
        w_tf = self._transformer_hit_rate
        w_total = w_lstm + w_tf or 1.0

        # Direction vote (majority)
        direction_votes: Dict[str, float] = {"up": 0.0, "down": 0.0, "sideways": 0.0}
        direction_votes[lstm_dir] = direction_votes.get(lstm_dir, 0.0) + w_lstm * lstm_conf
        direction_votes[tf_dir] = direction_votes.get(tf_dir, 0.0) + w_tf * tf_conf
        direction = max(direction_votes, key=lambda k: direction_votes[k])

        # Raw combined confidence
        raw_conf = (w_lstm * lstm_conf + w_tf * tf_conf) / w_total

        # Disagreement penalty: collapse to 0 if disagreement > threshold
        if disagreement > self._disagreement_threshold:
            raw_conf = 0.0

        # Platt scaling calibration
        calibrated_conf = self._platt_scale(raw_conf)

        # Price target from LSTM magnitude + transformer median
        lstm_mag = float(lstm_result.get("magnitude", 0.0))
        tf_quantiles = tf_result.get("quantiles", [[0, 0, 0]] * 4)
        tf_median = float(tf_quantiles[2][1]) if tf_quantiles else 0.0
        combined_mag = (w_lstm * lstm_mag + w_tf * tf_median) / w_total
        price_target = current_price * (1.0 + combined_mag) if current_price > 0 else 0.0

        return {
            "direction": direction,
            "confidence": round(calibrated_conf, 4),
            "price_target": round(price_target, 8),
            "uncertainty": round(tf_uncertainty, 6),
            "lstm_direction": lstm_dir,
            "transformer_direction": tf_dir,
            "disagreement": round(disagreement, 4),
            "magnitude": round(combined_mag, 6),
        }

    def record_outcome(self, lstm_correct: bool, transformer_correct: bool) -> None:
        """Update hit-rate EMAs after observing actual market move.

        Args:
            lstm_correct:        Whether LSTM direction prediction was correct.
            transformer_correct: Whether Transformer direction prediction was correct.
        """
        # EMA update
        self._lstm_hit_rate = (
            self._ema_alpha * float(lstm_correct)
            + (1.0 - self._ema_alpha) * self._lstm_hit_rate
        )
        self._transformer_hit_rate = (
            self._ema_alpha * float(transformer_correct)
            + (1.0 - self._ema_alpha) * self._transformer_hit_rate
        )

        # History lists for alternative weight calc
        self._lstm_history.append(1 if lstm_correct else 0)
        self._transformer_history.append(1 if transformer_correct else 0)
        if len(self._lstm_history) > self._max_history:
            self._lstm_history = self._lstm_history[-self._max_history :]
        if len(self._transformer_history) > self._max_history:
            self._transformer_history = self._transformer_history[-self._max_history :]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _platt_scale(self, raw_prob: float) -> float:
        """Apply Platt scaling: sigmoid(A * logit(p) + B)."""
        import math
        p = max(1e-7, min(1.0 - 1e-7, raw_prob))
        logit = math.log(p / (1.0 - p))
        scaled = self._platt_a * logit + self._platt_b
        calibrated = 1.0 / (1.0 + math.exp(-scaled))
        return float(max(0.0, min(1.0, calibrated)))

    def fit_platt_scaling(
        self, raw_probs: List[float], labels: List[int], lr: float = 0.01, epochs: int = 100
    ) -> None:
        """Fit Platt scaling parameters from calibration data.

        Args:
            raw_probs: List of raw model probabilities (0–1).
            labels:    Binary labels (1=correct, 0=incorrect).
            lr:        Learning rate for gradient descent.
            epochs:    Number of optimisation iterations.
        """
        import math

        a, b = self._platt_a, self._platt_b
        for _ in range(epochs):
            grad_a, grad_b = 0.0, 0.0
            for p, y in zip(raw_probs, labels):
                p_c = max(1e-7, min(1 - 1e-7, p))
                logit = math.log(p_c / (1 - p_c))
                pred = 1.0 / (1.0 + math.exp(-(a * logit + b)))
                err = pred - y
                grad_a += err * logit
                grad_b += err
            n = len(raw_probs) or 1
            a -= lr * grad_a / n
            b -= lr * grad_b / n
        self._platt_a = a
        self._platt_b = b
        logger.debug(f"EnsemblePredictor: Platt scaling fitted A={a:.4f} B={b:.4f}")
