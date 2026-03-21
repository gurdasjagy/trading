"""Walk-forward validator for ML models — prevents overfitting via time-series cross-validation.

This module implements proper walk-forward validation for time-series ML models:
- No data leakage (strict temporal ordering)
- Rolling window training and validation
- Out-of-sample performance metrics (Sharpe, hit rate, profit factor)
- Validation reports with statistical significance tests

Usage:
    validator = WalkForwardValidator(
        training_window_days=180,
        validation_window_days=30,
        step_forward_days=30,
    )
    
    result = await validator.validate(
        model=my_lstm_model,
        features=feature_array,
        targets=target_array,
        timestamps=timestamp_array,
    )
    
    if result.passed:
        print(f"Model validated: Sharpe={result.sharpe:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class ValidationWindow:
    """A single train/validation window."""
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    train_start_ts: datetime
    train_end_ts: datetime
    val_start_ts: datetime
    val_end_ts: datetime


@dataclass
class WindowResult:
    """Results from validating a single window."""
    window: ValidationWindow
    train_loss: float
    val_loss: float
    val_sharpe: float
    val_hit_rate: float
    val_profit_factor: float
    val_predictions: np.ndarray
    val_actuals: np.ndarray


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    passed: bool
    num_windows: int
    avg_val_sharpe: float
    std_val_sharpe: float
    avg_val_hit_rate: float
    avg_val_profit_factor: float
    sharpe_p_value: float  # Statistical significance test
    window_results: List[WindowResult] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)


class WalkForwardValidator:
    """Walk-forward validator for time-series ML models.
    
    Implements proper time-series cross-validation with no data leakage.
    Each window trains on historical data and validates on future data,
    then steps forward to the next window.
    
    Args:
        training_window_days: Size of training window in days.
        validation_window_days: Size of validation window in days.
        step_forward_days: Step size for rolling forward (default: same as validation window).
        min_sharpe: Minimum acceptable Sharpe ratio for validation.
        min_hit_rate: Minimum acceptable hit rate (fraction of correct predictions).
        min_profit_factor: Minimum acceptable profit factor (gross profit / gross loss).
    """

    def __init__(
        self,
        training_window_days: int = 180,
        validation_window_days: int = 30,
        step_forward_days: Optional[int] = None,
        min_sharpe: float = 0.5,
        min_hit_rate: float = 0.52,
        min_profit_factor: float = 1.2,
    ) -> None:
        self.training_window_days = training_window_days
        self.validation_window_days = validation_window_days
        self.step_forward_days = step_forward_days or validation_window_days
        self.min_sharpe = min_sharpe
        self.min_hit_rate = min_hit_rate
        self.min_profit_factor = min_profit_factor

        logger.info(
            "WalkForwardValidator initialized: train={}d val={}d step={}d",
            training_window_days,
            validation_window_days,
            self.step_forward_days,
        )

    async def validate(
        self,
        model: Any,
        features: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
    ) -> WalkForwardResult:
        """Run walk-forward validation on the model.
        
        Args:
            model: ML model with .fit(X, y) and .predict(X) methods.
            features: Feature array of shape (n_samples, n_features).
            targets: Target array of shape (n_samples,).
            timestamps: Timestamp array of shape (n_samples,) as Unix timestamps or datetime objects.
        
        Returns:
            WalkForwardResult with aggregated metrics and per-window results.
        """
        # Convert timestamps to datetime if needed
        if isinstance(timestamps[0], (int, float)):
            timestamps = np.array([datetime.fromtimestamp(ts) for ts in timestamps])

        # Generate windows
        windows = self._generate_windows(timestamps)
        logger.info("Generated {} walk-forward windows", len(windows))

        if len(windows) == 0:
            return WalkForwardResult(
                passed=False,
                num_windows=0,
                avg_val_sharpe=0.0,
                std_val_sharpe=0.0,
                avg_val_hit_rate=0.0,
                avg_val_profit_factor=0.0,
                sharpe_p_value=1.0,
                failure_reasons=["Insufficient data for walk-forward validation"],
            )

        # Validate each window
        window_results: List[WindowResult] = []
        for i, window in enumerate(windows):
            logger.info(
                "Validating window {}/{}: train {} → {} | val {} → {}",
                i + 1,
                len(windows),
                window.train_start_ts.date(),
                window.train_end_ts.date(),
                window.val_start_ts.date(),
                window.val_end_ts.date(),
            )

            result = await self._validate_window(model, features, targets, window)
            window_results.append(result)

        # Aggregate results
        return self._aggregate_results(window_results)

    def _generate_windows(self, timestamps: np.ndarray) -> List[ValidationWindow]:
        """Generate walk-forward windows from timestamps.
        
        Args:
            timestamps: Array of datetime objects.
        
        Returns:
            List of ValidationWindow objects.
        """
        windows: List[ValidationWindow] = []
        start_date = timestamps[0]
        end_date = timestamps[-1]

        current_date = start_date
        while True:
            train_start = current_date
            train_end = train_start + timedelta(days=self.training_window_days)
            val_start = train_end
            val_end = val_start + timedelta(days=self.validation_window_days)

            # Stop if validation window exceeds available data
            if val_end > end_date:
                break

            # Find indices for this window
            train_start_idx = np.searchsorted(timestamps, train_start)
            train_end_idx = np.searchsorted(timestamps, train_end)
            val_start_idx = train_end_idx
            val_end_idx = np.searchsorted(timestamps, val_end)

            # Skip if window is too small
            if train_end_idx - train_start_idx < 100 or val_end_idx - val_start_idx < 20:
                current_date += timedelta(days=self.step_forward_days)
                continue

            windows.append(
                ValidationWindow(
                    train_start_idx=train_start_idx,
                    train_end_idx=train_end_idx,
                    val_start_idx=val_start_idx,
                    val_end_idx=val_end_idx,
                    train_start_ts=timestamps[train_start_idx],
                    train_end_ts=timestamps[train_end_idx - 1],
                    val_start_ts=timestamps[val_start_idx],
                    val_end_ts=timestamps[val_end_idx - 1],
                )
            )

            # Step forward
            current_date += timedelta(days=self.step_forward_days)

        return windows

    async def _validate_window(
        self,
        model: Any,
        features: np.ndarray,
        targets: np.ndarray,
        window: ValidationWindow,
    ) -> WindowResult:
        """Validate a single window.
        
        Args:
            model: ML model.
            features: Full feature array.
            targets: Full target array.
            window: Window to validate.
        
        Returns:
            WindowResult with metrics for this window.
        """
        # Extract train and validation data
        X_train = features[window.train_start_idx : window.train_end_idx]
        y_train = targets[window.train_start_idx : window.train_end_idx]
        X_val = features[window.val_start_idx : window.val_end_idx]
        y_val = targets[window.val_start_idx : window.val_end_idx]

        # Train model on this window
        try:
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            train_loss = 0.0  # Placeholder
        except Exception as exc:
            logger.error("Training failed for window: {}", exc)
            train_loss = float('inf')

        # Validate
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(X_val)
            else:
                predictions = np.zeros_like(y_val)
            val_loss = float(np.mean((predictions - y_val) ** 2))
        except Exception as exc:
            logger.error("Prediction failed for window: {}", exc)
            predictions = np.zeros_like(y_val)
            val_loss = float('inf')

        # Calculate metrics
        sharpe = self._calculate_sharpe(predictions, y_val)
        hit_rate = self._calculate_hit_rate(predictions, y_val)
        profit_factor = self._calculate_profit_factor(predictions, y_val)

        return WindowResult(
            window=window,
            train_loss=train_loss,
            val_loss=val_loss,
            val_sharpe=sharpe,
            val_hit_rate=hit_rate,
            val_profit_factor=profit_factor,
            val_predictions=predictions,
            val_actuals=y_val,
        )

    def _aggregate_results(self, window_results: List[WindowResult]) -> WalkForwardResult:
        """Aggregate results from all windows.
        
        Args:
            window_results: List of WindowResult objects.
        
        Returns:
            WalkForwardResult with aggregated metrics.
        """
        if not window_results:
            return WalkForwardResult(
                passed=False,
                num_windows=0,
                avg_val_sharpe=0.0,
                std_val_sharpe=0.0,
                avg_val_hit_rate=0.0,
                avg_val_profit_factor=0.0,
                sharpe_p_value=1.0,
                failure_reasons=["No validation windows"],
            )

        # Extract metrics
        sharpes = [r.val_sharpe for r in window_results]
        hit_rates = [r.val_hit_rate for r in window_results]
        profit_factors = [r.val_profit_factor for r in window_results]

        avg_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes))
        avg_hit_rate = float(np.mean(hit_rates))
        avg_profit_factor = float(np.mean(profit_factors))

        # Statistical significance test (t-test against zero)
        if len(sharpes) > 1:
            t_stat = avg_sharpe / (std_sharpe / np.sqrt(len(sharpes)) + 1e-10)
            # Approximate p-value using normal distribution
            from scipy import stats
            p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        else:
            p_value = 1.0

        # Check pass/fail criteria
        failure_reasons: List[str] = []
        if avg_sharpe < self.min_sharpe:
            failure_reasons.append(f"Avg Sharpe {avg_sharpe:.3f} < {self.min_sharpe}")
        if avg_hit_rate < self.min_hit_rate:
            failure_reasons.append(f"Avg hit rate {avg_hit_rate:.3f} < {self.min_hit_rate}")
        if avg_profit_factor < self.min_profit_factor:
            failure_reasons.append(f"Avg profit factor {avg_profit_factor:.3f} < {self.min_profit_factor}")
        if p_value > 0.05:
            failure_reasons.append(f"Sharpe not statistically significant (p={p_value:.3f})")

        passed = len(failure_reasons) == 0

        logger.info(
            "Walk-forward validation {}: Sharpe={:.3f}±{:.3f} HitRate={:.3f} PF={:.3f} p={:.3f}",
            "PASSED" if passed else "FAILED",
            avg_sharpe,
            std_sharpe,
            avg_hit_rate,
            avg_profit_factor,
            p_value,
        )

        return WalkForwardResult(
            passed=passed,
            num_windows=len(window_results),
            avg_val_sharpe=avg_sharpe,
            std_val_sharpe=std_sharpe,
            avg_val_hit_rate=avg_hit_rate,
            avg_val_profit_factor=avg_profit_factor,
            sharpe_p_value=p_value,
            window_results=window_results,
            failure_reasons=failure_reasons,
        )

    @staticmethod
    def _calculate_sharpe(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Sharpe ratio from predictions and actuals."""
        if len(predictions) == 0:
            return 0.0
        returns = predictions * np.sign(actuals)  # Simplified: assume predictions are returns
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252))

    @staticmethod
    def _calculate_hit_rate(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate hit rate (fraction of correct direction predictions)."""
        if len(predictions) == 0:
            return 0.0
        correct = np.sum(np.sign(predictions) == np.sign(actuals))
        return float(correct / len(predictions))

    @staticmethod
    def _calculate_profit_factor(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(predictions) == 0:
            return 0.0
        pnl = predictions * actuals  # Simplified PnL
        gross_profit = np.sum(pnl[pnl > 0])
        gross_loss = abs(np.sum(pnl[pnl < 0]))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)
