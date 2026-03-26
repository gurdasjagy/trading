"""FEAT 12: ML Weight Publisher with Gradient-Boosted Model Training.

Trains a gradient-boosted model (LightGBM with XGBoost fallback) on recent
trade outcomes to produce per-symbol weights for the Rust execution engine.

Features used for training:
    - Recent trade outcomes (win/loss, PnL)
    - Current market regime (from /dev/shm/regime_weights)
    - Time of day / day of week effects
    - Correlation with BTC/ETH

Outputs per-symbol weights:
    - momentum_weight
    - mean_reversion_weight
    - volatility_weight
    - confidence_floor
    - max_position_scale

Writes to /dev/shm/ml_weights for Rust consumption using seqlock pattern.
"""

import mmap
import os
import struct
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class StrategyPerformanceTracker:
    """Tracks strategy performance over a rolling 24h window.

    Computes momentum_weight vs mean_reversion_weight based on which
    strategies performed better in recent history.
    """

    def __init__(self, window_hours: int = 24):
        self.window_seconds = window_hours * 3600
        # Per-symbol performance history:
        # {symbol_id: deque[(timestamp, return, strategy_type, features_dict)]}
        self.performance_history: Dict[int, deque] = {}

    def record_trade(
        self,
        symbol_id: int,
        return_pct: float,
        strategy_type: str,
        timestamp: Optional[float] = None,
        features: Optional[Dict[str, float]] = None,
    ):
        """Record a trade result with optional feature context.

        Args:
            symbol_id: Symbol ID (1=BTC, 2=ETH, etc.)
            return_pct: Trade return as percentage (e.g., 0.05 = 5%)
            strategy_type: "momentum" or "mean_reversion"
            timestamp: Unix timestamp (defaults to now)
            features: Optional dict of contextual features at trade time
        """
        if timestamp is None:
            timestamp = time.time()

        if symbol_id not in self.performance_history:
            self.performance_history[symbol_id] = deque(maxlen=2000)

        self.performance_history[symbol_id].append(
            (timestamp, return_pct, strategy_type, features or {})
        )

    def get_weights(self, symbol_id: int) -> Tuple[float, float]:
        """Calculate momentum_weight and mean_reversion_weight for a symbol.

        Returns:
            (momentum_weight, mean_reversion_weight) tuple
        """
        if symbol_id not in self.performance_history:
            return (1.0, 0.0)  # Default to momentum

        history = self.performance_history[symbol_id]
        if not history:
            return (1.0, 0.0)

        # Filter to last 24 hours
        cutoff = time.time() - self.window_seconds
        recent = [(ts, ret, stype, _) for ts, ret, stype, _ in history if ts >= cutoff]

        if not recent:
            return (1.0, 0.0)

        # Calculate average returns by strategy type
        momentum_returns = [ret for _, ret, stype, _ in recent if stype == "momentum"]
        mean_rev_returns = [
            ret for _, ret, stype, _ in recent if stype == "mean_reversion"
        ]

        avg_momentum = (
            sum(momentum_returns) / len(momentum_returns) if momentum_returns else 0.0
        )
        avg_mean_rev = (
            sum(mean_rev_returns) / len(mean_rev_returns) if mean_rev_returns else 0.0
        )

        # Normalize to weights (0.0 to 1.0)
        total = abs(avg_momentum) + abs(avg_mean_rev)
        if total > 0.0:
            momentum_weight = max(0.0, avg_momentum) / total
            mean_rev_weight = max(0.0, avg_mean_rev) / total
        else:
            momentum_weight = 1.0
            mean_rev_weight = 0.0

        return (momentum_weight, mean_rev_weight)

    def get_training_data(
        self, symbol_id: int, min_samples: int = 30
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract training data for the gradient-boosted model.

        Returns (X, y) where:
            X: Feature matrix with columns:
                [hour_sin, hour_cos, dow_sin, dow_cos, is_momentum,
                 regime_state, volatility_level, rolling_win_rate,
                 rolling_avg_return, btc_correlation]
            y: Trade return (target variable)
        """
        if symbol_id not in self.performance_history:
            return None

        history = list(self.performance_history[symbol_id])
        if len(history) < min_samples:
            return None

        X_rows: List[List[float]] = []
        y_values: List[float] = []

        for i, (ts, ret, stype, feats) in enumerate(history):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour = dt.hour + dt.minute / 60.0

            # Time features (cyclical encoding)
            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            dow = dt.weekday()
            dow_sin = np.sin(2 * np.pi * dow / 7.0)
            dow_cos = np.cos(2 * np.pi * dow / 7.0)

            # Strategy type encoding
            is_momentum = 1.0 if stype == "momentum" else 0.0

            # Context features (from features dict or defaults)
            regime_state = feats.get("regime_state", 2.0)
            volatility_level = feats.get("volatility_level", 0.5)
            btc_correlation = feats.get("btc_correlation", 0.5)

            # Rolling statistics (look back at recent trades)
            lookback = history[max(0, i - 20):i]
            if lookback:
                wins = sum(1 for _, r, _, _ in lookback if r > 0)
                rolling_win_rate = wins / len(lookback)
                rolling_avg_return = sum(r for _, r, _, _ in lookback) / len(lookback)
            else:
                rolling_win_rate = 0.5
                rolling_avg_return = 0.0

            X_rows.append([
                float(hour_sin), float(hour_cos),
                float(dow_sin), float(dow_cos),
                is_momentum,
                regime_state, volatility_level,
                rolling_win_rate, rolling_avg_return,
                btc_correlation,
            ])
            y_values.append(ret)

        return np.array(X_rows), np.array(y_values)


# ---------------------------------------------------------------------------
# FEAT 12: Gradient-Boosted ML Model Trainer
# ---------------------------------------------------------------------------


class GradientBoostedWeightTrainer:
    """Trains a gradient-boosted model to optimize per-symbol strategy weights.

    Uses LightGBM (preferred) with XGBoost fallback.  If neither is available,
    falls back to a simple heuristic-based approach.

    The model predicts expected return for each strategy type given current
    market conditions, then converts predictions into weight allocations.
    """

    FEATURE_NAMES = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_momentum",
        "regime_state", "volatility_level", "rolling_win_rate",
        "rolling_avg_return", "btc_correlation",
    ]

    def __init__(
        self,
        retrain_interval_secs: int = 3600,
        min_training_samples: int = 50,
    ):
        self._retrain_interval = retrain_interval_secs
        self._min_samples = min_training_samples
        self._last_train_time: float = 0.0
        self._models: Dict[int, Any] = {}  # Per-symbol models
        self._backend: Optional[str] = None
        self._detect_backend()

    def _detect_backend(self):
        """Detect available ML backend (LightGBM preferred, XGBoost fallback)."""
        try:
            import lightgbm  # noqa: F401
            self._backend = "lightgbm"
            logger.info("FEAT 12: Using LightGBM backend for ML weight training")
            return
        except ImportError:
            pass

        try:
            import xgboost  # noqa: F401
            self._backend = "xgboost"
            logger.info("FEAT 12: Using XGBoost backend for ML weight training")
            return
        except ImportError:
            pass

        logger.warning(
            "FEAT 12: Neither LightGBM nor XGBoost available. "
            "Using heuristic fallback. Install with: pip install lightgbm"
        )
        self._backend = None

    def train(self, symbol_id: int, X: np.ndarray, y: np.ndarray) -> bool:
        """Train a gradient-boosted model for a specific symbol."""
        if len(X) < self._min_samples:
            return False

        if self._backend == "lightgbm":
            return self._train_lightgbm(symbol_id, X, y)
        elif self._backend == "xgboost":
            return self._train_xgboost(symbol_id, X, y)
        return False

    def _train_lightgbm(self, symbol_id: int, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            import lightgbm as lgb

            params = {
                "objective": "regression", "metric": "rmse",
                "num_leaves": 15, "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 5, "verbose": -1, "n_jobs": 1, "seed": 42,
            }
            dataset = lgb.Dataset(X, label=y, feature_name=self.FEATURE_NAMES)
            model = lgb.train(
                params, dataset, num_boost_round=100,
                valid_sets=[dataset],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            self._models[symbol_id] = model
            logger.info("FEAT 12: LightGBM trained for symbol {} ({} samples)", symbol_id, len(X))
            return True
        except Exception as exc:
            logger.error("FEAT 12: LightGBM training failed for symbol {}: {}", symbol_id, exc)
            return False

    def _train_xgboost(self, symbol_id: int, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            import xgboost as xgb

            params = {
                "objective": "reg:squarederror", "max_depth": 4,
                "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.8, "seed": 42, "verbosity": 0,
            }
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.FEATURE_NAMES)
            model = xgb.train(params, dtrain, num_boost_round=100)
            self._models[symbol_id] = model
            logger.info("FEAT 12: XGBoost trained for symbol {} ({} samples)", symbol_id, len(X))
            return True
        except Exception as exc:
            logger.error("FEAT 12: XGBoost training failed for symbol {}: {}", symbol_id, exc)
            return False

    def predict_weights(
        self,
        symbol_id: int,
        current_regime: float = 2.0,
        current_volatility: float = 0.5,
        btc_correlation: float = 0.5,
    ) -> Dict[str, float]:
        """Predict optimal weights for a symbol given current conditions."""
        if symbol_id not in self._models:
            return self._default_weights()

        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
        dow = now.weekday()

        base_features = [
            np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
            0.0,  # is_momentum placeholder
            current_regime, current_volatility,
            0.5, 0.0,  # rolling_win_rate, rolling_avg_return
            btc_correlation,
        ]

        # Predict expected return for momentum
        mom_feats = base_features.copy()
        mom_feats[4] = 1.0
        momentum_pred = self._predict_single(symbol_id, np.array([mom_feats]))

        # Predict expected return for mean-reversion
        mr_feats = base_features.copy()
        mr_feats[4] = 0.0
        meanrev_pred = self._predict_single(symbol_id, np.array([mr_feats]))

        return self._predictions_to_weights(momentum_pred, meanrev_pred, current_volatility)

    def _predict_single(self, symbol_id: int, X: np.ndarray) -> float:
        model = self._models.get(symbol_id)
        if model is None:
            return 0.0
        try:
            if self._backend == "lightgbm":
                return float(model.predict(X)[0])
            elif self._backend == "xgboost":
                import xgboost as xgb
                dmat = xgb.DMatrix(X, feature_names=self.FEATURE_NAMES)
                return float(model.predict(dmat)[0])
        except Exception as exc:
            logger.warning("FEAT 12: Prediction failed for symbol {}: {}", symbol_id, exc)
        return 0.0

    def _predictions_to_weights(
        self, momentum_pred: float, meanrev_pred: float, volatility: float,
    ) -> Dict[str, float]:
        """Convert model predictions to weight allocations."""
        m_exp = np.exp(np.clip(momentum_pred * 10, -5, 5))
        r_exp = np.exp(np.clip(meanrev_pred * 10, -5, 5))
        total = m_exp + r_exp

        momentum_weight = float(m_exp / total)
        meanrev_weight = float(r_exp / total)

        avg_pred = (momentum_pred + meanrev_pred) / 2
        volatility_weight = float(np.clip(1.0 + avg_pred * 5, 0.3, 2.0))
        confidence_floor = float(np.clip(0.3 + volatility * 0.4, 0.1, 0.8))

        pred_strength = abs(momentum_pred) + abs(meanrev_pred)
        max_position_scale = float(
            np.clip(1.0 + pred_strength * 2 - volatility * 0.5, 0.25, 3.0)
        )

        return {
            "momentum_weight": momentum_weight,
            "mean_reversion_weight": meanrev_weight,
            "volatility_weight": volatility_weight,
            "confidence_floor": confidence_floor,
            "max_position_scale": max_position_scale,
        }

    def _default_weights(self) -> Dict[str, float]:
        return {
            "momentum_weight": 0.6, "mean_reversion_weight": 0.4,
            "volatility_weight": 1.0, "confidence_floor": 0.3,
            "max_position_scale": 1.0,
        }

    def should_retrain(self) -> bool:
        return (time.time() - self._last_train_time) >= self._retrain_interval

    def mark_trained(self):
        self._last_train_time = time.time()

class MlWeightPublisher:
    """Publishes calibrated ML weights to shared memory for the Rust engine.

    FEAT 12 Enhancement: Integrates gradient-boosted model training
    (LightGBM/XGBoost) to produce data-driven per-symbol weights based on:
        - Recent trade outcomes
        - Current market regime
        - Time of day / day of week effects
        - Correlation with BTC/ETH

    Uses seqlock pattern for lock-free reads on the Rust side.
    """

    SHM_SIZE = 65536
    MAGIC_BYTES = 0x4D4C5F5747485453
    MAX_SYMBOLS = 1024

    def __init__(
        self,
        shm_path: str = "/dev/shm/ml_weights",
        regime_shm_path: str = "/dev/shm/regime_weights",
        retrain_interval_secs: int = 3600,
    ):
        self.shm_path = shm_path
        self.regime_shm_path = regime_shm_path
        self.model_version = 0
        self.performance_tracker = StrategyPerformanceTracker()

        # FEAT 12: Gradient-boosted model trainer
        self._ml_trainer = GradientBoostedWeightTrainer(
            retrain_interval_secs=retrain_interval_secs,
            min_training_samples=50,
        )

        self._init_shm()
        self._init_regime_shm()

    def _init_shm(self):
        if not os.path.exists(self.shm_path):
            with open(self.shm_path, "wb") as f:
                f.write(b"\x00" * self.SHM_SIZE)

        fd = os.open(self.shm_path, os.O_RDWR)
        self.mmap = mmap.mmap(
            fd, self.SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        os.close(fd)

        # Write magic bytes initially
        struct.pack_into("<Q", self.mmap, 8, self.MAGIC_BYTES)

    def _init_regime_shm(self):
        """Initialize regime weights shared memory for reading."""
        try:
            if os.path.exists(self.regime_shm_path):
                fd = os.open(self.regime_shm_path, os.O_RDONLY)
                self.regime_mmap = mmap.mmap(
                    fd, 65536, mmap.MAP_SHARED, mmap.PROT_READ
                )
                os.close(fd)
            else:
                self.regime_mmap = None
        except Exception as e:
            logger.warning("Failed to init regime SHM: {}", e)
            self.regime_mmap = None

    def _read_regime_state(self) -> Optional[Dict[str, Any]]:
        """Read current regime state from shared memory using seqlock pattern."""
        if self.regime_mmap is None:
            return None

        try:
            max_retries = 10
            for _ in range(max_retries):
                seq_before = struct.unpack_from("<I", self.regime_mmap, 0)[0]
                if seq_before % 2 != 0:
                    continue

                magic = struct.unpack_from("<Q", self.regime_mmap, 4)[0]
                if magic != 0x5245474D5F574754:
                    return None

                volatility_regime = struct.unpack_from("<I", self.regime_mmap, 12)[0]

                seq_after = struct.unpack_from("<I", self.regime_mmap, 0)[0]
                if seq_before == seq_after:
                    regime_map = {0: "Low", 1: "Normal", 2: "High", 3: "Extreme"}
                    return {
                        "volatility_regime": regime_map.get(volatility_regime, "Normal"),
                        "regime_code": volatility_regime,
                    }
            return None
        except Exception as e:
            logger.warning("Failed to read regime state: {}", e)
            return None

    def train_models(self, symbol_ids: Optional[List[int]] = None):
        """FEAT 12: Train gradient-boosted models for all tracked symbols.

        Extracts training data from performance history, trains per-symbol
        models, and logs results.
        """
        if symbol_ids is None:
            symbol_ids = list(self.performance_tracker.performance_history.keys())

        trained_count = 0
        for sym_id in symbol_ids:
            data = self.performance_tracker.get_training_data(sym_id)
            if data is None:
                continue
            X, y = data
            if self._ml_trainer.train(sym_id, X, y):
                trained_count += 1

        if trained_count > 0:
            self._ml_trainer.mark_trained()
            logger.info(
                "FEAT 12: Trained ML models for {}/{} symbols",
                trained_count, len(symbol_ids),
            )

    def publish(self, symbol_weights: Dict[int, Dict[str, float]]):
        """Publish new weights with ML model predictions and regime adjustments.

        FEAT 12 Enhancement: If gradient-boosted models are trained, uses
        their predictions to generate weights. Otherwise falls back to
        performance-tracking + input blending.
        """
        # FEAT 12: Periodically retrain models
        if self._ml_trainer.should_retrain():
            self.train_models()

        # Read current regime for ML predictions
        regime_state = self._read_regime_state()
        current_regime = 2.0
        current_volatility = 0.5
        if regime_state:
            current_regime = float(regime_state.get("regime_code", 2))
            vol_map = {"Low": 0.2, "Normal": 0.4, "High": 0.7, "Extreme": 0.95}
            current_volatility = vol_map.get(
                regime_state.get("volatility_regime", "Normal"), 0.5
            )

        adjusted_weights: Dict[int, Dict[str, float]] = {}
        for symbol_id, weights in symbol_weights.items():
            # FEAT 12: Try ML model predictions first
            if symbol_id in self._ml_trainer._models:
                ml_weights = self._ml_trainer.predict_weights(
                    symbol_id,
                    current_regime=current_regime,
                    current_volatility=current_volatility,
                )
                # Blend ML predictions (60%) with input weights (40%)
                adjusted_weights[symbol_id] = {
                    "momentum_weight": (
                        ml_weights["momentum_weight"] * 0.6
                        + weights.get("momentum_weight", 1.0) * 0.4
                    ),
                    "mean_reversion_weight": (
                        ml_weights["mean_reversion_weight"] * 0.6
                        + weights.get("mean_reversion_weight", 0.0) * 0.4
                    ),
                    "volatility_weight": (
                        ml_weights["volatility_weight"] * 0.6
                        + weights.get("volatility_weight", 1.0) * 0.4
                    ),
                    "confidence_floor": (
                        ml_weights["confidence_floor"] * 0.6
                        + weights.get("confidence_floor", 0.0) * 0.4
                    ),
                    "max_position_scale": (
                        ml_weights["max_position_scale"] * 0.6
                        + weights.get("max_position_scale", 1.0) * 0.4
                    ),
                }
            else:
                # Fallback: performance-based weights blended with input
                perf_momentum, perf_mean_rev = (
                    self.performance_tracker.get_weights(symbol_id)
                )
                momentum_weight = (
                    weights.get("momentum_weight", 1.0) * 0.3 + perf_momentum * 0.7
                )
                mean_reversion_weight = (
                    weights.get("mean_reversion_weight", 0.0) * 0.3
                    + perf_mean_rev * 0.7
                )
                adjusted_weights[symbol_id] = {
                    "momentum_weight": momentum_weight,
                    "mean_reversion_weight": mean_reversion_weight,
                    "volatility_weight": weights.get("volatility_weight", 1.0),
                    "confidence_floor": weights.get("confidence_floor", 0.0),
                    "max_position_scale": weights.get("max_position_scale", 1.0),
                }

        # Apply regime-aware weight adjustments
        if regime_state:
            vol_regime = regime_state.get("volatility_regime", "Normal")
            for symbol_id in adjusted_weights:
                if vol_regime == "High":
                    adjusted_weights[symbol_id]["confidence_floor"] = max(
                        adjusted_weights[symbol_id]["confidence_floor"], 0.6
                    )
                    adjusted_weights[symbol_id]["max_position_scale"] *= 0.5
                elif vol_regime == "Extreme":
                    adjusted_weights[symbol_id]["confidence_floor"] = max(
                        adjusted_weights[symbol_id]["confidence_floor"], 0.8
                    )
                    adjusted_weights[symbol_id]["max_position_scale"] *= 0.25

        self.model_version += 1
        num_symbols = min(len(adjusted_weights), self.MAX_SYMBOLS)

        # Seqlock write start (odd)
        seq = struct.unpack_from("<I", self.mmap, 0)[0]
        struct.pack_into("<I", self.mmap, 0, seq + 1)

        struct.pack_into("<Q", self.mmap, 16, self.model_version)
        struct.pack_into("<I", self.mmap, 24, num_symbols)

        offset = 32
        for symbol_id, w in list(adjusted_weights.items())[: self.MAX_SYMBOLS]:
            struct.pack_into(
                "<HHfffff",
                self.mmap,
                offset,
                symbol_id,
                0,
                w.get("momentum_weight", 1.0),
                w.get("mean_reversion_weight", 0.0),
                w.get("volatility_weight", 1.0),
                w.get("confidence_floor", 0.0),
                w.get("max_position_scale", 1.0),
            )
            offset += 24

        # Seqlock write end (even)
        struct.pack_into("<I", self.mmap, 0, seq + 2)

        logger.debug(
            "FEAT 12: Published ML weights v{} for {} symbols (ML models: {})",
            self.model_version, num_symbols, len(self._ml_trainer._models),
        )


if __name__ == "__main__":
    # Test publisher with ML training
    pub = MlWeightPublisher()

    # Simulate some historical trades for training
    logger.info("Generating synthetic trade data for ML training...")
    for i in range(200):
        ts = time.time() - (200 - i) * 300
        pub.performance_tracker.record_trade(
            symbol_id=1, return_pct=np.random.normal(0.001, 0.005),
            strategy_type="momentum", timestamp=ts,
            features={
                "regime_state": float(np.random.choice([0, 1, 2, 3])),
                "volatility_level": np.random.uniform(0.1, 0.9),
                "btc_correlation": np.random.uniform(0.3, 0.9),
            },
        )
        pub.performance_tracker.record_trade(
            symbol_id=1, return_pct=np.random.normal(0.0005, 0.003),
            strategy_type="mean_reversion", timestamp=ts + 60,
            features={
                "regime_state": float(np.random.choice([0, 1, 2, 3])),
                "volatility_level": np.random.uniform(0.1, 0.9),
                "btc_correlation": np.random.uniform(0.3, 0.9),
            },
        )

    pub.train_models()

    test_weights = {
        1: {
            "momentum_weight": 0.8, "mean_reversion_weight": 0.2,
            "volatility_weight": 1.2, "confidence_floor": 0.4,
            "max_position_scale": 2.0,
        },
        2: {
            "momentum_weight": 0.5, "mean_reversion_weight": 0.6,
            "volatility_weight": 0.8, "confidence_floor": 0.3,
            "max_position_scale": 1.5,
        },
    }

    while True:
        pub.publish(test_weights)
        logger.info("Published model version {}", pub.model_version)
        time.sleep(60)
