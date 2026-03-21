"""LSTM neural network for price sequence prediction."""

import os
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    logger.debug("PyTorch not available — LSTMPriceModel will use linear regression fallback")


# ---------------------------------------------------------------------------
# PyTorch LSTM model (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):  # type: ignore[misc]
        """Stacked LSTM followed by a single linear output layer."""

        def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):  # type: ignore[override]
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class _BiLSTMNet(nn.Module):  # type: ignore[misc]
        """Bidirectional LSTM with multi-head attention and dual output heads.

        Architecture:
            - 2 BiLSTM layers (hidden_size units each, bidirectional → 2*hidden_size output)
            - Multi-head self-attention over the sequence
            - Dense(64) → ReLU
            - Direction head: Dense(3) with Softmax — [up, down, sideways] probabilities
            - Magnitude head: Dense(1) — predicted price-change magnitude
        """

        def __init__(
            self,
            input_size: int = 120,
            hidden_size: int = 128,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            lstm_out = hidden_size * 2  # bidirectional

            self.bilstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            self.attn = nn.MultiheadAttention(
                embed_dim=lstm_out,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(lstm_out)

            self.dense = nn.Sequential(
                nn.Linear(lstm_out, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Direction: up / down / sideways
            self.direction_head = nn.Linear(64, 3)
            # Magnitude: predicted price-change fraction
            self.magnitude_head = nn.Linear(64, 1)

        def forward(self, x):  # type: ignore[override]
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.bilstm(x)  # (batch, seq_len, lstm_out)
            # Self-attention over sequence
            attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
            attn_out = self.layer_norm(lstm_out + attn_out)
            # Use last timestep
            last = attn_out[:, -1, :]  # (batch, lstm_out)
            feat = self.dense(last)  # (batch, 64)

            direction_logits = self.direction_head(feat)  # (batch, 3)
            magnitude = self.magnitude_head(feat)  # (batch, 1)
            return direction_logits, magnitude


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LSTMPriceModel:
    """LSTM-based price predictor with a linear-regression fallback.

    When PyTorch is not installed (or training data is insufficient) the model
    silently falls back to a simple linear-regression trend extrapolation so
    that the rest of the system continues to function.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self._model = None
        self._is_trained = False
        self._price_min: float = 0.0
        self._price_max: float = 1.0
        # Linear regression fallback coefficients
        self._lr_slope: float = 0.0
        self._lr_intercept: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, price_data: List[float]) -> None:
        """Train the LSTM on *price_data*.

        Falls back to fitting a linear regression when torch is unavailable
        or training data is too short.

        Args:
            price_data: Ordered list of close prices (oldest first).
        """
        if len(price_data) < self.sequence_length + 1:
            logger.warning(
                f"LSTMPriceModel: insufficient training data "
                f"({len(price_data)} < {self.sequence_length + 1})"
            )
            self._fit_linear_fallback(price_data)
            return

        if not _TORCH_AVAILABLE:
            self._fit_linear_fallback(price_data)
            return

        try:
            self._train_lstm(price_data)
        except Exception as exc:
            logger.warning(f"LSTMPriceModel.train error: {exc} — falling back to linear regression")
            self._fit_linear_fallback(price_data)

    def predict(self, sequence: List[float]) -> float:
        """Predict the next price given the most recent *sequence* of prices.

        Args:
            sequence: List of the most recent close prices (oldest first).
                      Should have length ≥ :attr:`sequence_length`.

        Returns:
            Predicted next-bar price (float).
        """
        if not self._is_trained:
            logger.debug("LSTMPriceModel: model not trained — returning last price")
            return float(sequence[-1]) if sequence else 0.0

        if self._model is not None and _TORCH_AVAILABLE:
            return self._predict_lstm(sequence)

        return self._predict_linear(sequence)

    def save_model(self, path: str) -> None:
        """Save the trained LSTM weights to *path*.

        Args:
            path: File path (e.g. ``"models/lstm.pt"``).
        """
        if self._model is None:
            logger.warning("LSTMPriceModel.save_model: no LSTM model to save")
            return
        try:
            import torch

            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(self._model.state_dict(), path)
            logger.info(f"LSTMPriceModel saved to {path}")
        except Exception as exc:
            logger.warning(f"LSTMPriceModel.save_model error: {exc}")

    def load_model(self, path: str) -> None:
        """Load LSTM weights from *path*.

        Args:
            path: File path written by :meth:`save_model`.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("LSTMPriceModel.load_model: torch unavailable")
            return
        try:
            import torch

            net = _LSTMNet(hidden_size=self.hidden_size, num_layers=self.num_layers)
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            self._model = net
            self._is_trained = True
            logger.info(f"LSTMPriceModel loaded from {path}")
        except Exception as exc:
            logger.warning(f"LSTMPriceModel.load_model error: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers — LSTM path
    # ------------------------------------------------------------------

    def _train_lstm(self, price_data: List[float]) -> None:
        """Train the _LSTMNet on normalised price data."""
        import torch
        import torch.nn as nn

        prices = np.array(price_data, dtype=np.float32)
        self._price_min = float(prices.min())
        self._price_max = float(prices.max())
        price_range = self._price_max - self._price_min or 1.0
        norm = (prices - self._price_min) / price_range

        # Build sequences
        X, y = [], []
        for i in range(len(norm) - self.sequence_length):
            X.append(norm[i : i + self.sequence_length])
            y.append(norm[i + self.sequence_length])

        X_t = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

        net = _LSTMNet(hidden_size=self.hidden_size, num_layers=self.num_layers)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = net(X_t)
            loss = criterion(output, y_t)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"LSTM epoch {epoch + 1}/{self.epochs} loss={loss.item():.6f}")

        net.eval()
        self._model = net
        self._is_trained = True
        logger.info("LSTMPriceModel training complete")

    def _predict_lstm(self, sequence: List[float]) -> float:
        """Run a forward pass through the LSTM."""
        import torch

        prices = np.array(sequence[-self.sequence_length :], dtype=np.float32)
        price_range = self._price_max - self._price_min or 1.0
        norm = (prices - self._price_min) / price_range
        X = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = float(self._model(X).item())
        pred = pred_norm * price_range + self._price_min
        return float(pred)

    # ------------------------------------------------------------------
    # Internal helpers — linear regression fallback
    # ------------------------------------------------------------------

    def _fit_linear_fallback(self, price_data: List[float]) -> None:
        """Fit a simple OLS linear regression as fallback."""
        if len(price_data) < 2:
            self._is_trained = True
            return
        x = np.arange(len(price_data), dtype=float)
        y = np.array(price_data, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        self._lr_slope = float(coeffs[0])
        self._lr_intercept = float(coeffs[1])
        self._is_trained = True
        logger.debug("LSTMPriceModel: linear regression fallback fitted")

    def _predict_linear(self, sequence: List[float]) -> float:
        """Extrapolate one step using the linear trend."""
        if not sequence:
            return 0.0
        # Use the length of the training data as x reference
        n = len(sequence)
        return self._lr_slope * n + self._lr_intercept


# ---------------------------------------------------------------------------
# LSTMPredictor — production BiLSTM with attention for direction prediction
# ---------------------------------------------------------------------------


class LSTMPredictor:
    """Production-grade BiLSTM predictor for price direction and magnitude.

    Input:  ``(sequence_length, num_features)`` numpy array.
    Output: dict with keys:
            - ``direction_probs``: list of 3 floats [P(up), P(down), P(sideways)]
            - ``magnitude``: predicted price-change fraction (float)
            - ``confidence``: max probability of the direction classes (float)
            - ``direction``: the most probable direction string ("up"/"down"/"sideways")

    When PyTorch is unavailable a simple heuristic fallback is used.
    """

    DIRECTIONS = ["up", "down", "sideways"]

    def __init__(
        self,
        input_size: int = 120,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        weights_path: Optional[str] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self._model = None
        self._device = "cpu"

        if _TORCH_AVAILABLE:
            self._model = _BiLSTMNet(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
            if weights_path and os.path.isfile(weights_path):
                self._load_weights(weights_path)
            self._model.eval()
            logger.debug(f"LSTMPredictor: BiLSTM model initialised (weights={'loaded' if weights_path else 'random'})")
        else:
            logger.debug("LSTMPredictor: PyTorch not available — using heuristic fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> Dict[str, object]:
        """Generate a direction + magnitude prediction from *features*.

        Args:
            features: Numpy array of shape ``(sequence_length, num_features)``
                      or ``(num_features,)`` for a single timestep.

        Returns:
            Dict with ``direction_probs`` (list[float]), ``magnitude`` (float),
            ``confidence`` (float), ``direction`` (str).
        """
        if self._model is not None and _TORCH_AVAILABLE:
            return self._predict_torch(features)
        return self._predict_heuristic(features)

    def save_weights(self, path: str) -> None:
        """Save model weights to *path*."""
        if self._model is None:
            return
        try:
            import torch
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(self._model.state_dict(), path)
            logger.info(f"LSTMPredictor weights saved to {path}")
        except Exception as exc:
            logger.warning(f"LSTMPredictor.save_weights error: {exc}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_weights(self, path: str) -> None:
        import torch
        try:
            state = torch.load(path, map_location=self._device)
            self._model.load_state_dict(state)
            logger.info(f"LSTMPredictor weights loaded from {path}")
        except Exception as exc:
            logger.warning(f"LSTMPredictor._load_weights error: {exc} — using random weights")

    def _predict_torch(self, features: np.ndarray) -> Dict[str, object]:
        import torch
        import torch.nn.functional as F

        arr = np.array(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Pad / truncate feature dimension if needed
        if arr.shape[-1] != self.input_size:
            diff = self.input_size - arr.shape[-1]
            if diff > 0:
                arr = np.pad(arr, ((0, 0), (0, diff)))
            else:
                arr = arr[:, : self.input_size]

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

        with torch.no_grad():
            dir_logits, mag = self._model(x)
            dir_probs = F.softmax(dir_logits, dim=-1).squeeze(0).tolist()
            magnitude = float(mag.squeeze())

        confidence = float(max(dir_probs))
        direction = self.DIRECTIONS[int(np.argmax(dir_probs))]
        return {
            "direction_probs": dir_probs,
            "magnitude": magnitude,
            "confidence": confidence,
            "direction": direction,
        }

    @staticmethod
    def _predict_heuristic(features: np.ndarray) -> Dict[str, object]:
        """Fallback heuristic when PyTorch is unavailable."""
        arr = np.array(features, dtype=np.float32)
        if arr.size == 0:
            return {"direction_probs": [1 / 3, 1 / 3, 1 / 3], "magnitude": 0.0, "confidence": 0.33, "direction": "sideways"}
        # Use mean of last-row features as a weak bullish/bearish signal
        last = arr[-1] if arr.ndim > 1 else arr
        score = float(np.tanh(np.nanmean(last) * 0.5))
        p_up = max(0.0, score)
        p_down = max(0.0, -score)
        p_side = 1.0 - p_up - p_down
        probs = [p_up, p_down, p_side]
        idx = int(np.argmax(probs))
        return {
            "direction_probs": probs,
            "magnitude": score * 0.01,
            "confidence": float(probs[idx]),
            "direction": ["up", "down", "sideways"][idx],
        }
