"""Transformer-based time-series price predictor."""

import math
import os
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    logger.debug(
        "PyTorch not available — TransformerPriceModel will use linear regression fallback"
    )


# ---------------------------------------------------------------------------
# PyTorch Transformer model (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _PositionalEncoding(nn.Module):  # type: ignore[misc]
        """Standard sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x):  # type: ignore[override]
            x = x + self.pe[:, : x.size(1), :]
            return self.dropout(x)

    class _TransformerNet(nn.Module):  # type: ignore[misc]
        """Transformer encoder for univariate time-series forecasting."""

        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):  # type: ignore[override]
            # x: (batch, seq_len, 1)
            x = self.input_proj(x)  # (batch, seq_len, d_model)
            x = self.pos_enc(x)
            x = self.encoder(x)  # (batch, seq_len, d_model)
            return self.fc(x[:, -1, :])  # (batch, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TransformerPriceModel:
    """Transformer-based price predictor with a linear-regression fallback.

    Uses a Transformer encoder architecture over a sliding window of price
    observations.  Falls back gracefully to linear regression when PyTorch is
    not installed or training data is insufficient.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self._model = None
        self._is_trained = False
        self._price_min: float = 0.0
        self._price_max: float = 1.0
        self._lr_slope: float = 0.0
        self._lr_intercept: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, price_data: List[float]) -> None:
        """Train the model on *price_data*.

        Args:
            price_data: Ordered list of close prices (oldest first).
        """
        if len(price_data) < self.sequence_length + 1:
            logger.warning(
                f"TransformerPriceModel: insufficient data "
                f"({len(price_data)} < {self.sequence_length + 1})"
            )
            self._fit_linear_fallback(price_data)
            return

        if not _TORCH_AVAILABLE:
            self._fit_linear_fallback(price_data)
            return

        try:
            self._train_transformer(price_data)
        except Exception as exc:
            logger.warning(
                f"TransformerPriceModel.train error: {exc} — falling back to linear regression"
            )
            self._fit_linear_fallback(price_data)

    def predict(self, sequence: List[float]) -> float:
        """Predict the next price given *sequence* of recent prices.

        Args:
            sequence: Most recent close prices (oldest first, length ≥ sequence_length).

        Returns:
            Predicted next-bar price (float).
        """
        if not self._is_trained:
            logger.debug("TransformerPriceModel: not trained — returning last price")
            return float(sequence[-1]) if sequence else 0.0

        if self._model is not None and _TORCH_AVAILABLE:
            return self._predict_transformer(sequence)

        return self._predict_linear(sequence)

    def save_model(self, path: str) -> None:
        """Save model weights to *path*.

        Args:
            path: Destination file path.
        """
        if self._model is None:
            logger.warning("TransformerPriceModel.save_model: no model to save")
            return
        try:
            import torch

            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(self._model.state_dict(), path)
            logger.info(f"TransformerPriceModel saved to {path}")
        except Exception as exc:
            logger.warning(f"TransformerPriceModel.save_model error: {exc}")

    def load_model(self, path: str) -> None:
        """Load model weights from *path*.

        Args:
            path: File path written by :meth:`save_model`.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("TransformerPriceModel.load_model: torch unavailable")
            return
        try:
            import torch

            net = _TransformerNet(
                d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers
            )
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            self._model = net
            self._is_trained = True
            logger.info(f"TransformerPriceModel loaded from {path}")
        except Exception as exc:
            logger.warning(f"TransformerPriceModel.load_model error: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers — Transformer path
    # ------------------------------------------------------------------

    def _train_transformer(self, price_data: List[float]) -> None:
        """Train _TransformerNet on normalised price data."""
        import torch
        import torch.nn as nn

        prices = np.array(price_data, dtype=np.float32)
        self._price_min = float(prices.min())
        self._price_max = float(prices.max())
        price_range = self._price_max - self._price_min or 1.0
        norm = (prices - self._price_min) / price_range

        X, y = [], []
        for i in range(len(norm) - self.sequence_length):
            X.append(norm[i : i + self.sequence_length])
            y.append(norm[i + self.sequence_length])

        X_t = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

        net = _TransformerNet(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)
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
                logger.debug(f"Transformer epoch {epoch + 1}/{self.epochs} loss={loss.item():.6f}")

        net.eval()
        self._model = net
        self._is_trained = True
        logger.info("TransformerPriceModel training complete")

    def _predict_transformer(self, sequence: List[float]) -> float:
        """Run a forward pass through the Transformer."""
        import torch

        prices = np.array(sequence[-self.sequence_length :], dtype=np.float32)
        price_range = self._price_max - self._price_min or 1.0
        norm = (prices - self._price_min) / price_range
        X = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = float(self._model(X).item())
        return float(pred_norm * price_range + self._price_min)

    # ------------------------------------------------------------------
    # Internal helpers — linear regression fallback
    # ------------------------------------------------------------------

    def _fit_linear_fallback(self, price_data: List[float]) -> None:
        if len(price_data) < 2:
            self._is_trained = True
            return
        x = np.arange(len(price_data), dtype=float)
        y = np.array(price_data, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        self._lr_slope = float(coeffs[0])
        self._lr_intercept = float(coeffs[1])
        self._is_trained = True
        logger.debug("TransformerPriceModel: linear regression fallback fitted")

    def _predict_linear(self, sequence: List[float]) -> float:
        if not sequence:
            return 0.0
        n = len(sequence)
        return self._lr_slope * n + self._lr_intercept


# ---------------------------------------------------------------------------
# TFT-style TransformerPredictor — production multi-horizon predictor
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _VariableSelectionNetwork(nn.Module):  # type: ignore[misc]
        """Lightweight variable selection / feature gating network."""

        def __init__(self, input_size: int, num_features: int) -> None:
            super().__init__()
            self.fc = nn.Linear(input_size, num_features)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):  # type: ignore[override]
            # x: (batch, seq, input_size) → weights (batch, seq, num_features)
            weights = self.softmax(self.fc(x))
            return x * weights[..., : x.size(-1)]

    class _TFTNet(nn.Module):  # type: ignore[misc]
        """Simplified Temporal Fusion Transformer architecture.

        Supports multi-horizon quantile regression outputs.
        """

        def __init__(
            self,
            input_size: int = 120,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_horizons: int = 4,  # 5m, 15m, 1h, 4h
            num_quantiles: int = 3,  # 10th, 50th, 90th
        ) -> None:
            super().__init__()
            self.num_horizons = num_horizons
            self.num_quantiles = num_quantiles

            # Variable selection
            self.vsn = _VariableSelectionNetwork(input_size, input_size)

            # Learned positional embedding
            self.pos_embedding = nn.Embedding(1024, d_model)

            # Input projection
            self.input_proj = nn.Linear(input_size, d_model)
            self.dropout = nn.Dropout(dropout)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.layer_norm = nn.LayerNorm(d_model)

            # Quantile regression head: for each horizon output num_quantiles values
            self.quantile_head = nn.Linear(d_model, num_horizons * num_quantiles)

        def forward(self, x):  # type: ignore[override]
            # x: (batch, seq_len, input_size)
            x = self.vsn(x)
            x = self.input_proj(x)  # (batch, seq_len, d_model)

            # Learned positional encoding
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)
            x = self.dropout(x)

            enc = self.encoder(x)
            out = self.layer_norm(enc[:, -1, :])  # (batch, d_model) — last timestep

            quantiles = self.quantile_head(out)  # (batch, num_horizons * num_quantiles)
            return quantiles.view(-1, self.num_horizons, self.num_quantiles)


class TransformerPredictor:
    """Production TFT-style Transformer for multi-horizon quantile prediction.

    Input:  ``(sequence_length, num_features)`` numpy array.
    Output: dict with keys:
            - ``quantiles``: shape ``(num_horizons, 3)`` — [10th, 50th, 90th percentile]
            - ``horizons``: list of horizon labels ["5m","15m","1h","4h"]
            - ``direction``: dominant direction from 1h horizon median
            - ``confidence``: uncertainty-adjusted confidence
            - ``uncertainty``: average 80th-percentile spread across horizons

    Falls back to a simple heuristic when PyTorch is unavailable.
    """

    HORIZONS = ["5m", "15m", "1h", "4h"]

    def __init__(
        self,
        input_size: int = 120,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        weights_path: Optional[str] = None,
    ) -> None:
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_p = dropout

        self._model = None
        self._device = "cpu"

        if _TORCH_AVAILABLE:
            self._model = _TFTNet(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                num_horizons=len(self.HORIZONS),
                num_quantiles=3,
            )
            if weights_path and os.path.isfile(weights_path):
                self._load_weights(weights_path)
            self._model.eval()
            logger.debug(f"TransformerPredictor: TFT model initialised (weights={'loaded' if weights_path else 'random'})")
        else:
            logger.debug("TransformerPredictor: PyTorch not available — using heuristic fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> Dict[str, object]:
        """Generate multi-horizon quantile predictions from *features*.

        Args:
            features: Numpy array of shape ``(sequence_length, num_features)``
                      or ``(num_features,)``.

        Returns:
            Dict with ``quantiles``, ``horizons``, ``direction``,
            ``confidence``, ``uncertainty``.
        """
        if self._model is not None and _TORCH_AVAILABLE:
            return self._predict_torch(features)
        return self._predict_heuristic(features)

    def save_weights(self, path: str) -> None:
        if self._model is None:
            return
        try:
            import torch
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(self._model.state_dict(), path)
            logger.info(f"TransformerPredictor weights saved to {path}")
        except Exception as exc:
            logger.warning(f"TransformerPredictor.save_weights error: {exc}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_weights(self, path: str) -> None:
        import torch
        try:
            state = torch.load(path, map_location=self._device)
            self._model.load_state_dict(state)
            logger.info(f"TransformerPredictor weights loaded from {path}")
        except Exception as exc:
            logger.warning(f"TransformerPredictor._load_weights error: {exc} — using random weights")

    def _predict_torch(self, features: np.ndarray) -> Dict[str, object]:
        import torch

        arr = np.array(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != self.input_size:
            diff = self.input_size - arr.shape[-1]
            if diff > 0:
                arr = np.pad(arr, ((0, 0), (0, diff)))
            else:
                arr = arr[:, : self.input_size]

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
        with torch.no_grad():
            q_out = self._model(x)  # (1, 4, 3)
        quantiles = q_out.squeeze(0).cpu().numpy().tolist()  # (4, 3)

        # Use 1h median (index 2, quantile 1) to determine direction
        median_1h = quantiles[2][1]
        direction = "up" if median_1h > 0.001 else "down" if median_1h < -0.001 else "sideways"

        # Uncertainty: mean of (90th - 10th percentile spread) across horizons
        uncertainty = float(np.mean(
            [q[2] - q[0] for q in quantiles]
        ))

        # Confidence decreases with high uncertainty
        confidence = float(max(0.0, min(1.0, 1.0 - min(1.0, abs(uncertainty) * 5))))

        return {
            "quantiles": quantiles,
            "horizons": self.HORIZONS,
            "direction": direction,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }

    @staticmethod
    def _predict_heuristic(features: np.ndarray) -> Dict[str, object]:
        arr = np.array(features, dtype=np.float32)
        if arr.size == 0:
            neutral = [[0.0, 0.0, 0.0]] * 4
            return {"quantiles": neutral, "horizons": TransformerPredictor.HORIZONS, "direction": "sideways", "confidence": 0.0, "uncertainty": 0.0}
        last = arr[-1] if arr.ndim > 1 else arr
        score = float(np.tanh(np.nanmean(last) * 0.3))
        spread = abs(score) * 0.01
        quantiles = [[score - spread, score, score + spread]] * 4
        direction = "up" if score > 0.001 else "down" if score < -0.001 else "sideways"
        return {
            "quantiles": quantiles,
            "horizons": TransformerPredictor.HORIZONS,
            "direction": direction,
            "confidence": float(abs(score)),
            "uncertainty": spread,
        }
