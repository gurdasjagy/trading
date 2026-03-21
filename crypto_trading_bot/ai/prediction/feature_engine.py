"""Feature engineering pipeline for OHLCV data used by ML price predictors."""

from __future__ import annotations

import math
from typing import Optional  # noqa: F401

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngine:
    """Extracts and normalises a rich feature vector from OHLCV data.

    The output array has shape ``(sequence_length, num_features)`` where
    ``num_features`` is typically 120+.  All features are normalised using a
    rolling z-score (lookback=100) so that the values fed into the neural
    network are approximately standard-normal.

    Usage::

        engine = FeatureEngine(sequence_length=60)
        features = engine.extract_features(ohlcv_df, symbol="BTC/USDT")
        # features.shape == (60, <num_features>)
    """

    def __init__(self, sequence_length: int = 60, zscore_lookback: int = 100) -> None:
        self.sequence_length = sequence_length
        self.zscore_lookback = zscore_lookback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self, ohlcv_df: pd.DataFrame, symbol: str = ""  # noqa: ARG002
    ) -> np.ndarray:
        """Extract and normalise features from *ohlcv_df*.

        Args:
            ohlcv_df: DataFrame with columns ``open``, ``high``, ``low``,
                      ``close``, ``volume`` and a DatetimeIndex.
            symbol:   Symbol string (currently reserved for future use).

        Returns:
            Numpy array of shape ``(sequence_length, num_features)``.
            Returns zeros of the correct shape when data is insufficient.
        """
        if len(ohlcv_df) < self.sequence_length:
            logger.warning(
                f"FeatureEngine: insufficient rows ({len(ohlcv_df)} < {self.sequence_length})"
            )
            return np.zeros((self.sequence_length, self._num_features()), dtype=np.float32)

        df = ohlcv_df.copy()
        df = self._ensure_columns(df)

        # Build all feature columns
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)
        df = self._add_microstructure_features(df)
        df = self._add_temporal_features(df)

        # Select only feature columns (exclude raw OHLCV)
        feature_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]

        feat_df = df[feature_cols].copy()

        # Handle NaN: forward-fill then zero-fill
        feat_df = feat_df.ffill().fillna(0.0)

        # Rolling z-score normalisation
        feat_df = self._rolling_zscore(feat_df)

        # Final NaN cleanup after zscore
        feat_df = feat_df.ffill().fillna(0.0)

        # Take the last sequence_length rows
        arr = feat_df.iloc[-self.sequence_length :].values.astype(np.float32)

        if arr.shape[0] < self.sequence_length:
            pad = np.zeros(
                (self.sequence_length - arr.shape[0], arr.shape[1]), dtype=np.float32
            )
            arr = np.vstack([pad, arr])

        return arr

    # ------------------------------------------------------------------
    # Feature construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required OHLCV columns exist and are float."""
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)
        return df

    @staticmethod
    def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add return and price-level features."""
        c = df["close"]
        df["ret_1"] = c.pct_change(1)
        df["ret_5"] = c.pct_change(5)
        df["ret_15"] = c.pct_change(15)
        df["ret_60"] = c.pct_change(60)
        df["ret_240"] = c.pct_change(240)
        df["log_ret_1"] = np.log(c / c.shift(1))
        df["log_ret_5"] = np.log(c / c.shift(5))

        # ROC momentum at multiple windows
        for w in (5, 10, 20, 50):
            df[f"roc_{w}"] = (c - c.shift(w)) / (c.shift(w) + 1e-10)

        # Price position within recent range
        roll_high = df["high"].rolling(20).max()
        roll_low = df["low"].rolling(20).min()
        df["price_pos_20"] = (c - roll_low) / (roll_high - roll_low + 1e-10)

        roll_high50 = df["high"].rolling(50).max()
        roll_low50 = df["low"].rolling(50).min()
        df["price_pos_50"] = (c - roll_low50) / (roll_high50 - roll_low50 + 1e-10)

        # EMA features
        for span in (9, 21, 50, 100, 200):
            ema = c.ewm(span=span, adjust=False).mean()
            df[f"ema_{span}_dist"] = (c - ema) / (ema + 1e-10)

        # VWAP distance (approximate using OHLC4)
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap_approx"] = (typical * df["volume"]).rolling(20).sum() / (
            df["volume"].rolling(20).sum() + 1e-10
        )
        df["vwap_dist"] = (c - df["vwap_approx"]) / (df["vwap_approx"] + 1e-10)

        return df

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        v = df["volume"]
        c = df["close"]

        df["vol_ratio_20"] = v / (v.rolling(20).mean() + 1e-10)
        df["vol_ratio_5"] = v / (v.rolling(5).mean() + 1e-10)

        # OBV
        direction = np.sign(c.diff())
        obv = (v * direction).cumsum()
        df["obv_norm"] = obv / (obv.abs().rolling(50).mean() + 1e-10)

        # Volume-weighted return
        df["vwret_5"] = (c.pct_change(1) * v).rolling(5).sum() / (v.rolling(5).sum() + 1e-10)

        # Money flow (positive/negative)
        typical = (df["high"] + df["low"] + c) / 3.0
        raw_mf = typical * v
        pos_mf = raw_mf.where(typical > typical.shift(1), 0.0)
        neg_mf = raw_mf.where(typical < typical.shift(1), 0.0)
        df["mf_ratio"] = pos_mf.rolling(14).sum() / (
            neg_mf.rolling(14).sum().abs() + 1e-10
        )

        return df

    @staticmethod
    def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add realised-volatility features."""
        h, lo, c, o = df["high"], df["low"], df["close"], df["open"]

        # Parkinson volatility
        df["vol_park_20"] = np.sqrt(
            (1.0 / (4.0 * math.log(2))) * (np.log(h / lo) ** 2).rolling(20).mean()
        )

        # Garman-Klass volatility
        u = np.log(h / o)
        d = np.log(lo / o)
        c_norm = np.log(c / o)
        gk = 0.5 * u * d - (2 * math.log(2) - 1) * c_norm**2
        df["vol_gk_20"] = np.sqrt(gk.rolling(20).mean().abs())

        # ATR
        prev_c = c.shift(1)
        tr = pd.concat(
            [h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_norm"] = df["atr_14"] / (c + 1e-10)

        # Bollinger band width
        roll_mean = c.rolling(20).mean()
        roll_std = c.rolling(20).std()
        df["bb_width"] = (2 * roll_std) / (roll_mean + 1e-10)
        df["bb_pos"] = (c - (roll_mean - 2 * roll_std)) / (4 * roll_std + 1e-10)

        # Historical volatility (close-to-close)
        log_ret = np.log(c / c.shift(1))
        df["hv_20"] = log_ret.rolling(20).std() * math.sqrt(252)
        df["hv_5"] = log_ret.rolling(5).std() * math.sqrt(252)

        return df

    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum / oscillator features."""
        c = df["close"]

        # RSI(14)
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
        df["rsi_norm"] = (df["rsi_14"] - 50.0) / 50.0

        # RSI(7) for short-term
        avg_gain7 = gain.ewm(alpha=1 / 7, adjust=False).mean()
        avg_loss7 = loss.ewm(alpha=1 / 7, adjust=False).mean()
        rs7 = avg_gain7 / (avg_loss7 + 1e-10)
        df["rsi_7"] = 100.0 - 100.0 / (1.0 + rs7)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd - signal) / (c.abs() + 1e-10)

        # Stochastic %K %D
        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        stoch_k = 100 * (c - low14) / (high14 - low14 + 1e-10)
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_k.rolling(3).mean()

        # Williams %R
        df["wills_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-10)

        # ADX
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        dm_pos = (df["high"] - prev_high).clip(lower=0)
        dm_neg = (prev_low - df["low"]).clip(lower=0)
        dm_pos = dm_pos.where(dm_pos > dm_neg, 0.0)
        dm_neg = dm_neg.where(dm_neg > dm_pos, 0.0)
        tr = pd.concat(
            [df["high"] - df["low"],
             (df["high"] - c.shift(1)).abs(),
             (df["low"] - c.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean()
        di_pos = 100 * dm_pos.rolling(14).mean() / (atr14 + 1e-10)
        di_neg = 100 * dm_neg.rolling(14).mean() / (atr14 + 1e-10)
        dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)
        df["adx_14"] = dx.rolling(14).mean()

        return df

    @staticmethod
    def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add order-book / microstructure proxy features from OHLCV."""
        h, lo, c, o, v = df["high"], df["low"], df["close"], df["open"], df["volume"]

        # Bid-ask spread proxy: (high - low) / close
        df["spread_proxy"] = (h - lo) / (c + 1e-10)

        # Kyle's lambda proxy: abs(price change) / volume
        df["kyles_lambda"] = c.diff().abs() / (v + 1e-10)
        df["kyles_lambda"] = df["kyles_lambda"].rolling(10).mean()

        # Order book imbalance proxy: (close - open) / (high - low)
        df["ob_imbalance"] = (c - o) / (h - lo + 1e-10)

        # Amihud illiquidity: |return| / dollar_volume
        dollar_vol = c * v
        df["amihud"] = c.pct_change().abs() / (dollar_vol + 1e-10)
        df["amihud"] = df["amihud"].rolling(10).mean()

        # Candle body ratio
        df["body_ratio"] = (c - o).abs() / (h - lo + 1e-10)

        # Upper / lower wick ratio
        body_top = pd.concat([c, o], axis=1).max(axis=1)
        body_bot = pd.concat([c, o], axis=1).min(axis=1)
        df["upper_wick"] = (h - body_top) / (h - lo + 1e-10)
        df["lower_wick"] = (body_bot - lo) / (h - lo + 1e-10)

        return df

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time-of-day and day-of-week features."""
        idx = df.index
        if hasattr(idx, "hour"):
            hour = idx.hour
            dow = idx.dayofweek
        else:
            hour = pd.Series(0, index=df.index)
            dow = pd.Series(0, index=df.index)

        df["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * math.pi * dow / 7.0)

        return df

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _rolling_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score normalisation column-wise."""
        roll_mean = df.rolling(self.zscore_lookback, min_periods=10).mean()
        roll_std = df.rolling(self.zscore_lookback, min_periods=10).std()
        normalised = (df - roll_mean) / (roll_std + 1e-8)
        # Clip extreme values
        return normalised.clip(-5.0, 5.0)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _num_features(self) -> int:
        """Return the expected number of features (used for zero-shape fallback).

        This is computed as the sum of all feature columns added by the
        ``_add_*`` methods: 7 price returns + 4 ROC + 2 price-position + 5 EMA
        + 2 VWAP + 5 volume + 8 volatility + 8 momentum + 7 microstructure
        + 4 temporal = 52 total features.
        """
        return 52

    @property
    def num_features(self) -> int:
        """Number of features produced per timestep."""
        return self._num_features()
