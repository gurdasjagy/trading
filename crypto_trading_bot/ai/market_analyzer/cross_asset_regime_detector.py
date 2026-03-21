"""Cross-asset regime detection using multi-market correlation and HMM."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.debug(
        "hmmlearn not installed — HMM regime detection will be disabled. "
        "Install hmmlearn>=0.3.0 to enable this feature."
    )


class CrossAssetRegime(str, Enum):
    """Cross-asset market regimes."""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    DOLLAR_STRENGTH = "DOLLAR_STRENGTH"
    DOLLAR_WEAKNESS = "DOLLAR_WEAKNESS"
    INFLATION_FEAR = "INFLATION_FEAR"
    DEFLATION_FEAR = "DEFLATION_FEAR"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"


class CrossAssetRegimeDetector:
    """Detect market regimes using cross-asset analysis.

    Analyzes correlations and price movements across:
    - Crypto (BTC, ETH)
    - US Dollar Index (DXY)
    - US 10Y Treasury Yield
    - VIX (Volatility Index)
    - Gold (XAU/USD)
    - S&P 500
    - Oil (WTI)

    Uses PCA for dimensionality reduction and HMM for regime classification.
    """

    def __init__(
        self,
        correlation_window: int = 30,
        pca_components: int = 3,
        hmm_states: int = 5,
        cache_ttl: float = 900.0,  # 15 minutes
    ) -> None:
        """Initialize cross-asset regime detector.

        Args:
            correlation_window: Days for rolling correlation calculation.
            pca_components: Number of PCA components to extract.
            hmm_states: Number of hidden states in HMM.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.correlation_window = correlation_window
        self.pca_components = pca_components
        self.hmm_states = hmm_states
        self.cache_ttl = cache_ttl

        # Asset data cache: {symbol: (timestamp, DataFrame)}
        self._asset_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

        # Current regime
        self._current_regime: CrossAssetRegime = CrossAssetRegime.UNKNOWN
        self._regime_confidence: float = 0.0
        self._last_detection_time: float = 0.0

        # HMM model (trained incrementally)
        self._hmm_model: Optional[hmm.GaussianHMM] = None
        if HMM_AVAILABLE:
            self._hmm_model = hmm.GaussianHMM(
                n_components=hmm_states,
                covariance_type="diag",
                n_iter=100,
            )

        # PCA for dimensionality reduction
        self._pca: PCA = PCA(n_components=pca_components)
        self._scaler: StandardScaler = StandardScaler()

        # Asset symbols (mock data sources for now)
        self._asset_symbols = {
            "crypto": ["BTC/USDT", "ETH/USDT"],
            "dxy": "DXY",  # US Dollar Index
            "treasury_10y": "US10Y",
            "vix": "VIX",
            "gold": "XAU/USD",
            "sp500": "SPX",
            "oil": "WTI",
        }

        logger.info(f"CrossAssetRegimeDetector initialized (HMM={'enabled' if HMM_AVAILABLE else 'disabled'})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_regime(
        self,
        exchange: Optional[Any] = None,
    ) -> Tuple[CrossAssetRegime, float, Optional[np.ndarray]]:
        """Detect current cross-asset regime.

        Args:
            exchange: Exchange client for fetching data (optional, uses cache/mock if None).

        Returns:
            Tuple of (regime, confidence, transition_probabilities).
        """
        # Check cache
        now = time.time()
        if now - self._last_detection_time < self.cache_ttl:
            logger.debug(
                f"CrossAssetRegimeDetector: using cached regime {self._current_regime.value}"
            )
            return self._current_regime, self._regime_confidence, None

        try:
            # Fetch cross-asset data
            asset_data = await self._fetch_cross_asset_data(exchange)

            # Compute correlation matrix
            corr_matrix = self._compute_correlation_matrix(asset_data)

            # Detect regime from correlation patterns
            regime, confidence = self._classify_regime(asset_data, corr_matrix)

            # Update HMM if available
            transition_probs = None
            if HMM_AVAILABLE and self._hmm_model is not None:
                transition_probs = await self._update_hmm(asset_data)

            self._current_regime = regime
            self._regime_confidence = confidence
            self._last_detection_time = now

            logger.info(
                f"CrossAssetRegimeDetector: {regime.value} (confidence={confidence:.2f})"
            )

            return regime, confidence, transition_probs

        except Exception as exc:
            logger.error(f"CrossAssetRegimeDetector failed: {exc}")
            return CrossAssetRegime.UNKNOWN, 0.0, None

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_cross_asset_data(
        self, exchange: Optional[Any]
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all cross-asset symbols.

        Args:
            exchange: Exchange client (optional).

        Returns:
            Dict mapping symbol to OHLCV DataFrame.
        """
        data = {}

        # Fetch crypto data from exchange if available
        if exchange is not None:
            for symbol in self._asset_symbols["crypto"]:
                try:
                    # Check cache first
                    if symbol in self._asset_cache:
                        cached_ts, cached_df = self._asset_cache[symbol]
                        if time.time() - cached_ts < self.cache_ttl:
                            data[symbol] = cached_df
                            continue

                    # Fetch fresh data
                    ohlcv = await exchange.get_ohlcv(
                        symbol=symbol,
                        timeframe="1d",
                        limit=self.correlation_window * 2,
                    )
                    if ohlcv is not None and not ohlcv.empty:
                        data[symbol] = ohlcv
                        self._asset_cache[symbol] = (time.time(), ohlcv)
                except Exception as exc:
                    logger.debug(f"Failed to fetch {symbol}: {exc}")

        # Mock data for other assets (in production, fetch from Alpha Vantage, FRED, etc.)
        # For now, we'll generate synthetic correlated data
        data.update(self._generate_mock_asset_data())

        return data

    def _generate_mock_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Generate mock data for non-crypto assets.

        In production, replace with real API calls to:
        - DXY: Alpha Vantage, TradingView
        - US10Y: FRED API, Alpha Vantage
        - VIX: Alpha Vantage, Yahoo Finance
        - Gold: exchange data or Alpha Vantage
        - SPX: Alpha Vantage, Yahoo Finance
        - WTI: Alpha Vantage
        """
        n_days = self.correlation_window * 2
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_days, freq='D')

        # Generate synthetic returns with realistic correlations
        np.random.seed(int(time.time()) % 1000)

        # Base market factor
        market_factor = np.random.randn(n_days) * 0.01

        data = {}

        # DXY: negatively correlated with risk assets
        dxy_returns = -0.3 * market_factor + np.random.randn(n_days) * 0.005
        data["DXY"] = self._returns_to_ohlcv(dxy_returns, dates, base=100.0)

        # US10Y: positively correlated with inflation expectations
        us10y_returns = 0.2 * market_factor + np.random.randn(n_days) * 0.01
        data["US10Y"] = self._returns_to_ohlcv(us10y_returns, dates, base=4.0)

        # VIX: negatively correlated with risk-on
        vix_returns = -0.6 * market_factor + np.random.randn(n_days) * 0.02
        data["VIX"] = self._returns_to_ohlcv(vix_returns, dates, base=20.0)

        # Gold: safe haven, correlated with dollar weakness
        gold_returns = -0.4 * dxy_returns + np.random.randn(n_days) * 0.008
        data["XAU/USD"] = self._returns_to_ohlcv(gold_returns, dates, base=2000.0)

        # SPX: risk-on asset
        spx_returns = 0.8 * market_factor + np.random.randn(n_days) * 0.01
        data["SPX"] = self._returns_to_ohlcv(spx_returns, dates, base=4500.0)

        # WTI: commodity, inflation sensitive
        wti_returns = 0.3 * market_factor + np.random.randn(n_days) * 0.015
        data["WTI"] = self._returns_to_ohlcv(wti_returns, dates, base=80.0)

        return data

    @staticmethod
    def _returns_to_ohlcv(returns: np.ndarray, dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
        """Convert returns to OHLCV DataFrame."""
        prices = base * np.exp(np.cumsum(returns))
        # Ensure dates are tz-aware (UTC) to match exchange OHLCV data
        if dates.tz is None:
            dates = dates.tz_localize("UTC")
        df = pd.DataFrame({
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.randint(1000, 10000, size=len(prices)),
        }, index=dates)
        return df

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def _compute_correlation_matrix(
        self, asset_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute rolling correlation matrix across assets.

        Args:
            asset_data: Dict mapping symbol to OHLCV DataFrame.

        Returns:
            Correlation matrix as DataFrame.
        """
        # Extract close prices and compute returns
        returns_dict = {}
        for symbol, df in asset_data.items():
            if len(df) < 2:
                continue
            returns = df["close"].pct_change().dropna()
            # Normalize index to tz-naive to prevent join errors when mixing
            # tz-aware exchange data with mock data
            if hasattr(returns.index, 'tz') and returns.index.tz is not None:
                returns.index = returns.index.tz_localize(None)
            returns_dict[symbol] = returns

        if not returns_dict:
            return pd.DataFrame()

        # Align returns to common index
        returns_df = pd.DataFrame(returns_dict).dropna()

        if len(returns_df) < self.correlation_window:
            # Not enough data
            return returns_df.corr()

        # Rolling correlation (use last N days)
        recent_returns = returns_df.tail(self.correlation_window)
        corr_matrix = recent_returns.corr()

        return corr_matrix

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        asset_data: Dict[str, pd.DataFrame],
        corr_matrix: pd.DataFrame,
    ) -> Tuple[CrossAssetRegime, float]:
        """Classify regime from asset data and correlations.

        Uses heuristic rules based on cross-asset behavior.
        """
        # Extract recent returns
        returns = {}
        for symbol, df in asset_data.items():
            if len(df) < 2:
                continue
            recent_return = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]
            returns[symbol] = recent_return

        if len(returns) < 4:
            return CrossAssetRegime.UNKNOWN, 0.0

        # Define regime conditions
        btc_up = returns.get("BTC/USDT", 0) > 0.02
        stocks_up = returns.get("SPX", 0) > 0.01
        vix_down = returns.get("VIX", 0) < -0.05
        dxy_up = returns.get("DXY", 0) > 0.01
        dxy_down = returns.get("DXY", 0) < -0.01
        gold_up = returns.get("XAU/USD", 0) > 0.01
        gold_down = returns.get("XAU/USD", 0) < -0.01
        bonds_up = returns.get("US10Y", 0) < -0.02  # Yield down = bond price up
        bonds_down = returns.get("US10Y", 0) > 0.02

        # Liquidity crisis: everything down, VIX spike
        vix_spike = returns.get("VIX", 0) > 0.15
        all_down = all(returns.get(s, 0) < -0.03 for s in ["BTC/USDT", "SPX", "XAU/USD"])
        if vix_spike and all_down:
            return CrossAssetRegime.LIQUIDITY_CRISIS, 0.9

        # Risk on: stocks up, VIX down, crypto up, gold down
        if stocks_up and vix_down and btc_up and gold_down:
            return CrossAssetRegime.RISK_ON, 0.85

        # Risk off: stocks down, VIX up, crypto down, gold up
        if not stocks_up and not vix_down and not btc_up and gold_up:
            return CrossAssetRegime.RISK_OFF, 0.85

        # Dollar strength: DXY up, gold down, commodities down
        if dxy_up and gold_down:
            return CrossAssetRegime.DOLLAR_STRENGTH, 0.75

        # Dollar weakness: DXY down, gold up, commodities up
        if dxy_down and gold_up:
            return CrossAssetRegime.DOLLAR_WEAKNESS, 0.75

        # Inflation fear: gold up, bonds down (yields up), commodities up
        if gold_up and bonds_down:
            return CrossAssetRegime.INFLATION_FEAR, 0.70

        # Deflation fear: gold down, bonds up (yields down)
        if gold_down and bonds_up:
            return CrossAssetRegime.DEFLATION_FEAR, 0.70

        return CrossAssetRegime.NORMAL, 0.50

    # ------------------------------------------------------------------
    # HMM training
    # ------------------------------------------------------------------

    async def _update_hmm(self, asset_data: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Update HMM model with new data and return transition probabilities.

        Args:
            asset_data: Dict mapping symbol to OHLCV DataFrame.

        Returns:
            Transition probability matrix if HMM is fitted, else None.
        """
        if not HMM_AVAILABLE or self._hmm_model is None:
            return None

        try:
            # Extract features for HMM
            features = self._extract_hmm_features(asset_data)

            if len(features) < 10:
                return None

            # Fit HMM incrementally (partial_fit if available, else refit)
            self._hmm_model.fit(features)

            # Return transition matrix
            return self._hmm_model.transmat_

        except Exception as exc:
            logger.debug(f"HMM update failed: {exc}")
            return None

    def _extract_hmm_features(self, asset_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Extract feature matrix for HMM training.

        Args:
            asset_data: Dict mapping symbol to OHLCV DataFrame.

        Returns:
            Feature matrix (n_samples, n_features).
        """
        # Compute returns for each asset
        returns_list = []
        for symbol, df in asset_data.items():
            if len(df) < 2:
                continue
            returns = df["close"].pct_change().dropna()
            # Normalize index to tz-naive to prevent alignment errors
            if hasattr(returns.index, 'tz') and returns.index.tz is not None:
                returns.index = returns.index.tz_localize(None)
            returns_list.append(returns)

        if not returns_list:
            return np.array([])

        # Align all returns to common index
        returns_df = pd.DataFrame(
            {f"asset_{i}": ret for i, ret in enumerate(returns_list)}
        ).dropna()

        if returns_df.empty:
            return np.array([])

        # Standardize features
        features = self._scaler.fit_transform(returns_df.values)

        # Apply PCA
        if features.shape[1] >= self.pca_components:
            features = self._pca.fit_transform(features)

        return features

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> CrossAssetRegime:
        """Current detected regime."""
        return self._current_regime

    @property
    def regime_confidence(self) -> float:
        """Confidence score for current regime (0-1)."""
        return self._regime_confidence

    def __repr__(self) -> str:
        return f"CrossAssetRegimeDetector(regime={self._current_regime.value}, conf={self._regime_confidence:.2f})"
