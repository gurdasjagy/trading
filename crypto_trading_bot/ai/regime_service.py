"""Async background service that computes macro ``RegimeState`` every N minutes.

This is the **only** component that is allowed to call the LLM, the sentiment
analyzer, or any other slow/external service.  It writes a small JSON snapshot
to ``/dev/shm/regime_state.json`` (or a configurable path) so that the
deterministic hot-path (Rust strategy engine) can read it without any blocking
I/O or LLM latency.

Architecture
------------
::

    Python slow loop (5 min)                     Rust hot path (< 1 µs)
    ─────────────────────────────────            ───────────────────────
    RegimeService.run_forever()
      ├─ LLMClient (market analysis)   ──write──▶  /dev/shm/regime_state.json
      ├─ SentimentAnalyzer (news)                    │
      ├─ MarketRegimeDetector (OHLCV)                │
      ├─ CrossAssetRegimeDetector                  RegimeReader.get_current()
      └─ FearGreedMonitor                           StrategyEngine.evaluate()

Deployment
----------
Run as a separate asyncio task (``asyncio.create_task``) **or** as a
standalone service via Docker Compose (see ``docker-compose.yml``).

Environment variables
---------------------
``REGIME_OUTPUT_PATH``      Path to write regime JSON (default: ``/dev/shm/regime_state.json``).
``REGIME_INTERVAL_SECONDS`` Update cadence in seconds (default: ``300``).
``REGIME_TTL_SECONDS``      How long the Rust engine trusts the state (default: ``600``).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import struct
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# FEAT 11: Hidden Markov Model (HMM) Regime Detector
# ---------------------------------------------------------------------------


class HMMRegimeDetector:
    """Hidden Markov Model-based regime detector with 4 states.

    States:
        0 = Trending Up (momentum bullish)
        1 = Trending Down (momentum bearish)
        2 = Mean-Reverting (range-bound)
        3 = High Volatility (crisis / choppy)

    Uses hmmlearn's GaussianHMM with 2 features:
        - Log returns (captures direction)
        - Realized volatility (captures vol regime)

    The model supports online learning: parameters are updated hourly
    with new market data to adapt to changing conditions.
    """

    STATE_NAMES = ["trending_up", "trending_down", "mean_reverting", "high_volatility"]
    N_STATES = 4

    def __init__(
        self,
        lookback_periods: int = 100,
        vol_window: int = 20,
        online_update_interval_secs: int = 3600,
    ):
        self._lookback = lookback_periods
        self._vol_window = vol_window
        self._online_update_interval = online_update_interval_secs
        self._last_online_update: float = 0.0
        self._model = None
        self._is_fitted = False

        # Rolling data buffer for online learning
        self._price_buffer: List[float] = []
        self._max_buffer = 2000  # Keep last 2000 prices

        # State probabilities (uniform prior)
        self._state_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self._current_state = 2  # Default: mean-reverting

        self._init_model()

    def _init_model(self):
        """Initialize the HMM with sensible priors for financial data."""
        try:
            from hmmlearn.hmm import GaussianHMM

            self._model = GaussianHMM(
                n_components=self.N_STATES,
                covariance_type="full",
                n_iter=50,
                random_state=42,
                init_params="",  # We set initial params manually
            )

            # Initial transition matrix: states tend to persist
            self._model.transmat_ = np.array([
                [0.90, 0.03, 0.05, 0.02],  # Trending up: stays, rarely flips
                [0.03, 0.90, 0.05, 0.02],  # Trending down: stays
                [0.05, 0.05, 0.85, 0.05],  # Mean-reverting: more transitions
                [0.04, 0.04, 0.07, 0.85],  # High vol: somewhat sticky
            ])

            # Initial state distribution
            self._model.startprob_ = np.array([0.25, 0.25, 0.30, 0.20])

            # Initial means: [log_return, volatility]
            # Trending Up: positive returns, moderate vol
            # Trending Down: negative returns, moderate vol
            # Mean-Reverting: near-zero returns, low vol
            # High Volatility: near-zero returns, high vol
            self._model.means_ = np.array([
                [0.002, 0.01],   # Trending Up
                [-0.002, 0.01],  # Trending Down
                [0.0, 0.005],    # Mean-Reverting
                [0.0, 0.03],     # High Volatility
            ])

            # Initial covariances
            self._model.covars_ = np.array([
                [[0.0001, 0.0], [0.0, 0.0001]],  # Trending Up
                [[0.0001, 0.0], [0.0, 0.0001]],  # Trending Down
                [[0.00005, 0.0], [0.0, 0.00005]], # Mean-Reverting
                [[0.0005, 0.0], [0.0, 0.0005]],   # High Volatility
            ])

            logger.info("FEAT 11: HMM Regime Detector initialized with 4 states")
        except ImportError:
            logger.warning(
                "FEAT 11: hmmlearn not installed, HMM regime detection disabled. "
                "Install with: pip install hmmlearn"
            )
            self._model = None

    def _prepare_features(self, prices: List[float]) -> Optional[np.ndarray]:
        """Convert price series to [log_returns, realized_vol] feature matrix."""
        if len(prices) < self._vol_window + 2:
            return None

        prices_arr = np.array(prices)
        # Log returns
        log_returns = np.diff(np.log(prices_arr))

        # Realized volatility (rolling std of log returns)
        vol = np.array([
            np.std(log_returns[max(0, i - self._vol_window):i])
            if i >= self._vol_window else np.std(log_returns[:i + 1])
            for i in range(len(log_returns))
        ])

        # Stack features: [log_return, volatility]
        features = np.column_stack([log_returns, vol])

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def update_prices(self, prices: List[float]) -> None:
        """Add new price data to the buffer for online learning."""
        self._price_buffer.extend(prices)
        if len(self._price_buffer) > self._max_buffer:
            self._price_buffer = self._price_buffer[-self._max_buffer:]

    def fit(self, prices: List[float]) -> bool:
        """Fit the HMM on historical price data.

        Args:
            prices: List of close prices (at least vol_window + 2 elements)

        Returns:
            True if fitting succeeded
        """
        if self._model is None:
            return False

        features = self._prepare_features(prices)
        if features is None or len(features) < 10:
            logger.warning("FEAT 11: Not enough data to fit HMM (need >= {} prices)", self._vol_window + 12)
            return False

        try:
            self._model.fit(features)
            self._is_fitted = True
            self.update_prices(prices)
            logger.info("FEAT 11: HMM fitted on {} data points", len(features))
            return True
        except Exception as exc:
            logger.error("FEAT 11: HMM fitting failed: {}", exc)
            return False

    def predict(self, prices: List[float]) -> tuple[str, np.ndarray]:
        """Predict the current regime from recent prices.

        Args:
            prices: Recent price series

        Returns:
            (regime_name, state_probabilities) tuple
        """
        if self._model is None or not self._is_fitted:
            return self.STATE_NAMES[self._current_state], self._state_probs

        features = self._prepare_features(prices)
        if features is None or len(features) < 2:
            return self.STATE_NAMES[self._current_state], self._state_probs

        try:
            # Get posterior state probabilities
            log_prob, state_seq = self._model.decode(features, algorithm="viterbi")
            posteriors = self._model.predict_proba(features)

            # Use the last time step's state and probabilities
            self._current_state = int(state_seq[-1])
            self._state_probs = posteriors[-1]

            return self.STATE_NAMES[self._current_state], self._state_probs

        except Exception as exc:
            logger.warning("FEAT 11: HMM prediction failed: {}", exc)
            return self.STATE_NAMES[self._current_state], self._state_probs

    def maybe_online_update(self, prices: List[float]) -> bool:
        """Perform online learning if the update interval has elapsed.

        Re-fits the HMM on the accumulated price buffer plus new data.
        Called periodically (default: every hour).

        Returns:
            True if an update was performed
        """
        now = time.time()
        if now - self._last_online_update < self._online_update_interval:
            return False

        self.update_prices(prices)

        if len(self._price_buffer) < self._vol_window + 20:
            return False

        success = self.fit(self._price_buffer)
        if success:
            self._last_online_update = now
            logger.info(
                "FEAT 11: HMM online update completed with {} prices",
                len(self._price_buffer),
            )
        return success

    def write_regime_probabilities_shm(
        self, shm_path: str = "/dev/shm/regime_weights"
    ) -> bool:
        """Write HMM state probabilities to shared memory.

        Appends HMM probabilities to the existing regime_weights SHM file
        at a dedicated offset (starting at byte 112, within the _reserved area).

        Layout at offset 112:
            [112..116]  hmm_state         u32 (0=up, 1=down, 2=mean_rev, 3=high_vol)
            [116..120]  prob_trending_up   f32 (float, fixed-point 1e4)
            [120..124]  prob_trending_down f32
            [124..128]  prob_mean_rev     f32
            (128 is end of the 128-byte regime struct, so we stay within bounds
             using the _reserved area from bytes 88-112, offset 112 is at the edge)

        Actually, the reserved area is [88..112] = 24 bytes. Let's use offset 88.
        """
        try:
            if not os.path.exists(shm_path):
                return False

            fd = os.open(shm_path, os.O_RDWR)
            try:
                mm = mmap.mmap(fd, 128, access=mmap.ACCESS_WRITE)
                try:
                    # Write HMM data into the _reserved area starting at offset 88
                    # [88..92] hmm_state (u32)
                    # [92..96] prob_trending_up (i32, fixed-point 1e4)
                    # [96..100] prob_trending_down (i32, fixed-point 1e4)
                    # [100..104] prob_mean_rev (i32, fixed-point 1e4)
                    # [104..108] prob_high_vol (i32, fixed-point 1e4)
                    struct.pack_into(
                        "<Iiiii",
                        mm,
                        88,
                        self._current_state,
                        int(self._state_probs[0] * 10_000),
                        int(self._state_probs[1] * 10_000),
                        int(self._state_probs[2] * 10_000),
                        int(self._state_probs[3] * 10_000),
                    )
                    mm.flush()
                    return True
                finally:
                    mm.close()
            finally:
                os.close(fd)
        except Exception as exc:
            logger.warning("FEAT 11: Failed to write HMM probabilities to SHM: {}", exc)
            return False

    @property
    def current_state_name(self) -> str:
        """Current regime state name."""
        return self.STATE_NAMES[self._current_state]

    @property
    def state_probabilities(self) -> Dict[str, float]:
        """Current state probabilities as a dict."""
        return {
            name: float(prob)
            for name, prob in zip(self.STATE_NAMES, self._state_probs)
        }


# ---------------------------------------------------------------------------
# RegimeState model (mirrors the Rust regime.rs struct)
# ---------------------------------------------------------------------------


class RegimeState(BaseModel):
    """Macro regime state written by the Python service and read by Rust.

    All fields are intentionally simple scalars or lists of strings so that
    the JSON can be parsed by serde_json without any schema negotiation.
    """

    timestamp_ms: int = 0
    overall_regime: str = "unknown"
    volatility_regime: str = "high"
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    fear_greed_index: int = 50
    btc_dominance_trend: str = "flat"
    funding_rate_bias: str = "neutral"
    cross_asset_correlation: float = 0.0
    news_impact_score: float = 0.0
    recommended_position_scale: float = 0.5
    recommended_strategy_filter: List[str] = []
    blocked_strategies: List[str] = []
    max_leverage_override: Optional[int] = None
    ttl_seconds: int = 600

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def safe_default(cls) -> "RegimeState":
        """Conservative default state used when all data sources fail."""
        return cls(
            overall_regime="unknown",
            volatility_regime="high",
            recommended_position_scale=0.5,
            ttl_seconds=600,
        )


# ---------------------------------------------------------------------------
# RegimeService
# ---------------------------------------------------------------------------


class RegimeService:
    """Background service that periodically computes ``RegimeState``.

    Parameters
    ----------
    llm_client:
        :class:`~ai.llm_client.LLMClient` instance (optional).
    sentiment_analyzer:
        :class:`~ai.sentiment.analyzer.SentimentAnalyzer` (optional).
    regime_detector:
        :class:`~ai.market_analyzer.regime_detector.MarketRegimeDetector` (optional).
    cross_asset_detector:
        :class:`~ai.market_analyzer.cross_asset_regime_detector.CrossAssetRegimeDetector` (optional).
    news_sources:
        List of news-source objects that have an async ``fetch()`` method
        returning ``List[Dict[str, str]]``.  Each dict should have at least
        ``"content"`` and optionally ``"title"`` and ``"source"`` keys.
    output_path:
        Path to write the JSON snapshot.
        Default: env ``REGIME_OUTPUT_PATH`` or ``/dev/shm/regime_state.json``.
    interval_seconds:
        Update cadence.
        Default: env ``REGIME_INTERVAL_SECONDS`` or ``300`` (5 minutes).
    ttl_seconds:
        How long the Rust engine trusts the state before falling back to the
        safe default.
        Default: env ``REGIME_TTL_SECONDS`` or ``600`` (10 minutes).
    """

    _DEFAULT_PATH = "/dev/shm/regime_state.json"
    _DEFAULT_INTERVAL = 300
    _DEFAULT_TTL = 600

    def __init__(
        self,
        llm_client=None,
        sentiment_analyzer=None,
        regime_detector=None,
        cross_asset_detector=None,
        news_sources: Optional[List[Any]] = None,
        output_path: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._llm = llm_client
        self._sentiment = sentiment_analyzer
        self._regime_detector = regime_detector
        self._cross_asset_detector = cross_asset_detector
        self._news_sources = news_sources or []

        self._output_path = output_path or os.environ.get(
            "REGIME_OUTPUT_PATH", self._DEFAULT_PATH
        )
        self._interval = interval_seconds or int(
            os.environ.get("REGIME_INTERVAL_SECONDS", self._DEFAULT_INTERVAL)
        )
        self._ttl = ttl_seconds or int(
            os.environ.get("REGIME_TTL_SECONDS", self._DEFAULT_TTL)
        )

        self._current_state: RegimeState = RegimeState.safe_default()
        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._error_count: int = 0
        
        # ── FEAT 11: HMM Regime Detector ────────────────────────────
        self._hmm_detector = HMMRegimeDetector(
            lookback_periods=100,
            vol_window=20,
            online_update_interval_secs=3600,  # Re-fit every hour
        )
        logger.info("RegimeService: HMMRegimeDetector initialized")

        # ── Task 12: Gamma Exposure Writer initialization ────
        self._gamma_writer: Optional[Any] = None
        self._tasks: Dict[str, asyncio.Task] = {}
        try:
            from ai.market_analyzer.options_gamma_writer import GammaExposureWriter
            self._gamma_writer = GammaExposureWriter()
            logger.info("RegimeService: GammaExposureWriter initialized")
        except Exception as exc:
            logger.debug(f"RegimeService: GammaExposureWriter not available: {exc}")

        # ── Issue 4: SharedRegimeWriter for shared memory persistence ────
        # Writes to /dev/shm/regime_weights alongside the JSON file for
        # gradual migration.  The Rust engine reads from shared memory,
        # while legacy Python code reads the JSON file.
        self._shm_writer: Optional[Any] = None
        try:
            from ai.shared_regime_writer import SharedRegimeWriter

            regime_shm_path = os.environ.get(
                "REGIME_SHM_PATH", "/dev/shm/regime_weights"
            )
            self._shm_writer = SharedRegimeWriter(regime_shm_path)
            logger.info(
                f"RegimeService: SharedRegimeWriter enabled at {regime_shm_path}"
            )
        except Exception as exc:
            logger.debug(f"RegimeService: SharedRegimeWriter not available: {exc}")

        logger.info(
            f"RegimeService initialised — output={self._output_path} "
            f"interval={self._interval}s ttl={self._ttl}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        ohlcv_data: Optional[Any] = None,
        exchange: Optional[Any] = None,
    ) -> RegimeState:
        """Compute and persist one ``RegimeState`` snapshot.

        This is the method called by the trading bot's slow loop.  It can
        also be called directly for testing.

        Args:
            market_data: Dict with at least ``"symbols"`` (list of str) and
                optionally ``"prices"`` (dict), ``"btc_dominance"`` (float),
                ``"funding_rates"`` (dict).
            ohlcv_data:  Pandas DataFrame for single-symbol regime detection.
            exchange:    Exchange client for cross-asset data fetching.

        Returns:
            The computed :class:`RegimeState`.
        """
        try:
            state = await self._compute_regime_state(market_data, ohlcv_data, exchange)
            await self._persist(state)
            self._current_state = state
            self._last_update_ts = time.time()
            self._update_count += 1
            logger.info(
                f"RegimeService: {state.overall_regime} / "
                f"vol={state.volatility_regime} / "
                f"sentiment={state.sentiment_score:.2f} / "
                f"scale={state.recommended_position_scale:.2f}"
            )
            return state
        except Exception as exc:
            self._error_count += 1
            logger.error(f"RegimeService.run_once failed: {exc}", exc_info=True)
            # Keep the previous state if available; otherwise use safe default
            if self._last_update_ts > 0 and (time.time() - self._last_update_ts) < self._ttl:
                return self._current_state
            safe = RegimeState.safe_default()
            safe.timestamp_ms = int(time.time() * 1000)
            safe.ttl_seconds = self._ttl
            await self._persist(safe)
            return safe

    async def run_forever(
        self,
        market_data_provider=None,
        exchange: Optional[Any] = None,
    ) -> None:
        """Background loop that calls :meth:`run_once` every ``interval_seconds``.

        Designed to be run with ``asyncio.create_task(service.run_forever())``.

        Args:
            market_data_provider: Optional callable/object that returns a
                ``Dict[str, Any]`` with the market overview on each iteration.
                If ``None``, an empty market overview is used.
            exchange: Exchange client passed through to the cross-asset detector.
        """
        logger.info(f"RegimeService background loop started (interval={self._interval}s)")
        
        # Task 12: Start gamma exposure writer task
        if self._gamma_writer is not None:
            self._tasks["gamma_writer"] = asyncio.create_task(
                self._gamma_exposure_loop(), 
                name="gamma_writer"
            )
            logger.info("RegimeService: Gamma exposure writer task started")
        
        while True:
            try:
                md: Dict[str, Any] = {}
                if market_data_provider is not None:
                    if asyncio.iscoroutinefunction(market_data_provider):
                        md = await market_data_provider()
                    elif callable(market_data_provider):
                        md = market_data_provider()

                await self.run_once(market_data=md, exchange=exchange)
            except Exception as exc:
                logger.error(f"RegimeService loop iteration failed: {exc}")

            await asyncio.sleep(self._interval)

    @property
    def current_state(self) -> RegimeState:
        """Most recently computed :class:`RegimeState`."""
        return self._current_state

    @property
    def is_stale(self) -> bool:
        """``True`` if the current state is older than ``ttl_seconds``."""
        return (time.time() - self._last_update_ts) > self._ttl
    
    # Task 12: Gamma exposure writer loop
    async def _gamma_exposure_loop(self):
        """Background loop that updates gamma exposure every 5 minutes."""
        if self._gamma_writer is None:
            return
            
        logger.info("Gamma exposure loop started (interval=300s)")
        while True:
            try:
                await self._gamma_writer.update_and_publish()
            except Exception as exc:
                logger.error(f"Gamma exposure update failed: {exc}")
            await asyncio.sleep(300)  # 5 minutes

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    async def _compute_regime_state(
        self,
        market_data: Optional[Dict[str, Any]],
        ohlcv_data: Optional[Any],
        exchange: Optional[Any],
    ) -> RegimeState:
        """Gather all signals and assemble a :class:`RegimeState`."""
        if market_data is None:
            market_data = {}

        state = RegimeState(
            timestamp_ms=int(time.time() * 1000),
            ttl_seconds=self._ttl,
        )

        # ── 1. News sentiment ────────────────────────────────────────────
        news_items = await self._fetch_news()
        sent_score, sent_conf, news_impact = await self._analyse_sentiment(news_items)
        state.sentiment_score = sent_score
        state.sentiment_confidence = sent_conf
        state.news_impact_score = news_impact

        # ── 2. Single-asset regime (ADX / volatility) ────────────────────
        overall_regime, vol_regime = await self._detect_single_asset_regime(ohlcv_data)
        state.overall_regime = overall_regime
        state.volatility_regime = vol_regime

        # ── 3. Cross-asset regime ────────────────────────────────────────
        cross_regime, cross_conf = await self._detect_cross_asset_regime(exchange)
        state.cross_asset_correlation = cross_conf

        # ── 4. Market-specific signals ───────────────────────────────────
        state.btc_dominance_trend = _extract_btc_dominance_trend(market_data)
        state.funding_rate_bias = _extract_funding_rate_bias(market_data)
        state.fear_greed_index = _extract_fear_greed(market_data)

        # ── 5. FEAT 11: HMM regime detection ─────────────────────────
        prices = self._extract_prices_for_hmm(market_data)
        if prices and len(prices) >= 25:
            # Try online update (re-fit every hour)
            self._hmm_detector.maybe_online_update(prices)

            # If not fitted yet, do initial fit
            if not self._hmm_detector._is_fitted:
                self._hmm_detector.fit(prices)

            # Get HMM prediction
            hmm_regime, hmm_probs = self._hmm_detector.predict(prices)

            # Use HMM to refine the data-driven regime when it's uncertain
            hmm_regime_map = {
                "trending_up": "trending_bullish",
                "trending_down": "trending_bearish",
                "mean_reverting": "ranging",
                "high_volatility": "high_volatility",
            }
            mapped_hmm = hmm_regime_map.get(hmm_regime, "unknown")

            # If data-driven regime is unknown or HMM is very confident,
            # let HMM override
            max_hmm_prob = float(max(hmm_probs))
            if state.overall_regime == "unknown" and max_hmm_prob > 0.5:
                state.overall_regime = mapped_hmm
                logger.info(
                    f"FEAT 11: HMM overriding unknown regime → {mapped_hmm} "
                    f"(prob={max_hmm_prob:.3f})"
                )
            elif max_hmm_prob > 0.8 and mapped_hmm != state.overall_regime:
                # HMM very confident and disagrees — log but don't override
                # unless data-driven was ambiguous
                logger.info(
                    f"FEAT 11: HMM suggests {mapped_hmm} (prob={max_hmm_prob:.3f}) "
                    f"vs data-driven {state.overall_regime}"
                )

            # Write HMM probabilities to shared memory
            self._hmm_detector.write_regime_probabilities_shm()
        else:
            logger.debug("FEAT 11: Not enough price data for HMM detection")

        # ── 6. LLM market analysis (optional slow step) ──────────────────
        if self._llm is not None:
            llm_signals = await self._run_llm_analysis(market_data, state)
            if llm_signals:
                _merge_llm_signals(state, llm_signals)

        # ── 7. Compute strategy recommendations ─────────────────────────
        state.recommended_position_scale = _compute_position_scale(state)
        state.recommended_strategy_filter = _compute_allowed_strategies(state)
        state.blocked_strategies = _compute_blocked_strategies(state)
        state.max_leverage_override = _compute_leverage_override(state)

        return state

    def _extract_prices_for_hmm(
        self, market_data: Optional[Dict[str, Any]]
    ) -> List[float]:
        """Extract a price series from market_data for HMM input.

        Looks for:
            1. market_data["prices"]["BTC/USDT"] or first available symbol
            2. market_data["price_history"] as a list of floats
            3. market_data["ohlcv"] as a list of [t,o,h,l,c,v] candles
        """
        if not market_data:
            return []

        # Try explicit price_history
        history = market_data.get("price_history")
        if isinstance(history, list) and len(history) > 0:
            return [float(p) for p in history if isinstance(p, (int, float))]

        # Try ohlcv candles (use close price)
        ohlcv = market_data.get("ohlcv")
        if isinstance(ohlcv, list) and len(ohlcv) > 0:
            closes = []
            for candle in ohlcv:
                if isinstance(candle, (list, tuple)) and len(candle) >= 5:
                    closes.append(float(candle[4]))  # Close price
            if closes:
                return closes

        # Try prices dict (pick BTC/USDT or first)
        prices_dict = market_data.get("prices", {})
        if isinstance(prices_dict, dict):
            for key in ["BTC/USDT", "BTCUSDT"]:
                if key in prices_dict:
                    val = prices_dict[key]
                    if isinstance(val, list):
                        return [float(p) for p in val if isinstance(p, (int, float))]
            # Try first available
            for val in prices_dict.values():
                if isinstance(val, list) and len(val) > 10:
                    return [float(p) for p in val if isinstance(p, (int, float))]

        return []

    async def _fetch_news(self) -> List[Dict[str, str]]:
        """Fetch news from all configured news sources."""
        if not self._news_sources:
            return []
        items: List[Dict[str, str]] = []
        for source in self._news_sources:
            try:
                if asyncio.iscoroutinefunction(getattr(source, "fetch", None)):
                    fetched = await source.fetch()
                elif callable(getattr(source, "fetch", None)):
                    fetched = source.fetch()
                else:
                    continue
                if isinstance(fetched, list):
                    items.extend(fetched[:20])
            except Exception as exc:
                logger.debug(f"RegimeService: news source {source!r} failed: {exc}")
        return items[:50]

    async def _analyse_sentiment(
        self, news_items: List[Dict[str, str]]
    ) -> tuple[float, float, float]:
        """Return (sentiment_score, confidence, news_impact_score).

        All values are in [0, 1] or [-1, 1] as appropriate.
        """
        if not self._sentiment or not news_items:
            return 0.0, 0.0, 0.0

        texts = [
            item.get("content") or item.get("title", "")
            for item in news_items[:15]
            if item.get("content") or item.get("title")
        ]
        if not texts:
            return 0.0, 0.0, 0.0

        try:
            results = await self._sentiment.analyze_batch(texts)
            if not results:
                return 0.0, 0.0, 0.0
            scores = [r.score for r in results]
            confs = [r.confidence for r in results]
            avg_score = sum(scores) / len(scores)
            avg_conf = sum(confs) / len(confs)
            # News impact = normalised variance of sentiment (high variance = impactful)
            if len(scores) > 1:
                variance = statistics.variance(scores)
                impact = min(1.0, variance * 4.0)  # scale to [0, 1]
            else:
                impact = 0.0
            return avg_score, avg_conf, impact
        except Exception as exc:
            logger.warning(f"RegimeService sentiment analysis failed: {exc}")
            return 0.0, 0.0, 0.0

    async def _detect_single_asset_regime(
        self, ohlcv_data: Optional[Any]
    ) -> tuple[str, str]:
        """Return (overall_regime, volatility_regime) strings."""
        if self._regime_detector is None or ohlcv_data is None:
            return "unknown", "moderate"

        try:
            from ai.market_analyzer.regime_detector import MarketRegime

            regime_enum = await self._regime_detector.detect_regime(ohlcv_data)
            overall = _map_single_regime(regime_enum)
            vol_regime = _volatility_from_regime(regime_enum)
            return overall, vol_regime
        except Exception as exc:
            logger.warning(f"RegimeService: single-asset regime detection failed: {exc}")
            return "unknown", "moderate"

    async def _detect_cross_asset_regime(
        self, exchange: Optional[Any]
    ) -> tuple[str, float]:
        """Return (cross_asset_regime_name, confidence)."""
        if self._cross_asset_detector is None:
            return "unknown", 0.0
        try:
            regime, confidence, _ = await self._cross_asset_detector.detect_regime(exchange)
            return regime.value, confidence
        except Exception as exc:
            logger.warning(f"RegimeService: cross-asset regime detection failed: {exc}")
            return "unknown", 0.0

    async def _run_llm_analysis(
        self, market_data: Dict[str, Any], current_state: RegimeState
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM for a high-level market assessment.

        This is intentionally the **last** step and its results only
        *augment* the data-driven signals computed above.
        """
        if self._llm is None:
            return None
        try:
            from ai.prompt_engine import PromptEngine

            prompt = PromptEngine().build_market_confidence_prompt(
                market_overview={
                    "symbols": market_data.get("symbols", []),
                    "overall_regime": current_state.overall_regime,
                    "volatility": current_state.volatility_regime,
                },
                sentiment_score=current_state.sentiment_score,
                news_summary=f"fear_greed={current_state.fear_greed_index}",
            )
            result = await self._llm.analyze_market(prompt)
            if result and "error" not in result:
                return result
        except Exception as exc:
            logger.warning(f"RegimeService LLM analysis failed: {exc}")
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self, state: RegimeState) -> None:
        """Write the regime state to disk atomically AND to shared memory.

        Uses a write-then-rename pattern for the JSON file to avoid the
        Rust reader ever seeing a partial file.

        **Issue 4**: Also writes to ``/dev/shm/regime_weights`` via
        :class:`~ai.shared_regime_writer.SharedRegimeWriter` for the
        shared-memory-based cold-path architecture.
        """
        # 1. Write JSON file (legacy path)
        path = self._output_path
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            data = state.to_json()
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp_path, path)
            logger.debug(f"RegimeService: wrote {len(data)} bytes to {path}")
        except Exception as exc:
            logger.error(f"RegimeService: failed to persist state to {path}: {exc}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # 2. Write to shared memory (Issue 4 cold-path)
        if self._shm_writer is not None:
            try:
                from ai.shared_regime_writer import RegimeData

                regime_data = RegimeData(
                    overall_regime=state.overall_regime,
                    volatility_regime=state.volatility_regime,
                    sentiment_score=state.sentiment_score,
                    sentiment_confidence=state.sentiment_confidence,
                    fear_greed_index=state.fear_greed_index,
                    btc_dominance_trend=state.btc_dominance_trend,
                    funding_rate_bias=state.funding_rate_bias,
                    cross_asset_correlation=state.cross_asset_correlation,
                    news_impact_score=state.news_impact_score,
                    recommended_position_scale=state.recommended_position_scale,
                    max_leverage_override=state.max_leverage_override or 0,
                    ttl_seconds=state.ttl_seconds,
                )
                self._shm_writer.update(regime_data)
            except Exception as exc:
                logger.warning(
                    f"RegimeService: SharedRegimeWriter update failed: {exc}"
                )


# ---------------------------------------------------------------------------
# Pure helper functions (no dependencies → easily testable)
# ---------------------------------------------------------------------------


def _map_single_regime(regime_enum: Any) -> str:
    """Convert a ``MarketRegime`` enum value to a simplified regime string."""
    name = getattr(regime_enum, "value", str(regime_enum))
    mapping = {
        "STRONG_UPTREND": "trending_bullish",
        "WEAK_UPTREND": "trending_bullish",
        "STRONG_DOWNTREND": "trending_bearish",
        "WEAK_DOWNTREND": "trending_bearish",
        "RANGING": "ranging",
        "HIGH_VOLATILITY": "high_volatility",
        "LOW_VOLATILITY": "ranging",
        "CRASH": "trending_bearish",
        "UNKNOWN": "unknown",
    }
    return mapping.get(name, "unknown")


def _volatility_from_regime(regime_enum: Any) -> str:
    """Extract the volatility sub-regime string from a ``MarketRegime``."""
    name = getattr(regime_enum, "value", str(regime_enum))
    if name in ("HIGH_VOLATILITY", "CRASH"):
        return "high"
    if name == "LOW_VOLATILITY":
        return "low"
    if name == "UNKNOWN":
        return "moderate"
    return "moderate"


def _extract_btc_dominance_trend(market_data: Dict[str, Any]) -> str:
    """Extract BTC dominance trend from market overview dict."""
    dom = market_data.get("btc_dominance_change", 0.0)
    if isinstance(dom, (int, float)):
        if dom > 0.5:
            return "rising"
        if dom < -0.5:
            return "falling"
    return "flat"


def _extract_funding_rate_bias(market_data: Dict[str, Any]) -> str:
    """Classify funding rates as long_crowded / short_crowded / neutral."""
    rates = market_data.get("funding_rates", {})
    if not rates:
        return "neutral"
    values = [v for v in rates.values() if isinstance(v, (int, float))]
    if not values:
        return "neutral"
    avg = sum(values) / len(values)
    if avg > 0.001:
        return "long_crowded"
    if avg < -0.001:
        return "short_crowded"
    return "neutral"


def _extract_fear_greed(market_data: Dict[str, Any]) -> int:
    """Extract the Fear & Greed index from the market overview (default 50)."""
    val = market_data.get("fear_greed_index", 50)
    if isinstance(val, (int, float)):
        return max(0, min(100, int(val)))
    return 50


def _merge_llm_signals(state: RegimeState, llm_result: Dict[str, Any]) -> None:
    """Apply LLM directional bias to the regime state (in-place).

    The LLM result is used only as a **tiebreaker** when the data-driven
    regime is ``"unknown"`` or the overall_regime is ambiguous.
    """
    llm_direction = str(llm_result.get("direction", "neutral")).lower()
    llm_confidence = float(llm_result.get("confidence", 0.0))

    # Only override if the LLM is confident and the data-driven result is
    # uncertain.  Avoids hallucinated regime flips when the market data is
    # clear.
    if state.overall_regime == "unknown" and llm_confidence > 0.65:
        if llm_direction == "bullish":
            state.overall_regime = "trending_bullish"
        elif llm_direction == "bearish":
            state.overall_regime = "trending_bearish"
        else:
            state.overall_regime = "ranging"


def _compute_position_scale(state: RegimeState) -> float:
    """Compute the recommended position size multiplier.

    Rules (conservative bias):
    - CRASH / extreme volatility        → 0.0 (no new positions)
    - HIGH_VOLATILITY / high news impact → 0.5
    - RANGING / low volatility           → 0.75
    - trending_bullish  / trending_bearish → 1.0
    - Funding long_crowded (longs)        → -0.25
    """
    scale = 1.0

    if state.volatility_regime == "extreme":
        return 0.0
    if state.volatility_regime == "high" or state.news_impact_score > 0.7:
        scale = 0.5
    elif state.volatility_regime == "low" or state.overall_regime == "ranging":
        scale = 0.75
    elif state.overall_regime in ("trending_bullish", "trending_bearish"):
        scale = 1.0

    # Reduce if market is crowded on one side (carries reversal risk)
    if state.funding_rate_bias != "neutral":
        scale = max(0.25, scale - 0.25)

    return round(scale, 2)


def _compute_allowed_strategies(state: RegimeState) -> List[str]:
    """Return the list of strategy tags that work well in the current regime."""
    if state.overall_regime == "trending_bullish":
        return ["trend_following", "momentum", "breakout", "ema_ribbon"]
    if state.overall_regime == "trending_bearish":
        return ["trend_following", "momentum", "breakout", "short_strategies"]
    if state.overall_regime == "ranging":
        return ["mean_reversion", "market_making", "range_trading", "imbalance_maker"]
    if state.volatility_regime == "high":
        return ["volatility_breakout", "gamma_scalping"]
    # Unknown or extreme — allow nothing by default
    return []


def _compute_blocked_strategies(state: RegimeState) -> List[str]:
    """Return the list of strategy tags that should **not** be used now."""
    blocked = []
    if state.volatility_regime in ("high", "extreme"):
        blocked.extend(["mean_reversion", "market_making", "imbalance_maker"])
    if state.overall_regime in ("trending_bullish", "trending_bearish"):
        blocked.extend(["mean_reversion", "range_trading"])
    if state.news_impact_score > 0.8:
        blocked.extend(["market_making", "imbalance_maker"])
    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for s in blocked:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _compute_leverage_override(state: RegimeState) -> Optional[int]:
    """Return a hard leverage cap override if the regime warrants one."""
    if state.volatility_regime == "extreme":
        return 1
    if state.volatility_regime == "high":
        return 3
    if state.news_impact_score > 0.7:
        return 5
    return None
