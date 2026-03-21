"""Strategy manager — selects and rotates strategies based on market microstructure."""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from core.exceptions import StrategyError
from strategy.base_strategy import BaseStrategy, Signal
from strategy.strategies.mean_reversion_strategy import MeanReversionStrategy
from strategy.strategies.momentum_strategy import MomentumStrategy
from strategy.strategies.trend_following_strategy import TrendFollowingStrategy
from strategy.strategy_selector import StrategySelector

try:
    from ai.reinforcement.rl_strategy_optimizer import RLStrategyOptimizer  # noqa: F401
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.debug("RLStrategyOptimizer not available, using heuristic selection only")

try:
    from ai.reinforcement.capital_allocator import StrategyCapitalAllocator
    from ai.reinforcement.regime_predictor import RegimeTransitionPredictor
    RL_EXTRA_AVAILABLE = True
except ImportError:
    RL_EXTRA_AVAILABLE = False


class ConfidenceCalibrator:
    """Adjusts raw strategy confidence based on historical accuracy.

    After at least ``_min_trades_for_calibration`` trades have been recorded
    for a strategy, :meth:`calibrate` scales the raw confidence by the ratio
    of the observed win-rate to the average confidence that was present when
    trades were taken.  This corrects for over- or under-confident raw scores
    without requiring any persistence between restarts.
    """

    def __init__(self) -> None:
        # strategy_name -> {"wins": int, "losses": int, "total_confidence": float}
        self._stats: Dict[str, Dict[str, float]] = {}
        self._min_trades_for_calibration: int = 20

    def calibrate(self, strategy_name: str, raw_confidence: float) -> float:
        """Return calibrated confidence based on historical win rate.

        If fewer than ``_min_trades_for_calibration`` trades have been recorded
        for *strategy_name*, the raw value is returned unchanged.

        Args:
            strategy_name: Name of the strategy that produced the signal.
            raw_confidence: Uncalibrated confidence in ``[0.0, 1.0]``.

        Returns:
            Calibrated confidence clamped to ``[0.0, 1.0]``.
        """
        stats = self._stats.get(strategy_name)
        if stats is None or (stats["wins"] + stats["losses"]) < self._min_trades_for_calibration:
            return raw_confidence  # Not enough data yet

        total = stats["wins"] + stats["losses"]
        historical_win_rate = stats["wins"] / total
        avg_confidence = stats["total_confidence"] / total if total > 0 else 0.5

        # Scale raw confidence by (actual_win_rate / avg_confidence_when_traded)
        if avg_confidence > 0:
            calibration_factor = historical_win_rate / avg_confidence
            calibrated = raw_confidence * calibration_factor
        else:
            calibrated = raw_confidence

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, calibrated))

    def record_outcome(
        self, strategy_name: str, confidence_at_entry: float, won: bool
    ) -> None:
        """Record a trade outcome for future calibration.

        Args:
            strategy_name: Name of the strategy that generated the signal.
            confidence_at_entry: The (calibrated) confidence value that was
                present when the trade was entered.
            won: ``True`` if the trade closed with ``pnl >= 0``.
        """
        if strategy_name not in self._stats:
            self._stats[strategy_name] = {"wins": 0.0, "losses": 0.0, "total_confidence": 0.0}
        stats = self._stats[strategy_name]
        if won:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["total_confidence"] += confidence_at_entry


class StrategyManager:
    """Manages multiple strategies and routes signal generation.

    Responsibilities
    ----------------
    * Register / deregister strategies.
    * Enable / disable individual strategies at runtime.
    * Gather signals from all enabled strategies for a symbol.
    * Select the best-performing strategy for a given regime.
    * Track per-strategy performance metrics.
    """

    def __init__(self, rl_optimizer: Optional[Any] = None) -> None:
        self._strategies: Dict[str, BaseStrategy] = {}
        # performance: strategy_name -> {"wins": int, "losses": int, "pnl": float}
        self._performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"wins": 0.0, "losses": 0.0, "pnl": 0.0}
        )
        # per-regime win rates: strategy_name -> regime -> {"wins": int, "total": int}
        self._regime_performance: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "total": 0})
        )
        # Rolling performance tracker: strategy_name -> regime ->
        #   deque of (pnl, is_win) tuples (last 50 trades)
        self._rolling_trades: Dict[str, Dict[str, Deque[Tuple[float, bool]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=50))
        )
        # Track consecutive losses per strategy for auto-disabling
        self._consecutive_losses: Dict[str, int] = defaultdict(int)
        # Auto-disable timestamps: strategy_name -> re-enable unix timestamp
        self._disabled_until: Dict[str, float] = {}
        # Monthly reset timestamp
        self._last_monthly_reset: float = time.time()
        self._initialized = False
        # Intelligent strategy selector
        self._selector = StrategySelector(top_n=7)
        # Last selection scores for introspection
        self._last_selection_scores: Dict[str, Dict[str, Any]] = {}
        # RL strategy optimizer (optional)
        self._rl_optimizer = rl_optimizer
        # Minimum confidence a signal must have to be emitted (configurable)
        self._min_signal_confidence: float = 0.55
        # Signals older than this many seconds are treated as stale and discarded
        self._signal_expiry_seconds: float = 30.0
        # Calibration layer — adjusts raw confidence by historical win-rate
        self._confidence_calibrator = ConfidenceCalibrator()
        # Capital allocator (lazily initialised after strategies are registered)
        self._capital_allocator: Optional[Any] = None
        # Regime transition predictor for crash pre-activation
        self._regime_predictor: Optional[Any] = None
        if RL_EXTRA_AVAILABLE:
            self._regime_predictor = RegimeTransitionPredictor()
        self._register_default_strategies()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all registered strategies."""
        logger.info("Initializing StrategyManager …")
        for name, strategy in self._strategies.items():
            logger.debug(f"Loaded strategy: {name} (enabled={strategy.enabled})")
        # Lazily initialise the capital allocator with all registered strategy names
        if RL_EXTRA_AVAILABLE and self._capital_allocator is None:
            self._capital_allocator = StrategyCapitalAllocator(
                strategy_names=list(self._strategies.keys())
            )
        self._initialized = True
        logger.info(f"StrategyManager ready with {len(self._strategies)} strategies.")

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register_default_strategies(self) -> None:
        """Register all built-in concrete strategies (empty symbol list = all symbols)."""
        # Core strategies always imported at module level
        for strategy in (
            MomentumStrategy(symbols=[]),
            MeanReversionStrategy(symbols=[]),
            TrendFollowingStrategy(symbols=[]),
        ):
            self.add_strategy(strategy)

        # Extended strategy set (12 original + 31 new = 43 additional strategies loaded here;
        # together with the 3 core strategies above this totals 46) — import errors are caught
        # individually so that a single missing dependency does not prevent the others
        # from loading.
        _extra: List[tuple] = [
            # --- Original strategies ---
            ("strategy.strategies.ai_adaptive", "AIAdaptiveStrategy", {}),
            ("strategy.strategies.dca_strategy", "DCAStrategy", {}),
            ("strategy.strategies.funding_rate_arb", "FundingRateArbStrategy", {}),
            ("strategy.strategies.grid_trading", "GridTradingStrategy", {}),
            ("strategy.strategies.liquidation_hunter", "LiquidationHunterStrategy", {}),
            ("strategy.strategies.market_making", "MarketMakingStrategy", {}),
            ("strategy.strategies.news_momentum", "NewsMomentumStrategy", {}),
            ("strategy.strategies.scalping", "ScalpingStrategy", {}),
            ("strategy.strategies.sentiment_reversal", "SentimentReversalStrategy", {}),
            ("strategy.strategies.smart_money_flow", "SmartMoneyFlowStrategy", {}),
            ("strategy.strategies.technical_breakout", "TechnicalBreakoutStrategy", {}),
            ("strategy.strategies.whale_follower", "WhaleFollowerStrategy", {}),
            # --- New strategies (31) ---
            ("strategy.strategies.bollinger_squeeze", "BollingerSqueezeStrategy", {}),
            ("strategy.strategies.vwap_deviation", "VWAPDeviationStrategy", {}),
            ("strategy.strategies.ichimoku_cloud", "IchimokuCloudStrategy", {}),
            ("strategy.strategies.fibonacci_retracement", "FibonacciRetracementStrategy", {}),
            ("strategy.strategies.order_flow_imbalance", "OrderFlowImbalanceStrategy", {}),
            ("strategy.strategies.volume_profile", "VolumeProfileStrategy", {}),
            ("strategy.strategies.rsi_divergence", "RSIDivergenceStrategy", {}),
            ("strategy.strategies.macd_crossover", "MACDCrossoverStrategy", {}),
            ("strategy.strategies.ema_ribbon", "EMARibbonStrategy", {}),
            ("strategy.strategies.supertrend", "SupertrendStrategy", {}),
            ("strategy.strategies.keltner_channel", "KeltnerChannelStrategy", {}),
            ("strategy.strategies.donchian_breakout", "DonchianBreakoutStrategy", {}),
            ("strategy.strategies.parabolic_sar", "ParabolicSARStrategy", {}),
            ("strategy.strategies.stochastic_rsi", "StochasticRSIStrategy", {}),
            ("strategy.strategies.williams_r", "WilliamsRStrategy", {}),
            ("strategy.strategies.adx_trend", "ADXTrendStrategy", {}),
            ("strategy.strategies.pivot_point", "PivotPointStrategy", {}),
            ("strategy.strategies.harmonic_pattern", "HarmonicPatternStrategy", {}),
            ("strategy.strategies.elliott_wave", "ElliottWaveStrategy", {}),
            ("strategy.strategies.supply_demand_zone", "SupplyDemandZoneStrategy", {}),
            ("strategy.strategies.market_structure_break", "MarketStructureBreakStrategy", {}),
            ("strategy.strategies.fair_value_gap", "FairValueGapStrategy", {}),
            ("strategy.strategies.order_block", "OrderBlockStrategy", {}),
            ("strategy.strategies.accumulation_distribution", "AccDistStrategy", {}),
            ("strategy.strategies.on_chain_momentum", "OnChainMomentumStrategy", {}),
            ("strategy.strategies.correlation_divergence", "CorrelationDivergenceStrategy", {}),
            ("strategy.strategies.volatility_breakout", "VolatilityBreakoutStrategy", {}),
            ("strategy.strategies.time_based", "TimeBasedStrategy", {}),
            ("strategy.strategies.multi_timeframe_confluence", "MTFConfluenceStrategy", {}),
            ("strategy.strategies.range_breakout", "RangeBreakoutStrategy", {}),
            ("strategy.strategies.momentum_divergence", "MomentumDivergenceStrategy", {}),
        ]

        # --- Gold/Silver-specific forex strategies (15) ---
        _forex_strategies: List[tuple] = [
            ("strategy.strategies.forex.london_breakout", "LondonBreakoutStrategy", {}),
            ("strategy.strategies.forex.gold_dxy_inverse", "GoldDXYInverseStrategy", {}),
            ("strategy.strategies.forex.nfp_news_strategy", "NFPNewsStrategy", {}),
            ("strategy.strategies.forex.gold_mean_reversion", "GoldMeanReversionStrategy", {}),
            ("strategy.strategies.forex.gold_momentum_breakout", "GoldMomentumBreakoutStrategy", {}),
            ("strategy.strategies.forex.gold_fibonacci", "GoldFibonacciStrategy", {}),
            ("strategy.strategies.forex.gold_supply_demand", "GoldSupplyDemandStrategy", {}),
            ("strategy.strategies.forex.gold_rsi_divergence", "GoldRSIDivergenceStrategy", {}),
            ("strategy.strategies.forex.gold_bollinger_squeeze", "GoldBollingerSqueezeStrategy", {}),
            ("strategy.strategies.forex.gold_ichimoku", "GoldIchimokuStrategy", {}),
            ("strategy.strategies.forex.gold_vwap", "GoldVWAPStrategy", {}),
            ("strategy.strategies.forex.gold_scalping", "GoldScalpingStrategy", {}),
            ("strategy.strategies.forex.silver_gold_ratio", "SilverGoldRatioStrategy", {}),
            ("strategy.strategies.forex.gold_session_momentum", "GoldSessionMomentumStrategy", {}),
            ("strategy.strategies.forex.gold_safe_haven", "GoldSafeHavenStrategy", {}),
        ]
        _extra.extend(_forex_strategies)

        # --- 50 new quantitative strategies ---
        _new_strategies: List[tuple] = [
            # Category 1: Statistical Arbitrage (10)
            ("strategy.strategies.cointegration_pairs", "CointegrationPairsStrategy", {}),
            ("strategy.strategies.zscore_mean_reversion", "ZScoreMeanReversionStrategy", {}),
            ("strategy.strategies.ou_process", "OUProcessStrategy", {}),
            ("strategy.strategies.kalman_pairs", "KalmanPairsStrategy", {}),
            ("strategy.strategies.pca_factor", "PCAFactorStrategy", {}),
            ("strategy.strategies.hurst_exponent", "HurstExponentStrategy", {}),
            ("strategy.strategies.copula_dependence", "CopulaDependenceStrategy", {}),
            ("strategy.strategies.regime_stat_arb", "RegimeStatArbStrategy", {}),
            ("strategy.strategies.cross_sectional_momentum", "CrossSectionalMomentumStrategy", {}),
            ("strategy.strategies.dispersion_trading", "DispersionTradingStrategy", {}),
            # Category 2: Market Microstructure (8)
            ("strategy.strategies.vpin_toxicity", "VPINToxicityStrategy", {}),
            ("strategy.strategies.kyles_lambda", "KylesLambdaStrategy", {}),
            ("strategy.strategies.amihud_illiquidity", "AmihudIlliquidityStrategy", {}),
            ("strategy.strategies.tick_imbalance", "TickImbalanceStrategy", {}),
            ("strategy.strategies.dollar_bars", "DollarBarsStrategy", {}),
            ("strategy.strategies.entropy_strategy", "EntropyStrategy", {}),
            ("strategy.strategies.spoof_detector", "SpoofDetectorStrategy", {}),
            ("strategy.strategies.latency_arb", "LatencyArbStrategy", {}),
            # Category 3: Options/Volatility (7)
            ("strategy.strategies.iv_surface", "IVSurfaceStrategy", {}),
            ("strategy.strategies.gamma_scalping", "GammaScalpingStrategy", {}),
            ("strategy.strategies.vol_risk_premium", "VolRiskPremiumStrategy", {}),
            ("strategy.strategies.variance_swap", "VarianceSwapStrategy", {}),
            ("strategy.strategies.straddle_replication", "StraddleReplicationStrategy", {}),
            ("strategy.strategies.term_structure", "TermStructureStrategy", {}),
            ("strategy.strategies.vol_clustering", "VolClusteringStrategy", {}),
            # Category 4: ML-Inspired (8)
            ("strategy.strategies.xgboost_classifier", "XGBoostClassifierStrategy", {}),
            ("strategy.strategies.random_forest_regime", "RandomForestRegimeStrategy", {}),
            ("strategy.strategies.isolation_forest", "IsolationForestStrategy", {}),
            ("strategy.strategies.dbscan_breakout", "DBSCANBreakoutStrategy", {}),
            ("strategy.strategies.rl_agent_strategy", "RLAgentStrategy", {}),
            ("strategy.strategies.genetic_optimizer", "GeneticOptimizerStrategy", {}),
            ("strategy.strategies.bayesian_strategy", "BayesianStrategy", {}),
            ("strategy.strategies.nn_ensemble", "NNEnsembleStrategy", {}),
            # Category 5: Advanced Technical (10)
            ("strategy.strategies.wyckoff", "WyckoffStrategy", {}),
            ("strategy.strategies.market_profile", "MarketProfileStrategy", {}),
            ("strategy.strategies.auction_theory", "AuctionTheoryStrategy", {}),
            ("strategy.strategies.delta_divergence", "DeltaDivergenceStrategy", {}),
            ("strategy.strategies.footprint_chart", "FootprintChartStrategy", {}),
            ("strategy.strategies.renko_breakout", "RenkoBreakoutStrategy", {}),
            ("strategy.strategies.heikin_ashi", "HeikinAshiStrategy", {}),
            ("strategy.strategies.point_figure", "PointFigureStrategy", {}),
            ("strategy.strategies.kagi_reversal", "KagiReversalStrategy", {}),
            ("strategy.strategies.three_line_break", "ThreeLineBreakStrategy", {}),
            # Category 6: Macro & Sentiment (7)
            ("strategy.strategies.funding_momentum", "FundingMomentumStrategy", {}),
            ("strategy.strategies.oi_divergence", "OIDivergenceStrategy", {}),
            ("strategy.strategies.liquidation_cascade", "LiquidationCascadeStrategy", {}),
            ("strategy.strategies.whale_wallet", "WhaleWalletStrategy", {}),
            ("strategy.strategies.social_momentum", "SocialMomentumStrategy", {}),
            ("strategy.strategies.fear_greed_contrarian", "FearGreedContrarianStrategy", {}),
            ("strategy.strategies.defi_tvl_flow", "DeFiTVLFlowStrategy", {}),
        ]
        _extra.extend(_new_strategies)

        import importlib

        for module_path, class_name, kwargs in _extra:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                strategy = cls(symbols=[], **kwargs)
                # Only register if no strategy with this name is already present
                if strategy.name not in self._strategies:
                    self.add_strategy(strategy)
                else:
                    logger.debug(
                        f"Strategy {strategy.name!r} already registered — skipping {class_name}."
                    )
            except Exception as exc:
                logger.warning(
                    f"Could not load strategy {class_name} from {module_path}: {exc}"
                )

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Register *strategy* in the manager."""
        if strategy.name in self._strategies:
            logger.warning(f"Strategy {strategy.name!r} already registered — replacing.")
        self._strategies[strategy.name] = strategy
        logger.info(f"Strategy registered: {strategy.name!r}")

    def remove_strategy(self, name: str) -> None:
        """Remove a strategy by *name*."""
        if name not in self._strategies:
            raise StrategyError(f"Strategy {name!r} not found.", strategy=name)
        del self._strategies[name]
        logger.info(f"Strategy removed: {name!r}")

    def enable_strategy(self, name: str) -> None:
        """Enable a previously disabled strategy."""
        self._get_or_raise(name).enabled = True
        logger.info(f"Strategy enabled: {name!r}")

    def disable_strategy(self, name: str) -> None:
        """Disable a strategy without removing it."""
        self._get_or_raise(name).enabled = False
        logger.info(f"Strategy disabled: {name!r}")

    # ------------------------------------------------------------------
    # Intelligent strategy selection
    # ------------------------------------------------------------------

    def get_regime_appropriate_strategies(
        self, regime: str, volatility: str = "normal"
    ) -> List[str]:
        """Return names of strategies appropriate for *regime* and *volatility*.

        Delegates to :class:`~strategy.strategy_selector.StrategySelector` for
        intelligent microstructure-based selection when ``market_data`` is
        available.  This method provides a regime-only fallback used by legacy
        callers that do not pass full market data.

        Args:
            regime: Market regime label.
            volatility: Volatility regime label.

        Returns:
            List of strategy names that should run in the current conditions.
            An empty list means *all* enabled strategies should run (fallback).
        """
        # Normalise volatility labels from VolatilityAnalyzer ("medium" → "normal")
        vol_norm = volatility if volatility != "medium" else "normal"

        # Map volatility + regime to the effective regime key used for affinity lookup
        if regime == "crash":
            effective_regime = "crash"
        elif vol_norm == "extreme":
            effective_regime = "extreme"
        elif vol_norm == "high":
            effective_regime = "high_volatility"
        else:
            effective_regime = regime

        # Use StrategySelector with empty market data (regime affinity only)
        selected, scores = self._selector.select_strategies(
            self._strategies,
            market_data={},
            rolling_metrics={
                name: self.get_rolling_metrics(name) for name in self._strategies
            },
            regime=effective_regime,
        )
        self._last_selection_scores = scores
        return selected

    def get_strategy_selection_reasoning(self) -> Dict[str, str]:
        """Return why each strategy was selected or rejected in the last cycle.

        Returns:
            Mapping of strategy name → human-readable explanation string.
        """
        return self._selector.get_selection_reasoning(self._last_selection_scores)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def get_signals(self, symbol: str) -> List[Signal]:
        """Gather signals from all enabled strategies that cover *symbol*."""
        tasks = []
        names = []
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            if symbol not in strategy.symbols and strategy.symbols:
                continue
            tasks.append(strategy.generate_signal(symbol))
            names.append(name)

        if not tasks:
            logger.debug(f"No enabled strategies for {symbol}")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        signals: List[Signal] = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {name!r} raised an error for {symbol}: {result}")
            elif isinstance(result, Signal):
                signals.append(result)
        return signals

    async def evaluate_all(
        self,
        symbol: str,
        market_data: Dict[str, pd.DataFrame],
        sentiment_items: Optional[List] = None,
        regime: str = "unknown",
        volatility_regime: str = "normal",
        order_flow_signal: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Run intelligently-selected strategies' :meth:`~BaseStrategy.analyze`.

        Strategies are selected by :class:`~strategy.strategy_selector.StrategySelector`
        or :class:`~ai.reinforcement.rl_strategy_optimizer.RLStrategyOptimizer` (if enabled)
        based on current market microstructure, historical performance, and regime
        appropriateness — not just a static regime mapping.

        Strategies that return ``None`` (no signal) are silently skipped.
        When two or more strategies agree on the same *direction*, each of
        their confidence scores is boosted by +0.05 (capped at 1.0).

        Sentiment data from the :class:`~data.aggregator.DataAggregator` is
        optionally applied as a confidence modifier of up to ±10 % based on
        the average sentiment score of relevant :class:`~data.sources.base_source.DataItem`
        objects supplied via *sentiment_items*.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            market_data: Mapping of timeframe label to OHLCV DataFrame,
                e.g. ``{"15m": df_15m, "1h": df_1h, "4h": df_4h}``.
            sentiment_items: Optional list of :class:`~data.sources.base_source.DataItem`
                objects relevant to *symbol*.  When provided, the net sentiment
                is used to adjust signal confidence by ±0.10.
            regime: Current market regime label (e.g. ``"trending_up"``).
            volatility_regime: Current volatility regime label (e.g. ``"high"``).
            order_flow_signal: Optional order flow signal from OrderFlowAnalyzer.

        Returns:
            A list of signal dicts sorted by *confidence* (descending).
            The list contains at most one entry per strategy and reflects
            any consensus confidence boost.  The highest-confidence signal
            for *symbol* is always first.
        """
        primary_df = market_data.get("15m", pd.DataFrame())

        # Use RL optimizer if available, otherwise fall back to heuristic selector
        rolling_metrics_all = {
            name: self.get_rolling_metrics(name) for name in self._strategies
        }

        if self._rl_optimizer is not None:
            # Build context vector for RL selection
            context = self._build_rl_context(
                regime=regime,
                volatility_regime=volatility_regime,
                market_data=market_data,
                order_flow_signal=order_flow_signal,
            )

            # Get heuristic selection as fallback for warm-up period
            heuristic_names, _ = self._selector.select_strategies(
                self._strategies,
                market_data=market_data,
                rolling_metrics=rolling_metrics_all,
                regime=regime,
                symbol=symbol,
            )

            # Use RL optimizer to select strategies
            strategy_names_list = list(self._strategies.keys())
            selected_names, rl_metadata = self._rl_optimizer.select_strategies(
                context=context,
                strategy_names=strategy_names_list,
                heuristic_fallback=heuristic_names,
            )

            logger.debug(
                f"RL strategy selection: method={rl_metadata.get('method')}, "
                f"selected={selected_names[:3]}..."
            )
        else:
            # Use traditional heuristic selector
            selected_names, selection_scores = self._selector.select_strategies(
                self._strategies,
                market_data=market_data,
                rolling_metrics=rolling_metrics_all,
                regime=regime,
                symbol=symbol,
            )
            self._last_selection_scores = selection_scores

        # Fall back to all enabled strategies if selector returns empty list
        if not selected_names:
            selected_names = [
                name
                for name, s in self._strategies.items()
                if s.enabled and (not s.symbols or symbol in s.symbols)
            ]

        raw_signals: List[Dict[str, Any]] = []
        for strategy in self._strategies.values():
            if not strategy.enabled:
                continue
            if strategy.symbols and symbol not in strategy.symbols:
                continue
            if strategy.name not in selected_names:
                logger.debug(
                    f"Skipping {strategy.name!r} — not selected for regime={regime!r} "
                    f"vol={volatility_regime!r}"
                )
                continue
            try:
                sig = strategy.analyze(primary_df, symbol)
                if sig is not None:
                    sig_copy = dict(sig)  # defensive copy
                    # Stamp every signal with the current wall-clock time
                    sig_copy.setdefault("generated_at", time.time())
                    raw_signals.append(sig_copy)
            except Exception as exc:
                logger.error(
                    f"Strategy {strategy.name!r} raised during evaluate_all for {symbol}: {exc}"
                )

        if not raw_signals:
            return []

        # Consensus: count directions
        direction_counts: Dict[str, int] = defaultdict(int)
        for s in raw_signals:
            direction_counts[s["direction"]] += 1

        # Boost confidence by 0.05 for each strategy that joins the majority direction
        boosted: List[Dict[str, Any]] = []
        for s in raw_signals:
            entry = dict(s)
            if direction_counts[entry["direction"]] >= 2:
                entry["confidence"] = round(min(1.0, entry["confidence"] + 0.05), 3)
            boosted.append(entry)

        # Apply sentiment modifier (±10 % of confidence) when items are provided
        if sentiment_items:
            sentiment_modifier = self._compute_sentiment_modifier(sentiment_items)
            if sentiment_modifier != 0.0:
                logger.debug(
                    f"Applying sentiment modifier {sentiment_modifier:+.3f} to {symbol} signals"
                )
                for entry in boosted:
                    raw_conf = entry["confidence"]
                    entry["confidence"] = round(
                        min(1.0, max(0.0, raw_conf + sentiment_modifier)), 3
                    )
                    entry["sentiment_modifier"] = sentiment_modifier

        # Apply confidence calibration based on historical per-strategy accuracy
        for signal in boosted:
            strategy_name = signal.get("strategy", "unknown")
            raw_conf = signal.get("confidence", 0.5)
            signal["raw_confidence"] = raw_conf
            signal["confidence"] = self._confidence_calibrator.calibrate(strategy_name, raw_conf)

        # Filter out signals below the minimum confidence threshold
        boosted = [
            s for s in boosted if s.get("confidence", 0) >= self._min_signal_confidence
        ]

        # Filter out stale signals (only relevant for cached/pre-generated signals)
        now = time.time()
        boosted = [
            s for s in boosted
            if (now - s.get("generated_at", now)) < self._signal_expiry_seconds
        ]

        if not boosted:
            return []

        # Resolve directional conflicts: if both LONG and SHORT exist for a symbol,
        # keep only the higher-confidence signal.
        boosted = self._resolve_conflicts(boosted)

        # Sort by confidence descending so the best signal is first
        boosted.sort(key=lambda x: x["confidence"], reverse=True)
        return boosted

    def _resolve_conflicts(self, signals: List[Dict]) -> List[Dict]:
        """When both LONG and SHORT signals exist for a symbol, keep only the stronger one.

        Args:
            signals: List of signal dicts (may contain mixed directions per symbol).

        Returns:
            Filtered list with at most one direction per symbol.  When a conflict
            is detected, the signal with the highest ``confidence`` value wins.
        """
        by_symbol: Dict[str, List[Dict]] = {}
        for s in signals:
            sym = s.get("symbol", "")
            by_symbol.setdefault(sym, []).append(s)

        resolved: List[Dict] = []
        for sym, sym_signals in by_symbol.items():
            directions = {s.get("direction") for s in sym_signals}
            if "long" in directions and "short" in directions:
                # Conflict: keep only the highest-confidence signal
                best = max(sym_signals, key=lambda s: s.get("confidence", 0))
                logger.info(
                    f"Signal conflict for {sym}: {len(sym_signals)} signals, "
                    f"keeping {best.get('direction')} (conf={best.get('confidence', 0):.3f})"
                )
                resolved.append(best)
            else:
                resolved.extend(sym_signals)
        return resolved

    @staticmethod
    def _compute_sentiment_modifier(items: List) -> float:
        """Derive a confidence modifier in the range [-0.10, +0.10] from *items*.

        The modifier is the average ``metadata["sentiment_score"]`` across items
        that carry one, scaled to ±10 %.  Items without that key contribute a
        neutral score of 0.  Returns 0.0 when no items are provided.
        """
        if not items:
            return 0.0
        total = 0.0
        count = 0
        for item in items:
            meta = getattr(item, "metadata", None) or {}
            if "sentiment_score" in meta:
                total += float(meta["sentiment_score"])
                count += 1
        if count == 0:
            return 0.0
        avg_sentiment = total / count  # in [-1, 1]
        # Clamp to [-1, 1] then scale to ±0.10
        clamped = max(-1.0, min(1.0, avg_sentiment))
        return round(clamped * 0.10, 4)

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    async def select_strategy(self, symbol: str, regime: str) -> Optional[BaseStrategy]:
        """Select the best strategy for *symbol* given the current *regime*.

        Selection favours enabled strategies that:
        1. Cover *symbol* (or have an empty symbol list, meaning "all").
        2. Have the highest win-rate adjusted by total trades.

        Falls back to the first enabled strategy if no performance data exists.
        """
        candidates = [
            s
            for s in self._strategies.values()
            if s.enabled and (not s.symbols or symbol in s.symbols)
        ]
        if not candidates:
            logger.warning(f"No eligible strategies for {symbol} in regime {regime!r}")
            return None

        def _score(s: BaseStrategy) -> float:
            perf = self._performance[s.name]
            total = perf["wins"] + perf["losses"]
            if total == 0:
                return 0.5  # unknown — treat as neutral
            return perf["wins"] / total + perf["pnl"] * 0.01

        best = max(candidates, key=_score)
        logger.debug(f"Selected strategy {best.name!r} for {symbol} ({regime})")
        return best

    # ------------------------------------------------------------------
    # Forex / gold-silver strategy selection
    # ------------------------------------------------------------------

    # Names of all registered gold/silver forex strategies (order matters:
    # the first entry is the default fallback).
    _FOREX_STRATEGY_NAMES: List[str] = [
        "gold_momentum_breakout",
        "gold_fibonacci",
        "gold_mean_reversion",
        "london_breakout",
        "gold_ichimoku",
        "gold_bollinger_squeeze",
        "gold_rsi_divergence",
        "gold_vwap",
        "gold_supply_demand",
        "gold_dxy_inverse",
        "gold_session_momentum",
        "gold_safe_haven",
        "gold_scalping",
        "silver_gold_ratio",
        "nfp_news_strategy",
    ]

    async def select_forex_strategy(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        ai_brain: Any = None,
        news_available: bool = False,
    ) -> str:
        """Select the best forex strategy for *symbol*.

        Selection order:
        1. AI recommendation (when *ai_brain* is provided).
        2. News/session-based heuristic (when *news_available* is True).
        3. Best historically-performing forex strategy (fallback).

        Args:
            symbol: Trading pair symbol (e.g. ``"XAU/USD"``).
            market_data: Contextual market data dict (passed to AI brain).
            ai_brain: Optional AI brain object that exposes
                ``get_forex_strategy_recommendation(symbol, market_data)``.
            news_available: Whether live news/economic calendar data is
                available for heuristic session/event-based selection.

        Returns:
            Strategy name string (one of ``_FOREX_STRATEGY_NAMES``).
        """
        # 1. AI recommendation
        if ai_brain is not None:
            try:
                recommendation = await ai_brain.get_forex_strategy_recommendation(
                    symbol, market_data
                )
                if recommendation:
                    strategy_name = recommendation.get("strategy")
                    if strategy_name and strategy_name in self._strategies:
                        logger.info(
                            f"[ForexSelector] AI recommended {strategy_name!r} for {symbol}"
                        )
                        return strategy_name
            except Exception as exc:
                logger.warning(f"[ForexSelector] AI recommendation failed: {exc}")

        # 2. News / session-based heuristic
        if news_available:
            from datetime import datetime, timezone

            now_h = datetime.now(timezone.utc).hour

            # Within 2 hours of major news windows (NFP ~13:30 GMT, FOMC ~18:00 GMT)
            news_hours = {13, 14, 18, 19}
            if now_h in news_hours and "nfp_news_strategy" in self._strategies:
                logger.info(
                    "[ForexSelector] News window detected — selecting nfp_news_strategy"
                )
                return "nfp_news_strategy"

            # London session (08:00–16:00 GMT)
            if 8 <= now_h < 16 and "london_breakout" in self._strategies:
                logger.info(
                    "[ForexSelector] London session active — selecting london_breakout"
                )
                return "london_breakout"

        # 3. Best historically-performing forex strategy
        best = self._get_best_performing_forex_strategy(symbol)
        chosen = best or "gold_momentum_breakout"
        logger.info(f"[ForexSelector] Fallback to {chosen!r} for {symbol}")
        return chosen

    def _get_best_performing_forex_strategy(self, symbol: str) -> Optional[str]:
        """Return the forex strategy with the highest win-rate for *symbol*.

        Only considers enabled strategies that are present in
        ``_FOREX_STRATEGY_NAMES``.  Returns ``None`` when no performance
        data is available; callers should fall back to the first entry of
        ``_FOREX_STRATEGY_NAMES`` (``"gold_momentum_breakout"``) in that case.
        """
        best_name: Optional[str] = None
        best_score: float = -1.0

        for name in self._FOREX_STRATEGY_NAMES:
            if name not in self._strategies:
                continue
            if not self._is_enabled_now(name):
                continue
            perf = self._performance[name]
            total = perf["wins"] + perf["losses"]
            if total == 0:
                continue
            score = perf["wins"] / total + perf["pnl"] * 0.01
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    # ------------------------------------------------------------------
    # Capital allocation (Update 1)
    # ------------------------------------------------------------------

    def allocate_capital(
        self,
        total_capital: float,
        active_strategies: Optional[List[str]] = None,
        regime: str = "unknown",
    ) -> Dict[str, float]:
        """Allocate capital across strategies using Sharpe-ratio-weighted softmax.

        Args:
            total_capital: Total available capital in quote currency.
            active_strategies: Optional subset of strategies to allocate to.
                Defaults to all currently enabled strategies.
            regime: Current market regime (for rolling metrics lookup).

        Returns:
            ``{strategy_name: capital_amount}`` mapping.
        """
        if self._capital_allocator is None:
            # Fall back to uniform allocation
            candidates = active_strategies or [
                n for n, s in self._strategies.items() if s.enabled
            ]
            if not candidates:
                return {}
            per_strategy = total_capital / len(candidates)
            return {name: per_strategy for name in candidates}

        rolling_metrics_all = {
            name: self.get_rolling_metrics(name, regime)
            for name in self._strategies
        }

        # Auto-rebalance if interval elapsed
        if self._capital_allocator.should_rebalance():
            return self._capital_allocator.rebalance(
                rolling_metrics=rolling_metrics_all,
                total_capital=total_capital,
                active_strategies=active_strategies,
            )
        return self._capital_allocator.allocate(
            rolling_metrics=rolling_metrics_all,
            total_capital=total_capital,
            active_strategies=active_strategies,
        )

    def update_regime_predictor(
        self,
        log_return: float,
        volatility: float,
        volume_change: float,
        funding_rate: float = 0.0,
        current_regime_label: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """Feed a new bar's features to the regime transition predictor.

        Args:
            log_return: log(close_t / close_{t-1}).
            volatility: Normalised rolling volatility (0–1).
            volume_change: log(volume_t / rolling_mean_volume).
            funding_rate: Current funding rate.
            current_regime_label: Optional ground-truth regime label.

        Returns:
            Probability distribution over next regimes, or ``None`` if
            the predictor is not available.
        """
        if self._regime_predictor is None:
            return None
        self._regime_predictor.update(
            log_return=log_return,
            volatility=volatility,
            volume_change=volume_change,
            funding_rate=funding_rate,
            current_regime_label=current_regime_label,
        )
        next_regime_probs = self._regime_predictor.predict_next_regime()

        # Pre-activate crash protection strategies when warranted
        if self._regime_predictor.should_pre_activate_crash_protection():
            crash_strats = self._regime_predictor.get_crash_protection_strategies()
            for name in crash_strats:
                if name in self._strategies and not self._strategies[name].enabled:
                    self._strategies[name].enabled = True
                    logger.info(
                        "[StrategyManager] Pre-activated crash-protection strategy: {!r}", name
                    )
        return next_regime_probs

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def record_win(self, strategy_name: str, pnl: float, regime: str = "unknown") -> None:
        """Record a winning trade for *strategy_name*."""
        self._performance[strategy_name]["wins"] += 1
        self._performance[strategy_name]["pnl"] += pnl
        self._regime_performance[strategy_name][regime]["wins"] += 1
        self._regime_performance[strategy_name][regime]["total"] += 1
        # Rolling tracker
        self._rolling_trades[strategy_name][regime].append((pnl, True))
        self._consecutive_losses[strategy_name] = 0  # reset on win

    def record_loss(self, strategy_name: str, pnl: float, regime: str = "unknown") -> None:
        """Record a losing trade for *strategy_name*."""
        self._performance[strategy_name]["losses"] += 1
        self._performance[strategy_name]["pnl"] += pnl  # pnl is negative
        self._regime_performance[strategy_name][regime]["total"] += 1
        # Rolling tracker
        self._rolling_trades[strategy_name][regime].append((pnl, False))
        self._consecutive_losses[strategy_name] += 1
        # Auto-disable checks
        self._check_auto_disable(strategy_name)

    def get_regime_win_rate(self, strategy_name: str, regime: str) -> float:
        """Return the win rate of *strategy_name* in *regime* (0.0–1.0).

        Returns 0.5 when no trades have been recorded for the regime.
        """
        stats = self._regime_performance[strategy_name][regime]
        total = stats["total"]
        if total == 0:
            return 0.5
        return stats["wins"] / total

    def get_rolling_metrics(
        self, strategy_name: str, regime: str = "unknown"
    ) -> Dict[str, float]:
        """Return rolling performance metrics for the last 50 trades.

        Metrics computed:
        * ``win_rate``: fraction of winning trades (0–1)
        * ``avg_profit``: average PnL of winning trades
        * ``avg_loss``: average PnL of losing trades (negative)
        * ``profit_factor``: |total wins| / |total losses|
        * ``sharpe``: simplified Sharpe ratio of trade returns
        * ``max_drawdown``: maximum consecutive loss

        Args:
            strategy_name: Strategy to query.
            regime: Market regime to filter on (defaults to all regimes combined
                when no per-regime data is available).

        Returns:
            Dict of metric names to float values.
        """
        trades = list(self._rolling_trades[strategy_name][regime])
        if not trades:
            # Fall back to aggregating across all regimes
            all_trades: List[Tuple[float, bool]] = []
            for r_trades in self._rolling_trades[strategy_name].values():
                all_trades.extend(r_trades)
            trades = all_trades

        if not trades:
            return {
                "win_rate": 0.5,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 1.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
            }

        wins = [(pnl, w) for pnl, w in trades if w]
        losses = [(pnl, w) for pnl, w in trades if not w]

        win_rate = len(wins) / len(trades)
        avg_profit = sum(p for p, _ in wins) / len(wins) if wins else 0.0
        avg_loss = sum(p for p, _ in losses) / len(losses) if losses else 0.0

        total_win = sum(p for p, _ in wins)
        total_loss = abs(sum(p for p, _ in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else (1.0 if total_win == 0 else float("inf"))

        # Simplified Sharpe: mean / std of returns
        returns = [p for p, _ in trades]
        mean_r = sum(returns) / len(returns)
        if len(returns) > 1:
            variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = math.sqrt(variance)
            sharpe = mean_r / std_r if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown (max consecutive loss sum)
        max_dd = 0.0
        current_dd = 0.0
        for pnl, is_win in trades:
            if not is_win:
                current_dd += abs(pnl)
                max_dd = max(max_dd, current_dd)
            else:
                current_dd = 0.0

        return {
            "win_rate": round(win_rate, 4),
            "avg_profit": round(avg_profit, 6),
            "avg_loss": round(avg_loss, 6),
            "profit_factor": round(profit_factor, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
            "total_trades": len(trades),
        }

    def get_signal_weight(self, strategy_name: str, regime: str = "unknown") -> float:
        """Return a signal weight (0–1) based on rolling Sharpe ratio.

        Strategies with higher Sharpe ratios receive proportionally larger
        weight when signals are aggregated.  Disabled strategies return 0.

        Args:
            strategy_name: Strategy to query.
            regime: Current market regime.

        Returns:
            Weight in [0, 1].
        """
        if not self._is_enabled_now(strategy_name):
            return 0.0
        metrics = self.get_rolling_metrics(strategy_name, regime)
        sharpe = metrics.get("sharpe", 0.0)
        # Normalise: Sharpe of 2.0 = weight 1.0, linearly scaled
        weight = max(0.0, min(1.0, sharpe / 2.0))
        return round(weight, 4)

    def monthly_reset(self) -> None:
        """Re-enable all auto-disabled strategies and reset tracking.

        Should be called at the start of each calendar month to give all
        strategies a fresh opportunity to prove themselves.
        """
        now = time.time()
        for name in list(self._disabled_until.keys()):
            del self._disabled_until[name]
            if name in self._strategies:
                self._strategies[name].enabled = True
                logger.info("[StrategyManager] Monthly reset: re-enabled {!r}", name)
        self._consecutive_losses.clear()
        self._last_monthly_reset = now
        logger.info("[StrategyManager] Monthly performance reset completed")

    @property
    def performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of the current performance data."""
        return dict(self._performance)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_enabled_now(self, strategy_name: str) -> bool:
        """Return True if the strategy is currently enabled (not auto-disabled)."""
        until = self._disabled_until.get(strategy_name, 0.0)
        if time.time() >= until:
            if until > 0:
                # Timer expired — re-enable
                self._disabled_until.pop(strategy_name, None)
                if strategy_name in self._strategies:
                    self._strategies[strategy_name].enabled = True
                    logger.info(
                        "[StrategyManager] Auto-disable expired: re-enabled {!r}", strategy_name
                    )
            return True
        return False

    def _check_auto_disable(self, strategy_name: str) -> None:
        """Disable a strategy if it meets auto-disable criteria."""
        metrics_all = self.get_rolling_metrics(strategy_name)
        consec = self._consecutive_losses.get(strategy_name, 0)
        win_rate = metrics_all.get("win_rate", 0.5)
        total = metrics_all.get("total_trades", 0)

        now = time.time()
        disable_for: Optional[float] = None

        if consec >= 5:
            disable_for = 12.0 * 3600  # 12 hours for 5 consecutive losses
        elif total >= 20 and win_rate < 0.35:
            disable_for = 24.0 * 3600  # 24 hours for low win rate

        if disable_for is not None and strategy_name in self._strategies:
            self._strategies[strategy_name].enabled = False
            self._disabled_until[strategy_name] = now + disable_for
            logger.warning(
                "[StrategyManager] Auto-disabled {!r} for {:.0f}h "
                "(consec_losses={} win_rate={:.0%})",
                strategy_name,
                disable_for / 3600,
                consec,
                win_rate,
            )

    def _get_or_raise(self, name: str) -> BaseStrategy:
        if name not in self._strategies:
            raise StrategyError(f"Strategy {name!r} not found.", strategy=name)
        return self._strategies[name]

    def _build_rl_context(
        self,
        regime: str,
        volatility_regime: str,
        market_data: Dict[str, pd.DataFrame],
        order_flow_signal: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Build context feature vector for RL strategy selection.

        Returns a dict with ~25 features as specified in the RL optimizer design.
        """
        from datetime import datetime, timezone

        context = {}

        # Market regime (one-hot encoded: 9 regimes)
        regime_options = [
            "trending_up", "trending_down", "ranging", "high_volatility",
            "low_volatility", "crash", "extreme", "unknown", "normal"
        ]
        for r in regime_options:
            context[f"regime_{r}"] = 1.0 if regime == r else 0.0

        # Volatility regime (one-hot: low/normal/high/extreme = 4)
        vol_options = ["low", "normal", "high", "extreme"]
        for v in vol_options:
            context[f"volatility_{v}"] = 1.0 if volatility_regime == v else 0.0

        # Hour of day (cyclical encoding)
        now = datetime.now(tz=timezone.utc)
        hour = now.hour
        context["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        context["hour_cos"] = math.cos(2 * math.pi * hour / 24)

        # Day of week (cyclical encoding)
        day = now.weekday()
        context["day_sin"] = math.sin(2 * math.pi * day / 7)
        context["day_cos"] = math.cos(2 * math.pi * day / 7)

        # Recent win/loss streaks (normalized 0-1)
        # Aggregate across all strategies
        total_wins = sum(p["wins"] for p in self._performance.values())
        total_losses = sum(p["losses"] for p in self._performance.values())
        total_trades = total_wins + total_losses or 1
        context["win_rate"] = total_wins / total_trades

        # Recent streaks (simplified: use global consecutive win/loss tracking)
        context["win_streak"] = min(1.0, sum(1 for p in self._performance.values() if p.get("wins", 0) > p.get("losses", 0)) / len(self._strategies))
        context["loss_streak"] = 1.0 - context["win_streak"]

        # Current drawdown percentage (0-1) - aggregate across all strategies
        total_pnl = sum(p["pnl"] for p in self._performance.values())
        context["drawdown_pct"] = max(0.0, min(1.0, -total_pnl / 10000.0 if total_pnl < 0 else 0.0))

        # BTC 24h return (normalized) - extract from market data if available
        btc_return = 0.0
        if "4h" in market_data and not market_data["4h"].empty:
            df = market_data["4h"]
            if len(df) >= 6:
                btc_return = (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6]
        context["btc_24h_return"] = max(-1.0, min(1.0, btc_return))

        # Market correlation score (0-1) - placeholder (would need multi-asset data)
        context["correlation_score"] = 0.5

        # Spread percentile (0-1) - placeholder
        context["spread_percentile"] = 0.5

        # VPIN score (0-1) - placeholder
        context["vpin_score"] = 0.5

        # Funding rate (normalized) - placeholder
        context["funding_rate"] = 0.0

        # Sentiment score (-1 to 1) - placeholder
        context["sentiment_score"] = 0.0

        # Order flow signal integration
        if order_flow_signal is not None:
            context["order_flow_strength"] = getattr(order_flow_signal, "strength", 0.5)
            context["order_flow_imbalance"] = getattr(order_flow_signal, "imbalance", 0.0)
        else:
            context["order_flow_strength"] = 0.5
            context["order_flow_imbalance"] = 0.0

        return context

    def update_rl_optimizer(
        self,
        strategy_name: str,
        trade_outcome: Dict[str, Any],
        context: Dict[str, float],
    ) -> None:
        """Update RL optimizer with trade outcome.

        Args:
            strategy_name: Name of strategy that generated the trade.
            trade_outcome: Dict with trade outcome data (pnl, position_size, etc.).
            context: Context features at trade entry.
        """
        if self._rl_optimizer is None:
            return

        # Get strategy index
        strategy_names = list(self._strategies.keys())
        if strategy_name not in strategy_names:
            logger.warning(f"Strategy {strategy_name} not found for RL update")
            return

        strategy_index = strategy_names.index(strategy_name)

        # Update RL optimizer
        try:
            self._rl_optimizer.update(
                strategy_index=strategy_index,
                trade_outcome=trade_outcome,
                context=context,
                next_context=None,  # Could compute next context in future
            )
            logger.debug(f"RL optimizer updated for {strategy_name}")
        except Exception as exc:
            logger.error(f"Failed to update RL optimizer: {exc}")

    async def validate_strategy_for_live(self, strategy_name: str) -> Tuple[bool, List[str]]:
        """Validate a strategy for live trading using walk-forward validation.
        
        Loads the last 6 months of OHLCV data for the strategy's symbols,
        runs WalkForwardValidator.validate() with the strategy's ML model (if available),
        checks that at least 4 of 6 monthly windows have positive out-of-sample returns,
        and returns (passed, failure_reasons).
        
        Args:
            strategy_name: Name of the strategy to validate.
            
        Returns:
            Tuple of (passed: bool, failure_reasons: List[str]).
        """
        failure_reasons: List[str] = []
        
        # Check if strategy exists
        if strategy_name not in self._strategies:
            failure_reasons.append(f"Strategy {strategy_name!r} not found")
            return False, failure_reasons
        
        strategy = self._strategies[strategy_name]
        
        # Check if strategy has ML model
        if not hasattr(strategy, "model") or strategy.model is None:
            # Non-ML strategies pass validation automatically
            logger.info(
                "Strategy {} has no ML model — skipping walk-forward validation",
                strategy_name,
            )
            return True, []
        
        # Import WalkForwardValidator
        try:
            from ai.prediction.walk_forward_validator import WalkForwardValidator
        except ImportError as exc:
            failure_reasons.append(f"WalkForwardValidator not available: {exc}")
            return False, failure_reasons
        
        # Get strategy symbols (use all trading pairs if strategy has no specific symbols)
        symbols = strategy.symbols if strategy.symbols else []
        if not symbols:
            # Fall back to first trading pair from settings
            try:
                from config.settings import Settings
                settings = Settings.get_settings()
                symbols = getattr(getattr(settings, "exchange", None), "trading_pairs", [])[:1]
            except Exception:
                symbols = ["BTC/USDT"]  # Ultimate fallback
        
        if not symbols:
            failure_reasons.append("No symbols configured for validation")
            return False, failure_reasons
        
        # Load 6 months of OHLCV data for the first symbol
        symbol = symbols[0]
        try:
            # Get exchange from engine (if available)
            from core.engine import TradingEngine
            engine = getattr(TradingEngine, "_instance", None)
            if engine is None or engine.exchange is None:
                failure_reasons.append("Exchange not available for data loading")
                return False, failure_reasons
            
            # Load 6 months of 1h data (approx 4320 candles)
            df = await engine.exchange.get_ohlcv(symbol, timeframe="1h", limit=4320)
            if df is None or df.empty or len(df) < 100:
                failure_reasons.append(f"Insufficient data for {symbol} (got {len(df) if df is not None else 0} candles)")
                return False, failure_reasons
            
            # Extract features and targets from OHLCV data
            # Simplified: use close prices as features, returns as targets
            import numpy as np
            closes = df["close"].values
            features = np.column_stack([
                closes[:-1],  # Previous close
                df["volume"].values[:-1],  # Previous volume
            ])
            targets = np.diff(closes) / closes[:-1]  # Returns
            timestamps = df.index.values[1:]  # Align with targets
            
            # Run walk-forward validation
            validator = WalkForwardValidator(
                training_window_days=180,  # 6 months
                validation_window_days=30,  # 1 month
                step_forward_days=30,  # 1 month step
                min_sharpe=0.3,  # Lower threshold for crypto
                min_hit_rate=0.50,  # 50% hit rate minimum
                min_profit_factor=1.1,  # Modest profit factor
            )
            
            result = await validator.validate(
                model=strategy.model,
                features=features,
                targets=targets,
                timestamps=timestamps,
            )
            
            if not result.passed:
                failure_reasons.extend(result.failure_reasons)
                logger.warning(
                    "Strategy {} failed walk-forward validation: {}",
                    strategy_name,
                    ", ".join(result.failure_reasons),
                )
                return False, failure_reasons
            
            # Check that at least 4 of 6 monthly windows have positive returns
            positive_windows = sum(
                1 for w in result.window_results
                if w.val_sharpe > 0
            )
            if positive_windows < 4:
                failure_reasons.append(
                    f"Only {positive_windows}/6 windows have positive Sharpe "
                    f"(minimum 4 required)"
                )
                return False, failure_reasons
            
            logger.info(
                "Strategy {} passed walk-forward validation: "
                "Sharpe={:.3f}±{:.3f} HitRate={:.3f} PF={:.3f} ({}/{} windows positive)",
                strategy_name,
                result.avg_val_sharpe,
                result.std_val_sharpe,
                result.avg_val_hit_rate,
                result.avg_val_profit_factor,
                positive_windows,
                result.num_windows,
            )
            return True, []
            
        except Exception as exc:
            failure_reasons.append(f"Validation error: {exc}")
            logger.error("Walk-forward validation error for {}: {}", strategy_name, exc)
            return False, failure_reasons

    def __len__(self) -> int:
        return len(self._strategies)

    def __repr__(self) -> str:
        names = list(self._strategies.keys())
        return f"StrategyManager(strategies={names})"
