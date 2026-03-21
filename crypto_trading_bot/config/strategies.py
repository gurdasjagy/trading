"""Strategy parameter models and registry."""

from __future__ import annotations

from typing import Dict, List, Type

from pydantic import BaseModel, Field


class StrategyParams(BaseModel):
    """Base strategy configuration shared by all strategies."""

    name: str
    enabled: bool = True
    max_position_size_pct: float = 10.0
    timeframes: List[str] = Field(default=["1m", "5m", "15m"])


class NewsMomentumParams(StrategyParams):
    """Parameters for the news-driven momentum strategy."""

    name: str = "news_momentum"
    timeframes: List[str] = Field(default=["1m", "5m"])
    min_news_impact: str = "HIGH"
    momentum_window_minutes: int = 5
    volume_spike_threshold: float = 2.0
    max_news_age_minutes: int = 15


class FundingRateArbParams(StrategyParams):
    """Parameters for the funding-rate arbitrage strategy."""

    name: str = "funding_rate_arb"
    timeframes: List[str] = Field(default=["1h", "4h"])
    min_funding_rate: float = 0.0005
    max_funding_rate: float = -0.0005
    arb_threshold: float = 0.0003


class LiquidationHunterParams(StrategyParams):
    """Parameters for the liquidation-cascade hunting strategy."""

    name: str = "liquidation_hunter"
    timeframes: List[str] = Field(default=["1m", "3m", "5m"])
    min_liquidation_volume_usd: float = 50_000_000.0
    cascade_window_minutes: int = 60
    rsi_extreme_threshold: int = 20


class TechnicalBreakoutParams(StrategyParams):
    """Parameters for the technical-breakout strategy."""

    name: str = "technical_breakout"
    timeframes: List[str] = Field(default=["15m", "1h", "4h"])
    lookback_periods: int = 20
    volume_confirmation_multiplier: float = 1.5
    min_adx: int = 25


class GridTradingParams(StrategyParams):
    """Parameters for the grid-trading strategy."""

    name: str = "grid_trading"
    timeframes: List[str] = Field(default=["1m", "5m"])
    grid_levels: int = 10
    grid_spacing_pct: float = 0.5
    max_grids: int = 20


class DCAStrategyParams(StrategyParams):
    """Parameters for the dollar-cost averaging strategy."""

    name: str = "dca"
    timeframes: List[str] = Field(default=["1d"])
    base_amount_usd: float = 100.0
    multiplier: float = 2.0
    max_levels: int = 5
    interval_hours: int = 24


class ScalpingParams(StrategyParams):
    """Parameters for the high-frequency scalping strategy."""

    name: str = "scalping"
    timeframes: List[str] = Field(default=["1m"])
    min_spread_pct: float = 0.05
    max_hold_seconds: int = 120
    target_profit_pct: float = 0.1


class AIAdaptiveParams(StrategyParams):
    """Parameters for the AI-driven adaptive strategy."""

    name: str = "ai_adaptive"
    timeframes: List[str] = Field(default=["5m", "15m", "1h"])
    min_confidence: float = 0.7
    use_llm_reasoning: bool = True
    update_interval_seconds: int = 30


# ── New strategy parameter classes ───────────────────────────────────────────


class BollingerSqueezeParams(StrategyParams):
    """Parameters for the Bollinger Band squeeze breakout strategy."""

    name: str = "bollinger_squeeze"
    timeframes: List[str] = Field(default=["1h", "4h"])
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_ma: int = 20


class VWAPDeviationParams(StrategyParams):
    """Parameters for the VWAP deviation mean-reversion strategy."""

    name: str = "vwap_deviation"
    timeframes: List[str] = Field(default=["15m", "1h"])
    deviation_threshold: float = 2.0


class IchimokuCloudParams(StrategyParams):
    """Parameters for the Ichimoku Cloud strategy."""

    name: str = "ichimoku_cloud"
    timeframes: List[str] = Field(default=["1h", "4h"])
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52


class FibonacciRetracementParams(StrategyParams):
    """Parameters for the Fibonacci retracement bounce strategy."""

    name: str = "fibonacci_retracement"
    timeframes: List[str] = Field(default=["1h", "4h"])
    swing_lookback: int = 50
    tolerance_pct: float = 0.5


class OrderFlowImbalanceParams(StrategyParams):
    """Parameters for the order-flow imbalance strategy."""

    name: str = "order_flow_imbalance"
    timeframes: List[str] = Field(default=["5m", "15m"])
    imbalance_threshold: float = 0.6


class VolumeProfileParams(StrategyParams):
    """Parameters for the volume-profile strategy."""

    name: str = "volume_profile"
    timeframes: List[str] = Field(default=["1h", "4h"])
    num_bins: int = 20


class RSIDivergenceParams(StrategyParams):
    """Parameters for the RSI divergence strategy."""

    name: str = "rsi_divergence"
    timeframes: List[str] = Field(default=["1h", "4h"])
    rsi_period: int = 14
    lookback: int = 20


class MACDCrossoverParams(StrategyParams):
    """Parameters for the MACD crossover strategy."""

    name: str = "macd_crossover"
    timeframes: List[str] = Field(default=["1h", "4h"])
    fast: int = 12
    slow: int = 26
    signal: int = 9


class EMARibbonParams(StrategyParams):
    """Parameters for the EMA ribbon strategy."""

    name: str = "ema_ribbon"
    timeframes: List[str] = Field(default=["1h", "4h"])
    ema_periods: List[int] = Field(default=[8, 13, 21, 34, 55, 89])


class SupertrendParams(StrategyParams):
    """Parameters for the Supertrend strategy."""

    name: str = "supertrend"
    timeframes: List[str] = Field(default=["1h", "4h"])
    atr_period: int = 10
    multiplier: float = 3.0


class KeltnerChannelParams(StrategyParams):
    """Parameters for the Keltner Channel breakout strategy."""

    name: str = "keltner_channel"
    timeframes: List[str] = Field(default=["1h", "4h"])
    ema_period: int = 20
    atr_multiplier: float = 2.0


class DonchianBreakoutParams(StrategyParams):
    """Parameters for the Donchian Channel breakout strategy."""

    name: str = "donchian_breakout"
    timeframes: List[str] = Field(default=["4h", "1d"])
    entry_period: int = 20
    exit_period: int = 10


class ParabolicSARParams(StrategyParams):
    """Parameters for the Parabolic SAR strategy."""

    name: str = "parabolic_sar"
    timeframes: List[str] = Field(default=["1h", "4h"])
    step: float = 0.02
    max_step: float = 0.2


class StochasticRSIParams(StrategyParams):
    """Parameters for the Stochastic RSI strategy."""

    name: str = "stochastic_rsi"
    timeframes: List[str] = Field(default=["1h", "4h"])
    rsi_period: int = 14
    stoch_period: int = 14
    smooth_k: int = 3
    smooth_d: int = 3


class WilliamsRParams(StrategyParams):
    """Parameters for the Williams %R strategy."""

    name: str = "williams_r"
    timeframes: List[str] = Field(default=["1h", "4h"])
    period: int = 14
    ob_threshold: float = -20.0
    os_threshold: float = -80.0


class ADXTrendParams(StrategyParams):
    """Parameters for the ADX trend-confirmation strategy."""

    name: str = "adx_trend"
    timeframes: List[str] = Field(default=["1h", "4h"])
    adx_period: int = 14
    min_adx: float = 25.0


class PivotPointParams(StrategyParams):
    """Parameters for the pivot-point bounce strategy."""

    name: str = "pivot_point"
    timeframes: List[str] = Field(default=["1h", "4h"])
    tolerance_pct: float = 0.3


class HarmonicPatternParams(StrategyParams):
    """Parameters for the harmonic pattern strategy."""

    name: str = "harmonic_pattern"
    timeframes: List[str] = Field(default=["4h", "1d"])
    tolerance: float = 0.05


class ElliottWaveParams(StrategyParams):
    """Parameters for the Elliott Wave strategy."""

    name: str = "elliott_wave"
    timeframes: List[str] = Field(default=["4h", "1d"])
    min_wave_pct: float = 0.03


class SupplyDemandZoneParams(StrategyParams):
    """Parameters for the supply and demand zone strategy."""

    name: str = "supply_demand_zone"
    timeframes: List[str] = Field(default=["1h", "4h"])
    zone_lookback: int = 50
    min_move_pct: float = 0.02


class MarketStructureBreakParams(StrategyParams):
    """Parameters for the market structure break strategy."""

    name: str = "market_structure_break"
    timeframes: List[str] = Field(default=["1h", "4h"])
    swing_lookback: int = 20


class FairValueGapParams(StrategyParams):
    """Parameters for the fair value gap strategy."""

    name: str = "fair_value_gap"
    timeframes: List[str] = Field(default=["15m", "1h"])
    min_gap_pct: float = 0.003


class OrderBlockParams(StrategyParams):
    """Parameters for the order block strategy."""

    name: str = "order_block"
    timeframes: List[str] = Field(default=["1h", "4h"])
    lookback: int = 30


class AccDistParams(StrategyParams):
    """Parameters for the accumulation/distribution strategy."""

    name: str = "accumulation_distribution"
    timeframes: List[str] = Field(default=["1h", "4h"])
    ema_period: int = 20


class OnChainMomentumParams(StrategyParams):
    """Parameters for the on-chain momentum strategy."""

    name: str = "on_chain_momentum"
    timeframes: List[str] = Field(default=["1h", "4h"])
    volume_spike_multiplier: float = 2.0
    lookback: int = 20


class CorrelationDivergenceParams(StrategyParams):
    """Parameters for the correlation divergence strategy."""

    name: str = "correlation_divergence"
    timeframes: List[str] = Field(default=["1h", "4h"])
    corr_window: int = 20
    divergence_threshold: float = 0.3


class VolatilityBreakoutParams(StrategyParams):
    """Parameters for the volatility breakout strategy."""

    name: str = "volatility_breakout"
    timeframes: List[str] = Field(default=["1h", "4h"])
    atr_period: int = 14
    atr_ma_period: int = 20
    expansion_threshold: float = 1.5


class TimeBasedParams(StrategyParams):
    """Parameters for the time-of-day session strategy."""

    name: str = "time_based"
    timeframes: List[str] = Field(default=["1h"])
    active_sessions: List[str] = Field(default=["london", "new_york"])


class MTFConfluenceParams(StrategyParams):
    """Parameters for the multi-timeframe confluence strategy."""

    name: str = "mtf_confluence"
    timeframes: List[str] = Field(default=["15m", "1h", "4h"])
    min_confluence: int = 3


class RangeBreakoutParams(StrategyParams):
    """Parameters for the range breakout strategy."""

    name: str = "range_breakout"
    timeframes: List[str] = Field(default=["1h", "4h"])
    range_lookback: int = 20
    breakout_multiplier: float = 1.0


class MomentumDivergenceParams(StrategyParams):
    """Parameters for the momentum divergence strategy."""

    name: str = "momentum_divergence"
    timeframes: List[str] = Field(default=["1h", "4h"])
    lookback: int = 20


# Registry mapping strategy name → param class for dynamic discovery.
STRATEGY_REGISTRY: Dict[str, Type[StrategyParams]] = {
    # Original strategies
    "news_momentum": NewsMomentumParams,
    "funding_rate_arb": FundingRateArbParams,
    "liquidation_hunter": LiquidationHunterParams,
    "technical_breakout": TechnicalBreakoutParams,
    "grid_trading": GridTradingParams,
    "dca": DCAStrategyParams,
    "scalping": ScalpingParams,
    "ai_adaptive": AIAdaptiveParams,
    # New strategies
    "bollinger_squeeze": BollingerSqueezeParams,
    "vwap_deviation": VWAPDeviationParams,
    "ichimoku_cloud": IchimokuCloudParams,
    "fibonacci_retracement": FibonacciRetracementParams,
    "order_flow_imbalance": OrderFlowImbalanceParams,
    "volume_profile": VolumeProfileParams,
    "rsi_divergence": RSIDivergenceParams,
    "macd_crossover": MACDCrossoverParams,
    "ema_ribbon": EMARibbonParams,
    "supertrend": SupertrendParams,
    "keltner_channel": KeltnerChannelParams,
    "donchian_breakout": DonchianBreakoutParams,
    "parabolic_sar": ParabolicSARParams,
    "stochastic_rsi": StochasticRSIParams,
    "williams_r": WilliamsRParams,
    "adx_trend": ADXTrendParams,
    "pivot_point": PivotPointParams,
    "harmonic_pattern": HarmonicPatternParams,
    "elliott_wave": ElliottWaveParams,
    "supply_demand_zone": SupplyDemandZoneParams,
    "market_structure_break": MarketStructureBreakParams,
    "fair_value_gap": FairValueGapParams,
    "order_block": OrderBlockParams,
    "accumulation_distribution": AccDistParams,
    "on_chain_momentum": OnChainMomentumParams,
    "correlation_divergence": CorrelationDivergenceParams,
    "volatility_breakout": VolatilityBreakoutParams,
    "time_based": TimeBasedParams,
    "mtf_confluence": MTFConfluenceParams,
    "range_breakout": RangeBreakoutParams,
    "momentum_divergence": MomentumDivergenceParams,
}


def get_strategy_params(strategy_name: str, **overrides) -> StrategyParams:
    """Instantiate strategy params by name, optionally overriding defaults.

    Args:
        strategy_name: Key in ``STRATEGY_REGISTRY``.
        **overrides: Field overrides forwarded to the param constructor.

    Returns:
        A populated :class:`StrategyParams` subclass instance.

    Raises:
        ValueError: If *strategy_name* is not registered.
    """
    name = strategy_name.lower()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. " f"Available: {list(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name](**overrides)
