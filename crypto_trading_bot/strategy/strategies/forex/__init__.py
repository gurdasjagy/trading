"""Gold/Silver-specific forex trading strategies."""

from strategy.strategies.forex.gold_adx_trend import GoldAdxTrendStrategy
from strategy.strategies.forex.gold_asian_range import GoldAsianRangeStrategy
from strategy.strategies.forex.gold_atr_breakout import GoldAtrBreakoutStrategy
from strategy.strategies.forex.gold_donchian_channel import GoldDonchianChannelStrategy
from strategy.strategies.forex.gold_ema_ribbon import GoldEmaRibbonStrategy
from strategy.strategies.forex.gold_keltner_channel import GoldKeltnerChannelStrategy
from strategy.strategies.forex.gold_liquidity_sweep import GoldLiquiditySweepStrategy
from strategy.strategies.forex.gold_london_fix import GoldLondonFixStrategy
from strategy.strategies.forex.gold_momentum_breakout import GoldMomentumBreakoutStrategy
from strategy.strategies.forex.gold_multi_timeframe_confluence import (
    GoldMultiTimeframeConfluenceStrategy,
)
from strategy.strategies.forex.gold_ny_open_reversal import GoldNyOpenReversalStrategy
from strategy.strategies.forex.gold_order_block import GoldOrderBlockStrategy
from strategy.strategies.forex.gold_parabolic_sar import GoldParabolicSarStrategy
from strategy.strategies.forex.gold_range_expansion import GoldRangeExpansionStrategy
from strategy.strategies.forex.gold_stochastic_oversold import GoldStochasticOversoldStrategy
from strategy.strategies.forex.gold_supertrend import GoldSupertrendStrategy
from strategy.strategies.forex.gold_volatility_squeeze import GoldVolatilitySqueezeStrategy
from strategy.strategies.forex.gold_williams_r import GoldWilliamsRStrategy

__all__ = [
    "GoldAdxTrendStrategy",
    "GoldAsianRangeStrategy",
    "GoldAtrBreakoutStrategy",
    "GoldDonchianChannelStrategy",
    "GoldEmaRibbonStrategy",
    "GoldKeltnerChannelStrategy",
    "GoldLiquiditySweepStrategy",
    "GoldLondonFixStrategy",
    "GoldMomentumBreakoutStrategy",
    "GoldMultiTimeframeConfluenceStrategy",
    "GoldNyOpenReversalStrategy",
    "GoldOrderBlockStrategy",
    "GoldParabolicSarStrategy",
    "GoldRangeExpansionStrategy",
    "GoldStochasticOversoldStrategy",
    "GoldSupertrendStrategy",
    "GoldVolatilitySqueezeStrategy",
    "GoldWilliamsRStrategy",
]
