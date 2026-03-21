"""Config package — exports all public configuration symbols."""

from config.data_sources import (
    DATA_SOURCES,
    DataSourceConfig,
    DataSourceType,
)
from config.exchanges import (
    SUPPORTED_EXCHANGES,
    BingXConfig,
    BitgetConfig,
    ExchangeAPIConfig,
    GateIOConfig,
    MEXCConfig,
    get_exchange_config,
)
from config.gold_config import (
    GOLD_FUTURES_CONFIG,
    GOLD_TRADING_KNOWLEDGE,
)
from config.logging_config import configure_logging
from config.risk_profiles import (
    AGGRESSIVE_PROFILE,
    CONSERVATIVE_PROFILE,
    MODERATE_PROFILE,
    RiskProfile,
    RiskProfileConfig,
    get_risk_profile,
)
from config.settings import (
    AIConfig,
    ExchangeConfig,
    ForexConfig,
    MonitoringConfig,
    RiskConfig,
    Settings,
)
from config.settings import (
    DataSourceConfig as DataConfig,
)
from config.strategies import (
    STRATEGY_REGISTRY,
    AIAdaptiveParams,
    DCAStrategyParams,
    FundingRateArbParams,
    GridTradingParams,
    LiquidationHunterParams,
    NewsMomentumParams,
    ScalpingParams,
    StrategyParams,
    TechnicalBreakoutParams,
    get_strategy_params,
)

__all__ = [
    # settings
    "Settings",
    "ExchangeConfig",
    "AIConfig",
    "DataConfig",
    "RiskConfig",
    "MonitoringConfig",
    "ForexConfig",
    # exchanges
    "ExchangeAPIConfig",
    "MEXCConfig",
    "GateIOConfig",
    "BingXConfig",
    "BitgetConfig",
    "SUPPORTED_EXCHANGES",
    "get_exchange_config",
    # risk profiles
    "RiskProfile",
    "RiskProfileConfig",
    "CONSERVATIVE_PROFILE",
    "MODERATE_PROFILE",
    "AGGRESSIVE_PROFILE",
    "get_risk_profile",
    # strategies
    "StrategyParams",
    "NewsMomentumParams",
    "FundingRateArbParams",
    "LiquidationHunterParams",
    "TechnicalBreakoutParams",
    "GridTradingParams",
    "DCAStrategyParams",
    "ScalpingParams",
    "AIAdaptiveParams",
    "STRATEGY_REGISTRY",
    "get_strategy_params",
    # data sources
    "DataSourceType",
    "DataSourceConfig",
    "DATA_SOURCES",
    # gold futures
    "GOLD_FUTURES_CONFIG",
    "GOLD_TRADING_KNOWLEDGE",
    # logging
    "configure_logging",
]
