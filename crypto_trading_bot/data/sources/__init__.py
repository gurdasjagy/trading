"""Data sources package — exports base classes eagerly; concrete sources are lazy."""

from .base_source import BaseSource, DataItem, DataSourceType

__all__ = [
    "BaseSource",
    "BinanceFuturesSource",
    "CoinGeckoSource",
    "CryptoPanicSource",
    "DataItem",
    "DataSourceType",
    "DeribitOptionsSource",
    "ExchangeFlowMonitor",
    "FearGreedMonitor",
    "FundingRateMonitor",
    "NewsRSSMonitor",
    "RedditMonitor",
    "SantimentMonitor",
    "TokenUnlocksMonitor",
]


def __getattr__(name: str):  # noqa: N807
    """Lazily import concrete source classes on first access."""
    _lazy_map = {
        "BinanceFuturesSource": (".binance_futures", "BinanceFuturesSource"),
        "CoinGeckoSource": (".coingecko_source", "CoinGeckoSource"),
        "CryptoPanicSource": (".cryptopanic_source", "CryptoPanicSource"),
        "DeribitOptionsSource": (".deribit_options", "DeribitOptionsSource"),
        "ExchangeFlowMonitor": (".exchange_flow", "ExchangeFlowMonitor"),
        "FearGreedMonitor": (".fear_greed_monitor", "FearGreedMonitor"),
        "FundingRateMonitor": (".funding_rate_monitor", "FundingRateMonitor"),
        "NewsRSSMonitor": (".news_rss_monitor", "NewsRSSMonitor"),
        "RedditMonitor": (".reddit_monitor", "RedditMonitor"),
        "SantimentMonitor": (".santiment_monitor", "SantimentMonitor"),
        "TokenUnlocksMonitor": (".token_unlocks", "TokenUnlocksMonitor"),
    }
    if name in _lazy_map:
        module_path, class_name = _lazy_map[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
