"""Data source type definitions and predefined source configurations."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DataSourceType(str, Enum):
    """Enumeration of all supported data source categories."""

    TWITTER = "twitter"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    RSS_FEED = "rss_feed"
    ONCHAIN = "onchain"
    WEBSOCKET = "websocket"
    REST_API = "rest_api"


class DataSourceConfig(BaseModel):
    """Configuration for a single external data source."""

    name: str
    source_type: DataSourceType
    enabled: bool = True
    polling_interval: int = Field(default=30, description="Seconds between polls")
    max_retries: int = 3
    timeout: int = Field(default=30, description="Request timeout in seconds")
    url: Optional[str] = None
    extra: dict = Field(default_factory=dict, description="Source-specific overrides")


# ── Twitter ───────────────────────────────────────────────────────────────────
TWITTER_SOURCE = DataSourceConfig(
    name="twitter_stream",
    source_type=DataSourceType.TWITTER,
    enabled=True,
    polling_interval=15,
    max_retries=5,
    timeout=30,
    extra={"filter_keywords": ["crypto", "bitcoin", "ethereum", "BTC", "ETH"]},
)

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_SOURCE = DataSourceConfig(
    name="telegram_channels",
    source_type=DataSourceType.TELEGRAM,
    enabled=True,
    polling_interval=10,
    max_retries=3,
    timeout=20,
)

# ── Reddit ────────────────────────────────────────────────────────────────────
REDDIT_SOURCE = DataSourceConfig(
    name="reddit_stream",
    source_type=DataSourceType.REDDIT,
    enabled=True,
    polling_interval=30,
    max_retries=3,
    timeout=30,
    extra={"sort": "new", "limit": 100},
)

# ── RSS feeds ─────────────────────────────────────────────────────────────────
COINTELEGRAPH_RSS = DataSourceConfig(
    name="cointelegraph_rss",
    source_type=DataSourceType.RSS_FEED,
    enabled=True,
    polling_interval=60,
    max_retries=3,
    timeout=15,
    url="https://cointelegraph.com/rss",
)

COINDESK_RSS = DataSourceConfig(
    name="coindesk_rss",
    source_type=DataSourceType.RSS_FEED,
    enabled=True,
    polling_interval=60,
    max_retries=3,
    timeout=15,
    url="https://coindesk.com/arc/outboundfeeds/rss/",
)

DECRYPT_RSS = DataSourceConfig(
    name="decrypt_rss",
    source_type=DataSourceType.RSS_FEED,
    enabled=True,
    polling_interval=60,
    max_retries=3,
    timeout=15,
    url="https://decrypt.co/feed",
)

THEBLOCK_RSS = DataSourceConfig(
    name="theblock_rss",
    source_type=DataSourceType.RSS_FEED,
    enabled=True,
    polling_interval=60,
    max_retries=3,
    timeout=15,
    url="https://theblock.co/rss.xml",
)

CRYPTOSLATE_RSS = DataSourceConfig(
    name="cryptoslate_rss",
    source_type=DataSourceType.RSS_FEED,
    enabled=True,
    polling_interval=60,
    max_retries=3,
    timeout=15,
    url="https://cryptoslate.com/feed/",
)

BITCOIN_MAGAZINE_RSS = DataSourceConfig(
    name="bitcoin_magazine_rss",
    source_type=DataSourceType.RSS_FEED,
    enabled=True,
    polling_interval=60,
    max_retries=3,
    timeout=15,
    url="https://bitcoinmagazine.com/feed",
)

# ── On-chain data ─────────────────────────────────────────────────────────────
WHALE_ALERT_SOURCE = DataSourceConfig(
    name="whale_alert",
    source_type=DataSourceType.ONCHAIN,
    enabled=True,
    polling_interval=30,
    max_retries=3,
    timeout=20,
    url="https://api.whale-alert.io/v1/transactions",
    extra={"min_value_usd": 500_000},
)

GLASSNODE_SOURCE = DataSourceConfig(
    name="glassnode",
    source_type=DataSourceType.REST_API,
    enabled=True,
    polling_interval=300,
    max_retries=3,
    timeout=30,
    url="https://api.glassnode.com/v1/metrics",
)

# ── Sentiment indices ─────────────────────────────────────────────────────────
FEAR_AND_GREED_SOURCE = DataSourceConfig(
    name="fear_and_greed_index",
    source_type=DataSourceType.REST_API,
    enabled=True,
    polling_interval=3600,
    max_retries=3,
    timeout=15,
    url="https://api.alternative.me/fng/",
)

# ── Exchange WebSocket feeds ──────────────────────────────────────────────────
MEXC_WS_SOURCE = DataSourceConfig(
    name="mexc_websocket",
    source_type=DataSourceType.WEBSOCKET,
    enabled=True,
    polling_interval=0,
    max_retries=5,
    timeout=10,
    url="wss://wbs.mexc.com/ws",
)

GATEIO_WS_SOURCE = DataSourceConfig(
    name="gateio_websocket",
    source_type=DataSourceType.WEBSOCKET,
    enabled=True,
    polling_interval=0,
    max_retries=5,
    timeout=10,
    url="wss://api.gateio.ws/ws/v4/",
)

# ── Aggregated list of all enabled data sources ───────────────────────────────
DATA_SOURCES: List[DataSourceConfig] = [
    TWITTER_SOURCE,
    TELEGRAM_SOURCE,
    REDDIT_SOURCE,
    COINTELEGRAPH_RSS,
    COINDESK_RSS,
    DECRYPT_RSS,
    THEBLOCK_RSS,
    CRYPTOSLATE_RSS,
    BITCOIN_MAGAZINE_RSS,
    WHALE_ALERT_SOURCE,
    GLASSNODE_SOURCE,
    FEAR_AND_GREED_SOURCE,
    MEXC_WS_SOURCE,
    GATEIO_WS_SOURCE,
]
