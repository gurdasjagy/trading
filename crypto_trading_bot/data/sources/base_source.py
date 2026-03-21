"""Abstract base class for all data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class DataSourceType(str, Enum):
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    RSS_FEED = "rss_feed"
    ONCHAIN = "onchain"
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    DISCORD = "discord"


class DataItem(BaseModel):
    """Standardized data item from any source."""

    source_type: DataSourceType
    source_name: str
    content: str
    url: Optional[str] = None
    author: Optional[str] = None
    timestamp: datetime
    raw_data: Optional[Dict] = None
    metadata: Optional[Dict] = None
    relevance_score: float = 0.5  # 0-1
    urgency_score: float = 0.5  # 0-1
    mentioned_assets: List[str] = []


class BaseSource(ABC):
    """Abstract base class for all data sources."""

    # Common crypto asset keyword mapping
    _ASSET_KEYWORDS: Dict[str, List[str]] = {
        "BTC": ["bitcoin", "btc", "$btc"],
        "ETH": ["ethereum", "eth", "ether", "$eth"],
        "SOL": ["solana", "sol", "$sol"],
        "BNB": ["bnb", "binance coin"],
        "XRP": ["ripple", "xrp", "$xrp"],
        "ADA": ["cardano", "ada", "$ada"],
        "DOGE": ["dogecoin", "doge", "$doge"],
        "AVAX": ["avalanche", "avax", "$avax"],
        "LINK": ["chainlink", "link", "$link"],
        "DOT": ["polkadot", "dot", "$dot"],
        "MATIC": ["polygon", "matic", "$matic"],
        "NEAR": ["near protocol", "near", "$near"],
        "ARB": ["arbitrum", "arb", "$arb"],
    }

    _URGENT_KEYWORDS = ["breaking", "urgent", "alert", "crash", "pump", "dump", "hack", "exploit"]

    def __init__(self, name: str, source_type: DataSourceType, enabled: bool = True):
        self.name = name
        self.source_type = source_type
        self.enabled = enabled
        self._running = False
        self._items_collected = 0
        self._errors = 0
        self._last_update: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Begin data collection."""
        ...

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop data collection."""
        ...

    @abstractmethod
    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        """Fetch latest items from this source."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _extract_mentioned_assets(self, text: str) -> List[str]:
        """Extract mentioned crypto assets from text."""
        text_lower = text.lower()
        return [
            asset
            for asset, keywords in self._ASSET_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]

    def _calculate_urgency(self, text: str, author_influence: float = 0.5) -> float:
        """Calculate urgency score based on content and author influence."""
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in self._URGENT_KEYWORDS if kw in text_lower)
        urgency = min(1.0, 0.3 + (keyword_hits * 0.1) + (author_influence * 0.3))
        return urgency

    @property
    def status(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "running": self._running,
            "items_collected": self._items_collected,
            "errors": self._errors,
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }
