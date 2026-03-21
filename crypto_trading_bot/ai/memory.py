"""AI memory system for maintaining market and trade context across decisions."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel


class MemoryItem(BaseModel):
    """A single memory entry with optional expiry and relevance metadata."""

    context_type: str
    content: str
    relevance_score: float = 0.5
    timestamp: datetime = None  # type: ignore[assignment]
    expires_at: Optional[datetime] = None
    symbol: Optional[str] = None
    tags: List[str] = []

    def __init__(self, **data) -> None:  # type: ignore[override]
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(tz=timezone.utc)
        super().__init__(**data)


class AIMemory:
    """Manages AI memory for maintaining context across trading decisions.

    Stores time-bounded memory items and retrieves them filtered by type,
    symbol, and minimum relevance.  Expired items are removed automatically.
    When the store exceeds *max_items*, the least-relevant entries are evicted.
    """

    def __init__(
        self,
        max_items: int = 100,
        default_ttl_hours: int = 24,
    ) -> None:
        self._items: List[MemoryItem] = []
        self._max_items = max_items
        self._default_ttl = timedelta(hours=default_ttl_hours)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        context_type: str,
        content: str,
        symbol: Optional[str] = None,
        relevance: float = 0.5,
        ttl_hours: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a memory item with an optional TTL.

        Args:
            context_type: Logical category (e.g. "trade_signal", "news").
            content:       Human-readable content to remember.
            symbol:        Optional asset symbol this memory relates to.
            relevance:     0–1 score; lower-relevance items are evicted first.
            ttl_hours:     Override the default TTL (hours).
            tags:          Arbitrary searchable tags.
        """
        expires_at = datetime.now(tz=timezone.utc) + timedelta(
            hours=ttl_hours or self._default_ttl.seconds // 3600 or 24
        )
        item = MemoryItem(
            context_type=context_type,
            content=content,
            relevance_score=relevance,
            expires_at=expires_at,
            symbol=symbol,
            tags=tags or [],
        )
        self._items.append(item)
        self._cleanup()
        logger.debug(
            f"AIMemory stored [{context_type}] for {symbol or 'global'} (total={len(self._items)})"
        )

    def retrieve(
        self,
        context_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> List[MemoryItem]:
        """Retrieve memory items, most relevant and recent first.

        Args:
            context_type:   Filter by context type; ``None`` matches all.
            symbol:         Filter by symbol; ``None`` matches all (including global).
            limit:          Maximum number of items to return.
            min_relevance:  Exclude items with relevance below this threshold.

        Returns:
            List of matching :class:`MemoryItem` objects, sorted by
            (relevance, timestamp) descending.
        """
        self._cleanup()
        filtered = [
            item
            for item in self._items
            if (context_type is None or item.context_type == context_type)
            and (symbol is None or item.symbol == symbol or item.symbol is None)
            and item.relevance_score >= min_relevance
        ]
        filtered.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        return filtered[:limit]

    def get_context_for_symbol(self, symbol: str, max_chars: int = 2000) -> str:
        """Return a formatted context string for *symbol*, capped at *max_chars*.

        Useful for injecting into LLM prompts without exceeding context limits.
        """
        items = self.retrieve(symbol=symbol, limit=5, min_relevance=0.3)
        if not items:
            return "No recent context available."

        context_parts: List[str] = []
        total_chars = 0
        for item in items:
            part = f"[{item.context_type} - {item.timestamp.strftime('%H:%M')}]: {item.content}"
            if total_chars + len(part) > max_chars:
                break
            context_parts.append(part)
            total_chars += len(part)

        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Purge expired items and enforce the max-size limit."""
        now = datetime.now(tz=timezone.utc)
        self._items = [
            item for item in self._items if item.expires_at is None or item.expires_at > now
        ]
        if len(self._items) > self._max_items:
            # Evict least relevant items first
            self._items.sort(key=lambda x: x.relevance_score)
            self._items = self._items[-self._max_items :]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def item_count(self) -> int:
        """Number of live (non-expired) items currently in memory."""
        return len(self._items)
