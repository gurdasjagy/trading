"""Enums and data-classes for news classification."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List


class NewsCategory(str, Enum):
    """Category of a news article's primary subject matter."""

    REGULATORY = "REGULATORY"
    TECHNICAL = "TECHNICAL"
    ADOPTION = "ADOPTION"
    MARKET = "MARKET"
    MACRO = "MACRO"
    SECURITY = "SECURITY"
    PARTNERSHIP = "PARTNERSHIP"
    DEVELOPMENT = "DEVELOPMENT"
    UNKNOWN = "UNKNOWN"


class ImpactLevel(str, Enum):
    """Expected market-impact severity of a news article."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOISE = "NOISE"


class Direction(str, Enum):
    """Expected price-direction implication of a news article."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TimeHorizon(str, Enum):
    """Expected time horizon over which the news will have an impact."""

    IMMEDIATE = "IMMEDIATE"  # minutes to 1 hour
    SHORT = "SHORT"  # 1–24 hours
    MEDIUM = "MEDIUM"  # 1–7 days
    LONG = "LONG"  # > 7 days


@dataclass
class NewsClassification:
    """Structured result of a news-article classification.

    Attributes:
        category:        Primary topic category.
        impact:          Expected market-impact level.
        direction:       Bullish / bearish / neutral price direction.
        affected_assets: List of ticker symbols expected to be affected.
        time_horizon:    Time horizon for the expected impact.
        confidence:      Classifier confidence in [0, 1].
        reasoning:       Optional human-readable explanation.
        classified_at:   UTC timestamp of when the classification was produced.
    """

    category: NewsCategory
    impact: ImpactLevel
    direction: Direction
    affected_assets: List[str] = field(default_factory=list)
    time_horizon: TimeHorizon = TimeHorizon.SHORT
    confidence: float = 0.5
    reasoning: str = ""
    classified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
