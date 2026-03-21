"""Event-driven pub/sub system for the trading bot."""

import inspect
import threading
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List

from loguru import logger


class EventType(str, Enum):
    NEWS_RECEIVED = "news_received"
    SENTIMENT_UPDATE = "sentiment_update"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    PRICE_UPDATE = "price_update"
    RISK_ALERT = "risk_alert"
    CIRCUIT_BREAKER = "circuit_breaker"
    DAILY_TARGET_HIT = "daily_target_hit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MARKET_REGIME_CHANGE = "market_regime_change"
    WHALE_ALERT = "whale_alert"
    FUNDING_RATE_ALERT = "funding_rate_alert"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    SYSTEM_HEALTH = "system_health"
    POSITION_UPDATE = "position_update"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"


class EventBus:
    """Thread-safe, async-compatible publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Register a callback for the given event type."""
        with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed {callback.__name__!r} to {event_type.value!r}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed {callback.__name__!r} from {event_type.value!r}")
            except ValueError:
                logger.warning(f"Callback {callback.__name__!r} not found for {event_type.value!r}")

    def publish(self, event_type: EventType, data: Any = None) -> None:
        """Publish an event synchronously. Only invokes synchronous callbacks."""
        with self._lock:
            callbacks = list(self._subscribers[event_type])

        for callback in callbacks:
            if inspect.iscoroutinefunction(callback):
                logger.warning(
                    f"Skipping async callback {callback.__name__!r} in synchronous publish — "
                    "use publish_async() instead."
                )
                continue
            try:
                callback(event_type, data)
            except Exception as exc:
                logger.error(
                    f"Error in sync callback {callback.__name__!r} "
                    f"for event {event_type.value!r}: {exc}"
                )

    async def publish_async(self, event_type: EventType, data: Any = None) -> None:
        """Publish an event asynchronously, running both sync and async callbacks."""
        with self._lock:
            callbacks = list(self._subscribers[event_type])

        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as exc:
                logger.error(
                    f"Error in callback {callback.__name__!r} "
                    f"for event {event_type.value!r}: {exc}"
                )

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Return the number of subscribers for an event type."""
        with self._lock:
            return len(self._subscribers[event_type])

    def clear_subscribers(self) -> None:
        """Remove all subscribers from all event types."""
        with self._lock:
            self._subscribers.clear()
        logger.debug("All event bus subscribers cleared.")
