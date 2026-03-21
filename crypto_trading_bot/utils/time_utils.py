"""Timezone and time-related utilities for the trading bot."""

import time
from datetime import datetime, timedelta, timezone

# Timeframe string → seconds
_TIMEFRAME_SECONDS: dict = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}

# Timeframe string → pandas frequency string
_TIMEFRAME_PANDAS_FREQ: dict = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W",
}


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


def timestamp_ms() -> int:
    """Return the current UTC time in milliseconds."""
    return int(time.time() * 1000)


def from_timestamp_ms(ts: int) -> datetime:
    """Convert a millisecond UTC timestamp to a timezone-aware datetime."""
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)


def to_timestamp_ms(dt: datetime) -> int:
    """Convert a datetime to a millisecond UTC timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Examples:
        3725 → "1h 2m 5s"
        90   → "1m 30s"
        45   → "45s"
    """
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def is_market_hours() -> bool:
    """Crypto markets are open 24/7 — always returns True."""
    return True


def time_until_next_hour() -> float:
    """Return the number of seconds until the start of the next UTC hour."""
    now = utcnow()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return (next_hour - now).total_seconds()


def time_until_midnight_utc() -> float:
    """Return the number of seconds until the next UTC midnight."""
    now = utcnow()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (tomorrow - now).total_seconds()


def parse_timeframe(timeframe: str) -> int:
    """
    Convert a timeframe string to seconds.

    Examples:
        "1m"  → 60
        "1h"  → 3600
        "1d"  → 86400

    Raises:
        ValueError: for unrecognised timeframe strings.
    """
    try:
        return _TIMEFRAME_SECONDS[timeframe.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. " f"Supported: {list(_TIMEFRAME_SECONDS)}"
        )


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """
    Convert a timeframe string to a pandas frequency string.

    Examples:
        "1m" → "1min"
        "1h" → "1H"
        "1d" → "1D"

    Raises:
        ValueError: for unrecognised timeframe strings.
    """
    try:
        return _TIMEFRAME_PANDAS_FREQ[timeframe.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. " f"Supported: {list(_TIMEFRAME_PANDAS_FREQ)}"
        )


def round_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """
    Floor *dt* to the start of the given timeframe period.

    Examples:
        round_to_timeframe(datetime(2024,1,1,13,45), "1h") → datetime(2024,1,1,13,0)
    """
    seconds = parse_timeframe(timeframe)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = int(dt.timestamp())
    floored_ts = (ts // seconds) * seconds
    return datetime.fromtimestamp(floored_ts, tz=timezone.utc)
