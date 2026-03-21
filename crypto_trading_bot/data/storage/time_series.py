"""Time series data storage for OHLCV and indicator data."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TimeSeriesStore:
    """
    Stores and retrieves OHLCV and technical indicator time series data.
    Uses parquet files for efficient storage with pandas.
    """

    def __init__(self, base_dir: str = "data/timeseries"):
        self._base_dir = Path(base_dir)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ttl = 60  # seconds
        self._last_cache_time: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def store_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """Store OHLCV data for a symbol and timeframe."""
        if data.empty:
            return
        path = self._get_file_path(symbol, timeframe)
        data = self._normalise_df(data)
        data.to_parquet(path, index=True)
        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = data
        self._last_cache_time[cache_key] = _utcnow()
        logger.debug(f"Stored {len(data)} rows for {symbol} {timeframe}")

    async def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Load OHLCV data from storage, optionally sliced by date range."""
        cache_key = f"{symbol}_{timeframe}"
        df = self._get_from_cache(cache_key)
        if df is None:
            path = self._get_file_path(symbol, timeframe)
            if not path.exists():
                return None
            try:
                df = pd.read_parquet(path)
                self._cache[cache_key] = df
                self._last_cache_time[cache_key] = _utcnow()
            except Exception as e:
                logger.error(f"Failed to load parquet {path}: {e}")
                return None

        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df if not df.empty else None

    async def append_ohlcv(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> None:
        """Append new OHLCV rows to existing storage, deduplicating by index."""
        if new_data.empty:
            return
        existing = await self.load_ohlcv(symbol, timeframe)
        new_data = self._normalise_df(new_data)
        if existing is not None:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        else:
            combined = new_data
        await self.store_ohlcv(symbol, timeframe, combined)

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent closing price from any stored timeframe."""
        for timeframe in ("1m", "5m", "15m", "1h", "4h", "1d"):
            df = await self.load_ohlcv(symbol, timeframe)
            if df is not None and "close" in df.columns:
                return float(df["close"].iloc[-1])
        return None

    async def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Remove rows older than max_age_days from all stored files. Returns total rows removed."""
        cutoff = _utcnow() - timedelta(days=max_age_days)
        total_removed = 0
        for path in self._base_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(path)
                before = len(df)
                df = df[df.index >= cutoff]
                removed = before - len(df)
                if removed > 0:
                    df.to_parquet(path, index=True)
                    total_removed += removed
                    logger.debug(f"Removed {removed} old rows from {path.name}")
            except Exception as e:
                logger.warning(f"Cleanup error for {path}: {e}")
        # Invalidate cache
        self._cache.clear()
        self._last_cache_time.clear()
        logger.info(
            f"Cleanup complete: removed {total_removed} rows older than {max_age_days} days"
        )
        return total_removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        clean_symbol = symbol.replace("/", "_")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir / f"{clean_symbol}_{timeframe}.parquet"

    def _normalise_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column names and a DatetimeIndex."""
        df = df.copy()
        # Rename common exchange column variants
        rename_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)
            elif "date" in df.columns:
                df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.sort_index(inplace=True)
        return df

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        if cache_key not in self._cache:
            return None
        age = (_utcnow() - self._last_cache_time[cache_key]).total_seconds()
        if age > self._cache_ttl:
            del self._cache[cache_key]
            del self._last_cache_time[cache_key]
            return None
        return self._cache[cache_key]
