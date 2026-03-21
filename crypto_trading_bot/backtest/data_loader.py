"""Historical OHLCV data loader for backtesting."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

_CACHE_DIR = Path("data/backtest_cache")
_DB_PATH = _CACHE_DIR / "ohlcv.db"


class HistoricalDataLoader:
    """Loads and caches historical OHLCV data from exchange CCXT APIs."""

    def __init__(self, exchange_id: str = "binance", cache_dir: Optional[Path] = None) -> None:
        self.exchange_id = exchange_id
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for *symbol* between *start_date* and *end_date*.

        Tries cache first; fetches from exchange and caches on miss.
        """
        cached = self.load_from_cache(symbol, timeframe, start_date, end_date)
        if cached is not None and not cached.empty:
            logger.info(
                "Loaded {} bars from cache for {}/{} ({} → {})",
                len(cached),
                symbol,
                timeframe,
                start_date.date(),
                end_date.date(),
            )
            return cached

        logger.info(
            "Cache miss — fetching {}/{} from exchange ({} → {})",
            symbol,
            timeframe,
            start_date.date(),
            end_date.date(),
        )
        df = await self.download_and_cache(symbol, timeframe, start_date, end_date)
        return df

    async def fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from the exchange via CCXT."""
        try:
            import ccxt.async_support as ccxt  # type: ignore
        except ImportError as exc:
            raise ImportError("ccxt is required for exchange fetching. pip install ccxt") from exc

        exchange_cls = getattr(ccxt, self.exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown CCXT exchange: {self.exchange_id!r}")

        exchange = exchange_cls({"enableRateLimit": True})
        try:
            since_ms = int(since.timestamp() * 1000)
            until_ms = int(until.timestamp() * 1000)
            all_bars: list = []
            fetch_since = since_ms

            while fetch_since < until_ms:
                bars = await exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=1000)
                if not bars:
                    break
                bars = [b for b in bars if b[0] < until_ms]
                all_bars.extend(bars)
                if len(bars) < 1000:
                    break
                fetch_since = bars[-1][0] + 1

            if not all_bars:
                logger.warning("No OHLCV data returned for {}/{}", symbol, timeframe)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(
                all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df
        finally:
            await exchange.close()

    def load_from_cache(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        """Return cached OHLCV data or *None* if not found."""
        # Try SQLite first
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            table = self._table_name(symbol, timeframe)
            # Check table exists
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            if cur.fetchone() is None:
                conn.close()
                return None

            start_ts = start.isoformat()
            end_ts = end.isoformat()
            query = f'SELECT * FROM "{table}" WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp'  # noqa: S608
            df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
            conn.close()

            if df.empty:
                return None

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Cache read error: {}", exc)
            return None

    def save_to_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Persist OHLCV *df* to the local SQLite cache."""
        if df.empty:
            return
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            table = self._table_name(symbol, timeframe)
            df_reset = df.reset_index()
            df_reset["timestamp"] = df_reset["timestamp"].astype(str)
            df_reset.to_sql(table, conn, if_exists="replace", index=False)
            conn.close()
            logger.debug("Cached {} rows → table {!r}", len(df), table)
        except Exception as exc:
            logger.error("Failed to save cache for {}/{}: {}", symbol, timeframe, exc)

    async def download_and_cache(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch from exchange, cache, and return the OHLCV DataFrame."""
        df = await self.fetch_from_exchange(symbol, timeframe, start, end)
        self.save_to_cache(symbol, timeframe, df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Ensure the SQLite database file exists."""
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.close()
        except Exception as exc:
            logger.warning("Could not initialise cache DB: {}", exc)

    @staticmethod
    def _table_name(symbol: str, timeframe: str) -> str:
        """Derive a safe SQLite table name from symbol and timeframe."""
        import re

        safe_symbol = re.sub(r"[^A-Za-z0-9_]", "_", symbol)
        safe_tf = re.sub(r"[^A-Za-z0-9_]", "_", timeframe)
        return f"{safe_symbol}_{safe_tf}"
