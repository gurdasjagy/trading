"""Santiment on-chain analytics monitor.

Fetches development activity, network growth, social volume, and MVRV from
the Santiment GraphQL API.  Requires a Santiment API key.

API docs: https://academy.santiment.net/sanapi/
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


_GQL_QUERY = """
query {{
  getMetric(metric: "{metric}") {{
    timeseriesData(
      slug: "{slug}"
      from: "{from_dt}"
      to: "{to_dt}"
      interval: "1d"
      aggregation: LAST
    ) {{
      datetime
      value
    }}
  }}
}}
"""


class SantimentMonitor(BaseSource):
    """Fetches on-chain analytics from the Santiment API.

    Metrics monitored (daily):
    * ``dev_activity``: GitHub development activity — bullish when rising.
    * ``network_growth``: New wallet addresses — bullish when rising.
    * ``social_volume_total``: Social media mention volume.
    * ``mvrv_usd``: Market-Value-to-Realized-Value — extreme highs = bearish.
    * ``exchange_balance``: Net exchange balance change.

    Args:
        api_key: Santiment API key (required for most metrics).
        assets: List of Santiment slugs (default ``["bitcoin", "ethereum"]``).
        polling_interval: Seconds between API polls (default 3600 = 1 hour).
    """

    SANTIMENT_API = "https://api.santiment.net/graphql"

    _MVRV_HIGH = 3.0   # MVRV > 3 = overvalued
    _MVRV_LOW = 1.0    # MVRV < 1 = undervalued
    _DEV_SPIKE_MULTIPLIER = 2.0  # 2× recent avg = notable spike

    def __init__(
        self,
        api_key: str = "",
        assets: Optional[List[str]] = None,
        polling_interval: int = 3600,
    ) -> None:
        super().__init__("santiment", DataSourceType.REST_API)
        self._api_key = api_key
        self._assets = assets or ["bitcoin", "ethereum"]
        self._polling_interval = polling_interval
        self._items: List[DataItem] = []
        # Previous metric values for trend detection
        self._prev_values: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("SantimentMonitor: no API key — metrics will be empty")
        logger.info("SantimentMonitor started for assets: {}", self._assets)
        while self._running:
            try:
                new_items = await self._fetch_all()
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
            except Exception as exc:
                logger.error("SantimentMonitor error: {}", exc)
                self._errors += 1
            await asyncio.sleep(self._polling_interval)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            items = await self._fetch_all()
            self._items.extend(items)
        return self._items[-limit:]

    # ------------------------------------------------------------------
    # Fetching and analysis
    # ------------------------------------------------------------------

    async def _fetch_all(self) -> List[DataItem]:
        """Fetch all configured metrics for all assets."""
        items: List[DataItem] = []
        for slug in self._assets:
            for metric in ["dev_activity", "network_growth", "mvrv_usd", "exchange_balance"]:
                try:
                    value = await self._fetch_metric(slug, metric)
                    if value is not None:
                        new_items = self._analyse_metric(slug, metric, value)
                        items.extend(new_items)
                except Exception as exc:
                    logger.debug("Santiment {}/{}: {}", slug, metric, exc)
        self._items_collected += len(items)
        self._last_update = _utcnow()
        return items

    async def _fetch_metric(self, slug: str, metric: str) -> Optional[float]:
        """Fetch the latest daily value for *metric* and *slug*."""
        if not self._api_key:
            return None
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        from_dt = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        query = _GQL_QUERY.format(
            metric=metric, slug=slug, from_dt=from_dt, to_dt=to_dt
        )

        headers = {"Authorization": f"Apikey {self._api_key}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.SANTIMENT_API,
                    json={"query": query},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ts_data = (
                            data.get("data", {})
                            .get("getMetric", {})
                            .get("timeseriesData", [])
                        )
                        if ts_data:
                            return float(ts_data[-1].get("value", 0))
        except Exception as exc:
            logger.debug("Santiment API error for {}/{}: {}", slug, metric, exc)
        return None

    def _analyse_metric(
        self, slug: str, metric: str, value: float
    ) -> List[DataItem]:
        """Produce DataItems based on notable metric changes or levels."""
        items: List[DataItem] = []
        key = f"{slug}:{metric}"
        prev = self._prev_values.get(key, value)
        self._prev_values[key] = value

        asset_label = slug.upper()
        content = None
        sentiment_score = 0.0
        urgency = 0.4

        if metric == "mvrv_usd":
            if value > self._MVRV_HIGH:
                content = (
                    f"Santiment {asset_label}: MVRV = {value:.2f} "
                    f"(overvalued territory — historically bearish)"
                )
                sentiment_score = -0.5
                urgency = 0.7
            elif value < self._MVRV_LOW:
                content = (
                    f"Santiment {asset_label}: MVRV = {value:.2f} "
                    f"(undervalued territory — historically bullish)"
                )
                sentiment_score = 0.5
                urgency = 0.65

        elif metric == "dev_activity":
            if prev > 0 and value > prev * self._DEV_SPIKE_MULTIPLIER:
                content = (
                    f"Santiment {asset_label}: development activity spike "
                    f"({value:.1f} vs prev {prev:.1f}) — bullish signal"
                )
                sentiment_score = 0.3
                urgency = 0.5

        elif metric == "network_growth":
            if prev > 0 and value > prev * 1.5:
                content = (
                    f"Santiment {asset_label}: network growth spike "
                    f"({value:,.0f} new addresses) — adoption signal"
                )
                sentiment_score = 0.3
                urgency = 0.45

        elif metric == "exchange_balance":
            # Negative = coins leaving exchanges (bullish), positive = inflow (bearish)
            change = value - prev
            if abs(change) > abs(prev) * 0.02 and abs(change) > 0:
                direction = "outflow from" if change < 0 else "inflow to"
                sentiment_score = 0.3 if change < 0 else -0.3
                content = (
                    f"Santiment {asset_label}: significant exchange {direction} "
                    f"exchanges ({change:+,.0f}) — "
                    + ("accumulation signal" if change < 0 else "sell pressure signal")
                )
                urgency = 0.6

        if content:
            items.append(
                DataItem(
                    source_type=self.source_type,
                    source_name="santiment",
                    content=content,
                    timestamp=_utcnow(),
                    relevance_score=min(1.0, urgency + 0.1),
                    urgency_score=urgency,
                    mentioned_assets=[asset_label],
                    metadata={
                        "sentiment_score": sentiment_score,
                        "metric": metric,
                        "value": value,
                        "prev_value": prev,
                        "slug": slug,
                    },
                )
            )
        return items
