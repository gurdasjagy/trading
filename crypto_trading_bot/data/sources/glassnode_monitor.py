"""Glassnode on-chain analytics monitor."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GlassnodeMonitor(BaseSource):
    """Fetches on-chain analytics from Glassnode API."""

    API_BASE = "https://api.glassnode.com/v1/metrics"

    # Metric → (path, description, bullish_high)
    DEFAULT_METRICS: Dict[str, Tuple[str, str, bool]] = {
        "sopr": ("indicators/sopr", "Spent Output Profit Ratio", True),
        "nupl": ("indicators/nupl", "Net Unrealized Profit/Loss", True),
        "mvrv": ("market/mvrv", "Market-Value-to-Realized-Value", True),
        "active_addresses": ("addresses/active_count", "Active Addresses", True),
        "exchange_net_flow": (
            "transactions/transfers_volume_exchanges_net",
            "Exchange Net Flow",
            False,
        ),
    }

    def __init__(
        self,
        api_key: str = "",
        assets: Optional[List[str]] = None,
        polling_interval: int = 3600,  # Glassnode updates hourly
    ):
        super().__init__("glassnode", DataSourceType.REST_API)
        self._api_key = api_key
        self._assets = assets or ["BTC", "ETH"]
        self._polling_interval = polling_interval
        self._items: List[DataItem] = []
        self._prev_values: Dict[str, float] = {}  # "asset:metric" -> value

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("Glassnode: no API key – monitoring disabled.")
            self._running = False
            return
        logger.info("Glassnode Monitor started")
        while self._running:
            try:
                for asset in self._assets:
                    for metric_key in self.DEFAULT_METRICS:
                        item = await self.fetch_metric(asset, metric_key)
                        if item:
                            self._items.append(item)
                            self._items_collected += 1
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"Glassnode monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    async def fetch_metric(self, asset: str, metric: str) -> Optional[DataItem]:
        """Fetch a specific Glassnode metric for an asset."""
        if not self._api_key:
            return None
        metric_info = self.DEFAULT_METRICS.get(metric)
        if not metric_info:
            logger.warning(f"Glassnode: unknown metric '{metric}'")
            return None
        path, description, bullish_high = metric_info
        url = f"{self.API_BASE}/{path}"
        params = {"a": asset, "api_key": self._api_key, "i": "24h", "limit": 1}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data:
                            return None
                        latest = data[-1]
                        value = float(latest.get("v") or latest.get("value") or 0)
                        ts_raw = latest.get("t") or latest.get("timestamp", 0)
                        ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc).replace(tzinfo=None)
                        self._last_update = _utcnow()
                        return self._interpret_metric(
                            metric, value, asset, description, ts, bullish_high
                        )
                    else:
                        logger.debug(f"Glassnode API {resp.status} for {asset}/{metric}")
        except Exception as exc:
            logger.warning(f"Glassnode fetch_metric({asset}, {metric}) error: {exc}")
            self._errors += 1
        return None

    async def get_exchange_flows(self, asset: str) -> Dict[str, float]:
        """Get net exchange flows (positive = inflow, negative = outflow)."""
        if not self._api_key:
            return {}
        flows: Dict[str, float] = {}
        metrics = {
            "inflow": "transactions/transfers_volume_exchanges_inflow",
            "outflow": "transactions/transfers_volume_exchanges_outflow",
        }
        async with aiohttp.ClientSession() as session:
            for flow_type, path in metrics.items():
                params = {"a": asset, "api_key": self._api_key, "i": "24h", "limit": 1}
                try:
                    async with session.get(
                        f"{self.API_BASE}/{path}",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data:
                                flows[flow_type] = float(
                                    data[-1].get("v") or data[-1].get("value") or 0
                                )
                except Exception as exc:
                    logger.debug(f"Glassnode flow {flow_type} for {asset}: {exc}")
        if "inflow" in flows and "outflow" in flows:
            flows["net"] = flows["inflow"] - flows["outflow"]
        return flows

    async def get_realized_profit_loss(self, asset: str) -> Dict[str, float]:
        """Fetch realized profit and loss metrics for an asset."""
        if not self._api_key:
            return {}
        result: Dict[str, float] = {}
        paths = {
            "realized_profit": "indicators/realized_profit",
            "realized_loss": "indicators/realized_loss",
        }
        async with aiohttp.ClientSession() as session:
            for key, path in paths.items():
                params = {"a": asset, "api_key": self._api_key, "i": "24h", "limit": 1}
                try:
                    async with session.get(
                        f"{self.API_BASE}/{path}",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data:
                                result[key] = float(data[-1].get("v") or data[-1].get("value") or 0)
                except Exception as exc:
                    logger.debug(f"Glassnode realized P&L {key} for {asset}: {exc}")
        return result

    def _interpret_metric(
        self,
        metric: str,
        value: float,
        asset: str,
        description: str,
        timestamp: datetime,
        bullish_high: bool,
    ) -> DataItem:
        """Interpret a metric value and produce a DataItem with contextual signal."""
        prev_key = f"{asset}:{metric}"
        prev_value = self._prev_values.get(prev_key)
        self._prev_values[prev_key] = value

        # Generic interpretation: compare to prev
        if prev_value is not None and prev_value != 0:
            change_pct = ((value - prev_value) / abs(prev_value)) * 100
            trend = "rising" if change_pct > 0 else "falling"
            if bullish_high:
                signal = "bullish" if change_pct > 0 else "bearish"
            else:  # bearish when high (e.g. exchange net inflow = sell pressure)
                signal = "bearish" if change_pct > 0 else "bullish"
        else:
            trend = "stable"
            signal = "neutral"
            change_pct = 0.0

        content = (
            f"Glassnode {asset} {description}: {value:.4f} "
            f"({trend} {change_pct:+.2f}%) – Signal: {signal}"
        )
        relevance = 0.7 if signal != "neutral" else 0.4
        urgency = min(1.0, 0.2 + abs(change_pct) / 50.0) if signal != "neutral" else 0.2
        return DataItem(
            source_type=self.source_type,
            source_name=f"glassnode/{metric}",
            content=content,
            timestamp=timestamp,
            metadata={
                "asset": asset,
                "metric": metric,
                "description": description,
                "value": value,
                "prev_value": prev_value,
                "change_pct": change_pct,
                "signal": signal,
                "trend": trend,
            },
            relevance_score=relevance,
            urgency_score=urgency,
            mentioned_assets=[asset],
        )
