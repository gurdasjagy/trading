"""Whale Alert Source — monitors large BTC/ETH/SOL transfers."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class WhaleAlertSource(BaseSource):
    """Monitors large BTC/ETH/SOL transfers (>$1M) from whale-alert.io API."""

    WHALE_ALERT_API = "https://api.whale-alert.io/v1"
    WHALE_THRESHOLD_USD = 1_000_000  # $1M minimum

    def __init__(
        self,
        api_key: str = "",
        polling_interval: int = 300,  # 5 minutes
    ):
        super().__init__("whale_alert", DataSourceType.REST_API)
        self._api_key = api_key
        self._polling_interval = polling_interval
        self._seen_tx_ids: set = set()
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._api_key:
            logger.warning("WhaleAlert: no API key provided – monitoring disabled.")
            self._running = False
            return
        logger.info("Whale Alert Source started")
        while self._running:
            try:
                new_items = await self._fetch_whale_transfers()
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"WhaleAlert monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_whale_transfers()
        return self._items[-limit:]

    async def _fetch_whale_transfers(self) -> List[DataItem]:
        """Fetch large transfers from Whale Alert API."""
        items: List[DataItem] = []
        if not self._api_key:
            return items

        try:
            params = {
                "api_key": self._api_key,
                "min_value": self.WHALE_THRESHOLD_USD,
                "limit": 100,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.WHALE_ALERT_API}/transactions",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for tx in data.get("transactions", []):
                            tx_id = tx.get("hash") or tx.get("id", "")
                            if tx_id in self._seen_tx_ids:
                                continue
                            self._seen_tx_ids.add(tx_id)

                            # Filter for BTC/ETH/SOL only
                            symbol = (tx.get("symbol") or "").upper()
                            if symbol not in ["BTC", "ETH", "SOL"]:
                                continue

                            item = self._create_data_item(tx)
                            if item:
                                items.append(item)
                                self._items_collected += 1
                    else:
                        logger.warning(f"Whale Alert API status {resp.status}")
        except Exception as exc:
            logger.warning(f"Whale Alert fetch error: {exc}")
            self._errors += 1

        self._last_update = _utcnow()
        return items

    def _create_data_item(self, tx: dict) -> Optional[DataItem]:
        """Create a DataItem from a whale transaction."""
        try:
            symbol = (tx.get("symbol") or "").upper()
            amount = float(tx.get("amount") or 0)
            amount_usd = float(tx.get("amount_usd") or 0)
            from_owner = tx.get("from", {}).get("owner_type", "unknown")
            to_owner = tx.get("to", {}).get("owner_type", "unknown")
            tx_hash = tx.get("hash") or tx.get("id", "")
            ts_raw = tx.get("timestamp", 0)
            ts = (
                datetime.fromtimestamp(ts_raw, tz=timezone.utc).replace(tzinfo=None)
                if ts_raw
                else _utcnow()
            )

            # Classify transfer type
            is_to_exchange = to_owner in ("exchange",)
            is_from_exchange = from_owner in ("exchange",)
            if is_to_exchange:
                transfer_type = "exchange_inflow"
                signal_hint = "potential_sell_pressure"
            elif is_from_exchange:
                transfer_type = "exchange_outflow"
                signal_hint = "potential_accumulation"
            else:
                transfer_type = "whale_move"
                signal_hint = "monitor"

            content = (
                f"Whale Alert: {amount:,.0f} {symbol} "
                f"(~${amount_usd:,.0f}) moved – {transfer_type} ({signal_hint})"
            )

            # Relevance and urgency scale with transfer size
            # Base relevance 0.5, up to 1.0 for $100M+ transfers
            relevance_score = min(1.0, 0.5 + min(0.5, amount_usd / 100_000_000))
            # Base urgency 0.4, up to 1.0 for $50M+ transfers
            urgency_score = min(1.0, 0.4 + min(0.6, amount_usd / 50_000_000))

            return DataItem(
                source_type=self.source_type,
                source_name="whale_alert",
                content=content,
                url=f"https://whale-alert.io/transaction/{tx_hash}" if tx_hash else None,
                timestamp=ts,
                raw_data=tx,
                metadata={
                    "transfer_type": transfer_type,
                    "signal_hint": signal_hint,
                    "amount_usd": amount_usd,
                    "symbol": symbol,
                    "amount": amount,
                    "from_owner": from_owner,
                    "to_owner": to_owner,
                },
                relevance_score=relevance_score,
                urgency_score=urgency_score,
                mentioned_assets=[symbol],
            )
        except Exception as exc:
            logger.debug(f"WhaleAlert _create_data_item error: {exc}")
            return None
