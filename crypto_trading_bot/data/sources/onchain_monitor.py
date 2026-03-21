"""On-chain data monitor for whale movements and exchange flows."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class OnChainMonitor(BaseSource):
    """Monitors on-chain data for whale movements using Whale Alert and Etherscan APIs."""

    WHALE_ALERT_API = "https://api.whale-alert.io/v1"
    ETHERSCAN_API = "https://api.etherscan.io/api"
    BLOCKCHAIN_API = "https://blockchain.info"

    # Minimum transfer value in USD to consider as a whale movement
    WHALE_THRESHOLD_USD = 1_000_000

    def __init__(
        self,
        whale_alert_api_key: str = "",
        etherscan_api_key: str = "",
        polling_interval: int = 60,
    ):
        super().__init__("onchain", DataSourceType.ONCHAIN)
        self._whale_alert_key = whale_alert_api_key
        self._etherscan_key = etherscan_api_key
        self._polling_interval = polling_interval
        self._seen_tx_ids: set = set()
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._whale_alert_key and not self._etherscan_key:
            logger.warning(
                "OnChain: no API keys provided – monitoring will use public endpoints only."
            )
        logger.info("OnChain Monitor started")
        while self._running:
            try:
                new_items: List[DataItem] = []
                new_items.extend(await self.monitor_whale_transfers())
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"OnChain monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(60)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self.monitor_whale_transfers()
        return self._items[-limit:]

    async def monitor_whale_transfers(self) -> List[DataItem]:
        """Fetch large on-chain transfers from Whale Alert API."""
        items: List[DataItem] = []
        if not self._whale_alert_key:
            return await self._public_btc_fallback()
        try:
            params = {
                "api_key": self._whale_alert_key,
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
                            item = self._classify_transfer(tx)
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

    async def check_exchange_flows(self, asset: str = "BTC") -> Dict[str, float]:
        """Return estimated exchange inflow/outflow using Etherscan (ETH-family) or blockchain.info (BTC)."""
        flows: Dict[str, float] = {"inflow": 0.0, "outflow": 0.0, "net": 0.0}
        if asset == "ETH" and self._etherscan_key:
            try:
                # Etherscan: known exchange addresses (simplified)
                binance_addr = "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE"
                params = {
                    "module": "account",
                    "action": "txlist",
                    "address": binance_addr,
                    "startblock": 0,
                    "endblock": 99999999,
                    "sort": "desc",
                    "apikey": self._etherscan_key,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.ETHERSCAN_API,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for tx in (data.get("result") or [])[:100]:
                                value_eth = int(tx.get("value", 0)) / 1e18
                                if tx.get("to", "").lower() == binance_addr.lower():
                                    flows["inflow"] += value_eth
                                else:
                                    flows["outflow"] += value_eth
                            flows["net"] = flows["inflow"] - flows["outflow"]
            except Exception as exc:
                logger.warning(f"Etherscan exchange flow error: {exc}")
        return flows

    async def get_network_metrics(self, asset: str = "BTC") -> Dict[str, float]:
        """Fetch network-level metrics (hash rate, difficulty, mempool size)."""
        metrics: Dict[str, float] = {}
        if asset == "BTC":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.BLOCKCHAIN_API}/stats?format=json",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            metrics = {
                                "hash_rate": float(data.get("hash_rate", 0)),
                                "difficulty": float(data.get("difficulty", 0)),
                                "mempool_size": float(data.get("mempool_size", 0)),
                                "n_tx": float(data.get("n_tx", 0)),
                            }
            except Exception as exc:
                logger.warning(f"Blockchain.info metrics error: {exc}")
        return metrics

    def _classify_transfer(self, tx: dict) -> Optional[DataItem]:
        """Classify a transaction and build a DataItem."""
        try:
            symbol = (tx.get("symbol") or tx.get("blockchain") or "").upper()
            amount = float(tx.get("amount") or tx.get("amount_usd") or 0)
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

            # Classify direction
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
            assets = (
                [symbol]
                if symbol in self._ASSET_KEYWORDS
                else self._extract_mentioned_assets(content)
            )
            urgency = min(1.0, 0.4 + min(1.0, amount_usd / 50_000_000))
            return DataItem(
                source_type=self.source_type,
                source_name="onchain/whale_alert",
                content=content,
                url=f"https://whale-alert.io/transaction/{tx_hash}" if tx_hash else None,
                timestamp=ts,
                raw_data=tx,
                metadata={
                    "transfer_type": transfer_type,
                    "signal_hint": signal_hint,
                    "amount_usd": amount_usd,
                    "symbol": symbol,
                },
                relevance_score=min(1.0, 0.5 + min(0.5, amount_usd / 100_000_000)),
                urgency_score=urgency,
                mentioned_assets=assets,
            )
        except Exception as exc:
            logger.debug(f"_classify_transfer error: {exc}")
            return None

    async def _public_btc_fallback(self) -> List[DataItem]:
        """Use blockchain.info unconfirmed transactions as a very rough fallback."""
        items: List[DataItem] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BLOCKCHAIN_API}/unconfirmed-transactions?format=json",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for tx in (data.get("txs") or [])[:10]:
                            total_out = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
                            if total_out < 100:  # skip small txs (< 100 BTC)
                                continue
                            tx_hash = tx.get("hash", "")
                            if tx_hash in self._seen_tx_ids:
                                continue
                            self._seen_tx_ids.add(tx_hash)
                            content = f"Large unconfirmed BTC tx: {total_out:.2f} BTC"
                            items.append(
                                DataItem(
                                    source_type=self.source_type,
                                    source_name="onchain/blockchain_info",
                                    content=content,
                                    url=f"https://blockchain.info/tx/{tx_hash}",
                                    timestamp=_utcnow(),
                                    metadata={"amount_btc": total_out, "tx_hash": tx_hash},
                                    relevance_score=0.6,
                                    urgency_score=min(1.0, total_out / 1000),
                                    mentioned_assets=["BTC"],
                                )
                            )
                            self._items_collected += 1
        except Exception as exc:
            logger.debug(f"BTC public fallback error: {exc}")
        return items
