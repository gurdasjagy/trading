"""SOL Stake Monitor — monitors Solana validator stake distribution."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SolStakeMonitor(BaseSource):
    """Monitors SOL validator stake distribution changes via Solana RPC API."""

    SOLANA_RPC = "https://api.mainnet-beta.solana.com"
    UNSTAKE_THRESHOLD_SOL = 1_000_000  # 1M SOL

    def __init__(self, polling_interval: int = 3600):  # 1 hour
        super().__init__("sol_stake", DataSourceType.REST_API)
        self._polling_interval = polling_interval
        self._items: List[DataItem] = []
        self._prev_stake_data: Optional[Dict] = None

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("SOL Stake Monitor started")
        while self._running:
            try:
                items = await self._fetch_stake_changes()
                self._items.extend(items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"SolStake monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self._fetch_stake_changes()
        return self._items[-limit:]

    async def _fetch_stake_changes(self) -> List[DataItem]:
        """Fetch validator stake data and detect large unstaking events."""
        items: List[DataItem] = []

        try:
            # Call getVoteAccounts RPC method
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getVoteAccounts",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.SOLANA_RPC,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Solana RPC status {resp.status}")
                        return items
                    data = await resp.json()

            if "result" not in data:
                logger.warning(f"Solana RPC error: {data.get('error')}")
                return items

            result = data["result"]
            current_validators = result.get("current", [])
            delinquent_validators = result.get("delinquent", [])

            # Calculate total stake
            total_stake_lamports = sum(
                v.get("activatedStake", 0) for v in current_validators
            )
            total_stake_sol = total_stake_lamports / 1e9

            validator_count = len(current_validators)
            delinquent_count = len(delinquent_validators)

            # Compare with previous data to detect changes
            if self._prev_stake_data:
                prev_total = self._prev_stake_data.get("total_stake_sol", 0)
                stake_change_sol = total_stake_sol - prev_total

                # Detect large unstaking (negative change > threshold)
                if stake_change_sol < -self.UNSTAKE_THRESHOLD_SOL:
                    content = (
                        f"SOL Stake Alert: Large unstaking detected "
                        f"({abs(stake_change_sol):,.0f} SOL, {validator_count} validators) – "
                        f"Potential selling pressure"
                    )

                    # Relevance scales with stake change magnitude
                    relevance_score = min(1.0, abs(stake_change_sol) / 5_000_000)
                    urgency_score = min(1.0, abs(stake_change_sol) / 3_000_000)

                    item = DataItem(
                        source_type=self.source_type,
                        source_name=self.name,
                        content=content,
                        timestamp=_utcnow(),
                        metadata={
                            "stake_change_sol": stake_change_sol,
                            "total_stake_sol": total_stake_sol,
                            "validator_count": validator_count,
                            "delinquent_count": delinquent_count,
                            "signal": "selling_pressure",
                        },
                        relevance_score=relevance_score,
                        urgency_score=urgency_score,
                        mentioned_assets=["SOL"],
                    )
                    items.append(item)
                    self._items_collected += 1
                    logger.info(
                        f"SOL unstaking: {abs(stake_change_sol):,.0f} SOL "
                        f"({validator_count} validators)"
                    )

            # Store current data for next comparison
            self._prev_stake_data = {
                "total_stake_sol": total_stake_sol,
                "validator_count": validator_count,
                "delinquent_count": delinquent_count,
            }
            self._last_update = _utcnow()

        except Exception as exc:
            logger.warning(f"SolStake fetch error: {exc}")
            self._errors += 1

        return items
