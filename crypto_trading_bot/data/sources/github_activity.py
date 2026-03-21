"""GitHub development activity monitor for crypto projects."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GitHubActivityMonitor(BaseSource):
    """Tracks GitHub development activity for crypto projects."""

    GITHUB_API = "https://api.github.com"

    DEFAULT_REPOS = [
        "bitcoin/bitcoin",
        "ethereum/go-ethereum",
        "solana-labs/solana",
        "bnb-chain/bsc",
        "ripple/rippled",
        "cardano-foundation/cardano-node",
        "polkadot-fellows/runtimes",
        "near/nearcore",
        "OffchainLabs/nitro",  # Arbitrum
        "maticnetwork/bor",  # Polygon
    ]

    def __init__(
        self,
        github_token: str = "",
        repos: Optional[List[str]] = None,
        polling_interval: int = 3600,  # 1 hour
    ):
        super().__init__("github_activity", DataSourceType.REST_API)
        self._token = github_token
        self._repos = repos or self.DEFAULT_REPOS
        self._polling_interval = polling_interval
        self._seen_commit_shas: set = set()
        self._items: List[DataItem] = []

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def start_monitoring(self) -> None:
        self._running = True
        if not self._token:
            logger.warning("GitHub: no token – using unauthenticated API (60 req/hr limit).")
        logger.info(f"GitHub Activity Monitor started – tracking {len(self._repos)} repos")
        while self._running:
            try:
                for repo in self._repos:
                    new_items = await self.fetch_commits(repo)
                    self._items.extend(new_items)
                await self.track_developer_activity()
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"GitHub monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            for repo in self._repos[:3]:
                self._items.extend(await self.fetch_commits(repo))
        return self._items[-limit:]

    async def fetch_commits(self, repo: str, count: int = 10) -> List[DataItem]:
        """Fetch recent commits for a GitHub repository."""
        items: List[DataItem] = []
        url = f"{self.GITHUB_API}/repos/{repo}/commits"
        params = {"per_page": count}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        commits = await resp.json()
                        for commit in commits:
                            sha = commit.get("sha", "")
                            if sha in self._seen_commit_shas:
                                continue
                            self._seen_commit_shas.add(sha)
                            commit_data = commit.get("commit", {})
                            author_info = commit_data.get("author", {})
                            message = commit_data.get("message", "").split("\n")[0]
                            author_name = author_info.get("name", "unknown")
                            ts_str = author_info.get("date", "")
                            try:
                                ts = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=None)
                            except Exception:
                                ts = _utcnow()
                            asset = self._repo_to_asset(repo)
                            content = f"[{repo}] {author_name}: {message}"
                            significant = self.detect_significant_updates(message)
                            items.append(
                                DataItem(
                                    source_type=self.source_type,
                                    source_name=f"github/{repo}",
                                    content=content,
                                    url=commit.get("html_url"),
                                    author=author_name,
                                    timestamp=ts,
                                    raw_data={"sha": sha, "repo": repo},
                                    metadata={
                                        "repo": repo,
                                        "sha": sha[:8],
                                        "significant": significant,
                                    },
                                    relevance_score=0.8 if significant else 0.4,
                                    urgency_score=0.6 if significant else 0.2,
                                    mentioned_assets=[asset] if asset else [],
                                )
                            )
                            self._items_collected += 1
                    elif resp.status == 403:
                        logger.warning(f"GitHub rate limit hit for {repo}")
                    else:
                        logger.debug(f"GitHub API {resp.status} for {repo}")
        except Exception as exc:
            logger.warning(f"fetch_commits({repo}) error: {exc}")
            self._errors += 1
        return items

    async def track_developer_activity(self) -> Dict[str, int]:
        """Summarise contributor counts across tracked repos."""
        activity: Dict[str, int] = {}
        async with aiohttp.ClientSession() as session:
            for repo in self._repos:
                url = f"{self.GITHUB_API}/repos/{repo}/stats/contributors"
                try:
                    async with session.get(
                        url,
                        headers=self._headers(),
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            activity[repo] = len(data) if isinstance(data, list) else 0
                except Exception as exc:
                    logger.debug(f"track_developer_activity({repo}): {exc}")
        return activity

    def detect_significant_updates(self, message: str) -> bool:
        """Heuristically detect if a commit message suggests a significant update."""
        SIGNIFICANT_KEYWORDS = [
            "security",
            "fix",
            "vulnerability",
            "upgrade",
            "breaking",
            "consensus",
            "hard fork",
            "merge",
            "release",
            "critical",
            "patch",
            "update protocol",
        ]
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in SIGNIFICANT_KEYWORDS)

    def _repo_to_asset(self, repo: str) -> Optional[str]:
        """Map a repository slug to a known crypto asset ticker."""
        REPO_ASSET_MAP = {
            "bitcoin/bitcoin": "BTC",
            "ethereum/go-ethereum": "ETH",
            "solana-labs/solana": "SOL",
            "bnb-chain/bsc": "BNB",
            "ripple/rippled": "XRP",
            "cardano-foundation/cardano-node": "ADA",
            "polkadot-fellows/runtimes": "DOT",
            "near/nearcore": "NEAR",
            "OffchainLabs/nitro": "ARB",
            "maticnetwork/bor": "MATIC",
        }
        return REPO_ASSET_MAP.get(repo)
