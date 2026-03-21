"""SEC EDGAR filings monitor for crypto-related regulatory activity."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SECFilingsMonitor(BaseSource):
    """Monitors SEC EDGAR for crypto-related filings that may affect markets."""

    EDGAR_API = "https://efts.sec.gov/LATEST/search-index"
    EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22bitcoin%22&dateRange=custom&startdt={start}&enddt={end}&forms=8-K,S-1,13F"
    EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
    EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"

    CRYPTO_TERMS = [
        "bitcoin",
        "ethereum",
        "cryptocurrency",
        "digital asset",
        "blockchain",
        "crypto",
        "stablecoin",
        "defi",
        "nft",
        "web3",
        "virtual currency",
        "digital currency",
    ]

    # Filing types with their market impact significance
    HIGH_IMPACT_FORMS = {"8-K", "S-1", "S-11", "424B4", "SC 13G", "SC 13D"}
    MEDIUM_IMPACT_FORMS = {"13F", "10-K", "10-Q", "DEF 14A"}

    def __init__(
        self,
        polling_interval: int = 1800,  # 30 minutes
        forms: Optional[List[str]] = None,
    ):
        super().__init__("sec_filings", DataSourceType.REST_API)
        self._polling_interval = polling_interval
        self._forms = forms or list(self.HIGH_IMPACT_FORMS | self.MEDIUM_IMPACT_FORMS)
        self._seen_accession_numbers: set = set()
        self._items: List[DataItem] = []

    async def start_monitoring(self) -> None:
        self._running = True
        logger.info("SEC Filings Monitor started (EDGAR API – no key required)")
        while self._running:
            try:
                new_items = await self.fetch_recent_filings()
                self._items.extend(new_items)
                if len(self._items) > 500:
                    self._items = self._items[-500:]
                await asyncio.sleep(self._polling_interval)
            except Exception as exc:
                logger.error(f"SEC filings monitoring error: {exc}")
                self._errors += 1
                await asyncio.sleep(300)

    async def stop_monitoring(self) -> None:
        self._running = False

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        if not self._items:
            await self.fetch_recent_filings()
        return self._items[-limit:]

    async def fetch_recent_filings(self) -> List[DataItem]:
        """Search EDGAR full-text search for recent crypto-related filings."""
        new_items: List[DataItem] = []
        async with aiohttp.ClientSession() as session:
            for term in self.CRYPTO_TERMS[:5]:  # limit queries
                params = {
                    "q": f'"{term}"',
                    "forms": ",".join(self._forms[:8]),
                    "dateRange": "custom",
                    "startdt": (datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                    "enddt": (datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                }
                try:
                    async with session.get(
                        "https://efts.sec.gov/LATEST/search-index",
                        params=params,
                        headers={"User-Agent": "CryptoTradingBot research@example.com"},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            hits = data.get("hits", {}).get("hits", [])
                            for hit in hits:
                                src = hit.get("_source", {})
                                acc = src.get("accession_no", hit.get("_id", ""))
                                if acc in self._seen_accession_numbers:
                                    continue
                                self._seen_accession_numbers.add(acc)
                                filing = self._build_filing_dict(src, acc)
                                if not self._filter_crypto_relevant(filing):
                                    continue
                                item = self._classify_impact(filing)
                                if item:
                                    new_items.append(item)
                                    self._items_collected += 1
                        else:
                            logger.debug(f"EDGAR search status {resp.status} for '{term}'")
                except Exception as exc:
                    logger.debug(f"EDGAR fetch error for '{term}': {exc}")
        self._last_update = _utcnow()
        return new_items

    def _build_filing_dict(self, src: dict, accession: str) -> dict:
        """Normalise EDGAR hit source into a filing dict."""
        return {
            "accession": accession,
            "form_type": src.get("form_type", ""),
            "company": src.get("entity_name") or src.get("company_name", "Unknown"),
            "cik": src.get("file_num") or src.get("cik", ""),
            "filed_at": src.get("period_of_report") or src.get("file_date", ""),
            "description": src.get("description") or src.get("period_of_report", ""),
            "url": src.get("file_date", ""),
        }

    def _filter_crypto_relevant(self, filing: dict) -> bool:
        """Return True if the filing is clearly related to crypto/digital assets."""
        text = f"{filing.get('company','')} {filing.get('description','')}".lower()
        return any(term in text for term in self.CRYPTO_TERMS)

    def _classify_impact(self, filing: dict) -> Optional[DataItem]:
        """Estimate market impact of the filing and produce a DataItem."""
        form_type = filing.get("form_type", "")
        company = filing.get("company", "Unknown")
        accession = filing.get("accession", "")
        filed_at = filing.get("filed_at", "")
        try:
            ts = datetime.strptime(filed_at, "%Y-%m-%d") if filed_at else _utcnow()
        except Exception:
            ts = _utcnow()

        if form_type in self.HIGH_IMPACT_FORMS:
            impact = "high"
            relevance = 0.9
            urgency = 0.8
        elif form_type in self.MEDIUM_IMPACT_FORMS:
            impact = "medium"
            relevance = 0.7
            urgency = 0.5
        else:
            impact = "low"
            relevance = 0.5
            urgency = 0.3

        description = filing.get("description", "")
        content = (
            f"SEC Filing ({form_type}): {company} filed {form_type}. "
            f"Impact: {impact}. {description}"
        )
        assets = self._extract_mentioned_assets(content)
        return DataItem(
            source_type=self.source_type,
            source_name="sec_filings/edgar",
            content=content,
            url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&accession={accession}",
            timestamp=ts,
            raw_data=filing,
            metadata={
                "form_type": form_type,
                "company": company,
                "accession": accession,
                "impact_level": impact,
            },
            relevance_score=relevance,
            urgency_score=urgency,
            mentioned_assets=assets,
        )
