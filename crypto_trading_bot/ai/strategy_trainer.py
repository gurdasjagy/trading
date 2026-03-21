"""AI Strategy Trainer — DeepSeek V3 via GateRouter API.

Collects news events + market outcomes, feeds them to DeepSeek V3 via GateRouter,
builds a lookup table (news_category -> expected_impact -> strategy_adjustment),
and writes results to /dev/shm/ai_strategy_weights for the Rust engine.

Runs daily as a scheduled task (via cron or asyncio.create_task).
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel


class NewsEvent(BaseModel):
    """A single news event with market outcome."""
    timestamp: int  # Unix timestamp (seconds)
    category: str  # e.g., "fed_rate_decision", "cpi_data", "exchange_hack"
    content: str  # News headline or summary
    sentiment: float  # -1.0 to 1.0
    market_impact_pct: float  # BTC price change 1h after event
    volatility_spike: bool  # True if ATR spiked >50% within 4h


class StrategyAdjustment(BaseModel):
    """AI-recommended strategy adjustment for a news category."""
    category: str
    recommended_position_scale: float  # 0.0 to 1.0
    recommended_leverage: int  # 1 to 125
    blocked_strategies: List[str]  # e.g., ["mean_reversion", "market_making"]
    allowed_strategies: List[str]  # e.g., ["trend_following", "breakout"]
    confidence: float  # 0.0 to 1.0


class AIStrategyTrainer:
    """Trains strategy adjustments using DeepSeek V3 via GateRouter API.

    Architecture:
    1. Collect news events + market outcomes from the past 30 days
    2. Feed them to DeepSeek V3 via GateRouter API (llm_client.py)
    3. Parse the LLM's recommendations into a lookup table
    4. Write the table to /dev/shm/ai_strategy_weights (JSON)
    5. The Rust engine reads this file and applies adjustments in real-time

    Usage:
        trainer = AIStrategyTrainer(llm_client=llm_client)
        await trainer.run_daily_training()
    """

    _DEFAULT_OUTPUT_PATH = "/dev/shm/ai_strategy_weights"
    _DEFAULT_TRAINING_INTERVAL = 86400  # 24 hours

    def __init__(
        self,
        llm_client=None,
        news_collector=None,
        market_data_provider=None,
        output_path: Optional[str] = None,
    ) -> None:
        """Initialize the AI strategy trainer.

        Args:
            llm_client: LLMClient instance (must have GateRouter API key configured)
            news_collector: Object with async fetch() method returning List[Dict]
            market_data_provider: Object with async get_ohlcv() method
            output_path: Path to write strategy weights JSON
        """
        self._llm = llm_client
        self._news_collector = news_collector
        self._market_data = market_data_provider
        self._output_path = output_path or os.environ.get(
            "AI_STRATEGY_WEIGHTS_PATH", self._DEFAULT_OUTPUT_PATH
        )
        self._training_interval = int(
            os.environ.get("AI_TRAINING_INTERVAL_SECONDS", self._DEFAULT_TRAINING_INTERVAL)
        )
        self._last_training_ts: float = 0.0
        self._training_count: int = 0
        self._error_count: int = 0

        logger.info(
            f"AIStrategyTrainer initialized — output={self._output_path} "
            f"interval={self._training_interval}s"
        )

    async def run_daily_training(self) -> Dict[str, StrategyAdjustment]:
        """Run one training cycle: collect data, query LLM, write results.

        Returns:
            Dict mapping news category to StrategyAdjustment.
        """
        try:
            logger.info("Starting AI strategy training cycle...")
            start_time = time.time()

            # Step 1: Collect news events + market outcomes
            events = await self._collect_training_data()
            if not events:
                logger.warning("No training data collected — skipping cycle")
                return {}

            logger.info(f"Collected {len(events)} news events for training")

            # Step 2: Query DeepSeek V3 via GateRouter API
            adjustments = await self._query_llm_for_adjustments(events)
            if not adjustments:
                logger.warning("LLM returned no adjustments — using defaults")
                adjustments = self._get_default_adjustments()

            logger.info(f"LLM returned {len(adjustments)} strategy adjustments")

            # Step 3: Write to /dev/shm/ai_strategy_weights
            await self._persist_adjustments(adjustments)

            elapsed = time.time() - start_time
            self._last_training_ts = time.time()
            self._training_count += 1

            logger.info(
                f"AI strategy training completed in {elapsed:.1f}s "
                f"(cycle #{self._training_count})"
            )

            return adjustments

        except Exception as exc:
            self._error_count += 1
            logger.error(f"AI strategy training failed: {exc}", exc_info=True)
            return {}

    async def run_forever(self) -> None:
        """Background loop that runs training every 24 hours.

        Designed to be run with asyncio.create_task(trainer.run_forever()).
        """
        logger.info(
            f"AI strategy trainer background loop started (interval={self._training_interval}s)"
        )
        while True:
            try:
                await self.run_daily_training()
            except Exception as exc:
                logger.error(f"AI training loop iteration failed: {exc}")

            await asyncio.sleep(self._training_interval)

    async def _collect_training_data(self) -> List[NewsEvent]:
        """Collect news events + market outcomes from the past 30 days.

        Returns:
            List of NewsEvent objects with market impact data.
        """
        if self._news_collector is None:
            logger.debug("No news collector configured — returning empty list")
            return []

        try:
            # Fetch news from the past 30 days
            news_items = await self._news_collector.fetch()
            if not news_items:
                return []

            events: List[NewsEvent] = []
            for item in news_items[:100]:  # Limit to 100 most recent
                # Extract fields
                content = item.get("content") or item.get("title", "")
                if not content:
                    continue

                timestamp = item.get("timestamp", int(time.time()))
                category = self._categorize_news(content)
                sentiment = item.get("sentiment", 0.0)

                # Fetch market impact (BTC price change 1h after event)
                market_impact_pct = await self._get_market_impact(timestamp)
                volatility_spike = await self._check_volatility_spike(timestamp)

                events.append(
                    NewsEvent(
                        timestamp=timestamp,
                        category=category,
                        content=content[:200],  # Truncate for LLM context
                        sentiment=sentiment,
                        market_impact_pct=market_impact_pct,
                        volatility_spike=volatility_spike,
                    )
                )

            return events

        except Exception as exc:
            logger.warning(f"Failed to collect training data: {exc}")
            return []

    def _categorize_news(self, content: str) -> str:
        """Categorize news content into a predefined category.

        Categories:
        - fed_rate_decision
        - cpi_data
        - exchange_hack
        - regulatory_news
        - whale_movement
        - technical_upgrade
        - market_crash
        - unknown
        """
        content_lower = content.lower()

        if any(kw in content_lower for kw in ["fed", "interest rate", "fomc"]):
            return "fed_rate_decision"
        if any(kw in content_lower for kw in ["cpi", "inflation", "pce"]):
            return "cpi_data"
        if any(kw in content_lower for kw in ["hack", "exploit", "breach"]):
            return "exchange_hack"
        if any(kw in content_lower for kw in ["sec", "regulation", "ban", "law"]):
            return "regulatory_news"
        if any(kw in content_lower for kw in ["whale", "large transfer", "billion"]):
            return "whale_movement"
        if any(kw in content_lower for kw in ["upgrade", "fork", "halving"]):
            return "technical_upgrade"
        if any(kw in content_lower for kw in ["crash", "dump", "liquidation"]):
            return "market_crash"

        return "unknown"

    async def _get_market_impact(self, event_timestamp: int) -> float:
        """Calculate BTC price change 1h after the event.

        Args:
            event_timestamp: Unix timestamp of the news event (seconds)

        Returns:
            Price change percentage (e.g., 2.5 for +2.5%, -3.0 for -3.0%)
        """
        if self._market_data is None:
            return 0.0

        try:
            # Fetch OHLCV data for 1h before and 1h after the event
            ohlcv = await self._market_data.get_ohlcv(
                symbol="BTC/USDT",
                timeframe="1h",
                since=event_timestamp - 3600,
                limit=3,
            )
            if len(ohlcv) < 2:
                return 0.0

            # Price change = (close_after - close_before) / close_before * 100
            close_before = ohlcv[0][4]  # Close price 1h before
            close_after = ohlcv[-1][4]  # Close price 1h after
            return ((close_after - close_before) / close_before) * 100.0

        except Exception as exc:
            logger.debug(f"Failed to get market impact for ts={event_timestamp}: {exc}")
            return 0.0

    async def _check_volatility_spike(self, event_timestamp: int) -> bool:
        """Check if ATR spiked >50% within 4h of the event.

        Args:
            event_timestamp: Unix timestamp of the news event (seconds)

        Returns:
            True if volatility spiked, False otherwise
        """
        if self._market_data is None:
            return False

        try:
            # Fetch OHLCV data for 4h before and 4h after the event
            ohlcv = await self._market_data.get_ohlcv(
                symbol="BTC/USDT",
                timeframe="1h",
                since=event_timestamp - 14400,
                limit=9,
            )
            if len(ohlcv) < 5:
                return False

            # Calculate ATR before and after
            atr_before = self._calculate_atr(ohlcv[:4])
            atr_after = self._calculate_atr(ohlcv[4:])

            return atr_after > atr_before * 1.5

        except Exception as exc:
            logger.debug(f"Failed to check volatility spike for ts={event_timestamp}: {exc}")
            return False

    def _calculate_atr(self, ohlcv: List[List[float]]) -> float:
        """Calculate Average True Range from OHLCV data.

        Args:
            ohlcv: List of [timestamp, open, high, low, close, volume]

        Returns:
            ATR value
        """
        if len(ohlcv) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(ohlcv)):
            high = ohlcv[i][2]
            low = ohlcv[i][3]
            prev_close = ohlcv[i - 1][4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    async def _query_llm_for_adjustments(
        self, events: List[NewsEvent]
    ) -> Dict[str, StrategyAdjustment]:
        """Query DeepSeek V3 via GateRouter API for strategy adjustments.

        Args:
            events: List of NewsEvent objects with market impact data

        Returns:
            Dict mapping news category to StrategyAdjustment
        """
        if self._llm is None:
            logger.warning("No LLM client configured — using default adjustments")
            return self._get_default_adjustments()

        try:
            # Build prompt for DeepSeek V3
            prompt = self._build_training_prompt(events)

            # Query LLM via GateRouter API
            response = await self._llm.query_json(
                prompt=prompt,
                system_prompt=(
                    "You are an expert cryptocurrency trading strategist. "
                    "Analyze the provided news events and market outcomes, "
                    "then recommend strategy adjustments for each news category. "
                    "Respond ONLY with valid JSON."
                ),
            )

            if "error" in response:
                logger.warning(f"LLM query failed: {response['error']}")
                return self._get_default_adjustments()

            # Parse LLM response into StrategyAdjustment objects
            adjustments = self._parse_llm_response(response)
            return adjustments

        except Exception as exc:
            logger.error(f"LLM query failed: {exc}", exc_info=True)
            return self._get_default_adjustments()

    def _build_training_prompt(self, events: List[NewsEvent]) -> str:
        """Build the LLM prompt from collected news events.

        Args:
            events: List of NewsEvent objects

        Returns:
            Formatted prompt string
        """
        # Group events by category
        by_category: Dict[str, List[NewsEvent]] = {}
        for event in events:
            if event.category not in by_category:
                by_category[event.category] = []
            by_category[event.category].append(event)

        # Build prompt
        prompt_parts = [
            "Analyze the following news events and their market outcomes:",
            "",
        ]

        for category, cat_events in by_category.items():
            avg_impact = sum(e.market_impact_pct for e in cat_events) / len(cat_events)
            vol_spike_pct = (
                sum(1 for e in cat_events if e.volatility_spike) / len(cat_events) * 100
            )

            prompt_parts.append(f"## {category}")
            prompt_parts.append(f"- Events: {len(cat_events)}")
            prompt_parts.append(f"- Avg market impact: {avg_impact:.2f}%")
            prompt_parts.append(f"- Volatility spike rate: {vol_spike_pct:.0f}%")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "For each news category, recommend:",
                "1. recommended_position_scale (0.0 to 1.0)",
                "2. recommended_leverage (1 to 125)",
                "3. blocked_strategies (list of strategy names to avoid)",
                "4. allowed_strategies (list of strategy names that work well)",
                "5. confidence (0.0 to 1.0)",
                "",
                "Respond with JSON in this format:",
                "{",
                '  "fed_rate_decision": {',
                '    "recommended_position_scale": 0.5,',
                '    "recommended_leverage": 3,',
                '    "blocked_strategies": ["mean_reversion", "market_making"],',
                '    "allowed_strategies": ["trend_following", "breakout"],',
                '    "confidence": 0.8',
                "  },",
                "  ...",
                "}",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_llm_response(
        self, response: Dict[str, Any]
    ) -> Dict[str, StrategyAdjustment]:
        """Parse LLM JSON response into StrategyAdjustment objects.

        Args:
            response: LLM response dict

        Returns:
            Dict mapping category to StrategyAdjustment
        """
        adjustments: Dict[str, StrategyAdjustment] = {}

        for category, data in response.items():
            if not isinstance(data, dict):
                continue

            try:
                adj = StrategyAdjustment(
                    category=category,
                    recommended_position_scale=float(
                        data.get("recommended_position_scale", 0.5)
                    ),
                    recommended_leverage=int(data.get("recommended_leverage", 5)),
                    blocked_strategies=data.get("blocked_strategies", []),
                    allowed_strategies=data.get("allowed_strategies", []),
                    confidence=float(data.get("confidence", 0.5)),
                )
                adjustments[category] = adj
            except Exception as exc:
                logger.warning(f"Failed to parse adjustment for {category}: {exc}")

        return adjustments

    def _get_default_adjustments(self) -> Dict[str, StrategyAdjustment]:
        """Return conservative default strategy adjustments.

        Used when LLM query fails or no training data is available.
        """
        return {
            "fed_rate_decision": StrategyAdjustment(
                category="fed_rate_decision",
                recommended_position_scale=0.3,
                recommended_leverage=3,
                blocked_strategies=["mean_reversion", "market_making"],
                allowed_strategies=["trend_following"],
                confidence=0.5,
            ),
            "cpi_data": StrategyAdjustment(
                category="cpi_data",
                recommended_position_scale=0.4,
                recommended_leverage=3,
                blocked_strategies=["market_making"],
                allowed_strategies=["trend_following", "breakout"],
                confidence=0.5,
            ),
            "exchange_hack": StrategyAdjustment(
                category="exchange_hack",
                recommended_position_scale=0.0,
                recommended_leverage=1,
                blocked_strategies=["mean_reversion", "market_making", "trend_following"],
                allowed_strategies=[],
                confidence=0.9,
            ),
            "regulatory_news": StrategyAdjustment(
                category="regulatory_news",
                recommended_position_scale=0.2,
                recommended_leverage=2,
                blocked_strategies=["market_making"],
                allowed_strategies=["trend_following"],
                confidence=0.6,
            ),
            "market_crash": StrategyAdjustment(
                category="market_crash",
                recommended_position_scale=0.0,
                recommended_leverage=1,
                blocked_strategies=["mean_reversion", "market_making", "trend_following"],
                allowed_strategies=[],
                confidence=0.95,
            ),
            "unknown": StrategyAdjustment(
                category="unknown",
                recommended_position_scale=0.5,
                recommended_leverage=5,
                blocked_strategies=[],
                allowed_strategies=["trend_following", "mean_reversion"],
                confidence=0.3,
            ),
        }

    async def _persist_adjustments(
        self, adjustments: Dict[str, StrategyAdjustment]
    ) -> None:
        """Write strategy adjustments to /dev/shm/ai_strategy_weights.

        Uses atomic write-then-rename to avoid partial reads.

        Args:
            adjustments: Dict mapping category to StrategyAdjustment
        """
        path = Path(self._output_path)
        tmp_path = path.with_suffix(".tmp")

        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Build JSON output
            output = {
                "timestamp": int(time.time()),
                "training_count": self._training_count,
                "adjustments": {
                    cat: adj.model_dump() for cat, adj in adjustments.items()
                },
            }

            # Write to temp file
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2)

            # Atomic rename
            tmp_path.replace(path)

            logger.info(
                f"AI strategy weights written to {path} ({len(adjustments)} categories)"
            )

        except Exception as exc:
            logger.error(f"Failed to persist strategy adjustments: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
