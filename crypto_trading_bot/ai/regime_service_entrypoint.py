"""Regime Service entrypoint — Docker Compose ``command`` target.

Run with:
    python -m ai.regime_service_entrypoint

Modes:
    1. **Standalone** (default): Runs the RegimeService loop independently.
    2. **Orchestrator** (``USE_COLD_PATH_ORCHESTRATOR=1``): Delegates to
       :class:`~core.cold_path_orchestrator.ColdPathOrchestrator` which
       manages regime, sentiment, and health monitoring together.

Environment variables
---------------------
USE_COLD_PATH_ORCHESTRATOR   Set to "1" or "true" to use the new cold-path
                             orchestrator instead of standalone regime loop.
REGIME_OUTPUT_PATH           Path to write regime JSON (default: /dev/shm/regime_state.json)
REGIME_INTERVAL_SECONDS      Update cadence in seconds (default: 300)
REGIME_TTL_SECONDS           State TTL for the Rust reader (default: 600)
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger


def _use_orchestrator() -> bool:
    """Check if the cold-path orchestrator should be used."""
    val = os.environ.get("USE_COLD_PATH_ORCHESTRATOR", "").strip().lower()
    return val in ("1", "true", "yes", "on")


async def _main_standalone() -> None:
    """Original standalone regime service loop (backward compatible)."""
    logger.info("Regime Service starting (standalone mode)…")

    from ai.regime_service import RegimeService

    # Optional integrations — import only if available
    llm_client = None
    sentiment_analyzer = None
    regime_detector = None
    cross_asset_detector = None
    news_sources = []

    try:
        from ai.llm_client import LLMClient
        llm_client = LLMClient()
        logger.info("LLMClient loaded")
    except Exception as exc:
        logger.warning(f"LLMClient not available: {exc}")

    try:
        from ai.sentiment.analyzer import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("SentimentAnalyzer loaded")
    except Exception as exc:
        logger.warning(f"SentimentAnalyzer not available: {exc}")

    try:
        from ai.market_analyzer.regime_detector import MarketRegimeDetector
        regime_detector = MarketRegimeDetector()
        logger.info("MarketRegimeDetector loaded")
    except Exception as exc:
        logger.warning(f"MarketRegimeDetector not available: {exc}")

    try:
        from ai.market_analyzer.cross_asset_regime_detector import CrossAssetRegimeDetector
        cross_asset_detector = CrossAssetRegimeDetector()
        logger.info("CrossAssetRegimeDetector loaded")
    except Exception as exc:
        logger.warning(f"CrossAssetRegimeDetector not available: {exc}")

    # Load news sources
    _news_source_classes = [
        ("data.sources.news_rss_monitor", "NewsRSSMonitor"),
        ("data.sources.cryptopanic_source", "CryptoPanicSource"),
        ("data.sources.reddit_monitor", "RedditMonitor"),
    ]
    for module_path, class_name in _news_source_classes:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            news_sources.append(cls())
            logger.info(f"News source loaded: {class_name}")
        except Exception as exc:
            logger.debug(f"News source {class_name} not available: {exc}")

    service = RegimeService(
        llm_client=llm_client,
        sentiment_analyzer=sentiment_analyzer,
        regime_detector=regime_detector,
        cross_asset_detector=cross_asset_detector,
        news_sources=news_sources,
    )

    # Write an initial safe-default state immediately so the bot doesn't start
    # with a stale file.
    await service.run_once()

    logger.info("Regime Service loop running…")
    await service.run_forever()


async def _main_orchestrator() -> None:
    """Cold-path orchestrator mode (Issue 4)."""
    logger.info("Regime Service starting (cold-path orchestrator mode)…")

    from config.settings import Settings
    from core.cold_path_orchestrator import ColdPathOrchestrator

    settings = Settings.get_settings()
    orchestrator = ColdPathOrchestrator(settings)
    await orchestrator.start()


async def _main() -> None:
    """Route to the appropriate entrypoint based on configuration."""
    if _use_orchestrator():
        await _main_orchestrator()
    else:
        await _main_standalone()


if __name__ == "__main__":
    asyncio.run(_main())

