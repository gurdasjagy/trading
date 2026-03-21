"""Tests for CryptoPanic and CoinGecko data sources, and sentiment modifier."""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.sources.base_source import DataItem, DataSourceType
from data.sources.coingecko_source import CoinGeckoSource
from data.sources.cryptopanic_source import CryptoPanicSource
from strategy.strategy_manager import StrategyManager

# ── CryptoPanicSource ─────────────────────────────────────────────────────────


class TestCryptoPanicSourceInit:
    def test_disabled_without_api_key(self):
        """CryptoPanicSource is disabled when no auth_token is provided."""
        src = CryptoPanicSource()
        assert src.enabled is False

    def test_enabled_with_api_key(self):
        """CryptoPanicSource is enabled when an auth_token is provided."""
        src = CryptoPanicSource(auth_token="test-key-123")
        assert src.enabled is True

    def test_name_and_source_type(self):
        src = CryptoPanicSource(auth_token="key")
        assert src.name == "cryptopanic"
        assert src.source_type == DataSourceType.REST_API

    def test_default_polling_interval(self):
        """Default polling interval should be 300 seconds (5 minutes)."""
        src = CryptoPanicSource(auth_token="key")
        assert src._polling_interval == 300


class TestCryptoPanicSourceParsePost:
    def _make_source(self) -> CryptoPanicSource:
        return CryptoPanicSource(auth_token="test-key")

    def test_parse_valid_post_returns_data_item(self):
        src = self._make_source()
        post = {
            "title": "Bitcoin surges past $70,000",
            "published_at": "2024-01-15T10:30:00Z",
            "url": "https://cryptopanic.com/news/1/btc-surge",
            "kind": "news",
            "votes": {"positive": 5, "negative": 1, "important": 2},
            "currencies": [{"code": "BTC"}],
        }
        item = src._parse_post(post)
        assert item is not None
        assert isinstance(item, DataItem)
        assert "BTC" in item.mentioned_assets
        assert 0.0 <= item.relevance_score <= 1.0
        assert 0.0 <= item.urgency_score <= 1.0

    def test_parse_post_no_title_returns_none(self):
        src = self._make_source()
        item = src._parse_post({"title": "", "currencies": []})
        assert item is None

    def test_parse_post_extracts_sentiment_from_votes(self):
        src = self._make_source()
        post = {
            "title": "BTC price analysis",
            "published_at": "2024-01-15T10:00:00Z",
            "votes": {"positive": 10, "negative": 0, "important": 0},
            "currencies": [{"code": "BTC"}],
        }
        item = src._parse_post(post)
        assert item is not None
        # positive-only votes → sentiment_score should be 1.0
        assert item.metadata is not None
        assert item.metadata["sentiment_score"] == pytest.approx(1.0)

    def test_parse_post_negative_sentiment(self):
        src = self._make_source()
        post = {
            "title": "ETH crash incoming",
            "published_at": "2024-01-15T10:00:00Z",
            "votes": {"positive": 0, "negative": 10, "important": 0},
            "currencies": [{"code": "ETH"}],
        }
        item = src._parse_post(post)
        assert item is not None
        assert item.metadata["sentiment_score"] == pytest.approx(-1.0)

    def test_parse_post_no_votes_neutral_sentiment(self):
        src = self._make_source()
        post = {
            "title": "SOL news update",
            "published_at": "2024-01-15T10:00:00Z",
            "votes": {},
            "currencies": [{"code": "SOL"}],
        }
        item = src._parse_post(post)
        assert item is not None
        assert item.metadata["sentiment_score"] == pytest.approx(0.0)

    def test_parse_post_currency_list_populates_mentioned_assets(self):
        src = self._make_source()
        post = {
            "title": "Market update",
            "published_at": "2024-01-15T10:00:00Z",
            "votes": {},
            "currencies": [{"code": "BTC"}, {"code": "ETH"}, {"code": "SOL"}],
        }
        item = src._parse_post(post)
        assert item is not None
        assert "BTC" in item.mentioned_assets
        assert "ETH" in item.mentioned_assets
        assert "SOL" in item.mentioned_assets

    @pytest.mark.asyncio
    async def test_fetch_latest_without_api_key_returns_empty(self):
        src = CryptoPanicSource()  # no key
        result = await src.fetch_latest()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_and_cache_with_mocked_response(self):
        src = CryptoPanicSource(auth_token="fake-key")
        mock_data = {
            "results": [
                {
                    "title": "Bitcoin hits new all-time high",
                    "published_at": "2024-01-20T12:00:00Z",
                    "url": "https://cryptopanic.com/news/1",
                    "kind": "news",
                    "votes": {"positive": 8, "negative": 1, "important": 1},
                    "currencies": [{"code": "BTC"}],
                }
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "data.sources.cryptopanic_source.aiohttp.ClientSession", return_value=mock_session
        ):
            await src._fetch_and_cache()

        assert len(src._items) == 1
        assert src._items[0].content == "Bitcoin hits new all-time high"
        assert "BTC" in src._items[0].mentioned_assets

    @pytest.mark.asyncio
    async def test_fetch_and_cache_handles_401(self):
        src = CryptoPanicSource(auth_token="bad-key")

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "data.sources.cryptopanic_source.aiohttp.ClientSession", return_value=mock_session
        ):
            await src._fetch_and_cache()  # should not raise

        assert src._items == []


# ── CoinGeckoSource ───────────────────────────────────────────────────────────


class TestCoinGeckoSourceInit:
    def test_always_enabled(self):
        """CoinGeckoSource requires no API key and is always enabled."""
        src = CoinGeckoSource()
        assert src.enabled is True

    def test_name_and_source_type(self):
        src = CoinGeckoSource()
        assert src.name == "coingecko"
        assert src.source_type == DataSourceType.REST_API

    def test_default_polling_interval(self):
        src = CoinGeckoSource()
        assert src._polling_interval == 300

    def test_custom_notable_move_pct(self):
        src = CoinGeckoSource(notable_move_pct=10.0)
        assert src._notable_move_pct == 10.0


class TestCoinGeckoSourceMarketCoinToItem:
    def _make_source(self) -> CoinGeckoSource:
        return CoinGeckoSource()

    def _make_coin(self, change_24h: float = 2.0) -> dict:
        return {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 65_000.0,
            "price_change_percentage_24h": change_24h,
            "market_cap": 1_200_000_000_000.0,
            "total_volume": 30_000_000_000.0,
        }

    def test_returns_data_item(self):
        src = self._make_source()
        item = src._market_coin_to_item(self._make_coin())
        assert item is not None
        assert isinstance(item, DataItem)

    def test_notable_move_flagged(self):
        src = self._make_source()
        item = src._market_coin_to_item(self._make_coin(change_24h=8.0))
        assert item is not None
        assert item.metadata["is_notable"] is True
        assert "[NOTABLE MOVE]" in item.content

    def test_small_move_not_notable(self):
        src = self._make_source()
        item = src._market_coin_to_item(self._make_coin(change_24h=1.5))
        assert item is not None
        assert item.metadata["is_notable"] is False
        assert "[NOTABLE MOVE]" not in item.content

    def test_negative_notable_move(self):
        """A large negative move should also be flagged as notable."""
        src = self._make_source()
        item = src._market_coin_to_item(self._make_coin(change_24h=-7.5))
        assert item is not None
        assert item.metadata["is_notable"] is True

    def test_symbol_in_mentioned_assets(self):
        src = self._make_source()
        item = src._market_coin_to_item(self._make_coin())
        assert item is not None
        assert "BTC" in item.mentioned_assets

    def test_scores_in_valid_range(self):
        src = self._make_source()
        item = src._market_coin_to_item(self._make_coin(change_24h=12.0))
        assert item is not None
        assert 0.0 <= item.relevance_score <= 1.0
        assert 0.0 <= item.urgency_score <= 1.0

    def test_missing_price_change_handled(self):
        src = self._make_source()
        coin = {
            "id": "ethereum",
            "symbol": "eth",
            "name": "Ethereum",
            "current_price": 3_000.0,
            "price_change_percentage_24h": None,
            "market_cap": 400_000_000_000.0,
            "total_volume": 15_000_000_000.0,
        }
        item = src._market_coin_to_item(coin)
        assert item is not None
        assert item.metadata["change_24h_pct"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_fetch_markets_with_mocked_response(self):
        src = CoinGeckoSource()
        mock_data = [
            {
                "id": "bitcoin",
                "symbol": "btc",
                "name": "Bitcoin",
                "current_price": 65_000.0,
                "price_change_percentage_24h": 6.5,
                "market_cap": 1_300_000_000_000.0,
                "total_volume": 40_000_000_000.0,
            }
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "data.sources.coingecko_source.aiohttp.ClientSession", return_value=mock_session
        ):
            items = await src._fetch_markets()

        assert len(items) == 1
        assert items[0].metadata["is_notable"] is True  # 6.5% > 5%
        assert "BTC" in items[0].mentioned_assets

    @pytest.mark.asyncio
    async def test_rate_limit_returns_empty(self):
        src = CoinGeckoSource()

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "data.sources.coingecko_source.aiohttp.ClientSession", return_value=mock_session
        ):
            result = await src._get_json("https://api.coingecko.com/test")

        assert result is None


# ── StrategyManager sentiment modifier ───────────────────────────────────────


def _make_sentiment_item(sentiment_score: float) -> MagicMock:
    """Return a mock DataItem-like object with the given sentiment_score in metadata."""
    item = MagicMock()
    item.metadata = {"sentiment_score": sentiment_score}
    return item


class TestStrategyManagerSentimentModifier:
    def test_compute_modifier_positive_sentiment(self):
        items = [_make_sentiment_item(1.0)]
        modifier = StrategyManager._compute_sentiment_modifier(items)
        assert modifier == pytest.approx(0.10)

    def test_compute_modifier_negative_sentiment(self):
        items = [_make_sentiment_item(-1.0)]
        modifier = StrategyManager._compute_sentiment_modifier(items)
        assert modifier == pytest.approx(-0.10)

    def test_compute_modifier_neutral_sentiment(self):
        items = [_make_sentiment_item(0.0)]
        modifier = StrategyManager._compute_sentiment_modifier(items)
        assert modifier == pytest.approx(0.0)

    def test_compute_modifier_averages_multiple_items(self):
        items = [_make_sentiment_item(1.0), _make_sentiment_item(-1.0)]
        modifier = StrategyManager._compute_sentiment_modifier(items)
        assert modifier == pytest.approx(0.0)

    def test_compute_modifier_empty_list(self):
        modifier = StrategyManager._compute_sentiment_modifier([])
        assert modifier == pytest.approx(0.0)

    def test_compute_modifier_items_without_sentiment_ignored(self):
        item_no_meta = MagicMock()
        item_no_meta.metadata = {}
        modifier = StrategyManager._compute_sentiment_modifier([item_no_meta])
        assert modifier == pytest.approx(0.0)

    def test_compute_modifier_capped_at_plus_10(self):
        # Even with extreme sentiment, modifier should be capped at 0.10
        items = [_make_sentiment_item(100.0)]
        modifier = StrategyManager._compute_sentiment_modifier(items)
        assert modifier == pytest.approx(0.10)

    def test_compute_modifier_capped_at_minus_10(self):
        items = [_make_sentiment_item(-100.0)]
        modifier = StrategyManager._compute_sentiment_modifier(items)
        assert modifier == pytest.approx(-0.10)

    @pytest.mark.asyncio
    async def test_evaluate_all_applies_sentiment_modifier(self):
        """evaluate_all with positive sentiment items should boost confidence."""
        import numpy as np
        import pandas as pd

        mgr = StrategyManager()

        # Inject a concrete strategy that always returns a signal
        class AlwaysLongStrategy:
            name = "always_long"
            enabled = True
            symbols: List[str] = []

            def analyze(self, ohlcv, symbol=""):
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": 50_000.0,
                    "atr": 100.0,
                    "confidence": 0.60,
                    "strategy": "always_long",
                    "timeframe": "15m",
                }

        mgr._strategies = {"always_long": AlwaysLongStrategy()}  # type: ignore[assignment]

        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame(
            {
                "close": rng.uniform(49_000, 51_000, n),
                "open": rng.uniform(49_000, 51_000, n),
                "high": rng.uniform(51_000, 52_000, n),
                "low": rng.uniform(48_000, 49_000, n),
                "volume": rng.uniform(100, 10_000, n),
            }
        )

        positive_items = [_make_sentiment_item(1.0)]  # modifier = +0.10
        signals = await mgr.evaluate_all("BTC/USDT", {"15m": df}, sentiment_items=positive_items)

        assert len(signals) == 1
        # Base confidence 0.60 + modifier 0.10 = 0.70
        assert signals[0]["confidence"] == pytest.approx(0.70)
        assert signals[0]["sentiment_modifier"] == pytest.approx(0.10)

    @pytest.mark.asyncio
    async def test_evaluate_all_without_sentiment_unchanged(self):
        """evaluate_all without sentiment items should not modify confidence."""
        import numpy as np
        import pandas as pd

        mgr = StrategyManager()

        class AlwaysLongStrategy:
            name = "always_long"
            enabled = True
            symbols: List[str] = []

            def analyze(self, ohlcv, symbol=""):
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": 50_000.0,
                    "atr": 100.0,
                    "confidence": 0.60,
                    "strategy": "always_long",
                    "timeframe": "15m",
                }

        mgr._strategies = {"always_long": AlwaysLongStrategy()}  # type: ignore[assignment]

        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame(
            {
                "close": rng.uniform(49_000, 51_000, n),
                "open": rng.uniform(49_000, 51_000, n),
                "high": rng.uniform(51_000, 52_000, n),
                "low": rng.uniform(48_000, 49_000, n),
                "volume": rng.uniform(100, 10_000, n),
            }
        )

        signals = await mgr.evaluate_all("BTC/USDT", {"15m": df})

        assert len(signals) == 1
        assert signals[0]["confidence"] == pytest.approx(0.60)
        assert "sentiment_modifier" not in signals[0]
