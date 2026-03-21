"""Tests for the extended LLMClient: new AI providers and quota-switching."""

from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.llm_client import LLMClient, _is_quota_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**kwargs) -> LLMClient:
    """Return an LLMClient configured with fake keys for all providers."""
    defaults = dict(
        openai_api_key="sk-openai-test",
        anthropic_api_key="sk-anthropic-test",
        gemini_api_key="AIzaGeminiTest",
        grok_api_key="xai-grok-test",
        openrouter_api_key="sk-or-test",
        use_local_first=False,
    )
    defaults.update(kwargs)
    return LLMClient(**defaults)


def _openai_like_response(content: str = "hello") -> MagicMock:
    """Return a minimal mock mimicking an OpenAI-compatible chat response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.total_tokens = 10
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# _is_quota_error helper
# ---------------------------------------------------------------------------


class TestIsQuotaError:
    def test_rate_limit_keyword(self):
        assert _is_quota_error(Exception("rate limit exceeded"))

    def test_429_status_code(self):
        exc = Exception("too many requests")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_quota_error(exc)

    def test_402_status_code(self):
        exc = Exception("payment required")
        exc.status_code = 402  # type: ignore[attr-defined]
        assert _is_quota_error(exc)

    def test_quota_keyword(self):
        assert _is_quota_error(Exception("insufficient_quota"))

    def test_resource_exhausted(self):
        assert _is_quota_error(Exception("RESOURCE_EXHAUSTED: quota exceeded"))

    def test_billing_keyword(self):
        assert _is_quota_error(Exception("billing account inactive"))

    def test_generic_error_not_quota(self):
        assert not _is_quota_error(Exception("network timeout"))

    def test_connection_error_not_quota(self):
        assert not _is_quota_error(ConnectionError("connection refused"))


# ---------------------------------------------------------------------------
# Quota tracking mechanism
# ---------------------------------------------------------------------------


class TestQuotaTracking:
    def test_mark_quota_exceeded_makes_provider_unavailable(self):
        client = _make_client()
        assert client._is_provider_available("openai")
        client._mark_quota_exceeded("openai")
        assert not client._is_provider_available("openai")

    def test_quota_resets_after_timeout(self):
        client = _make_client()
        client._QUOTA_RESET_SECONDS = 0.01  # very short for test
        client._mark_quota_exceeded("openai")
        assert not client._is_provider_available("openai")
        time.sleep(0.02)
        assert client._is_provider_available("openai")

    def test_unknown_provider_is_available(self):
        client = _make_client()
        assert client._is_provider_available("unknown_provider")

    def test_get_quota_status_shows_exhausted(self):
        client = _make_client()
        client._mark_quota_exceeded("openai")
        status = client.get_quota_status()
        assert "openai" in status
        assert "exhausted" in status["openai"]

    def test_get_quota_status_empty_when_no_exhausted(self):
        client = _make_client()
        assert client.get_quota_status() == {}

    def test_multiple_providers_tracked_independently(self):
        client = _make_client()
        client._mark_quota_exceeded("openai")
        client._mark_quota_exceeded("anthropic")
        assert not client._is_provider_available("openai")
        assert not client._is_provider_available("anthropic")
        assert client._is_provider_available("gemini_flash")


# ---------------------------------------------------------------------------
# Provider list builder
# ---------------------------------------------------------------------------


class TestBuildProviderList:
    def test_all_providers_present_when_all_keys_set(self):
        client = _make_client(use_local_first=False)
        names = [name for name, _ in client._build_provider_list()]
        assert "gemini_flash_lite" in names
        assert "gemini_flash" in names
        assert "grok" in names
        assert "openrouter" in names
        assert "openai" in names
        assert "anthropic" in names

    def test_gemini_flash_lite_before_gemini_flash(self):
        """Flash Lite (higher free quota) should appear before Flash."""
        client = _make_client(use_local_first=False)
        names = [name for name, _ in client._build_provider_list()]
        assert names.index("gemini_flash_lite") < names.index("gemini_flash")

    def test_free_providers_before_paid(self):
        """Gemini / Grok / OpenRouter appear before OpenAI / Anthropic."""
        client = _make_client(use_local_first=False)
        names = [name for name, _ in client._build_provider_list()]
        last_free = max(
            names.index("gemini_flash_lite"),
            names.index("gemini_flash"),
            names.index("grok"),
            names.index("openrouter"),
        )
        first_paid = min(names.index("openai"), names.index("anthropic"))
        assert last_free < first_paid

    def test_ollama_first_when_use_local_first(self):
        client = _make_client(use_local_first=True, ollama_base_url="http://localhost:11434")
        names = [name for name, _ in client._build_provider_list()]
        assert names[0] == "ollama"

    def test_no_openai_when_key_absent(self):
        client = LLMClient(openai_api_key="", gemini_api_key="AIzaTest", use_local_first=False)
        names = [name for name, _ in client._build_provider_list()]
        assert "openai" not in names

    def test_no_gemini_when_key_absent(self):
        client = LLMClient(openai_api_key="sk-test", gemini_api_key="", use_local_first=False)
        names = [name for name, _ in client._build_provider_list()]
        assert "gemini_flash" not in names
        assert "gemini_flash_lite" not in names


# ---------------------------------------------------------------------------
# Automatic quota-aware switching in query()
# ---------------------------------------------------------------------------


class TestQueryQuotaSwitching:
    @pytest.mark.asyncio
    async def test_skips_quota_exhausted_provider_uses_next(self):
        """When the first provider is quota-exhausted, the next is used."""
        client = _make_client(use_local_first=False)

        call_log = []

        async def fake_gemini_flash_lite(p, sp, t, mt, jm):
            call_log.append("gemini_flash_lite")
            raise Exception("429 quota exceeded")

        async def fake_gemini_flash(p, sp, t, mt, jm):
            call_log.append("gemini_flash")
            return "response from gemini_flash"

        with (
            patch.object(client, "_query_gemini_flash_lite", fake_gemini_flash_lite),
            patch.object(client, "_query_gemini_flash", fake_gemini_flash),
            patch.object(client, "_query_grok", AsyncMock(side_effect=Exception("skip"))),
            patch.object(client, "_query_openrouter", AsyncMock(side_effect=Exception("skip"))),
            patch.object(client, "_query_openai", AsyncMock(side_effect=Exception("skip"))),
            patch.object(client, "_query_anthropic", AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await client.query("test prompt")

        assert result == "response from gemini_flash"
        assert "gemini_flash_lite" in call_log
        # gemini_flash_lite should now be marked as quota-exceeded
        assert not client._is_provider_available("gemini_flash_lite")

    @pytest.mark.asyncio
    async def test_returns_error_when_all_providers_quota_exceeded(self):
        """Returns error sentinel when every provider is quota-exhausted."""
        client = _make_client(use_local_first=False)

        async def quota_fail(p, sp, t, mt, jm):
            raise Exception("429 too many requests")

        with (
            patch.object(client, "_query_gemini_flash_lite", quota_fail),
            patch.object(client, "_query_gemini_flash", quota_fail),
            patch.object(client, "_query_grok", quota_fail),
            patch.object(client, "_query_openrouter", quota_fail),
            patch.object(client, "_query_openai", quota_fail),
            patch.object(client, "_query_anthropic", quota_fail),
        ):
            result = await client.query("test prompt")

        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_skips_already_marked_exhausted_provider(self):
        """A provider pre-marked as exhausted is never called."""
        client = _make_client(use_local_first=False)
        client._mark_quota_exceeded("gemini_flash_lite")

        gemini_flash_lite_called = False

        async def should_not_be_called(p, sp, t, mt, jm):
            nonlocal gemini_flash_lite_called
            gemini_flash_lite_called = True
            return "bad"

        async def good_response(p, sp, t, mt, jm):
            return "good response"

        with (
            patch.object(client, "_query_gemini_flash_lite", should_not_be_called),
            patch.object(client, "_query_gemini_flash", good_response),
        ):
            result = await client.query("test prompt")

        assert result == "good response"
        assert not gemini_flash_lite_called

    @pytest.mark.asyncio
    async def test_non_quota_error_does_not_mark_exhausted(self):
        """A plain network error does NOT mark the provider as quota-exhausted."""
        client = _make_client(use_local_first=False)

        async def network_error(p, sp, t, mt, jm):
            raise ConnectionError("connection reset")

        async def good_response(p, sp, t, mt, jm):
            return "fallback ok"

        with (
            patch.object(client, "_query_gemini_flash_lite", network_error),
            patch.object(client, "_query_gemini_flash", good_response),
        ):
            await client.query("test prompt")

        # gemini_flash_lite should NOT be quota-exhausted — it was a network error
        assert client._is_provider_available("gemini_flash_lite")


# ---------------------------------------------------------------------------
# New provider smoke tests (calls mocked at the HTTP client level)
# ---------------------------------------------------------------------------


class TestNewProviderMethods:
    """Verify the new _query_* methods call the right endpoints and parse responses."""

    def _mock_openai_client(self, content: str = "test response") -> MagicMock:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_openai_like_response(content)
        )
        return mock_client

    @pytest.mark.asyncio
    async def test_query_gemini_flash_lite_calls_correct_model(self):
        client = _make_client()
        mock_openai_instance = self._mock_openai_client("gemini lite response")

        with patch("openai.AsyncOpenAI", return_value=mock_openai_instance):
            result = await client._query_gemini_flash_lite("prompt", "system", 0.3, 100, False)

        assert result == "gemini lite response"
        call_kwargs = mock_openai_instance.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == client.gemini_flash_lite_model

    @pytest.mark.asyncio
    async def test_query_gemini_flash_calls_correct_model(self):
        client = _make_client()
        mock_openai_instance = self._mock_openai_client("gemini flash response")

        with patch("openai.AsyncOpenAI", return_value=mock_openai_instance):
            result = await client._query_gemini_flash("prompt", "system", 0.3, 100, False)

        assert result == "gemini flash response"
        call_kwargs = mock_openai_instance.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == client.gemini_flash_model

    @pytest.mark.asyncio
    async def test_query_grok_uses_xai_base_url(self):
        client = _make_client()
        mock_openai_instance = self._mock_openai_client("grok response")

        with patch("openai.AsyncOpenAI", return_value=mock_openai_instance) as MockOpenAI:
            result = await client._query_grok("prompt", "system", 0.3, 100, False)

        assert result == "grok response"
        init_kwargs = MockOpenAI.call_args.kwargs
        assert "x.ai" in init_kwargs.get("base_url", "")

    @pytest.mark.asyncio
    async def test_query_openrouter_uses_openrouter_base_url(self):
        client = _make_client()
        mock_openai_instance = self._mock_openai_client("openrouter response")

        with patch("openai.AsyncOpenAI", return_value=mock_openai_instance) as MockOpenAI:
            result = await client._query_openrouter("prompt", "system", 0.3, 100, False)

        assert result == "openrouter response"
        init_kwargs = MockOpenAI.call_args.kwargs
        assert "openrouter" in init_kwargs.get("base_url", "")

    @pytest.mark.asyncio
    async def test_query_gemini_uses_google_base_url(self):
        client = _make_client()
        mock_openai_instance = self._mock_openai_client("gemini base url response")

        with patch("openai.AsyncOpenAI", return_value=mock_openai_instance) as MockOpenAI:
            result = await client._query_gemini(
                "gemini-2.5-flash", "prompt", "system", 0.3, 100, False
            )

        assert result == "gemini base url response"
        init_kwargs = MockOpenAI.call_args.kwargs
        base_url = init_kwargs.get("base_url", "")
        # Gemini uses Google's generativelanguage API endpoint
        assert base_url.startswith("https://generativelanguage.googleapis.com/")


# ---------------------------------------------------------------------------
# Settings — new API key fields
# ---------------------------------------------------------------------------


class TestSettingsNewAPIKeys:
    def test_gemini_key_field_defaults_none(self):
        from config.settings import Settings

        s = Settings(TRADING_MODE="paper")
        assert s.gemini_api_key is None

    def test_grok_key_field_defaults_none(self):
        from config.settings import Settings

        s = Settings(TRADING_MODE="paper")
        assert s.grok_api_key is None

    def test_openrouter_key_field_defaults_none(self):
        from config.settings import Settings

        s = Settings(TRADING_MODE="paper")
        assert s.openrouter_api_key is None

    def test_gemini_key_populated_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
        from config.settings import Settings

        s = Settings(TRADING_MODE="paper")
        assert s.gemini_api_key == "AIza-test-key"

    def test_grok_key_populated_from_env(self, monkeypatch):
        monkeypatch.setenv("GROK_API_KEY", "xai-test-key")
        from config.settings import Settings

        s = Settings(TRADING_MODE="paper")
        assert s.grok_api_key == "xai-test-key"

    def test_openrouter_key_populated_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        from config.settings import Settings

        s = Settings(TRADING_MODE="paper")
        assert s.openrouter_api_key == "sk-or-test-key"


# ---------------------------------------------------------------------------
# AIConfig — new model fields
# ---------------------------------------------------------------------------


class TestAIConfigNewModels:
    def test_gemini_flash_model_default(self):
        from config.settings import AIConfig

        cfg = AIConfig()
        assert cfg.gemini_flash_model == "gemini-2.5-flash"

    def test_gemini_flash_lite_model_default(self):
        from config.settings import AIConfig

        cfg = AIConfig()
        assert cfg.gemini_flash_lite_model == "gemini-2.5-flash-lite"

    def test_grok_model_default(self):
        from config.settings import AIConfig

        cfg = AIConfig()
        assert cfg.grok_model == "grok-3-mini"

    def test_openrouter_model_default(self):
        from config.settings import AIConfig

        cfg = AIConfig()
        assert "free" in cfg.openrouter_model.lower() or "mistral" in cfg.openrouter_model.lower()
