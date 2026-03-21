"""Unified LLM client supporting GateRouter, OpenAI, Anthropic, Ollama, Gemini, Grok, and OpenRouter.

Provider fallback chain (default, GateRouter-first when configured):
  GateRouter (DeepSeek V3) → Ollama → Gemini Flash Lite → Gemini Flash → Grok → OpenRouter → OpenAI → Anthropic

When a provider's quota is exhausted (HTTP 429 / 402 or a rate-limit exception),
it is automatically skipped and the next provider in the chain is tried.  A
quota-exceeded provider is re-enabled after ``_QUOTA_RESET_SECONDS`` (default 1 h).
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from utils.rate_limiter import RateLimiter

# Error keywords that signal a quota / rate-limit problem rather than a
# transient network error.  Case-insensitive substring match.
_QUOTA_ERROR_KEYWORDS: Tuple[str, ...] = (
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "insufficient_quota",
    "exceeded",
    "billing",
    "payment required",
    "credits",
    "resource_exhausted",
)


def _is_quota_error(exc: Exception) -> bool:
    """Return True when *exc* looks like a provider quota / rate-limit error."""
    msg = str(exc).lower()
    if any(kw in msg for kw in _QUOTA_ERROR_KEYWORDS):
        return True
    # Check HTTP status codes embedded in common SDK exceptions
    for attr in ("status_code", "status", "code"):
        code = getattr(exc, attr, None)
        if code in (429, 402):
            return True
    return False


class LLMResponse:
    """Structured response from an LLM provider."""

    def __init__(
        self,
        content: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.content = content
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.cost_usd = cost_usd


class LLMClient:
    """
    Unified LLM client with automatic quota-aware provider fallback.

    Default fallback chain (free providers first):
      1. Local Ollama       — free, private, requires local installation
      2. Gemini Flash Lite  — Google free tier, highest quota
      3. Gemini Flash       — Google free tier, stronger model
      4. Grok               — xAI free tier
      5. OpenRouter         — aggregator, some free models available
      6. OpenAI GPT-4o      — paid, best reasoning
      7. Anthropic Claude   — paid, strong backup

    Quota switching
    ---------------
    When a provider returns a quota / rate-limit error (HTTP 429/402, or an
    exception message containing quota keywords) it is marked as *exhausted*.
    Exhausted providers are skipped for ``_QUOTA_RESET_SECONDS`` (default
    3600 s = 1 h), after which they are automatically re-enabled.
    """

    # Cost per 1K tokens (approximate, USD)
    COST_PER_1K: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "ollama": {"input": 0.0, "output": 0.0},
        # Gemini free-tier models — zero cost on free quota
        "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
        "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},
        # Grok free tier
        "grok-3-mini": {"input": 0.0, "output": 0.0},
        # OpenRouter — varies by model; default to zero (user picks model)
        "openrouter": {"input": 0.0, "output": 0.0},
        # GateRouter — pay-per-token via Gate Pay, DeepSeek V3 pricing
        "gaterouter": {"input": 0.00027, "output": 0.0011},
    }

    # How long (seconds) to wait before retrying a quota-exceeded provider
    _QUOTA_RESET_SECONDS: float = 3600.0  # 1 hour

    def __init__(
        self,
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        gemini_api_key: str = "",
        grok_api_key: str = "",
        openrouter_api_key: str = "",
        gaterouter_api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
        openai_model: str = "gpt-4o",
        anthropic_model: str = "claude-3-5-sonnet-20241022",
        gemini_flash_model: str = "gemini-2.5-flash",
        gemini_flash_lite_model: str = "gemini-2.5-flash-lite",
        grok_model: str = "grok-3-mini",
        openrouter_model: str = "mistralai/mistral-7b-instruct:free",
        gaterouter_model: str = "deepseek/deepseek-chat",
        ollama_model: str = "llama3:8b",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        use_local_first: bool = True,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.grok_api_key = grok_api_key
        self.openrouter_api_key = openrouter_api_key
        self.gaterouter_api_key = gaterouter_api_key
        self.ollama_base_url = ollama_base_url
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.gemini_flash_model = gemini_flash_model
        self.gemini_flash_lite_model = gemini_flash_lite_model
        self.grok_model = grok_model
        self.openrouter_model = openrouter_model
        self.gaterouter_model = gaterouter_model
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_first = use_local_first
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._rate_limiter = RateLimiter(requests_per_second=2.0)
        # Response cache: prompt -> (unix_timestamp, result_dict)
        self._response_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._CACHE_TTL_SECONDS: float = 300.0  # 5 minutes
        self._MAX_RETRIES: int = 3
        # Quota tracking: provider_name -> monotonic timestamp when it was exhausted
        # None means not exhausted (or reset has elapsed)
        self._quota_exhausted_at: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Quota tracking helpers
    # ------------------------------------------------------------------

    def _mark_quota_exceeded(self, provider: str) -> None:
        """Record that *provider* has just hit its quota."""
        self._quota_exhausted_at[provider] = time.monotonic()
        logger.warning(
            f"Provider '{provider}' quota exceeded — skipping for "
            f"{self._QUOTA_RESET_SECONDS / 60:.0f} min."
        )

    def _is_provider_available(self, provider: str) -> bool:
        """Return True when *provider* is not currently quota-exhausted."""
        exhausted_at = self._quota_exhausted_at.get(provider)
        if exhausted_at is None:
            return True
        elapsed = time.monotonic() - exhausted_at
        if elapsed >= self._QUOTA_RESET_SECONDS:
            # Reset — provider may be tried again
            del self._quota_exhausted_at[provider]
            logger.info(f"Provider '{provider}' quota reset — re-enabling.")
            return True
        return False

    def get_quota_status(self) -> Dict[str, Any]:
        """Return a snapshot of quota status for all tracked providers.

        Returns a dict mapping provider name to one of:
        - ``"available"``     — not currently exhausted
        - ``"exhausted(Xm)"`` — exhausted, Xm minutes remaining until reset
        """
        now = time.monotonic()
        status: Dict[str, Any] = {}
        for provider, exhausted_at in self._quota_exhausted_at.items():
            remaining = self._QUOTA_RESET_SECONDS - (now - exhausted_at)
            if remaining > 0:
                status[provider] = f"exhausted({remaining / 60:.0f}m remaining)"
            else:
                status[provider] = "available"
        return status

    # ------------------------------------------------------------------
    # Provider list builder
    # ------------------------------------------------------------------

    def _build_provider_list(self) -> List[Tuple[str, Callable]]:
        """Build the ordered provider list based on configured keys.

        When GateRouter API key is set, it becomes the **top-priority** provider
        (DeepSeek V3 premium). Free providers (Ollama, Gemini, Grok, OpenRouter)
        are placed before paid providers (OpenAI, Anthropic) so that the bot
        avoids charges until all free options are exhausted.
        """
        providers: List[Tuple[str, Callable]] = []

        # GateRouter.ai — HIGHEST PRIORITY when configured (user has premium plan)
        if self.gaterouter_api_key:
            providers.append(("gaterouter", self._query_gaterouter))

        if self.use_local_first and self.ollama_base_url:
            providers.append(("ollama", self._query_ollama))

        if self.gemini_api_key:
            providers.append(("gemini_flash_lite", self._query_gemini_flash_lite))
            providers.append(("gemini_flash", self._query_gemini_flash))

        if self.grok_api_key:
            providers.append(("grok", self._query_grok))

        if self.openrouter_api_key:
            providers.append(("openrouter", self._query_openrouter))

        if self.openai_api_key:
            providers.append(("openai", self._query_openai))

        if self.anthropic_api_key:
            providers.append(("anthropic", self._query_anthropic))

        if not self.use_local_first and self.ollama_base_url:
            if not any(name == "ollama" for name, _ in providers):
                providers.append(("ollama", self._query_ollama))

        return providers

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    async def query(
        self,
        prompt: str,
        system_prompt: str = "You are an expert cryptocurrency trading analyst.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Query LLM with automatic quota-aware provider fallback.

        Tries providers in order (free-first by default).  When a provider
        raises a quota / rate-limit error it is marked as exhausted and
        skipped; the next available provider is tried instead.  Non-quota
        errors (network issues, bad responses) are logged and also cause a
        fallback to the next provider.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        providers = self._build_provider_list()

        for provider_name, provider_fn in providers:
            if not self._is_provider_available(provider_name):
                logger.debug(f"Skipping quota-exhausted provider: {provider_name}")
                continue
            try:
                await self._rate_limiter.acquire()
                response = await provider_fn(prompt, system_prompt, temp, tokens, json_mode)
                logger.debug(
                    f"LLM response from {provider_name}: {len(response)} chars"
                )
                return response
            except Exception as e:
                if _is_quota_error(e):
                    self._mark_quota_exceeded(provider_name)
                else:
                    logger.warning(
                        f"LLM provider {provider_name} failed: {e}, trying next..."
                    )

        return '{"error": "All LLM providers failed"}'

    async def query_json(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Query LLM and parse the response as JSON.

        Strips markdown fences if present before attempting to parse.
        """
        if not system_prompt:
            system_prompt = "You are a trading analyst. Always respond with valid JSON only."
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no other text."
        response = await self.query(full_prompt, system_prompt, json_mode=True)
        try:
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return {"error": "Invalid JSON response", "raw": response[:200]}

    async def analyze_market(self, prompt: str) -> Dict[str, Any]:
        """Send a structured market-analysis prompt and return the parsed JSON response.

        Implements:
        - Response caching: identical prompts within 5 minutes return the cached result.
        - Retry logic: up to 3 attempts with exponential back-off (1 s, 2 s, 4 s).

        Args:
            prompt: The market analysis prompt to send to the LLM.

        Returns:
            Parsed JSON dict.  On unrecoverable failure returns
            ``{"error": "<reason>"}``.
        """
        # Cache lookup
        now = time.monotonic()
        cached = self._response_cache.get(prompt)
        if cached is not None:
            cached_at, cached_result = cached
            if now - cached_at < self._CACHE_TTL_SECONDS:
                logger.debug("analyze_market: cache hit (age={:.0f}s)", now - cached_at)
                return cached_result

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                result = await self.query_json(prompt)
                if "error" not in result:
                    self._response_cache[prompt] = (time.monotonic(), result)
                    return result
                # Treat an LLM-level error as a retriable failure
                last_exc = Exception(result.get("error", "LLM error"))
            except Exception as exc:
                last_exc = exc

            if attempt < self._MAX_RETRIES:
                backoff = 2 ** (attempt - 1)  # 1 s, 2 s, 4 s
                logger.warning(
                    "analyze_market attempt {}/{} failed: {} — retrying in {}s",
                    attempt,
                    self._MAX_RETRIES,
                    last_exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        logger.error("analyze_market failed after {} attempts: {}", self._MAX_RETRIES, last_exc)
        return {"error": str(last_exc)}

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    async def _query_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query OpenAI ChatCompletion API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)
        kwargs: Dict[str, Any] = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        cost = (
            (
                usage.prompt_tokens * self.COST_PER_1K[self.openai_model]["input"] / 1000
                + usage.completion_tokens * self.COST_PER_1K[self.openai_model]["output"] / 1000
            )
            if usage
            else 0.0
        )
        self._total_cost += cost
        self._total_tokens += tokens_used
        return response.choices[0].message.content or ""

    async def _query_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query Anthropic Messages API."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.anthropic_api_key)
        effective_prompt = f"{prompt}\n\nRespond with valid JSON only." if json_mode else prompt
        message = await client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": effective_prompt}],
        )
        tokens_used = message.usage.input_tokens + message.usage.output_tokens
        cost = (
            message.usage.input_tokens * self.COST_PER_1K[self.anthropic_model]["input"] / 1000
            + message.usage.output_tokens * self.COST_PER_1K[self.anthropic_model]["output"] / 1000
        )
        self._total_cost += cost
        self._total_tokens += tokens_used
        return message.content[0].text if message.content else ""

    async def _query_ollama(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query local Ollama chat endpoint."""
        import aiohttp

        url = f"{self.ollama_base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama returned HTTP {resp.status}")
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    async def _query_gemini(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query Google Gemini via its OpenAI-compatible REST endpoint.

        Gemini exposes an OpenAI-compatible API at
        ``https://generativelanguage.googleapis.com/v1beta/openai/``,
        so we can reuse the ``openai.AsyncOpenAI`` client with a custom
        ``base_url``.  This avoids pulling in the ``google-generativeai``
        package as a hard dependency.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        self._total_tokens += tokens_used
        # Gemini free tier — no cost tracking needed, but keep structure consistent
        return response.choices[0].message.content or ""

    async def _query_gemini_flash(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query Gemini 2.5 Flash (more capable free-tier model)."""
        return await self._query_gemini(
            self.gemini_flash_model, prompt, system_prompt, temperature, max_tokens, json_mode
        )

    async def _query_gemini_flash_lite(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query Gemini 2.5 Flash Lite (highest free-tier quota)."""
        return await self._query_gemini(
            self.gemini_flash_lite_model, prompt, system_prompt, temperature, max_tokens, json_mode
        )

    async def _query_grok(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query xAI Grok via its OpenAI-compatible API endpoint.

        Grok exposes an OpenAI-compatible API at ``https://api.x.ai/v1``.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.grok_api_key,
            base_url="https://api.x.ai/v1",
        )
        kwargs: Dict[str, Any] = {
            "model": self.grok_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        self._total_tokens += tokens_used
        return response.choices[0].message.content or ""

    async def _query_openrouter(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query OpenRouter — an AI aggregator with many free models.

        OpenRouter exposes an OpenAI-compatible API at
        ``https://openrouter.ai/api/v1``.  Set ``OPENROUTER_MODEL`` in
        ``.env`` to choose the model (default:
        ``mistralai/mistral-7b-instruct:free``).
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/trading-bot",
                "X-Title": "CryptoTradingBot",
            },
        )
        kwargs: Dict[str, Any] = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        self._total_tokens += tokens_used
        return response.choices[0].message.content or ""

    async def _query_gaterouter(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Query GateRouter.ai — OpenAI-compatible LLM router by Gate.io.

        GateRouter exposes an OpenAI-compatible API at
        ``https://api.gaterouter.ai/openai``.  Supports smart routing (model="auto")
        or explicit model selection (e.g. ``deepseek/deepseek-chat`` for DeepSeek V3).
        Payment is via Gate Pay credits — no extra setup needed.

        API path: ``/openai/v1`` (not ``/v1``).
        Model ID format: ``provider/model-name`` (e.g. ``deepseek/deepseek-chat``).
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.gaterouter_api_key,
            base_url="https://api.gaterouter.ai/openai",
        )
        kwargs: Dict[str, Any] = {
            "model": self.gaterouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        cost = (
            (
                usage.prompt_tokens * self.COST_PER_1K["gaterouter"]["input"] / 1000
                + usage.completion_tokens * self.COST_PER_1K["gaterouter"]["output"] / 1000
            )
            if usage
            else 0.0
        )
        self._total_cost += cost
        self._total_tokens += tokens_used
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost in USD across all queries."""
        return self._total_cost

    @property
    def total_tokens_used(self) -> int:
        """Cumulative token count across all queries."""
        return self._total_tokens
