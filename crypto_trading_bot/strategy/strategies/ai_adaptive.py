"""AI-adaptive strategy — reads regime state from the background service.

## Architecture (Deterministic Edge AI)

The previous design called the LLM **synchronously in the hot path** (every
``generate_signal()`` call).  This created unavoidable 200-2000 ms latency
spikes that caused missed entries, widened fills, and wasted API credits.

The new design **completely decouples** LLM calls from signal generation:

1. :class:`~ai.regime_service.RegimeService` runs as a background task every
   ``REGIME_INTERVAL_SECONDS`` (default 5 min).  It calls the LLM, sentiment
   analyzer, and all slow data sources, then writes a
   :class:`~ai.regime_service.RegimeState` snapshot to
   ``/dev/shm/regime_state.json`` (or an in-memory object).

2. This strategy reads the **pre-computed** ``RegimeState`` in O(1) time.
   Zero LLM calls, zero network I/O, zero latency in the hot path.

3. The ``AIBrain`` instance is still supported for backward compatibility but
   is **never** called during signal generation.  It is still available to the
   regime service (slow loop).

Fallback behaviour
------------------
If the regime service is unavailable or its state is stale, the strategy
falls back to a **neutral signal** rather than gambling on an outdated LLM
response.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


def _load_regime_state_from_file(path: str) -> Optional[Dict[str, Any]]:
    """Load the latest regime state JSON from *path*.

    Returns ``None`` if the file is absent, unreadable, or empty.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


class AIAdaptiveStrategy(BaseStrategy):
    """Regime-aware signal generator backed by the background AI regime service.

    This strategy generates signals by reading the pre-computed
    :class:`~ai.regime_service.RegimeState` (written every 5 minutes by the
    background service) instead of calling the LLM in the hot path.

    Parameters
    ----------
    symbols:           Symbols to trade.
    timeframe:         OHLCV timeframe (informational; not used for LLM calls).
    enabled:           Whether the strategy is active.
    ai_brain:          Legacy AIBrain instance (retained for backward compat).
                       No longer called during signal generation.
    regime_service:    :class:`~ai.regime_service.RegimeService` instance.
                       If provided, ``get_current_state()`` is called directly.
    regime_state_path: Path to the JSON file written by the regime service.
                       Default: env ``REGIME_OUTPUT_PATH`` or
                       ``/dev/shm/regime_state.json``.
    regime_stale_ttl:  Seconds after which a cached regime state is considered
                       stale and a neutral signal is returned (default: 900).
    """

    _DEFAULT_REGIME_PATH = "/dev/shm/regime_state.json"

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        ai_brain: Optional[Any] = None,
        regime_service: Optional[Any] = None,
        regime_state_path: Optional[str] = None,
        regime_stale_ttl: int = 900,
    ) -> None:
        super().__init__(
            name="ai_adaptive",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        # Legacy — kept for backward compatibility; not used in hot path
        self._ai_brain = ai_brain
        self._regime_service = regime_service
        self._regime_state_path = regime_state_path or os.environ.get(
            "REGIME_OUTPUT_PATH", self._DEFAULT_REGIME_PATH
        )
        self._regime_stale_ttl = regime_stale_ttl

        # In-memory cache to avoid repeated disk reads within the same
        # strategy evaluation cycle.
        self._cached_regime: Optional[Dict[str, Any]] = None
        self._cached_regime_ts: float = 0.0
        self._cache_ttl_s: float = 30.0  # 30-second local in-memory TTL

    # ------------------------------------------------------------------
    # Setters for dependency injection
    # ------------------------------------------------------------------

    def set_ai_brain(self, brain: Any) -> None:
        """Inject the legacy AIBrain instance (used only by the regime service)."""
        self._ai_brain = brain

    def set_regime_service(self, service: Any) -> None:
        """Inject the :class:`~ai.regime_service.RegimeService` instance."""
        self._regime_service = service

    # ------------------------------------------------------------------
    # Hot-path signal generation (O(1), no LLM calls)
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        """Generate a trading signal from the current regime state.

        This method never calls the LLM.  It reads the pre-computed
        ``RegimeState`` and returns a signal consistent with the current
        macro regime.
        """
        regime = self._get_regime_state()

        if regime is None:
            return self._neutral_signal(symbol, "Regime state unavailable")

        # Check staleness
        ts_ms = regime.get("timestamp_ms", 0)
        if ts_ms > 0:
            age_s = time.time() - ts_ms / 1000.0
            if age_s > self._regime_stale_ttl:
                return self._neutral_signal(
                    symbol, f"Regime state stale ({age_s:.0f}s old)"
                )

        overall = regime.get("overall_regime", "unknown")
        vol_regime = regime.get("volatility_regime", "high")
        sentiment = float(regime.get("sentiment_score", 0.0))
        confidence = float(regime.get("sentiment_confidence", 0.0))
        position_scale = float(regime.get("recommended_position_scale", 0.5))
        blocked = regime.get("blocked_strategies", [])

        # Hard block
        if "ai_adaptive" in blocked or overall == "unknown":
            return self._neutral_signal(symbol, f"Regime '{overall}' blocks ai_adaptive")

        # Extreme volatility guard
        if vol_regime in ("extreme", "high") and position_scale < 0.25:
            return self._neutral_signal(
                symbol, f"Volatility regime '{vol_regime}' — no new positions"
            )

        # Map regime + sentiment to a direction
        direction = self._regime_to_direction(overall, sentiment)
        if direction == "neutral":
            return self._neutral_signal(symbol, f"Regime '{overall}' → hold")

        # Scale confidence by the regime's position scale and sentiment confidence
        effective_confidence = max(0.1, min(1.0, confidence * position_scale))

        # Determine leverage (honour override if present)
        max_lev = regime.get("max_leverage_override")
        leverage = 2 if max_lev is None else min(2, int(max_lev))

        reasoning = (
            f"Regime={overall} vol={vol_regime} "
            f"sentiment={sentiment:.2f} scale={position_scale:.2f}"
        )
        logger.debug(f"[{self.name}] {symbol}: {direction} ({reasoning})")

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=effective_confidence,
            confidence=effective_confidence,
            strategy_name=self.name,
            reasoning=reasoning,
            leverage=leverage,
        )

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close positions when the regime flips against them."""
        regime = self._get_regime_state()
        if regime is None:
            return False

        overall = regime.get("overall_regime", "unknown")
        side = str(getattr(position, "side", "long")).lower()

        # Close longs when regime turns bearish or unknown
        if side == "long" and overall in ("trending_bearish", "unknown"):
            logger.info(
                f"[{self.name}] Closing long on {getattr(position, 'symbol', '?')} "
                f"— regime={overall}"
            )
            return True

        # Close shorts when regime turns bullish or unknown
        if side == "short" and overall in ("trending_bullish", "unknown"):
            logger.info(
                f"[{self.name}] Closing short on {getattr(position, 'symbol', '?')} "
                f"— regime={overall}"
            )
            return True

        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        """Return order parameters scaled by the current regime recommendation."""
        defaults: Dict[str, Any] = {
            "position_size_pct": 0.05,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "leverage": 2,
        }
        regime = self._get_regime_state()
        if regime is None:
            return defaults

        scale = float(regime.get("recommended_position_scale", 0.5))
        max_lev = regime.get("max_leverage_override")
        leverage = max_lev if max_lev is not None else defaults["leverage"]

        # Widen stops and targets in volatile regimes
        vol = regime.get("volatility_regime", "moderate")
        sl_mult = 1.5 if vol in ("high", "extreme") else 1.0
        tp_mult = 1.5 if vol in ("high", "extreme") else 1.0

        return {
            "position_size_pct": round(defaults["position_size_pct"] * scale, 4),
            "stop_loss_pct": round(defaults["stop_loss_pct"] * sl_mult, 4),
            "take_profit_pct": round(defaults["take_profit_pct"] * tp_mult, 4),
            "leverage": int(leverage),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_regime_state(self) -> Optional[Dict[str, Any]]:
        """Return the latest regime state dict (in-memory or from file/service).

        Uses a 30-second in-process cache to avoid repeated disk reads.
        """
        now = time.time()

        # 1. Try the injected regime service first (no disk I/O)
        if self._regime_service is not None:
            try:
                state = self._regime_service.current_state
                if state is not None:
                    return state.model_dump() if hasattr(state, "model_dump") else dict(state)
            except Exception:
                pass

        # 2. Use the in-memory cache if fresh
        if self._cached_regime is not None and (now - self._cached_regime_ts) < self._cache_ttl_s:
            return self._cached_regime

        # 3. Read from the shared-memory file written by the regime service
        data = _load_regime_state_from_file(self._regime_state_path)
        if data is not None:
            self._cached_regime = data
            self._cached_regime_ts = now
            return data

        return None

    @staticmethod
    def _regime_to_direction(overall_regime: str, sentiment_score: float) -> str:
        """Map the overall regime + sentiment to ``"long"``, ``"short"``, or
        ``"neutral"``."""
        if overall_regime == "trending_bullish":
            return "long"
        if overall_regime == "trending_bearish":
            return "short"
        if overall_regime in ("ranging", "high_volatility"):
            # In ranging markets follow sentiment if it's strong enough
            if sentiment_score > 0.3:
                return "long"
            if sentiment_score < -0.3:
                return "short"
        return "neutral"

