"""AI-enhanced trade execution module.

Wraps the standard :class:`~.trade_executor.TradeExecutor` with an AI confidence
gate so that signals with low AI-assessed confidence can be filtered out or
size-reduced before execution.

The AI gate is **optional** — if no AI brain is provided the module behaves
identically to the raw ``TradeExecutor``.

Usage::

    from execution.ai_trade_execution import AITradeExecutor

    executor = AITradeExecutor(
        exchange=exchange,
        order_manager=order_manager,
        position_manager=position_manager,
        ai_brain=ai_brain,          # optional
        min_ai_confidence=0.55,     # signals below this are skipped
    )
    result = await executor.execute_trade(signal)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from .trade_executor import TradeExecutor

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange
    from exchange.order_manager import OrderManager
    from exchange.position_manager import PositionManager

# Default threshold below which the AI gate blocks trade execution.
_DEFAULT_MIN_AI_CONFIDENCE: float = 0.50
# Confidence boost when AI direction aligns with signal direction.
_AI_CONFIDENCE_BOOST: float = 0.15
# Confidence penalty when AI direction conflicts with signal direction.
_AI_CONFIDENCE_PENALTY: float = -0.15


class AITradeExecutor(TradeExecutor):
    """Trade executor with an optional AI confidence gate.

    Extends :class:`~.trade_executor.TradeExecutor` by injecting the AI
    brain's market assessment as a pre-execution filter.  Signals whose
    effective confidence falls below *min_ai_confidence* are logged and
    skipped without placing any orders.

    Args:
        exchange: Connected exchange instance (paper or live).
        order_manager: :class:`~exchange.order_manager.OrderManager` instance.
        position_manager: :class:`~exchange.position_manager.PositionManager` instance.
        ai_brain: Optional :class:`~ai.brain.AIBrain` instance.  When ``None``
            the executor behaves identically to the base ``TradeExecutor``.
        min_ai_confidence: Minimum AI confidence score (0–1) required to
            allow trade execution.  Defaults to 0.50.
    """

    def __init__(
        self,
        exchange: "BaseExchange",
        order_manager: "OrderManager",
        position_manager: "PositionManager",
        ai_brain: Optional[Any] = None,
        min_ai_confidence: float = _DEFAULT_MIN_AI_CONFIDENCE,
    ) -> None:
        super().__init__(
            exchange=exchange,
            order_manager=order_manager,
            position_manager=position_manager,
        )
        self._ai_brain = ai_brain
        self._min_ai_confidence = min_ai_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validated trade signal with an optional AI confidence gate.

        If an AI brain is configured, this method queries it for a market
        confidence modifier.  When the resulting confidence is below
        *min_ai_confidence* the trade is skipped and a structured failure
        result is returned.

        Args:
            signal: Trade signal dict (see :meth:`TradeExecutor.execute_trade`).

        Returns:
            Execution result dict (see :meth:`TradeExecutor.execute_trade`).
            On AI-gated skip the dict contains
            ``{"success": False, "skipped": True, "reason": "ai_confidence_too_low"}``.
        """
        if self._ai_brain is not None:
            try:
                ai_result = await self._ai_brain.get_confidence_modifier(
                    market_overview={"symbols": [signal.get("symbol", "")]},
                    sentiment_score=float(signal.get("sentiment_score", 0.0)),
                    news_summary=signal.get("news_summary", ""),
                )
                if ai_result is not None:
                    ai_confidence = float(ai_result.get("confidence", 1.0))
                    ai_direction = str(ai_result.get("direction", "neutral")).lower()
                    signal_direction = str(signal.get("direction", "long")).lower()

                    # Direction conflict: AI bullish but signal is short (or vice versa)
                    direction_conflict = (
                        (ai_direction == "bullish" and signal_direction == "short")
                        or (ai_direction == "bearish" and signal_direction == "long")
                    )

                    # Apply AI modifier to signal's confidence
                    base_confidence = float(signal.get("confidence", 0.5))
                    modifier = _AI_CONFIDENCE_BOOST if not direction_conflict else _AI_CONFIDENCE_PENALTY
                    effective_confidence = max(0.0, min(1.0, base_confidence + modifier * ai_confidence))
                    signal = {**signal, "confidence": effective_confidence}

                    logger.debug(
                        "AITradeExecutor: symbol={} base_conf={:.3f} ai_conf={:.3f} "
                        "ai_dir={} effective_conf={:.3f}",
                        signal.get("symbol"),
                        base_confidence,
                        ai_confidence,
                        ai_direction,
                        effective_confidence,
                    )

                    if effective_confidence < self._min_ai_confidence:
                        logger.info(
                            "AITradeExecutor: signal SKIPPED — effective confidence {:.3f} < "
                            "threshold {:.3f} (symbol={} direction={})",
                            effective_confidence,
                            self._min_ai_confidence,
                            signal.get("symbol"),
                            signal.get("direction"),
                        )
                        return {
                            "success": False,
                            "skipped": True,
                            "reason": "ai_confidence_too_low",
                            "symbol": signal.get("symbol", ""),
                            "direction": signal.get("direction", ""),
                            "effective_confidence": effective_confidence,
                            "min_required": self._min_ai_confidence,
                        }
            except Exception as exc:
                # AI gate failure is non-fatal — proceed with original signal
                logger.warning(
                    "AITradeExecutor: AI gate error for {} — proceeding without AI filter: {}",
                    signal.get("symbol"),
                    exc,
                )

        return await super().execute_trade(signal)

    @property
    def ai_brain(self) -> Optional[Any]:
        """Return the configured AI brain, or ``None``."""
        return self._ai_brain

    @ai_brain.setter
    def ai_brain(self, value: Optional[Any]) -> None:
        """Replace the AI brain at runtime (e.g. after a model upgrade)."""
        self._ai_brain = value
        logger.info(
            "AITradeExecutor: AI brain updated to {}",
            type(value).__name__ if value is not None else "None",
        )

    @property
    def min_ai_confidence(self) -> float:
        """Minimum confidence threshold for trade execution."""
        return self._min_ai_confidence

    @min_ai_confidence.setter
    def min_ai_confidence(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"min_ai_confidence must be in [0, 1], got {value}")
        self._min_ai_confidence = value
        logger.info("AITradeExecutor: min_ai_confidence updated to {:.3f}", value)
