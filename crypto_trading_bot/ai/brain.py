"""Central AI decision engine combining sentiment, regime, and LLM reasoning.

.. note:: **Slow-loop component — not for the hot path.**

    :class:`AIBrain` makes LLM API calls and may take 200 ms – 2 000 ms per
    invocation.  It is designed to run in the
    :class:`~ai.regime_service.RegimeService` background loop every
    ``REGIME_INTERVAL_SECONDS`` (default 5 minutes).

    Strategy signal generators (e.g. :class:`~strategy.strategies.ai_adaptive.AIAdaptiveStrategy`)
    do **not** call this class at signal-generation time.  Instead they read the
    pre-computed :class:`~ai.regime_service.RegimeState` from
    ``/dev/shm/regime_state.json`` (O(1) file read with a 30-second in-process
    cache).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class TradeDecision(BaseModel):
    """Output of the AI Brain for a single symbol analysis."""

    symbol: str
    direction: TradeDirection
    confidence: float  # 0–1
    reasoning: str
    suggested_leverage: int = 1
    suggested_stop_loss_pct: float = 2.0
    suggested_take_profit_pct: float = 3.0
    should_enter: bool = False
    risk_level: str = "MEDIUM"
    key_factors: List[str] = []
    timestamp: datetime = None  # type: ignore[assignment]

    def __init__(self, **data) -> None:  # type: ignore[override]
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(tz=timezone.utc)
        super().__init__(**data)


class NewsImpact(BaseModel):
    """Distilled impact assessment for a single news item."""

    title: str
    direction: str  # BULLISH / BEARISH / NEUTRAL
    impact_level: str  # CRITICAL / HIGH / MEDIUM / LOW
    affected_symbols: List[str] = []
    confidence: float = 0.5
    action_required: bool = False


class AIBrain:
    """Central AI decision engine.

    Combines sentiment analysis, technical indicator signals, market regime
    detection, and (optionally) LLM reasoning to produce :class:`TradeDecision`
    objects.  When an LLM is unavailable the engine falls back to a transparent
    rule-based approach so the bot can always produce a decision.

    Composition
    -----------
    - ``llm_client``       — :class:`~ai.llm_client.LLMClient` (optional)
    - ``sentiment_analyzer`` — :class:`~ai.sentiment.analyzer.SentimentAnalyzer` (optional)
    - ``news_classifier``  — :class:`~ai.news_classifier.classifier.NewsClassifier` (optional)
    - ``regime_detector``  — :class:`~ai.market_analyzer.regime_detector.MarketRegimeDetector` (optional)
    - ``memory``           — :class:`~ai.memory.AIMemory` (optional)
    """

    MIN_CONFIDENCE_TO_TRADE: float = 0.65
    MAX_LEVERAGE: int = 10

    def __init__(
        self,
        llm_client=None,
        sentiment_analyzer=None,
        news_classifier=None,
        regime_detector=None,
        memory=None,
    ) -> None:
        self._llm = llm_client
        self._sentiment = sentiment_analyzer
        self._news_classifier = news_classifier
        self._regime_detector = regime_detector
        self._memory = memory
        self._decisions_made: int = 0
        self._correct_decisions: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> TradeDecision:
        """Analyze *symbol* and return a :class:`TradeDecision`.

        Args:
            symbol:      Trading pair, e.g. ``"BTC/USDT"``.
            market_data: Must contain ``"price"`` (float).  Optionally
                         ``"indicators"`` (dict) and ``"ohlcv"`` (DataFrame).
            context:     Optional dict with ``"news"`` (list), ``"balance"``
                         (float), ``"open_positions"`` (list).

        Returns:
            :class:`TradeDecision` — never raises, falls back to HOLD on error.
        """
        if context is None:
            context = {}
        try:
            logger.debug(f"AI Brain analyzing {symbol}")

            current_price: float = market_data.get("price", 0.0)
            indicators: Dict[str, Any] = market_data.get("indicators", {})
            news_items: List[Dict] = context.get("news", [])
            balance: float = context.get("balance", 10_000.0)
            open_positions: List[Dict] = context.get("open_positions", [])

            # Sentiment ---------------------------------------------------
            sentiment_score = await self._compute_sentiment(news_items)

            # Market regime -----------------------------------------------
            regime = await self._compute_regime(market_data)

            # OHLCV summary -----------------------------------------------
            ohlcv_summary = self._build_ohlcv_summary(market_data, current_price)

            # News summary ------------------------------------------------
            news_summary = self._summarize_news(news_items)

            # LLM decision (best path) ------------------------------------
            if self._llm:
                decision = await self._llm_decision(
                    symbol,
                    current_price,
                    ohlcv_summary,
                    indicators,
                    sentiment_score,
                    news_summary,
                    open_positions,
                    balance,
                    regime,
                )
                if decision is not None:
                    self._decisions_made += 1
                    return decision

            # Rule-based fallback -----------------------------------------
            decision = self._rule_based_decision(
                symbol, current_price, indicators, sentiment_score, regime
            )
            self._decisions_made += 1
            return decision

        except Exception as exc:
            logger.error(f"AI Brain error for {symbol}: {exc}", exc_info=True)
            return TradeDecision(
                symbol=symbol,
                direction=TradeDirection.HOLD,
                confidence=0.0,
                reasoning=f"Analysis error: {exc}",
                should_enter=False,
            )

    async def analyze_news(
        self,
        title: str,
        content: str = "",
        source: str = "",
    ) -> NewsImpact:
        """Classify a news item and return a :class:`NewsImpact`.

        Falls back to a neutral LOW-impact result if the classifier is
        unavailable or raises.
        """
        if self._news_classifier:
            try:
                classification = await self._news_classifier.classify(title, content, source)
                return NewsImpact(
                    title=title,
                    direction=classification.direction.value,
                    impact_level=classification.impact_level.value,
                    affected_symbols=[f"{a}/USDT" for a in classification.affected_assets],
                    confidence=classification.confidence,
                    action_required=classification.impact_level.value in ("CRITICAL", "HIGH"),
                )
            except Exception as exc:
                logger.warning(f"News analysis error: {exc}")

        return NewsImpact(title=title, direction="NEUTRAL", impact_level="LOW")

    async def get_confidence_modifier(
        self,
        market_overview: Dict[str, Any],
        sentiment_score: float = 0.0,
        news_summary: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return an AI-based market assessment for use as a confidence modifier.

        This is called **once per trading cycle** (not per symbol) to produce a
        high-level directional view.  Individual strategy signals can then be
        boosted or reduced by 15% depending on whether the AI direction agrees.

        The method returns ``None`` silently when no LLM client is configured or
        when the LLM call fails, so the trading cycle degrades gracefully.

        Args:
            market_overview: Dict with at least ``"symbols"`` (list of str).
            sentiment_score: Aggregate sentiment in [-1, 1].
            news_summary:    Recent news bullet points.

        Returns:
            Dict with ``"direction"`` (``"bullish"``/``"bearish"``/``"neutral"``),
            ``"confidence"``, ``"key_levels"``, and ``"risk_assessment"`` — or
            ``None`` if AI is unavailable or fails.
        """
        if self._llm is None:
            return None
        try:
            from ai.prompt_engine import PromptEngine

            prompt = PromptEngine().build_market_confidence_prompt(
                market_overview=market_overview,
                sentiment_score=sentiment_score,
                news_summary=news_summary,
            )
            result = await self._llm.analyze_market(prompt)
            if result and "error" not in result:
                return result
        except Exception as exc:
            logger.warning(f"AI confidence modifier failed: {exc}")
        return None

    async def should_exit_trade(
        self,
        position: Dict[str, Any],
        current_data: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Determine whether *position* should be closed now.

        Checks hard stop-loss and take-profit levels.  Returns a
        ``(should_exit, reason)`` tuple.
        """
        entry_price: float = position.get("entry_price", 0.0)
        current_price: float = current_data.get("price", 0.0)
        side: str = position.get("side", "long")

        if not current_price or not entry_price:
            return False, "Insufficient price data"

        stop_loss: Optional[float] = position.get("stop_loss")
        if stop_loss is not None:
            if (side == "long" and current_price <= stop_loss) or (
                side == "short" and current_price >= stop_loss
            ):
                return True, "Stop loss triggered"

        take_profit: Optional[float] = position.get("take_profit")
        if take_profit is not None:
            if (side == "long" and current_price >= take_profit) or (
                side == "short" and current_price <= take_profit
            ):
                return True, "Take profit reached"

        return False, ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compute_sentiment(self, news_items: List[Dict]) -> float:
        """Average sentiment score across *news_items* using the analyzer."""
        if not self._sentiment or not news_items:
            return 0.0
        texts = [item.get("content", "") for item in news_items[:10] if item.get("content")]
        if not texts:
            return 0.0
        results = await self._sentiment.analyze_batch(texts)
        return sum(r.score for r in results) / len(results) if results else 0.0

    async def _compute_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect market regime; returns 'UNKNOWN' when data or detector is absent."""
        if self._regime_detector and "ohlcv" in market_data:
            try:
                regime_enum = await self._regime_detector.detect_regime(market_data["ohlcv"])
                return regime_enum.value
            except Exception as exc:
                logger.warning(f"Regime detection failed: {exc}")
        return "UNKNOWN"

    @staticmethod
    def _build_ohlcv_summary(market_data: Dict[str, Any], current_price: float) -> Dict[str, float]:
        """Build the 24-bar OHLCV summary dict expected by the prompt engine."""
        ohlcv = market_data.get("ohlcv")
        if ohlcv is None or len(ohlcv) == 0:
            return {}

        lookback = min(24, len(ohlcv))
        open_price = float(ohlcv["open"].iloc[-lookback])
        change_pct = (
            (
                (current_price - float(ohlcv["close"].iloc[-lookback]))
                / float(ohlcv["close"].iloc[-lookback])
                * 100
            )
            if lookback > 1
            else 0.0
        )

        return {
            "open_24h": open_price,
            "high_24h": float(ohlcv["high"].tail(lookback).max()),
            "low_24h": float(ohlcv["low"].tail(lookback).min()),
            "volume_24h": float(ohlcv["volume"].tail(lookback).sum()),
            "change_pct": change_pct,
        }

    async def _llm_decision(
        self,
        symbol: str,
        current_price: float,
        ohlcv_summary: Dict[str, float],
        indicators: Dict[str, Any],
        sentiment_score: float,
        news_summary: str,
        open_positions: List[Dict],
        balance: float,
        regime: str,
    ) -> Optional[TradeDecision]:
        """Query the LLM and parse the result; returns ``None`` on failure."""
        try:
            from ai.prompt_engine import PromptEngine

            prompt = PromptEngine().build_trade_analysis_prompt(
                symbol=symbol,
                current_price=current_price,
                ohlcv_summary=ohlcv_summary,
                indicators=indicators,
                sentiment_score=sentiment_score,
                news_summary=news_summary,
                open_positions=open_positions,
                balance_usd=balance,
                market_regime=regime,
            )
            llm_result = await self._llm.query_json(prompt)
            if llm_result and "error" not in llm_result:
                return self._parse_llm_decision(symbol, llm_result)
        except Exception as exc:
            logger.warning(f"LLM decision failed for {symbol}: {exc}, falling back to rule-based")
        return None

    def _parse_llm_decision(self, symbol: str, result: Dict[str, Any]) -> TradeDecision:
        """Convert a validated LLM JSON dict into a :class:`TradeDecision`."""
        try:
            direction_str = result.get("direction", "hold").lower()
            direction = (
                TradeDirection(direction_str)
                if direction_str in ("long", "short", "hold")
                else TradeDirection.HOLD
            )
            confidence = float(result.get("confidence", 0.5))
            should_enter = (
                bool(result.get("should_trade", False))
                and confidence >= self.MIN_CONFIDENCE_TO_TRADE
            )
            reasoning = (
                f"{result.get('entry_reasoning', '')} | Risks: {result.get('risk_concerns', '')}"
            )
            leverage = min(int(result.get("suggested_leverage", 1)), self.MAX_LEVERAGE)

            return TradeDecision(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                reasoning=reasoning.strip(" |"),
                suggested_leverage=leverage,
                suggested_stop_loss_pct=float(result.get("suggested_stop_loss_pct", 2.0)),
                suggested_take_profit_pct=float(result.get("suggested_take_profit_pct", 3.0)),
                should_enter=should_enter,
                key_factors=result.get("key_factors", []),
            )
        except Exception as exc:
            logger.warning(f"Failed to parse LLM decision: {exc}")
            return TradeDecision(
                symbol=symbol,
                direction=TradeDirection.HOLD,
                confidence=0.0,
                reasoning="LLM response parse error",
                should_enter=False,
            )

    def _rule_based_decision(
        self,
        symbol: str,
        price: float,
        indicators: Dict[str, Any],
        sentiment: float,
        regime: str,
    ) -> TradeDecision:
        """Simple rule-based fallback decision when the LLM is unavailable."""
        score = 0.0
        reasons: List[str] = []

        # RSI signal
        rsi = float(indicators.get("rsi", 50))
        if rsi < 30:
            score += 1.5
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 1.5
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # Sentiment
        if sentiment > 0.3:
            score += 1.0
            reasons.append("Positive market sentiment")
        elif sentiment < -0.3:
            score -= 1.0
            reasons.append("Negative market sentiment")

        # Regime adjustment
        if regime in ("STRONG_UPTREND", "WEAK_UPTREND"):
            score += 0.5
        elif regime in ("STRONG_DOWNTREND", "WEAK_DOWNTREND"):
            score -= 0.5
        elif regime in ("HIGH_VOLATILITY", "CRASH"):
            score = 0.0
            reasons = ["High volatility — holding off without LLM confirmation"]

        confidence = min(abs(score) / 4.0, 1.0)
        should_enter = confidence >= self.MIN_CONFIDENCE_TO_TRADE

        if score > 1.5 and should_enter:
            direction = TradeDirection.LONG
        elif score < -1.5 and should_enter:
            direction = TradeDirection.SHORT
        else:
            direction = TradeDirection.HOLD
            should_enter = False

        return TradeDecision(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "No clear signal",
            should_enter=should_enter,
            suggested_leverage=2,
            suggested_stop_loss_pct=2.0,
            suggested_take_profit_pct=3.0,
        )

    @staticmethod
    def _summarize_news(news_items: List[Dict]) -> str:
        """Produce a brief bullet-list summary of up to 5 news items."""
        if not news_items:
            return "No recent news"
        bullets = [f"- {item.get('content', '')[:100]}" for item in news_items[:5]]
        return "\n".join(bullets)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def decision_count(self) -> int:
        """Total number of :meth:`analyze` calls that produced a decision."""
        return self._decisions_made


# ---------------------------------------------------------------------------
# Module-level helpers (importable without the full engine dependency chain)
# ---------------------------------------------------------------------------


def apply_ai_confidence_modifier(signal: Dict[str, Any], ai_signal: Dict[str, Any]) -> float:
    """Apply the AI directional view as a ±15% confidence modifier.

    This is a module-level function so that it can be imported and tested
    without pulling in the full :class:`~core.engine.TradingEngine` (and its
    heavy exchange dependency chain).

    The AI assessment is a **confidence modifier only** — it never alters
    position sizes, stop-losses, or take-profits, and does not override the
    risk manager's decisions.

    Rules
    -----
    * AI bullish + signal long  → +15% confidence
    * AI bearish + signal short → +15% confidence
    * AI bearish + signal long  → −15% confidence
    * AI bullish + signal short → −15% confidence
    * AI neutral or missing     → no change

    Args:
        signal:    Strategy signal dict with ``"direction"`` and ``"confidence"``.
        ai_signal: AI market analysis dict with ``"direction"``.

    Returns:
        Adjusted confidence clamped to [0.0, 1.0].
    """
    current_confidence = float(signal.get("confidence", 0.5))
    signal_direction = signal.get("direction", "neutral")
    ai_direction = str(ai_signal.get("direction", "neutral")).lower()

    agrees = (ai_direction == "bullish" and signal_direction == "long") or (
        ai_direction == "bearish" and signal_direction == "short"
    )
    disagrees = (ai_direction == "bearish" and signal_direction == "long") or (
        ai_direction == "bullish" and signal_direction == "short"
    )

    if agrees:
        modifier = 0.15
    elif disagrees:
        modifier = -0.15
    else:
        modifier = 0.0

    return round(min(1.0, max(0.0, current_confidence + modifier)), 3)
