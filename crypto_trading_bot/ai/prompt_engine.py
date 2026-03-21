"""Dynamic prompt construction for different AI trading-analysis tasks."""

from typing import Any, Dict, List


class PromptEngine:
    """Builds optimized prompts for different trading analysis tasks."""

    SYSTEM_PROMPTS: Dict[str, str] = {
        "TRADE_ANALYST": (
            "You are an expert cryptocurrency trader and quantitative analyst with 10+ years of"
            " experience.\nYou analyze market data, news, sentiment, and technical indicators to"
            " make precise trading decisions.\nAlways be objective, risk-aware, and base decisions"
            " on data. Never be overconfident.\nAlways respond with valid JSON."
        ),
        "NEWS_CLASSIFIER": (
            "You are a financial news analyst specializing in cryptocurrency markets.\n"
            "Classify news articles by impact, direction, and urgency. Be precise and objective.\n"
            "Always respond with valid JSON."
        ),
        "RISK_ANALYST": (
            "You are a risk management expert for cryptocurrency trading.\n"
            "Your primary goal is capital preservation. Be conservative and risk-averse.\n"
            "Always respond with valid JSON."
        ),
    }

    # ------------------------------------------------------------------
    # Trade analysis
    # ------------------------------------------------------------------

    def build_trade_analysis_prompt(
        self,
        symbol: str,
        current_price: float,
        ohlcv_summary: Dict[str, Any],
        indicators: Dict[str, Any],
        sentiment_score: float,
        news_summary: str,
        open_positions: List[Dict[str, Any]],
        balance_usd: float,
        market_regime: str = "UNKNOWN",
    ) -> str:
        """Build a comprehensive trade-analysis prompt for the LLM."""
        return f"""Analyze {symbol} for a trading opportunity.

CURRENT DATA:
- Price: ${current_price:,.4f}
- Market Regime: {market_regime}
- Sentiment Score: {sentiment_score:.2f} (-1=very bearish, +1=very bullish)
- Account Balance: ${balance_usd:,.2f}
- Open Positions: {len(open_positions)}

PRICE ACTION (24h):
- Open: ${ohlcv_summary.get('open_24h', 0):,.4f}
- High: ${ohlcv_summary.get('high_24h', 0):,.4f}
- Low: ${ohlcv_summary.get('low_24h', 0):,.4f}
- Volume: {ohlcv_summary.get('volume_24h', 0):,.0f}
- Change: {ohlcv_summary.get('change_pct', 0):+.2f}%

TECHNICAL INDICATORS:
- RSI (14): {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- EMA 20: {indicators.get('ema_20', 'N/A')}
- EMA 50: {indicators.get('ema_50', 'N/A')}
- Bollinger Band Position: {indicators.get('bb_position', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}
- Volume vs Average: {indicators.get('volume_ratio', 'N/A')}

RECENT NEWS:
{news_summary[:500] if news_summary else 'No significant news'}

Provide your analysis as JSON:
{{
  "should_trade": true/false,
  "direction": "long" or "short" or "hold",
  "confidence": 0.0-1.0,
  "suggested_leverage": 1-10,
  "entry_reasoning": "why to enter",
  "risk_concerns": "any concerns",
  "suggested_stop_loss_pct": 1.0-5.0,
  "suggested_take_profit_pct": 1.0-10.0,
  "time_horizon": "1h/4h/24h",
  "key_factors": ["factor1", "factor2"]
}}"""

    # ------------------------------------------------------------------
    # News classification
    # ------------------------------------------------------------------

    def build_news_classification_prompt(
        self,
        title: str,
        content: str,
        source: str,
    ) -> str:
        """Build a news-classification prompt."""
        return f"""Classify this crypto news article:

SOURCE: {source}
TITLE: {title}
CONTENT: {content[:1000]}

Respond with JSON:
{{
  "category": "REGULATORY|TECHNICAL|ADOPTION|MARKET|MACRO|SECURITY|PARTNERSHIP|DEVELOPMENT",
  "impact_level": "CRITICAL|HIGH|MEDIUM|LOW|NOISE",
  "direction": "BULLISH|BEARISH|NEUTRAL",
  "affected_assets": ["BTC", "ETH"],
  "time_horizon": "IMMEDIATE|SHORT|MEDIUM|LONG",
  "confidence": 0.0-1.0,
  "summary": "1 sentence summary",
  "is_fake_or_recycled": true/false
}}"""

    # ------------------------------------------------------------------
    # Sentiment synthesis
    # ------------------------------------------------------------------

    def build_sentiment_synthesis_prompt(
        self,
        sources_data: List[Dict[str, Any]],
    ) -> str:
        """Build a sentiment-synthesis prompt from multiple source data items."""
        sources_text = "\n".join(
            [
                f"- {d.get('source')}: {d.get('content', '')[:200]} (score: {d.get('score', 0):.2f})"
                for d in sources_data[:10]
            ]
        )
        return f"""Synthesize the following crypto sentiment signals:

{sources_text}

Respond with JSON:
{{
  "overall_sentiment": -1.0 to 1.0,
  "label": "very_bearish|bearish|neutral|bullish|very_bullish",
  "confidence": 0.0-1.0,
  "key_themes": ["theme1", "theme2"],
  "conflicting_signals": true/false,
  "reasoning": "brief explanation"
}}"""

    # ------------------------------------------------------------------
    # Cycle-level market confidence
    # ------------------------------------------------------------------

    def build_market_confidence_prompt(
        self,
        market_overview: Dict[str, Any],
        sentiment_score: float = 0.0,
        news_summary: str = "",
    ) -> str:
        """Build a cycle-level market direction and confidence prompt.

        This prompt is sent **once per trading cycle** (not per symbol) to
        produce a high-level market assessment that is used to apply a ±15%
        confidence modifier to individual strategy signals.

        Args:
            market_overview: Dict with at least ``"symbols"`` (list of str).
            sentiment_score: Aggregate sentiment in [-1, 1].
            news_summary:    Recent news bullet points (truncated to 500 chars).

        Returns:
            A prompt string requesting JSON with ``direction``, ``confidence``,
            ``key_levels``, and ``risk_assessment`` fields.
        """
        symbols_str = ", ".join(market_overview.get("symbols", []))
        return f"""Analyze overall cryptocurrency market conditions.

ACTIVE SYMBOLS: {symbols_str or 'N/A'}
AGGREGATE SENTIMENT: {sentiment_score:.2f} (-1=very bearish, +1=very bullish)

RECENT NEWS:
{news_summary[:500] if news_summary else 'No significant news'}

Provide a market-wide assessment as JSON:
{{
  "direction": "bullish" or "bearish" or "neutral",
  "confidence": 0.0-1.0,
  "key_levels": {{"support": 0.0, "resistance": 0.0}},
  "risk_assessment": "brief risk summary string",
  "reasoning": "brief explanation"
}}"""

    # ------------------------------------------------------------------
    # Daily plan
    # ------------------------------------------------------------------

    def build_daily_plan_prompt(
        self,
        portfolio_state: Dict[str, Any],
        market_overview: str,
    ) -> str:
        """Build a daily trading-plan prompt."""
        return f"""Create a trading plan for today based on:

PORTFOLIO STATE:
- Balance: ${portfolio_state.get('balance', 0):,.2f}
- Open Positions: {portfolio_state.get('open_positions', 0)}
- Daily P&L so far: ${portfolio_state.get('daily_pnl', 0):+,.2f}
- Win Rate (last 30 trades): {portfolio_state.get('win_rate', 0):.1%}

MARKET OVERVIEW:
{market_overview[:500]}

Respond with JSON:
{{
  "market_sentiment": "bullish|bearish|neutral",
  "recommended_pairs": ["BTC/USDT", "ETH/USDT"],
  "max_trades_today": 1-10,
  "risk_level": "conservative|moderate|aggressive",
  "focus_strategies": ["strategy1", "strategy2"],
  "avoid_trades": "any trades to avoid today",
  "key_events_to_watch": ["event1"],
  "daily_target_pct": 0.5-2.0
}}"""
