"""Intelligent strategy selector — scores strategies against current market microstructure."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
from loguru import logger

from strategy.base_strategy import BaseStrategy

# ---------------------------------------------------------------------------
# Regime-to-strategy affinity map (20% weight in final score)
# ---------------------------------------------------------------------------

_REGIME_AFFINITY: Dict[str, Dict[str, float]] = {
    "trending_up": {
        "trend_following": 1.0,
        "momentum": 1.0,
        "technical_breakout": 0.9,
        "ai_adaptive": 0.85,
        "smart_money_flow": 0.85,
        "whale_follower": 0.8,
        "ema_ribbon": 1.0,
        "supertrend": 1.0,
        "adx_trend": 1.0,
        "donchian_breakout": 0.9,
        "parabolic_sar": 0.9,
        "macd_crossover": 0.85,
        "market_structure_break": 0.8,
        "elliott_wave": 0.75,
        "on_chain_momentum": 0.8,
    },
    "trending_down": {
        "trend_following": 1.0,
        "mean_reversion": 0.7,
        "scalping": 0.75,
        "ai_adaptive": 0.85,
        "smart_money_flow": 0.85,
        "whale_follower": 0.8,
        "ema_ribbon": 1.0,
        "supertrend": 1.0,
        "adx_trend": 1.0,
        "donchian_breakout": 0.9,
        "parabolic_sar": 0.9,
        "macd_crossover": 0.85,
        "market_structure_break": 0.8,
        "momentum_divergence": 0.8,
    },
    "ranging": {
        "mean_reversion": 1.0,
        "grid_trading": 1.0,
        "market_making": 1.0,
        "scalping": 0.9,
        "funding_rate_arb": 0.9,
        "sentiment_reversal": 0.85,
        "vwap_deviation": 1.0,
        "stochastic_rsi": 0.9,
        "williams_r": 0.9,
        "bollinger_squeeze": 0.85,
        "range_breakout": 1.0,
        "pivot_point": 0.9,
        "volume_profile": 0.9,
        "accumulation_distribution": 0.85,
    },
    "high_volatility": {
        "scalping": 1.0,
        "liquidation_hunter": 1.0,
        "funding_rate_arb": 0.9,
        "ai_adaptive": 0.9,
        "volatility_breakout": 1.0,
        "bollinger_squeeze": 0.85,
        "keltner_channel": 0.85,
        "order_flow_imbalance": 0.9,
        "fair_value_gap": 0.8,
        "order_block": 0.8,
    },
    "extreme": {
        "scalping": 1.0,
        "liquidation_hunter": 1.0,
        "funding_rate_arb": 0.9,
        "ai_adaptive": 0.9,
        "volatility_breakout": 0.9,
    },
    "crash": {
        "dca": 1.0,
        "sentiment_reversal": 0.9,
        "mean_reversion": 0.7,
        "accumulation_distribution": 0.8,
    },
    "low_volatility": {
        "grid_trading": 1.0,
        "market_making": 1.0,
        "funding_rate_arb": 0.9,
        "mean_reversion": 0.85,
        "vwap_deviation": 0.8,
        "bollinger_squeeze": 0.9,
        "range_breakout": 0.85,
        "time_based": 0.75,
    },
}

# ---------------------------------------------------------------------------
# Market-condition scoring thresholds
# ---------------------------------------------------------------------------

# ADX thresholds for trend strength
_ADX_STRONG = 30.0
_ADX_MODERATE = 20.0

# ATR/price ratio for volatility classification
_VOLATILITY_HIGH = 0.025
_VOLATILITY_LOW = 0.008

# RSI extremes
_RSI_OB = 65.0
_RSI_OS = 35.0

# Bollinger Band Width expansion threshold (normalised)
_BBW_EXPAND = 0.05

# Strategies that prefer trending conditions
_TREND_STRATEGIES = frozenset(
    {
        "trend_following",
        "momentum",
        "ema_ribbon",
        "supertrend",
        "adx_trend",
        "donchian_breakout",
        "parabolic_sar",
        "macd_crossover",
        "technical_breakout",
        "market_structure_break",
        "elliott_wave",
        "ichimoku_cloud",
        "fibonacci_retracement",
        "on_chain_momentum",
    }
)

# Strategies that prefer ranging/low-volatility conditions
_RANGE_STRATEGIES = frozenset(
    {
        "mean_reversion",
        "grid_trading",
        "market_making",
        "vwap_deviation",
        "stochastic_rsi",
        "williams_r",
        "range_breakout",
        "pivot_point",
        "volume_profile",
        "accumulation_distribution",
        "bollinger_squeeze",
    }
)

# Strategies that prefer high-volatility / breakout conditions
_VOLATILITY_STRATEGIES = frozenset(
    {
        "scalping",
        "liquidation_hunter",
        "volatility_breakout",
        "keltner_channel",
        "order_flow_imbalance",
        "fair_value_gap",
        "order_block",
    }
)

# ---------------------------------------------------------------------------
# Session-to-strategy affinity (bonus applied to session score component)
# Sessions in UTC hours: Asian 0–8, London 8–16, New York 13–21
# ---------------------------------------------------------------------------

_SESSION_AFFINITY: Dict[str, Dict[str, float]] = {
    "asian": {
        # Asian session: low volume, range-bound — prefer ranging strategies
        "mean_reversion": 1.0,
        "grid_trading": 1.0,
        "market_making": 1.0,
        "vwap_deviation": 0.9,
        "stochastic_rsi": 0.9,
        "williams_r": 0.9,
        "funding_rate_arb": 0.9,
        "range_breakout": 0.85,
        "pivot_point": 0.85,
        "dca": 0.8,
    },
    "london": {
        # London open: high momentum, breakouts common
        "technical_breakout": 1.0,
        "donchian_breakout": 1.0,
        "volatility_breakout": 1.0,
        "ema_ribbon": 0.9,
        "supertrend": 0.9,
        "trend_following": 0.9,
        "adx_trend": 0.9,
        "macd_crossover": 0.85,
        "market_structure_break": 0.85,
    },
    "new_york": {
        # New York session: high liquidity, momentum trading
        "momentum": 1.0,
        "smart_money_flow": 1.0,
        "whale_follower": 0.9,
        "on_chain_momentum": 0.9,
        "liquidation_hunter": 0.9,
        "scalping": 0.85,
        "order_flow_imbalance": 0.85,
        "ai_adaptive": 0.8,
        "news_momentum": 0.8,
    },
    "overlap": {
        # London/NY overlap: highest volume — all strategies active
        "momentum": 1.0,
        "technical_breakout": 1.0,
        "volatility_breakout": 1.0,
        "liquidation_hunter": 1.0,
        "smart_money_flow": 0.95,
        "scalping": 0.9,
        "trend_following": 0.9,
        "ai_adaptive": 0.9,
    },
}


class StrategySelector:
    """Score and select the best strategies for current market conditions.

    Selection criteria (weighted)
    --------------------------------
    * ``market_condition_fit`` (40%): How well the strategy suits the current
      microstructure (ADX, BBW, RSI, volatility).
    * ``historical_performance`` (30%): Rolling win-rate and Sharpe ratio
      from past trades.
    * ``regime_appropriateness`` (20%): Affinity from the
      :data:`_REGIME_AFFINITY` mapping.
    * ``diversity_bonus`` (10%): Prefer strategies that diversify the set
      across trend/range/volatility categories.

    The selector is deterministic — given identical inputs it always
    returns the same ordered list.
    """

    def __init__(self, top_n: int = 7) -> None:
        self._top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_strategies(
        self,
        strategies: Dict[str, BaseStrategy],
        market_data: Dict[str, pd.DataFrame],
        rolling_metrics: Dict[str, Dict[str, float]],
        regime: str = "unknown",
        symbol: str = "",
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Score all enabled strategies and return the top N names.

        Args:
            strategies: Mapping of strategy name → :class:`BaseStrategy`.
            market_data: Timeframe → OHLCV DataFrame (at least ``"15m"``).
            rolling_metrics: strategy name → metrics dict (``win_rate``,
                ``sharpe``, ``total_trades`` …).
            regime: Current market regime label.
            symbol: Trading pair symbol (for logging).

        Returns:
            A 2-tuple of:
            * List of selected strategy names (best first).
            * Dict of strategy name → detailed scoring breakdown (for logging).
        """
        micro = self._analyse_microstructure(market_data)
        scores: Dict[str, Dict[str, Any]] = {}

        for name, strategy in strategies.items():
            if not strategy.enabled:
                continue
            if strategy.symbols and symbol and symbol not in strategy.symbols:
                continue

            score_detail = self._score_strategy(name, micro, rolling_metrics.get(name, {}), regime)
            scores[name] = score_detail

        if not scores:
            return [], {}

        # Apply diversity bonus after initial scoring
        selected_names = self._apply_diversity_bonus_and_rank(scores)
        top = selected_names[: self._top_n]

        self._log_selection(top, scores, regime, symbol, micro)
        return top, scores

    def get_selection_reasoning(
        self, scores: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Return a human-readable explanation for each strategy's score.

        Args:
            scores: The second element returned by :meth:`select_strategies`.

        Returns:
            Mapping of strategy name → explanation string.
        """
        reasoning: Dict[str, str] = {}
        for name, detail in scores.items():
            total = detail.get("total_score", 0.0)
            mc = detail.get("market_condition_fit", 0.0)
            hp = detail.get("historical_performance", 0.0)
            ra = detail.get("regime_appropriateness", 0.0)
            db = detail.get("diversity_bonus", 0.0)
            selected = detail.get("selected", False)
            status = "SELECTED" if selected else "rejected"
            reasoning[name] = (
                f"[{status}] total={total:.3f} "
                f"(market_fit={mc:.3f} hist_perf={hp:.3f} "
                f"regime={ra:.3f} diversity={db:.3f})"
            )
        return reasoning

    # ------------------------------------------------------------------
    # Microstructure analysis
    # ------------------------------------------------------------------

    def _analyse_microstructure(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute market microstructure indicators from available data.

        Returns a dict with keys:
        ``adx``, ``rsi``, ``atr_pct``, ``bbw``, ``trend_strength``,
        ``volatility_level``, ``momentum_direction``.
        """
        df15 = market_data.get("15m")
        df1h = market_data.get("1h")
        if df15 is not None and not df15.empty:
            df = df15
        elif df1h is not None and not df1h.empty:
            df = df1h
        else:
            df = pd.DataFrame()
        defaults: Dict[str, float] = {
            "adx": 20.0,
            "rsi": 50.0,
            "atr_pct": 0.015,
            "bbw": 0.03,
            "trend_strength": 0.5,
            "volatility_level": 0.5,
            "momentum_direction": 0.0,
        }
        if df.empty or len(df) < 30:
            return defaults

        try:
            closes = df["close"]
            highs = df["high"]
            lows = df["low"]

            # ADX
            adx_df = ta.adx(highs, lows, closes, length=14)
            adx_val = 20.0
            if adx_df is not None and not adx_df.empty:
                col = [c for c in adx_df.columns if c.startswith("ADX_")]
                if col:
                    v = adx_df[col[0]].iloc[-1]
                    adx_val = float(v) if not pd.isna(v) else 20.0

            # RSI
            rsi_s = ta.rsi(closes, length=14)
            rsi_val = 50.0
            if rsi_s is not None:
                v = rsi_s.iloc[-1]
                rsi_val = float(v) if not pd.isna(v) else 50.0

            # ATR % of price
            atr_s = ta.atr(highs, lows, closes, length=14)
            atr_val = 0.015
            if atr_s is not None:
                v = atr_s.iloc[-1]
                last_close = float(closes.iloc[-1])
                if not pd.isna(v) and last_close > 0:
                    atr_val = float(v) / last_close

            # Bollinger Band Width (normalised by middle band)
            bbands = ta.bbands(closes, length=20, std=2.0)
            bbw_val = 0.03
            if bbands is not None:
                w_col = [c for c in bbands.columns if "BBW" in c]
                if w_col:
                    v = bbands[w_col[0]].iloc[-1]
                    bbw_val = float(v) if not pd.isna(v) else 0.03

            # Trend strength: normalise ADX 0→50 to 0→1
            trend_strength = min(1.0, adx_val / 50.0)

            # Volatility level: normalise ATR%
            if atr_val >= _VOLATILITY_HIGH:
                volatility_level = 1.0
            elif atr_val <= _VOLATILITY_LOW:
                volatility_level = 0.0
            else:
                volatility_level = (atr_val - _VOLATILITY_LOW) / (
                    _VOLATILITY_HIGH - _VOLATILITY_LOW
                )

            # Momentum direction: sign and magnitude from MACD histogram
            macd_df = ta.macd(closes, fast=12, slow=26, signal=9)
            momentum_direction = 0.0
            if macd_df is not None:
                h_col = [c for c in macd_df.columns if "MACDh_" in c]
                if h_col:
                    v = macd_df[h_col[0]].iloc[-1]
                    if not pd.isna(v):
                        momentum_direction = float(v)

            return {
                "adx": adx_val,
                "rsi": rsi_val,
                "atr_pct": atr_val,
                "bbw": bbw_val,
                "trend_strength": trend_strength,
                "volatility_level": volatility_level,
                "momentum_direction": momentum_direction,
            }
        except Exception as exc:
            logger.warning(f"[StrategySelector] Microstructure analysis failed: {exc}")
            return defaults

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_strategy(
        self,
        name: str,
        micro: Dict[str, float],
        metrics: Dict[str, float],
        regime: str,
    ) -> Dict[str, Any]:
        """Compute the four score components for one strategy."""
        mc_fit = self._market_condition_fit(name, micro)
        hist_perf = self._historical_performance_score(metrics)
        regime_score = self._regime_score(name, regime)
        session_score = self._session_score(name)
        # Diversity bonus is applied later in aggregate
        return {
            "market_condition_fit": round(mc_fit, 4),
            "historical_performance": round(hist_perf, 4),
            "regime_appropriateness": round(regime_score, 4),
            "session_score": round(session_score, 4),
            "diversity_bonus": 0.0,  # filled in by _apply_diversity_bonus_and_rank
            "total_score": 0.0,  # filled in by _apply_diversity_bonus_and_rank
            "selected": False,
        }

    def _market_condition_fit(self, name: str, micro: Dict[str, float]) -> float:
        """Score how well the strategy fits the current microstructure (0–1)."""
        adx = micro["adx"]
        vol = micro["volatility_level"]
        rsi = micro["rsi"]

        if name in _TREND_STRATEGIES:
            # Prefer high ADX (strong trend)
            base = min(1.0, adx / _ADX_STRONG)
            # Penalise if RSI is in extreme territory (over-extended)
            if rsi > 80 or rsi < 20:
                base *= 0.7
            return base

        if name in _RANGE_STRATEGIES:
            # Prefer low ADX (weak trend / ranging)
            base = 1.0 - min(1.0, adx / _ADX_STRONG)
            # Bonus if RSI is in mid range (not trending hard)
            if _RSI_OS < rsi < _RSI_OB:
                base = min(1.0, base + 0.15)
            return base

        if name in _VOLATILITY_STRATEGIES:
            # Prefer high volatility
            return vol

        # Default: neutral fit
        return 0.5

    def _historical_performance_score(self, metrics: Dict[str, float]) -> float:
        """Convert rolling metrics to a [0, 1] score (0.5 when no data)."""
        total = metrics.get("total_trades", 0)
        if total == 0:
            return 0.5  # unknown — neutral

        win_rate = metrics.get("win_rate", 0.5)  # 0–1
        sharpe = metrics.get("sharpe", 0.0)
        profit_factor = metrics.get("profit_factor", 1.0)

        # Normalise each component to [0, 1]
        wr_score = win_rate  # already 0–1
        sharpe_score = max(0.0, min(1.0, (sharpe + 1.0) / 4.0))  # [-1,3] → [0,1]
        pf_score = max(0.0, min(1.0, (profit_factor - 0.5) / 2.0))  # [0.5,2.5] → [0,1]

        # Weight them: win_rate 50%, Sharpe 30%, PF 20%
        score = 0.5 * wr_score + 0.3 * sharpe_score + 0.2 * pf_score

        # Dampen confidence when trade count is low (< 10)
        confidence = min(1.0, total / 10.0)
        return score * confidence + 0.5 * (1.0 - confidence)

    def _regime_score(self, name: str, regime: str) -> float:
        """Look up affinity for the current regime (0–1, default 0.5)."""
        regime_map = _REGIME_AFFINITY.get(regime, {})
        return regime_map.get(name, 0.5)

    @staticmethod
    def _current_session() -> str:
        """Return the active trading session name based on current UTC hour.

        Sessions (UTC):
        * ``"asian"``   — 00:00–07:59
        * ``"london"``  — 08:00–12:59
        * ``"overlap"`` — 13:00–15:59  (London / New York overlap)
        * ``"new_york"``— 16:00–20:59
        * ``"off"``     — 21:00–23:59  (low-liquidity period)
        """
        hour = datetime.now(tz=timezone.utc).hour
        if 0 <= hour < 8:
            return "asian"
        if 8 <= hour < 13:
            return "london"
        if 13 <= hour < 16:
            return "overlap"
        if 16 <= hour < 21:
            return "new_york"
        return "off"

    def _session_score(self, name: str) -> float:
        """Return a session-affinity score for the current UTC session (0–1).

        Strategies not listed in the session affinity map receive a neutral
        score of 0.5, allowing them to compete on other criteria.

        Args:
            name: Strategy name.

        Returns:
            Session affinity score in [0, 1].
        """
        session = self._current_session()
        if session == "off":
            # Outside major sessions — slightly reduce all scores to encourage caution
            return 0.4
        session_map = _SESSION_AFFINITY.get(session, {})
        return session_map.get(name, 0.5)

    def _apply_diversity_bonus_and_rank(
        self, scores: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Add diversity bonuses, compute final scores, and return ranked names."""
        # Initial ranking by weighted sum (no diversity yet)
        # Weights: market_fit 35%, hist_perf 25%, regime 20%, session 10%
        # (diversity bonus 10% applied below)
        def _raw(d: Dict[str, Any]) -> float:
            return (
                0.35 * d["market_condition_fit"]
                + 0.25 * d["historical_performance"]
                + 0.20 * d["regime_appropriateness"]
                + 0.10 * d.get("session_score", 0.5)
            )

        # Sort by raw score desc
        ranked = sorted(scores.keys(), key=lambda n: _raw(scores[n]), reverse=True)

        categories_selected: List[str] = []  # "trend" / "range" / "volatility"
        for name in ranked:
            cat = self._category(name)
            bonus = 0.0
            if cat not in categories_selected:
                bonus = 0.10  # diversity bonus
                categories_selected.append(cat)
            scores[name]["diversity_bonus"] = round(bonus, 4)
            scores[name]["total_score"] = round(_raw(scores[name]) + bonus, 4)

        # Re-rank with final scores
        ranked_final = sorted(
            scores.keys(), key=lambda n: scores[n]["total_score"], reverse=True
        )
        # Mark selected
        for i, name in enumerate(ranked_final):
            scores[name]["selected"] = i < self._top_n

        return ranked_final

    @staticmethod
    def _category(name: str) -> str:
        if name in _TREND_STRATEGIES:
            return "trend"
        if name in _RANGE_STRATEGIES:
            return "range"
        if name in _VOLATILITY_STRATEGIES:
            return "volatility"
        return "other"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_selection(
        self,
        top: List[str],
        scores: Dict[str, Dict[str, Any]],
        regime: str,
        symbol: str,
        micro: Dict[str, float],
    ) -> None:
        session = self._current_session()
        logger.info(
            "[StrategySelector] symbol={} regime={} session={} "
            "adx={:.1f} atr_pct={:.3f} rsi={:.1f}",
            symbol or "ALL",
            regime,
            session,
            micro["adx"],
            micro["atr_pct"],
            micro["rsi"],
        )
        for name in top:
            d = scores[name]
            logger.info(
                "[StrategySelector]  ✓ {} | total={:.3f} "
                "(fit={:.3f} hist={:.3f} regime={:.3f} session={:.3f} diversity={:.3f})",
                name,
                d["total_score"],
                d["market_condition_fit"],
                d["historical_performance"],
                d["regime_appropriateness"],
                d.get("session_score", 0.5),
                d["diversity_bonus"],
            )
        rejected = [n for n in scores if n not in top]
        if rejected:
            logger.debug(
                "[StrategySelector]  ✗ rejected: {}",
                ", ".join(rejected[:10]) + ("…" if len(rejected) > 10 else ""),
            )
