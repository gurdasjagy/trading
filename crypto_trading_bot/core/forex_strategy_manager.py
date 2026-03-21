"""Forex Strategy Manager — loads, scores and selects gold trading strategies."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# ---------------------------------------------------------------------------
# ForexStrategyManager
# ---------------------------------------------------------------------------


class ForexStrategyManager:
    """Loads all strategies from strategy/strategies/forex/ and selects the best.

    Features:
    * Automatic discovery of all strategy classes in the forex strategies dir.
    * Session-aware strategy weighting (London favours breakout, Asian favours range).
    * Market-regime fitness scoring per strategy.
    * Rolling win-rate tracking per strategy.
    * Minimum confluence threshold (≥ 3 strategies agreeing before entry).
    * Automatic disabling of persistently underperforming strategies.
    """

    # Minimum number of strategies that must agree on direction for entry
    MIN_CONFLUENCE = 3

    # Strategies with win rate below this after MIN_TRADES_TO_EVALUATE are disabled
    MIN_WIN_RATE_THRESHOLD = 0.35
    MIN_TRADES_TO_EVALUATE = 20

    # Win-rate defaults before enough data is collected
    MIN_TRADES_FOR_WINRATE = 5   # minimum trades before using real win rate
    DEFAULT_WIN_RATE = 0.6       # assumed win rate when trade history is sparse

    # Module path for forex strategies
    FOREX_STRATEGY_MODULES = [
        "strategy.strategies.forex.gold_momentum_breakout",
        "strategy.strategies.forex.gold_fibonacci",
        "strategy.strategies.forex.gold_mean_reversion",
        "strategy.strategies.forex.gold_rsi_divergence",
        "strategy.strategies.forex.gold_bollinger_squeeze",
        "strategy.strategies.forex.gold_ichimoku",
        "strategy.strategies.forex.gold_safe_haven",
        "strategy.strategies.forex.gold_scalping",
        "strategy.strategies.forex.gold_session_momentum",
        "strategy.strategies.forex.gold_supply_demand",
        "strategy.strategies.forex.gold_vwap",
        "strategy.strategies.forex.gold_dxy_inverse",
        "strategy.strategies.forex.asian_range_breakout",
        "strategy.strategies.forex.london_breakout",
        "strategy.strategies.forex.nfp_news_strategy",
        "strategy.strategies.forex.forex_mean_reversion_session",
        "strategy.strategies.forex.carry_trade",
        "strategy.strategies.forex.central_bank_divergence",
        "strategy.strategies.forex.correlation_pairs",
        # New strategies (Upgrade 2)
        "strategy.strategies.forex.gold_supertrend",
        "strategy.strategies.forex.gold_ema_ribbon",
        "strategy.strategies.forex.gold_adx_trend",
        "strategy.strategies.forex.gold_donchian_channel",
        "strategy.strategies.forex.gold_parabolic_sar",
        "strategy.strategies.forex.gold_keltner_channel",
        "strategy.strategies.forex.gold_stochastic_oversold",
        "strategy.strategies.forex.gold_williams_r",
        "strategy.strategies.forex.gold_atr_breakout",
        "strategy.strategies.forex.gold_volatility_squeeze",
        "strategy.strategies.forex.gold_range_expansion",
        "strategy.strategies.forex.gold_london_fix",
        "strategy.strategies.forex.gold_ny_open_reversal",
        "strategy.strategies.forex.gold_asian_range",
        "strategy.strategies.forex.gold_order_block",
        "strategy.strategies.forex.gold_liquidity_sweep",
        "strategy.strategies.forex.gold_multi_timeframe_confluence",
    ]

    # Session → preferred strategy types (used for weighting)
    SESSION_WEIGHTS: Dict[str, Dict[str, float]] = {
        "London": {
            "breakout": 1.5,
            "trend": 1.3,
            "range": 0.7,
            "volatility": 1.2,
            "scalping": 1.0,
        },
        "New York": {
            "breakout": 1.2,
            "trend": 1.4,
            "range": 0.8,
            "volatility": 1.3,
            "scalping": 0.9,
        },
        "Tokyo": {
            "breakout": 0.7,
            "trend": 0.8,
            "range": 1.5,
            "volatility": 0.6,
            "scalping": 1.2,
        },
        "Sydney": {
            "breakout": 0.8,
            "trend": 0.9,
            "range": 1.3,
            "volatility": 0.7,
            "scalping": 1.1,
        },
    }

    # Strategy type classification by name keywords
    STRATEGY_TYPES: Dict[str, str] = {
        "breakout": "breakout",
        "momentum": "trend",
        "trend": "trend",
        "ema_ribbon": "trend",
        "adx": "trend",
        "donchian": "breakout",
        "parabolic": "trend",
        "supertrend": "trend",
        "mean_reversion": "range",
        "stochastic": "range",
        "williams": "range",
        "keltner": "range",
        "bollinger": "volatility",
        "atr_breakout": "volatility",
        "volatility_squeeze": "volatility",
        "range_expansion": "volatility",
        "scalping": "scalping",
        "vwap": "trend",
        "london_fix": "breakout",
        "london": "breakout",
        "asian": "range",
        "ny_open": "breakout",
        "fibonacci": "range",
        "ichimoku": "trend",
        "rsi_divergence": "range",
        "supply_demand": "range",
        "order_block": "range",
        "liquidity_sweep": "breakout",
        "session_momentum": "trend",
        "safe_haven": "trend",
        "dxy_inverse": "trend",
        "multi_timeframe": "trend",
    }

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self._symbols = symbols or ["XAU/USDT"]
        self._strategies: List[BaseStrategy] = []
        self._exchange: Any = None
        # Performance tracking: strategy_name → {trades, wins, losses}
        self._perf: Dict[str, Dict[str, int]] = {}

    def set_exchange(self, exchange: Any) -> None:
        """Attach an exchange client to all loaded strategies."""
        self._exchange = exchange
        for s in self._strategies:
            s._exchange = exchange

    async def load_strategies(self) -> int:
        """Dynamically load all forex strategy classes.

        Returns:
            Number of strategies successfully loaded.
        """
        loaded = 0
        for module_path in self.FOREX_STRATEGY_MODULES:
            try:
                module = importlib.import_module(module_path)
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(cls, BaseStrategy)
                        and cls is not BaseStrategy
                        and getattr(cls, "enabled", True)
                    ):
                        strategy = cls(symbols=self._symbols)
                        if self._exchange:
                            strategy._exchange = self._exchange
                        self._strategies.append(strategy)
                        self._perf[strategy.name] = {"trades": 0, "wins": 0, "losses": 0}
                        logger.debug("Loaded forex strategy: {}", strategy.name)
                        loaded += 1
            except ImportError:
                logger.debug("Forex strategy module not found (skipping): {}", module_path)
            except Exception as e:
                logger.warning("Failed to load forex strategy {}: {}", module_path, e)

        logger.info("ForexStrategyManager: loaded {} strategies", loaded)
        return loaded

    def _get_strategy_type(self, strategy_name: str) -> str:
        """Classify strategy by name."""
        name_lower = strategy_name.lower()
        for keyword, stype in self.STRATEGY_TYPES.items():
            if keyword in name_lower:
                return stype
        return "trend"  # default

    def _get_session_multiplier(self, strategy_name: str, session: str) -> float:
        """Return session-based weighting for a strategy."""
        if not session or session == "Closed":
            return 1.0
        # Pick first session name if multiple active (e.g. "London, New York")
        primary_session = session.split(",")[0].strip()
        weights = self.SESSION_WEIGHTS.get(primary_session, {})
        stype = self._get_strategy_type(strategy_name)
        return weights.get(stype, 1.0)

    def _get_win_rate(self, strategy_name: str) -> float:
        """Return rolling win rate for a strategy."""
        perf = self._perf.get(strategy_name, {})
        trades = perf.get("trades", 0)
        wins = perf.get("wins", 0)
        if trades < self.MIN_TRADES_FOR_WINRATE:
            return self.DEFAULT_WIN_RATE
        return wins / trades

    def record_trade_result(self, strategy_name: str, win: bool) -> None:
        """Record a trade outcome for performance tracking.

        Automatically disables persistently underperforming strategies.
        """
        if strategy_name not in self._perf:
            self._perf[strategy_name] = {"trades": 0, "wins": 0, "losses": 0}
        self._perf[strategy_name]["trades"] += 1
        if win:
            self._perf[strategy_name]["wins"] += 1
        else:
            self._perf[strategy_name]["losses"] += 1

        # Auto-disable underperformers
        trades = self._perf[strategy_name]["trades"]
        if trades >= self.MIN_TRADES_TO_EVALUATE:
            win_rate = self._perf[strategy_name]["wins"] / trades
            if win_rate < self.MIN_WIN_RATE_THRESHOLD:
                for s in self._strategies:
                    if s.name == strategy_name and s.enabled:
                        s.enabled = False
                        logger.warning(
                            "Auto-disabled underperforming strategy {} (win_rate={:.1%})",
                            strategy_name, win_rate
                        )

    async def evaluate_all(
        self,
        symbol: str,
        ohlcv_data: Dict[str, pd.DataFrame],  # timeframe → DataFrame
        ticker: Any,
        session: str,
        regime: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        """Run all enabled strategies and return a consensus signal.

        Args:
            symbol: Trading symbol.
            ohlcv_data: Dict of timeframe → OHLCV DataFrame.
            ticker: Current ticker data.
            session: Current trading session name.
            regime: Market regime label.

        Returns:
            Consensus signal dict or ``None`` if confluence is insufficient.
        """
        long_votes: List[Tuple[float, str]] = []  # (weighted_strength, strategy_name)
        short_votes: List[Tuple[float, str]] = []

        for strategy in self._strategies:
            if not strategy.enabled:
                continue
            try:
                sig: Signal = await strategy.generate_signal(symbol)
                if sig.direction == "neutral":
                    continue

                # Weight = signal strength × confidence × session multiplier × win_rate
                session_mult = self._get_session_multiplier(strategy.name, session)
                win_rate_weight = self._get_win_rate(strategy.name)
                weight = sig.strength * sig.confidence * session_mult * win_rate_weight

                if sig.direction == "long":
                    long_votes.append((weight, strategy.name))
                elif sig.direction == "short":
                    short_votes.append((weight, strategy.name))

            except Exception as e:
                logger.debug("Strategy {} error for {}: {}", strategy.name, symbol, e)

        total_long = len(long_votes)
        total_short = len(short_votes)

        logger.debug(
            "Consensus for {}: {} long votes, {} short votes (min confluence={})",
            symbol, total_long, total_short, self.MIN_CONFLUENCE
        )

        # Require minimum confluence
        if total_long < self.MIN_CONFLUENCE and total_short < self.MIN_CONFLUENCE:
            return None

        if total_long >= self.MIN_CONFLUENCE and total_long > total_short:
            direction = "long"
            votes = long_votes
        elif total_short >= self.MIN_CONFLUENCE and total_short > total_long:
            direction = "short"
            votes = short_votes
        else:
            return None  # tie or neither meets threshold

        if not votes:
            return None

        # Aggregate strength
        total_weight = sum(w for w, _ in votes)
        avg_strength = total_weight / len(votes)
        strategy_names = [n for _, n in votes]

        return {
            "symbol": symbol,
            "direction": direction,
            "strength": round(min(avg_strength, 1.0), 4),
            "confidence": round(min(avg_strength, 1.0), 4),
            "strategies": strategy_names,
            "long_votes": total_long,
            "short_votes": total_short,
            "session": session,
            "regime": regime,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return strategy performance statistics."""
        return {
            "total": len(self._strategies),
            "enabled": sum(1 for s in self._strategies if s.enabled),
            "performance": {
                name: {
                    "trades": p["trades"],
                    "win_rate": round(p["wins"] / p["trades"], 3) if p["trades"] > 0 else 0.0,
                }
                for name, p in self._perf.items()
            },
        }
