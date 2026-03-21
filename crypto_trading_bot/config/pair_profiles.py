"""Per-trading-pair optimised configuration profiles.

Each profile contains recommended settings for leverage, stop-loss / take-profit
multipliers, preferred strategies, and position-size limits tuned to the
historical behaviour of the pair.

Usage::

    from config.pair_profiles import get_pair_profile

    profile = get_pair_profile("BTC/USDT")
    leverage = profile["leverage_range"]      # (min, max)
    sl_mult  = profile["sl_atr_multiplier"]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Profile type alias (plain dict for easy JSON serialisation)
# ---------------------------------------------------------------------------
PairProfile = Dict[str, Any]

# ---------------------------------------------------------------------------
# Default fallback profile (conservative)
# ---------------------------------------------------------------------------
_DEFAULT_PROFILE: PairProfile = {
    "leverage_range": (1, 3),
    "sl_atr_multiplier": 2.0,
    "tp_rr_multipliers": (1.0, 2.0, 3.0),
    "tp_proportions": (0.30, 0.30, 0.40),
    "max_position_pct": 0.05,  # % of total capital
    "preferred_strategies": ["trend_following", "mean_reversion"],
    "strategy_bias": "neutral",  # "trend", "momentum", "mean_reversion", "neutral"
    "volatility_category": "medium",  # "low", "medium", "high"
}

# ---------------------------------------------------------------------------
# Per-pair profiles
# ---------------------------------------------------------------------------
_PROFILES: Dict[str, PairProfile] = {
    # ── Bitcoin ────────────────────────────────────────────────────────────
    "BTC/USDT": {
        "leverage_range": (3, 5),
        "sl_atr_multiplier": 2.5,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.20,
        "preferred_strategies": [
            "trend_following",
            "ema_ribbon",
            "supertrend",
            "donchian_breakout",
            "whale_follower",
            "on_chain_momentum",
        ],
        "strategy_bias": "trend",
        "volatility_category": "low",
    },
    # ── Ethereum ───────────────────────────────────────────────────────────
    "ETH/USDT": {
        "leverage_range": (5, 7),
        "sl_atr_multiplier": 2.0,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.15,
        "preferred_strategies": [
            "momentum",
            "macd_crossover",
            "technical_breakout",
            "smart_money_flow",
            "on_chain_momentum",
        ],
        "strategy_bias": "momentum",
        "volatility_category": "medium",
    },
    # ── Solana ─────────────────────────────────────────────────────────────
    "SOL/USDT": {
        "leverage_range": (5, 10),
        "sl_atr_multiplier": 1.8,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.10,
        "preferred_strategies": [
            "volatility_breakout",
            "donchian_breakout",
            "technical_breakout",
            "momentum",
            "liquidation_hunter",
        ],
        "strategy_bias": "momentum",
        "volatility_category": "high",
    },
    # ── BNB ────────────────────────────────────────────────────────────────
    "BNB/USDT": {
        "leverage_range": (3, 7),
        "sl_atr_multiplier": 2.0,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.10,
        "preferred_strategies": [
            "trend_following",
            "momentum",
            "mean_reversion",
            "technical_breakout",
        ],
        "strategy_bias": "neutral",
        "volatility_category": "medium",
    },
    # ── XRP ────────────────────────────────────────────────────────────────
    "XRP/USDT": {
        "leverage_range": (3, 5),
        "sl_atr_multiplier": 2.2,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.08,
        "preferred_strategies": [
            "trend_following",
            "mean_reversion",
            "sentiment_reversal",
        ],
        "strategy_bias": "mean_reversion",
        "volatility_category": "medium",
    },
    # ── ADA ────────────────────────────────────────────────────────────────
    "ADA/USDT": {
        "leverage_range": (3, 5),
        "sl_atr_multiplier": 2.2,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.08,
        "preferred_strategies": [
            "mean_reversion",
            "grid_trading",
            "dca",
            "sentiment_reversal",
        ],
        "strategy_bias": "mean_reversion",
        "volatility_category": "high",
    },
    # ── DOGE ───────────────────────────────────────────────────────────────
    "DOGE/USDT": {
        "leverage_range": (3, 5),
        "sl_atr_multiplier": 2.5,
        "tp_rr_multipliers": (1.0, 2.0, 3.0),
        "tp_proportions": (0.30, 0.30, 0.40),
        "max_position_pct": 0.06,
        "preferred_strategies": [
            "sentiment_reversal",
            "mean_reversion",
            "scalping",
        ],
        "strategy_bias": "mean_reversion",
        "volatility_category": "high",
    },
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_pair_profile(symbol: str) -> PairProfile:
    """Return the profile for *symbol*, falling back to the default profile.

    The lookup strips the ``/USDT`` suffix variants so ``"BTCUSDT"`` and
    ``"BTC/USDT"`` both resolve to the BTC profile.

    Args:
        symbol: Trading pair symbol (e.g. ``"BTC/USDT"`` or ``"BTCUSDT"``).

    Returns:
        A copy of the :data:`PairProfile` dict so callers can modify it safely.
    """
    # Normalise: insert slash if missing, uppercase
    normalised = symbol.upper()
    if "/" not in normalised and normalised.endswith("USDT"):
        normalised = normalised[:-4] + "/USDT"
    return dict(_PROFILES.get(normalised, _DEFAULT_PROFILE))


def list_profiles() -> List[str]:
    """Return the list of symbols with explicit profiles.

    Returns:
        Sorted list of symbol strings.
    """
    return sorted(_PROFILES.keys())


def get_max_leverage(symbol: str) -> int:
    """Convenience helper — return the maximum leverage for *symbol*.

    Args:
        symbol: Trading pair symbol.

    Returns:
        Maximum leverage as an integer.
    """
    profile = get_pair_profile(symbol)
    leverage_range: Tuple[int, int] = profile["leverage_range"]
    return int(leverage_range[1])


def get_preferred_strategies(symbol: str) -> List[str]:
    """Return the list of preferred strategy names for *symbol*.

    Args:
        symbol: Trading pair symbol.

    Returns:
        List of strategy name strings.
    """
    return list(get_pair_profile(symbol).get("preferred_strategies", []))
