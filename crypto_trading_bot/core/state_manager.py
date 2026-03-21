"""Global state management for the trading bot."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


class BotStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MarketRegime(str, Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    UNKNOWN = "unknown"


@dataclass
class GlobalState:
    status: BotStatus = BotStatus.STARTING
    circuit_breaker_active: bool = False
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_balance: float = 0.0
    available_balance: float = 0.0
    open_positions_count: int = 0
    consecutive_losses: int = 0
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    last_update: datetime = field(default_factory=_utcnow)
    trade_count_today: int = 0
    win_count_today: int = 0
    loss_count_today: int = 0


class StateManager:
    """Singleton async-safe global state manager."""

    _instance: Optional["StateManager"] = None

    @classmethod
    def get_instance(cls) -> "StateManager":
        """Return the singleton StateManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._state = GlobalState()
        self._state_lock = asyncio.Lock()
        self._symbol_states: Dict[str, Dict] = {}
        self._starting_equity: float = 10_000.0
        self._realized_pnl_today: float = 0.0

    async def update_state(self, **kwargs) -> None:
        """Thread-safe update of one or more GlobalState fields."""
        async with self._state_lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
                else:
                    logger.warning(f"Unknown state field: {key!r}")
            self._state.last_update = _utcnow()
        logger.debug(f"State updated: {kwargs}")

    async def get_state(self) -> GlobalState:
        """Return a snapshot of the current global state."""
        async with self._state_lock:
            return self._state

    async def update_symbol_state(self, symbol: str, data: dict) -> None:
        """Update per-symbol state data."""
        async with self._state_lock:
            if symbol not in self._symbol_states:
                self._symbol_states[symbol] = {}
            self._symbol_states[symbol].update(data)
            self._symbol_states[symbol]["last_update"] = _utcnow()

    async def get_symbol_state(self, symbol: str) -> dict:
        """Return per-symbol state; returns empty dict if not found."""
        async with self._state_lock:
            return dict(self._symbol_states.get(symbol, {}))

    async def record_trade_result(self, pnl: float, won: bool) -> None:
        """Update P&L and win/loss counters after a trade closes."""
        async with self._state_lock:
            self._state.daily_pnl += pnl
            self._state.trade_count_today += 1
            if won:
                self._state.win_count_today += 1
                self._state.consecutive_losses = 0
            else:
                self._state.loss_count_today += 1
                self._state.consecutive_losses += 1

            if self._state.total_balance > 0:
                self._state.daily_pnl_pct = (
                    self._state.daily_pnl / self._state.total_balance
                ) * 100
            self._state.last_update = _utcnow()

        logger.info(
            f"Trade recorded: pnl={pnl:+.4f}, won={won}, "
            f"daily_pnl={self._state.daily_pnl:+.4f}, "
            f"consecutive_losses={self._state.consecutive_losses}"
        )

    async def reset_daily_stats(self) -> None:
        """Reset all daily statistics (call at UTC midnight)."""
        async with self._state_lock:
            self._state.daily_pnl = 0.0
            self._state.daily_pnl_pct = 0.0
            self._state.trade_count_today = 0
            self._state.win_count_today = 0
            self._state.loss_count_today = 0
            self._state.consecutive_losses = 0
            self._state.circuit_breaker_active = False
            self._state.last_update = _utcnow()
            self._realized_pnl_today = 0.0
        logger.info("Daily stats reset.")

    def record_realized_pnl(self, pnl: float) -> None:
        """Add realized P&L from a closed or partially closed position."""
        self._realized_pnl_today += pnl

    def compute_accurate_pnl(
        self,
        positions: List[Dict],
        realized_pnl_today: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute accurate P&L including both unrealized and realized components.

        Args:
            positions: List of position dicts with unrealized_pnl.
            realized_pnl_today: Sum of realized P&L from closed/partially-closed
                positions today.  Defaults to the internally tracked value.

        Returns:
            Dict with total_pnl, unrealized_pnl, realized_pnl, and pnl_pct.
        """
        if realized_pnl_today is None:
            realized_pnl_today = self._realized_pnl_today
        unrealized = sum(float(p.get("unrealized_pnl", 0)) for p in positions)
        total = unrealized + realized_pnl_today
        starting_equity = self._starting_equity or 10_000.0
        pnl_pct = (total / starting_equity) * 100.0 if starting_equity > 0 else 0.0
        return {
            "total_pnl": total,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized_pnl_today,
            "pnl_pct": pnl_pct,
        }

    def is_trading_allowed(self) -> bool:
        """Return True only when the bot is running and the circuit breaker is off."""
        return self._state.status == BotStatus.RUNNING and not self._state.circuit_breaker_active
