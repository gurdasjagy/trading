"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class Signal:
    """Trading signal produced by a strategy."""

    symbol: str
    direction: str  # "long" / "short" / "neutral"
    strength: float  # 0.0–1.0
    confidence: float  # 0.0–1.0
    strategy_name: str
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 1

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.direction not in ("long", "short", "neutral"):
            raise ValueError(f"Invalid direction: {self.direction!r}")


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement :meth:`generate_signal`,
    :meth:`should_close`, and :meth:`calculate_parameters`.
    """

    def __init__(
        self,
        name: str,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
    ) -> None:
        self._name = name
        self._enabled = enabled
        self._symbols = symbols
        self._timeframe = timeframe
        self._exchange: Any = None  # set externally after construction

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    @property
    def timeframe(self) -> str:
        return self._timeframe

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate_signal(self, symbol: str) -> Signal:
        """Analyse *symbol* and return a :class:`Signal`."""

    @abstractmethod
    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Return *True* if an open *position* should be closed given *data*."""

    @abstractmethod
    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        """Return order parameters (size, stop-loss, take-profit, leverage) for a trade."""

    # ------------------------------------------------------------------
    # Shared helpers for subclasses
    # ------------------------------------------------------------------

    async def _get_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from the attached exchange client."""
        tf = timeframe or self._timeframe
        if self._exchange is None:
            logger.warning(f"[{self._name}] No exchange attached — returning empty DataFrame")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        try:
            return await self._exchange.get_ohlcv(symbol, tf, limit)
        except Exception as exc:
            logger.error(f"[{self._name}] Failed to fetch OHLCV for {symbol}: {exc}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    @staticmethod
    def _calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Return the most recent RSI value for *prices*."""
        if len(prices) < period + 1:
            return 50.0
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))

    @staticmethod
    def _calculate_macd(
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Dict[str, float]:
        """Return MACD line, signal line, and histogram."""

        def _ema(data: List[float], span: int) -> List[float]:
            k = 2.0 / (span + 1)
            result = [data[0]]
            for p in data[1:]:
                result.append(p * k + result[-1] * (1 - k))
            return result

        if len(prices) < slow:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        ema_fast = _ema(prices, fast)
        ema_slow = _ema(prices, slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = _ema(macd_line, signal)
        histogram = macd_line[-1] - signal_line[-1]
        return {
            "macd": macd_line[-1],
            "signal": signal_line[-1],
            "histogram": histogram,
        }

    @staticmethod
    def _calculate_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Return the most recent ATR value."""
        if len(ohlcv) < period + 1:
            return 0.0
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        trs = []
        for i in range(1, len(ohlcv)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return 0.0
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    @staticmethod
    def _calculate_ema(prices: List[float], period: int) -> float:
        """Return the most recent EMA value."""
        if not prices:
            return 0.0
        k = 2.0 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    # ------------------------------------------------------------------
    # Neutral signal helper
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """Analyse *ohlcv* DataFrame and return a signal dict or ``None``.

        Subclasses may override this method to provide OHLCV-driven analysis
        without requiring an attached exchange.  The base implementation
        always returns ``None`` so that existing strategies continue to work
        unchanged.

        Args:
            ohlcv: OHLCV DataFrame with columns open/high/low/close/volume.
            symbol: Trading pair symbol (e.g. ``"BTC/USDT"``).

        Returns:
            A signal dict with keys ``symbol``, ``direction``, ``entry_price``,
            ``atr``, ``confidence``, ``strategy``, ``timeframe`` or ``None``
            when no actionable signal is detected.
        """
        return None

    def _neutral_signal(self, symbol: str, reason: str = "No clear signal") -> Signal:
        """Return a neutral :class:`Signal` for *symbol*."""
        return Signal(
            symbol=symbol,
            direction="neutral",
            strength=0.0,
            confidence=0.0,
            strategy_name=self._name,
            reasoning=reason,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, enabled={self._enabled})"
