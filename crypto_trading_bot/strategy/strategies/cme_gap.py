"""CME Gap Strategy — trades gap fills on Monday open."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class CMEGapStrategy(BaseStrategy):
    """Detects and trades CME futures gaps on Monday open.
    
    Strategy Logic:
    - CME futures close Friday 21:00 UTC, reopen Sunday 17:00 UTC
    - If BTC/ETH price gaps >1% from Friday close, generates gap-fill signal
    - Base confidence: 70% (gaps tend to fill)
    - Only active on Sunday 17:00-23:59 UTC and Monday 00:00-12:00 UTC
    - Stop-loss at 2% from entry
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        min_gap_pct: float = 1.0,
    ) -> None:
        super().__init__(
            name="cme_gap",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._min_gap_pct = min_gap_pct
        self._cme_close_price: Dict[str, float] = {}  # symbol -> Friday close price

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            # Only trade BTC/ETH (CME futures symbols)
            base_symbol = symbol.split("/")[0]
            if base_symbol not in ("BTC", "ETH"):
                return self._neutral_signal(symbol, "Not a CME futures symbol")
            
            # Check if we're in the gap-fill window
            now = datetime.now(timezone.utc)
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            hour = now.hour
            
            # Active window: Sunday 17:00-23:59 or Monday 00:00-12:00
            is_active = (
                (weekday == 6 and hour >= 17) or  # Sunday evening
                (weekday == 0 and hour < 12)      # Monday morning
            )
            
            if not is_active:
                return self._neutral_signal(symbol, "Outside CME gap-fill window")
            
            # Fetch current price
            ohlcv = await self._get_ohlcv(symbol, limit=100)
            if len(ohlcv) < 50:
                return self._neutral_signal(symbol, "Insufficient OHLCV data")
            
            current_price = float(ohlcv["close"].iloc[-1])
            
            # Get Friday close price (21:00 UTC)
            # Look back to find Friday's 21:00 candle
            friday_close = self._get_friday_close(ohlcv)
            if friday_close is None or friday_close <= 0:
                return self._neutral_signal(symbol, "Could not determine Friday close")
            
            # Calculate gap
            gap_pct = abs(current_price - friday_close) / friday_close * 100.0
            
            if gap_pct < self._min_gap_pct:
                return self._neutral_signal(
                    symbol,
                    f"Gap too small: {gap_pct:.2f}% < {self._min_gap_pct:.0f}%"
                )
            
            # Determine direction: trade toward Friday close (gap fill)
            if current_price > friday_close:
                # Price gapped up -> short to fill gap
                direction = "short"
                stop_loss = current_price * 1.02
                take_profit = friday_close
            else:
                # Price gapped down -> long to fill gap
                direction = "long"
                stop_loss = current_price * 0.98
                take_profit = friday_close
            
            # Base confidence 70%, bonus for larger gaps
            confidence = min(0.85, 0.70 + (gap_pct - self._min_gap_pct) / 20.0)
            strength = min(1.0, gap_pct / 5.0)
            
            return Signal(
                symbol=symbol,
                direction=direction,
                strength=round(strength, 3),
                confidence=round(confidence, 3),
                strategy_name=self.name,
                reasoning=(
                    f"CME gap: {gap_pct:.2f}% gap from Friday close ${friday_close:.2f} "
                    f"to current ${current_price:.2f}, targeting gap fill"
                ),
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    def _get_friday_close(self, ohlcv: pd.DataFrame) -> float | None:
        """Find Friday 21:00 UTC close price from OHLCV data."""
        try:
            # Ensure timestamp column exists
            if "timestamp" not in ohlcv.columns:
                if ohlcv.index.name == "timestamp":
                    ohlcv = ohlcv.reset_index()
                else:
                    return None
            
            # Convert timestamp to datetime
            ohlcv["dt"] = pd.to_datetime(ohlcv["timestamp"], unit="ms")
            
            # Filter for Friday (weekday=4) at hour 21
            friday_candles = ohlcv[
                (ohlcv["dt"].dt.weekday == 4) &
                (ohlcv["dt"].dt.hour == 21)
            ]
            
            if friday_candles.empty:
                return None
            
            # Return most recent Friday close
            return float(friday_candles["close"].iloc[-1])
        except Exception as exc:
            logger.debug(f"_get_friday_close error: {exc}")
            return None

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if gap is filled or we exit the active window."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=5)
        if ohlcv.empty:
            return False
        
        current_price = float(ohlcv["close"].iloc[-1])
        entry_price = float(getattr(position, "entry_price", 0.0))
        side = str(getattr(position, "side", "long")).lower()
        
        # Close if gap is filled (price crossed entry in target direction)
        if side == "long" and current_price >= entry_price * 1.01:
            return True
        if side == "short" and current_price <= entry_price * 0.99:
            return True
        
        # Close if we're past Monday 12:00 UTC
        now = datetime.now(timezone.utc)
        if now.weekday() == 0 and now.hour >= 12:
            return True
        if now.weekday() > 0:
            return True
        
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.03,
            "leverage": 3,
        }
