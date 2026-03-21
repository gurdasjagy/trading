"""News momentum strategy — trades on breaking high-impact news."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class NewsMomentumStrategy(BaseStrategy):
    """Trades based on breaking news sentiment.

    Rules
    -----
    * Only acts on HIGH or CRITICAL impact news.
    * Ignores news older than 15 minutes.
    * Requires volume confirmation before entry.
    * Maximum 5 % risk per trade; always sets a stop-loss.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "5m",
        enabled: bool = True,
        max_news_age_minutes: int = 15,
    ) -> None:
        super().__init__(
            name="news_momentum",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._max_age = timedelta(minutes=max_news_age_minutes)
        self._news_cache: List[Dict[str, Any]] = []

    def update_news(self, news_items: List[Dict[str, Any]]) -> None:
        """Push fresh news items (from a data source) into the strategy."""
        self._news_cache = news_items

    async def generate_signal(self, symbol: str) -> Signal:
        """Generate a signal for *symbol* based on recent high-impact news."""
        try:
            relevant = self._filter_relevant_news(symbol)
            if not relevant:
                return self._neutral_signal(symbol, "No relevant high-impact news found")

            bullish = sum(1 for n in relevant if n.get("sentiment", "neutral").lower() == "bullish")
            bearish = sum(1 for n in relevant if n.get("sentiment", "neutral").lower() == "bearish")

            ohlcv = await self._get_ohlcv(symbol, limit=20)
            volume_ok = self._confirm_volume(ohlcv)

            if not volume_ok:
                return self._neutral_signal(symbol, "Volume confirmation failed")

            if bullish > bearish and bullish >= 1:
                strength = min(1.0, 0.5 + bullish * 0.15)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=strength,
                    confidence=0.7,
                    strategy_name=self.name,
                    reasoning=f"Bullish news ({bullish} items); volume confirmed",
                    leverage=2,
                )
            if bearish > bullish and bearish >= 1:
                strength = min(1.0, 0.5 + bearish * 0.15)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=strength,
                    confidence=0.7,
                    strategy_name=self.name,
                    reasoning=f"Bearish news ({bearish} items); volume confirmed",
                    leverage=2,
                )

            return self._neutral_signal(symbol, "Mixed news sentiment — no edge")
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if news sentiment reversed or is stale."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        relevant = self._filter_relevant_news(symbol)
        if not relevant:
            # No fresh news — close the news-driven position
            return True
        side = getattr(position, "side", "long")
        bearish_count = sum(1 for n in relevant if n.get("sentiment", "").lower() == "bearish")
        bullish_count = sum(1 for n in relevant if n.get("sentiment", "").lower() == "bullish")
        if side == "long" and bearish_count > bullish_count:
            return True
        if side == "short" and bullish_count > bearish_count:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        """Return conservative parameters (max 5 % risk, always stop-loss)."""
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = min(0.05, (atr / last_price) * 1.5) if last_price > 0 else 0.02
        tp_pct = sl_pct * 2.0
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct,
            "leverage": 2,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_relevant_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Return HIGH/CRITICAL news for *symbol* within the age limit."""
        now = datetime.now(tz=timezone.utc)
        base = symbol.replace("/USDT", "").replace("/USD", "")
        result = []
        for item in self._news_cache:
            impact = item.get("impact_level", "LOW").upper()
            if impact not in ("HIGH", "CRITICAL"):
                continue
            ts = item.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, tz=timezone.utc)
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if now - ts > self._max_age:
                continue
            assets = item.get("affected_symbols", [])
            if base in assets or not assets:
                result.append(item)
        return result

    @staticmethod
    def _confirm_volume(ohlcv: Any) -> bool:
        """Return True if the latest volume exceeds the 10-bar average."""
        if ohlcv is None or len(ohlcv) < 2:
            return False
        volumes = ohlcv["volume"].values
        avg = float(volumes[:-1].mean()) if len(volumes) > 1 else 0.0
        current = float(volumes[-1])
        return avg > 0 and current > avg * 1.2
