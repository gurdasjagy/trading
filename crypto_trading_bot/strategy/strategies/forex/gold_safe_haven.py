"""Gold Safe Haven Flow strategy — long gold when risk-off signals fire."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal

# Default VIX proxy thresholds
_VIX_SPIKE_THRESHOLD = 25.0   # VIX above this = fear / risk-off
_SPX_DROP_THRESHOLD = -0.02   # S&P 500 daily return below this = risk-off


class GoldSafeHavenStrategy(BaseStrategy):
    """Safe Haven Flow strategy for Gold.

    When equity markets crash (VIX spike, S&P 500 drop), gold rallies as a
    safe haven.  This strategy monitors risk-sentiment indicators and goes
    long gold during risk-off events.

    Risk-off detection
    ------------------
    * If ``vix_value`` is provided and above ``vix_threshold``, risk-off.
    * If ``spx_ohlcv`` is provided and the last bar's return is below
      ``spx_drop_threshold``, risk-off.
    * As a proxy without external data: use gold's own momentum vs a longer
      MA — a sharp gold rally in the absence of normal technical signals
      indicates safe-haven buying.  Only generate signals when a genuine
      spike is detected (price up > 0.5 × ATR in the last bar).

    Strategy only generates LONG signals (gold is a safe haven; shorting
    in panic is high-risk and not recommended here).
    """

    _STRATEGY_NAME = "gold_safe_haven"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        vix_threshold: float = _VIX_SPIKE_THRESHOLD,
        spx_drop_threshold: float = _SPX_DROP_THRESHOLD,
        momentum_atr_mult: float = 0.5,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._vix_threshold = vix_threshold
        self._spx_drop_threshold = spx_drop_threshold
        self._momentum_atr_mult = momentum_atr_mult
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        symbol: str = "",
        vix_value: Optional[float] = None,
        spx_ohlcv: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyse gold OHLCV with optional risk-sentiment data.

        Args:
            ohlcv: XAU/USD OHLCV DataFrame.
            symbol: Trading symbol.
            vix_value: Current VIX level (optional).
            spx_ohlcv: S&P 500 OHLCV DataFrame (optional); last bar return used.
        """
        if ohlcv is None or len(ohlcv) < 50:
            return None

        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        closes = ohlcv["close"].tolist()
        curr_price = closes[-1]
        prev_price = closes[-2]

        risk_off = False
        risk_sources: List[str] = []

        # External risk signal: VIX
        if vix_value is not None and vix_value >= self._vix_threshold:
            risk_off = True
            risk_sources.append(f"VIX={vix_value:.1f}")

        # External risk signal: S&P 500 drop
        if spx_ohlcv is not None and len(spx_ohlcv) >= 2:
            spx_return = (
                float(spx_ohlcv["close"].iloc[-1]) / float(spx_ohlcv["close"].iloc[-2]) - 1.0
            )
            if spx_return <= self._spx_drop_threshold:
                risk_off = True
                risk_sources.append(f"SPX_ret={spx_return:.2%}")

        # Proxy: gold price spike on current bar
        price_change = curr_price - prev_price
        if price_change >= self._momentum_atr_mult * atr:
            risk_off = True
            risk_sources.append(f"gold_spike={price_change:.4f}")

        if not risk_off:
            return None

        # Additional RSI filter: avoid entering into extreme over-bought territory
        rsi = self._calculate_rsi(closes)
        if rsi > 80:
            return None

        # Confidence based on number of confirming risk signals
        base_conf = 0.55 + min(len(risk_sources), 3) * 0.12
        if vix_value is not None:
            vix_factor = min(1.0, (vix_value - self._vix_threshold) / 20.0)
            base_conf += vix_factor * 0.08
        confidence = round(min(0.9, base_conf), 3)

        return {
            "symbol": symbol,
            "direction": "long",
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "risk_sources": risk_sources,
            "rsi": round(rsi, 2),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=100)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No safe-haven signal")

            atr = sig["atr"]
            entry = sig["entry_price"]
            stop_loss = entry - atr * 2.0
            take_profit = entry + atr * 4.0

            return Signal(
                symbol=symbol,
                direction="long",
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"Safe-haven gold long: {', '.join(sig['risk_sources'])}, "
                    f"RSI={sig['rsi']:.1f}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        if ohlcv.empty:
            return False
        # Close safe-haven position when RSI > 70 (risk exhausted)
        closes = ohlcv["close"].tolist()
        rsi = self._calculate_rsi(closes)
        return rsi > 70

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 2.0) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 2.5),
            "leverage": 2,
        }
