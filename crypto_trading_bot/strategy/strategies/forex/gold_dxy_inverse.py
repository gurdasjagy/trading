"""Gold-Dollar Inverse Correlation strategy — long gold when USD weakens."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldDXYInverseStrategy(BaseStrategy):
    """Gold-Dollar Inverse Correlation strategy for XAU/USD.

    Gold moves inversely to USD strength.  When the USD weakens (proxied by
    the recent DXY-like behaviour of OHLCV momentum), go long gold; when USD
    strengthens, go short gold.

    Because direct DXY data is not always available, this strategy uses price
    momentum of the gold instrument itself as a proxy for dollar weakness:
    a declining USD environment typically produces an *accelerating* gold
    up-move, while a strengthening USD produces a prolonged gold down-move.

    If a second DataFrame (dxy_ohlcv) is passed via ``analyze``, real DXY
    correlation is computed (correlation between gold closes and DXY closes
    over the lookback window).
    """

    _STRATEGY_NAME = "gold_dxy_inverse"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        lookback: int = 20,
        correlation_threshold: float = -0.5,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._lookback = lookback
        self._correlation_threshold = correlation_threshold
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        symbol: str = "",
        dxy_ohlcv: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        if ohlcv is None or len(ohlcv) < max(self._lookback + 5, 50):
            return None

        closes = ohlcv["close"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        curr_price = closes[-1]

        if dxy_ohlcv is not None and len(dxy_ohlcv) >= self._lookback:
            # Real DXY correlation path
            gold_slice = pd.Series(closes[-self._lookback:])
            dxy_slice = dxy_ohlcv["close"].iloc[-self._lookback:].reset_index(drop=True)
            if len(dxy_slice) == self._lookback:
                correlation = float(gold_slice.corr(dxy_slice))
            else:
                correlation = None
        else:
            correlation = None

        # Proxy: use gold's own momentum vs a longer MA to infer USD direction
        short_ma = self._calculate_ema(closes, 10)
        long_ma = self._calculate_ema(closes, self._lookback)

        if long_ma <= 0:
            return None

        momentum_ratio = (short_ma - long_ma) / long_ma

        # If real DXY correlation available, use it; otherwise rely on momentum
        if correlation is not None:
            if correlation > self._correlation_threshold:
                # Correlation not strongly inverse — no signal
                return None
            direction = "long" if momentum_ratio > 0 else "short"
            confidence_base = min(0.9, abs(correlation) * 0.8 + 0.1)
        else:
            # Proxy-only path: require a meaningful momentum divergence
            threshold = 0.005  # 0.5 % divergence
            if abs(momentum_ratio) < threshold:
                return None
            direction = "long" if momentum_ratio > 0 else "short"
            confidence_base = min(0.75, 0.45 + min(abs(momentum_ratio) / 0.02, 1.0) * 0.3)

        rsi = self._calculate_rsi(closes)
        # Avoid over-bought longs or over-sold shorts
        if direction == "long" and rsi > 75:
            return None
        if direction == "short" and rsi < 25:
            return None

        confidence = round(confidence_base, 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "momentum_ratio": round(momentum_ratio, 6),
            "correlation": round(correlation, 4) if correlation is not None else None,
            "rsi": round(rsi, 2),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=150)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No DXY-inverse signal")

            direction = sig["direction"]
            atr = sig["atr"]
            entry = sig["entry_price"]

            if direction == "long":
                stop_loss = entry - atr * 1.5
                take_profit = entry + atr * 3.0
            else:
                stop_loss = entry + atr * 1.5
                take_profit = entry - atr * 3.0

            return Signal(
                symbol=symbol,
                direction=direction,
                strength=sig["confidence"],
                confidence=sig["confidence"],
                strategy_name=self.name,
                reasoning=(
                    f"DXY-inverse {direction}: momentum={sig['momentum_ratio']:.4f}, "
                    f"RSI={sig['rsi']:.1f}, corr={sig['correlation']}"
                ),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                leverage=3,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        if ohlcv.empty:
            return False
        sig = self.analyze(ohlcv, symbol)
        return sig is not None and sig["direction"] != str(
            getattr(position, "side", "long")
        ).lower()

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=60)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.04, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 3,
        }
