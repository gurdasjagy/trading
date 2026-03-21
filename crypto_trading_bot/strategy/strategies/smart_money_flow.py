"""Smart money flow strategy — tracks institutional accumulation/distribution."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class SmartMoneyFlowStrategy(BaseStrategy):
    """Detects institutional (smart money) accumulation or distribution.

    Combines three indicators
    -------------------------
    * **OBV trend** — On-Balance Volume direction.
    * **CMF** — Chaikin Money Flow (> 0 = buying pressure).
    * **On-chain net flow** — injected externally (positive = inflow to exchanges).
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="smart_money_flow",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._onchain_cache: Dict[str, float] = {}  # symbol -> net flow USD (+ = inflow)

    def update_onchain_flow(self, symbol: str, net_flow_usd: float) -> None:
        """Inject the latest on-chain net exchange flow for *symbol*."""
        self._onchain_cache[symbol] = net_flow_usd

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, limit=50)
            if len(ohlcv) < 20:
                return self._neutral_signal(symbol, "Insufficient OHLCV data")

            obv_trend = self._obv_trend(ohlcv)
            cmf = self._cmf(ohlcv)
            onchain = self._onchain_cache.get(symbol, 0.0)

            # Score: +1 per bullish factor, -1 per bearish factor
            score = 0.0
            reasons = []

            if obv_trend > 0:
                score += 1.0
                reasons.append("OBV rising")
            elif obv_trend < 0:
                score -= 1.0
                reasons.append("OBV falling")

            if cmf > 0.05:
                score += 1.0
                reasons.append(f"CMF={cmf:.3f} (buying pressure)")
            elif cmf < -0.05:
                score -= 1.0
                reasons.append(f"CMF={cmf:.3f} (selling pressure)")

            # Outflow from exchange = accumulation = bullish
            if onchain < -500_000:
                score += 1.0
                reasons.append(f"On-chain outflow ${abs(onchain)/1e6:.1f}M (accumulation)")
            elif onchain > 500_000:
                score -= 1.0
                reasons.append(f"On-chain inflow ${onchain/1e6:.1f}M (distribution)")

            reasoning = "; ".join(reasons) if reasons else "No clear smart-money signal"

            if score >= 2.0:
                strength = min(1.0, score / 3.0)
                return Signal(
                    symbol=symbol,
                    direction="long",
                    strength=round(strength, 3),
                    confidence=0.7,
                    strategy_name=self.name,
                    reasoning=f"Smart money accumulation — {reasoning}",
                    leverage=2,
                )
            if score <= -2.0:
                strength = min(1.0, abs(score) / 3.0)
                return Signal(
                    symbol=symbol,
                    direction="short",
                    strength=round(strength, 3),
                    confidence=0.7,
                    strategy_name=self.name,
                    reasoning=f"Smart money distribution — {reasoning}",
                    leverage=2,
                )

            return self._neutral_signal(symbol, f"Mixed smart-money signals (score={score:.1f})")
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=30)
        if ohlcv.empty:
            return False
        cmf = self._cmf(ohlcv)
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and cmf < -0.1:
            return True
        if side == "short" and cmf > 0.1:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.025
        return {
            "position_size_pct": 0.07,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.12, sl_pct * 2.5),
            "leverage": 2,
        }

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obv_trend(ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Return slope direction of OBV over the last *period* bars (+/- 1)."""
        closes = ohlcv["close"].values
        volumes = ohlcv["volume"].values
        obv = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        recent = obv[-period:]
        if len(recent) < 2:
            return 0.0
        return 1.0 if recent[-1] > recent[0] else -1.0

    @staticmethod
    def _cmf(ohlcv: pd.DataFrame, period: int = 20) -> float:
        """Return the Chaikin Money Flow value."""
        if len(ohlcv) < period:
            return 0.0
        df = ohlcv.tail(period)
        hl = df["high"] - df["low"]
        hl = hl.replace(0, 1e-9)
        mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl * df["volume"]
        total_vol = df["volume"].sum()
        return float(mfv.sum() / total_vol) if total_vol > 0 else 0.0
