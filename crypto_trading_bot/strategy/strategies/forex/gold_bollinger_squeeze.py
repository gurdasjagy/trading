"""Gold Bollinger Band Squeeze strategy — trade breakouts from low-volatility compression."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class GoldBollingerSqueezeStrategy(BaseStrategy):
    """Gold Bollinger Band Squeeze strategy.

    When Bollinger Bands narrow (squeeze), gold is about to make a big
    directional move.  This strategy detects the squeeze and trades the
    breakout direction with volume confirmation.

    Signal logic
    ------------
    * Detect squeeze: current BBW < MA of BBW over ``squeeze_ma`` periods.
    * Breakout: price closes outside the bands after a squeeze.
    * Volume must exceed ``volume_multiplier`` × average to confirm.
    * Confidence based on squeeze depth and volume ratio.

    This is a gold-specialised variant of BollingerSqueezeStrategy with
    gold-specific parameters (wider band, longer squeeze MA) and volume
    confirmation.
    """

    _STRATEGY_NAME = "gold_bollinger_squeeze"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        enabled: bool = True,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_ma: int = 30,
        volume_multiplier: float = 1.3,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            name=self._STRATEGY_NAME,
            symbols=symbols or [],
            timeframe=timeframe,
            enabled=enabled,
        )
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._squeeze_ma = squeeze_ma
        self._volume_multiplier = volume_multiplier
        self._atr_period = atr_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bbands(
        closes: pd.Series, period: int, std_mult: float
    ) -> Optional[Dict[str, float]]:
        """Return upper, middle, lower bands and BBW for the latest bar."""
        if len(closes) < period:
            return None
        rolling = closes.rolling(period)
        mid = rolling.mean().iloc[-1]
        std = rolling.std().iloc[-1]
        if pd.isna(mid) or pd.isna(std):
            return None
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        return {"upper": upper, "mid": mid, "lower": lower, "bbw": upper - lower}

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, ohlcv: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        min_rows = self._bb_period + self._squeeze_ma + 10
        if ohlcv is None or len(ohlcv) < min_rows:
            return None

        closes = ohlcv["close"]
        volumes = ohlcv["volume"].tolist()
        atr = self._calculate_atr(ohlcv)
        if atr <= 0:
            return None

        bands = self._compute_bbands(closes, self._bb_period, self._bb_std)
        if bands is None:
            return None

        # Compute BBW series for squeeze detection
        rolling_mid = closes.rolling(self._bb_period).mean()
        rolling_std = closes.rolling(self._bb_period).std()
        bbw_series = (rolling_mid + self._bb_std * rolling_std) - (rolling_mid - self._bb_std * rolling_std)

        bbw_ma = float(bbw_series.rolling(self._squeeze_ma).mean().iloc[-1])
        curr_bbw = float(bbw_series.iloc[-1])

        if pd.isna(bbw_ma) or bbw_ma == 0:
            return None

        in_squeeze = curr_bbw < bbw_ma

        if not in_squeeze:
            return None

        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])

        # Volume confirmation
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        curr_vol = volumes[-1]
        volume_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

        if volume_ratio < self._volume_multiplier:
            return None

        direction: Optional[str] = None
        if curr_price > bands["upper"] and prev_price <= bands["upper"]:
            direction = "long"
        elif curr_price < bands["lower"] and prev_price >= bands["lower"]:
            direction = "short"

        if direction is None:
            return None

        squeeze_ratio = max(0.0, 1.0 - curr_bbw / bbw_ma)
        vol_factor = min(1.0, (volume_ratio - self._volume_multiplier) / 2.0)
        confidence = round(min(0.9, 0.52 + squeeze_ratio * 0.25 + vol_factor * 0.13), 3)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": curr_price,
            "atr": atr,
            "confidence": confidence,
            "strategy": self._STRATEGY_NAME,
            "timeframe": self._timeframe,
            "upper_band": round(bands["upper"], 4),
            "lower_band": round(bands["lower"], 4),
            "bbw": round(curr_bbw, 6),
            "bbw_ma": round(bbw_ma, 6),
            "volume_ratio": round(volume_ratio, 3),
        }

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            ohlcv = await self._get_ohlcv(symbol, self._timeframe, limit=200)
            sig = self.analyze(ohlcv, symbol)
            if sig is None:
                return self._neutral_signal(symbol, "No Bollinger squeeze on gold")

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
                    f"Gold BB squeeze breakout {direction}: "
                    f"BBW={sig['bbw']:.4f}<MA={sig['bbw_ma']:.4f}, "
                    f"vol_ratio={sig['volume_ratio']:.2f}"
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
        if sig is None:
            return False
        curr_price = float(ohlcv["close"].iloc[-1])
        side = str(getattr(position, "side", "long")).lower()
        if side == "long" and curr_price < sig["upper_band"]:
            return True
        if side == "short" and curr_price > sig["lower_band"]:
            return True
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=100)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 1.0
        sl_pct = (atr / last_price * 1.5) if last_price > 0 else 0.02
        return {
            "position_size_pct": 0.05,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.10, sl_pct * 2.0),
            "leverage": 3,
        }
