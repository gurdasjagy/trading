"""Technical indicators — 50+ indicators via pandas-ta."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

try:
    import pandas_ta as ta  # type: ignore

    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.warning("pandas-ta not installed — some indicators will be unavailable")


class TechnicalIndicators:
    """Calculates a comprehensive set of technical indicators.

    All methods accept either a :class:`pandas.DataFrame` with OHLCV columns
    (``open``, ``high``, ``low``, ``close``, ``volume``) or a plain list of
    prices, depending on the indicator.

    The :meth:`calculate_all` convenience method returns every indicator as
    a single flat dictionary.
    """

    # ------------------------------------------------------------------
    # Master calculator
    # ------------------------------------------------------------------

    def calculate_all(self, ohlcv_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all available indicators and return a flat dict."""
        result: Dict[str, Any] = {}
        closes = ohlcv_df["close"].tolist() if "close" in ohlcv_df.columns else []

        try:
            result["rsi_14"] = self.rsi(closes)
            result["rsi_7"] = self.rsi(closes, period=7)
        except Exception as exc:
            logger.warning(f"RSI calculation failed: {exc}")

        try:
            macd_vals = self.macd(closes)
            result.update(macd_vals)
        except Exception as exc:
            logger.warning(f"MACD calculation failed: {exc}")

        try:
            bb = self.bollinger_bands(closes)
            result.update(bb)
        except Exception as exc:
            logger.warning(f"Bollinger Bands calculation failed: {exc}")

        try:
            result["ema_9"] = self.ema(closes, 9)
            result["ema_21"] = self.ema(closes, 21)
            result["ema_50"] = self.ema(closes, 50)
            result["ema_200"] = self.ema(closes, 200)
        except Exception as exc:
            logger.warning(f"EMA calculation failed: {exc}")

        try:
            result["sma_20"] = self.sma(closes, 20)
            result["sma_50"] = self.sma(closes, 50)
            result["sma_200"] = self.sma(closes, 200)
        except Exception as exc:
            logger.warning(f"SMA calculation failed: {exc}")

        try:
            result["atr_14"] = self.atr(ohlcv_df)
        except Exception as exc:
            logger.warning(f"ATR calculation failed: {exc}")

        try:
            result["vwap"] = self.vwap(ohlcv_df)
        except Exception as exc:
            logger.warning(f"VWAP calculation failed: {exc}")

        try:
            adx_vals = self.adx(ohlcv_df)
            result.update(adx_vals)
        except Exception as exc:
            logger.warning(f"ADX calculation failed: {exc}")

        try:
            stoch = self.stochastic(ohlcv_df)
            result.update(stoch)
        except Exception as exc:
            logger.warning(f"Stochastic calculation failed: {exc}")

        try:
            ichi = self.ichimoku(ohlcv_df)
            result.update(ichi)
        except Exception as exc:
            logger.warning(f"Ichimoku calculation failed: {exc}")

        try:
            fib = self.fibonacci_levels(ohlcv_df)
            result["fibonacci"] = fib
        except Exception as exc:
            logger.warning(f"Fibonacci calculation failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Individual indicators
    # ------------------------------------------------------------------

    def rsi(self, prices: List[float], period: int = 14) -> float:
        """Return the most recent RSI value."""
        if _HAS_PANDAS_TA and len(prices) >= period + 1:
            series = pd.Series(prices)
            result = ta.rsi(series, length=period)
            if result is not None and not result.empty:
                val = result.iloc[-1]
                if pd.notna(val):
                    return float(val)
        return self._rsi_fallback(prices, period)

    def macd(
        self,
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Dict[str, float]:
        """Return MACD line, signal line, and histogram."""
        if _HAS_PANDAS_TA and len(prices) >= slow:
            series = pd.Series(prices)
            result = ta.macd(series, fast=fast, slow=slow, signal=signal)
            if result is not None and not result.empty:
                cols = result.columns.tolist()
                last = result.iloc[-1]
                if not last.isna().all():
                    macd_col = next(
                        (
                            c
                            for c in cols
                            if "MACD_" in c
                            and "s" not in c.lower()[-2:]
                            and "h" not in c.lower()[-2:]
                        ),
                        cols[0],
                    )
                    sig_col = next(
                        (c for c in cols if "MACDs_" in c or c.endswith("s")),
                        cols[1] if len(cols) > 1 else cols[0],
                    )
                    hist_col = next(
                        (c for c in cols if "MACDh_" in c or c.endswith("h")),
                        cols[2] if len(cols) > 2 else cols[0],
                    )
                    return {
                        "macd": float(last.get(macd_col, 0.0) or 0.0),
                        "macd_signal": float(last.get(sig_col, 0.0) or 0.0),
                        "macd_histogram": float(last.get(hist_col, 0.0) or 0.0),
                    }
        return self._macd_fallback(prices, fast, slow, signal)

    def bollinger_bands(
        self, prices: List[float], period: int = 20, std: float = 2.0
    ) -> Dict[str, float]:
        """Return upper, middle, and lower Bollinger Bands."""
        if _HAS_PANDAS_TA and len(prices) >= period:
            series = pd.Series(prices)
            result = ta.bbands(series, length=period, std=std)
            if result is not None and not result.empty:
                last = result.iloc[-1]
                cols = result.columns.tolist()
                upper = next((c for c in cols if "BBU" in c), None)
                mid = next((c for c in cols if "BBM" in c), None)
                lower = next((c for c in cols if "BBL" in c), None)
                if upper and mid and lower:
                    return {
                        "bb_upper": float(last[upper] or 0.0),
                        "bb_middle": float(last[mid] or 0.0),
                        "bb_lower": float(last[lower] or 0.0),
                    }
        sma = self.sma(prices, period)
        if len(prices) >= period:
            import math

            segment = prices[-period:]
            variance = sum((p - sma) ** 2 for p in segment) / period
            sd = math.sqrt(variance) * std
            return {"bb_upper": sma + sd, "bb_middle": sma, "bb_lower": sma - sd}
        return {"bb_upper": 0.0, "bb_middle": 0.0, "bb_lower": 0.0}

    def ema(self, prices: List[float], period: int) -> float:
        """Return the most recent EMA value."""
        if not prices:
            return 0.0
        if _HAS_PANDAS_TA and len(prices) >= period:
            series = pd.Series(prices)
            result = ta.ema(series, length=period)
            if result is not None and not result.empty:
                val = result.iloc[-1]
                if pd.notna(val):
                    return float(val)
        k = 2.0 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def sma(self, prices: List[float], period: int) -> float:
        """Return the most recent SMA value."""
        if len(prices) < period:
            return float(prices[-1]) if prices else 0.0
        return sum(prices[-period:]) / period

    def atr(self, ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Return the most recent ATR value."""
        if _HAS_PANDAS_TA and len(ohlcv) >= period:
            result = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=period)
            if result is not None and not result.empty:
                val = result.iloc[-1]
                if pd.notna(val):
                    return float(val)
        return self._atr_fallback(ohlcv, period)

    def vwap(self, ohlcv: pd.DataFrame) -> float:
        """Return VWAP for the session."""
        if _HAS_PANDAS_TA and not ohlcv.empty:
            try:
                result = ta.vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
                if result is not None and not result.empty:
                    val = result.iloc[-1]
                    if pd.notna(val):
                        return float(val)
            except Exception:
                pass
        typical = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        total_vol = ohlcv["volume"].sum()
        return float((typical * ohlcv["volume"]).sum() / total_vol) if total_vol > 0 else 0.0

    def adx(self, ohlcv: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Return ADX, +DI, and -DI."""
        if _HAS_PANDAS_TA and len(ohlcv) >= period * 2:
            result = ta.adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=period)
            if result is not None and not result.empty:
                last = result.iloc[-1]
                cols = result.columns.tolist()
                adx_col = next((c for c in cols if c.startswith("ADX_")), None)
                dmp_col = next((c for c in cols if "DMP_" in c), None)
                dmn_col = next((c for c in cols if "DMN_" in c), None)
                if adx_col:
                    return {
                        "adx": float(last.get(adx_col, 0.0) or 0.0),
                        "di_plus": float(last.get(dmp_col, 0.0) or 0.0) if dmp_col else 0.0,
                        "di_minus": float(last.get(dmn_col, 0.0) or 0.0) if dmn_col else 0.0,
                    }
        return {"adx": 0.0, "di_plus": 0.0, "di_minus": 0.0}

    def stochastic(
        self,
        ohlcv: pd.DataFrame,
        k: int = 14,
        d: int = 3,
        smooth: int = 3,
    ) -> Dict[str, float]:
        """Return Stochastic %K and %D."""
        if _HAS_PANDAS_TA and len(ohlcv) >= k + d:
            result = ta.stoch(
                ohlcv["high"],
                ohlcv["low"],
                ohlcv["close"],
                k=k,
                d=d,
                smooth_k=smooth,
            )
            if result is not None and not result.empty:
                last = result.iloc[-1]
                cols = result.columns.tolist()
                k_col = next((c for c in cols if "STOCHk_" in c), None)
                d_col = next((c for c in cols if "STOCHd_" in c), None)
                return {
                    "stoch_k": float(last.get(k_col, 0.0) or 0.0) if k_col else 0.0,
                    "stoch_d": float(last.get(d_col, 0.0) or 0.0) if d_col else 0.0,
                }
        return {"stoch_k": 0.0, "stoch_d": 0.0}

    def ichimoku(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Return the latest Ichimoku cloud values."""
        if _HAS_PANDAS_TA and len(ohlcv) >= 52:
            try:
                span_a, span_b, tenkan, kijun = ta.ichimoku(
                    ohlcv["high"], ohlcv["low"], ohlcv["close"]
                )[:4]
                return {
                    "ichimoku_tenkan": (
                        float(tenkan.iloc[-1]) if tenkan is not None and not tenkan.empty else 0.0
                    ),
                    "ichimoku_kijun": (
                        float(kijun.iloc[-1]) if kijun is not None and not kijun.empty else 0.0
                    ),
                    "ichimoku_span_a": (
                        float(span_a.iloc[-1]) if span_a is not None and not span_a.empty else 0.0
                    ),
                    "ichimoku_span_b": (
                        float(span_b.iloc[-1]) if span_b is not None and not span_b.empty else 0.0
                    ),
                }
            except Exception:
                pass
        return {
            "ichimoku_tenkan": 0.0,
            "ichimoku_kijun": 0.0,
            "ichimoku_span_a": 0.0,
            "ichimoku_span_b": 0.0,
        }

    def fibonacci_levels(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Return Fibonacci retracement levels from the recent swing."""
        if ohlcv.empty:
            return {}
        high = float(ohlcv["high"].max())
        low = float(ohlcv["low"].min())
        diff = high - low
        return {
            "fib_0": high,
            "fib_236": high - diff * 0.236,
            "fib_382": high - diff * 0.382,
            "fib_500": high - diff * 0.500,
            "fib_618": high - diff * 0.618,
            "fib_786": high - diff * 0.786,
            "fib_100": low,
        }

    # ------------------------------------------------------------------
    # Pure-Python fallbacks (no pandas-ta dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi_fallback(prices: List[float], period: int) -> float:
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
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    @staticmethod
    def _macd_fallback(prices: List[float], fast: int, slow: int, signal: int) -> Dict[str, float]:
        def _ema(data: List[float], span: int) -> List[float]:
            k = 2.0 / (span + 1)
            out = [data[0]]
            for p in data[1:]:
                out.append(p * k + out[-1] * (1 - k))
            return out

        if len(prices) < slow:
            return {"macd": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0}
        ef = _ema(prices, fast)
        es = _ema(prices, slow)
        ml = [f - s for f, s in zip(ef, es)]
        sl = _ema(ml, signal)
        return {
            "macd": ml[-1],
            "macd_signal": sl[-1],
            "macd_histogram": ml[-1] - sl[-1],
        }

    @staticmethod
    def _atr_fallback(ohlcv: pd.DataFrame, period: int) -> float:
        if len(ohlcv) < period + 1:
            return 0.0
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        trs = [
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            for i in range(1, len(ohlcv))
        ]
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr
