"""Dynamic leverage optimisation based on market conditions."""

from __future__ import annotations

from loguru import logger


class LeverageOptimizer:
    """Dynamically calculates optimal leverage based on market conditions."""

    # Volatility regime → leverage multiplier
    _VOLATILITY_MULTIPLIERS = {
        "low": 1.2,
        "normal": 1.0,
        "high": 0.6,
        "extreme": 0.3,
    }

    # Market regime → leverage multiplier
    _REGIME_MULTIPLIERS = {
        "trending_up": 1.1,
        "trending_down": 0.8,
        "ranging": 0.9,
        "breakout": 1.0,
        "reversal": 0.7,
        "unknown": 0.8,
    }

    def __init__(self, default_leverage: int = 5, max_leverage: int = 20) -> None:
        self._default_leverage = default_leverage
        self._max_leverage = max_leverage

    def calculate_optimal_leverage(
        self,
        symbol: str,
        max_leverage: int = 20,
        volatility_regime: str = "normal",
        market_regime: str = "unknown",
    ) -> int:
        """Calculate optimal leverage for *symbol* given current market conditions.

        Args:
            symbol: Trading symbol.
            max_leverage: Hard ceiling on leverage.
            volatility_regime: Current volatility regime label.
            market_regime: Current market regime label.

        Returns:
            Recommended leverage as an integer.
        """
        base = self._default_leverage
        base = self.adjust_for_volatility(base, volatility_regime)
        base = self.adjust_for_regime(base, market_regime)
        result = min(base, max_leverage, self._max_leverage)
        result = max(result, 1)
        logger.debug(
            "Optimal leverage for {}: base={} vol_regime={} mkt_regime={} result={}",
            symbol,
            self._default_leverage,
            volatility_regime,
            market_regime,
            result,
        )
        return result

    def adjust_for_volatility(self, base_leverage: int, volatility_regime: str) -> int:
        """Scale *base_leverage* according to the volatility regime.

        Args:
            base_leverage: Starting leverage value.
            volatility_regime: One of ``"low"``, ``"normal"``, ``"high"``, ``"extreme"``.

        Returns:
            Adjusted leverage as an integer.
        """
        mult = self._VOLATILITY_MULTIPLIERS.get(volatility_regime, 1.0)
        result = max(1, int(base_leverage * mult))
        logger.debug(
            "Volatility-adjusted leverage: base={} regime={} mult={} result={}",
            base_leverage,
            volatility_regime,
            mult,
            result,
        )
        return result

    def adjust_for_regime(self, base_leverage: int, market_regime: str) -> int:
        """Scale *base_leverage* according to the market regime.

        Args:
            base_leverage: Starting leverage value.
            market_regime: Market structure label.

        Returns:
            Adjusted leverage as an integer.
        """
        mult = self._REGIME_MULTIPLIERS.get(market_regime, 0.8)
        result = max(1, int(base_leverage * mult))
        logger.debug(
            "Regime-adjusted leverage: base={} regime={} mult={} result={}",
            base_leverage,
            market_regime,
            mult,
            result,
        )
        return result

    def calculate_volatility_based_leverage(
        self,
        symbol: str,
        realized_vol: float,
        predicted_vol: float,
        max_leverage: int,
        btc_daily_change: float = 0.0,
    ) -> int:
        """Calculate leverage directly from annualised volatility numbers.

        Args:
            symbol: Trading symbol (used for safe-leverage cap lookup).
            realized_vol: Annualised realized volatility (e.g. 0.80 = 80 %).
            predicted_vol: Annualised predicted volatility.
            max_leverage: Caller-supplied hard ceiling on leverage.
            btc_daily_change: BTC daily % change as a decimal (negative = drop).
                When BTC drops > 5 %, altcoin leverage is halved.

        Returns:
            Recommended leverage as an integer (minimum 1).
        """
        # Use the higher of realized and predicted volatility for conservatism
        vol = max(realized_vol, predicted_vol)

        if vol > 1.50:
            leverage = max(1, int(max_leverage * 0.1))
        elif vol > 0.80:
            leverage = max(1, int(max_leverage * 0.3))
        elif vol > 0.50:
            leverage = max(1, int(max_leverage * 0.5))
        elif vol > 0.30:
            leverage = max(1, int(max_leverage * 0.7))
        else:
            leverage = max(1, int(max_leverage * 0.9))

        # BTC dominance / contagion factor: when BTC crashes, reduce altcoin risk
        base_symbol = symbol.split("/")[0].upper()
        if base_symbol not in ("BTC",) and btc_daily_change < -0.05:
            leverage = max(1, leverage // 2)
            logger.info(
                "BTC daily drop {:.1%} — halving altcoin leverage for {} to {}x",
                btc_daily_change,
                symbol,
                leverage,
            )

        # Never exceed the symbol-specific safe cap
        safe_cap = self.get_max_safe_leverage(symbol)
        leverage = min(leverage, safe_cap, max_leverage, self._max_leverage)
        leverage = max(leverage, 1)

        logger.debug(
            "Volatility-based leverage for {}: rv={:.2%} pv={:.2%} result={}x",
            symbol,
            realized_vol,
            predicted_vol,
            leverage,
        )
        return leverage

    def get_max_safe_leverage(self, symbol: str) -> int:
        """Return the maximum safe leverage for *symbol*.

        Conservative symbols (BTC, ETH) support higher leverage than
        smaller-cap assets.

        Args:
            symbol: Trading symbol.

        Returns:
            Maximum safe leverage.
        """
        safe_caps = {
            "BTC/USDT": 20,
            "ETH/USDT": 15,
            "BNB/USDT": 10,
            "SOL/USDT": 10,
            "XRP/USDT": 10,
        }
        cap = safe_caps.get(symbol, 5)
        logger.debug("Max safe leverage for {}: {}", symbol, cap)
        return cap
