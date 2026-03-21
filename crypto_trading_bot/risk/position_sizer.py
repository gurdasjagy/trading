"""Position sizing algorithms: Kelly, Fixed Fraction, Fixed Ratio, Volatility-adjusted, Equal-weight.

All sizing methods in this module return values in **USDT (quote currency)** unless otherwise
noted.  Before submitting an order to a futures exchange, callers must convert the USDT amount
to base currency units (e.g. BTC, ETH) using :meth:`PositionSizer.convert_to_base_units`.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class SizingMethod(str, Enum):
    KELLY = "kelly"
    FIXED_FRACTION = "fixed_fraction"
    FIXED_RATIO = "fixed_ratio"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    EQUAL_WEIGHT = "equal_weight"


class PositionSizer:
    """Implements multiple position sizing methods."""

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> float:
        """Calculate half-Kelly position size.

        Args:
            win_rate: Historical win rate (0–1).
            avg_win: Average winning trade return.
            avg_loss: Average losing trade return (positive value).
            capital: Total available capital.

        Returns:
            Recommended position size in currency units.
        """
        if avg_loss <= 1e-10 or capital <= 0:
            return 0.0
        try:
            loss_rate = 1.0 - win_rate
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
            kelly_pct = win_rate - (loss_rate / win_loss_ratio) if win_loss_ratio > 0 else 0.0
            # Apply half-Kelly to reduce risk
            half_kelly = max(0.0, kelly_pct / 2.0)
            # Cap at 25% of capital
            half_kelly = min(half_kelly, 0.25)
            size = capital * half_kelly
            logger.debug(
                "Kelly size: win_rate={} kelly_pct={:.4f} half_kelly={:.4f} size={:.2f}",
                win_rate,
                kelly_pct,
                half_kelly,
                size,
            )
            return size
        except Exception as exc:
            logger.error("Kelly size calculation failed: {}", exc)
            return 0.0

    def fixed_fraction_size(self, capital: float, risk_pct: float) -> float:
        """Calculate fixed-fraction position size.

        Args:
            capital: Total available capital.
            risk_pct: Fraction of capital to risk per trade (0–1).

        Returns:
            Recommended position size in currency units.
        """
        if capital <= 0 or risk_pct <= 0:
            return 0.0
        size = capital * min(risk_pct, 1.0)
        logger.debug(
            "Fixed fraction size: capital={} risk_pct={} size={:.2f}", capital, risk_pct, size
        )
        return size

    def volatility_adjusted_size(
        self,
        capital: float,
        atr: float,
        entry_price: float,
        max_risk_pct: float = 0.02,
    ) -> float:
        """Calculate volatility-adjusted position size.

        Sizes the position so that one ATR move equals *max_risk_pct* of capital.

        Args:
            capital: Total available capital.
            atr: Average True Range of the instrument.
            entry_price: Intended entry price.
            max_risk_pct: Maximum fraction of capital to risk per ATR move.

        Returns:
            Recommended position size in base units.
        """
        if atr <= 0 or entry_price <= 0 or capital <= 0:
            return 0.0
        try:
            risk_amount = capital * max_risk_pct
            size = risk_amount / atr
            logger.debug(
                "Volatility-adjusted size: atr={} risk_amount={:.2f} size={:.6f}",
                atr,
                risk_amount,
                size,
            )
            return size
        except Exception as exc:
            logger.error("Volatility-adjusted size calculation failed: {}", exc)
            return 0.0

    def fixed_ratio_size(
        self,
        capital: float,
        delta: float = 1000.0,
        initial_contracts: float = 1.0,
    ) -> float:
        """Calculate fixed-ratio position size.

        Uses the Ryan Jones fixed-ratio formula: N = 0.5 * (1 + sqrt(1 + 8*P/delta)).

        Args:
            capital: Current capital / profit above starting capital.
            delta: Dollar increment required to add one contract.
            initial_contracts: Starting number of contracts.

        Returns:
            Number of contracts to trade.
        """
        if delta <= 0:
            return initial_contracts
        try:
            import math

            n = 0.5 * (1.0 + math.sqrt(1.0 + 8.0 * max(capital, 0) / delta))
            contracts = max(initial_contracts, n)
            logger.debug(
                "Fixed ratio size: capital={} delta={} contracts={:.2f}", capital, delta, contracts
            )
            return contracts
        except Exception as exc:
            logger.error("Fixed ratio size calculation failed: {}", exc)
            return initial_contracts

    def equal_weight_size(self, capital: float, num_positions: int) -> float:
        """Return an equal-weight allocation.

        Args:
            capital: Total capital to distribute.
            num_positions: Number of positions to distribute across.

        Returns:
            Capital allocation per position.
        """
        if num_positions <= 0 or capital <= 0:
            return 0.0
        return capital / num_positions

    def adaptive_size(
        self,
        capital: float,
        atr: float,
        entry_price: float,
        volatility_regime: str = "normal",
        win_streak: int = 0,
        loss_streak: int = 0,
        realized_vol: float = 0.5,
        win_rate: float = 0.5,
        avg_win: float = 0.01,
        avg_loss: float = 0.01,
    ) -> float:
        """Calculate an adaptive position size that accounts for volatility and streaks.

        Base sizing rules:
        * Risk 1 % of capital per trade.
        * Volatility adjustment: divide by (realized_vol / 0.5) — higher vol = smaller size.
        * Win streak bonus: after 3 consecutive wins, +10 % per extra win (max +30 %).
        * Loss streak penalty: after 2 losses reduce by 25 %; after 3, by 50 %.
        * Hard cap: never risk more than 2 % of capital.
        * Extreme vol cap: if volatility_regime == ``"extreme"``, cap at 0.5 % risk.
        * Kelly override: if Kelly suggests > 2× adaptive size, cap at adaptive size.

        Args:
            capital: Total available capital (quote currency).
            atr: Average True Range of the instrument.
            entry_price: Intended entry price.
            volatility_regime: One of ``"low"``, ``"normal"``, ``"high"``, ``"extreme"``.
            win_streak: Number of consecutive recent wins.
            loss_streak: Number of consecutive recent losses.
            realized_vol: Realised volatility (annualised, as a decimal).
                Default 0.5 represents 50 % annualised vol.
            win_rate: Historical win rate (0–1) for Kelly calculation.
            avg_win: Average winning trade return (for Kelly).
            avg_loss: Average losing trade return, positive value (for Kelly).

        Returns:
            Recommended position size in currency units (USDT).
        """
        if capital <= 0:
            return 0.0

        # Base risk: 1 % of capital
        base_risk_pct = 0.01

        # Extreme volatility: hard cap at 0.5 %
        if volatility_regime == "extreme":
            base_risk_pct = min(base_risk_pct, 0.005)

        # Volatility adjustment
        reference_vol = 0.5  # baseline annualised vol
        if realized_vol > 0:
            vol_factor = reference_vol / realized_vol
        else:
            vol_factor = 1.0
        adjusted_risk_pct = base_risk_pct * vol_factor

        # Win streak bonus: +10 % per win above 3 (max +30 %)
        if win_streak >= 3:
            bonus = min(0.30, (win_streak - 2) * 0.10)
            adjusted_risk_pct *= (1.0 + bonus)

        # Loss streak penalty
        if loss_streak >= 3:
            adjusted_risk_pct *= 0.50
        elif loss_streak >= 2:
            adjusted_risk_pct *= 0.75

        # Hard cap at 2 %
        adjusted_risk_pct = min(adjusted_risk_pct, 0.02)

        # Convert risk percentage to position size using ATR
        if atr > 0 and entry_price > 0:
            risk_amount = capital * adjusted_risk_pct
            size = risk_amount / atr
        else:
            size = capital * adjusted_risk_pct

        # Kelly override: cap if Kelly is more than 2× the adaptive size
        kelly = self.kelly_size(win_rate, avg_win, avg_loss, capital)
        if kelly > 0 and kelly > size * 2.0:
            kelly = size  # cap Kelly at adaptive size

        logger.debug(
            "Adaptive size: capital={} regime={} win_streak={} loss_streak={} "
            "risk_pct={:.4f} size={:.4f}",
            capital,
            volatility_regime,
            win_streak,
            loss_streak,
            adjusted_risk_pct,
            size,
        )
        return round(size, 8)

    def pair_specific_size(
        self,
        capital: float,
        symbol: str,
        atr: float = 0.0,
        entry_price: float = 0.0,
        win_streak: int = 0,
        loss_streak: int = 0,
        realized_vol: float = 0.5,
        win_rate: float = 0.5,
        avg_win: float = 0.01,
        avg_loss: float = 0.01,
    ) -> float:
        """Calculate a position size using the pair-specific profile settings.

        Applies the ``max_position_pct`` cap from the pair profile, then delegates
        to :meth:`adaptive_size` for streak / volatility adjustments.

        Args:
            capital: Total available capital (USDT).
            symbol: Trading pair symbol (e.g. ``"BTC/USDT"``).
            atr: Average True Range (used for ATR-based sizing).
            entry_price: Intended entry price.
            win_streak: Consecutive winning trades.
            loss_streak: Consecutive losing trades.
            realized_vol: Realised annualised volatility (decimal).
            win_rate: Historical win rate (0–1) for Kelly component.
            avg_win: Average winning trade return.
            avg_loss: Average losing trade return (positive value).

        Returns:
            Recommended position size in USDT, capped by the pair profile's
            ``max_position_pct``.
        """
        try:
            from config.pair_profiles import get_pair_profile  # avoid circular at module level
        except ImportError:
            logger.warning("pair_profiles not available; falling back to adaptive_size")
            return self.adaptive_size(
                capital=capital,
                atr=atr,
                entry_price=entry_price,
                win_streak=win_streak,
                loss_streak=loss_streak,
                realized_vol=realized_vol,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )

        profile = get_pair_profile(symbol)
        max_position_pct: float = profile.get("max_position_pct", 0.10)
        volatility_category: str = profile.get("volatility_category", "medium")

        # Map volatility category to regime for adaptive_size
        _VOL_MAP = {"low": "low", "medium": "normal", "high": "high"}
        regime = _VOL_MAP.get(volatility_category, "normal")

        base_size = self.adaptive_size(
            capital=capital,
            atr=atr,
            entry_price=entry_price,
            volatility_regime=regime,
            win_streak=win_streak,
            loss_streak=loss_streak,
            realized_vol=realized_vol,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        # Cap at the pair's maximum position percentage
        cap = capital * max_position_pct
        size = min(base_size, cap)

        logger.debug(
            "Pair-specific size: symbol={} max_pct={:.0%} base={:.4f} capped={:.4f}",
            symbol,
            max_position_pct,
            base_size,
            size,
        )
        return round(size, 8)

    @staticmethod
    def convert_to_base_units(usdt_size: float, price: float) -> float:
        """Convert a USDT position size to base currency units.

        All sizing methods in this class return values in USDT (quote currency).
        CCXT futures exchanges require order amounts in **base currency** units
        (e.g. BTC, ETH).  Use this utility to safely perform the conversion
        before placing any order.

        Args:
            usdt_size: Position size in USDT (quote currency).
            price: Current market price of the base currency in USDT.

        Returns:
            Equivalent position size in base currency units, or 0.0 when
            *price* is zero or negative (to avoid division by zero).
        """
        if price <= 0 or usdt_size <= 0:
            return 0.0
        return usdt_size / price

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
        max_kelly_fraction: float = 0.25,
    ) -> float:
        """Calculate a Kelly-Criterion position size capped at *max_kelly_fraction*.

        Uses the full Kelly formula capped at *max_kelly_fraction* of Kelly (default
        25 %) to be conservative.  A fractional Kelly reduces the risk of ruin while
        still capturing much of the theoretical Kelly growth rate.

        Args:
            win_rate: Historical win rate (0–1).
            avg_win: Average winning trade return (positive, as a decimal fraction).
            avg_loss: Average losing trade return (positive, as a decimal fraction).
            equity: Total available equity in USDT.
            max_kelly_fraction: Cap Kelly percentage at this fraction (default 0.25
                = 25 % of the full Kelly recommendation).

        Returns:
            Recommended position size in USDT.
        """
        if avg_loss <= 0 or equity <= 0 or win_rate <= 0:
            return 0.0
        try:
            loss_rate = 1.0 - win_rate
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
            kelly_pct = win_rate - (loss_rate / win_loss_ratio) if win_loss_ratio > 0 else 0.0
            # Apply fractional Kelly
            fractional_kelly = max(0.0, kelly_pct * max_kelly_fraction)
            # Hard cap: never exceed max_kelly_fraction of equity
            fractional_kelly = min(fractional_kelly, max_kelly_fraction)
            size = equity * fractional_kelly
            logger.debug(
                "calculate_kelly_size: win_rate={} kelly_pct={:.4f} "
                "fraction={} size={:.2f}",
                win_rate,
                kelly_pct,
                max_kelly_fraction,
                size,
            )
            return size
        except Exception as exc:
            logger.error("calculate_kelly_size failed: {}", exc)
            return 0.0

    def calculate_volatility_adjusted_size(
        self,
        equity: float,
        atr: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """Calculate a volatility-adjusted position size using ATR and stop-loss distance.

        Position size is derived from the maximum dollar risk per trade
        (``equity × risk_per_trade_pct``) divided by the distance between
        *entry_price* and *stop_loss*.  The ATR acts as a sanity guard: if the
        stop-loss distance is smaller than one ATR (tight stop), the size is
        reduced proportionally; if the stop-loss is wider than one ATR (loose
        stop), the size is increased up to the natural ATR-based limit.

        Args:
            equity: Total available equity in USDT.
            atr: Average True Range of the instrument in USDT.
            risk_per_trade_pct: Fraction of equity to risk per trade (e.g. 0.01
                for 1 %).
            entry_price: Intended entry price.
            stop_loss: Stop-loss price.

        Returns:
            Recommended position size in base currency units (same contract unit
            as *amount* on the exchange).
        """
        if equity <= 0 or entry_price <= 0 or atr <= 0:
            return 0.0
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return 0.0
        try:
            risk_amount = equity * risk_per_trade_pct
            # Size = risk $ / SL distance gives position in base units
            size = risk_amount / sl_distance
            # ATR adjustment: scale size so that 1 ATR move = risk_amount
            atr_size = risk_amount / atr
            # Use the geometric mean to blend both: avoid extremes
            blended = math.sqrt(size * atr_size) if size > 0 and atr_size > 0 else size
            logger.debug(
                "calculate_volatility_adjusted_size: equity={} atr={} "
                "sl_dist={:.4f} size={:.6f} atr_size={:.6f} blended={:.6f}",
                equity,
                atr,
                sl_distance,
                size,
                atr_size,
                blended,
            )
            return round(blended, 8)
        except Exception as exc:
            logger.error("calculate_volatility_adjusted_size failed: {}", exc)
            return 0.0

    def adjust_for_correlation(
        self,
        base_size: float,
        new_symbol: str,
        existing_positions: List[dict],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """Reduce *base_size* when the new position is correlated with open positions.

        If multiple positions are highly correlated (e.g. BTC long and ETH long),
        adding another correlated position increases portfolio risk beyond what
        individual position limits suggest.  This method reduces the new size in
        proportion to how correlated it is with the current book.

        Args:
            base_size: Initial position size in USDT before correlation adjustment.
            new_symbol: Symbol of the new position being sized (e.g. ``"ETH/USDT"``).
            existing_positions: List of dicts each with at least a ``"symbol"`` key.
            correlation_matrix: Optional nested dict of pairwise correlations,
                e.g. ``{"BTC/USDT": {"ETH/USDT": 0.85}}``.  When ``None`` or a
                pair is missing, a default correlation of 0.5 is assumed for
                non-identical assets.

        Returns:
            Adjusted position size in USDT (always ≤ *base_size*).
        """
        if not existing_positions or base_size <= 0:
            return base_size

        _DEFAULT_CORR = 0.5
        _corr: Dict[str, Dict[str, float]] = correlation_matrix or {}

        max_corr = 0.0
        for pos in existing_positions:
            sym = pos.get("symbol", "")
            if not sym or sym == new_symbol:
                continue
            # Look up correlation symmetrically
            corr = (
                _corr.get(new_symbol, {}).get(sym)
                or _corr.get(sym, {}).get(new_symbol)
                or _DEFAULT_CORR
            )
            max_corr = max(max_corr, float(corr))

        if max_corr <= 0:
            return base_size

        # Reduction factor: at 100 % correlation, size → 0; at 0 %, no change.
        reduction_factor = 1.0 - max_corr
        adjusted = base_size * reduction_factor
        logger.debug(
            "adjust_for_correlation: symbol={} max_corr={:.2f} "
            "base_size={:.4f} adjusted={:.4f}",
            new_symbol,
            max_corr,
            base_size,
            adjusted,
        )
        return round(max(adjusted, 0.0), 8)

    async def profit_maximizer_size(
        self,
        capital: float,
        symbol: str,
        strategy_name: str,
        atr: float = 0.0,
        entry_price: float = 0.0,
        win_streak: int = 0,
        loss_streak: int = 0,
        realized_vol: float = 0.5,
        win_rate: float = 0.5,
        avg_win: float = 0.03,
        avg_loss: float = 0.02,
        daily_profit: float = 0.0,
        profit_maximizer=None,
    ) -> float:
        """Calculate position size incorporating profit maximization multipliers.

        This is the PRIMARY sizing method that combines all profit maximization
        subsystems:
        - Base adaptive sizing
        - Compound growth multiplier
        - Momentum scaling multiplier
        - Time-of-day multiplier
        - Pair profitability multiplier
        - Drawdown recovery multiplier

        Args:
            capital: Available capital
            symbol: Trading pair
            strategy_name: Name of strategy
            atr: Average True Range
            entry_price: Entry price
            win_streak: Consecutive wins
            loss_streak: Consecutive losses
            realized_vol: Realized volatility
            win_rate: Historical win rate
            avg_win: Average win
            avg_loss: Average loss
            daily_profit: Profit for current day
            profit_maximizer: ProfitMaximizer instance

        Returns:
            Final position size in USDT
        """
        # Get base size using pair-specific adaptive sizing
        base_size = self.pair_specific_size(
            capital=capital,
            symbol=symbol,
            atr=atr,
            entry_price=entry_price,
            win_streak=win_streak,
            loss_streak=loss_streak,
            realized_vol=realized_vol,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        # If no profit maximizer, return base size
        if profit_maximizer is None:
            return base_size

        # Get combined multiplier from profit maximizer
        try:
            combined_multiplier = await profit_maximizer.get_combined_multiplier(
                strategy_name=strategy_name,
                symbol=symbol,
                daily_profit=daily_profit,
            )

            # Apply multiplier
            final_size = base_size * combined_multiplier

            # Hard cap: never exceed 2% of equity per trade (safety rail)
            max_size = capital * 0.02
            final_size = min(final_size, max_size)

            logger.debug(
                "profit_maximizer_size: base={:.4f} multiplier={:.3f} final={:.4f} (cap={:.4f})",
                base_size,
                combined_multiplier,
                final_size,
                max_size,
            )

            return round(final_size, 8)

        except Exception as exc:
            logger.error("profit_maximizer_size failed: {} — falling back to base size", exc)
            return base_size

    def calculate_size(self, method: SizingMethod | str, **kwargs: Any) -> float:
        """Dispatch to the appropriate sizing method.

        Args:
            method: One of the :class:`SizingMethod` values.
            **kwargs: Arguments forwarded to the selected method.

        Returns:
            Calculated position size.
        """
        method = SizingMethod(method)
        dispatch = {
            SizingMethod.KELLY: self.kelly_size,
            SizingMethod.FIXED_FRACTION: self.fixed_fraction_size,
            SizingMethod.FIXED_RATIO: self.fixed_ratio_size,
            SizingMethod.VOLATILITY_ADJUSTED: self.volatility_adjusted_size,
            SizingMethod.EQUAL_WEIGHT: self.equal_weight_size,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(f"Unknown sizing method: {method}")
        try:
            return fn(**kwargs)
        except TypeError as exc:
            logger.error("Wrong arguments for sizing method {}: {}", method, exc)
            return 0.0
