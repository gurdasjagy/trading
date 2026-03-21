"""Forex-specific risk management with lot sizing, pip-based SL/TP, and spread protection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger


@dataclass
class ForexTradeApproval:
    """Result returned by :meth:`ForexRiskManager.validate_forex_trade`."""

    approved: bool
    lot_size: float
    leverage: int
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pips: float
    take_profit_pips: float
    margin_required: float
    risk_amount: float       # USD amount at risk
    reward_amount: float     # USD potential reward
    risk_reward_ratio: float
    rejection_reason: Optional[str] = None


class ForexRiskManager:
    """Manages all forex-specific risk calculations.

    Key forex risk concepts:

    * Position sizing by lot (not by USDT amount).
    * Risk per trade as % of account equity.
    * Pip-based stop loss and take profit.
    * Spread cost consideration.
    * Margin requirements.
    * Maximum drawdown per session.
    * Correlation risk between gold and silver.
    """

    # ------------------------------------------------------------------
    # Forex pair specifications
    # ------------------------------------------------------------------

    PAIR_SPECS: Dict[str, Dict] = {
        "XAU/USD": {
            "pip_size": 0.01,
            "contract_size": 1,       # 1 oz per unit
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 500.0,
            "typical_spread_pips": 3.0,
            "max_acceptable_spread_pips": 8.0,
            "default_sl_pips": 200,   # $2.00 for gold
            "default_tp_pips": 400,   # $4.00 for gold
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,
            # Gold-specific institutional params
            "max_daily_trades": 8,
            "max_concurrent_positions": 2,
            "min_time_between_trades": 300,   # 5 minutes
            "weekend_close_hour": 20,          # Friday 20:00 UTC
            "max_position_hold_hours": 48,
            "funding_rate_threshold": -0.01,
            "correlation_with_dxy": -0.85,
            "key_levels": [2000, 2050, 2100, 2500, 2800, 3000, 3050, 3100],
            "session_preferences": {
                "london": {"weight": 1.2, "max_trades": 4},
                "new_york": {"weight": 1.1, "max_trades": 3},
                "london_ny_overlap": {"weight": 1.5, "max_trades": 3},
                "asian": {"weight": 0.7, "max_trades": 2},
                "sydney": {"weight": 0.5, "max_trades": 1},
            },
        },
        "XAG/USD": {
            "pip_size": 0.001,
            "contract_size": 5000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 5.0,
            "max_acceptable_spread_pips": 15.0,
            "default_sl_pips": 50,
            "default_tp_pips": 100,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,
        },
        # Major forex pairs (0.0 pip spread for Exness Raw Spread)
        "EURUSD": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.5,
            "max_acceptable_spread_pips": 2.0,
            "default_sl_pips": 20,
            "default_tp_pips": 40,
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 2.0,
        },
        "GBPUSD": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.8,
            "max_acceptable_spread_pips": 2.5,
            "default_sl_pips": 25,
            "default_tp_pips": 50,
            "atr_multiplier_sl": 1.2,
            "atr_multiplier_tp": 2.0,
        },
        "USDJPY": {
            "pip_size": 0.01,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.5,
            "max_acceptable_spread_pips": 2.0,
            "default_sl_pips": 20,
            "default_tp_pips": 40,
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 2.0,
        },
        "AUDUSD": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.6,
            "max_acceptable_spread_pips": 2.0,
            "default_sl_pips": 20,
            "default_tp_pips": 40,
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 2.0,
        },
        "USDCAD": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.7,
            "max_acceptable_spread_pips": 2.0,
            "default_sl_pips": 20,
            "default_tp_pips": 40,
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 2.0,
        },
        "USDCHF": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.8,
            "max_acceptable_spread_pips": 2.0,
            "default_sl_pips": 20,
            "default_tp_pips": 40,
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 2.0,
        },
        "GBPJPY": {
            "pip_size": 0.01,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 1.5,
            "max_acceptable_spread_pips": 4.0,
            "default_sl_pips": 30,
            "default_tp_pips": 60,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,
        },
        "EURJPY": {
            "pip_size": 0.01,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 1.2,
            "max_acceptable_spread_pips": 3.0,
            "default_sl_pips": 25,
            "default_tp_pips": 50,
            "atr_multiplier_sl": 1.2,
            "atr_multiplier_tp": 2.0,
        },
        "NZDUSD": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.8,
            "max_acceptable_spread_pips": 2.5,
            "default_sl_pips": 20,
            "default_tp_pips": 40,
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 2.0,
        },
        "EURGBP": {
            "pip_size": 0.0001,
            "contract_size": 100000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 0.9,
            "max_acceptable_spread_pips": 2.5,
            "default_sl_pips": 15,
            "default_tp_pips": 30,
            "atr_multiplier_sl": 0.8,
            "atr_multiplier_tp": 1.5,
        },
        # Exness-specific symbols (without slash)
        "XAUUSD": {
            "pip_size": 0.01,
            "contract_size": 100,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 1.5,
            "max_acceptable_spread_pips": 5.0,
            "default_sl_pips": 200,
            "default_tp_pips": 400,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,
        },
        "XAGUSD": {
            "pip_size": 0.001,
            "contract_size": 5000,
            "min_lot": 0.01,
            "lot_step": 0.01,
            "typical_spread_pips": 2.0,
            "max_acceptable_spread_pips": 8.0,
            "default_sl_pips": 50,
            "default_tp_pips": 100,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,
        },
    }

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._session_pnl: float = 0.0
        self._session_trades: int = 0
        self._max_session_drawdown: float = 0.0

        # Pull configurable limits from settings if available, otherwise use defaults
        forex_cfg = getattr(settings, "forex", None) if settings is not None else None
        self._daily_loss_limit_pct: float = (
            getattr(forex_cfg, "daily_loss_limit_pct", 3.0) if forex_cfg else 3.0
        )
        self._risk_per_trade_pct: float = (
            getattr(forex_cfg, "risk_per_trade_pct", 1.0) if forex_cfg else 1.0
        )
        self._max_open_forex_trades: int = (
            getattr(forex_cfg, "max_open_trades", 5) if forex_cfg else 5
        )
        self._open_trade_count: int = 0

        # --- New tracking fields (Upgrade 2) ---
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._in_recovery_mode: bool = False
        self._recovery_mode_level: int = 0   # 0=off 1=light 2=medium 3=hard 4=stopped
        self._peak_equity: float = 0.0
        self._max_drawdown_pct_seen: float = 0.0
        self._session_pnl_breakdown: Dict[str, float] = {
            "london": 0.0,
            "new_york": 0.0,
            "asian": 0.0,
            "sydney": 0.0,
        }
        # Per-session trade counts (for exposure tracking)
        self._session_trade_counts: Dict[str, int] = {
            "london": 0, "new_york": 0, "asian": 0, "sydney": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_forex_trade(
        self,
        symbol: str,
        direction: str,
        equity: float,
        current_price: float,
        spread_pips: float,
        atr: float = 0.0,
        leverage: int = 20,
    ) -> ForexTradeApproval:
        """Validate and size a forex trade.

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.
            direction: ``"long"`` or ``"short"``.
            equity: Account equity in USD.
            current_price: Current market price.
            spread_pips: Current bid-ask spread in pips.
            atr: Average True Range (optional; used for ATR-based SL/TP).
            leverage: Leverage multiplier.

        Returns:
            :class:`ForexTradeApproval` with lot size, SL/TP prices, and
            margin info.  Check ``approval.approved`` before placing the order;
            if ``False``, ``rejection_reason`` explains why.
        """
        spec = self.PAIR_SPECS.get(symbol)
        if not spec:
            return ForexTradeApproval(
                approved=False, lot_size=0, leverage=0,
                stop_loss_price=0, take_profit_price=0,
                stop_loss_pips=0, take_profit_pips=0,
                margin_required=0, risk_amount=0, reward_amount=0,
                risk_reward_ratio=0,
                rejection_reason=f"Unknown forex pair: {symbol}",
            )

        # Check spread
        if spread_pips > spec["max_acceptable_spread_pips"]:
            return ForexTradeApproval(
                approved=False, lot_size=0, leverage=0,
                stop_loss_price=0, take_profit_price=0,
                stop_loss_pips=0, take_profit_pips=0,
                margin_required=0, risk_amount=0, reward_amount=0,
                risk_reward_ratio=0,
                rejection_reason=(
                    f"Spread too wide: {spread_pips:.1f} pips > "
                    f"max {spec['max_acceptable_spread_pips']}"
                ),
            )

        # Check max open trades
        if self._open_trade_count >= self._max_open_forex_trades:
            return ForexTradeApproval(
                approved=False, lot_size=0, leverage=0,
                stop_loss_price=0, take_profit_price=0,
                stop_loss_pips=0, take_profit_pips=0,
                margin_required=0, risk_amount=0, reward_amount=0,
                risk_reward_ratio=0,
                rejection_reason=f"Max open trades reached: {self._open_trade_count}",
            )

        # Check daily loss limit
        daily_loss_pct = (
            abs(self._session_pnl / equity * 100)
            if equity > 0 and self._session_pnl < 0
            else 0.0
        )
        if daily_loss_pct >= self._daily_loss_limit_pct:
            return ForexTradeApproval(
                approved=False, lot_size=0, leverage=0,
                stop_loss_price=0, take_profit_price=0,
                stop_loss_pips=0, take_profit_pips=0,
                margin_required=0, risk_amount=0, reward_amount=0,
                risk_reward_ratio=0,
                rejection_reason=f"Daily loss limit reached: {daily_loss_pct:.1f}%",
            )

        # Calculate SL/TP in pips (ATR-based if available, else defaults)
        if atr > 0:
            sl_pips = (atr * spec["atr_multiplier_sl"]) / spec["pip_size"]
            tp_pips = (atr * spec["atr_multiplier_tp"]) / spec["pip_size"]
        else:
            sl_pips = float(spec["default_sl_pips"])
            tp_pips = float(spec["default_tp_pips"])

        # Account for spread in SL (spread eats into profit)
        effective_sl_pips = sl_pips + spread_pips

        # Calculate lot size based on risk per trade
        risk_amount = equity * (self._risk_per_trade_pct / 100)
        pip_value_per_lot = spec["pip_size"] * spec["contract_size"]
        if effective_sl_pips > 0 and pip_value_per_lot > 0:
            lot_size = risk_amount / (effective_sl_pips * pip_value_per_lot)
        else:
            lot_size = spec["min_lot"]

        # Round to lot step
        lot_step = spec["lot_step"]
        lot_size = max(spec["min_lot"], round(lot_size / lot_step) * lot_step)

        # Calculate margin required
        margin_required = (lot_size * spec["contract_size"] * current_price) / leverage

        # Check if we have enough margin (don't use more than 50% of equity)
        if equity > 0 and margin_required > equity * 0.5:
            lot_size = (equity * 0.5 * leverage) / (spec["contract_size"] * current_price)
            lot_size = max(spec["min_lot"], round(lot_size / lot_step) * lot_step)
            margin_required = (lot_size * spec["contract_size"] * current_price) / leverage

        # Calculate SL/TP prices
        sl_price_distance = sl_pips * spec["pip_size"]
        tp_price_distance = tp_pips * spec["pip_size"]

        if direction == "long":
            stop_loss_price = current_price - sl_price_distance
            take_profit_price = current_price + tp_price_distance
        else:
            stop_loss_price = current_price + sl_price_distance
            take_profit_price = current_price - tp_price_distance

        # Calculate actual risk and reward amounts
        actual_risk = lot_size * effective_sl_pips * pip_value_per_lot
        actual_reward = lot_size * tp_pips * pip_value_per_lot
        rr_ratio = actual_reward / actual_risk if actual_risk > 0 else 0.0

        # Minimum R:R of 1.5
        if rr_ratio < 1.5:
            return ForexTradeApproval(
                approved=False, lot_size=0, leverage=0,
                stop_loss_price=0, take_profit_price=0,
                stop_loss_pips=0, take_profit_pips=0,
                margin_required=0, risk_amount=0, reward_amount=0,
                risk_reward_ratio=round(rr_ratio, 2),
                rejection_reason=f"Risk/reward too low: {rr_ratio:.2f} < 1.5",
            )

        logger.debug(
            "Forex trade approved: {} {} lot={} sl={} tp={} R:R={}",
            symbol, direction, round(lot_size, 2),
            round(stop_loss_price, 2), round(take_profit_price, 2), round(rr_ratio, 2),
        )

        return ForexTradeApproval(
            approved=True,
            lot_size=round(lot_size, 2),
            leverage=leverage,
            stop_loss_price=round(stop_loss_price, 2),
            take_profit_price=round(take_profit_price, 2),
            stop_loss_pips=round(sl_pips, 1),
            take_profit_pips=round(tp_pips, 1),
            margin_required=round(margin_required, 2),
            risk_amount=round(actual_risk, 2),
            reward_amount=round(actual_reward, 2),
            risk_reward_ratio=round(rr_ratio, 2),
        )

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result for session tracking.

        Args:
            pnl: Profit/loss in USD (positive = profit, negative = loss).
        """
        self._session_pnl += pnl
        self._session_trades += 1
        if self._session_pnl < self._max_session_drawdown:
            self._max_session_drawdown = self._session_pnl

    def increment_open_trades(self) -> None:
        """Increment the open trade counter (call when a forex trade opens)."""
        self._open_trade_count += 1

    def decrement_open_trades(self) -> None:
        """Decrement the open trade counter (call when a forex trade closes)."""
        self._open_trade_count = max(0, self._open_trade_count - 1)

    def reset_session(self) -> None:
        """Reset session stats (call at the start of each trading day)."""
        self._session_pnl = 0.0
        self._session_trades = 0
        self._max_session_drawdown = 0.0

    @property
    def session_pnl(self) -> float:
        """Cumulative PnL for the current session."""
        return self._session_pnl

    @property
    def session_trades(self) -> int:
        """Number of completed trades in the current session."""
        return self._session_trades

    # ------------------------------------------------------------------
    # Session-based risk adjustment
    # ------------------------------------------------------------------

    def get_session_risk_multiplier(self) -> float:
        """Return risk multiplier based on current trading session.

        Returns tighter stops during high-volatility sessions (London/NY overlap)
        and wider stops during low-volatility Asian session.
        """
        now = datetime.now(tz=timezone.utc)
        hour = now.hour

        # London/NY overlap (13:00-16:00 UTC) — highest volatility
        if 13 <= hour < 16:
            return 0.7  # Tighter stops (30% reduction)

        # London session (08:00-16:00 UTC) — high volatility
        if 8 <= hour < 16:
            return 0.85  # Slightly tighter stops (15% reduction)

        # New York session (13:00-21:00 UTC) — high volatility
        if 13 <= hour < 21:
            return 0.85

        # Asian session (00:00-09:00 UTC) — low volatility
        if 0 <= hour < 9:
            return 1.2  # Wider stops (20% increase)

        # Sydney session (22:00-07:00 UTC) — low volatility
        if hour >= 22 or hour < 7:
            return 1.2

        return 1.0  # Normal

    def adjust_risk_for_news(
        self, base_risk_pct: float, minutes_until_news: int, news_impact: str
    ) -> float:
        """Adjust risk percentage based on proximity to high-impact news events.

        Args:
            base_risk_pct: Base risk per trade (%).
            minutes_until_news: Minutes until next high-impact event.
            news_impact: Event impact level ("low", "medium", "high").

        Returns:
            Adjusted risk percentage (reduced near high-impact events).
        """
        if news_impact == "high" and minutes_until_news < 30:
            return base_risk_pct * 0.5  # 50% reduction
        if news_impact == "medium" and minutes_until_news < 15:
            return base_risk_pct * 0.7  # 30% reduction
        return base_risk_pct

    def check_correlation_risk(
        self, symbol: str, direction: str, open_positions: list
    ) -> tuple[bool, str]:
        """Check correlation risk with existing positions.

        Prevents opening inverse correlated positions simultaneously
        (e.g., long EURUSD + long USDCHF).

        Args:
            symbol: Pair to trade.
            direction: "long" or "short".
            open_positions: List of currently open positions.

        Returns:
            (allowed, reason): Tuple of bool and rejection reason if not allowed.
        """
        # Define inverse correlation pairs
        inverse_pairs = {
            ("EURUSD", "USDCHF"): True,
            ("GBPUSD", "USDCHF"): True,
            ("AUDUSD", "USDCHF"): True,
            ("NZDUSD", "USDCHF"): True,
        }

        # Normalize symbols (remove slashes)
        norm_symbol = symbol.replace("/", "")

        for pos in open_positions:
            pos_symbol = pos.get("symbol", "").replace("/", "")
            pos_side = pos.get("side", "")

            # Check if inverse correlated
            pair_key = tuple(sorted([norm_symbol, pos_symbol]))
            if pair_key in inverse_pairs:
                # Same direction on inverse pairs = hedged OK
                if pos_side == direction:
                    continue
                else:
                    return (
                        False,
                        f"Correlation risk: {symbol} {direction} conflicts with {pos_symbol} {pos_side}",
                    )

        return (True, "")

    def calculate_max_margin_usage(self, equity: float) -> float:
        """Calculate maximum allowed margin usage (30% of equity).

        Args:
            equity: Account equity in USD.

        Returns:
            Maximum margin allowed in USD.
        """
        return equity * 0.3

    def adjust_lot_size_for_drawdown(
        self, base_lot_size: float, consecutive_losses: int
    ) -> float:
        """Reduce lot size after consecutive losses, increase after wins.

        Args:
            base_lot_size: Calculated lot size from risk per trade.
            consecutive_losses: Number of consecutive losing trades (negative for wins).

        Returns:
            Adjusted lot size.
        """
        if consecutive_losses >= 3:
            # After 3 losses, reduce lot size by 30%
            return base_lot_size * 0.7
        elif consecutive_losses >= 5:
            # After 5 losses, reduce lot size by 50%
            return base_lot_size * 0.5
        elif consecutive_losses <= -3:
            # After 3 wins, increase lot size by 20%
            return base_lot_size * 1.2
        return base_lot_size

    def get_session_name(self) -> str:
        """Return the name of the current trading session."""
        now = datetime.now(tz=timezone.utc)
        hour = now.hour

        sessions = []
        # London session (08:00-16:00 UTC)
        if 8 <= hour < 16:
            sessions.append("London")
        # New York session (13:00-21:00 UTC)
        if 13 <= hour < 21:
            sessions.append("New York")
        # Tokyo session (00:00-09:00 UTC)
        if 0 <= hour < 9:
            sessions.append("Tokyo")
        # Sydney session (22:00-07:00 UTC)
        if hour >= 22 or hour < 7:
            sessions.append("Sydney")

        return ", ".join(sessions) if sessions else "Closed"

    def validate_margin_level(
        self, equity: float, used_margin: float
    ) -> tuple[bool, str]:
        """Validate margin level (must be above 100% for new trades).

        Args:
            equity: Account equity.
            used_margin: Currently used margin.

        Returns:
            (valid, reason): Tuple of bool and reason if invalid.
        """
        if used_margin <= 0:
            return (True, "")

        margin_level = (equity / used_margin) * 100

        if margin_level < 100:
            return (
                False,
                f"Margin level too low: {margin_level:.1f}% (minimum 100%)",
            )

        return (True, "")

    def calculate_dynamic_lot_size(
        self,
        symbol: str,
        equity: float,
        stop_loss_pips: float,
        leverage: int = 20,
        atr_factor: float = 1.0,
    ) -> float:
        """Calculate lot size dynamically using risk % of equity and ATR scaling.

        The base formula is:
          risk_amount = equity * risk_per_trade_pct / 100
          pip_value   = pip_size * contract_size * lot_size
          lot_size    = risk_amount / (stop_loss_pips * pip_value_per_lot)

        The result is further scaled by:
        * Drawdown recovery mode (reduces size when losing)
        * ATR factor (optional caller-supplied volatility scalar)
        * Session risk multiplier

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.
            equity: Current account equity in USD.
            stop_loss_pips: Intended stop loss distance in pips.
            leverage: Leverage multiplier.
            atr_factor: Volatility scalar (1.0 = neutral; <1 = widen SL; >1 = tight SL).

        Returns:
            Lot size rounded to pair's ``lot_step``, clamped to [min_lot, max_lot].
        """
        spec = self.PAIR_SPECS.get(symbol) or self.PAIR_SPECS.get(symbol.replace("/", ""))
        if not spec or stop_loss_pips <= 0 or equity <= 0:
            return spec["min_lot"] if spec else 0.01

        pip_size = spec["pip_size"]
        contract_size = spec["contract_size"]
        min_lot = spec.get("min_lot", 0.01)
        lot_step = spec.get("lot_step", 0.01)
        max_lot = spec.get("max_lot", 500.0)

        # Risk amount in USD
        risk_amount = equity * (self._risk_per_trade_pct / 100.0)

        # Pip value per 1 lot (USD per pip per standard lot)
        pip_value_per_lot = pip_size * contract_size

        # Base lot size from risk formula
        if pip_value_per_lot <= 0:
            return min_lot
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Apply drawdown/recovery scaling
        recovery_scale = self._get_recovery_lot_scale()
        lot_size *= recovery_scale

        # Apply ATR factor (wider ATR → smaller lot)
        if atr_factor > 0:
            lot_size /= atr_factor

        # Apply session risk multiplier.
        # session_mult: 1.5 = high-risk session (reduce lots), 0.7 = low-risk (increase lots).
        # We invert so that high-risk sessions produce smaller lots.
        session_mult = self.get_session_risk_multiplier()
        if session_mult > 0:
            lot_size *= (1.0 / session_mult)

        # Apply consecutive-loss reduction
        if self._consecutive_losses >= 5:
            lot_size *= 0.5
        elif self._consecutive_losses >= 3:
            lot_size *= 0.7

        # Apply consecutive-win increase (capped at 50% increase)
        if self._consecutive_wins >= 3:
            lot_size *= min(1.5, 1.0 + self._consecutive_wins * 0.1)

        # Round to lot_step and clamp
        if lot_step > 0:
            lot_size = math.floor(lot_size / lot_step) * lot_step
        lot_size = max(min_lot, min(max_lot, lot_size))
        return round(lot_size, 2)

    def enter_recovery_mode(self, equity: float) -> None:
        """Update peak equity and check if drawdown recovery mode should be entered.

        Recovery levels:
          0 — normal trading
          1 — light drawdown   (>= 5%)  → reduce lot by 20%
          2 — medium drawdown  (>= 10%) → reduce lot by 40%
          3 — heavy drawdown   (>= 15%) → reduce lot by 60%
          4 — trading stopped  (>= 20%) → block new trades

        Call this after every trade result or equity update.
        """
        if equity > self._peak_equity:
            self._peak_equity = equity
            # Exiting drawdown — reduce recovery level gradually
            if self._in_recovery_mode and self._recovery_mode_level > 0:
                self._recovery_mode_level = max(0, self._recovery_mode_level - 1)
                if self._recovery_mode_level == 0:
                    self._in_recovery_mode = False
                    logger.info("ForexRiskManager: exited recovery mode (new equity peak {})", equity)
            return

        if self._peak_equity <= 0:
            self._peak_equity = equity
            return

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
        self._max_drawdown_pct_seen = max(self._max_drawdown_pct_seen, drawdown_pct)

        prev_level = self._recovery_mode_level
        # Levels: 1=light(2%), 2=medium(3%), 3=heavy(5%), 4=stopped(7%) — matches docstring
        if drawdown_pct >= 7.0:
            self._recovery_mode_level = 4
        elif drawdown_pct >= 5.0:
            self._recovery_mode_level = 3
        elif drawdown_pct >= 3.0:
            self._recovery_mode_level = 2
        elif drawdown_pct >= 2.0:
            self._recovery_mode_level = 1
        else:
            self._recovery_mode_level = 0

        if self._recovery_mode_level > 0:
            self._in_recovery_mode = True
            if self._recovery_mode_level != prev_level:
                logger.warning(
                    "ForexRiskManager: recovery mode level {} (drawdown={:.1f}%)",
                    self._recovery_mode_level,
                    drawdown_pct,
                )
        else:
            self._in_recovery_mode = False

    def is_trading_stopped_by_drawdown(self) -> bool:
        """Return True if drawdown is severe enough to halt all new trades."""
        return self._recovery_mode_level >= 4

    def check_session_exposure(self, session_name: str, symbol: str) -> tuple:
        """Check whether opening a new position is allowed given current session exposure.

        Args:
            session_name: Current session (``london``, ``new_york``, ``asian``, ``sydney``).
            symbol: Symbol to trade (used for gold-specific session limits).

        Returns:
            ``(allowed, reason)`` tuple.
        """
        # Gold-specific session limits
        spec = self.PAIR_SPECS.get(symbol) or self.PAIR_SPECS.get(symbol.replace("/", ""))
        if spec:
            session_prefs = spec.get("session_preferences", {})
            session_key = _normalise_session(session_name)
            pref = session_prefs.get(session_key)
            if pref:
                max_trades = pref.get("max_trades", 99)
                current = self._session_trade_counts.get(session_key, 0)
                if current >= max_trades:
                    return (
                        False,
                        f"Session limit reached: {symbol} allows {max_trades} trades in {session_name} session (current={current})",
                    )

        # General max concurrent positions check
        if self._open_trade_count >= self._max_open_forex_trades:
            return (
                False,
                f"Max open trades reached ({self._open_trade_count}/{self._max_open_forex_trades})",
            )

        return (True, "")

    def record_trade_result_extended(
        self,
        pnl: float,
        session_name: str = "",
        *,
        equity: float = 0.0,
    ) -> None:
        """Extended trade result recording that tracks consecutive wins/losses and session PnL.

        Call this instead of (or in addition to) :meth:`record_trade_result`.
        """
        self.record_trade_result(pnl)  # Update base session PnL / trade count

        won = pnl > 0

        # Consecutive streak tracking
        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        # Session PnL breakdown
        session_key = _normalise_session(session_name)
        if session_key in self._session_pnl_breakdown:
            self._session_pnl_breakdown[session_key] += pnl

        # Update recovery mode if equity provided
        if equity > 0:
            self.enter_recovery_mode(equity)

    def record_session_trade(self, session_name: str) -> None:
        """Increment the per-session trade count when a new trade is opened."""
        key = _normalise_session(session_name)
        if key in self._session_trade_counts:
            self._session_trade_counts[key] += 1

    def reset_session_counts(self) -> None:
        """Reset per-session trade counts at the start of each day."""
        for key in self._session_trade_counts:
            self._session_trade_counts[key] = 0
        self.reset_session()

    def _get_recovery_lot_scale(self) -> float:
        """Return lot size scalar based on recovery mode level."""
        scales = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.0}
        return scales.get(self._recovery_mode_level, 1.0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _normalise_session(raw: str) -> str:
    """Map various session name spellings to canonical lowercase keys."""
    raw_lower = (raw or "").lower().strip()
    if "london" in raw_lower and ("new" in raw_lower or "ny" in raw_lower):
        return "london_ny_overlap"
    if "london" in raw_lower:
        return "london"
    if "new" in raw_lower or "ny" in raw_lower or "york" in raw_lower:
        return "new_york"
    if "asian" in raw_lower or "tokyo" in raw_lower or "japan" in raw_lower:
        return "asian"
    if "sydney" in raw_lower or "australia" in raw_lower:
        return "sydney"
    return raw_lower or "unknown"


# ---------------------------------------------------------------------------
# ForexTakeProfitManager
# ---------------------------------------------------------------------------


class ForexTakeProfitManager:
    """Manage multi-target take-profit logic for forex trades.

    Implements a 3-tier TP ladder:
      * TP1 — 25% close at 1×R (risk-to-reward = 1)
      * TP2 — 50% close at 2×R
      * TP3 — 25% close at 3×R (full close)

    Also manages:
    * Break-even SL move after TP1 hit
    * Partial close tracking so the same TP is not triggered twice
    """

    TP_RATIOS = [
        {"level": 1, "close_pct": 25, "rr_multiple": 1.0},
        {"level": 2, "close_pct": 50, "rr_multiple": 2.0},
        {"level": 3, "close_pct": 25, "rr_multiple": 3.0},
    ]

    def calculate_tp_prices(
        self,
        entry: float,
        side: str,
        sl_pips: float,
        pip_size: float,
    ) -> list:
        """Return [tp1, tp2, tp3] prices for the given trade.

        Args:
            entry: Entry price.
            side: ``"long"`` or ``"short"``.
            sl_pips: Stop-loss distance in pips (positive value).
            pip_size: Pip size for the pair.

        Returns:
            List of three TP prices: ``[tp1, tp2, tp3]``.
        """
        direction = 1 if side == "long" else -1
        return [
            round(entry + direction * tp["rr_multiple"] * sl_pips * pip_size, 5)
            for tp in self.TP_RATIOS
        ]

    def check_tp_hit(
        self,
        current_price: float,
        entry: float,
        side: str,
        tp_prices: list,
        tp_hits: list,
    ) -> Optional[int]:
        """Check which TP level was hit (if any).

        Args:
            current_price: Live price.
            entry: Entry price.
            side: ``"long"`` or ``"short"``.
            tp_prices: List of TP prices [tp1, tp2, tp3].
            tp_hits: Booleans indicating already-hit TPs.

        Returns:
            TP level (1/2/3) or None.
        """
        for i, (tp_price, already_hit) in enumerate(zip(tp_prices, tp_hits)):
            if already_hit:
                continue
            if side == "long" and current_price >= tp_price:
                return i + 1
            if side == "short" and current_price <= tp_price:
                return i + 1
        return None

    def get_close_pct(self, tp_level: int) -> float:
        """Return the percentage of position to close at a given TP level."""
        for tp in self.TP_RATIOS:
            if tp["level"] == tp_level:
                return tp["close_pct"]
        return 100.0

    def calculate_break_even_sl(
        self, entry: float, side: str, spread_pips: float, pip_size: float
    ) -> float:
        """Calculate break-even SL price (entry + spread buffer).

        After TP1 is hit, move SL to break-even (entry + spread cost).
        """
        buffer_pips = max(spread_pips * 1.5, 2.0)  # At least 2 pip buffer
        direction = 1 if side == "long" else -1
        return round(entry + direction * buffer_pips * pip_size, 5)


# ---------------------------------------------------------------------------
# GoldTrailingStop
# ---------------------------------------------------------------------------


class GoldTrailingStop:
    """Trailing stop implementation optimised for XAU/USD (gold).

    Gold has high volatility so the trailing stop uses ATR-based distance
    rather than a fixed pip offset.  After TP1 is hit the trailing stop
    activates at 1×ATR behind the highest-water-mark price.
    """

    def __init__(
        self,
        entry: float,
        side: str,
        atr: float,
        activation_price: float,
        atr_multiplier: float = 1.0,
    ) -> None:
        """
        Args:
            entry: Entry price.
            side: ``"long"`` or ``"short"``.
            atr: ATR value in price units.
            activation_price: Price at which trailing stop activates (usually TP1).
            atr_multiplier: Trail distance = atr * atr_multiplier.
        """
        self._entry = entry
        self._side = side
        self._atr = atr
        self._atr_multiplier = atr_multiplier
        self._activation_price = activation_price
        self._active = False
        self._hwm: float = entry  # Highest-water-mark (or lowest for short)
        self._trail_sl: float = entry  # Current trailing SL price

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stop_price(self) -> float:
        return self._trail_sl

    def update(self, current_price: float) -> float:
        """Update the trailing stop with the latest price.

        Args:
            current_price: Latest market price.

        Returns:
            Updated trailing SL price.
        """
        trail_dist = self._atr * self._atr_multiplier

        if not self._active:
            # Check activation
            if self._side == "long" and current_price >= self._activation_price:
                self._active = True
                self._hwm = current_price
                self._trail_sl = current_price - trail_dist
                logger.debug(
                    "GoldTrailingStop: activated for LONG at {} SL={}",
                    current_price,
                    self._trail_sl,
                )
            elif self._side == "short" and current_price <= self._activation_price:
                self._active = True
                self._hwm = current_price
                self._trail_sl = current_price + trail_dist
                logger.debug(
                    "GoldTrailingStop: activated for SHORT at {} SL={}",
                    current_price,
                    self._trail_sl,
                )
            return self._trail_sl

        # Update HWM and trail SL
        if self._side == "long":
            if current_price > self._hwm:
                self._hwm = current_price
                self._trail_sl = self._hwm - trail_dist
        else:
            if current_price < self._hwm:
                self._hwm = current_price
                self._trail_sl = self._hwm + trail_dist

        return self._trail_sl

    def is_stopped_out(self, current_price: float) -> bool:
        """Return True if the trailing stop has been triggered."""
        if not self._active:
            return False
        if self._side == "long":
            return current_price <= self._trail_sl
        return current_price >= self._trail_sl


# ---------------------------------------------------------------------------
# ForexProfitCompounder
# ---------------------------------------------------------------------------


class ForexProfitCompounder:
    """Grow lot sizes proportionally as the account equity increases.

    The compounder adjusts the base lot size each day based on the ratio
    of current equity to starting equity.  This allows the position size
    to scale naturally with account growth without exceeding risk limits.
    """

    def __init__(
        self,
        base_lot: float = 0.01,
        compounding_threshold_pct: float = 5.0,
        max_multiplier: float = 5.0,
    ) -> None:
        """
        Args:
            base_lot: Starting (minimum) lot size.
            compounding_threshold_pct: Minimum equity growth (%) before compounding.
            max_multiplier: Hard cap on the lot multiplier.
        """
        self._original_base_lot = base_lot
        self._base_lot = base_lot
        self._threshold_pct = compounding_threshold_pct
        self._max_multiplier = max_multiplier
        self._base_lot_multiplier: float = 1.0
        self._compound_growth: float = 0.0
        self._starting_equity: float = 0.0

    def initialise(self, starting_equity: float) -> None:
        """Set the reference equity for compounding calculations."""
        self._starting_equity = starting_equity

    def update(self, current_equity: float) -> float:
        """Update the compounding multiplier based on equity growth.

        Args:
            current_equity: Current account equity.

        Returns:
            Adjusted base lot size.
        """
        if self._starting_equity <= 0 or current_equity <= 0:
            return self._base_lot

        growth_pct = (current_equity - self._starting_equity) / self._starting_equity * 100.0
        self._compound_growth = max(0.0, growth_pct)

        if growth_pct >= self._threshold_pct:
            # Compound proportionally to equity growth, capped by max_multiplier
            raw_multiplier = current_equity / self._starting_equity
            self._base_lot_multiplier = min(raw_multiplier, self._max_multiplier)
        else:
            self._base_lot_multiplier = 1.0

        self._base_lot = round(self._original_base_lot * self._base_lot_multiplier, 2)
        return self._base_lot

    @property
    def base_lot(self) -> float:
        return max(self._original_base_lot, self._base_lot)

    @property
    def multiplier(self) -> float:
        return self._base_lot_multiplier


# ---------------------------------------------------------------------------
# ForexMarginMonitor
# ---------------------------------------------------------------------------


class ForexMarginMonitor:
    """Monitor margin level in real time and trigger protective actions.

    Margin levels:
      * >= 200%  — healthy
      * 150–200% — warning: reduce new trade sizes
      * 100–150% — alert: no new trades
      * < 100%   — critical: close least-profitable position
    """

    LEVEL_HEALTHY = "healthy"
    LEVEL_WARNING = "warning"
    LEVEL_ALERT = "alert"
    LEVEL_CRITICAL = "critical"

    def __init__(
        self,
        warning_pct: float = 150.0,
        alert_pct: float = 100.0,
        critical_pct: float = 80.0,
    ) -> None:
        self._warning_pct = warning_pct
        self._alert_pct = alert_pct
        self._critical_pct = critical_pct

    def get_margin_level(self, equity: float, used_margin: float) -> str:
        """Return margin level string for the given equity/margin values."""
        if used_margin <= 0:
            return self.LEVEL_HEALTHY
        level_pct = (equity / used_margin) * 100.0
        if level_pct >= self._warning_pct:
            return self.LEVEL_HEALTHY
        if level_pct >= self._alert_pct:
            return self.LEVEL_WARNING
        if level_pct >= self._critical_pct:
            return self.LEVEL_ALERT
        return self.LEVEL_CRITICAL

    def can_open_new_trade(self, equity: float, used_margin: float) -> tuple:
        """Return (allowed, reason) for opening a new trade given current margin."""
        level = self.get_margin_level(equity, used_margin)
        if level == self.LEVEL_HEALTHY:
            return (True, "")
        if level == self.LEVEL_WARNING:
            return (True, f"Margin warning: level {level} — consider reducing size")
        if level == self.LEVEL_ALERT:
            return (False, f"Margin alert: level < {self._alert_pct}% — no new trades")
        return (False, f"Margin critical: level < {self._critical_pct}% — close positions")

    def get_size_multiplier(self, equity: float, used_margin: float) -> float:
        """Return a position-size scalar (0–1) based on margin health."""
        if used_margin <= 0:
            return 1.0
        level_pct = (equity / used_margin) * 100.0
        if level_pct >= self._warning_pct:
            return 1.0
        if level_pct >= self._alert_pct:
            # Linear reduction from 1.0 at warning to 0.3 at alert
            range_pct = self._warning_pct - self._alert_pct
            pos = level_pct - self._alert_pct
            return max(0.3, pos / range_pct) if range_pct > 0 else 0.3
        return 0.0  # Stop trading


# ---------------------------------------------------------------------------
# ForexNewsFilter
# ---------------------------------------------------------------------------


class ForexNewsFilter:
    """Block or size-down trades around high-impact news events.

    Integrates with the economic calendar to detect upcoming FOMC, NFP,
    CPI, and other tier-1 events that cause large gold price spikes.
    """

    # High-impact gold-relevant event keywords
    GOLD_HIGH_IMPACT_EVENTS = {
        "FOMC", "Federal Reserve", "Fed Rate", "CPI", "NFP",
        "Non-Farm Payroll", "Unemployment", "GDP", "PCE",
        "Retail Sales", "ISM", "JOLTS",
    }

    def __init__(
        self,
        block_minutes_before: int = 30,
        block_minutes_after: int = 15,
        size_reduction_minutes: int = 60,
    ) -> None:
        """
        Args:
            block_minutes_before: Minutes before event to block new trades.
            block_minutes_after: Minutes after event to block new trades.
            size_reduction_minutes: Minutes before/after where size is reduced 50%.
        """
        self._block_before = block_minutes_before
        self._block_after = block_minutes_after
        self._size_reduction_minutes = size_reduction_minutes
        self._upcoming_events: list = []

    def update_events(self, events: list) -> None:
        """Update the list of upcoming economic events.

        Args:
            events: List of dicts with at least ``time`` (datetime) and
                    ``title`` (str), optionally ``impact`` (str).
        """
        self._upcoming_events = [
            e for e in events
            if any(kw.lower() in e.get("title", "").lower() for kw in self.GOLD_HIGH_IMPACT_EVENTS)
            or e.get("impact", "").lower() == "high"
        ]

    def check_trade_allowed(
        self, symbol: str, current_time: Optional[datetime] = None
    ) -> tuple:
        """Return ``(allowed, reason, size_multiplier)`` for the given symbol.

        Args:
            symbol: Forex symbol to check (used to filter FX-relevant events).
            current_time: Override current UTC time (default: ``datetime.now``).

        Returns:
            ``(allowed: bool, reason: str, size_multiplier: float)`` tuple.
        """
        now = current_time or datetime.now(tz=timezone.utc)

        for event in self._upcoming_events:
            event_time = event.get("time")
            if event_time is None:
                continue
            if not isinstance(event_time, datetime):
                try:
                    event_time = datetime.fromisoformat(str(event_time))
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

            delta = (event_time - now).total_seconds() / 60.0  # minutes

            if -self._block_after <= delta <= self._block_before:
                return (
                    False,
                    f"High-impact news: '{event.get('title', 'Event')}' at {event_time.strftime('%H:%M')} UTC",
                    0.0,
                )

            if -self._size_reduction_minutes <= delta <= self._size_reduction_minutes:
                return (
                    True,
                    f"Near news event: '{event.get('title', 'Event')}' — size reduced 50%",
                    0.5,
                )

        return (True, "", 1.0)

