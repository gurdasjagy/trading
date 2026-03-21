"""Take-profit management with multiple TP levels and trailing take-profits."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from exchange.position_manager import PositionTracker


class TakeProfitEngine:
    """Manages take-profit targets including multiple TP levels."""

    # TP level proportions: TP1 30%, TP2 30%, TP3 40%
    DEFAULT_TP_PROPORTIONS = [0.30, 0.30, 0.40]
    # R:R multipliers relative to stop distance: TP1 at 1.5R, TP2 at 2R, TP3 at 3R
    # TP1 must be at least 1.5R to satisfy the minimum risk_reward_min requirement.
    DEFAULT_RR_MULTIPLIERS = [1.5, 2.0, 3.0]

    def calculate_tp_levels(
        self,
        entry: float,
        direction: str,
        atr: float,
        risk_reward: float = 2.0,
        stop_distance: float | None = None,
    ) -> List[dict]:
        """Calculate multiple take-profit price levels.

        TP levels are calculated relative to the actual stop distance when
        *stop_distance* is provided, otherwise ATR is used as the base distance.
        TP1 is placed at 1.5× the stop distance, which meets the minimum
        ``risk_reward_min`` of 1.5 required by the risk manager.

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range used when *stop_distance* is not given.
            risk_reward: Retained for backward compatibility; not used in the
                distance calculation.  Callers should pass *stop_distance*
                instead.
            stop_distance: Actual distance from entry to stop-loss.  When
                provided, TP levels are set to ``rr_mult × stop_distance``.

        Returns:
            List of dicts with keys ``price``, ``proportion``, and ``rr_ratio``.
        """
        if entry <= 0 or atr <= 0:
            logger.warning("Invalid entry ({}) or ATR ({}) for TP calculation", entry, atr)
            return []

        # Use the actual stop distance as the base unit so every TP level
        # represents a genuine multiple of the risk taken.
        base_distance = stop_distance if (stop_distance is not None and stop_distance > 0) else atr

        levels: List[dict] = []
        for proportion, rr_mult in zip(self.DEFAULT_TP_PROPORTIONS, self.DEFAULT_RR_MULTIPLIERS):
            distance = base_distance * rr_mult
            if direction == "long":
                price = entry + distance
            else:
                price = max(entry - distance, 0.0)
            levels.append(
                {
                    "price": round(price, 8),
                    "proportion": proportion,
                    "rr_ratio": rr_mult,
                }
            )
        logger.debug(
            "TP levels for {} entry={}: {}",
            direction,
            entry,
            [(lvl["price"], lvl["proportion"]) for lvl in levels],
        )
        return levels

    def calculate_resistance_based_tp(
        self,
        entry: float,
        direction: str,
        resistance_levels: list,
    ) -> List[dict]:
        """Place take-profit levels at key resistance (or support) price levels.

        For **long** positions TP levels are the resistance levels *above* entry.
        For **short** positions they are the support levels *below* entry.
        Levels are sorted from nearest to farthest and wrapped in the standard
        TP-level dict format.

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            resistance_levels: List of price levels representing resistance / support.

        Returns:
            List of TP-level dicts (``price``, ``proportion``, ``rr_ratio``),
            or an empty list when no suitable levels are found.
        """
        if not resistance_levels or entry <= 0:
            return []

        if direction == "long":
            candidates = sorted([lvl for lvl in resistance_levels if lvl > entry])
        else:
            candidates = sorted([lvl for lvl in resistance_levels if lvl < entry], reverse=True)

        if not candidates:
            return []

        # Distribute proportions across available levels (cap at 3 levels)
        levels = candidates[:3]
        proportions = [0.5, 0.3, 0.2][: len(levels)]
        # Pad in case fewer than 3 levels exist
        if len(proportions) < len(levels):
            proportions = [1.0 / len(levels)] * len(levels)

        result = []
        for price, proportion in zip(levels, proportions):
            risk_distance = abs(entry * 0.02)  # fallback: 2% of entry as risk estimate
            reward_distance = abs(price - entry)
            rr = round(reward_distance / risk_distance, 2) if risk_distance > 0 else 0.0
            result.append(
                {
                    "price": round(price, 8),
                    "proportion": proportion,
                    "rr_ratio": rr,
                }
            )

        logger.debug(
            "Resistance-based TP levels for {} entry={}: {}",
            direction,
            entry,
            [(lvl["price"], lvl["proportion"]) for lvl in result],
        )
        return result

    def calculate_dynamic_tp(
        self,
        entry: float,
        direction: str,
        atr: float,
        volatility_regime: str,
        risk_reward_min: float = 1.5,
    ) -> List[dict]:
        """Calculate TP levels using regime-appropriate risk-reward ratios.

        In trending markets wider TPs are used (3:1 / 4:1 / 6:1); in ranging
        or low-volatility markets tighter TPs are preferred (1.5:1 / 2:1 / 2.5:1).

        Args:
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            atr: Average True Range used as the base stop distance.
            volatility_regime: Current volatility regime label.
            risk_reward_min: Minimum R:R for TP1 (overrides regime default when
                the regime's TP1 is below this value).

        Returns:
            List of TP-level dicts (``price``, ``proportion``, ``rr_ratio``).
        """
        if entry <= 0 or atr <= 0:
            logger.warning(
                "Invalid entry ({}) or ATR ({}) for dynamic TP calculation", entry, atr
            )
            return []

        # Regime → (TP1 RR, TP2 RR, TP3 RR)
        _RR_MAP = {
            "low": (1.5, 2.0, 2.5),
            "normal": (2.0, 3.0, 4.5),
            "medium": (2.0, 3.0, 4.5),
            "high": (3.0, 4.0, 6.0),
            "extreme": (3.0, 4.0, 6.0),
        }
        rr_levels = _RR_MAP.get(volatility_regime, (2.0, 3.0, 4.5))

        # Ensure TP1 meets the minimum R:R requirement
        rr_levels = (
            max(rr_levels[0], risk_reward_min),
            max(rr_levels[1], risk_reward_min + 1.0),
            max(rr_levels[2], risk_reward_min + 2.0),
        )

        result = []
        for rr_mult, proportion in zip(rr_levels, self.DEFAULT_TP_PROPORTIONS):
            distance = atr * rr_mult
            if direction == "long":
                price = entry + distance
            else:
                price = max(entry - distance, 0.0)
            result.append(
                {
                    "price": round(price, 8),
                    "proportion": proportion,
                    "rr_ratio": rr_mult,
                }
            )

        logger.debug(
            "Dynamic TP levels for {} entry={} regime={}: {}",
            direction,
            entry,
            volatility_regime,
            [(lvl["price"], lvl["rr_ratio"]) for lvl in result],
        )
        return result

    def should_take_partial(
        self,
        position: "PositionTracker",
        current_price: float,
        tp_level: dict,
    ) -> bool:
        """Return True if *current_price* has reached or passed *tp_level*.

        Args:
            position: Current position tracker.
            current_price: Latest market price.
            tp_level: A TP-level dict as returned by :meth:`calculate_tp_levels`.

        Returns:
            ``True`` if a partial close should be triggered.
        """
        direction = position.position.side.value.lower()
        tp_price = tp_level.get("price", 0.0)
        if direction == "long":
            return current_price >= tp_price
        else:
            return current_price <= tp_price

    def update_trailing_tp(
        self,
        position: "PositionTracker",
        current_price: float,
    ) -> float:
        """Adjust the highest / lowest observed price to support a trailing TP.

        Returns the updated trailing reference price.

        Args:
            position: Current position tracker.
            current_price: Latest market price.

        Returns:
            Updated trailing reference price.
        """
        direction = position.position.side.value.lower()
        if direction == "long":
            if current_price > position.highest_price:
                position.highest_price = current_price
            return position.highest_price
        else:
            if position.lowest_price == 0.0 or current_price < position.lowest_price:
                position.lowest_price = current_price
            return position.lowest_price

    def calculate_rr_ratio(
        self,
        entry: float,
        stop: float,
        target: float,
    ) -> float:
        """Calculate the risk-reward ratio for a given entry, stop, and target.

        Args:
            entry: Entry price.
            stop: Stop-loss price.
            target: Take-profit price.

        Returns:
            Risk-reward ratio (reward / risk).  Returns 0.0 on invalid inputs.
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk == 0:
            logger.warning("Zero risk in R:R calculation (entry={} stop={})", entry, stop)
            return 0.0
        rr = reward / risk
        logger.debug("R:R ratio: entry={} stop={} target={} rr={:.2f}", entry, stop, target, rr)
        return rr

    def time_based_profit_check(
        self,
        position: "PositionTracker",
        current_price: float,
        min_profit_pct: float = 0.005,
        min_hours: float = 4.0,
        momentum_fading: bool = False,
    ) -> bool:
        """Return True if a time-based take-profit should be triggered.

        Takes profit when the trade has been open for at least *min_hours*,
        is currently in profit by at least *min_profit_pct*, and momentum
        is flagged as fading.

        Args:
            position: Current position tracker with ``opened_at`` timestamp.
            current_price: Latest market price.
            min_profit_pct: Minimum unrealised profit fraction to consider
                taking profit (default 0.5 %).
            min_hours: Minimum trade duration in hours before this check
                applies.
            momentum_fading: Pass ``True`` when a momentum indicator (e.g.
                MACD histogram, RSI slope) signals weakening trend.

        Returns:
            ``True`` if a time-based profit close should be executed.
        """
        try:
            entry = position.position.entry_price
            direction = position.position.side.value.lower()
            opened_at = position.opened_at
        except AttributeError as exc:
            logger.error("time_based_profit_check: invalid position object — {}", exc)
            return False

        if entry <= 0 or current_price <= 0:
            return False

        # Check hold time
        elapsed_hours = (
            datetime.now(tz=timezone.utc) - opened_at
        ).total_seconds() / 3600.0
        if elapsed_hours < min_hours:
            return False

        # Check profit
        if direction == "long":
            profit_pct = (current_price - entry) / entry
        else:
            profit_pct = (entry - current_price) / entry

        in_profit = profit_pct >= min_profit_pct
        if in_profit and momentum_fading:
            logger.info(
                "Time-based TP triggered: elapsed={:.1f}h profit={:.2%} momentum_fading=True",
                elapsed_hours,
                profit_pct,
            )
            return True
        return False

    def should_move_to_breakeven(
        self,
        position: "PositionTracker",
        current_price: float,
        tp1_hit: bool = False,
        breakeven_buffer_pct: float = 0.001,
    ) -> Optional[float]:
        """Return the breakeven stop price when conditions are met.

        After TP1 is hit the stop-loss should be moved to the entry price
        (plus a small buffer) to lock in a risk-free trade.

        Args:
            position: Current position tracker.
            current_price: Latest market price.
            tp1_hit: Pass ``True`` once the first take-profit level has been
                closed.
            breakeven_buffer_pct: Small buffer above/below entry to avoid
                immediate stop-out on spread (default 0.1 %).

        Returns:
            New breakeven stop price when the stop should be moved, or
            ``None`` if conditions are not yet met.
        """
        if not tp1_hit:
            return None
        try:
            entry = position.position.entry_price
            direction = position.position.side.value.lower()
        except AttributeError:
            return None

        if entry <= 0:
            return None

        if direction == "long":
            breakeven_stop = entry * (1 + breakeven_buffer_pct)
            # Only move if current stop is below breakeven
            existing_sl = getattr(position, "stop_loss", None)
            if existing_sl is not None and existing_sl >= breakeven_stop:
                return None
        else:
            breakeven_stop = entry * (1 - breakeven_buffer_pct)
            existing_sl = getattr(position, "stop_loss", None)
            if existing_sl is not None and existing_sl <= breakeven_stop:
                return None

        logger.info(
            "Moving SL to breakeven: entry={} dir={} breakeven_stop={:.4f}",
            entry,
            direction,
            breakeven_stop,
        )
        return round(breakeven_stop, 8)
