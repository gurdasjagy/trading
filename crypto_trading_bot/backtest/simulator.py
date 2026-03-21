"""Trade execution simulator with realistic fees, slippage, and funding rates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger


class TradeSimulator:
    """Simulates trade execution with realistic market conditions."""

    # ------------------------------------------------------------------
    # Entry / exit simulation
    # ------------------------------------------------------------------

    def simulate_entry(
        self,
        signal: dict,
        ohlcv_row: dict,
        capital: float,
        fee_rate: float = 0.001,
    ) -> dict:
        """Simulate a trade entry based on *signal* and the current OHLCV bar.

        Args:
            signal: Signal dict with keys ``side`` (``"long"``/``"short"``),
                ``size`` (fraction of *capital* to risk), and optional
                ``stop_loss``/``take_profit``.
            ohlcv_row: Single OHLCV bar as a dict with keys ``open``, ``high``,
                ``low``, ``close``, ``volume`` and ``timestamp``.
            capital: Available capital in quote currency.
            fee_rate: Taker fee fraction (default 0.1%).

        Returns:
            Position dict describing the simulated entry.
        """
        side = signal.get("side", "long")
        size_fraction = float(signal.get("size", 0.1))
        raw_price = float(ohlcv_row.get("open", ohlcv_row.get("close", 0)))

        entry_price = self.apply_slippage(raw_price, side)
        position_value = capital * size_fraction
        quantity = position_value / entry_price
        fee = self.calculate_fee(quantity, entry_price, fee_rate)
        net_capital_used = position_value + fee

        position = {
            "symbol": signal.get("symbol", ""),
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "entry_fee": fee,
            "capital_used": net_capital_used,
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "entry_time": ohlcv_row.get("timestamp", datetime.now(timezone.utc)),
            "status": "open",
            "pnl": 0.0,
            "funding_paid": 0.0,
        }
        logger.debug(
            "Simulated entry: {} {} @ {:.4f}, qty={:.6f}, fee={:.4f}",
            side,
            position["symbol"],
            entry_price,
            quantity,
            fee,
        )
        return position

    def simulate_exit(
        self,
        position: dict,
        ohlcv_row: dict,
        reason: str = "manual",
        fee_rate: float = 0.001,
    ) -> dict:
        """Simulate closing *position* at the current OHLCV bar.

        Args:
            position: Open position dict as returned by :meth:`simulate_entry`.
            ohlcv_row: Current OHLCV bar.
            reason: Exit reason label (e.g. ``"sl"``, ``"tp"``, ``"signal"``).
            fee_rate: Taker fee fraction.

        Returns:
            Updated position dict with ``pnl``, ``exit_price``, ``exit_time``,
            and ``status`` fields populated.
        """
        side = position["side"]
        raw_price = float(ohlcv_row.get("close", ohlcv_row.get("open", 0)))
        exit_side = "short" if side == "long" else "long"
        exit_price = self.apply_slippage(raw_price, exit_side)
        quantity = position["quantity"]
        exit_fee = self.calculate_fee(quantity, exit_price, fee_rate)

        if side == "long":
            gross_pnl = (exit_price - position["entry_price"]) * quantity
        else:
            gross_pnl = (position["entry_price"] - exit_price) * quantity

        net_pnl = (
            gross_pnl
            - exit_fee
            - position.get("entry_fee", 0.0)
            - position.get("funding_paid", 0.0)
        )

        exit_time = ohlcv_row.get("timestamp", datetime.now(timezone.utc))
        entry_time = position.get("entry_time", exit_time)
        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
            duration_hours = (exit_time - entry_time).total_seconds() / 3600
        else:
            duration_hours = 0.0

        position = dict(position)  # shallow copy — don't mutate caller's dict
        position.update(
            {
                "exit_price": exit_price,
                "exit_fee": exit_fee,
                "exit_time": exit_time,
                "exit_reason": reason,
                "pnl": net_pnl,
                "pnl_pct": (
                    (net_pnl / position["capital_used"] * 100)
                    if position.get("capital_used")
                    else 0.0
                ),
                "duration_hours": duration_hours,
                "status": "closed",
            }
        )
        logger.debug(
            "Simulated exit: {} @ {:.4f}, pnl={:.4f} ({})",
            reason,
            exit_price,
            net_pnl,
            position["symbol"],
        )
        return position

    # ------------------------------------------------------------------
    # Price adjustments
    # ------------------------------------------------------------------

    def apply_slippage(self, price: float, side: str, slippage_pct: float = 0.0005) -> float:
        """Return *price* adjusted for market slippage.

        Longs pay slightly more; shorts receive slightly less.

        Args:
            price: Raw market price.
            side: ``"long"`` or ``"short"``.
            slippage_pct: Slippage fraction (default 0.05%).

        Returns:
            Adjusted execution price.
        """
        if side == "long":
            return price * (1 + slippage_pct)
        return price * (1 - slippage_pct)

    def calculate_fee(self, amount: float, price: float, fee_rate: float) -> float:
        """Return the trading fee in quote currency.

        Args:
            amount: Quantity of base asset.
            price: Execution price in quote currency.
            fee_rate: Fee fraction (e.g. 0.001 for 0.1%).

        Returns:
            Fee amount in quote currency.
        """
        return amount * price * fee_rate

    # ------------------------------------------------------------------
    # SL / TP detection
    # ------------------------------------------------------------------

    def check_sl_tp_hit(self, position: dict, ohlcv_row: dict) -> Optional[str]:
        """Check whether the current bar triggers a stop-loss or take-profit.

        Args:
            position: Open position dict with optional ``stop_loss`` and
                ``take_profit`` keys.
            ohlcv_row: Current OHLCV bar with ``high`` and ``low`` keys.

        Returns:
            ``"sl"`` if stop-loss hit, ``"tp"`` if take-profit hit, or ``None``.
        """
        sl = position.get("stop_loss")
        tp = position.get("take_profit")
        side = position.get("side", "long")
        high = float(ohlcv_row.get("high", 0))
        low = float(ohlcv_row.get("low", 0))

        if side == "long":
            if sl is not None and low <= float(sl):
                return "sl"
            if tp is not None and high >= float(tp):
                return "tp"
        else:  # short
            if sl is not None and high >= float(sl):
                return "sl"
            if tp is not None and low <= float(tp):
                return "tp"

        return None

    # ------------------------------------------------------------------
    # Funding rate simulation
    # ------------------------------------------------------------------

    def simulate_funding_payment(self, position: dict, funding_rate: float) -> float:
        """Calculate the funding payment for *position* at the given rate.

        Positive funding rate means longs pay shorts; negative means the reverse.

        Args:
            position: Open position dict with ``side``, ``quantity``, and
                ``entry_price`` fields.
            funding_rate: Current funding rate (signed fraction, e.g. 0.0001).

        Returns:
            Funding payment in quote currency (positive = cost, negative = income).
        """
        notional = position["quantity"] * position["entry_price"]
        if position["side"] == "long":
            payment = notional * funding_rate
        else:
            payment = -notional * funding_rate
        logger.debug(
            "Funding payment: {:.6f} for {} {} (rate={:.6f})",
            payment,
            position["side"],
            position.get("symbol", ""),
            funding_rate,
        )
        return payment
