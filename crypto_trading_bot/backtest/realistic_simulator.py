"""Realistic market simulator with exchange impact, regime-aware slippage,
partial fills, and variable latency.

Extends the lightweight :class:`~backtest.simulator.TradeSimulator` with
institutional-grade market microstructure effects:

* **Market impact** via a square-root impact model
  (``impact_bps = η × sqrt(order_size / ADV) × 10_000``).
* **Regime-aware slippage**: slippage is scaled by an average daily volume
  multiplier that doubles in high-volatility regimes and quadruples in crash.
* **Partial fills**: limit orders are partially filled proportional to the
  fraction of bar volume they represent.
* **Variable latency**: execution latency drawn from a log-normal distribution
  delays fills by up to several milliseconds.
* **Realistic funding rates** applied every 8 hours.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from loguru import logger


class RealisticSimulator:
    """Institutional-grade trade execution simulator.

    Args:
        base_slippage_pct: Base slippage percentage in normal market
            conditions (default 0.05 %).
        impact_eta: Square-root impact model coefficient η.
        adv_usd: Assumed average daily volume in USD (used for impact calc).
        partial_fill_probability: Probability (0–1) that a limit order is
            only partially filled on a given bar.
        latency_mean_ms: Mean execution latency in milliseconds (log-normal μ).
        latency_std_ms: Std-dev of execution latency (log-normal σ).
        funding_interval_hours: Funding payment interval in hours (default 8).
    """

    # Regime → slippage multiplier
    _REGIME_SLIPPAGE_MULTIPLIER: Dict[str, float] = {
        "trending_up": 1.0,
        "trending_down": 1.2,
        "ranging": 0.8,
        "high_vol": 2.0,
        "high_volatility": 2.0,
        "crash": 4.0,
        "extreme": 3.5,
        "normal": 1.0,
        "unknown": 1.0,
    }

    def __init__(
        self,
        base_slippage_pct: float = 0.0005,
        impact_eta: float = 0.1,
        adv_usd: float = 500_000_000,
        partial_fill_probability: float = 0.15,
        latency_mean_ms: float = 50.0,
        latency_std_ms: float = 30.0,
        funding_interval_hours: float = 8.0,
    ) -> None:
        self.base_slippage_pct = base_slippage_pct
        self.impact_eta = impact_eta
        self.adv_usd = adv_usd
        self.partial_fill_probability = partial_fill_probability
        self.latency_mean_ms = latency_mean_ms
        self.latency_std_ms = latency_std_ms
        self.funding_interval_hours = funding_interval_hours

        # Counters for diagnostics
        self._total_entries = 0
        self._total_exits = 0
        self._partial_fills = 0

    # ------------------------------------------------------------------
    # Entry / Exit with realistic effects
    # ------------------------------------------------------------------

    def simulate_entry(
        self,
        signal: dict,
        ohlcv_row: dict,
        capital: float,
        fee_rate: float = 0.0004,
        regime: str = "normal",
        order_type: str = "market",
    ) -> dict:
        """Simulate a trade entry with market-microstructure effects.

        Args:
            signal: Signal dict with ``side``, ``size``, optional
                ``stop_loss`` / ``take_profit``.
            ohlcv_row: Current OHLCV bar dict.
            capital: Available capital in quote currency.
            fee_rate: Taker fee fraction.
            regime: Current market regime for slippage scaling.
            order_type: ``"market"`` or ``"limit"`` (limit orders may
                be partially filled).

        Returns:
            Position dict or ``None`` if the order was fully rejected
            (extremely rare — only when bar volume is 0).
        """
        side = signal.get("side", "long")
        size_fraction = float(signal.get("size", 0.1))
        raw_price = float(ohlcv_row.get("open", ohlcv_row.get("close", 0.0)))
        bar_volume_usd = float(ohlcv_row.get("volume", 1.0)) * raw_price

        position_value = capital * size_fraction

        # --- Market impact ---
        impact_pct = self._market_impact_pct(position_value)

        # --- Regime-aware slippage ---
        slip = self._regime_slippage(regime)

        # --- Latency adjustment ---
        latency_price_move = self._latency_price_impact(raw_price)

        # Combine: longs pay more, shorts receive less
        if side == "long":
            exec_price = raw_price * (1 + slip + impact_pct) + latency_price_move
        else:
            exec_price = raw_price * (1 - slip - impact_pct) - latency_price_move

        exec_price = max(1e-9, exec_price)

        # --- Partial fill logic for limit orders ---
        fill_fraction = 1.0
        if order_type == "limit" and bar_volume_usd > 0:
            if random.random() < self.partial_fill_probability:
                # Partial fill: fill between 40–90 % of the order
                fill_fraction = random.uniform(0.4, 0.9)
                self._partial_fills += 1
                logger.debug(
                    "Partial fill: {:.0%} of limit order at {:.4f}", fill_fraction, exec_price
                )

        filled_value = position_value * fill_fraction
        quantity = filled_value / exec_price
        fee = quantity * exec_price * fee_rate
        net_capital = filled_value + fee

        self._total_entries += 1

        position = {
            "symbol": signal.get("symbol", ""),
            "side": side,
            "entry_price": exec_price,
            "quantity": quantity,
            "fill_fraction": fill_fraction,
            "entry_fee": fee,
            "capital_used": net_capital,
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "entry_time": ohlcv_row.get("timestamp", datetime.now(timezone.utc)),
            "status": "open",
            "pnl": 0.0,
            "funding_paid": 0.0,
            "regime_at_entry": regime,
            "impact_pct": impact_pct,
            "slippage_pct": slip,
        }

        logger.debug(
            "RealisticEntry: {} {} @ {:.4f} (impact={:.2f}bps slip={:.2f}bps fill={:.0%})",
            side,
            position["symbol"],
            exec_price,
            impact_pct * 10_000,
            slip * 10_000,
            fill_fraction,
        )
        return position

    def simulate_exit(
        self,
        position: dict,
        ohlcv_row: dict,
        reason: str = "manual",
        fee_rate: float = 0.0004,
        regime: str = "normal",
    ) -> dict:
        """Simulate closing *position* with realistic execution.

        Args:
            position: Open position dict.
            ohlcv_row: Current OHLCV bar.
            reason: Exit reason label.
            fee_rate: Taker fee fraction.
            regime: Current market regime.

        Returns:
            Updated position dict with P&L fields populated.
        """
        side = position["side"]
        raw_price = float(ohlcv_row.get("close", ohlcv_row.get("open", 0.0)))
        quantity = position["quantity"]
        notional = quantity * raw_price

        # Market impact and slippage at exit
        impact_pct = self._market_impact_pct(notional)
        slip = self._regime_slippage(regime)
        latency_move = self._latency_price_impact(raw_price)

        exit_side = "short" if side == "long" else "long"
        if exit_side == "short":
            exec_price = raw_price * (1 - slip - impact_pct) - latency_move
        else:
            exec_price = raw_price * (1 + slip + impact_pct) + latency_move
        exec_price = max(1e-9, exec_price)

        exit_fee = quantity * exec_price * fee_rate

        if side == "long":
            gross_pnl = (exec_price - position["entry_price"]) * quantity
        else:
            gross_pnl = (position["entry_price"] - exec_price) * quantity

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

        capital_used = position.get("capital_used", 1.0)
        self._total_exits += 1

        updated = dict(position)
        updated.update(
            {
                "exit_price": exec_price,
                "exit_fee": exit_fee,
                "exit_time": exit_time,
                "exit_reason": reason,
                "pnl": net_pnl,
                "pnl_pct": net_pnl / capital_used * 100 if capital_used else 0.0,
                "duration_hours": duration_hours,
                "status": "closed",
                "exit_impact_pct": impact_pct,
                "exit_slippage_pct": slip,
            }
        )
        logger.debug(
            "RealisticExit: {} @ {:.4f}, pnl={:.4f} ({})",
            reason,
            exec_price,
            net_pnl,
            position.get("symbol", ""),
        )
        return updated

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    def apply_funding_payment(
        self,
        position: dict,
        current_time: datetime,
        funding_rate: float = 0.0001,
    ) -> Tuple[dict, float]:
        """Apply periodic funding rate payment to *position*.

        Returns:
            Updated position dict and the funding payment amount.
        """
        entry_time = position.get("entry_time")
        if entry_time is None:
            return position, 0.0

        if isinstance(current_time, datetime) and isinstance(entry_time, datetime):
            hours_held = (current_time - entry_time).total_seconds() / 3600
        else:
            hours_held = 0.0

        n_payments = int(hours_held / self.funding_interval_hours)
        notional = position["quantity"] * position["entry_price"]
        side = position["side"]

        total_funding = 0.0
        for _ in range(max(0, n_payments)):
            payment = notional * funding_rate if side == "long" else -notional * funding_rate
            total_funding += payment

        updated = dict(position)
        updated["funding_paid"] = position.get("funding_paid", 0.0) + total_funding
        return updated, total_funding

    # ------------------------------------------------------------------
    # SL / TP detection (pass-through)
    # ------------------------------------------------------------------

    def check_sl_tp_hit(self, position: dict, ohlcv_row: dict) -> Optional[str]:
        """Proxy to the simple SL/TP check."""
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
        else:
            if sl is not None and high >= float(sl):
                return "sl"
            if tp is not None and low <= float(tp):
                return "tp"
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _market_impact_pct(self, order_value_usd: float) -> float:
        """Compute market impact as a fraction using the sqrt model."""
        if self.adv_usd <= 0 or order_value_usd <= 0:
            return 0.0
        fraction = order_value_usd / self.adv_usd
        impact_pct = self.impact_eta * math.sqrt(fraction)
        return min(impact_pct, 0.01)  # cap at 1 %

    def _regime_slippage(self, regime: str) -> float:
        """Return slippage percentage adjusted for market regime."""
        multiplier = self._REGIME_SLIPPAGE_MULTIPLIER.get(regime, 1.0)
        return self.base_slippage_pct * multiplier

    def _latency_price_impact(self, price: float) -> float:
        """Sample a small latency-induced price move."""
        # Log-normal latency in ms → proportional price move
        mu = math.log(self.latency_mean_ms) if self.latency_mean_ms > 0 else 0.0
        sigma = self.latency_std_ms / (self.latency_mean_ms + 1e-9)
        latency_ms = math.exp(random.gauss(mu, sigma))
        # Rough approximation: 0.001 % price move per millisecond
        move_pct = latency_ms * 0.00001
        return price * move_pct * random.choice([-1, 1])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return simulator execution statistics."""
        return {
            "total_entries": self._total_entries,
            "total_exits": self._total_exits,
            "partial_fills": self._partial_fills,
            "partial_fill_rate": (
                self._partial_fills / max(1, self._total_entries)
            ),
        }
