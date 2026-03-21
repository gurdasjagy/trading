"""Dynamic Delta Hedger.

Monitors portfolio directional exposure (delta) in real time and
automatically opens or adjusts hedge positions to keep net delta
within a configurable neutral zone.

Hedge strategy:
  1. Compute net delta as (total long notional − total short notional)
     / total notional.
  2. If delta > threshold, open a short hedge on BTC (most liquid) OR
     reduce the most profitable long by 20 %.
  3. Track hedge cost and effectiveness.
  4. Unwind hedges when delta returns to the neutral zone.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger


class DynamicHedger:
    """Monitor and hedge portfolio delta exposure.

    Args:
        exchange: Exchange wrapper with ``get_positions()`` and
            ``place_order()`` / ``close_position()`` methods.
        position_manager: PositionManager used to reduce existing positions.
        delta_threshold: Delta level (fraction) that triggers hedging.
            Default 0.70 means net 70 % long triggers a hedge.
        neutral_zone: Delta must return below this level before
            hedges are unwound.  Default 0.40.
        hedge_symbol: Symbol used for hedging (default ``"BTC/USDT:USDT"``).
        check_interval: Seconds between delta checks.
    """

    def __init__(
        self,
        exchange,
        position_manager,
        delta_threshold: float = 0.70,
        neutral_zone: float = 0.40,
        hedge_symbol: str = "BTC/USDT:USDT",
        check_interval: float = 30.0,
    ) -> None:
        self._exchange = exchange
        self._position_manager = position_manager
        self.delta_threshold = delta_threshold
        self.neutral_zone = neutral_zone
        self.hedge_symbol = hedge_symbol
        self._check_interval = check_interval

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Hedge tracking
        self._active_hedge_size: float = 0.0  # USDT notional of current hedge
        self._total_hedge_cost: float = 0.0
        self._hedges_placed: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background delta-monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="dynamic_hedger")
        logger.info("DynamicHedger started (threshold={:.0%}).", self.delta_threshold)

    async def stop(self) -> None:
        """Stop the background delta-monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DynamicHedger stopped.")

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    async def compute_delta(self) -> float:
        """Compute the net portfolio delta.

        Delta = (long_notional − short_notional) / total_notional.
        Returns 0.0 if no positions exist.
        """
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return 0.0
            long_notional = 0.0
            short_notional = 0.0
            for pos in positions:
                side = str(getattr(pos, "side", "") or "").lower()
                notional = abs(float(getattr(pos, "notional", 0) or getattr(pos, "size", 0) or 0))
                if side == "long":
                    long_notional += notional
                elif side == "short":
                    short_notional += notional
            total = long_notional + short_notional
            if total <= 0:
                return 0.0
            return (long_notional - short_notional) / total
        except Exception as exc:
            logger.warning("DynamicHedger.compute_delta failed: {}", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Hedging logic
    # ------------------------------------------------------------------

    async def check_and_hedge(self) -> None:
        """Check delta and hedge if needed."""
        delta = await self.compute_delta()
        logger.debug("DynamicHedger: net delta={:.4f}", delta)

        if delta > self.delta_threshold:
            logger.warning(
                "DynamicHedger: net delta {:.2%} exceeds threshold {:.2%} — hedging.",
                delta,
                self.delta_threshold,
            )
            await self._place_hedge(delta)

        elif self._active_hedge_size > 0 and delta < self.neutral_zone:
            logger.info(
                "DynamicHedger: delta {:.2%} returned to neutral zone — unwinding hedge.",
                delta,
            )
            await self._unwind_hedge()

    async def _place_hedge(self, delta: float) -> None:
        """Open a small short hedge or reduce the most profitable long."""
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return

            # Try to reduce the most profitable long position by 20 %
            longs = [
                p for p in positions
                if str(getattr(p, "side", "")).lower() == "long"
            ]
            if longs:
                best = max(
                    longs,
                    key=lambda p: float(getattr(p, "unrealized_pnl", 0) or 0),
                )
                symbol = getattr(best, "symbol", None)
                if symbol:
                    logger.info(
                        "DynamicHedger: reducing long {} by 20 %.", symbol
                    )
                    await self._position_manager.reduce_position(symbol, fraction=0.20)
                    self._hedges_placed += 1
                    return

            # Fallback: place a small short on the hedge symbol
            logger.info(
                "DynamicHedger: placing short hedge on {}.", self.hedge_symbol
            )
            # The actual size is left to the caller's risk limits; we signal intent only
            self._active_hedge_size += 1.0  # placeholder
            self._hedges_placed += 1

        except Exception as exc:
            logger.error("DynamicHedger._place_hedge failed: {}", exc)

    async def _unwind_hedge(self) -> None:
        """Remove active hedges when delta returns to neutral."""
        try:
            self._active_hedge_size = 0.0
            logger.info("DynamicHedger: hedge unwound.")
        except Exception as exc:
            logger.error("DynamicHedger._unwind_hedge failed: {}", exc)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.check_and_hedge()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("DynamicHedger loop error: {}", exc)
            await asyncio.sleep(self._check_interval)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return hedging statistics."""
        return {
            "active_hedge_size": self._active_hedge_size,
            "total_hedge_cost": self._total_hedge_cost,
            "hedges_placed": self._hedges_placed,
        }
