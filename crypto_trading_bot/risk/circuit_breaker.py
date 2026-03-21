"""Emergency circuit-breaker — halts all trading activity when risk limits are breached."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional

from loguru import logger


class CircuitBreaker:
    """Emergency stop mechanism that closes positions and disables trading.

    Triggers:
    - Daily loss > 5 %
    - Single trade loss > 3 %
    - 5 consecutive losses
    - Exchange API errors
    - Abnormal market movement (> 10 % in 5 min)
    - Data feed failure
    - Low AI confidence
    - Unrealised PnL < -3 %

    Actions on trigger:
    - Close ALL positions
    - Cancel ALL orders
    - Disable new trades
    - Send emergency alerts
    - Log state snapshot
    - Require manual reset
    """

    def __init__(self) -> None:
        self._triggered: bool = False
        self._trigger_reason: Optional[str] = None
        self._triggered_at: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._emergency_close_verified: bool = False

        # Callbacks injected by the bot engine
        self._close_all_positions_cb: Optional[Callable] = None
        self._cancel_all_orders_cb: Optional[Callable] = None
        self._send_alert_cb: Optional[Callable] = None
        self._verify_positions_cb: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def register_callbacks(
        self,
        close_positions: Callable,
        cancel_orders: Callable,
        send_alert: Callable,
        verify_positions: Optional[Callable] = None,
    ) -> None:
        """Register async callbacks invoked on circuit-breaker activation.

        Args:
            close_positions: Coroutine function that closes all open positions.
            cancel_orders: Coroutine function that cancels all open orders.
            send_alert: Coroutine function that sends an emergency alert message.
            verify_positions: Optional coroutine function that returns the list of
                open positions, used to verify the emergency close succeeded.
        """
        self._close_all_positions_cb = close_positions
        self._cancel_all_orders_cb = cancel_orders
        self._send_alert_cb = send_alert
        self._verify_positions_cb = verify_positions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_triggered(self) -> bool:
        """Return True if the circuit breaker is currently active."""
        return self._triggered

    async def check(
        self,
        daily_loss_pct: float = 0.0,
        last_trade_loss_pct: float = 0.0,
        consecutive_losses: int = 0,
        api_error: bool = False,
        abnormal_market: bool = False,
        data_feed_failure: bool = False,
        low_ai_confidence: bool = False,
        unrealised_pnl_pct: float = 0.0,
    ) -> bool:
        """Evaluate all circuit-breaker conditions and trigger if any are met.

        Args:
            daily_loss_pct: Today's cumulative loss as a positive percentage.
            last_trade_loss_pct: Loss on the most recent trade as a positive percentage.
            consecutive_losses: Number of consecutive losing trades.
            api_error: True if an exchange API error has occurred.
            abnormal_market: True if an abnormal price move (>10 % in 5 min) was detected.
            data_feed_failure: True if the market data feed has failed.
            low_ai_confidence: True if the AI model confidence is below threshold.
            unrealised_pnl_pct: Current unrealised PnL as a signed percentage.

        Returns:
            ``True`` if the circuit breaker is (now) triggered.
        """
        if self._triggered:
            return True

        reason: Optional[str] = None

        if daily_loss_pct >= 5.0:
            reason = f"Daily loss limit reached: {daily_loss_pct:.2f}%"
        elif last_trade_loss_pct >= 3.0:
            reason = f"Single trade loss limit reached: {last_trade_loss_pct:.2f}%"
        elif consecutive_losses >= 5:
            reason = f"Consecutive losses limit reached: {consecutive_losses}"
        elif api_error:
            reason = "Exchange API error"
        elif abnormal_market:
            reason = "Abnormal market movement detected (>10% in 5min)"
        elif data_feed_failure:
            reason = "Market data feed failure"
        elif low_ai_confidence:
            reason = "Low AI model confidence"
        elif unrealised_pnl_pct <= -3.0:
            reason = f"Unrealised PnL critically low: {unrealised_pnl_pct:.2f}%"

        if reason:
            await self.trigger(reason)
            return True
        return False

    async def trigger(self, reason: str) -> None:
        """Activate the circuit breaker and execute emergency actions.

        Args:
            reason: Human-readable description of the trigger condition.
        """
        async with self._lock:
            if self._triggered:
                return
            self._triggered = True
            self._trigger_reason = reason
            self._triggered_at = datetime.now(tz=timezone.utc)

        logger.critical("🚨 CIRCUIT BREAKER TRIGGERED: {}", reason)
        await self.emergency_close_all()

        if self._send_alert_cb:
            try:
                await self._send_alert_cb(
                    f"🚨 CIRCUIT BREAKER ACTIVATED\nReason: {reason}\n"
                    f"Time: {self._triggered_at.isoformat()}\n"
                    "All positions closed. Manual reset required."
                )
            except Exception as exc:
                logger.error("Failed to send circuit-breaker alert: {}", exc)

        logger.critical(
            "Circuit breaker state snapshot — reason='{}' triggered_at={}",
            reason,
            self._triggered_at,
        )

    async def reset(self) -> None:
        """Reset the circuit breaker — requires manual intervention.

        This does NOT automatically re-enable trading; the calling code
        should verify conditions are safe before resuming.
        """
        async with self._lock:
            was_triggered = self._triggered
            self._triggered = False
            self._trigger_reason = None
            self._triggered_at = None

        if was_triggered:
            logger.warning("Circuit breaker manually reset — trading may resume")

    async def emergency_close_all(self) -> None:
        """Close all positions and cancel all orders with retry logic and verification.

        Retries each phase up to 3 times with 2-second delays between attempts.
        After closing, verifies positions are actually closed via the verify_positions
        callback (if registered).  Sets ``_emergency_close_verified`` accordingly.
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 2.0

        logger.warning("Emergency close-all initiated")

        # Phase 1: Cancel all orders (with retries)
        for attempt in range(MAX_RETRIES):
            try:
                if self._cancel_all_orders_cb:
                    await asyncio.wait_for(self._cancel_all_orders_cb(), timeout=10.0)
                    logger.info("All orders cancelled (circuit breaker)")
                    break
            except Exception as exc:
                logger.error(
                    "Cancel orders attempt {}/{} failed: {}",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)

        # Phase 2: Close all positions (with retries)
        for attempt in range(MAX_RETRIES):
            try:
                if self._close_all_positions_cb:
                    await asyncio.wait_for(self._close_all_positions_cb(), timeout=15.0)
                    logger.info("All positions closed (circuit breaker)")
                    break
            except Exception as exc:
                logger.error(
                    "Close positions attempt {}/{} failed: {}",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)

        # Phase 3: Verify positions are actually closed
        if self._verify_positions_cb:
            try:
                positions = await self._verify_positions_cb()
                self._emergency_close_verified = len(positions) == 0
                if not self._emergency_close_verified:
                    logger.critical(
                        "EMERGENCY CLOSE FAILED: {} positions still open!",
                        len(positions),
                    )
                    if self._send_alert_cb:
                        try:
                            await self._send_alert_cb(
                                f"🚨 EMERGENCY CLOSE VERIFICATION FAILED\n"
                                f"{len(positions)} positions still open after all retries!\n"
                                "Immediate manual intervention required."
                            )
                        except Exception as alert_exc:
                            logger.error(
                                "Failed to send emergency verification alert: {}", alert_exc
                            )
            except Exception as verify_exc:
                self._emergency_close_verified = False
                logger.critical(
                    "EMERGENCY CLOSE VERIFICATION ERROR: could not verify positions: {}",
                    verify_exc,
                )

    @property
    def trigger_info(self) -> dict:
        """Return current circuit-breaker state as a dict."""
        return {
            "triggered": self._triggered,
            "reason": self._trigger_reason,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "emergency_close_verified": self._emergency_close_verified,
        }
