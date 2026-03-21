"""Real-Time Margin Monitor.

Monitors margin utilisation every 5 seconds and automatically takes
defensive action when utilisation reaches critical levels:

  50 %  → info log
  70 %  → warning alert
  85 %  → close the least profitable position
  95 %  → close ALL positions immediately
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from loguru import logger


class MarginMonitor:
    """Monitor exchange margin utilisation and enforce automated safety levels.

    Args:
        exchange: Exchange wrapper (must support ``get_balance()`` and
            ``get_positions()``).
        position_manager: PositionManager instance for closing positions.
        alert_manager: Optional alerting object with ``send_alert(msg)``; if
            None, alerts are logged only.
        check_interval: Seconds between margin checks (default 5).
    """

    LEVEL_INFO = 0.50
    LEVEL_WARNING = 0.70
    LEVEL_DANGER = 0.85
    LEVEL_EMERGENCY = 0.95

    def __init__(
        self,
        exchange,
        position_manager,
        alert_manager=None,
        check_interval: float = 5.0,
    ) -> None:
        self._exchange = exchange
        self._position_manager = position_manager
        self._alert_manager = alert_manager
        self._check_interval = check_interval
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        # History of (timestamp, utilisation) for the dashboard
        self._history: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background margin-check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="margin_monitor")
        logger.info("MarginMonitor started (interval={}s).", self._check_interval)

    async def stop(self) -> None:
        """Stop the background margin-check loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MarginMonitor stopped.")

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    async def check_margin(self) -> dict:
        """Fetch balance and compute margin utilisation.

        Returns:
            Dict with keys: ``equity``, ``used_margin``, ``free_margin``,
            ``utilisation`` (0–1), ``level`` (str).
        """
        result = {
            "equity": 0.0,
            "used_margin": 0.0,
            "free_margin": 0.0,
            "utilisation": 0.0,
            "level": "ok",
        }
        try:
            balance = await self._exchange.get_balance()
            if balance is None:
                return result

            # Try to extract equity and margin from the balance object/dict
            if hasattr(balance, "total"):
                equity = float(balance.total.get("USDT", 0.0) or 0.0)
                free = float(balance.free.get("USDT", 0.0) or 0.0)
            elif isinstance(balance, dict):
                equity = float(
                    balance.get("total", {}).get("USDT", 0.0)
                    or balance.get("equity", 0.0)
                    or 0.0
                )
                free = float(
                    balance.get("free", {}).get("USDT", 0.0)
                    or balance.get("availableBalance", 0.0)
                    or 0.0
                )
            else:
                equity = 0.0
                free = 0.0

            used = max(0.0, equity - free)
            utilisation = used / equity if equity > 0 else 0.0

            result["equity"] = equity
            result["used_margin"] = used
            result["free_margin"] = free
            result["utilisation"] = utilisation

            if utilisation >= self.LEVEL_EMERGENCY:
                result["level"] = "emergency"
            elif utilisation >= self.LEVEL_DANGER:
                result["level"] = "danger"
            elif utilisation >= self.LEVEL_WARNING:
                result["level"] = "warning"
            elif utilisation >= self.LEVEL_INFO:
                result["level"] = "info"
            else:
                result["level"] = "ok"

            # Record history (keep last 1440 entries ≈ 2h at 5-s interval)
            self._history.append((time.time(), utilisation))
            if len(self._history) > 1440:
                self._history = self._history[-1440:]

        except Exception as exc:
            logger.warning("MarginMonitor.check_margin failed: {}", exc)

        return result

    async def enforce_margin_limits(self) -> None:
        """Evaluate margin and take automated action if needed."""
        info = await self.check_margin()
        util = info["utilisation"]
        level = info["level"]

        if level == "ok":
            return

        msg = (
            f"Margin utilisation: {util:.1%} "
            f"(equity={info['equity']:.2f} used={info['used_margin']:.2f})"
        )

        if level == "info":
            logger.info("MarginMonitor [INFO] {}", msg)

        elif level == "warning":
            logger.warning("MarginMonitor [WARNING] {}", msg)
            await self._alert(f"⚠️ Margin Warning: {msg}")

        elif level == "danger":
            logger.error("MarginMonitor [DANGER] {} — closing least profitable position.", msg)
            await self._alert(f"🔴 Margin Danger: {msg} — auto-closing weakest position.")
            await self._close_least_profitable()

        elif level == "emergency":
            logger.critical("MarginMonitor [EMERGENCY] {} — closing ALL positions!", msg)
            await self._alert(f"🚨 Margin EMERGENCY: {msg} — closing ALL positions NOW!")
            await self._close_all_positions()

    # ------------------------------------------------------------------
    # History accessor
    # ------------------------------------------------------------------

    def get_history(self) -> list:
        """Return the margin utilisation history as a list of (timestamp, utilisation)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Background loop that calls enforce_margin_limits every N seconds."""
        while self._running:
            try:
                await self.enforce_margin_limits()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("MarginMonitor loop error: {}", exc)
            await asyncio.sleep(self._check_interval)

    async def _close_least_profitable(self) -> None:
        """Close the single least profitable open position."""
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return
            # Find least profitable (highest unrealised loss)
            worst = min(positions, key=lambda p: float(getattr(p, "unrealized_pnl", 0) or 0))
            symbol = getattr(worst, "symbol", None)
            if symbol and self._position_manager is not None:
                logger.warning(
                    "MarginMonitor: closing least profitable position: {}", symbol
                )
                await self._position_manager.close_position(symbol)
        except Exception as exc:
            logger.error("MarginMonitor._close_least_profitable failed: {}", exc)

    async def _close_all_positions(self) -> None:
        """Close all open positions immediately."""
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return
            for pos in positions:
                symbol = getattr(pos, "symbol", None)
                if symbol and self._position_manager is not None:
                    try:
                        await self._position_manager.close_position(symbol)
                    except Exception as exc:
                        logger.error(
                            "MarginMonitor: failed to close {}: {}", symbol, exc
                        )
        except Exception as exc:
            logger.error("MarginMonitor._close_all_positions failed: {}", exc)

    async def _alert(self, message: str) -> None:
        """Send an alert via the alert manager (if available) or log it."""
        logger.warning("MarginMonitor alert: {}", message)
        if self._alert_manager is not None:
            try:
                await self._alert_manager.send_alert(message)
            except Exception as exc:
                logger.debug("MarginMonitor alert send failed: {}", exc)
