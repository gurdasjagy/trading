"""Background task for sending periodic summaries of queued alerts.

This module implements a background task that runs every 5 minutes to collect
queued alerts from the AlertManager and send them as a single summary message,
bypassing rate limits.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Dict, List

from loguru import logger

if TYPE_CHECKING:
    from monitoring.alerting import AlertManager, AlertType


class AlertSummaryTask:
    """Background task that sends periodic summaries of queued alerts.
    
    Runs every 5 minutes to collect alerts that were queued due to rate limiting
    and sends them as a single summary message with force=True to bypass limits.
    
    Args:
        alert_manager: AlertManager instance to collect queued alerts from.
        interval_seconds: How often to send summaries (default: 300 = 5 minutes).
    """

    def __init__(
        self,
        alert_manager: "AlertManager",
        interval_seconds: int = 300,
    ) -> None:
        self.alert_manager = alert_manager
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background summary task."""
        if self._running:
            logger.warning("AlertSummaryTask already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="alert_summary_task")
        logger.info("AlertSummaryTask started (interval={}s)", self.interval_seconds)

    async def stop(self) -> None:
        """Stop the background summary task."""
        if not self._running:
            return
        
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AlertSummaryTask stopped")

    async def _run_loop(self) -> None:
        """Main loop that sends summaries every interval_seconds."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                await self._send_summary()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("AlertSummaryTask error: {}", exc)

    async def _send_summary(self) -> None:
        """Collect queued alerts and send as a summary message."""
        try:
            # Collect queued alerts from AlertManager
            queued_alerts = self._collect_queued_alerts()
            
            if not queued_alerts:
                logger.debug("No queued alerts to summarize")
                return
            
            # Format summary message
            summary_lines = ["📊 **Queued Alert Summary**\n"]
            total_count = 0
            
            for alert_type, messages in queued_alerts.items():
                count = len(messages)
                total_count += count
                summary_lines.append(f"• {alert_type}: {count} alert(s)")
            
            summary_lines.append(f"\n**Total**: {total_count} queued alert(s)")
            summary_message = "\n".join(summary_lines)
            
            # Send summary with force=True to bypass rate limits
            success = await self.alert_manager.send_alert(
                summary_message,
                force=True,
            )
            
            if success:
                # Clear queued alerts after successful send
                self._clear_queued_alerts()
                logger.info("Sent alert summary: {} total alerts", total_count)
            else:
                logger.warning("Failed to send alert summary")
                
        except Exception as exc:
            logger.error("Error sending alert summary: {}", exc)

    def _collect_queued_alerts(self) -> Dict[str, List[str]]:
        """Collect queued alerts from AlertManager.
        
        Returns:
            Dict mapping alert type to list of queued message strings.
        """
        queued: Dict[str, List[str]] = {}
        
        # Access AlertManager's _queued_alerts dict
        if not hasattr(self.alert_manager, "_queued_alerts"):
            return queued
        
        for alert_type, messages in self.alert_manager._queued_alerts.items():
            if messages:
                # Convert AlertType enum to string for display
                type_str = alert_type.value if hasattr(alert_type, "value") else str(alert_type)
                queued[type_str] = list(messages)
        
        return queued

    def _clear_queued_alerts(self) -> None:
        """Clear all queued alerts from AlertManager after sending summary."""
        if not hasattr(self.alert_manager, "_queued_alerts"):
            return
        
        for alert_type in list(self.alert_manager._queued_alerts.keys()):
            self.alert_manager._queued_alerts[alert_type].clear()
