"""Liquidation risk monitor for leveraged positions.

Monitors positions against liquidation thresholds and provides early warning
alerts to prevent forced liquidation events.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from loguru import logger
from dataclasses import dataclass

from exchange.base_exchange import Position, PositionSide


@dataclass
class LiquidationRiskAlert:
    """Liquidation risk alert data class."""

    symbol: str
    liquidation_price: float
    current_price: float
    distance_pct: float
    margin_ratio: float
    risk_level: str  # "low", "medium", "high", "critical"
    timestamp: datetime


class LiquidationRiskMonitor:
    """Monitor liquidation risk for leveraged positions.

    Continuously tracks position liquidation prices and issues alerts when
    positions approach liquidation thresholds.

    Args:
        warning_threshold_pct: Distance from liquidation to trigger warning (%)
        critical_threshold_pct: Distance from liquidation to trigger critical alert (%)
        check_interval_seconds: How often to check positions (seconds)
    """

    # Risk thresholds
    CRITICAL_THRESHOLD = 2.0  # 2% from liquidation
    HIGH_THRESHOLD = 5.0  # 5% from liquidation
    MEDIUM_THRESHOLD = 10.0  # 10% from liquidation

    def __init__(
        self,
        warning_threshold_pct: float = 10.0,
        critical_threshold_pct: float = 2.0,
        check_interval_seconds: int = 10,
    ):
        self.warning_threshold_pct = warning_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.check_interval_seconds = check_interval_seconds

        self._active = False
        self._alert_callbacks: List[Callable] = []
        self._last_alerts: Dict[str, LiquidationRiskAlert] = {}
        self._monitor_task: Optional[asyncio.Task] = None

    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback function to be called when liquidation risk is detected.

        Args:
            callback: Async function that receives LiquidationRiskAlert
        """
        self._alert_callbacks.append(callback)
        logger.info(f"Registered liquidation alert callback: {callback.__name__}")

    async def start_monitoring(
        self,
        get_positions_func: Callable,
        get_ticker_func: Callable,
    ) -> None:
        """Start continuous monitoring of positions.

        Args:
            get_positions_func: Async function that returns List[Position]
            get_ticker_func: Async function(symbol) that returns Ticker
        """
        if self._active:
            logger.warning("Liquidation monitor already active")
            return

        self._active = True
        logger.info("Starting liquidation risk monitoring")

        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(get_positions_func, get_ticker_func)
        )

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._active = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped liquidation risk monitoring")

    async def _monitoring_loop(
        self,
        get_positions_func: Callable,
        get_ticker_func: Callable,
    ) -> None:
        """Main monitoring loop."""
        while self._active:
            try:
                # Get current positions
                positions = await get_positions_func()

                if not positions:
                    await asyncio.sleep(self.check_interval_seconds)
                    continue

                # Check each position for liquidation risk
                for position in positions:
                    try:
                        await self._check_position_risk(
                            position, get_ticker_func
                        )
                    except Exception as exc:
                        logger.error(
                            f"Error checking liquidation risk for {position.symbol}: {exc}"
                        )

                await asyncio.sleep(self.check_interval_seconds)

            except Exception as exc:
                logger.error(f"Error in liquidation monitoring loop: {exc}")
                await asyncio.sleep(5.0)

    async def _check_position_risk(
        self,
        position: Position,
        get_ticker_func: Callable,
    ) -> None:
        """Check liquidation risk for a single position."""
        # Skip if no liquidation price (spot trading)
        if position.liquidation_price == 0.0:
            return

        # Get current price
        try:
            ticker = await get_ticker_func(position.symbol)
            current_price = ticker.last or position.current_price
        except Exception as exc:
            logger.debug(f"Could not fetch ticker for {position.symbol}: {exc}")
            current_price = position.current_price

        if current_price == 0.0:
            return

        # Calculate distance to liquidation
        liquidation_price = position.liquidation_price
        distance_pct = self._calculate_liquidation_distance(
            current_price, liquidation_price, position.side
        )

        # Determine risk level
        risk_level = self._determine_risk_level(distance_pct)

        # Only alert if risk level is medium or higher
        if risk_level != "low":
            alert = LiquidationRiskAlert(
                symbol=position.symbol,
                liquidation_price=liquidation_price,
                current_price=current_price,
                distance_pct=distance_pct,
                margin_ratio=position.margin_ratio,
                risk_level=risk_level,
                timestamp=datetime.utcnow(),
            )

            # Check if this is a new alert or risk level has increased
            should_alert = self._should_send_alert(position.symbol, alert)

            if should_alert:
                logger.warning(
                    f"LIQUIDATION RISK {risk_level.upper()}: {position.symbol} "
                    f"@ {current_price:.2f}, liquidation @ {liquidation_price:.2f} "
                    f"({distance_pct:.2f}% distance), margin ratio: {position.margin_ratio:.2f}%"
                )

                # Store alert
                self._last_alerts[position.symbol] = alert

                # Trigger callbacks
                for callback in self._alert_callbacks:
                    try:
                        await callback(alert)
                    except Exception as exc:
                        logger.error(
                            f"Error in liquidation alert callback {callback.__name__}: {exc}"
                        )

    def _calculate_liquidation_distance(
        self,
        current_price: float,
        liquidation_price: float,
        side: PositionSide,
    ) -> float:
        """Calculate percentage distance from current price to liquidation price.

        Returns:
            Positive percentage representing distance to liquidation
        """
        if liquidation_price == 0.0 or current_price == 0.0:
            return 100.0  # Safe default

        if side == PositionSide.LONG:
            # For longs, liquidation is below current price
            # Distance = (current - liq) / current * 100
            distance_pct = ((current_price - liquidation_price) / current_price) * 100
        else:
            # For shorts, liquidation is above current price
            # Distance = (liq - current) / liq * 100
            distance_pct = ((liquidation_price - current_price) / liquidation_price) * 100

        return max(0.0, distance_pct)

    def _determine_risk_level(self, distance_pct: float) -> str:
        """Determine risk level based on distance to liquidation."""
        if distance_pct <= self.CRITICAL_THRESHOLD:
            return "critical"
        elif distance_pct <= self.HIGH_THRESHOLD:
            return "high"
        elif distance_pct <= self.MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "low"

    def _should_send_alert(
        self, symbol: str, new_alert: LiquidationRiskAlert
    ) -> bool:
        """Determine if we should send an alert for this position.

        Avoids spamming by only alerting when:
        - First time seeing this symbol at risk
        - Risk level has increased
        - At least 1 minute since last alert for same risk level
        """
        if symbol not in self._last_alerts:
            return True

        last_alert = self._last_alerts[symbol]

        # Alert if risk level increased
        risk_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if risk_levels[new_alert.risk_level] > risk_levels[last_alert.risk_level]:
            return True

        # Alert if it's been more than 1 minute since last alert
        time_since_last = (new_alert.timestamp - last_alert.timestamp).total_seconds()
        if time_since_last > 60:
            return True

        return False

    def calculate_safe_leverage(
        self,
        current_price: float,
        stop_loss_price: float,
        position_side: PositionSide,
        liquidation_buffer_pct: float = 20.0,
    ) -> float:
        """Calculate maximum safe leverage given a stop loss.

        Ensures liquidation price is significantly beyond stop loss price.

        Args:
            current_price: Entry price
            stop_loss_price: Stop loss price
            position_side: Long or short
            liquidation_buffer_pct: Buffer between SL and liquidation (%)

        Returns:
            Maximum safe leverage multiplier
        """
        if current_price == 0.0 or stop_loss_price == 0.0:
            return 1.0

        # Calculate stop loss distance as percentage
        if position_side == PositionSide.LONG:
            sl_distance_pct = ((current_price - stop_loss_price) / current_price) * 100
        else:
            sl_distance_pct = ((stop_loss_price - current_price) / current_price) * 100

        if sl_distance_pct <= 0:
            return 1.0  # Invalid stop loss

        # Liquidation should be buffer% beyond stop loss
        # For long: liq_distance = sl_distance + buffer
        liquidation_distance_pct = sl_distance_pct + liquidation_buffer_pct

        # Calculate safe leverage
        # Leverage = 1 / (liquidation_distance_pct / 100)
        # But account for maintenance margin (typically 0.5% for crypto)
        maintenance_margin_pct = 0.5

        safe_leverage = (100 / liquidation_distance_pct) * (1 - maintenance_margin_pct / 100)
        safe_leverage = max(1.0, safe_leverage)  # At least 1x

        logger.debug(
            f"Safe leverage calculation: SL distance {sl_distance_pct:.2f}%, "
            f"safe leverage {safe_leverage:.2f}x"
        )

        return safe_leverage

    def estimate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        position_side: PositionSide,
        maintenance_margin_rate: float = 0.005,  # 0.5% typical for crypto
    ) -> float:
        """Estimate liquidation price for a position.

        Args:
            entry_price: Position entry price
            leverage: Leverage multiplier
            position_side: Long or short
            maintenance_margin_rate: Exchange maintenance margin rate

        Returns:
            Estimated liquidation price
        """
        if entry_price == 0.0 or leverage <= 0:
            return 0.0

        # Liquidation price formula:
        # For long: liq = entry * (1 - (1/leverage) + maintenance_margin)
        # For short: liq = entry * (1 + (1/leverage) - maintenance_margin)

        if position_side == PositionSide.LONG:
            liq_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
        else:
            liq_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate)

        return max(0.0, liq_price)

    def get_current_risk_summary(self) -> Dict[str, any]:
        """Get summary of current liquidation risks.

        Returns:
            Dict with risk statistics
        """
        if not self._last_alerts:
            return {
                "total_positions_at_risk": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
            }

        risk_counts = {"critical": 0, "high": 0, "medium": 0}

        for alert in self._last_alerts.values():
            if alert.risk_level in risk_counts:
                risk_counts[alert.risk_level] += 1

        return {
            "total_positions_at_risk": len(self._last_alerts),
            "critical_count": risk_counts["critical"],
            "high_count": risk_counts["high"],
            "medium_count": risk_counts["medium"],
            "alerts": [
                {
                    "symbol": alert.symbol,
                    "risk_level": alert.risk_level,
                    "distance_pct": alert.distance_pct,
                    "current_price": alert.current_price,
                    "liquidation_price": alert.liquidation_price,
                }
                for alert in sorted(
                    self._last_alerts.values(),
                    key=lambda x: x.distance_pct,
                )
            ],
        }
