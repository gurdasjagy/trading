"""Anti-gaming protection - detect and protect against exchange manipulation."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange


@dataclass
class SpoofingEvent:
    """Detected spoofing event."""

    timestamp: float
    symbol: str
    side: str  # "bid" or "ask"
    price: float
    volume: float
    appeared_disappeared_count: int


@dataclass
class ManipulationAlert:
    """Alert for potential market manipulation."""

    timestamp: float
    symbol: str
    alert_type: str  # "spoofing", "wash_trading", "price_manipulation"
    severity: str  # "low", "medium", "high"
    description: str
    recommended_action: str


class AntiGamingProtection:
    """Protect against exchange manipulation and adversarial behavior.

    Protection Mechanisms:
    1. Spoofing Detection - Large orders that appear/disappear rapidly
    2. Front-running Protection - Randomize order timing
    3. Wash Trading Detection - Suspicious volume patterns
    4. Price Manipulation Detection - Artificial spikes near SL levels
    5. Order Timing Jitter - Add random delays to avoid pattern detection

    When manipulation detected:
    - Delay execution
    - Widen stop loss temporarily
    - Alert operator
    - Log event for analysis
    """

    # Detection thresholds
    SPOOFING_MIN_SIZE_USDT = 50000.0  # Minimum order size to track
    SPOOFING_APPEARANCE_THRESHOLD = 3  # Appear/disappear 3+ times
    SPOOFING_WINDOW_SECONDS = 60.0  # Detection window

    WASH_TRADING_VOLUME_SPIKE = 5.0  # 5x normal volume
    WASH_TRADING_SPREAD_TIGHTNESS = 0.001  # 0.1% spread

    PRICE_MANIP_SPIKE_THRESHOLD = 0.02  # 2% sudden move
    PRICE_MANIP_REVERSION_THRESHOLD = 0.015  # 1.5% reversion

    # Front-running protection
    TIMING_JITTER_MIN_SECONDS = 0.0
    TIMING_JITTER_MAX_SECONDS = 3.0

    # Cooldown after manipulation detected
    MANIPULATION_COOLDOWN_SECONDS = 300.0  # 5 minutes

    def __init__(self, exchange: BaseExchange):
        """Initialize anti-gaming protection.

        Args:
            exchange: Exchange interface
        """
        self._exchange = exchange

        # Spoofing detection state
        self._orderbook_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=60)  # 60 seconds of history
        )
        self._large_orders: Dict[str, Dict] = defaultdict(dict)  # symbol -> {price: count}
        self._spoofing_events: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )

        # Wash trading detection
        self._volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )

        # Price manipulation detection
        self._price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=120)  # 2 minutes
        )

        # Manipulation alerts
        self._alerts: deque = deque(maxlen=100)

        # Cooldown tracking
        self._symbol_cooldowns: Dict[str, float] = {}

        logger.info("AntiGamingProtection initialized")

    async def check_execution_safety(
        self,
        symbol: str,
        side: str,
        stop_loss: Optional[float] = None
    ) -> Tuple[bool, Optional[ManipulationAlert]]:
        """Check if execution is safe from manipulation.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            stop_loss: Stop loss price (to check for manipulation near SL)

        Returns:
            Tuple of (is_safe, alert_if_unsafe)
        """
        # Check cooldown
        if symbol in self._symbol_cooldowns:
            cooldown_until = self._symbol_cooldowns[symbol]
            if time.time() < cooldown_until:
                return False, ManipulationAlert(
                    timestamp=time.time(),
                    symbol=symbol,
                    alert_type="cooldown",
                    severity="medium",
                    description=f"Symbol in cooldown until {cooldown_until - time.time():.0f}s",
                    recommended_action="wait"
                )

        # Run all checks in parallel
        spoofing_task = asyncio.create_task(self._check_spoofing(symbol))
        wash_trading_task = asyncio.create_task(self._check_wash_trading(symbol))
        price_manip_task = asyncio.create_task(self._check_price_manipulation(symbol, stop_loss))

        results = await asyncio.gather(
            spoofing_task,
            wash_trading_task,
            price_manip_task,
            return_exceptions=True
        )

        # Check for any alerts
        for result in results:
            if isinstance(result, ManipulationAlert):
                self._store_alert(result)
                if result.severity in ["medium", "high"]:
                    # Set cooldown
                    self._symbol_cooldowns[symbol] = time.time() + self.MANIPULATION_COOLDOWN_SECONDS
                    return False, result

        return True, None

    async def _check_spoofing(self, symbol: str) -> Optional[ManipulationAlert]:
        """Detect spoofing (large orders appearing/disappearing rapidly).

        Args:
            symbol: Trading symbol

        Returns:
            ManipulationAlert if spoofing detected, None otherwise
        """
        try:
            orderbook = await self._exchange.get_orderbook(symbol, limit=20)
            if not orderbook:
                return None

            current_time = time.time()

            # Track large orders
            large_bids = [
                (bid[0], bid[1] * bid[0])
                for bid in orderbook.get('bids', [])
                if bid[1] * bid[0] > self.SPOOFING_MIN_SIZE_USDT
            ]

            large_asks = [
                (ask[0], ask[1] * ask[0])
                for ask in orderbook.get('asks', [])
                if ask[1] * ask[0] > self.SPOOFING_MIN_SIZE_USDT
            ]

            # Store orderbook snapshot
            self._orderbook_history[symbol].append({
                'timestamp': current_time,
                'large_bids': large_bids,
                'large_asks': large_asks
            })

            # Need at least 10 seconds of history
            if len(self._orderbook_history[symbol]) < 10:
                return None

            # Check for orders appearing/disappearing
            recent_snapshots = list(self._orderbook_history[symbol])[-10:]

            # Track order appearances at each price level
            price_appearances = defaultdict(lambda: {'count': 0, 'last_seen': 0.0})

            for snapshot in recent_snapshots:
                for price, volume in snapshot['large_bids'] + snapshot['large_asks']:
                    price_key = f"{price:.2f}"
                    if current_time - price_appearances[price_key]['last_seen'] > 5.0:
                        # Order reappeared after disappearing
                        price_appearances[price_key]['count'] += 1
                    price_appearances[price_key]['last_seen'] = current_time

            # Check for spoofing pattern
            for price_key, data in price_appearances.items():
                if data['count'] >= self.SPOOFING_APPEARANCE_THRESHOLD:
                    return ManipulationAlert(
                        timestamp=current_time,
                        symbol=symbol,
                        alert_type="spoofing",
                        severity="high",
                        description=f"Large order at {price_key} appeared/disappeared {data['count']} times",
                        recommended_action="delay_execution"
                    )

        except Exception as e:
            logger.debug("Spoofing check failed: {}", e)

        return None

    async def _check_wash_trading(self, symbol: str) -> Optional[ManipulationAlert]:
        """Detect wash trading (suspicious volume patterns).

        Wash trading indicators:
        - Sudden volume spike with tight spreads
        - Volume >> normal with no price movement

        Args:
            symbol: Trading symbol

        Returns:
            ManipulationAlert if wash trading suspected, None otherwise
        """
        try:
            ticker = await self._exchange.get_ticker(symbol)
            if not ticker:
                return None

            current_time = time.time()
            current_volume = ticker.volume_24h if hasattr(ticker, 'volume_24h') else 0.0

            # Store volume
            self._volume_history[symbol].append({
                'timestamp': current_time,
                'volume': current_volume
            })

            # Need history for comparison
            if len(self._volume_history[symbol]) < 30:
                return None

            # Calculate average volume (excluding current)
            recent_volumes = [
                v['volume']
                for v in list(self._volume_history[symbol])[:-1]
            ]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0

            # Check for volume spike
            if avg_volume > 0 and current_volume > avg_volume * self.WASH_TRADING_VOLUME_SPIKE:
                # Check spread tightness
                spread_pct = (ticker.ask - ticker.bid) / ticker.last if ticker.bid > 0 else 0.01

                if spread_pct < self.WASH_TRADING_SPREAD_TIGHTNESS:
                    return ManipulationAlert(
                        timestamp=current_time,
                        symbol=symbol,
                        alert_type="wash_trading",
                        severity="medium",
                        description=f"Volume spike {current_volume/avg_volume:.1f}x with tight spread {spread_pct:.3%}",
                        recommended_action="monitor"
                    )

        except Exception as e:
            logger.debug("Wash trading check failed: {}", e)

        return None

    async def _check_price_manipulation(
        self,
        symbol: str,
        stop_loss: Optional[float] = None
    ) -> Optional[ManipulationAlert]:
        """Detect price manipulation (artificial spikes near SL levels).

        Price manipulation indicators:
        - Sudden spike followed by quick reversion
        - Price action targeting stop loss levels

        Args:
            symbol: Trading symbol
            stop_loss: Stop loss price to monitor

        Returns:
            ManipulationAlert if manipulation suspected, None otherwise
        """
        try:
            ticker = await self._exchange.get_ticker(symbol)
            if not ticker:
                return None

            current_time = time.time()
            current_price = ticker.last

            # Store price
            self._price_history[symbol].append({
                'timestamp': current_time,
                'price': current_price
            })

            # Need history for spike detection
            if len(self._price_history[symbol]) < 20:
                return None

            recent_prices = list(self._price_history[symbol])

            # Check for spike and reversion pattern
            # Look for: price spike → quick reversion within 2 minutes
            for i in range(len(recent_prices) - 10, len(recent_prices)):
                if i < 2:
                    continue

                base_price = recent_prices[i - 2]['price']
                spike_price = recent_prices[i]['price']
                current_price_check = recent_prices[-1]['price']

                # Calculate spike and reversion
                spike_pct = abs(spike_price - base_price) / base_price
                reversion_pct = abs(current_price_check - spike_price) / spike_price

                # Check for manipulation pattern
                if (spike_pct > self.PRICE_MANIP_SPIKE_THRESHOLD and
                    reversion_pct > self.PRICE_MANIP_REVERSION_THRESHOLD):

                    # Extra suspicious if near stop loss
                    near_stop_loss = False
                    if stop_loss:
                        distance_to_sl = abs(spike_price - stop_loss) / stop_loss
                        if distance_to_sl < 0.005:  # Within 0.5% of SL
                            near_stop_loss = True

                    severity = "high" if near_stop_loss else "medium"
                    description = f"Price spike {spike_pct:.1%} followed by reversion {reversion_pct:.1%}"
                    if near_stop_loss:
                        description += f" near stop loss {stop_loss}"

                    return ManipulationAlert(
                        timestamp=current_time,
                        symbol=symbol,
                        alert_type="price_manipulation",
                        severity=severity,
                        description=description,
                        recommended_action="widen_stop_loss" if near_stop_loss else "delay_execution"
                    )

        except Exception as e:
            logger.debug("Price manipulation check failed: {}", e)

        return None

    def get_execution_delay(self) -> float:
        """Get randomized execution delay for front-running protection.

        Returns:
            Delay in seconds (0 to TIMING_JITTER_MAX_SECONDS)
        """
        import random
        delay = random.uniform(self.TIMING_JITTER_MIN_SECONDS, self.TIMING_JITTER_MAX_SECONDS)
        logger.debug("Front-running protection: adding {:.2f}s jitter", delay)
        return delay

    def get_widened_stop_loss(
        self,
        original_stop_loss: float,
        current_price: float,
        side: str
    ) -> float:
        """Get temporarily widened stop loss to avoid manipulation.

        Widen SL by 0.5% to avoid stop hunting.

        Args:
            original_stop_loss: Original SL price
            current_price: Current market price
            side: "long" or "short"

        Returns:
            Widened stop loss price
        """
        distance = abs(current_price - original_stop_loss)
        widening_factor = 1.5  # Widen by 50%

        if side == "long":
            widened_sl = current_price - (distance * widening_factor)
        else:
            widened_sl = current_price + (distance * widening_factor)

        logger.info(
            "Widening stop loss from {:.2f} to {:.2f} for manipulation protection",
            original_stop_loss,
            widened_sl
        )

        return widened_sl

    def _store_alert(self, alert: ManipulationAlert) -> None:
        """Store manipulation alert.

        Args:
            alert: ManipulationAlert to store
        """
        self._alerts.append(alert)

        logger.warning(
            "Manipulation alert: {} {} [{}] - {}",
            alert.symbol,
            alert.alert_type,
            alert.severity,
            alert.description
        )

    def get_recent_alerts(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[ManipulationAlert]:
        """Get recent manipulation alerts.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of alerts

        Returns:
            List of recent alerts
        """
        alerts = list(self._alerts)

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]

        return alerts[-limit:]

    def get_protection_summary(self, symbol: str) -> Dict:
        """Get protection summary for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with protection status
        """
        # Check cooldown
        in_cooldown = False
        cooldown_remaining = 0.0
        if symbol in self._symbol_cooldowns:
            cooldown_until = self._symbol_cooldowns[symbol]
            if time.time() < cooldown_until:
                in_cooldown = True
                cooldown_remaining = cooldown_until - time.time()

        # Count recent alerts
        recent_alerts = self.get_recent_alerts(symbol, limit=5)
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.alert_type] += 1

        return {
            "symbol": symbol,
            "in_cooldown": in_cooldown,
            "cooldown_remaining_seconds": cooldown_remaining,
            "recent_alerts": len(recent_alerts),
            "alert_breakdown": dict(alert_counts),
            "recommended_delay_seconds": self.get_execution_delay(),
            "status": "unsafe" if in_cooldown else "safe"
        }

    def clear_cooldown(self, symbol: str) -> None:
        """Clear cooldown for symbol (manual override).

        Args:
            symbol: Trading symbol
        """
        if symbol in self._symbol_cooldowns:
            del self._symbol_cooldowns[symbol]
            logger.info("Cooldown cleared for {}", symbol)
