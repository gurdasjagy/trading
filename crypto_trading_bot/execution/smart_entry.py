"""Smart entry optimizer — limit orders, TWAP, VWAP tracking, and order book imbalance."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class SmartEntryOptimizer:
    """Optimises trade entries using limit orders, TWAP, and VWAP tracking.

    Strategy
    --------
    * For **longs**: place a limit order at the lower Bollinger Band or
      nearest support level; fall back to a market order after a timeout.
    * For **shorts**: place a limit order at the upper Bollinger Band or
      nearest resistance level.
    * For **large orders**: split into 3–5 chunks spread over 2–5 minutes
      (TWAP execution).
    * **VWAP filter**: only enter when price is favourable relative to VWAP.
    * **Order book imbalance**: prefer entries when bid/ask imbalance
      confirms the intended direction.
    """

    # Seconds to wait for a limit order fill before falling back to market
    _LIMIT_TIMEOUT_SECONDS: float = 60.0
    # Minimum chunks for TWAP orders
    _TWAP_MIN_CHUNKS: int = 3
    _TWAP_MAX_CHUNKS: int = 5
    # TWAP total duration (seconds)
    _TWAP_DURATION_SECONDS: float = 300.0  # 5 minutes
    # Imbalance ratio threshold for confirmation
    _IMBALANCE_THRESHOLD: float = 3.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_limit_price(
        self,
        direction: str,
        current_price: float,
        bollinger_lower: Optional[float] = None,
        bollinger_upper: Optional[float] = None,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
    ) -> float:
        """Return the optimal limit order price for an entry.

        For longs the price is the lower Bollinger Band or nearest support
        below current price (whichever is higher / less risky).  For shorts
        it is the upper Bollinger Band or nearest resistance above current
        price.

        Args:
            direction: ``"long"`` or ``"short"``.
            current_price: Latest market price.
            bollinger_lower: Lower Bollinger Band price.
            bollinger_upper: Upper Bollinger Band price.
            support_levels: Sorted list of key support levels.
            resistance_levels: Sorted list of key resistance levels.

        Returns:
            Recommended limit order price.
        """
        if direction == "long":
            candidates: List[float] = []
            if bollinger_lower and bollinger_lower < current_price:
                candidates.append(bollinger_lower)
            if support_levels:
                below = [lvl for lvl in support_levels if lvl < current_price]
                if below:
                    candidates.append(max(below))  # nearest support below
            if candidates:
                limit_price = max(candidates)
            else:
                # Fallback: 0.5 % below current price
                limit_price = current_price * 0.995
        else:
            candidates = []
            if bollinger_upper and bollinger_upper > current_price:
                candidates.append(bollinger_upper)
            if resistance_levels:
                above = [lvl for lvl in resistance_levels if lvl > current_price]
                if above:
                    candidates.append(min(above))  # nearest resistance above
            if candidates:
                limit_price = min(candidates)
            else:
                # Fallback: 0.5 % above current price
                limit_price = current_price * 1.005

        logger.debug(
            "[SmartEntry] limit_price={:.4f} dir={} current={:.4f}",
            limit_price,
            direction,
            current_price,
        )
        return round(limit_price, 8)

    def calculate_twap_schedule(
        self,
        total_size: float,
        direction: str,
        n_chunks: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Build a TWAP execution schedule.

        Args:
            total_size: Total order size (base units).
            direction: ``"long"`` or ``"short"``.
            n_chunks: Number of chunks (default: adaptive 3–5).
            duration_seconds: Total execution window in seconds.

        Returns:
            List of ``(delay_seconds, chunk_size)`` tuples.  The first
            chunk has ``delay=0``.
        """
        n = n_chunks if n_chunks else self._adaptive_chunk_count(total_size)
        n = max(self._TWAP_MIN_CHUNKS, min(self._TWAP_MAX_CHUNKS, n))
        duration = duration_seconds or self._TWAP_DURATION_SECONDS
        interval = duration / (n - 1) if n > 1 else 0.0

        chunk_size = round(total_size / n, 8)
        remainder = round(total_size - chunk_size * (n - 1), 8)

        schedule: List[Tuple[float, float]] = []
        for i in range(n):
            delay = round(i * interval, 1)
            size = remainder if i == n - 1 else chunk_size
            schedule.append((delay, size))

        logger.debug(
            "[SmartEntry] TWAP schedule: {} chunks over {:.0f}s dir={}",
            n,
            duration,
            direction,
        )
        return schedule

    def is_vwap_favorable(
        self,
        direction: str,
        current_price: float,
        vwap: float,
    ) -> bool:
        """Return True when price is on the favourable side of VWAP.

        * Long: price should be *below* VWAP (buying discount).
        * Short: price should be *above* VWAP (selling premium).

        A small tolerance (0.1 %) is allowed so entries near VWAP are not
        rejected.

        Args:
            direction: ``"long"`` or ``"short"``.
            current_price: Latest market price.
            vwap: Volume-Weighted Average Price for the session.

        Returns:
            ``True`` if the entry is VWAP-favourable.
        """
        if vwap <= 0:
            return True  # no VWAP data available — don't block the trade
        tolerance = vwap * 0.001
        if direction == "long":
            return current_price <= vwap + tolerance
        else:
            return current_price >= vwap - tolerance

    def calculate_vwap(self, ohlcv: "Any") -> float:
        """Calculate VWAP from an OHLCV DataFrame.

        Args:
            ohlcv: pandas DataFrame with columns ``high``, ``low``,
                ``close``, ``volume``.

        Returns:
            VWAP value, or 0.0 if data is insufficient.
        """
        try:
            import pandas as pd

            df = ohlcv
            if df is None or (hasattr(df, "empty") and df.empty):
                return 0.0
            typical_price = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
            vwap = (typical_price * df["volume"].astype(float)).sum() / df["volume"].astype(float).sum()
            return float(vwap)
        except Exception as exc:
            logger.warning("[SmartEntry] VWAP calculation failed: {}", exc)
            return 0.0

    def validate_orderbook_depth(
        self,
        orderbook: Dict[str, Any],
        required_notional: float,
        side: str,
    ) -> Tuple[bool, str]:
        """Check if the order book has sufficient depth to absorb the trade.

        Ensures that there is at least 3× the required notional value of
        cumulative depth in the top 10 levels on the relevant side of the
        book.  If the depth is insufficient the trade would likely suffer
        significant market impact slippage.

        Args:
            orderbook: Dict with ``bids`` and ``asks`` lists of
                ``[price, size]`` entries.
            required_notional: Trade notional value in USDT.
            side: ``"buy"`` (check asks) or ``"sell"`` (check bids).

        Returns:
            Tuple of (is_sufficient: bool, reason: str).  When
            ``is_sufficient`` is ``True`` the reason string is empty.
        """
        book_side = orderbook.get("asks" if side == "buy" else "bids", [])
        cumulative = 0.0
        for level in book_side[:10]:
            price = float(level[0])
            size = float(level[1])
            cumulative += price * size
            if cumulative >= required_notional * 3:  # Need 3× depth
                return True, ""
        return (
            False,
            f"Insufficient depth: {cumulative:.0f} USDT available, "
            f"need {required_notional * 3:.0f}",
        )

    def check_order_book_imbalance(
        self,
        orderbook: Dict[str, Any],
        direction: str,
    ) -> Tuple[bool, float]:
        """Check whether order book imbalance confirms the entry direction.

        A ratio > ``_IMBALANCE_THRESHOLD`` : 1 (bid vs ask, or vice versa)
        constitutes strong directional pressure.

        Args:
            orderbook: Dict with ``bids`` and ``asks`` lists of [price, size].
            direction: ``"long"`` or ``"short"``.

        Returns:
            Tuple of (confirmed: bool, imbalance_ratio: float).
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        bid_vol = sum(float(r[1]) for r in bids)
        ask_vol = sum(float(r[1]) for r in asks)

        if ask_vol == 0 and bid_vol == 0:
            return True, 1.0  # no data — don't block

        if ask_vol == 0:
            ratio = float("inf")
        else:
            ratio = bid_vol / ask_vol if bid_vol > ask_vol else ask_vol / bid_vol

        if direction == "long":
            confirmed = bid_vol >= ask_vol * (1 / self._IMBALANCE_THRESHOLD)
        else:
            confirmed = ask_vol >= bid_vol * (1 / self._IMBALANCE_THRESHOLD)

        logger.debug(
            "[SmartEntry] imbalance ratio={:.2f} dir={} confirmed={}",
            ratio,
            direction,
            confirmed,
        )
        return confirmed, round(ratio, 3)

    def get_entry_recommendation(
        self,
        direction: str,
        current_price: float,
        total_size: float,
        bollinger_lower: Optional[float] = None,
        bollinger_upper: Optional[float] = None,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
        vwap: Optional[float] = None,
        orderbook: Optional[Dict[str, Any]] = None,
        is_large_order: bool = False,
    ) -> Dict[str, Any]:
        """Build a complete entry recommendation.

        Combines limit price calculation, VWAP check, order book imbalance,
        and optionally a TWAP schedule for large orders.

        Args:
            direction: ``"long"`` or ``"short"``.
            current_price: Latest market price.
            total_size: Order size in base units.
            bollinger_lower: Lower Bollinger Band.
            bollinger_upper: Upper Bollinger Band.
            support_levels: Key support price levels.
            resistance_levels: Key resistance price levels.
            vwap: Session VWAP.
            orderbook: Order book snapshot dict.
            is_large_order: When True, build a TWAP schedule.

        Returns:
            Dict with keys:
            * ``order_type``: ``"limit"`` or ``"market"``
            * ``limit_price``: Recommended limit price (0 for market orders)
            * ``use_twap``: Whether to use TWAP execution
            * ``twap_schedule``: List of (delay_s, size) tuples when TWAP is used
            * ``vwap_favorable``: Whether VWAP filter passed
            * ``imbalance_confirmed``: Whether order book confirms direction
            * ``imbalance_ratio``: Bid/ask ratio
            * ``limit_timeout_seconds``: Time to wait before falling back
        """
        limit_price = self.calculate_limit_price(
            direction=direction,
            current_price=current_price,
            bollinger_lower=bollinger_lower,
            bollinger_upper=bollinger_upper,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
        )

        vwap_ok = self.is_vwap_favorable(direction, current_price, vwap or 0.0)

        imbalance_ok, imbalance_ratio = (True, 1.0)
        if orderbook:
            imbalance_ok, imbalance_ratio = self.check_order_book_imbalance(orderbook, direction)

        use_twap = is_large_order
        twap_schedule: List[Tuple[float, float]] = []
        if use_twap:
            twap_schedule = self.calculate_twap_schedule(total_size, direction)

        return {
            "order_type": "limit",
            "limit_price": limit_price,
            "use_twap": use_twap,
            "twap_schedule": twap_schedule,
            "vwap_favorable": vwap_ok,
            "imbalance_confirmed": imbalance_ok,
            "imbalance_ratio": imbalance_ratio,
            "limit_timeout_seconds": self._LIMIT_TIMEOUT_SECONDS,
        }

    def calculate_entry_delay(
        self,
        direction: str,
        orderbook: Dict[str, Any],
        max_delay_seconds: float = 5.0,
    ) -> float:
        """Calculate how long to delay entry based on order book imbalance.

        If the imbalance is heavily against the trade direction (e.g., 5x more
        asks than bids for a long entry), delay the entry to anticipate a
        better fill price as the imbalance resolves.

        Args:
            direction: "long" or "short"
            orderbook: Order book with bids/asks
            max_delay_seconds: Maximum delay

        Returns:
            Recommended delay in seconds (0 = enter immediately)
        """
        bids = orderbook.get("bids", [])[:10]  # Top 10 levels
        asks = orderbook.get("asks", [])[:10]

        bid_vol = sum(float(r[1]) for r in bids) if bids else 0
        ask_vol = sum(float(r[1]) for r in asks) if asks else 0

        if bid_vol == 0 and ask_vol == 0:
            return 0.0

        imbalance_ratio = 1.0
        if direction == "long":
            # For longs, heavy ask pressure (sellers) means price may drop - delay
            if ask_vol > 0 and bid_vol > 0:
                imbalance_ratio = ask_vol / bid_vol
            else:
                imbalance_ratio = 1.0

            if imbalance_ratio >= 5.0:
                delay = max_delay_seconds
            elif imbalance_ratio >= 3.0:
                delay = max_delay_seconds * 0.6
            elif imbalance_ratio >= 2.0:
                delay = max_delay_seconds * 0.3
            else:
                delay = 0.0
        else:
            # For shorts, heavy bid pressure (buyers) means price may rise - delay
            if bid_vol > 0 and ask_vol > 0:
                imbalance_ratio = bid_vol / ask_vol
            else:
                imbalance_ratio = 1.0

            if imbalance_ratio >= 5.0:
                delay = max_delay_seconds
            elif imbalance_ratio >= 3.0:
                delay = max_delay_seconds * 0.6
            elif imbalance_ratio >= 2.0:
                delay = max_delay_seconds * 0.3
            else:
                delay = 0.0

        if delay > 0:
            logger.info(
                "[SmartEntry] Delaying {} entry by {:.1f}s due to adverse imbalance ratio {:.1f}x",
                direction, delay, imbalance_ratio
            )

        return delay

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adaptive_chunk_count(self, total_size: float) -> int:
        """Return an adaptive number of TWAP chunks based on order size."""
        # Simple heuristic: larger size → more chunks
        if total_size <= 0.01:
            return self._TWAP_MIN_CHUNKS
        elif total_size <= 0.1:
            return 4
        else:
            return self._TWAP_MAX_CHUNKS
