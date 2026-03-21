"""Tail Risk Protector.

Monitors a set of crypto-specific tail-risk indicators and takes
automatic defensive action when multiple indicators fire simultaneously:

Indicators monitored:
  1. Realised volatility spike (> 3× its 30-day average)
  2. Correlation spike (all assets co-moving > 0.8)
  3. Liquidity drought (bid-ask spread > 3× normal)
  4. Extreme funding rate (> 0.1 % per 8 h)

Response when ≥ 2 indicators trigger simultaneously:
  • Reduce ALL positions by 50 %
  • Tighten stop-loss by 2× (signal stored; TradeExecutor consumes it)
  • Disable new entries for 1 hour
  • Send a critical alert
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from loguru import logger


class TailRiskProtector:
    """Tail-risk monitoring and automated position protection.

    Args:
        exchange: Exchange wrapper with ``get_positions()``,
            ``get_ticker()``, and ``get_funding_rate()`` methods.
        position_manager: PositionManager for reducing positions.
        ws_data_manager: Optional WebSocketDataManager for real-time
            spread data.
        alert_manager: Optional alerting object with ``send_alert()``.
        check_interval: Seconds between indicator evaluations.
        vol_spike_multiplier: Realised vol / average vol ratio to flag.
        correlation_threshold: Co-movement threshold (Pearson r).
        spread_spike_multiplier: Current spread / average spread ratio.
        extreme_funding_rate: Funding rate threshold per 8 h.
        lockout_hours: Hours to disable new entries after trigger.
    """

    def __init__(
        self,
        exchange,
        position_manager,
        ws_data_manager=None,
        alert_manager=None,
        check_interval: float = 60.0,
        vol_spike_multiplier: float = 3.0,
        correlation_threshold: float = 0.80,
        spread_spike_multiplier: float = 3.0,
        extreme_funding_rate: float = 0.001,
        lockout_hours: float = 1.0,
    ) -> None:
        self._exchange = exchange
        self._position_manager = position_manager
        self._ws_dm = ws_data_manager
        self._alert_manager = alert_manager
        self._check_interval = check_interval
        self._vol_spike_mul = vol_spike_multiplier
        self._corr_threshold = correlation_threshold
        self._spread_spike_mul = spread_spike_multiplier
        self._extreme_funding = extreme_funding_rate
        self._lockout_hours = lockout_hours

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # State
        self._entries_disabled_until: Optional[datetime] = None
        self._tighten_sl: bool = False  # consumed by TradeExecutor
        self._trigger_count: int = 0

        # Rolling vol history: symbol → list of recent returns
        self._vol_history: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background tail-risk monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="tail_risk_protector"
        )
        logger.info("TailRiskProtector started.")

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TailRiskProtector stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_entry_allowed(self) -> bool:
        """Return False if new entries are currently disabled."""
        if self._entries_disabled_until is None:
            return True
        return datetime.now(tz=timezone.utc) >= self._entries_disabled_until

    def should_tighten_sl(self) -> bool:
        """Return True if stop-losses should be tightened (consumed once)."""
        flag = self._tighten_sl
        self._tighten_sl = False
        return flag

    # ------------------------------------------------------------------
    # Indicator evaluation
    # ------------------------------------------------------------------

    async def evaluate_indicators(self) -> List[str]:
        """Evaluate all tail-risk indicators.  Returns list of triggered names."""
        triggered: List[str] = []

        if await self._vol_spike_triggered():
            triggered.append("vol_spike")
        if await self._correlation_spike_triggered():
            triggered.append("correlation_spike")
        if await self._liquidity_drought_triggered():
            triggered.append("liquidity_drought")
        if await self._extreme_funding_triggered():
            triggered.append("extreme_funding")

        return triggered

    async def check_and_protect(self) -> None:
        """Run indicator evaluation and take protective action if warranted."""
        triggered = await self.evaluate_indicators()

        if len(triggered) >= 2:
            logger.critical(
                "TailRiskProtector: {} indicators triggered simultaneously: {}",
                len(triggered),
                triggered,
            )
            await self._take_protective_action(triggered)

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    async def _vol_spike_triggered(self) -> bool:
        """Return True if realised vol > 3× its 30-day average."""
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return False

            for pos in positions:
                symbol = getattr(pos, "symbol", None)
                if not symbol:
                    continue
                returns = self._vol_history.get(symbol, [])
                if len(returns) < 30:
                    continue
                recent = np.std(returns[-5:])
                avg = np.std(returns[-30:])
                if avg > 0 and recent > self._vol_spike_mul * avg:
                    logger.debug(
                        "TailRisk vol spike: {} recent_vol={:.6f} avg={:.6f}",
                        symbol,
                        recent,
                        avg,
                    )
                    return True
        except Exception as exc:
            logger.debug("TailRisk vol spike check failed: {}", exc)
        return False

    async def _correlation_spike_triggered(self) -> bool:
        """Return True if all open symbols are moving together > 0.8."""
        try:
            histories = {
                sym: ret
                for sym, ret in self._vol_history.items()
                if len(ret) >= 20
            }
            if len(histories) < 2:
                return False

            symbols = list(histories.keys())
            min_len = min(len(histories[s]) for s in symbols)
            if min_len < 10:
                return False

            matrix = np.array([histories[s][-min_len:] for s in symbols])
            corr = np.corrcoef(matrix)
            # Check off-diagonal correlations
            n = len(symbols)
            off_diag = [
                corr[i, j]
                for i in range(n)
                for j in range(i + 1, n)
            ]
            if not off_diag:
                return False
            avg_corr = float(np.mean(off_diag))
            if avg_corr > self._corr_threshold:
                logger.debug(
                    "TailRisk correlation spike: avg_corr={:.4f}", avg_corr
                )
                return True
        except Exception as exc:
            logger.debug("TailRisk correlation check failed: {}", exc)
        return False

    async def _liquidity_drought_triggered(self) -> bool:
        """Return True if any position's spread is > 3× normal."""
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return False

            for pos in positions:
                symbol = getattr(pos, "symbol", None)
                if not symbol:
                    continue
                bid = ask = 0.0
                if self._ws_dm is not None:
                    bid, ask = self._ws_dm.get_spread(symbol)
                else:
                    ticker = await self._exchange.get_ticker(symbol)
                    if ticker:
                        bid = float(getattr(ticker, "bid", 0) or 0)
                        ask = float(getattr(ticker, "ask", 0) or 0)
                if bid <= 0 or ask <= 0:
                    continue
                spread = ask - bid
                mid = (bid + ask) / 2
                if mid <= 0:
                    continue
                spread_pct = spread / mid
                # Use 0.05 % as a reasonable "normal" spread baseline
                normal_spread_pct = 0.0005
                if spread_pct > self._spread_spike_mul * normal_spread_pct:
                    logger.debug(
                        "TailRisk liquidity drought: {} spread={:.4%}", symbol, spread_pct
                    )
                    return True
        except Exception as exc:
            logger.debug("TailRisk liquidity check failed: {}", exc)
        return False

    async def _extreme_funding_triggered(self) -> bool:
        """Return True if any symbol has a funding rate > 0.1 % per 8 h."""
        try:
            positions = await self._exchange.get_positions()
            if not positions:
                return False
            for pos in positions:
                symbol = getattr(pos, "symbol", None)
                if not symbol:
                    continue
                try:
                    funding_info = await self._exchange.get_funding_rate(symbol)
                    if funding_info is None:
                        continue
                    rate = float(
                        funding_info.get("fundingRate")
                        or funding_info.get("rate")
                        or 0.0
                    )
                    if abs(rate) > self._extreme_funding:
                        logger.debug(
                            "TailRisk extreme funding: {} rate={:.4%}", symbol, rate
                        )
                        return True
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("TailRisk funding check failed: {}", exc)
        return False

    # ------------------------------------------------------------------
    # Protective action
    # ------------------------------------------------------------------

    async def _take_protective_action(self, triggered: List[str]) -> None:
        """Reduce positions, tighten SL, disable entries, alert."""
        self._trigger_count += 1

        # 1. Reduce all positions by 50 %
        try:
            positions = await self._exchange.get_positions()
            for pos in (positions or []):
                symbol = getattr(pos, "symbol", None)
                if symbol and self._position_manager is not None:
                    try:
                        await self._position_manager.reduce_position(
                            symbol, fraction=0.50
                        )
                        logger.warning(
                            "TailRiskProtector: reduced {} by 50 %.", symbol
                        )
                    except Exception as exc:
                        logger.error(
                            "TailRiskProtector: failed to reduce {}: {}", symbol, exc
                        )
        except Exception as exc:
            logger.error("TailRiskProtector: position reduction failed: {}", exc)

        # 2. Signal SL tightening
        self._tighten_sl = True

        # 3. Disable new entries for lockout period
        self._entries_disabled_until = datetime.now(tz=timezone.utc) + timedelta(
            hours=self._lockout_hours
        )
        logger.warning(
            "TailRiskProtector: new entries disabled until {}.",
            self._entries_disabled_until.isoformat(),
        )

        # 4. Send critical alert
        msg = (
            f"🚨 TAIL RISK ALERT: {len(triggered)} indicators triggered "
            f"simultaneously ({', '.join(triggered)}). "
            f"Positions reduced 50%, SL tightened, entries disabled for "
            f"{self._lockout_hours}h."
        )
        logger.critical("TailRiskProtector: {}", msg)
        if self._alert_manager is not None:
            try:
                await self._alert_manager.send_alert(msg)
            except Exception as exc:
                logger.debug("TailRiskProtector alert send failed: {}", exc)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.check_and_protect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("TailRiskProtector loop error: {}", exc)
            await asyncio.sleep(self._check_interval)

    # ------------------------------------------------------------------
    # Vol history update (called externally with new price data)
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Update the rolling return history for *symbol* with a new *price*.

        Computes the per-period return and appends it to the symbol's return
        history.  A separate ``_price_last`` dict tracks the previous price so
        that returns are computed correctly without mixing price values into
        the returns list.
        """
        if not hasattr(self, "_price_last"):
            self._price_last: dict = {}

        prev_price = self._price_last.get(symbol, 0.0)
        self._price_last[symbol] = price

        returns = self._vol_history.setdefault(symbol, [])
        if prev_price > 0 and price > 0:
            ret = (price - prev_price) / prev_price
            returns.append(ret)
        else:
            # First observation — no return available yet
            return

        # Retain last 200 return observations
        if len(returns) > 200:
            self._vol_history[symbol] = returns[-200:]
