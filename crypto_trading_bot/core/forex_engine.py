"""Dedicated Forex Trading Engine for gold-focused trading on Gate.io TradFi."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from core.forex_state_manager import ForexStateManager
from core.forex_strategy_manager import ForexStrategyManager
from exchange.exchange_factory import create_exchange
from execution.forex_trade_executor import ForexTradeExecutor
from risk.forex_risk_manager import ForexRiskManager


class ForexTradingEngine:
    """Dedicated forex trading engine for gold-focused trading on Gate.io TradFi.

    Features:

    * 30-second main trading cycle.
    * Session-aware trading (London / NY / Tokyo / Sydney).
    * Weekend / market-hours detection.
    * ForexRiskManager integration for every signal.
    * Spread monitoring — rejects trades with excessive spread.
    * News-event pausing (FOMC, CPI, NFP ±15 min).
    * Multi-timeframe OHLCV fetching (5m, 15m, 1h, 4h).
    * Confluence scoring via ForexStrategyManager.
    * Telegram alerts for trade opens/closes.
    * State persistence via ForexStateManager.
    """

    CYCLE_INTERVAL = 30  # seconds between main trading cycles

    # Timeframes to load for multi-timeframe analysis
    TIMEFRAMES = ["5m", "15m", "1h", "4h"]
    OHLCV_LIMIT = 100

    # Minutes before/after a high-impact news event to pause trading
    NEWS_PROXIMITY_MINUTES = 15

    # State file for engine-level persistence
    STATE_FILE = Path("data/forex_engine_state.json")

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._running = False
        self._exchange: Any = None

        # Sub-components (initialized in start())
        self._risk_manager = ForexRiskManager(settings)
        self._state_manager = ForexStateManager()
        self._strategy_manager: Optional[ForexStrategyManager] = None
        self._trade_executor: Optional[ForexTradeExecutor] = None
        self._alert_manager: Optional[Any] = None

        # Engine-level tracking
        self._current_session = ""
        self._cycle_count = 0
        self._total_trades = 0
        self._pause_until: float = 0.0  # epoch seconds; 0 = not paused

        # Resolve trading pairs from settings
        forex_cfg = getattr(settings, "forex", None)
        self._trading_pairs: List[str] = (
            getattr(forex_cfg, "trading_pairs", ["XAU/USDT"])
            if forex_cfg else ["XAU/USDT"]
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all subsystems and start the trading loop."""
        logger.info("ForexTradingEngine: starting (pairs={})", self._trading_pairs)

        # Initialize exchange
        self._exchange = create_exchange(self._settings)
        await self._exchange.connect()
        logger.info("ForexTradingEngine: exchange connected ({})", self._exchange.name)

        # Initialize alert manager if available
        try:
            from monitoring.alerting import AlertManager
            self._alert_manager = AlertManager(self._settings)
        except Exception:
            pass

        # Initialize strategy manager
        self._strategy_manager = ForexStrategyManager(symbols=self._trading_pairs)
        self._strategy_manager.set_exchange(self._exchange)
        n = await self._strategy_manager.load_strategies()
        logger.info("ForexTradingEngine: {} forex strategies loaded", n)

        # Initialize trade executor
        self._trade_executor = ForexTradeExecutor(self._exchange, self._risk_manager)

        # Load persisted state
        await self._state_manager.load()

        # Reconcile positions from previous session
        await self._trade_executor.reconcile_positions()

        # Subscribe to user data stream for real-time position tracking
        try:
            await self._exchange.start_user_data_stream(self._on_user_data)
        except Exception as e:
            logger.debug("ForexTradingEngine: user data stream not available — {}", e)

        self._running = True
        logger.info("ForexTradingEngine: running — entering main loop")
        await self._forex_trading_loop()

    async def stop(self) -> None:
        """Shutdown the engine gracefully."""
        logger.info("ForexTradingEngine: stopping")
        self._running = False
        await self._state_manager.save()
        if self._exchange:
            try:
                await self._exchange.disconnect()
            except Exception:
                pass
        logger.info("ForexTradingEngine: stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _forex_trading_loop(self) -> None:
        """Main event loop — runs _forex_trading_cycle every CYCLE_INTERVAL seconds."""
        while self._running:
            try:
                await self._forex_trading_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ForexTradingEngine: cycle error — {}", e)
            await asyncio.sleep(self.CYCLE_INTERVAL)

    async def _forex_trading_cycle(self) -> None:
        """One iteration of the forex trading loop."""
        self._cycle_count += 1

        # Check market hours (skip weekends)
        if self._is_market_closed():
            if self._cycle_count % 20 == 1:  # log every ~10 minutes
                logger.info("ForexTradingEngine: market closed (weekend)")
            return

        # Check if we are paused (e.g. news event imminent)
        if self._pause_until > time.time():
            remaining = self._pause_until - time.time()
            logger.info("ForexTradingEngine: paused for {:.0f}s (news event)", remaining)
            return

        # Update session
        session = self._risk_manager.get_session_name()
        if session != self._current_session:
            logger.info("ForexTradingEngine: session changed → {}", session)
            if self._current_session:
                await self._state_manager.start_new_session(session)
            self._current_session = session

        if session == "Closed":
            if self._cycle_count % 20 == 1:
                logger.info("ForexTradingEngine: no active session — skipping")
            return

        # Manage existing positions (break-even, trailing SL)
        await self._manage_open_positions()

        # Process each trading pair
        for symbol in self._trading_pairs:
            try:
                await self._process_forex_symbol(symbol, session)
            except Exception as e:
                logger.error("ForexTradingEngine: error processing {} — {}", symbol, e)

    async def _process_forex_symbol(self, symbol: str, session: str) -> None:
        """Evaluate all strategies for a single forex symbol."""
        if self._trade_executor and symbol in self._trade_executor.active_trades:
            return  # already in a trade for this symbol

        # Fetch multi-timeframe OHLCV data
        ohlcv_data: Dict[str, pd.DataFrame] = {}
        for tf in self.TIMEFRAMES:
            try:
                df = await self._exchange.get_ohlcv(symbol, tf, self.OHLCV_LIMIT)
                if df is not None and len(df) >= 20:
                    ohlcv_data[tf] = df
            except Exception as e:
                logger.debug("ForexTradingEngine: OHLCV {} {} failed — {}", symbol, tf, e)

        if not ohlcv_data:
            logger.debug("ForexTradingEngine: no OHLCV data for {} — skipping", symbol)
            return

        # Get ticker
        try:
            ticker = await self._exchange.get_ticker(symbol)
        except Exception as e:
            logger.warning("ForexTradingEngine: ticker error for {} — {}", symbol, e)
            return

        if ticker.last <= 0:
            return

        # Evaluate strategies for consensus signal
        await self._evaluate_forex_strategies(symbol, ohlcv_data, ticker, session)

    async def _evaluate_forex_strategies(
        self,
        symbol: str,
        ohlcv_data: Dict[str, pd.DataFrame],
        ticker: Any,
        session: str,
    ) -> None:
        """Run strategy manager to get consensus, then validate and execute."""
        if not self._strategy_manager:
            return

        # Detect market regime from 1h data
        regime = self._detect_regime(ohlcv_data.get("1h"))

        consensus = await self._strategy_manager.evaluate_all(
            symbol=symbol,
            ohlcv_data=ohlcv_data,
            ticker=ticker,
            session=session,
            regime=regime,
        )

        if consensus is None:
            return  # insufficient confluence

        logger.info(
            "ForexTradingEngine: consensus {} {} (strength={:.2f}, votes={}/{})",
            consensus["direction"], symbol,
            consensus["strength"],
            consensus.get("long_votes", 0) if consensus["direction"] == "long"
            else consensus.get("short_votes", 0),
            len(self._strategy_manager._strategies),
        )

        # Final risk check
        try:
            balance = await self._exchange.get_balance()
            equity = balance.usdt_total
        except Exception:
            equity = 0.0

        if equity <= 0:
            logger.warning("ForexTradingEngine: zero equity — skipping trade")
            return

        # Check news filter
        if self._is_news_imminent():
            logger.info(
                "ForexTradingEngine: news event imminent — pausing trading {}min",
                self.NEWS_PROXIMITY_MINUTES,
            )
            self._pause_until = time.time() + self.NEWS_PROXIMITY_MINUTES * 60
            return

        # Execute trade
        if not self._trade_executor:
            return

        result = await self._trade_executor.execute_forex_trade(
            signal=consensus,
            equity=equity,
        )

        if result and result.get("status") == "opened":
            self._total_trades += 1
            await self._send_trade_alert(result, ticker, session)

    async def _manage_open_positions(self) -> None:
        """Call update_position_management for each active trade."""
        if not self._trade_executor:
            return
        for symbol in list(self._trade_executor.active_trades.keys()):
            try:
                await self._trade_executor.update_position_management(symbol)
            except Exception as e:
                logger.debug("ForexTradingEngine: position management error {} — {}", symbol, e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_market_closed(self) -> bool:
        """Return True if forex markets are closed (Saturday / Sunday UTC)."""
        now = datetime.now(tz=timezone.utc)
        # Forex closed: Saturday 22:00 UTC → Sunday 22:00 UTC
        # Approximation: skip all of Saturday and most of Sunday
        weekday = now.weekday()  # Monday=0, Sunday=6
        if weekday == 5:  # Saturday — fully closed
            return True
        if weekday == 6 and now.hour < 22:  # Sunday before 22:00 UTC
            return True
        return False

    def _is_news_imminent(self) -> bool:
        """Simple check for high-impact news events (FOMC, CPI, NFP).

        Returns True when we are within 15 minutes of a known event time.
        This uses a hardcoded schedule; a more complete implementation
        would query an economic calendar API.
        """
        now = datetime.now(tz=timezone.utc)
        # NFP: first Friday of each month at 13:30 UTC
        # FOMC: 8 times per year, typically Wednesday 19:00 UTC
        # CPI: typically around 8th of the month at 13:30 UTC
        # Simplified: check for known high-impact hours
        high_impact_times = [
            (13, 30),  # NFP / CPI release time (UTC)
            (19, 0),   # FOMC statement
            (14, 0),   # Fed speeches
        ]
        for hour, minute in high_impact_times:
            delta_minutes = abs((now.hour * 60 + now.minute) - (hour * 60 + minute))
            if delta_minutes <= self.NEWS_PROXIMITY_MINUTES:
                return True
        return False

    def _detect_regime(self, ohlcv: Optional[pd.DataFrame]) -> str:
        """Simple market regime detection from OHLCV data."""
        if ohlcv is None or len(ohlcv) < 30:
            return "unknown"
        closes = ohlcv["close"].tolist()
        # Simple regime: compare recent EMA with older EMA
        ema_fast = self._ema(closes, 10)
        ema_slow = self._ema(closes, 30)
        if ema_fast > ema_slow * 1.001:
            return "trending_up"
        elif ema_fast < ema_slow * 0.999:
            return "trending_down"
        else:
            return "ranging"

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        """Return EMA of *prices*."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        k = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = p * k + ema * (1 - k)
        return ema

    async def _send_trade_alert(
        self, result: Dict[str, Any], ticker: Any, session: str
    ) -> None:
        """Send Telegram alert for a new trade."""
        if not self._alert_manager:
            return
        try:
            direction = result.get("direction", "?").upper()
            symbol = result.get("symbol", "?")
            fill_price = result.get("fill_price", 0.0)
            lot_size = result.get("lot_size", 0.0)
            sl = result.get("stop_loss", 0.0)
            tp = result.get("take_profit", 0.0)

            message = (
                f"🟢 <b>Forex Trade Opened</b>\n"
                f"📊 {direction} {symbol}\n"
                f"💰 Entry: {fill_price:.2f}\n"
                f"📦 Lot size: {lot_size}\n"
                f"🛑 SL: {sl:.2f}\n"
                f"🎯 TP: {tp:.2f}\n"
                f"🕐 Session: {session}"
            )
            from monitoring.alerting import AlertType
            await self._alert_manager.send_alert(message=message, alert_type=AlertType.TRADE_OPENED)
        except Exception as e:
            logger.debug("ForexTradingEngine: alert error — {}", e)

    async def _on_user_data(self, data: Dict[str, Any]) -> None:
        """Handle real-time user data updates from WebSocket."""
        channel = data.get("channel", "")
        if channel == "futures.orders":
            # Order filled → check if it was SL/TP → mark trade closed
            result = data.get("result", {})
            if isinstance(result, list):
                for order in result:
                    if order.get("status") == "finished" and order.get("reduce_only"):
                        contract = order.get("contract", "")
                        logger.info(
                            "ForexTradingEngine: reduce-only order filled — {} (SL/TP hit)",
                            contract
                        )

    # ------------------------------------------------------------------
    # Health / dashboard interface
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for health-check / dashboard."""
        session_stats = self._state_manager.get_current_session_stats()
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "total_trades": self._total_trades,
            "current_session": self._current_session,
            "session_pnl_pips": session_stats.pips_pnl,
            "session_pnl_usdt": round(session_stats.usdt_pnl, 2),
            "session_win_rate": round(session_stats.win_rate, 3),
            "active_trades": list(
                self._trade_executor.active_trades.keys()
            ) if self._trade_executor else [],
            "strategy_stats": self._strategy_manager.get_stats() if self._strategy_manager else {},
        }
