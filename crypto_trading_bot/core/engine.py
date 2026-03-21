"""Main trading engine that orchestrates all subsystems.

.. deprecated:: Issue 4
    This monolithic engine is replaced by:
      - Rust engine (rust_engine/src/main.rs) for the hot path
        (order execution, orderbook management, signal evaluation).
      - core/cold_path_orchestrator.py for the cold path
        (regime detection, AI sentiment, health monitoring).

    Kept for reference, backtesting compatibility, and legacy deployments.
    New development should use the ColdPathOrchestrator instead.

    See: core/cold_path_orchestrator.py, ai/regime_computer.py,
         ai/sentiment_service.py
"""

import asyncio
import json
import signal
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ai.market_analyzer.order_flow_analyzer import OrderFlowAnalyzer
from ai.market_analyzer.regime_detector import MarketRegime, MarketRegimeDetector
from ai.market_analyzer.volatility_analyzer import VolatilityAnalyzer
from core.event_bus import EventBus, EventType
from core.exceptions import CircuitBreakerError
from core.health_check import HealthChecker
from core.scheduler import TaskScheduler
from core.state_manager import BotStatus, StateManager
from core.state_persistence import StatePersistence
from exchange.exchange_factory import create_exchange
from exchange.order_manager import OrderManager
from exchange.position_manager import PositionManager
from execution.trade_executor import TradeExecutor
from monitoring.alerting import AlertManager, AlertType
from risk.crash_protector import CrashProtector
from risk.forex_risk_manager import ForexRiskManager
from risk.risk_manager import RiskManager
from strategy.strategy_manager import StrategyManager
from utils.time_utils import time_until_midnight_utc

if TYPE_CHECKING:
    from ai.market_analyzer.cross_asset_regime_detector import CrossAssetRegimeDetector

# Map MarketRegime enum values to the string labels used by the risk system
_REGIME_LABEL_MAP: Dict[MarketRegime, str] = {
    MarketRegime.STRONG_UPTREND: "trending_up",
    MarketRegime.WEAK_UPTREND: "trending_up",
    MarketRegime.STRONG_DOWNTREND: "trending_down",
    MarketRegime.WEAK_DOWNTREND: "trending_down",
    MarketRegime.RANGING: "ranging",
    MarketRegime.HIGH_VOLATILITY: "ranging",  # strategy selection uses volatility label for this
    MarketRegime.LOW_VOLATILITY: "low_volatility",
    MarketRegime.CRASH: "crash",
    MarketRegime.UNKNOWN: "unknown",
}

# BTC benchmark symbol for market-wide regime detection
_BTC_BENCHMARK = "BTC/USDT"
_REGIME_TIMEFRAME = "4h"
_REGIME_CANDLES = 100

# Data integrity / heartbeat constants
# Timeframe-aware stale thresholds: expected candle interval + 60 s buffer
_STALE_DATA_THRESHOLD_BY_TIMEFRAME: Dict[str, int] = {
    "1m": 60 + 60,
    "3m": 180 + 60,
    "5m": 300 + 60,
    "15m": 900 + 60,
    "30m": 1800 + 60,
    "1h": 3600 + 60,
    "2h": 7200 + 60,
    "4h": 14400 + 60,
    "1d": 86400 + 60,
}
_STALE_DATA_THRESHOLD_SECONDS = 60     # fallback for unknown timeframes
_HEARTBEAT_TIMEOUT_SECONDS = 300       # force reconnect if no API call in 5 minutes
_STATE_CHECKPOINT_INTERVAL_SECONDS = 300  # save state every 5 minutes
_STATE_CHECKPOINT_FILE = Path("data") / "engine_state.json"
# Maximum allowed gap between consecutive candles as a multiplier of the expected interval
_GAP_TOLERANCE_MULTIPLIER = 2.5

# Cooldown periods (seconds) between repeated alerts of the same type per symbol
_ALERT_COOLDOWN_SECONDS: Dict[str, int] = {
    "liquidation_warning": 3600,   # 1 hour between warning alerts
    "liquidation_danger": 900,     # 15 min between danger alerts
    "liquidation_critical": 60,    # 1 min between critical alerts (before auto-close)
    "risk_warning": 1800,          # 30 min between generic risk warnings
    "funding_rate": 3600,          # 1 hour between funding alerts
}


class TradingEngine:
    """
    Central orchestrator for the trading bot.

    Manages lifecycle of all subsystems and runs the main trading loop.
    """

    # Maximum number of symbols processed concurrently in each trading cycle
    _API_SEMAPHORE = asyncio.Semaphore(3)

    def __init__(self, settings) -> None:
        self.settings = settings

        # Core subsystems — populated in _initialize_subsystems
        self.state_manager = None
        self.event_bus = None
        self.scheduler = None
        self.health_checker = None

        # Optional subsystems — populated by higher-level wiring
        self.exchange = None
        self.order_manager = None
        self.position_manager = None
        self.strategy_manager = None
        self.data_aggregator = None
        self.ai_brain = None
        self.risk_manager = None
        self.trade_executor = None
        self.alert_manager = None
        self.trade_tracker = None

        # Forex-specific subsystems
        self.forex_exchange = None
        self.forex_risk_manager = None

        # Real-time data hub — created in _initialize_subsystems
        self.realtime_hub = None

        # Local order book cache — created in _initialize_subsystems
        self.local_orderbook_manager = None

        # Market analysis subsystems — populated in _initialize_subsystems
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.volatility_analyzer: Optional[VolatilityAnalyzer] = None
        self.cross_asset_regime_detector: Optional[CrossAssetRegimeDetector] = None
        self.order_flow_analyzer: Optional[OrderFlowAnalyzer] = None

        # Profit maximization subsystems — populated in _initialize_subsystems
        self.profit_maximizer = None
        self.parameter_tuner = None
        self.meta_learner = None

        # State persistence — created in _initialize_subsystems (Upgrade 2)
        self.state_persistence: Optional[StatePersistence] = None

        # Current detected market regime (available to dashboard / monitoring)
        self.current_market_regime: str = "unknown"
        self.current_volatility_regime: str = "normal"
        self.current_cross_asset_regime: str = "NORMAL"
        self._previous_market_regime: str = "unknown"

        self._running = False
        self._shutdown_event = asyncio.Event()
        self._start_time: Optional[datetime] = None
        self._cycle_count: int = 0
        self._fast_cycle_count: int = 0
        # AI market analysis: (cached_at, result) with 15-min TTL
        self._ai_analysis_cache: Optional[Tuple[datetime, dict]] = None
        self._AI_CACHE_TTL_SECONDS: float = 900.0  # 15 minutes

        # Crash protector (MR2: crash detection & auto-protection)
        self.crash_protector: CrashProtector = CrashProtector()

        # Heartbeat tracking for reconnection logic
        self._last_successful_api_call: float = time.time()
        self._last_checkpoint_ts: float = time.time()

        # Previous-cycle position snapshot for detecting closed positions
        self._previous_positions: Dict[str, dict] = {}

        # Per-symbol cooldown: tracks when a position was last closed per symbol
        self._symbol_last_closed: Dict[str, float] = {}

        # Per-alert cooldown tracker: "symbol:alert_type" -> last_sent_timestamp
        self._alert_cooldowns: Dict[str, float] = {}

        # New subsystems initialised lazily in _initialize_subsystems
        self._economic_filter: Optional[Any] = None    # Task 7
        self._portfolio_risk_manager: Optional[Any] = None  # Task 8
        self.anti_liquidation: Optional[Any] = None  # Anti-liquidation manager

        # Update 1: WebSocket data manager for zero-latency price cache
        self.ws_data_manager = None
        # Update 2: Margin monitor (runs every 5 s in fast cycle)
        self.margin_monitor = None

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all subsystems and start the main trading loop."""
        logger.info("Starting Trading Engine…")
        self._start_time = datetime.now(tz=timezone.utc)
        await self._initialize_subsystems()
        self._register_signal_handlers()
        await self._start_background_tasks()
        logger.info("Trading Engine started successfully.")
        await self.run()

    async def run(self) -> None:
        """Main trading loop — dual-interval architecture.

        * **Fast loop** (every ``fast_loop_interval`` seconds, default 5 s):
          Updates trailing stops, checks SL/TP, syncs positions, updates dashboard.
          Implemented by :meth:`_fast_cycle`.

        * **Slow loop** (every ``slow_loop_interval`` seconds, default 30 s):
          Fetches OHLCV, runs strategies, generates and executes signals.
          Implemented by :meth:`_slow_cycle` (the original ``_trading_cycle``).

        Both loops share the same ``_cycle_count`` counter (incremented on each
        slow cycle) and run concurrently as separate async tasks.
        """
        self._running = True
        self._cycle_count = 0
        self._fast_cycle_count = 0

        trading_loop_cfg = getattr(self.settings, "trading_loop", None)
        fast_interval: int = getattr(trading_loop_cfg, "fast_loop_interval", 5) if trading_loop_cfg else 5
        if trading_loop_cfg is not None:
            slow_interval: int = getattr(trading_loop_cfg, "slow_loop_interval", 30)
        elif hasattr(self.settings, "data_sources"):
            slow_interval = self.settings.data_sources.polling_interval_seconds
        else:
            slow_interval = 30

        async def _fast_loop() -> None:
            while self._running and not self._shutdown_event.is_set():
                try:
                    self._fast_cycle_count += 1
                    await self._fast_cycle(self._fast_cycle_count)
                    await asyncio.sleep(fast_interval)
                except asyncio.CancelledError:
                    logger.info("Fast trading loop cancelled.")
                    break
                except Exception as exc:
                    logger.error(f"Error in fast cycle #{self._fast_cycle_count}: {exc}")
                    await asyncio.sleep(fast_interval)

        async def _slow_loop() -> None:
            while self._running and not self._shutdown_event.is_set():
                try:
                    self._cycle_count += 1
                    await self._trading_cycle(self._cycle_count)
                    await asyncio.sleep(slow_interval)
                except asyncio.CancelledError:
                    logger.info("Slow trading loop cancelled.")
                    break
                except Exception as exc:
                    logger.error(f"Error in trading cycle #{self._cycle_count}: {exc}")
                    await asyncio.sleep(5)

        try:
            fast_task = asyncio.create_task(_fast_loop())
            slow_task = asyncio.create_task(_slow_loop())
            await asyncio.gather(fast_task, slow_task)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — stopping trading engine.")
            await self.stop()

    async def _fast_cycle(self, cycle_num: int) -> None:
        """Fast monitoring cycle (runs every ``fast_loop_interval`` seconds).

        Responsibilities:
        * Update trailing stops for open positions.
        * Enforce SL/TP overlays (paper mode).
        * Sync position state from the exchange.
        * Check funding rates and record costs.
        * Emit a lightweight dashboard update event.
        * (Update 2) Run margin utilisation check via MarginMonitor.
        """
        # Guard: skip if circuit breaker is active or bot is paused
        if self.state_manager is not None and not self.state_manager.is_trading_allowed():
            return

        trading_pairs: List[str] = getattr(
            getattr(self.settings, "exchange", None), "trading_pairs", []
        )

        if self.position_manager is not None and self.exchange is not None:
            for symbol in trading_pairs:
                try:
                    # Update 1: prefer WebSocket cache over REST call
                    last_price: Optional[float] = None
                    if (
                        self.ws_data_manager is not None
                        and self.ws_data_manager.is_ready(symbol)
                    ):
                        last_price = self.ws_data_manager.get_price(symbol)

                    if last_price and last_price > 0:
                        await self.position_manager.update_trailing_stop(symbol, last_price)
                        await self.position_manager.update_trailing_take_profit(
                            symbol, last_price
                        )
                    else:
                        # REST fallback when WS cache is not yet populated
                        ticker = await self.exchange.get_ticker(symbol)
                        if ticker is not None:
                            await self.position_manager.update_trailing_stop(symbol, ticker.last)
                            await self.position_manager.update_trailing_take_profit(
                                symbol, ticker.last
                            )
                except Exception as exc:
                    logger.debug(f"Fast cycle trailing stop update error for {symbol}: {exc}")

        # b. Enforce SL/TP overlays in paper mode
        is_paper = getattr(self.settings, "is_paper_trading", False)
        if self.position_manager is not None and is_paper:
            try:
                await self.position_manager._enforce_risk_overlays()
            except Exception as exc:
                logger.debug(f"Fast cycle SL/TP enforcement error: {exc}")

        # c. Sync open positions from exchange (lightweight)
        if self.exchange is not None:
            try:
                if self.position_manager is not None:
                    await self.position_manager.sync_positions()
            except Exception as exc:
                logger.debug(f"Fast cycle position sync error: {exc}")

        # d. Check funding rates for open positions and record costs
        await self._check_funding_rates()

        # e. Check liquidation proximity for all open positions
        await self._check_liquidation_proximity()

        logger.debug(f"Fast cycle #{cycle_num} complete")

    async def stop(self) -> None:
        """Graceful shutdown sequence.

        1. Stop the trading loop.
        2. Cancel ALL pending entry limit orders (not SL/TP) to prevent new positions.
        3. Do NOT close existing positions — they keep SL/TP protection on the exchange.
        4. Verify every open position has at least one SL order; place emergency SL if missing.
        5. Save a shutdown checkpoint to data/shutdown_state.json.
        6. Disconnect exchange and stop subsystems.
        """
        logger.info("Shutting down Trading Engine…")
        self._running = False
        self._shutdown_event.set()

        # Step 2: Cancel pending entry limit orders
        if self.trade_executor is not None:
            try:
                cancelled = await self.trade_executor.cancel_stale_entry_orders(max_age_minutes=0)
                if cancelled:
                    logger.info("Shutdown: cancelled {} pending entry orders", cancelled)
            except Exception as exc:
                logger.warning("Shutdown: could not cancel entry orders: {}", exc)

        # Steps 3 & 4: Verify SL coverage for open positions
        if self.exchange is not None and self.position_manager is not None:
            await self._ensure_shutdown_sltp_coverage()

        # Step 5: Save shutdown checkpoint
        await self._save_shutdown_checkpoint()

        # Upgrade 2: persist full in-memory state before shutdown
        if self.state_persistence is not None:
            try:
                await self.state_persistence.save_state()
                logger.info("StatePersistence: state saved on shutdown.")
            except Exception as exc:
                logger.warning("StatePersistence: save_state on shutdown failed: {}", exc)

        if self.scheduler is not None:
            self.scheduler.stop()

        if self.trade_tracker is not None:
            try:
                await self.trade_tracker.stop()
            except Exception as exc:
                logger.warning("Error stopping trade tracker: {}", exc)

        # Update 1: stop WebSocketDataManager
        if self.ws_data_manager is not None:
            try:
                await self.ws_data_manager.stop()
            except Exception as exc:
                logger.warning("Error stopping WebSocketDataManager: {}", exc)

        # Update 2: stop MarginMonitor
        if self.margin_monitor is not None:
            try:
                await self.margin_monitor.stop()
            except Exception as exc:
                logger.warning("Error stopping MarginMonitor: {}", exc)

        if self.exchange is not None:
            try:
                await self.exchange.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting exchange: {}", exc)

        if self.state_manager is not None:
            await self.state_manager.update_state(status=BotStatus.STOPPED)

        logger.info("Trading Engine stopped.")

    async def _ensure_shutdown_sltp_coverage(self) -> None:
        """Verify every open position has at least one SL order; place emergency SL if missing."""
        try:
            positions = await self.exchange.get_positions()
            if not positions:
                return
            open_orders = await self.exchange.get_open_orders()
            symbols_with_sl = {
                o.symbol for o in open_orders
                if hasattr(o, "type") and str(getattr(o, "type", "")).lower() in (
                    "stop_loss", "stop", "stop_market"
                )
            }
            for pos in positions:
                if pos.symbol not in symbols_with_sl:
                    logger.warning(
                        "Shutdown: {} has no SL — placing emergency SL at 3%", pos.symbol
                    )
                    try:
                        from exchange.base_exchange import OrderSide, PositionSide
                        is_long = pos.side == PositionSide.LONG
                        current_price = pos.current_price or pos.mark_price or pos.entry_price
                        if current_price <= 0:
                            ticker = await self.exchange.get_ticker(pos.symbol)
                            current_price = ticker.last
                        sl_price = current_price * (0.97 if is_long else 1.03)
                        close_side = OrderSide.SELL if is_long else OrderSide.BUY
                        await self.exchange.create_stop_loss_order(
                            pos.symbol, close_side, pos.amount, sl_price
                        )
                        logger.info(
                            "Shutdown: emergency SL placed for {} @ {:.4f}",
                            pos.symbol, sl_price,
                        )
                    except Exception as exc:
                        logger.error(
                            "Shutdown: failed to place emergency SL for {}: {}", pos.symbol, exc
                        )
        except Exception as exc:
            logger.warning("Shutdown SL coverage check failed: {}", exc)

    async def _save_shutdown_checkpoint(self) -> None:
        """Save current position state to data/shutdown_state.json."""
        import json
        import os
        from datetime import datetime as dt

        checkpoint: dict = {
            "timestamp": dt.utcnow().isoformat(),
            "cycle_count": self._cycle_count,
            "positions": [],
        }
        try:
            if self.exchange is not None:
                positions = await self.exchange.get_positions()
                open_orders = await self.exchange.get_open_orders()
                for pos in positions:
                    pos_orders = [
                        {"id": o.id, "type": str(o.type), "symbol": o.symbol, "price": o.price}
                        for o in open_orders if o.symbol == pos.symbol
                    ]
                    checkpoint["positions"].append({
                        "symbol": pos.symbol,
                        "side": str(pos.side),
                        "amount": pos.amount,
                        "entry_price": pos.entry_price,
                        "current_price": pos.current_price,
                        "liquidation_price": pos.liquidation_price,
                        "orders": pos_orders,
                    })
        except Exception as exc:
            logger.warning("Could not populate shutdown checkpoint positions: {}", exc)

        try:
            os.makedirs("data", exist_ok=True)
            checkpoint_path = "data/shutdown_state.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            logger.info("Shutdown checkpoint saved to {}", checkpoint_path)
        except Exception as exc:
            logger.warning("Could not save shutdown checkpoint: {}", exc)

    async def switch_mode(self, new_mode: str) -> dict:
        """Safely switch between any supported trading mode at runtime.

        Supported modes:

        * ``futures/live``    — live futures trading on configured exchange
        * ``futures/testnet`` — testnet futures trading
        * ``futures/paper``   — paper futures trading (simulated)
        * ``forex/live``      — live forex trading on Gate.io TradFi
        * ``forex/demo``      — demo forex trading on Gate.io TradFi testnet

        Legacy flat names (``"paper"``, ``"live"``, ``"testnet"``) are still
        accepted for backwards compatibility.

        Steps:

        1. Validate the new mode.
        2. Check API keys for live/testnet/forex modes.
        3. Pause the trading loop.
        4. Close all open positions and cancel all pending orders.
        5. Disconnect the current exchange.
        6. Update ``settings.trading_mode``.
        7. Re-initialise the exchange via ``_initialize_exchange()``.
        8. Re-wire TradeExecutor with the new exchange.
        9. Resume the trading loop.

        Args:
            new_mode: One of the mode strings listed above.

        Returns:
            dict with ``success``, ``mode``, and ``message`` keys.
        """
        # ------------------------------------------------------------------
        # 1. Normalise and validate
        # ------------------------------------------------------------------
        _VALID_MODES: dict = {
            # Canonical slash-separated names → settings value
            "futures/live":    {"trading_mode": "live",       "market": "futures"},
            "futures/testnet": {"trading_mode": "testnet",    "market": "futures"},
            "futures/paper":   {"trading_mode": "paper",      "market": "futures"},
            "forex/live":      {"trading_mode": "forex_live", "market": "forex"},
            "forex/demo":      {"trading_mode": "forex_demo", "market": "forex"},
            # Legacy flat names (backwards compat)
            "paper":    {"trading_mode": "paper",      "market": "futures"},
            "live":     {"trading_mode": "live",       "market": "futures"},
            "testnet":  {"trading_mode": "testnet",    "market": "futures"},
        }

        new_mode_normalized = new_mode.lower().strip()
        if new_mode_normalized not in _VALID_MODES:
            return {
                "success": False,
                "error": (
                    f"Invalid mode: {new_mode!r}. "
                    f"Use one of: {list(_VALID_MODES.keys())}"
                ),
            }

        mode_config = _VALID_MODES[new_mode_normalized]
        settings_mode = mode_config["trading_mode"]

        current_mode = getattr(self.settings, "trading_mode", "paper").lower()
        if settings_mode == current_mode:
            return {
                "success": True,
                "mode": current_mode,
                "message": f"Already in {new_mode_normalized} mode.",
            }

        # ------------------------------------------------------------------
        # 2. Safety checks: require API keys for non-paper modes
        # ------------------------------------------------------------------
        if settings_mode in ("live", "testnet"):
            api_key = getattr(self.settings, "exchange_api_key", None) or ""
            if not api_key:
                exchange_id = getattr(
                    getattr(self.settings, "exchange", None), "primary_exchange", "mexc"
                ) or "mexc"
                api_key = getattr(self.settings, f"{exchange_id}_api_key", None) or ""
            if not api_key:
                return {
                    "success": False,
                    "mode": current_mode,
                    "message": (
                        f"Cannot switch to {new_mode_normalized} mode: no API key configured. "
                        "Set EXCHANGE_API_KEY (or exchange-specific key) in your environment."
                    ),
                }

        if settings_mode in ("forex_live", "forex_demo"):
            gateio_key = (
                getattr(self.settings, "gateio_api_key", None)
                or getattr(self.settings, "exchange_api_key", None)
                or ""
            )
            if not gateio_key:
                return {
                    "success": False,
                    "mode": current_mode,
                    "message": (
                        f"Cannot switch to {new_mode_normalized} mode: no Gate.io API key configured. "
                        "Set GATEIO_API_KEY or EXCHANGE_API_KEY in your environment."
                    ),
                }

        logger.info("Switching trading mode: {} → {}", current_mode, new_mode_normalized)

        # ------------------------------------------------------------------
        # 3. Pause the trading loop
        # ------------------------------------------------------------------
        was_running = self._running
        self._running = False

        closed_positions = 0
        cancelled_orders = 0
        errors: List[str] = []

        try:
            # 4. Close all open positions and cancel orders on the current exchange
            if self.exchange is not None:
                try:
                    positions = await self.exchange.get_positions()
                    for pos in positions:
                        try:
                            await self.exchange.close_position(pos.symbol)
                            closed_positions += 1
                        except Exception as exc:
                            errors.append(f"close {pos.symbol}: {exc}")
                except Exception as exc:
                    errors.append(f"get_positions: {exc}")

                try:
                    open_orders = await self.exchange.get_open_orders()
                    for order in open_orders:
                        try:
                            await self.exchange.cancel_order(order.id, order.symbol)
                            cancelled_orders += 1
                        except Exception as exc:
                            errors.append(f"cancel {order.id}: {exc}")
                except Exception as exc:
                    errors.append(f"get_open_orders: {exc}")

                # 5. Disconnect current exchange
                try:
                    await self.exchange.disconnect()
                except Exception as exc:
                    errors.append(f"disconnect: {exc}")
                self.exchange = None
                self.order_manager = None
                self.position_manager = None

            # 6. Update the mode setting
            self.settings.trading_mode = settings_mode

            # 7. Re-initialise the exchange with the new mode
            await self._initialize_exchange()

            # 8. Re-wire TradeExecutor
            if (
                self.exchange is not None
                and self.order_manager is not None
                and self.position_manager is not None
            ):
                self.trade_executor = TradeExecutor(
                    exchange=self.exchange,
                    order_manager=self.order_manager,
                    position_manager=self.position_manager,
                )
                logger.info("TradeExecutor re-wired to new exchange.")

            # Re-register circuit breaker callbacks
            if (
                self.risk_manager is not None
                and self.trade_executor is not None
                and self.order_manager is not None
            ):
                self.risk_manager._circuit_breaker.register_callbacks(
                    close_positions=self._cb_close_all_positions,
                    cancel_orders=self._cb_cancel_all_orders,
                    send_alert=self._cb_send_alert,
                )

            # Restart paper SL/TP monitor if switching to paper
            if settings_mode == "paper":
                try:
                    from exchange.paper_exchange import PaperExchange

                    if isinstance(self.exchange, PaperExchange):
                        asyncio.create_task(
                            self.exchange.run_sl_tp_monitor(), name="paper_sl_tp_monitor"
                        )
                except Exception as exc:
                    logger.debug(f"Could not restart paper SL/TP monitor: {exc}")

            # Stop the old realtime hub and create a new one for the new exchange
            if self.realtime_hub is not None:
                try:
                    await self.realtime_hub.stop()
                except Exception as exc:
                    logger.debug(f"Could not stop old realtime hub: {exc}")
                self.realtime_hub = None
            await self._initialize_realtime_hub()
            # Re-wire the broadcast function from any active dashboard
            if self.realtime_hub is not None and hasattr(self, "_dashboard"):
                try:
                    dash_app = getattr(self._dashboard, "app", None)
                    if dash_app and hasattr(dash_app, "state"):
                        bcast = getattr(dash_app.state, "broadcast_update", None)
                        if bcast is not None:
                            self.realtime_hub.set_broadcast_fn(bcast)
                except Exception as exc:
                    logger.debug(f"Could not re-wire hub broadcast fn: {exc}")

        except Exception as exc:
            errors.append(f"switch_mode critical error: {exc}")
            logger.error("Mode switch failed: {}", exc)
            # Restore previous running state
            self._running = was_running
            return {
                "success": False,
                "mode": current_mode,
                "message": f"Mode switch failed: {exc}",
                "errors": errors,
            }
        finally:
            # 9. Resume the trading loop if it was running
            if was_running:
                self._running = True

        logger.info(
            "Trading mode switched: {} → {} (closed={} cancelled={} errors={})",
            current_mode,
            new_mode_normalized,
            closed_positions,
            cancelled_orders,
            len(errors),
        )
        return {
            "success": True,
            "mode": settings_mode,
            "message": f"Switched to {new_mode_normalized} mode successfully.",
            "closed_positions": closed_positions,
            "cancelled_orders": cancelled_orders,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Internal — trading loop
    # ------------------------------------------------------------------

    async def _trading_cycle(self, cycle_num: int) -> None:
        """
        Single iteration of the main trading loop.

        Steps:
        1. Circuit-breaker / paused guard
        2. Detect market regime and volatility (BTC/USDT 4h benchmark)
        3. For each symbol: fetch multi-timeframe OHLCV (15m, 1h, 4h) + ticker
        4. Run only regime-appropriate strategies
        5. Score and rank signals (confluence boost already applied by evaluate_all)
        6. Execute only the TOP signal per symbol
        7. Sync open positions from exchange
        8. Update risk manager with real regime and volatility data
        9. Enforce SL/TP for paper trading mode
        10. Evaluate daily P&L limits
        11. Emit events for dashboard/monitoring
        12. Log cycle summary
        """
        # 1. Check circuit breaker and paused state
        if self.state_manager is not None and not self.state_manager.is_trading_allowed():
            logger.warning(f"Cycle #{cycle_num}: trading not allowed (circuit breaker or paused).")
            return

        # 1b. Heartbeat check — force reconnect if exchange has been silent too long
        await self._check_heartbeat()

        # 1c. Crash protector pre-cycle check
        crash_level = self.crash_protector.get_current_level()
        if self.crash_protector.is_circuit_breaker_active():
            logger.warning(
                f"Cycle #{cycle_num}: crash circuit breaker active (level={crash_level.value}) — skipping."
            )
            return

        # 1d. Economic calendar kill switch (Task 7)
        if hasattr(self, "_economic_filter") and self._economic_filter is not None:
            try:
                allowed, reason = await self._economic_filter.is_trading_allowed()
                if not allowed:
                    logger.warning("Cycle #{}: {}", cycle_num, reason)
                    return
            except Exception as _ec_exc:
                logger.debug("Economic calendar check error: {}", _ec_exc)

        logger.debug(f"Trading cycle #{cycle_num} start (crash_level={crash_level.value})")

        signals_processed = 0
        trades_executed = 0
        trading_pairs: List[str] = getattr(
            getattr(self.settings, "exchange", None), "trading_pairs", []
        )

        # Refresh the aggregator cache once per cycle so that
        # get_items_for_symbol() returns up-to-date results per symbol.
        if self.data_aggregator is not None:
            try:
                await self.data_aggregator.collect_latest(max_age_minutes=60, limit_per_source=50)
            except Exception as exc:
                logger.warning(f"Data aggregator collect_latest error: {exc}")

        # AI market analysis — called once per cycle, cached for 15 minutes.
        ai_market_signal = await self._get_ai_market_analysis(trading_pairs)

        # 2. Detect market regime and volatility using BTC/USDT 4h benchmark
        await self._detect_market_regime()

        # 2a. Detect cross-asset regime (cached for 15 minutes)
        if self.cross_asset_regime_detector is not None:
            try:
                regime, confidence, transition_probs = await self.cross_asset_regime_detector.detect_regime(
                    exchange=self.exchange
                )
                self.current_cross_asset_regime = regime.value
                logger.debug(
                    f"Cross-asset regime: {regime.value} (confidence={confidence:.2f})"
                )
            except Exception as exc:
                logger.debug(f"Cross-asset regime detection failed: {exc}")

        # 2b. Check for CRITICAL negative news events — reduce exposure if detected
        await self._check_emergency_exposure_reduction()

        # 3-6. Per-symbol: fetch multi-timeframe data → strategies → risk → execute
        if self.exchange is not None and self.strategy_manager is not None:

            async def _process_symbol(symbol: str) -> Tuple[int, int]:
                """Process a single symbol: fetch data, run strategies, execute trades."""
                async with TradingEngine._API_SEMAPHORE:
                    _signals = 0
                    _trades = 0
                    try:
                        # a. Fetch multi-timeframe OHLCV (15m, 1h, 4h)
                        market_data: Dict[str, pd.DataFrame] = {}
                        primary_df: Optional[pd.DataFrame] = None

                        for timeframe, limit in [("15m", 100), ("1h", 100), ("4h", 100)]:
                            try:
                                tf_df = await self.exchange.get_ohlcv(symbol, timeframe, limit=limit)
                                if tf_df is not None and not tf_df.empty:
                                    if tf_df.index.name == "timestamp":
                                        tf_df = tf_df.reset_index()
                                    required_cols = {"open", "high", "low", "close", "volume"}
                                    if not required_cols - set(tf_df.columns):
                                        # Data integrity: check for stale data
                                        if self._is_data_stale(tf_df, symbol, timeframe):
                                            logger.warning(
                                                f"Stale data for {symbol} {timeframe} — skipping"
                                            )
                                            continue
                                        # Data integrity: check for gaps in candles
                                        if self._has_data_gaps(tf_df, symbol, timeframe):
                                            logger.warning(
                                                f"Data gaps for {symbol} {timeframe} — skipping"
                                            )
                                            continue
                                        market_data[timeframe] = tf_df
                                        if timeframe == "15m":
                                            primary_df = tf_df
                                        # Record successful API call
                                        self._last_successful_api_call = time.time()
                            except Exception as exc:
                                logger.debug(f"Could not fetch {timeframe} OHLCV for {symbol}: {exc}")

                        if primary_df is None or primary_df.empty:
                            logger.warning(f"No usable 15m OHLCV for {symbol} — skipping symbol.")
                            return _signals, _trades

                        # b. Fetch current ticker
                        ticker = None
                        try:
                            ticker = await self.exchange.get_ticker(symbol)
                            self._last_successful_api_call = time.time()
                            # Update crash protector with BTC price
                            if symbol == _BTC_BENCHMARK and ticker is not None:
                                self.crash_protector.update_price(float(ticker.last))
                        except Exception as exc:
                            logger.warning(f"Could not fetch ticker for {symbol}: {exc}")

                        # c. Sentiment items for this symbol
                        symbol_items = (
                            await self.data_aggregator.get_items_for_symbol(symbol)
                            if self.data_aggregator is not None
                            else []
                        )

                        # c2. Get order flow signal (if available)
                        order_flow_signal = None
                        if self.order_flow_analyzer is not None:
                            try:
                                order_flow_signal = self.order_flow_analyzer.analyze()
                            except Exception as exc:
                                logger.debug(f"Order flow analysis failed for {symbol}: {exc}")

                        # d. Run regime-appropriate strategies (evaluate_all filters internally)
                        signals = await self.strategy_manager.evaluate_all(
                            symbol,
                            market_data,
                            sentiment_items=symbol_items,
                            regime=self.current_market_regime,
                            volatility_regime=self.current_volatility_regime,
                            order_flow_signal=order_flow_signal,
                        )

                        # Log generated signals for better observability
                        if signals:
                            logger.debug(
                                f"Strategy signal generated for {symbol}: "
                                f"{len(signals)} signal(s), top confidence={signals[0].get('confidence', 0):.3f} "
                                f"direction={signals[0].get('direction')} strategy={signals[0].get('strategy')}"
                            )
                        else:
                            logger.debug(f"No signals generated for {symbol} in current regime")

                        # e. Apply AI confidence modifier (±15%) to each signal.
                        if ai_market_signal is not None:
                            for s in signals:
                                original = s.get("confidence", 0.5)
                                s["confidence"] = self._apply_ai_confidence_modifier(
                                    s, ai_market_signal
                                )
                                if s["confidence"] != original:
                                    logger.debug(
                                        f"AI modifier applied to {symbol} signal: "
                                        f"{original:.3f} → {s['confidence']:.3f} "
                                        f"(ai_direction={ai_market_signal.get('direction')})"
                                    )

                        _signals += len(signals)

                        # f. Execute only the TOP signal per symbol (highest confidence after sort)
                        top_signal = signals[0] if signals else None
                        signals_to_execute = [top_signal] if top_signal is not None else []

                        for signal in signals_to_execute:
                            try:
                                logger.debug(
                                    f"Processing signal: {signal.get('symbol')} {signal.get('direction')} "
                                    f"confidence={signal.get('confidence', 0):.3f} "
                                    f"strategy={signal.get('strategy')}"
                                )

                                # Check same-symbol conflict prevention
                                sig_direction = signal.get("direction", "long")
                                sig_confidence = float(signal.get("confidence", 0.0))
                                if self._has_conflicting_position(symbol, sig_direction, sig_confidence):
                                    logger.info(
                                        "Skipping {} {} — conflicting position exists",
                                        symbol, sig_direction,
                                    )
                                    continue

                                # Check per-symbol cooldown after recent position close
                                risk_cfg = getattr(self.settings, "risk", None)
                                cooldown_secs = getattr(risk_cfg, "symbol_cooldown_seconds", 60)
                                last_closed = self._symbol_last_closed.get(symbol, 0.0)
                                if last_closed > 0 and (time.time() - last_closed) < cooldown_secs:
                                    remaining = cooldown_secs - (time.time() - last_closed)
                                    logger.info(
                                        "Skipping {} — in cooldown ({:.0f}s remaining)",
                                        symbol, remaining,
                                    )
                                    continue

                                market_type = signal.get("market_type", "futures")

                                # Forex-specific risk validation
                                if self.forex_risk_manager is not None and market_type == "forex":
                                    forex_exchange = self.forex_exchange or self.exchange
                                    if forex_exchange is not None:
                                        try:
                                            spread_info = await forex_exchange.get_spread(symbol)
                                            spread_pips = spread_info.get("spread_pips", 0.0)
                                        except Exception as exc:
                                            logger.debug(f"Could not get spread for {symbol}: {exc}")
                                            spread_pips = 0.0

                                        # Get equity from account balance
                                        equity = 0.0
                                        try:
                                            balance_info = await forex_exchange.get_balance()
                                            if hasattr(balance_info, "usdt_total"):
                                                equity = float(balance_info.usdt_total)
                                            else:
                                                equity = float(balance_info.get("total", 0.0))
                                        except Exception as exc:
                                            logger.debug(f"Could not get forex equity: {exc}")

                                        forex_cfg = getattr(self.settings, "forex", None)
                                        fx_leverage = getattr(forex_cfg, "default_leverage", 20) if forex_cfg else 20

                                        forex_approval = self.forex_risk_manager.validate_forex_trade(
                                            symbol=symbol,
                                            direction=signal.get("direction", "long"),
                                            equity=equity,
                                            current_price=float(signal.get("entry_price", 0.0)),
                                            spread_pips=spread_pips,
                                            atr=float(signal.get("atr", 0.0)),
                                            leverage=fx_leverage,
                                        )
                                        if not forex_approval.approved:
                                            logger.info(
                                                "Forex trade rejected: {} — {}",
                                                symbol,
                                                forex_approval.rejection_reason,
                                            )
                                            continue

                                        # Enrich signal with forex approval data
                                        signal = {
                                            **signal,
                                            "lot_size": forex_approval.lot_size,
                                            "stop_loss": forex_approval.stop_loss_price,
                                            "take_profit": forex_approval.take_profit_price,
                                            "leverage": forex_approval.leverage,
                                            "margin_required": forex_approval.margin_required,
                                        }

                                if self.risk_manager is not None:
                                    approval = await self.risk_manager.validate_trade(signal)
                                    if not approval.approved:
                                        logger.info(
                                            f"Signal rejected: {signal.get('symbol')} "
                                            f"{signal.get('direction')} "
                                            f"— reason: {approval.rejection_reason}"
                                        )
                                        continue

                                    logger.debug(
                                        f"Signal approved: {signal.get('symbol')} "
                                        f"size={approval.position_size:.2f} USDT "
                                        f"sl={approval.stop_loss:.4f} leverage={approval.leverage}x"
                                    )

                                    approved_signal = {
                                        **signal,
                                        "position_size": approval.position_size,
                                        "stop_loss": approval.stop_loss,
                                        "take_profit_levels": approval.take_profit_levels,
                                        "leverage": approval.leverage,
                                    }
                                else:
                                    approved_signal = signal

                                # Task 8: Portfolio correlation check
                                if (
                                    hasattr(self, "_portfolio_risk_manager")
                                    and self._portfolio_risk_manager is not None
                                ):
                                    try:
                                        _existing_positions = []
                                        if self.position_manager is not None:
                                            _all_trackers = await self.position_manager.get_all_positions()
                                            _existing_positions = [
                                                {
                                                    "symbol": t.position.symbol,
                                                    "amount": t.position.amount,
                                                    "entry_price": t.position.entry_price,
                                                    "side": t.position.side.value,
                                                }
                                                for t in _all_trackers
                                            ]
                                        _equity = await self._get_current_equity()
                                        _should_reduce, _adj_size, _reason = (
                                            self._portfolio_risk_manager.should_reduce_new_position(
                                                new_symbol=symbol,
                                                new_size_usdt=approved_signal.get("position_size", 0),
                                                existing_positions=_existing_positions,
                                                equity=_equity,
                                            )
                                        )
                                        if _should_reduce:
                                            approved_signal = dict(approved_signal)
                                            approved_signal["position_size"] = _adj_size
                                            logger.info("Portfolio risk adjustment: {}", _reason)
                                    except Exception as _prm_exc:
                                        logger.debug("Portfolio risk check error: {}", _prm_exc)

                                if self.trade_executor is not None:
                                    result = await self.trade_executor.execute_trade(approved_signal)
                                    if result.get("success"):
                                        _trades += 1
                                        logger.info(
                                            f"Trade executed: {result.get('symbol')} "
                                            f"{result.get('direction')} "
                                            f"order_id={result.get('order_id')}"
                                        )
                                    else:
                                        logger.error(f"Trade execution failed for {symbol}: {result.get('error')}")

                                    if result.get("success"):
                                        trade_record = {
                                            "symbol": signal.get("symbol", ""),
                                            "direction": signal.get("direction", ""),
                                            "entry_price": result.get("filled_price", 0.0),
                                            "size": result.get("size", 0.0),
                                            "size_usdt": result.get("size_usdt", 0.0),
                                            "leverage": result.get("leverage", 1),
                                            "strategy": signal.get("strategy", "unknown"),
                                            "order_id": result.get("order_id", ""),
                                        }
                                        tj = getattr(self, "trade_journal", None)
                                        if tj is not None:
                                            try:
                                                tj.record_entry(
                                                    trade_record,
                                                    signal.get("reasoning", ""),
                                                    {},
                                                )
                                            except Exception as rec_exc:
                                                logger.debug(f"trade_journal.record_entry error: {rec_exc}")
                                        mc = getattr(self, "metrics_collector", None)
                                        if mc is not None:
                                            try:
                                                mc.record_trade(trade_record)
                                            except Exception as rec_exc:
                                                logger.debug(f"metrics_collector.record_trade error: {rec_exc}")
                                        if self.trade_tracker is not None:
                                            try:
                                                from core.trade_tracker import TrackedTrade

                                                tracked = TrackedTrade(
                                                    trade_id=result.get("order_id", ""),
                                                    symbol=signal.get("symbol", ""),
                                                    side=signal.get("direction", ""),
                                                    entry_price=result.get("filled_price", 0.0),
                                                    entry_time=datetime.now(tz=timezone.utc),
                                                    amount=result.get("size", 0.0),
                                                    leverage=result.get("leverage", 1),
                                                    strategy=signal.get("strategy", "unknown"),
                                                    market_type=signal.get("market_type", "futures"),
                                                )
                                                await self.trade_tracker.register_trade(tracked)
                                            except Exception as tt_exc:
                                                logger.debug(f"trade_tracker.register_trade error: {tt_exc}")
                                    if self.alert_manager is not None:
                                        try:
                                            # Enrich result with signal context for detailed alert
                                            alert_data = dict(result)
                                            alert_data.setdefault("strategy", signal.get("strategy", ""))
                                            alert_data["stop_loss"] = approved_signal.get("stop_loss")
                                            alert_data["take_profit_levels"] = approved_signal.get("take_profit_levels", [])
                                            alert_data["take_profit"] = (
                                                approved_signal.get("take_profit_levels", [None])[0]
                                                if approved_signal.get("take_profit_levels")
                                                else approved_signal.get("take_profit")
                                            )
                                            alert_data["leverage"] = approved_signal.get("leverage", result.get("leverage", 1))
                                            alert_data["position_size"] = approved_signal.get("position_size", 0)
                                            alert_data["mode"] = self.settings.trading_mode
                                            alert_data["exchange"] = getattr(self.exchange, "name", "unknown")
                                            await self.alert_manager.send_trade_open_alert_and_pin(
                                                alert_data,
                                                mode=self.settings.trading_mode,
                                            )
                                        except Exception as alert_exc:
                                            logger.debug(f"Trade alert error: {alert_exc}")

                            except CircuitBreakerError:
                                logger.critical(
                                    f"Circuit breaker triggered for {symbol} — stopping cycle"
                                )
                                raise
                            except Exception as exc:
                                logger.error(f"Trade execution error for signal {signal}: {exc}")

                        # g. Update trailing stops for open positions on this symbol
                        if self.position_manager is not None and ticker is not None:
                            try:
                                await self.position_manager.update_trailing_stop(symbol, ticker.last)
                            except Exception as exc:
                                logger.debug(f"Trailing stop update for {symbol}: {exc}")

                    except asyncio.CancelledError:
                        raise
                    except CircuitBreakerError:
                        raise
                    except Exception as exc:
                        logger.error(f"Error processing {symbol} in cycle #{cycle_num}: {exc}")

                    return _signals, _trades

            # Execute all symbols in parallel (up to _API_SEMAPHORE concurrency)
            gather_results = await asyncio.gather(
                *[_process_symbol(sym) for sym in trading_pairs],
                return_exceptions=True,
            )

            # Aggregate results; honour circuit breaker if raised by any symbol
            for gather_result in gather_results:
                if isinstance(gather_result, CircuitBreakerError):
                    if self.state_manager is not None:
                        await self.state_manager.update_state(circuit_breaker_active=True)
                    return
                if isinstance(gather_result, tuple):
                    signals_processed += gather_result[0]
                    trades_executed += gather_result[1]

        # 7. Sync open positions from exchange
        positions = []
        if self.exchange is not None:
            try:
                positions = await self.exchange.get_positions()
            except Exception as exc:
                logger.error(f"Failed to sync positions from exchange: {exc}")

        # 7b. Detect closed positions by comparing with the previous cycle snapshot
        #     and record trade results in the risk manager and strategy manager.
        current_position_keys = {
            f"{p.symbol}:{p.side.value}" for p in positions
        }
        for key, prev in list(self._previous_positions.items()):
            if key not in current_position_keys:
                # Position was closed between cycles — record the result.
                symbol_closed = prev.get("symbol", "")
                pnl = float(prev.get("unrealized_pnl", 0.0))
                entry = float(prev.get("entry_price", 0.0))
                amount = float(prev.get("amount", 0.0))
                position_value = entry * amount
                pnl_pct = (pnl / position_value) if position_value > 0 else 0.0
                trade_id = f"{symbol_closed}_{prev.get('side', '')}_{cycle_num}"
                strategy_name = prev.get("strategy", "unknown")
                regime = self.current_market_regime
                try:
                    if self.risk_manager is not None:
                        await self.risk_manager.record_trade_result(pnl, trade_id, pnl_pct)
                    if self.strategy_manager is not None:
                        if pnl >= 0:
                            self.strategy_manager.record_win(strategy_name, pnl, regime)
                        else:
                            self.strategy_manager.record_loss(strategy_name, pnl, regime)

                        # Update RL optimizer with trade outcome
                        if self.strategy_manager._rl_optimizer is not None:
                            # Build trade outcome dict for RL update
                            trade_outcome = {
                                "pnl": pnl,
                                "position_size": position_value,
                                "atr": 1.0,  # Would need to fetch from market data
                                "max_drawdown": abs(min(0, pnl)),  # Simplified
                                "holding_time": 3600.0,  # Would track actual time
                                "expected_holding_time": 3600.0,
                                "spread_cost": 0.0,  # Would need order book data
                                "strategy_regime": strategy_name,  # Strategy's target regime
                                "actual_regime": regime,
                            }
                            # Build context (simplified - would use actual market data)
                            context = self.strategy_manager._build_rl_context(
                                regime=regime,
                                volatility_regime=self.current_volatility_regime,
                                market_data={},
                                order_flow_signal=None,
                            )
                            self.strategy_manager.update_rl_optimizer(
                                strategy_name=strategy_name,
                                trade_outcome=trade_outcome,
                                context=context,
                            )

                        # Feed outcome to the confidence calibrator
                        if hasattr(self.strategy_manager, "_confidence_calibrator"):
                            confidence_at_entry = float(prev.get("entry_confidence", 0.5))
                            self.strategy_manager._confidence_calibrator.record_outcome(
                                strategy_name=strategy_name,
                                confidence_at_entry=confidence_at_entry,
                                won=(pnl >= 0),
                            )

                    logger.info(
                        f"Closed position recorded: {key} pnl={pnl:+.4f} pnl_pct={pnl_pct:+.4f}"
                    )
                except Exception as exc:
                    logger.debug(f"Trade result recording error for {key}: {exc}")

        # Update previous positions snapshot for next cycle
        self._previous_positions = {
            f"{p.symbol}:{p.side.value}": {
                "symbol": p.symbol,
                "side": p.side.value,
                "amount": p.amount,
                "entry_price": p.entry_price,
                "unrealized_pnl": p.unrealized_pnl,
                "strategy": getattr(p, "strategy", "unknown"),
                "entry_confidence": getattr(p, "entry_confidence", 0.5),
            }
            for p in positions
        }
        # Guard against unbounded growth (e.g. after many symbol rotations)
        if len(self._previous_positions) > 1000:
            self._previous_positions.clear()

        # Purge _symbol_last_closed entries older than 1 hour to prevent unbounded growth
        _one_hour_ago = time.time() - 3600.0
        stale_keys = [
            k for k, v in self._symbol_last_closed.items() if v < _one_hour_ago
        ]
        for k in stale_keys:
            del self._symbol_last_closed[k]

        if self.risk_manager is not None:
            try:
                equity = await self._get_current_equity()
                position_dicts = [p.model_dump() for p in positions]
                self.risk_manager.update_market_state(
                    volatility_regime=self.current_volatility_regime,
                    market_regime=self.current_market_regime,
                    equity=equity,
                    open_positions=position_dicts,
                )
            except Exception as exc:
                logger.error(f"Failed to update risk manager market state: {exc}")

        # 9. Check and enforce SL/TP for paper trading mode
        is_paper = getattr(self.settings, "is_paper_trading", False)
        if self.position_manager is not None and is_paper:
            try:
                await self.position_manager._enforce_risk_overlays()
            except Exception as exc:
                logger.debug(f"Paper trading SL/TP enforcement error: {exc}")

        # 10. Check daily P&L limits (use explicit positive loss so that 0% never triggers)
        if self.state_manager is not None:
            state = await self.state_manager.get_state()
            max_daily_loss_pct = (
                getattr(self.settings.risk, "max_daily_loss_pct", 5.0)
                if hasattr(self.settings, "risk")
                else 5.0
            )
            daily_loss_pct = -state.daily_pnl_pct
            if daily_loss_pct > 0 and daily_loss_pct >= max_daily_loss_pct:
                logger.critical(
                    f"Daily loss limit reached: {daily_loss_pct:.2f}% >= "
                    f"{max_daily_loss_pct:.2f}%. Activating circuit breaker."
                )
                await self.state_manager.update_state(circuit_breaker_active=True)

        # 11. Emit cycle summary event for dashboard / monitoring
        if self.event_bus is not None:
            try:
                crash_state = self.crash_protector.get_state()
                await self.event_bus.publish_async(
                    EventType.SYSTEM_HEALTH,
                    {
                        "cycle": cycle_num,
                        "signals_processed": signals_processed,
                        "trades_executed": trades_executed,
                        "open_positions": len(positions),
                        "market_regime": self.current_market_regime,
                        "volatility_regime": self.current_volatility_regime,
                        "crash_level": crash_state.level.value,
                        "circuit_breaker": crash_state.circuit_breaker_active,
                    },
                )
            except Exception as exc:
                logger.debug(f"Event bus publish error: {exc}")

        # 12. State checkpoint — save every 5 minutes
        await self._maybe_save_checkpoint()

        # 13. Log cycle summary
        logger.info(
            f"Trading cycle #{cycle_num} complete — "
            f"symbols={len(trading_pairs)} "
            f"signals={signals_processed} "
            f"trades={trades_executed} "
            f"positions={len(positions)} "
            f"regime={self.current_market_regime} "
            f"volatility={self.current_volatility_regime}"
        )

    async def _detect_market_regime(self) -> None:
        """Detect market regime and volatility using the BTC/USDT 4h benchmark.

        Updates ``self.current_market_regime`` and ``self.current_volatility_regime``.
        Logs a prominent warning when the regime changes between cycles.
        """
        if self.exchange is None or self.regime_detector is None or self.volatility_analyzer is None:
            return

        try:
            btc_df = await self.exchange.get_ohlcv(_BTC_BENCHMARK, _REGIME_TIMEFRAME, limit=_REGIME_CANDLES)
            if btc_df is None or btc_df.empty:
                logger.debug("Could not fetch BTC 4h OHLCV for regime detection.")
                return

            if btc_df.index.name == "timestamp":
                btc_df = btc_df.reset_index()

            required_cols = {"open", "high", "low", "close", "volume"}
            if required_cols - set(btc_df.columns):
                logger.debug("BTC OHLCV missing columns for regime detection.")
                return

            # Detect market regime via ADX + trend analysis
            market_regime_enum = await self.regime_detector.detect_regime(btc_df)
            new_market_regime = _REGIME_LABEL_MAP.get(market_regime_enum, "unknown")

            # Detect volatility regime from close prices
            prices = btc_df["close"].tolist()
            new_volatility_regime = self.volatility_analyzer.detect_volatility_regime(
                _BTC_BENCHMARK, prices=prices
            )
            # Normalise "medium" → "normal" for consistency with risk system labels
            if new_volatility_regime == "medium":
                new_volatility_regime = "normal"

            # Log regime changes prominently
            if new_market_regime != self._previous_market_regime:
                logger.warning(
                    "⚡ MARKET REGIME CHANGE: {} → {} (volatility={})",
                    self._previous_market_regime,
                    new_market_regime,
                    new_volatility_regime,
                )
                self._previous_market_regime = new_market_regime

            self.current_market_regime = new_market_regime
            self.current_volatility_regime = new_volatility_regime

            logger.debug(
                "Regime detected: market={} volatility={}",
                self.current_market_regime,
                self.current_volatility_regime,
            )

        except Exception as exc:
            logger.warning(f"Regime detection failed: {exc} — keeping previous regime values.")

    async def _check_emergency_exposure_reduction(self) -> None:
        """Reduce all position sizes by 50% when a CRITICAL negative news event is detected.

        Scans the most recent data items from the aggregator.  An emergency is
        triggered when at least one item has ``urgency_score >= 0.9`` and a
        bearish sentiment (``relevance_score < 0.0`` or urgency hint keywords),
        or when the item's metadata indicates a CRITICAL impact level.
        """
        if self.data_aggregator is None or self.position_manager is None:
            return

        try:
            # Collect a broad set of recent items (not symbol-specific)
            recent_items = await self.data_aggregator.get_items_for_symbol("BTC/USDT", limit=20)

            is_emergency = False
            for item in recent_items:
                urgency = float(getattr(item, "urgency_score", 0.0))
                meta = getattr(item, "metadata", {}) or {}
                impact = str(meta.get("impact_level", "")).upper()
                direction = str(meta.get("direction", "")).upper()

                if impact == "CRITICAL" and direction in ("BEARISH", "NEGATIVE"):
                    is_emergency = True
                    logger.critical(
                        "🚨 CRITICAL negative news detected: '{}…' (source={})",
                        item.content[:80],
                        item.source_name,
                    )
                    break

                if urgency >= 0.9:
                    content_lower = item.content.lower()
                    crash_keywords = ["crash", "hack", "exploit", "ban", "seized", "collapse"]
                    if any(kw in content_lower for kw in crash_keywords):
                        is_emergency = True
                        logger.critical(
                            "🚨 High-urgency crash keyword detected in news: '{}…' (source={})",
                            item.content[:80],
                            item.source_name,
                        )
                        break

            if not is_emergency:
                return

            # Reduce all open position sizes by 50%
            logger.critical(
                "Emergency exposure reduction triggered — reducing all positions by 50%"
            )
            try:
                summary = await self.position_manager.get_position_summary()
                positions = summary.get("positions", [])
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    amount = float(pos.get("amount", 0.0))
                    if amount <= 0 or not symbol:
                        continue
                    reduce_amount = amount * 0.5
                    try:
                        await self.position_manager.reduce_position(symbol, reduce_amount)
                        logger.info(
                            f"Emergency reduce: {symbol} by {reduce_amount:.6f} (50% of {amount:.6f})"
                        )
                    except Exception as exc:
                        logger.warning(f"Could not reduce position for {symbol}: {exc}")
            except Exception as exc:
                logger.error(f"Emergency exposure reduction error: {exc}")

            # Send an alert
            if self.alert_manager is not None:
                try:
                    await self.alert_manager.send_alert(
                        "🚨 EMERGENCY: CRITICAL negative news detected. "
                        "All position sizes reduced by 50%."
                    )
                except Exception as exc:
                    logger.debug(f"Emergency alert send error: {exc}")

        except Exception as exc:
            logger.debug(f"_check_emergency_exposure_reduction error: {exc}")

    # ------------------------------------------------------------------
    # Internal — initialisation
    # ------------------------------------------------------------------

    async def _initialize_subsystems(self) -> None:
        """Initialise all core subsystems in dependency order."""
        self.event_bus = EventBus()
        self.state_manager = StateManager.get_instance()
        self.health_checker = HealthChecker()
        self.scheduler = TaskScheduler()
        self.scheduler.start()

        await self.state_manager.update_state(status=BotStatus.RUNNING)
        logger.info("Core subsystems initialised.")

        # Instantiate market analysis components
        if self.regime_detector is None:
            self.regime_detector = MarketRegimeDetector()
            logger.info("MarketRegimeDetector initialised.")
        if self.volatility_analyzer is None:
            self.volatility_analyzer = VolatilityAnalyzer()
            logger.info("VolatilityAnalyzer initialised.")
        _ai_enabled = self.settings.ai.enabled if self.settings.ai is not None else True
        if self.cross_asset_regime_detector is None and _ai_enabled:
            from ai.market_analyzer.cross_asset_regime_detector import (  # noqa: PLC0415
                CrossAssetRegimeDetector,
            )
            self.cross_asset_regime_detector = CrossAssetRegimeDetector()
            logger.info("CrossAssetRegimeDetector initialised.")
        elif not _ai_enabled:
            logger.info(
                "CrossAssetRegimeDetector skipped — AI features are disabled "
                "(USE_AI=off / AI__ENABLED=off)."
            )
        if self.order_flow_analyzer is None:
            self.order_flow_analyzer = OrderFlowAnalyzer()
            logger.info("OrderFlowAnalyzer initialised.")

        # Create and connect the exchange (paper or live via factory)
        await self._initialize_exchange()

        # Create and start the local order book cache (zero-latency WebSocket book)
        await self._initialize_local_orderbook()

        # Create RiskManager (unless one was already wired externally, e.g. in tests)
        if self.risk_manager is None:
            self.risk_manager = RiskManager(settings=self.settings)
            logger.info("RiskManager initialised.")

        # Initialize Forex subsystems if enabled
        if self.settings.enable_forex_trading:
            try:
                # Initialize forex exchange if trading mode is forex-specific
                trading_mode = getattr(self.settings, "trading_mode", "paper").lower()
                if trading_mode in ("forex_live", "forex_demo", "forex_exness_live", "forex_exness_demo", "forex_exness_paper"):
                    # Forex exchange is created via exchange_factory in _initialize_exchange()
                    # and assigned to self.exchange, so we can reference it
                    self.forex_exchange = self.exchange
                    logger.info("Forex exchange assigned from main exchange.")

                # Initialize ForexRiskManager
                if self.forex_risk_manager is None:
                    self.forex_risk_manager = ForexRiskManager(settings=self.settings)
                    logger.info("ForexRiskManager initialised.")
            except Exception as exc:
                logger.warning(f"Forex initialization failed: {exc} — Forex trading will be unavailable.")
        else:
            logger.debug("Forex trading is disabled (ENABLE_FOREX_TRADING=false).")

        # Create StrategyManager and register all built-in strategies
        # Initialize RL optimizer if enabled
        rl_optimizer = None
        try:
            from ai.reinforcement.rl_strategy_optimizer import RLStrategyOptimizer

            # Count strategies (46 as per spec)
            num_strategies = 46  # Will be validated after strategy registration
            rl_optimizer = RLStrategyOptimizer(
                num_strategies=num_strategies,
                context_dim=25,
                decay_factor=0.995,
                epsilon=0.1,
                warm_up_trades=100,
                top_k=7,
                min_trades_for_thompson=10,
            )
            logger.info("RLStrategyOptimizer initialised.")
        except Exception as exc:
            logger.warning(f"RLStrategyOptimizer initialisation failed: {exc} — using heuristic only")

        if self.strategy_manager is None:
            self.strategy_manager = StrategyManager(rl_optimizer=rl_optimizer)
            await self.strategy_manager.initialize()

        # Create TradeExecutor once exchange and managers are ready
        if (
            self.trade_executor is None
            and self.exchange is not None
            and self.order_manager is not None
            and self.position_manager is not None
        ):
            self.trade_executor = TradeExecutor(
                exchange=self.exchange,
                order_manager=self.order_manager,
                position_manager=self.position_manager,
                local_orderbook_manager=self.local_orderbook_manager,
            )
            logger.info("TradeExecutor initialised.")

        # Create and start TradeTracker for real-time position monitoring
        if self.trade_tracker is None and self.exchange is not None:
            from core.trade_tracker import TradeTracker

            self.trade_tracker = TradeTracker(
                exchange=self.exchange,
                position_manager=self.position_manager,
                settings=self.settings,
            )
            await self.trade_tracker.start()
            logger.info("TradeTracker initialised and started.")

        # Create AlertManager (unless one was already wired externally, e.g. in tests)
        if self.alert_manager is None:
            self.alert_manager = AlertManager(settings=self.settings)
            logger.info("AlertManager initialised.")

        # Register circuit breaker callbacks so it can close positions / send alerts
        if self.trade_executor is not None and self.order_manager is not None:
            self.risk_manager._circuit_breaker.register_callbacks(
                close_positions=self._cb_close_all_positions,
                cancel_orders=self._cb_cancel_all_orders,
                send_alert=self._cb_send_alert,
            )
            logger.info("Circuit breaker callbacks registered.")

        # Set starting equity for percentage-based P&L calculations
        await self._set_starting_equity()

        # Wire up the DataAggregator with CoinGecko (free, always on) and
        # CryptoPanic (only when an API key is configured).
        if self.data_aggregator is None:
            await self._initialize_data_aggregator()

        # Wire up the AI Brain when an LLM API key is present (optional).
        if self.ai_brain is None:
            await self._initialize_ai_brain()

        # Create and start the real-time data hub
        await self._initialize_realtime_hub()

        # Initialize profit maximization subsystems
        await self._initialize_profit_maximization()

        # Reconcile exchange state with the database before generating new signals.
        await self._reconcile_startup_state()

        # Initialize Economic Calendar Filter (Task 7)
        try:
            from risk.economic_filter import EconomicCalendarFilter

            _risk_cfg_ec = getattr(self.settings, "risk", None)
            _ec_enabled = getattr(_risk_cfg_ec, "enable_economic_filter", True)
            _ec_buffer = getattr(_risk_cfg_ec, "economic_event_buffer_minutes", 15)
            if _ec_enabled:
                self._economic_filter = EconomicCalendarFilter(
                    buffer_minutes_before=_ec_buffer,
                    buffer_minutes_after=_ec_buffer,
                )
                logger.info("EconomicCalendarFilter initialised.")
            else:
                self._economic_filter = None
        except Exception as exc:
            logger.warning("EconomicCalendarFilter initialisation failed: {}", exc)
            self._economic_filter = None

        # Initialize Portfolio Risk Manager (Task 8)
        try:
            from risk.portfolio_risk_manager import PortfolioRiskManager

            self._portfolio_risk_manager: Optional[Any] = PortfolioRiskManager(
                max_portfolio_risk_pct=5.0,
                max_correlated_exposure_pct=3.0,
            )
            logger.info("PortfolioRiskManager initialised.")
        except Exception as exc:
            logger.warning("PortfolioRiskManager initialisation failed: {}", exc)
            self._portfolio_risk_manager = None

        # Initialize AntiLiquidationManager
        try:
            from risk.anti_liquidation import AntiLiquidationManager  # noqa: PLC0415

            self.anti_liquidation = AntiLiquidationManager()
            logger.info("AntiLiquidationManager initialised.")
        except Exception as exc:
            logger.warning("AntiLiquidationManager initialisation failed: {}", exc)
            self.anti_liquidation = None

        # Update 1: Initialize WebSocketDataManager
        try:
            from exchange.ws_data_manager import WebSocketDataManager  # noqa: PLC0415

            trading_pairs: List[str] = getattr(
                getattr(self.settings, "exchange", None), "trading_pairs", []
            )
            if self.exchange is not None and trading_pairs:
                self.ws_data_manager = WebSocketDataManager(
                    exchange=self.exchange,
                    symbols=trading_pairs,
                )
                await self.ws_data_manager.start()
                logger.info(
                    "WebSocketDataManager initialised for {} symbol(s).",
                    len(trading_pairs),
                )
            else:
                logger.info(
                    "WebSocketDataManager skipped — no exchange or trading pairs configured."
                )
        except Exception as exc:
            logger.warning(
                "WebSocketDataManager initialisation failed: {} — REST fallback active.", exc
            )
            self.ws_data_manager = None

        # Update 2: Initialize MarginMonitor
        try:
            from risk.margin_monitor import MarginMonitor  # noqa: PLC0415

            if self.exchange is not None and self.position_manager is not None:
                self.margin_monitor = MarginMonitor(
                    exchange=self.exchange,
                    position_manager=self.position_manager,
                    alert_manager=self.alert_manager,
                    check_interval=5.0,
                )
                await self.margin_monitor.start()
                logger.info("MarginMonitor initialised.")
            else:
                logger.info(
                    "MarginMonitor skipped — exchange or position_manager not available."
                )
        except Exception as exc:
            logger.warning("MarginMonitor initialisation failed: {}", exc)
            self.margin_monitor = None

        # Upgrade 2: Initialise StatePersistence and restore state from DB
        try:
            self.state_persistence = StatePersistence(engine=self)
            await self.state_persistence.restore_state()
            logger.info("StatePersistence initialised and state restored.")
        except Exception as exc:
            logger.warning("StatePersistence initialisation failed: {} — continuing without state restore.", exc)
            self.state_persistence = None

    async def _initialize_local_orderbook(self) -> None:
        """Create and start the LocalOrderBookManager for zero-latency book access."""
        if self.exchange is None:
            return
        try:
            from exchange.local_orderbook import LocalOrderBookManager
            from exchange.paper_exchange import PaperExchange

            trading_pairs: List[str] = getattr(
                getattr(self.settings, "exchange", None), "trading_pairs", []
            )
            if not trading_pairs:
                logger.info("LocalOrderBookManager: no trading pairs configured — skipping.")
                return

            self.local_orderbook_manager = LocalOrderBookManager(
                exchange=self.exchange,
                symbols=trading_pairs,
            )
            await self.local_orderbook_manager.start()
            logger.info(
                "LocalOrderBookManager initialised with {} symbol(s).",
                len(trading_pairs),
            )

            # Wire into PaperExchange so its market orders can use local books
            if isinstance(self.exchange, PaperExchange):
                self.exchange._local_orderbook_manager = self.local_orderbook_manager
                logger.info("LocalOrderBookManager wired into PaperExchange.")
        except Exception as exc:
            logger.warning(
                "LocalOrderBookManager initialisation failed: {} — continuing without it.", exc
            )

    async def _initialize_realtime_hub(self) -> None:
        """Create and start the RealtimeHub for live dashboard streaming."""
        try:
            from core.realtime_hub import RealtimeHub
            from exchange.paper_exchange import PaperExchange

            trading_pairs: List[str] = getattr(
                getattr(self.settings, "exchange", None), "trading_pairs", []
            )

            paper_exchange = self.exchange if isinstance(self.exchange, PaperExchange) else None

            self.realtime_hub = RealtimeHub(
                exchange=self.exchange,
                position_manager=self.position_manager,
                order_manager=self.order_manager,
                paper_exchange=paper_exchange,
            )
            if trading_pairs:
                await self.realtime_hub.start(trading_pairs)
                logger.info("RealtimeHub started with {} symbol(s).", len(trading_pairs))
            else:
                logger.info("RealtimeHub created but no trading pairs configured — not started.")
        except Exception as exc:
            logger.warning("RealtimeHub initialisation failed: {} — continuing without hub.", exc)

    async def _initialize_profit_maximization(self) -> None:
        """Initialize profit maximization subsystems (ParameterTuner, MetaLearner, ProfitMaximizer)."""
        try:
            from ai.profit_maximizer import ProfitMaximizer
            from ai.reinforcement.meta_learner import MetaLearner
            from ai.reinforcement.parameter_tuner import DynamicParameterTuner

            # Initialize ProfitMaximizer
            if self.profit_maximizer is None:
                self.profit_maximizer = ProfitMaximizer(settings=self.settings)
                logger.info("ProfitMaximizer initialized")

            # Initialize ParameterTuner
            if self.parameter_tuner is None:
                self.parameter_tuner = DynamicParameterTuner(settings=self.settings)
                logger.info("DynamicParameterTuner initialized")

            # Initialize MetaLearner with strategy names
            if self.meta_learner is None and self.strategy_manager is not None:
                strategy_names = []
                if hasattr(self.strategy_manager, "_strategies"):
                    strategy_names = [s.name for s in self.strategy_manager._strategies.values()]
                self.meta_learner = MetaLearner(strategy_names=strategy_names)
                logger.info("MetaLearner initialized with {} strategies", len(strategy_names))

        except Exception as exc:
            logger.warning("Profit maximization subsystems initialization failed: {} — continuing without them.", exc)

    async def _reconcile_startup_state(self) -> None:
        """Reconcile in-memory state with exchange and database on startup.

        Delegates to :class:`~core.reconciliation.StateReconciler`.  Failures
        are logged as warnings but never prevent the bot from starting.
        """
        if self.exchange is None or self.position_manager is None:
            logger.debug("Skipping state reconciliation: exchange or position_manager not ready.")
            return
        try:
            # Ensure database tables exist before querying
            from data.storage.models import create_tables
            await create_tables()
        except Exception as exc:
            logger.warning("Could not create database tables: {} — skipping reconciliation.", exc)
            return
        try:
            from core.reconciliation import StateReconciler

            reconciler = StateReconciler(
                exchange=self.exchange,
                position_manager=self.position_manager,
                risk_manager=self.risk_manager,
                alert_manager=self.alert_manager,
                settings=self.settings,
            )
            summary = await reconciler.reconcile_state()
            logger.info(
                "Startup reconciliation complete: %s",
                summary,
            )
        except Exception as exc:
            logger.warning("Startup state reconciliation failed: {} — continuing.", exc)

    async def _initialize_ai_brain(self) -> None:
        """Create the AI Brain when at least one LLM API key is configured.

        Supported providers (free-first fallback order):
          Ollama → Gemini Flash Lite → Gemini Flash → Grok → OpenRouter →
          OpenAI → Anthropic

        The AI Brain is **optional**: if no API key is configured or if
        AI is disabled via settings.ai.enabled, the method logs an
        informational message and returns without setting ``self.ai_brain``.
        The rest of the engine degrades gracefully — AI confidence modifiers
        are simply skipped.
        """
        # Check if AI is enabled
        ai_cfg = self.settings.ai
        ai_enabled = ai_cfg.enabled if ai_cfg is not None else True

        if not ai_enabled:
            logger.info(
                "AI features are disabled (USE_AI=off / AI__ENABLED=off). "
                "Bot will rely purely on technical strategies without AI analysis or news checking."
            )
            return

        openai_key = getattr(self.settings, "openai_api_key", None)
        anthropic_key = getattr(self.settings, "anthropic_api_key", None)
        gemini_key = getattr(self.settings, "gemini_api_key", None)
        grok_key = getattr(self.settings, "grok_api_key", None)
        openrouter_key = getattr(self.settings, "openrouter_api_key", None)
        gaterouter_key = getattr(self.settings, "gaterouter_api_key", None)

        any_cloud_key = any([openai_key, anthropic_key, gemini_key, grok_key, openrouter_key, gaterouter_key])
        if not any_cloud_key:
            logger.info(
                "No LLM API key configured (GATEROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "GEMINI_API_KEY, GROK_API_KEY, OPENROUTER_API_KEY) — "
                "AI market analysis will be skipped."
            )
            return

        try:
            from ai.brain import AIBrain
            from ai.llm_client import LLMClient

            llm_client = LLMClient(
                openai_api_key=openai_key or "",
                anthropic_api_key=anthropic_key or "",
                gemini_api_key=gemini_key or "",
                grok_api_key=grok_key or "",
                openrouter_api_key=openrouter_key or "",
                gaterouter_api_key=gaterouter_key or "",
                openai_model=getattr(ai_cfg, "openai_model", "gpt-4o"),
                anthropic_model=getattr(ai_cfg, "anthropic_model", "claude-3-5-sonnet-20241022"),
                gemini_flash_model=getattr(ai_cfg, "gemini_flash_model", "gemini-2.5-flash"),
                gemini_flash_lite_model=getattr(
                    ai_cfg, "gemini_flash_lite_model", "gemini-2.5-flash-lite"
                ),
                grok_model=getattr(ai_cfg, "grok_model", "grok-3-mini"),
                openrouter_model=getattr(
                    ai_cfg, "openrouter_model", "mistralai/mistral-7b-instruct:free"
                ),
                gaterouter_model=getattr(
                    ai_cfg, "gaterouter_model", "deepseek/deepseek-chat"
                ),
                # Always attempt free providers (Gemini, Grok, OpenRouter) first;
                # fall back to Ollama at the end when use_local_first=False.
                use_local_first=False,
            )
            self.ai_brain = AIBrain(llm_client=llm_client)

            # Initialize WalkForwardValidator for ML model validation
            try:
                from ai.prediction.walk_forward_validator import WalkForwardValidator

                self.walk_forward_validator = WalkForwardValidator(
                    train_window_months=6,
                    validation_window_months=1,
                    step_months=1,
                    min_samples=100,
                )
                logger.info("WalkForwardValidator initialized (6mo train, 1mo validate, 1mo step)")
            except Exception as val_exc:
                logger.warning("WalkForwardValidator initialization failed: {} — continuing without validation", val_exc)
                self.walk_forward_validator = None

            active_providers = [
                name
                for name, key in [
                    ("GateRouter", gaterouter_key),
                    ("Gemini", gemini_key),
                    ("Grok", grok_key),
                    ("OpenRouter", openrouter_key),
                    ("OpenAI", openai_key),
                    ("Anthropic", anthropic_key),
                ]
                if key
            ]
            logger.info(
                "AI Brain initialised. Active providers (fallback order): {}.",
                " → ".join(active_providers) if active_providers else "Ollama only",
            )
        except Exception as exc:
            logger.warning("Failed to initialise AI Brain: {} — continuing without AI.", exc)

    async def _initialize_data_aggregator(self) -> None:
        """Create the DataAggregator and register configured data sources.

        Data sources are optional — if any import or initialisation fails the
        engine logs a warning and continues without sentiment data.  The trading
        loop already handles ``self.data_aggregator is None`` gracefully.

        If AI is disabled (settings.ai.enabled=False), the data aggregator will
        not be initialized to avoid unnecessary API calls to news and sentiment sources.
        """
        # Check if AI is enabled - skip data aggregator if AI is disabled
        ai_enabled = self.settings.ai.enabled if self.settings.ai is not None else True

        if not ai_enabled:
            logger.info(
                "Data aggregator (news/sentiment sources) disabled because AI is disabled "
                "(USE_AI=off / AI__ENABLED=off). Enable AI to use news and sentiment analysis."
            )
            return

        try:
            from data.aggregator import DataAggregator
            from data.sources.coingecko_source import CoinGeckoSource
            from data.sources.cryptopanic_source import CryptoPanicSource

            aggregator = DataAggregator()

            # CoinGecko is always registered — it requires no API key
            aggregator.register_source(CoinGeckoSource())

            # CryptoPanic only when the operator has provided an API key
            cryptopanic_key = getattr(self.settings, "cryptopanic_api_key", None)
            if cryptopanic_key:
                aggregator.register_source(CryptoPanicSource(auth_token=cryptopanic_key))
                logger.info("CryptoPanic source registered.")
            else:
                logger.info("CRYPTOPANIC_API_KEY not set — CryptoPanic source will not be active.")

            # RSS news feeds — always active (no API key needed)
            try:
                from data.sources.news_rss_monitor import NewsRSSMonitor

                rss_feeds = getattr(
                    getattr(self.settings, "data_sources", None), "rss_feeds", []
                )
                aggregator.register_source(NewsRSSMonitor(feeds=rss_feeds))
                logger.info("RSS news source registered ({} feeds).", len(rss_feeds))
            except Exception as exc:
                logger.debug("RSS source not available: {}", exc)

            # Reddit — only when API credentials are configured
            reddit_client_id = getattr(self.settings, "reddit_client_id", None)
            reddit_client_secret = getattr(self.settings, "reddit_client_secret", None)
            if reddit_client_id and reddit_client_secret:
                try:
                    from data.sources.reddit_monitor import RedditMonitor

                    aggregator.register_source(
                        RedditMonitor(
                            client_id=reddit_client_id,
                            client_secret=reddit_client_secret,
                        )
                    )
                    logger.info("Reddit source registered.")
                except Exception as exc:
                    logger.debug("Reddit source not available: {}", exc)
            else:
                logger.info("REDDIT_CLIENT_ID/SECRET not set — Reddit source will not be active.")

            # Twitter/X — only when Bearer token is configured
            twitter_bearer = getattr(self.settings, "twitter_bearer_token", None)
            if twitter_bearer:
                try:
                    from data.sources.twitter_monitor import TwitterMonitor

                    aggregator.register_source(TwitterMonitor(bearer_token=twitter_bearer))
                    logger.info("Twitter source registered.")
                except Exception as exc:
                    logger.debug("Twitter source not available: {}", exc)
            else:
                logger.info(
                    "TWITTER_BEARER_TOKEN not set — Twitter source will not be active."
                )

            # Fear & Greed Index — no API key needed
            try:
                from data.sources.fear_greed_monitor import FearGreedMonitor

                aggregator.register_source(FearGreedMonitor())
                logger.info("Fear & Greed Index source registered.")
            except Exception as exc:
                logger.debug("Fear & Greed source not available: {}", exc)

            # Funding Rate monitor — no API key needed (uses public endpoints)
            try:
                from data.sources.funding_rate_monitor import FundingRateMonitor

                aggregator.register_source(FundingRateMonitor())
                logger.info("Funding Rate monitor registered.")
            except Exception as exc:
                logger.debug("Funding Rate source not available: {}", exc)

            self.data_aggregator = aggregator
            await aggregator.start_all_sources()
            logger.info(
                "DataAggregator initialised with {} enabled source(s).",
                aggregator.enabled_count,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialise DataAggregator: {} — continuing without data sources.", exc
            )

    async def _initialize_exchange(self) -> None:
        """Create the exchange via the factory and wire dependent managers.

        Retries the connection up to 3 times before giving up.  In live mode a
        failed connection is fatal — the bot exits with a clear error message.
        In paper mode a failure is non-fatal; the engine continues without a
        live price feed (useful for offline unit tests).
        """
        _MAX_RETRIES = 3
        _RETRY_DELAY = 5  # seconds between attempts

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                exchange = create_exchange(self.settings)
                await exchange.connect()
                self.exchange = exchange
                logger.info(
                    "Exchange initialised: {} (mode={}, attempt={}/{})",
                    exchange.name,
                    getattr(self.settings, "trading_mode", "paper"),
                    attempt,
                    _MAX_RETRIES,
                )
                break
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Exchange connection attempt {}/{} failed: {} — retrying in {}s…",
                        attempt,
                        _MAX_RETRIES,
                        exc,
                        _RETRY_DELAY,
                    )
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    mode = getattr(self.settings, "trading_mode", "paper").lower()
                    if mode == "live":
                        logger.critical(
                            "Exchange connection failed after {} attempts: {}\n"
                            "Cannot start live trading without an exchange connection. "
                            "Check your API keys, network connectivity, and exchange status, "
                            "then restart the bot.",
                            _MAX_RETRIES,
                            exc,
                        )
                        raise SystemExit(1) from exc
                    else:
                        logger.error(
                            "Exchange connection failed after {} attempts: {} — "
                            "continuing without live exchange (paper trading only).",
                            _MAX_RETRIES,
                            exc,
                        )
                        return

        # Wire OrderManager and PositionManager with the connected exchange
        self.order_manager = OrderManager(exchange=self.exchange)
        self.position_manager = PositionManager(exchange=self.exchange)
        logger.info("OrderManager and PositionManager wired to exchange")

    # ------------------------------------------------------------------
    # Internal — AI helpers
    # ------------------------------------------------------------------

    async def _get_ai_market_analysis(self, trading_pairs: List[str]) -> Optional[Dict]:
        """Return a cached or fresh AI market assessment.

        The analysis is called **once per trading cycle** and cached for 15
        minutes to minimise API costs.  Returns ``None`` when the AI brain is
        not configured or when the LLM call fails.

        Args:
            trading_pairs: Active symbol list passed into the market overview.

        Returns:
            Dict with ``"direction"``, ``"confidence"``, ``"key_levels"``, and
            ``"risk_assessment"`` — or ``None``.
        """
        if self.ai_brain is None:
            return None

        now = datetime.now(tz=timezone.utc)
        if self._ai_analysis_cache is not None:
            cached_at, cached_result = self._ai_analysis_cache
            age_seconds = (now - cached_at).total_seconds()
            if age_seconds < self._AI_CACHE_TTL_SECONDS:
                logger.debug("AI market analysis: using cached result (age={:.0f}s)", age_seconds)
                return cached_result

        try:
            result = await self.ai_brain.get_confidence_modifier(
                market_overview={"symbols": trading_pairs},
                sentiment_score=0.0,
                news_summary="",
            )
            if result is not None:
                self._ai_analysis_cache = (now, result)
                logger.info(
                    "AI market analysis: direction={} confidence={:.2f}",
                    result.get("direction", "unknown"),
                    float(result.get("confidence", 0.0)),
                )
            return result
        except Exception as exc:
            logger.warning("AI market analysis error: {} — skipping AI modifier", exc)
            return None

    @staticmethod
    def _apply_ai_confidence_modifier(signal: Dict, ai_signal: Dict) -> float:
        """Apply the AI directional view as a ±15% confidence modifier.

        Delegates to :func:`ai.brain.apply_ai_confidence_modifier` so that the
        pure function can be imported and tested without the full engine
        dependency chain.

        The AI assessment is a **confidence modifier only** — it never alters
        position sizes, stop-losses, or take-profits, and does not override the
        risk manager's decisions.
        """
        from ai.brain import apply_ai_confidence_modifier

        return apply_ai_confidence_modifier(signal, ai_signal)

    async def _start_background_tasks(self) -> None:
        """Launch all long-running background async tasks."""
        asyncio.create_task(self._health_check_loop(), name="health_check_loop")
        asyncio.create_task(self._daily_reset_loop(), name="daily_reset_loop")

        # Start the paper-trading SL/TP monitor when using the paper exchange.
        try:
            from exchange.paper_exchange import PaperExchange

            if isinstance(self.exchange, PaperExchange):
                asyncio.create_task(
                    self.exchange.run_sl_tp_monitor(), name="paper_sl_tp_monitor"
                )
                logger.info("Paper trading SL/TP monitor started.")
        except Exception as exc:
            logger.debug(f"Could not start paper SL/TP monitor: {exc}")

        # Periodically cancel stale entry limit orders to prevent orphaned orders
        # from triggering at unexpected times (e.g. during sudden price moves).
        asyncio.create_task(self._orphan_order_cleanup_loop(), name="orphan_order_cleanup")

        # Start dashboard alongside the trading engine
        asyncio.create_task(self._start_dashboard(), name="dashboard")

        # Task 1B: Unprotected position watchdog
        if self.position_manager is not None:
            asyncio.create_task(
                self.position_manager.watchdog_unprotected_positions(),
                name="unprotected_position_watchdog",
            )
            logger.info("Unprotected position watchdog task started.")

        # Task 6: WebSocket-driven position monitor (falls back to REST polling)
        if self.position_manager is not None and self.exchange is not None:
            asyncio.create_task(
                self.position_manager.start_ws_monitor(self.exchange),
                name="position_ws_monitor",
            )
            logger.info("WebSocket position monitor task started.")

        logger.info("Background tasks started.")

    async def _start_dashboard(self) -> None:
        """Launch the monitoring dashboard as a background task."""
        try:
            from monitoring.dashboard import TradingDashboard

            port = (
                self.settings.monitoring.dashboard_port
                if hasattr(self.settings, "monitoring")
                else 8080
            )
            dashboard = TradingDashboard(
                settings=self.settings,
                engine=self,
                realtime_hub=self.realtime_hub,
            )
            self._dashboard = dashboard
            logger.info("Starting dashboard on 0.0.0.0:{}", port)
            await dashboard.run(host="0.0.0.0", port=port)
        except Exception as exc:
            logger.error("Dashboard failed to start: {}", exc)

    def _register_signal_handlers(self) -> None:
        """Register SIGINT / SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

    # ------------------------------------------------------------------
    # Internal — background loops
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Periodic health checks every 60 seconds."""
        while self._running:
            try:
                await self.health_checker.run_all_checks()
                report = await self.health_checker.get_health_report()
                logger.debug(f"Health: {report['overall']}")

                if self.event_bus is not None:
                    await self.event_bus.publish_async(EventType.SYSTEM_HEALTH, report)
            except Exception as exc:
                logger.error(f"Health check loop error: {exc}")
            await asyncio.sleep(60)

    async def _daily_reset_loop(self) -> None:
        """Reset daily statistics at UTC midnight."""
        while self._running:
            try:
                wait_seconds = time_until_midnight_utc()
                logger.info(f"Daily reset scheduled in {wait_seconds:.0f}s")
                await asyncio.sleep(wait_seconds)

                # Log daily summary before resetting
                if self.state_manager is not None:
                    state = await self.state_manager.get_state()
                    logger.info(
                        f"Daily summary — trades={state.trade_count_today} "
                        f"wins={state.win_count_today} losses={state.loss_count_today} "
                        f"pnl={state.daily_pnl:+.4f} ({state.daily_pnl_pct:+.2f}%)"
                    )
                    # Send end-of-day alert
                    if self.alert_manager is not None:
                        summary = {
                            "date": str(date.today()),
                            "total_pnl": state.daily_pnl,
                            "win_rate": (
                                state.win_count_today / state.trade_count_today * 100.0
                                if state.trade_count_today
                                else 0.0
                            ),
                            "trade_count": state.trade_count_today,
                        }
                        try:
                            await self.alert_manager.send_daily_report(summary)
                        except Exception as alert_exc:
                            logger.debug(f"Daily report alert error: {alert_exc}")

                # Reset circuit breaker if it was triggered by daily loss limits only
                if (
                    self.risk_manager is not None
                    and self.risk_manager._circuit_breaker.is_triggered()
                ):
                    trigger_reason = self.risk_manager._circuit_breaker.trigger_info.get(
                        "reason", ""
                    )
                    # Only auto-reset if triggered by daily loss limits (not market crashes
                    # or API errors)
                    daily_loss_reasons = ["daily loss", "daily trading limit"]
                    is_daily_loss_trigger = any(
                        kw in (trigger_reason or "").lower() for kw in daily_loss_reasons
                    )
                    if is_daily_loss_trigger:
                        await self.risk_manager._circuit_breaker.reset()
                        logger.info(
                            "Circuit breaker reset at daily reset "
                            "(was triggered by daily loss limit)."
                        )
                    else:
                        logger.warning(
                            "Circuit breaker NOT reset at daily reset — triggered by '{}'. "
                            "Manual reset required.",
                            trigger_reason,
                        )

                # Reset daily stats in state manager
                if self.state_manager is not None:
                    await self.state_manager.reset_daily_stats()
                    logger.info("Daily stats reset at midnight UTC.")

                # Refresh starting equity after the daily reset
                await self._set_starting_equity()
            except Exception as exc:
                logger.error(f"Daily reset loop error: {exc}")
                await asyncio.sleep(60)

    async def _orphan_order_cleanup_loop(self) -> None:
        """Cancel stale entry limit orders every 5 minutes.

        Limit entry orders that remain unfilled for more than 30 minutes are
        cancelled automatically.  This prevents orphaned orders from triggering
        at unexpected prices during sudden market moves while the bot is alive.
        """
        _INTERVAL_SECONDS = 300  # 5 minutes
        _MAX_AGE_MINUTES = 30
        while self._running:
            await asyncio.sleep(_INTERVAL_SECONDS)
            try:
                if self.trade_executor is not None:
                    cancelled = await self.trade_executor.cancel_stale_entry_orders(
                        max_age_minutes=_MAX_AGE_MINUTES
                    )
                    if cancelled:
                        logger.info(
                            "Orphan order cleanup: cancelled {} stale limit order(s).", cancelled
                        )
            except Exception as exc:
                logger.debug("Orphan order cleanup error: {}", exc)

    # ------------------------------------------------------------------
    # Internal — equity helpers
    # ------------------------------------------------------------------

    async def _get_current_equity(self) -> float:
        """Return the current portfolio equity.

        For live mode, queries the exchange balance.  For paper mode (or when
        the exchange is unavailable), falls back to the configured paper
        trading balance.

        Returns:
            Current portfolio equity in quote currency.
        """
        paper_balance: float = getattr(self.settings, "paper_trading_balance", 10_000.0)
        if self.exchange is not None:
            try:
                balance = await self.exchange.get_balance()
                equity = balance.usdt_total if balance.usdt_total else balance.usdt_free
                if equity:
                    logger.debug(f"Equity fetched from exchange: {equity:.2f}")
                    return equity
            except Exception as exc:
                logger.error(f"Failed to fetch equity from exchange, using paper balance: {exc}")
        return paper_balance

    async def _set_starting_equity(self) -> None:
        """Set today's starting equity on the DailyPnLManager and StateManager.

        Called once at startup and again after each daily reset so that
        percentage-based P&L calculations use the correct denominator.
        """
        if self.risk_manager is None:
            logger.debug("Risk manager not wired — skipping starting equity setup.")
            return

        try:
            equity = await self._get_current_equity()
            # DailyPnLManager uses starting equity for percentage calculations
            self.risk_manager._daily_pnl.set_starting_equity(equity)
            # Also seed _portfolio_equity so the first trading cycle can size positions
            # correctly without waiting for update_market_state (called at end of cycle).
            self.risk_manager.update_market_state(
                volatility_regime=self.current_volatility_regime,
                market_regime=self.current_market_regime,
                equity=equity,
                open_positions=[],
            )
            # StateManager tracks the total balance for its own pnl_pct calc
            if self.state_manager is not None:
                await self.state_manager.update_state(total_balance=equity)
            logger.info(f"Starting equity initialised: {equity:.2f}")
        except Exception as exc:
            logger.error(f"Failed to set starting equity: {exc}")

    # ------------------------------------------------------------------
    # Internal — circuit breaker callbacks
    # ------------------------------------------------------------------

    async def _cb_close_all_positions(self) -> None:
        """Emergency callback: close all open positions on circuit breaker trigger."""
        if self.trade_executor is not None:
            await self.trade_executor.close_all_positions("circuit_breaker")

    async def _cb_cancel_all_orders(self) -> None:
        """Emergency callback: cancel all open orders on circuit breaker trigger."""
        if self.order_manager is not None:
            await self.order_manager.cancel_all_orders()

    async def _cb_send_alert(self, message: str) -> None:
        """Emergency callback: send an alert on circuit breaker trigger."""
        logger.warning(f"Circuit breaker alert: {message}")
        if self.alert_manager is not None:
            try:
                await self.alert_manager.send_alert(message, level="critical")
            except Exception as exc:
                logger.error(f"Failed to send circuit breaker alert: {exc}")

    # ------------------------------------------------------------------
    # Data integrity helpers (MR2: smart reconnection & data integrity)
    # ------------------------------------------------------------------

    def _is_data_stale(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> bool:
        """Return True if the most recent candle is older than the stale threshold.

        The threshold is timeframe-aware: for 15m data it is 900+60s, for 1h
        data 3600+60s, etc.  Unknown timeframes fall back to 60 s.

        Args:
            df: OHLCV DataFrame (should have a ``timestamp`` column or be indexed by it).
            symbol: Symbol for logging.
            timeframe: Timeframe label for logging.

        Returns:
            ``True`` if data is stale.
        """
        try:
            now = datetime.now(tz=timezone.utc).timestamp()
            ts_col = None
            if "timestamp" in df.columns:
                ts_col = df["timestamp"]
            elif df.index.name == "timestamp":
                ts_col = df.index.to_series()

            if ts_col is None:
                return False  # can't determine staleness without a timestamp

            last_ts = float(ts_col.iloc[-1])
            # Handle millisecond timestamps
            if last_ts > 1e12:
                last_ts /= 1000.0

            threshold = _STALE_DATA_THRESHOLD_BY_TIMEFRAME.get(
                timeframe, _STALE_DATA_THRESHOLD_SECONDS
            )
            age = now - last_ts
            if age > threshold:
                logger.warning(
                    f"[DataIntegrity] Stale {timeframe} data for {symbol}: "
                    f"age={age:.0f}s > threshold={threshold}s"
                )
                return True
        except Exception as exc:
            logger.debug(f"[DataIntegrity] Staleness check failed for {symbol} {timeframe}: {exc}")
        return False

    def _has_data_gaps(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> bool:
        """Return True if the OHLCV data has unexpected candle gaps.

        A gap is detected when a consecutive candle pair differs by more than
        2× the expected candle interval.

        Args:
            df: OHLCV DataFrame with a timestamp column.
            symbol: Symbol for logging.
            timeframe: Timeframe label (e.g. ``"15m"``).

        Returns:
            ``True`` if gaps are detected.
        """
        try:
            _TF_SECONDS: Dict[str, int] = {
                "1m": 60, "3m": 180, "5m": 300, "15m": 900,
                "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
            }
            expected_interval = _TF_SECONDS.get(timeframe, 0)
            if expected_interval == 0:
                return False

            ts_col = None
            if "timestamp" in df.columns:
                ts_col = df["timestamp"].astype(float)
            elif df.index.name == "timestamp":
                ts_col = df.index.to_series().astype(float)

            if ts_col is None or len(ts_col) < 2:
                return False

            ts_values = ts_col.values
            # Handle millisecond timestamps
            if ts_values[-1] > 1e12:
                ts_values = ts_values / 1000.0

            diffs = [ts_values[i + 1] - ts_values[i] for i in range(len(ts_values) - 1)]
            max_gap = max(diffs) if diffs else 0
            if max_gap > expected_interval * _GAP_TOLERANCE_MULTIPLIER:
                logger.warning(
                    f"[DataIntegrity] Gap detected in {symbol} {timeframe}: "
                    f"max_gap={max_gap:.0f}s expected={expected_interval}s"
                )
                return True
        except Exception as exc:
            logger.debug(f"[DataIntegrity] Gap check failed for {symbol} {timeframe}: {exc}")
        return False

    async def _check_heartbeat(self) -> None:
        """Force reconnect if no successful API call has been made recently."""
        if self.exchange is None:
            return
        elapsed = time.time() - self._last_successful_api_call
        if elapsed > _HEARTBEAT_TIMEOUT_SECONDS:
            logger.warning(
                f"[Heartbeat] No successful API call for {elapsed:.0f}s — forcing reconnect"
            )
            try:
                await self.exchange.disconnect()
                await asyncio.sleep(2)
                await self.exchange.connect()
                self._last_successful_api_call = time.time()
                logger.info("[Heartbeat] Exchange reconnected successfully")
            except Exception as exc:
                logger.error(f"[Heartbeat] Reconnect failed: {exc}")

    async def _maybe_save_checkpoint(self) -> None:
        """Save engine state to disk if the checkpoint interval has elapsed."""
        now = time.time()
        if now - self._last_checkpoint_ts < _STATE_CHECKPOINT_INTERVAL_SECONDS:
            return
        try:
            checkpoint = {
                "timestamp": now,
                "cycle_count": self._cycle_count,
                "market_regime": self.current_market_regime,
                "volatility_regime": self.current_volatility_regime,
                "crash_level": self.crash_protector.get_current_level().value,
                "start_time": self._start_time.isoformat() if self._start_time else None,
            }
            checkpoint_path = _STATE_CHECKPOINT_FILE
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as fh:
                json.dump(checkpoint, fh, indent=2)
            self._last_checkpoint_ts = now
            logger.debug(f"[Checkpoint] State saved to {checkpoint_path}")
        except Exception as exc:
            logger.debug(f"[Checkpoint] Failed to save state: {exc}")

        # Upgrade 2: also persist full in-memory state to DB
        if self.state_persistence is not None:
            try:
                await self.state_persistence.save_state()
            except Exception as exc:
                logger.debug(f"[Checkpoint] StatePersistence.save_state failed: {exc}")

    async def _check_funding_rates(self) -> None:
        """Check funding rates for all open positions and record costs.

        Called from the fast cycle.  If a funding rate is extremely adverse
        (worse than ``max_funding_rate_tolerance``), logs a warning so the
        operator can act.  Actual funding payments are recorded via
        :meth:`~exchange.position_manager.PositionManager.record_funding_cost`.
        """
        if self.position_manager is None or self.exchange is None:
            return

        try:
            trackers = await self.position_manager.get_all_positions()
            if not trackers:
                return

            tolerance = getattr(
                getattr(self.settings, "risk", None), "max_funding_rate_tolerance", -0.05
            )

            for tracker in trackers:
                symbol = tracker.position.symbol
                try:
                    raw_funding = await self.exchange.get_funding_rate(symbol)
                    # get_funding_rate returns a float directly
                    if isinstance(raw_funding, dict):
                        rate = float(raw_funding.get("fundingRate", 0.0))
                    else:
                        rate = float(raw_funding) if raw_funding is not None else 0.0
                    rate_pct: float = rate * 100.0

                    # Record a snapshot on the tracker
                    await self.position_manager.record_funding_cost(symbol, 0.0)  # event-based

                    if rate_pct < tolerance:
                        pnl = tracker.position.unrealized_pnl
                        logger.warning(
                            "Adverse funding rate for {}: {:.4f}% (tolerance={:.4f}%) pnl={:.4f}",
                            symbol,
                            rate_pct,
                            tolerance,
                            pnl,
                        )
                        # Risk manager advisory check
                        if self.risk_manager is not None:
                            should_close = self.risk_manager.should_reduce_for_funding(
                                symbol, rate_pct, pnl
                            )
                            if should_close:
                                logger.warning(
                                    "Auto-closing {} due to adverse funding rate {:.4f}%",
                                    symbol,
                                    rate_pct,
                                )
                                try:
                                    if self.position_manager is not None:
                                        await self.position_manager.close_position(
                                            symbol, reason="adverse_funding_rate"
                                        )
                                    else:
                                        await self.exchange.close_position(symbol)
                                    if self.alert_manager is not None:
                                        await self.alert_manager.send_alert(
                                            f"⚠️ Auto-closed {symbol} due to adverse "
                                            f"funding rate: {rate_pct:.4f}%"
                                        )
                                except Exception as close_exc:
                                    logger.error(
                                        "Failed to auto-close {} for funding: {}",
                                        symbol,
                                        close_exc,
                                    )
                except Exception as exc:
                    logger.debug(f"Funding rate check error for {symbol}: {exc}")
        except Exception as exc:
            logger.debug(f"_check_funding_rates error: {exc}")


    def _should_send_alert(self, symbol: str, alert_type: str) -> bool:
        """Return True if enough time has elapsed since the last alert of this type for symbol.

        Uses the module-level *_ALERT_COOLDOWN_SECONDS* mapping to determine the
        minimum interval between repeated alerts.  Unknown alert types default to
        a 30-minute cooldown.
        """
        cooldown = _ALERT_COOLDOWN_SECONDS.get(alert_type, 1800)
        key = f"{symbol}:{alert_type}"
        last_sent = self._alert_cooldowns.get(key, 0.0)
        if time.time() - last_sent >= cooldown:
            self._alert_cooldowns[key] = time.time()
            return True
        return False

    async def _check_liquidation_proximity(self) -> None:
        """Check all open positions for liquidation proximity and alert/close as needed.

        Uses AntiLiquidationManager for sophisticated per-position and portfolio-level
        risk assessment, with automatic protective actions (close/reduce/alert).
        """
        if self.position_manager is None or self.exchange is None:
            return
        try:
            positions = await self.exchange.get_positions()
            if not positions:
                return

            equity = await self._get_current_equity()

            # Build position dicts for portfolio assessment
            position_dicts = []
            for pos in positions:
                position_dicts.append({
                    "symbol": pos.symbol,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price or pos.mark_price or pos.entry_price,
                    "leverage": pos.leverage,
                    "side": pos.side.value.lower(),
                    "size": pos.amount,
                    "margin": pos.margin,
                    "liquidation_price": pos.liquidation_price,
                })

            # Individual position screening
            if self.anti_liquidation is not None:
                results = self.anti_liquidation.screen_all_positions(position_dicts)
                for result in results:
                    action = result.get("recommended_action")
                    symbol = result.get("symbol", "")
                    risk_info = result.get("liquidation_risk", {})

                    if action == "close_position":
                        logger.critical("🚨 LIQUIDATION CRITICAL: {} - auto-closing!", symbol)
                        try:
                            await self.exchange.close_position(symbol)
                            if self.alert_manager and self._should_send_alert(symbol, "liquidation_critical"):
                                await self.alert_manager.send_liquidation_alert(
                                    {
                                        "symbol": symbol,
                                        "side": result.get("side", ""),
                                        "size": result.get("size", 0.0),
                                        "entry_price": result.get("entry_price", 0.0),
                                        "liquidation_price": risk_info.get("liquidation_price", 0.0),
                                        "margin_used": result.get("margin", 0.0),
                                        "leverage": result.get("leverage", 1),
                                        "dist_pct": risk_info.get("distance_pct", 0.0) * 100,
                                        "strategy": result.get("strategy", "—"),
                                    },
                                    mode=self.settings.trading_mode,
                                )
                        except Exception as exc:
                            logger.error("Failed to auto-close {}: {}", symbol, exc)

                    elif action == "reduce_50pct":
                        logger.warning(
                            "⚠️ LIQUIDATION DANGER: {} - reducing by 50%", symbol
                        )
                        try:
                            if self.trade_executor:
                                await self.trade_executor.execute_partial_close(
                                    symbol, 0.5, reason="anti_liquidation_reduce"
                                )
                            if self.alert_manager and self._should_send_alert(symbol, "liquidation_danger"):
                                dist_pct = risk_info.get("distance_pct", 0.0)
                                await self.alert_manager.send_typed_alert(
                                    AlertType.RISK_WARNING,
                                    {
                                        "warning": (
                                            f"⚠️ Liquidation DANGER for {symbol}: "
                                            f"reducing 50% — {dist_pct:.1%} from liquidation"
                                        ),
                                        "drawdown_pct": dist_pct * 100,
                                    },
                                )
                        except Exception as exc:
                            logger.error("Failed to reduce {}: {}", symbol, exc)

                    elif action == "alert":
                        dist_pct = risk_info.get("distance_pct", 0.0)
                        # Only send if within the warning distance threshold (default 40%)
                        # and cooldown has expired — prevents alert spam every 5 s
                        warning_threshold = getattr(
                            self.anti_liquidation, "_WARNING_DISTANCE_PCT", 0.40
                        )
                        if dist_pct <= warning_threshold and self.alert_manager and self._should_send_alert(symbol, "liquidation_warning"):
                            try:
                                await self.alert_manager.send_typed_alert(
                                    AlertType.RISK_WARNING,
                                    {
                                        "warning": (
                                            f"⚠️ Liquidation risk for {symbol}: "
                                            f"{dist_pct:.1%} from liquidation price"
                                        ),
                                        "drawdown_pct": dist_pct * 100,
                                    },
                                )
                            except Exception:
                                pass

                # Portfolio-level assessment
                portfolio_risk = self.anti_liquidation.assess_portfolio(
                    position_dicts, equity
                )
                if portfolio_risk.get("recommended_action") == "reduce_positions":
                    logger.warning(
                        "Portfolio risk elevated: margin={:.1%} leverage={:.2f}x",
                        portfolio_risk["margin_usage_pct"],
                        portfolio_risk["portfolio_leverage"],
                    )
        except Exception as exc:
            logger.debug("_check_liquidation_proximity error: {}", exc)

    def _has_conflicting_position(
        self,
        symbol: str,
        direction: str,
        signal_confidence: float = 0.0,
    ) -> bool:
        """Return True if there is any open position for *symbol*.

        Any existing position for a symbol — whether same-direction or
        opposite — blocks a new trade.  On futures exchanges, sending a
        same-direction order against an existing position increases the size
        of that position rather than opening a separate one, so the
        ``max_open_positions`` counter never increments and the bot would
        grow the position indefinitely.  Blocking unconditionally prevents
        position stacking.

        Note: Symbol lookup normalises by stripping the ``:USDT`` swap suffix so
        the check works regardless of whether the symbol is stored as
        ``SOL/USDT`` or ``SOL/USDT:USDT``.
        """
        if self.position_manager is None:
            return False
        try:
            # Normalise: strip swap suffix for consistent comparison.
            norm_symbol = symbol.split(":")[0]
            tracker = self.position_manager.get_position_sync(norm_symbol)
            if tracker is None:
                # Also try with the swap suffix in case positions are keyed that way.
                base, _, quote = norm_symbol.partition("/")
                if quote:
                    swap_symbol = f"{norm_symbol}:{quote}"
                    tracker = self.position_manager.get_position_sync(swap_symbol)
            if tracker is None:
                return False
            # Any open position for this symbol blocks a new trade.
            from exchange.base_exchange import PositionSide
            existing_side = getattr(
                getattr(tracker, "position", None), "side", None
            )
            existing_label = (
                "LONG" if existing_side == PositionSide.LONG else "SHORT"
            ) if existing_side else "UNKNOWN"
            logger.info(
                "Skipping {} signal for {} — {} position already open (no stacking)",
                direction, symbol, existing_label,
            )
            return True
        except Exception as exc:
            logger.debug("_has_conflicting_position error for {}: {}", symbol, exc)
        return False
