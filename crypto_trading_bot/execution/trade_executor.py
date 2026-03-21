"""Trade executor — position reconciliation and slow-loop lifecycle management.

In the upgraded architecture all **real-time** order placement (entry, SL, TP)
is performed by the Rust ``trading_engine`` binary (see
``rust_engine/src/execution_gateway.rs``).  This module is now responsible
only for:

* **Reconciliation**: comparing Rust-reported positions against exchange REST
  state and emitting alerts on discrepancies.
* **Slow-loop lifecycle**: leverage/margin setup, position book-keeping, and
  historical PnL recording — tasks that do not need sub-millisecond latency.
* **Back-testing / paper-trading**: the full ``execute_trade`` pipeline is
  retained so that strategies can be simulated without the Rust binary.

If you need to force a Python-path live trade (e.g. for an emergency close),
call :meth:`execute_trade` directly.  A ``DeprecationWarning`` will be emitted
to remind callers that the Rust path is preferred.
"""

from __future__ import annotations

import asyncio
import math
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange
    from exchange.order_manager import OrderManager
    from exchange.position_manager import PositionManager

from config.settings import Settings
from exchange.base_exchange import MarginType
from .adaptive_execution_engine import AdaptiveExecutionEngine
from .anti_gaming import AntiGamingProtection
from .execution_optimizer import ExecutionOptimizer
from .execution_quality_analyzer import ExecutionQualityAnalyzer
from .fee_calculator import FeeCalculator
from .gateio_fee_optimizer import GateioFeeOptimizer
from .latency_monitor import LatencyMonitor
from .order_flow_execution import OrderFlowExecutionEngine
from .slippage_estimator import SlippageEstimator
from .smart_entry import SmartEntryOptimizer
from .smart_exit_engine import SmartExitEngine

try:
    from rust_trading_engine.execution_prep import (
        MarketInfo as RustMarketInfo,
        FeeTable as RustFeeTable,
        compute_execution_plan,
    )
    _USE_RUST_EXEC = True
except ImportError:
    _USE_RUST_EXEC = False


class TradeExecutor:
    """Executes trades with smart order management.

    Execution flow:
    1. Receive signal
    2. Validate with risk module (caller's responsibility)
    3. Set leverage
    4. Place entry order (potentially split)
    5. Wait for fill
    6. Place SL / TP orders
    7. Monitor via PositionManager
    """

    # Maximum fraction of average volume per chunk when splitting
    _MAX_CHUNK_VOLUME_FRACTION = 0.02
    # Maximum individual TWAP delay in seconds (safety cap)
    _MAX_TWAP_DELAY_SECONDS: float = 30.0

    def __init__(
        self,
        exchange: BaseExchange,
        order_manager: OrderManager,
        position_manager: PositionManager,
        local_orderbook_manager: Optional[Any] = None,
        enable_advanced_execution: bool = True,
    ) -> None:
        self._exchange = exchange
        self._order_manager = order_manager
        self._position_manager = position_manager
        self._local_orderbook_manager = local_orderbook_manager
        self._optimizer = ExecutionOptimizer()
        self._fee_calc = FeeCalculator()
        self._slippage = SlippageEstimator()
        self._smart_entry = SmartEntryOptimizer()
        self._fee_optimizer = GateioFeeOptimizer()
        # Lazily initialised Telegram alerter
        self._telegram_alerter: Optional[Any] = None

        # Rust execution pre-computation engine (Phase 4)
        if _USE_RUST_EXEC:
            self._rust_fee_table = RustFeeTable()
        else:
            self._rust_fee_table = None

        # Advanced execution components (can be disabled for backwards compatibility)
        self._enable_advanced_execution = enable_advanced_execution
        if enable_advanced_execution:
            self._latency_monitor = LatencyMonitor(market_type="crypto")
            self._execution_quality_analyzer = ExecutionQualityAnalyzer()
            self._adaptive_engine = AdaptiveExecutionEngine(
                exchange=exchange,
                order_manager=order_manager,
                local_orderbook_manager=local_orderbook_manager
            )
            self._smart_exit_engine = SmartExitEngine(
                exchange=exchange,
                position_manager=position_manager
            )
            self._order_flow_engine = OrderFlowExecutionEngine(
                exchange=exchange,
                order_flow_analyzer=None  # Can be connected if available
            )
            self._anti_gaming = AntiGamingProtection(exchange=exchange)
            logger.info("TradeExecutor initialized with advanced execution features")
        else:
            self._latency_monitor = None
            self._execution_quality_analyzer = None
            self._adaptive_engine = None
            self._smart_exit_engine = None
            self._order_flow_engine = None
            self._anti_gaming = None
            logger.info("TradeExecutor initialized in compatibility mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_trade(self, signal: dict) -> dict:
        """Execute a validated trade signal end-to-end (Python fallback path).

        .. deprecated::
            In production, trade signals are handled entirely by the Rust
            ``trading_engine`` binary.  Use this method only for paper-trading,
            back-testing, or emergency manual overrides.

        Args:
            signal: Trade signal dict containing at least:
                ``symbol``, ``direction``, ``position_size``, ``stop_loss``,
                ``take_profit_levels``, ``leverage``, and ``strategy``.

        Returns:
            Execution result dict with keys ``success``, ``symbol``,
            ``order_id``, ``filled_price``, ``sl_order_id``,
            ``tp_order_ids``, and ``error`` (on failure).
        """
        # Note: Rust path (rust_engine/) is for future use; Python path remains active for paper/live trading
        symbol: str = signal.get("symbol", "")
        direction: str = signal.get("direction", "long")
        size: float = signal.get("position_size", 0.0)
        stop_loss: float = signal.get("stop_loss", 0.0)
        tp_levels: List[float] = signal.get("take_profit_levels", [])
        leverage: int = signal.get("leverage", Settings.get_settings().exchange.default_leverage)
        strategy: str = signal.get("strategy", "unknown")
        # Percentage-based SL/TP fallbacks (used when absolute prices are not provided)
        stop_loss_pct: float = signal.get("stop_loss_pct", 0.02)
        take_profit_pct: float = signal.get("take_profit_pct", 0.04)

        # Advanced execution: Start latency tracking
        execution_id = f"{symbol}_{int(time.time() * 1000)}"
        latency_tracking = None
        if self._enable_advanced_execution and self._latency_monitor:
            latency_tracking = self._latency_monitor.start_tracking(
                execution_id=execution_id,
                symbol=symbol,
                signal_timestamp=time.time()
            )

        logger.info("Executing trade: {} {} size={} USDT lev={}x", symbol, direction, size, leverage)

        # Advanced execution: Anti-gaming protection
        if self._enable_advanced_execution and self._anti_gaming:
            is_safe, alert = await self._anti_gaming.check_execution_safety(
                symbol=symbol,
                side=direction,
                stop_loss=stop_loss
            )
            if not is_safe:
                logger.warning("Anti-gaming protection triggered: {}", alert.description if alert else "unknown")
                # Add jitter delay for front-running protection
                jitter_delay = self._anti_gaming.get_execution_delay()
                await asyncio.sleep(jitter_delay)

        if size <= 0:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Position size is zero or negative",
            }

        try:
            # Import exchange enums at runtime to avoid circular import issues
            from exchange.base_exchange import OrderSide  # type: ignore[import]

            # 1. Fetch current price
            ticker = await self._exchange.get_ticker(symbol)
            current_price = ticker.last
            if current_price <= 0:
                return {"success": False, "symbol": symbol, "error": "Could not fetch current price"}

            # 2. Resolve market info (needed for both Rust and Python sizing paths)
            # Many exchanges (like Gate.io) require amounts in CONTRACTS, not base currency!
            markets = {}
            if hasattr(self._exchange, "_client") and getattr(self._exchange._client, "markets", None):
                markets = self._exchange._client.markets
            elif hasattr(self._exchange, "get_markets"):
                markets = await self._exchange.get_markets()

            # Force resolution to the Futures/Swap symbol (e.g. BTC/USDT -> BTC/USDT:USDT)
            original_symbol = symbol
            if hasattr(self._exchange, "_resolve_swap_symbol"):
                symbol = self._exchange._resolve_swap_symbol(symbol)
            elif ":" not in symbol and f"{symbol}:USDT" in markets:
                symbol = f"{symbol}:USDT"

            # Look up market info for the resolved swap symbol, falling back to original
            market_info = markets.get(symbol, {}) or markets.get(original_symbol, {})
            is_contract = bool(market_info.get("contract", False) or market_info.get("swap", False) or "swap" in market_info.get("type", ""))
            contract_size = float(market_info.get("contractSize") or 1.0)
            is_inverse = bool(market_info.get("inverse", False))
            limits = market_info.get("limits", {})
            min_amount = float((limits.get("amount") or {}).get("min") or 0.0)
            step_size = float(market_info.get("precision", {}).get("amount") or 0.0)

            # Determine exchange name (used for both Rust and Python fee paths)
            _fee_exchange_name = (
                getattr(self._exchange, "_exchange_id", "")
                or getattr(self._exchange, "name", "unknown")
            )

            # 2b. Rust pre-submission computation (Phase 4) — replaces fee + sizing + precision
            _skip_python_sizing = False
            amount_to_order = 0.0
            if _USE_RUST_EXEC and self._rust_fee_table is not None:
                try:
                    _rust_market = RustMarketInfo(
                        symbol=symbol,
                        is_contract=is_contract,
                        is_inverse=is_inverse,
                        contract_size=contract_size,
                        min_amount=min_amount,
                        step_size=step_size,
                        price_precision=float(market_info.get("precision", {}).get("price", 0)),
                    )
                    _plan = compute_execution_plan(
                        market_info=_rust_market,
                        fee_table=self._rust_fee_table,
                        exchange_name=_fee_exchange_name,
                        current_price=current_price,
                        position_size_usdt=size,
                        leverage=leverage,
                        direction=direction,
                        signal_confidence=float(signal.get("confidence", 0.5)),
                        book_imbalance=0.0,
                        spread_bps=5.0,
                        expected_profit_pct=float(signal.get("expected_profit_pct", 1.0)),
                        max_entry_slippage_pct=float(signal.get("max_entry_slippage_pct", 0.005)),
                    )
                    if not _plan.is_viable:
                        return {"success": False, "symbol": symbol, "error": _plan.rejection_reason}
                    amount_to_order = _plan.amount_to_order
                    size = _plan.fee_adjusted_size
                    _skip_python_sizing = True
                    logger.debug(
                        "Rust execution plan for {}: amount={} fee_adjusted_size={:.4f}",
                        symbol, amount_to_order, size,
                    )
                except Exception as _rust_exc:
                    logger.debug("Rust execution prep failed: {} — falling back to Python", _rust_exc)
                    _skip_python_sizing = False

            if not _skip_python_sizing:
                # 2c. Fee-aware position sizing: deduct estimated round-trip fees (Python fallback)
                try:
                    _fee_rates = {
                        "mexc": 0.0006,
                        "gateio": 0.0005,
                        "bingx": 0.0005,
                        "bitget": 0.0006,
                    }
                    _entry_fee_rate = _fee_rates.get(_fee_exchange_name.lower(), 0.0006)
                    _total_notional_for_fee = size * leverage
                    _estimated_round_trip_fee = _total_notional_for_fee * _entry_fee_rate * 2
                    _fee_buffer = _estimated_round_trip_fee * 1.1  # 10% safety margin on fees

                    if _fee_buffer >= size:
                        return {
                            "success": False,
                            "symbol": symbol,
                            "error": (
                                f"Position size ({size:.2f} USDT) is too small to cover "
                                f"estimated fees ({_fee_buffer:.2f} USDT)"
                            ),
                        }

                    size_after_fees = size - _fee_buffer
                    logger.debug(
                        "Fee-adjusted size for {}: original={:.4f} fees={:.4f} adjusted={:.4f}",
                        symbol, size, _fee_buffer, size_after_fees,
                    )
                    size = size_after_fees
                except Exception as fee_exc:
                    logger.debug("Fee deduction calculation failed: {} - using original size", fee_exc)

                # 3. Correctly Calculate Contracts vs Base Currency (Python fallback)
                if is_contract:
                    # Contract-based market (perpetual swaps/futures)
                    if is_inverse:
                        # Inverse contracts (like BitMEX): amount is in USD value
                        amount_to_order = size / contract_size
                        logger.debug(
                            "Contract market (inverse): {:.4f} USDT → {:.2f} contracts (contractSize={:.4f})",
                            size,
                            amount_to_order,
                            contract_size,
                        )
                    else:
                        # Linear contracts: convert USDT → base units → contracts.
                        # Gate.io and most perpetual exchanges require WHOLE INTEGER contracts.
                        # Multiply margin by leverage to get the true notional position size.
                        total_notional_usdt = size * leverage
                        amount_to_order = math.floor((total_notional_usdt / current_price) / contract_size)
                        # math.floor returns a float in Python; cast to int explicitly so that
                        # {:d} formatting and downstream integer-only logic works correctly.
                        amount_to_order = int(amount_to_order)
                        logger.debug(
                            "Contract market (linear): {:.4f} USDT * {}x leverage = {:.4f} notional → {:d} contracts (contractSize={:.4f})",
                            size,
                            leverage,
                            total_notional_usdt,
                            amount_to_order,
                            contract_size,
                        )
                    # If the allocated size is too small for 1 contract, try rounding up to 1.
                    # The pre-trade balance check (below) will reject the trade if margin
                    # is actually insufficient.
                    if amount_to_order < 1:
                        min_contract_cost = (1 * contract_size * current_price) / leverage
                        logger.info(
                            "Rounded up to minimum 1 contract for {} (cost≈{:.2f} USDT, "
                            "original calc < 1). Balance check will validate margin.",
                            symbol, min_contract_cost,
                        )
                        amount_to_order = 1
                else:
                    # Spot or base-currency market: simple USDT → base conversion
                    amount_to_order = size / current_price
                    logger.debug(
                        "Base currency market: {:.4f} USDT / {:.4f} price = {:.8f} base units",
                        size,
                        current_price,
                        amount_to_order,
                    )

                # 4. Validate Exchange Minimums and Step Precision (Python fallback)
                # Round down to the nearest valid step size (e.g., whole integer contracts).
                # For contract markets this is a no-op when step_size == 1 (already floored above).
                if step_size > 0:
                    amount_to_order = math.floor(amount_to_order / step_size) * step_size

                # Block the trade if rounding reduced the size to 0 or below the minimum
                if amount_to_order <= 0 or (min_amount > 0 and amount_to_order < min_amount):
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": (
                            f"Order size is too small. Calculated size: {amount_to_order} "
                            f"(Min required: {min_amount}, Step size: {step_size})."
                        ),
                    }

            # Pre-trade balance check: verify sufficient free margin before placing order
            try:
                live_balance = await self._exchange.get_balance()
                # For contract markets, amount_to_order is in CONTRACTS; multiply by
                # contract_size to convert to base-currency units before pricing in USDT.
                if is_contract:
                    position_size_usdt = amount_to_order * contract_size * current_price
                else:
                    position_size_usdt = amount_to_order * current_price
                required_margin = position_size_usdt / max(leverage, 1)
                buffer_margin = required_margin * 1.1  # 10% safety buffer
                if live_balance.usdt_free < buffer_margin:
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": (
                            f"Insufficient balance: required={buffer_margin:.2f} USDT "
                            f"(incl. 10% buffer), available={live_balance.usdt_free:.2f} USDT"
                        ),
                    }
                if live_balance.usdt_free < required_margin * 2:
                    logger.warning(
                        "Low margin warning for {}: balance={:.2f} required={:.2f} USDT",
                        symbol, live_balance.usdt_free, required_margin,
                    )
                logger.info(
                    "Pre-trade balance OK for {}: free={:.2f} required={:.2f} USDT",
                    symbol, live_balance.usdt_free, required_margin,
                )
            except Exception as bal_exc:
                logger.warning("Pre-trade balance check failed for {}: {}", symbol, bal_exc)

            # Set margin type before setting leverage.
            # This prevents a losing trade from draining the entire account
            # balance (which happens under Cross Margin mode).
            _settings = Settings.get_settings()
            _risk_cfg = getattr(_settings, "risk", None)
            _margin_mode_str = _risk_cfg.margin_mode if _risk_cfg is not None else "isolated"
            _margin_type = MarginType.ISOLATED if _margin_mode_str == "isolated" else MarginType.CROSS
            try:
                await asyncio.wait_for(
                    self._exchange.set_margin_type(symbol, _margin_type),
                    timeout=3.0,
                )
                logger.info("Margin type set to {} for {}", _margin_type.value.upper(), symbol)
            except asyncio.TimeoutError:
                logger.debug("Margin type setting timed out for {} — continuing", symbol)
            except Exception as margin_exc:
                exc_str = str(margin_exc).lower()
                if any(kw in exc_str for kw in ("no need to change", "already", "not support", "skipped")):
                    logger.debug(
                        "Margin type already {} for {} (or not supported): {}",
                        _margin_type.value.upper(), symbol, margin_exc,
                    )
                else:
                    logger.warning(
                        "Could not set {} margin for {}: {}. Proceeding with current margin mode.",
                        _margin_type.value.upper(), symbol, margin_exc,
                    )

            # Respect the risk manager's calculated leverage. Only enforce the exchange minimum (1x).
            leverage = max(1, leverage)
            logger.debug("Using leverage {}x for {} (from risk manager)", leverage, symbol)
            try:
                await self._exchange.set_leverage(symbol, leverage)
            except Exception as lev_exc:
                logger.warning(
                    "set_leverage failed for {} ({}x): {}. Checking current leverage.",
                    symbol,
                    leverage,
                    lev_exc,
                )
                # If the exchange already has adequate leverage, we can proceed.
                try:
                    positions = await self._exchange.get_positions(symbol)
                    current_leverage = (
                        positions[0].leverage if positions else 0
                    )
                    if current_leverage >= leverage:
                        logger.info(
                            "Current leverage {}x >= required {}x for {} — continuing.",
                            current_leverage,
                            leverage,
                            symbol,
                        )
                    elif not positions:
                        # No open position yet (new entry): cannot verify leverage.
                        # Proceed and let the exchange use its default leverage.
                        logger.warning(
                            "No open position for {} to verify leverage; proceeding with trade.",
                            symbol,
                        )
                    else:
                        raise RuntimeError(
                            f"Unable to set leverage to {leverage}x for {symbol}: {lev_exc}"
                        ) from lev_exc
                except RuntimeError:
                    raise
                except Exception as pos_exc:
                    logger.warning(
                        "Could not verify current leverage for {}: {}. Proceeding anyway.",
                        symbol,
                        pos_exc,
                    )

            # 4. Cancel stale entry limit orders for this symbol before taking a new position.
            # Only cancel limit/entry order types — do NOT cancel SL/TP orders here because
            # an existing position may still rely on them for protection (Bug #4).
            try:
                existing_orders = await self._exchange.get_open_orders(symbol)
                for existing_order in existing_orders:
                    order_type_str = str(getattr(existing_order, "type", "")).lower()
                    # Skip stop-loss and take-profit orders — they belong to an existing position
                    if any(kw in order_type_str for kw in ("stop", "take_profit", "trigger")):
                        continue
                    info = getattr(existing_order, "info", {}) or {}
                    if info.get("reduceOnly"):
                        continue
                    # Cancel orphan entry limit orders only
                    try:
                        await self._order_manager.cancel_order(existing_order.id, symbol)
                        logger.debug(f"Cancelled stale entry order {existing_order.id} for {original_symbol}")
                    except Exception as cancel_exc:
                        logger.debug(f"Could not cancel order {existing_order.id}: {cancel_exc}")
            except Exception as e:
                logger.warning(f"Could not check/cancel orders for {original_symbol}: {e}")

            side = OrderSide.BUY if direction == "long" else OrderSide.SELL
            chunks = self._split_order(amount_to_order, max_chunk=amount_to_order, min_order_size=min_amount)  # single chunk by default

            # Use SmartEntryOptimizer to determine limit price and TWAP schedule
            try:
                orderbook = self._get_local_or_rest_orderbook_sync(symbol)
                if orderbook is None:
                    orderbook = await self._exchange.get_orderbook(symbol, limit=10)
            except Exception:
                orderbook = None

            entry_recommendation = self._smart_entry.get_entry_recommendation(
                direction=direction,
                current_price=current_price,
                total_size=amount_to_order,
                orderbook=orderbook,
                is_large_order=(amount_to_order > 0.1),
            )

            # Use TWAP schedule for large orders
            if entry_recommendation.get("use_twap") and entry_recommendation.get("twap_schedule"):
                twap_chunks = [sz for _, sz in entry_recommendation["twap_schedule"]]
                twap_delays = [delay for delay, _ in entry_recommendation["twap_schedule"]]
            else:
                twap_chunks = chunks
                twap_delays = [0.0] * len(chunks)

            # Task 9: Check order book imbalance and delay if adverse
            if orderbook is not None:
                entry_delay = self._smart_entry.calculate_entry_delay(
                    direction=direction,
                    orderbook=orderbook,
                    max_delay_seconds=5.0,
                )
                if entry_delay > 0:
                    logger.info(
                        "Delaying entry for {} by {:.1f}s due to adverse order book",
                        symbol, entry_delay,
                    )
                    await asyncio.sleep(entry_delay)
                    # Refresh orderbook and price after delay
                    try:
                        ticker = await self._exchange.get_ticker(symbol)
                        current_price = ticker.last
                        orderbook = await self._exchange.get_orderbook(symbol, limit=10)
                    except Exception:
                        pass

            # Gate.io book analysis + fee optimization (Phase 3: System 3 enhanced)
            _book_state: Dict[str, Any] = {}
            _fee_optimizer_order_type: str = "market"
            _execution_route: Dict[str, Any] = {}
            try:
                from exchange.gateio_book_analyzer import GateioBookAnalyzer
                _book_analyzer = GateioBookAnalyzer(self._exchange)
                _book_state = await _book_analyzer.analyze_book(original_symbol, depth=20)
                _signal_confidence: float = float(signal.get("confidence", 0.5))

                # Phase 3: System 3 - Calculate optimal execution route
                _execution_route = self._fee_optimizer.calculate_optimal_execution_route(
                    signal_confidence=_signal_confidence,
                    spread_bps=_book_state.get("spread_bps", 5.0),
                    book_state=_book_state,
                    position_notional=size,
                    maker_fee_bps=self._fee_optimizer.maker_fee * 10000,  # Convert to bps
                    taker_fee_bps=self._fee_optimizer.taker_fee * 10000,
                )

                _fee_optimizer_order_type = _execution_route.get("order_type", "market")
                logger.info(
                    "Execution route for {}: {} - {}",
                    original_symbol,
                    _fee_optimizer_order_type,
                    _execution_route.get("reasoning", ""),
                )

                # Override TWAP settings if execution route recommends it
                if _execution_route.get("order_type") == "twap" and _execution_route.get("use_iceberg"):
                    recommended_chunks = _execution_route.get("chunk_count", 2)
                    chunk_size = amount_to_order / recommended_chunks
                    twap_chunks = [chunk_size] * recommended_chunks
                    twap_delays = [15.0] * recommended_chunks
                    logger.info(
                        "Using smart routing TWAP for {}: {} chunks of {:.4f} each",
                        original_symbol,
                        recommended_chunks,
                        chunk_size,
                    )

                # Fee viability check — reject trade when break-even exceeds expected profit
                _expected_profit_pct: float = float(signal.get("expected_profit_pct", 1.0))
                _exit_price_est = current_price * (
                    1 + _expected_profit_pct / 100 if direction == "long"
                    else 1 - _expected_profit_pct / 100
                )
                _cost_breakdown = self._fee_optimizer.calculate_trade_cost(
                    entry_price=current_price,
                    exit_price=_exit_price_est,
                    amount=amount_to_order,
                    leverage=leverage,
                    direction=direction,
                )
                if not self._fee_optimizer.trade_is_viable(
                    _cost_breakdown,
                    expected_profit_pct=_expected_profit_pct,
                    min_profit_to_cost_ratio=2.0,
                ):
                    logger.warning(
                        "Trade {} rejected: break-even={:.4f}% exceeds expected profit {:.4f}%",
                        original_symbol,
                        _cost_breakdown.get("break_even_pct", 0.0),
                        _expected_profit_pct,
                    )
                    return {
                        "success": False,
                        "symbol": original_symbol,
                        "error": (
                            f"Fee check failed: break-even {_cost_breakdown.get('break_even_pct', 0):.4f}% "
                            f"≥ 50% of expected profit {_expected_profit_pct:.4f}%"
                        ),
                    }

                logger.debug(
                    "Book analysis for {}: imbalance={:.3f} spread={:.2f}bps "
                    "recommended_order_type={}",
                    original_symbol,
                    _book_state.get("imbalance", 0.0),
                    _book_state.get("spread_bps", 0.0),
                    _fee_optimizer_order_type,
                )

                # For large orders (> 5% of visible book depth), prefer iceberg via TWAP
                _bid_depth = _book_state.get("bid_depth_usdt", 0.0)
                _ask_depth = _book_state.get("ask_depth_usdt", 0.0)
                _visible_depth = _bid_depth if direction == "long" else _ask_depth
                _notional = amount_to_order * current_price
                if _visible_depth > 0 and _notional > _visible_depth * 0.05:
                    logger.info(
                        "Large order detected for {} ({:.0f} USDT > 5% of {:.0f} USDT depth) "
                        "— routing via TWAP/iceberg",
                        original_symbol, _notional, _visible_depth,
                    )
                    if not entry_recommendation.get("use_twap"):
                        entry_recommendation["use_twap"] = True
                        n_chunks = min(10, max(2, int(_notional / (_visible_depth * 0.02))))
                        chunk_size = amount_to_order / n_chunks
                        twap_chunks = [chunk_size] * n_chunks
                        twap_delays = [15.0] * n_chunks
            except Exception as _ba_exc:
                logger.debug("Book analysis skipped for {}: {}", original_symbol, _ba_exc)

            # Task 5: Determine slippage cap from settings
            _settings_for_slip = Settings.get_settings()
            _risk_cfg_for_slip = getattr(_settings_for_slip, "risk", None)
            _entry_slippage_pct = getattr(_risk_cfg_for_slip, "max_entry_slippage_pct", 0.01)

            # Task 1A: Build atomic SL/TP params for the entry order
            entry_params: Dict[str, Any] = {}
            if stop_loss > 0:
                entry_params["stopLoss"] = {
                    "triggerPrice": stop_loss,
                    "type": "market",
                }
            if tp_levels:
                # Attach only the first TP level atomically; remaining TPs placed separately
                entry_params["takeProfit"] = {
                    "triggerPrice": tp_levels[0],
                    "type": "market",
                }

            # Task 4: Check maker entry preference
            _settings_for_maker = Settings.get_settings()
            _risk_cfg_for_maker = getattr(_settings_for_maker, "risk", None)
            _use_maker = getattr(_risk_cfg_for_maker, "use_maker_entries", False)
            _maker_max_wait = getattr(_risk_cfg_for_maker, "maker_entry_max_wait_seconds", 15.0)

            entry_order = None

            if _use_maker and not entry_recommendation.get("use_twap"):
                # Task 4: Use post-only maker entry
                entry_order = await self._execute_maker_entry(
                    symbol=symbol,
                    side=side,
                    amount=amount_to_order,
                    current_price=current_price,
                    strategy=strategy,
                    max_total_wait_seconds=_maker_max_wait,
                )
            else:
                # Standard TWAP/market order logic with atomic SL/TP (Task 1A) and slippage (Task 5)
                for i, (chunk, delay) in enumerate(zip(twap_chunks, twap_delays)):
                    if i > 0 and delay > 0:
                        await asyncio.sleep(min(delay, self._MAX_TWAP_DELAY_SECONDS))
                    # Pass entry_params (atomic SL/TP) and slippage_pct to place_market_order
                    order_kwargs: Dict[str, Any] = {
                        "slippage_pct": _entry_slippage_pct,
                        "leverage": leverage,  # Pass leverage so it's set atomically with the order
                    }
                    order_kwargs.update(entry_params)
                    entry_order = await self._order_manager.place_market_order(
                        symbol=symbol,
                        side=side,
                        amount=chunk,
                        strategy=strategy,
                        **order_kwargs,
                    )
                    if len(twap_chunks) > 1:
                        await asyncio.sleep(0.5)  # brief pause between chunks

            if entry_order is None:
                return {"success": False, "symbol": symbol, "error": "No entry order placed"}

            filled_price = entry_order.price or 0.0
            sl_order_id: Optional[str] = None
            tp_order_ids: List[str] = []

            # Wait for the entry order to be fully filled with proper confirmation
            confirmed_order = await self._wait_for_fill(entry_order.id, symbol)
            if confirmed_order is not None and confirmed_order.price:
                filled_price = confirmed_order.price
                logger.info(
                    "Fill confirmed for {}: price={:.4f} filled={}",
                    symbol, filled_price, confirmed_order.filled,
                )
                # Log slippage vs expected price
                if current_price > 0:
                    slippage_pct = abs(filled_price - current_price) / current_price * 100
                    logger.info(
                        "Slippage for {}: expected={:.4f} actual={:.4f} slippage={:.3f}%",
                        symbol, current_price, filled_price, slippage_pct,
                    )
                # Log Gate.io execution quality metrics
                _mid_price = _book_state.get("mid_price", 0.0)
                if _mid_price > 0 and filled_price > 0:
                    _slippage_vs_mid_bps = (
                        abs(filled_price - _mid_price) / _mid_price * 10_000
                    )
                    _is_maker = _fee_optimizer_order_type in ("post_only", "limit_passive")
                    logger.info(
                        "Execution quality for {}: slippage_vs_mid={:.2f}bps "
                        "order_type={} maker={}",
                        original_symbol,
                        _slippage_vs_mid_bps,
                        _fee_optimizer_order_type,
                        _is_maker,
                    )

            # Pre-register the symbol as "pending SL" so the watchdog does not
            # place an emergency SL during the window between entry fill and
            # actual SL order placement (which can take 1-3 seconds).
            try:
                self._position_manager.mark_sl_pending(symbol)
            except Exception:
                pass

            # Fetch actual position to get confirmed contract amount for SL/TP sizing
            actual_pos = await self._position_manager.get_position(symbol)
            if not actual_pos or actual_pos.position.amount <= 0:
                logger.warning(f"Position did not open correctly for {symbol}, using order amount")
                actual_amount = amount_to_order
            else:
                actual_amount = actual_pos.position.amount
                logger.debug(f"Actual filled position for {symbol}: {actual_amount}")

            # Import PositionSide for determining position side
            from exchange.base_exchange import PositionSide  # type: ignore[import]
            pos_side = actual_pos.position.side if actual_pos else (
                PositionSide.LONG if direction == "long" else PositionSide.SHORT
            )

            is_long = pos_side == PositionSide.LONG
            price_for_sl_tp = filled_price if filled_price > 0 else current_price

            # Percentage-based SL/TP fallback: if the signal did not supply absolute
            # prices, derive them from the fill price so native exchange orders are
            # always placed regardless of how the signal was constructed.
            if stop_loss <= 0 and price_for_sl_tp > 0:
                stop_loss = (
                    price_for_sl_tp * (1 - stop_loss_pct)
                    if is_long
                    else price_for_sl_tp * (1 + stop_loss_pct)
                )
                logger.info(
                    "Derived SL from {}% pct: {} → {:.4f}",
                    stop_loss_pct * 100,
                    symbol,
                    stop_loss,
                )
            if not tp_levels and price_for_sl_tp > 0:
                tp_price_derived = (
                    price_for_sl_tp * (1 + take_profit_pct)
                    if is_long
                    else price_for_sl_tp * (1 - take_profit_pct)
                )
                tp_levels = [tp_price_derived]
                logger.info(
                    "Derived TP from {}% pct: {} → {:.4f}",
                    take_profit_pct * 100,
                    symbol,
                    tp_price_derived,
                )

            # Place stop-loss (Task 1A: skip if atomic SL was accepted by exchange)
            atomic_sl_placed = False
            atomic_tp_placed = False
            if entry_params.get("stopLoss") or entry_params.get("takeProfit"):
                try:
                    open_orders = await self._exchange.get_open_orders(symbol)
                    for o in open_orders:
                        order_type_str = str(getattr(o, "type", "")).lower()
                        if "stop" in order_type_str:
                            atomic_sl_placed = True
                            sl_order_id = o.id
                        if "take_profit" in order_type_str or "tp" in order_type_str:
                            atomic_tp_placed = True
                            tp_order_ids.append(o.id)
                    if atomic_sl_placed and sl_order_id:
                        # Mark position as protected immediately so watchdog does not
                        # place a duplicate emergency SL before we return.
                        try:
                            self._position_manager.mark_position_protected(symbol, sl_order_id)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.debug("Could not verify atomic SL/TP: {}", exc)

            if stop_loss > 0 and not atomic_sl_placed:
                sl_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
                try:
                    sl_order = await self._order_manager.place_stop_loss(
                        symbol=symbol,
                        side=sl_side,
                        amount=actual_amount,
                        stop_price=stop_loss,
                        strategy=strategy,
                    )
                    sl_order_id = sl_order.id
                    # Immediately mark the position as protected in the watchdog cache
                    # so it does not place a duplicate emergency SL while the order
                    # is being registered on the exchange.
                    try:
                        self._position_manager.mark_position_protected(symbol, sl_order_id)
                    except Exception:
                        pass
                    # Persist SL immediately to prevent watchdog race conditions
                    await self._persist_active_order(
                        exchange_id=sl_order_id,
                        symbol=symbol,
                        strategy=strategy,
                        order_type="stop_loss",
                        side="sell" if direction == "long" else "buy",
                        amount=actual_amount,
                        stop_loss=stop_loss if stop_loss > 0 else None,
                    )
                except Exception as exc:
                    logger.error("Failed to place stop-loss for {}: {}", symbol, exc)
                    # Clear the pending-SL guard so the watchdog can place an
                    # emergency SL if the executor failed to do so.
                    try:
                        self._position_manager.clear_sl_pending(symbol)
                    except Exception:
                        pass

            # Place take-profit orders safely (ensuring min_amount rules)
            # Skip the first TP if it was atomically placed (Task 1A)
            tp_start_idx = 1 if atomic_tp_placed and tp_levels else 0
            tp_proportions = [0.25, 0.50, 0.25]
            remaining_amount = actual_amount
            for i, (tp_price, proportion) in enumerate(zip(tp_levels, tp_proportions)):
                if i < tp_start_idx:
                    # First TP was placed atomically — deduct its allocated portion
                    # and skip placing a separate order for it
                    allocated = actual_amount * proportion
                    remaining_amount = actual_amount - allocated
                    continue
                tp_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY

                # Bug #3 validation: ensure TP price is on the correct side of entry.
                # For LONG trades, TP must be ABOVE entry (profit taken when price rises).
                # For SHORT trades, TP must be BELOW entry (profit taken when price drops).
                if price_for_sl_tp > 0:
                    if is_long and tp_price <= price_for_sl_tp:
                        logger.warning(
                            "Skipping invalid LONG TP{} for {}: tp={:.4f} <= entry={:.4f}",
                            i + 1, original_symbol, tp_price, price_for_sl_tp,
                        )
                        continue
                    if not is_long and tp_price >= price_for_sl_tp:
                        logger.warning(
                            "Skipping invalid SHORT TP{} for {}: tp={:.4f} >= entry={:.4f}",
                            i + 1, original_symbol, tp_price, price_for_sl_tp,
                        )
                        continue

                # If it's the last TP or position is too small, dump the rest
                if i == len(tp_levels) - 1 or (remaining_amount * proportion) < min_amount:
                    tp_amount = remaining_amount
                else:
                    tp_amount = remaining_amount * proportion

                if tp_amount < min_amount:
                    logger.debug(f"Skipping TP{i+1} for {original_symbol}: amount {tp_amount:.6f} < min {min_amount:.6f}")
                    continue  # Skip if chunk is too small

                try:
                    tp_order = await self._order_manager.place_take_profit(
                        symbol=symbol,
                        side=tp_side,
                        amount=tp_amount,
                        tp_price=tp_price,
                        strategy=strategy,
                    )
                    tp_order_ids.append(tp_order.id)
                    remaining_amount -= tp_amount
                    if remaining_amount <= 0:
                        break
                except ValueError as ve:
                    # TP price is no longer valid (price moved past it) — skip gracefully
                    logger.warning("Skipping TP{} for {} — price moved past target: {}", i + 1, original_symbol, ve)
                except Exception as exc:
                    logger.error("Failed to place TP{} for {}: {}", i + 1, original_symbol, exc)

            result = {
                "success": True,
                "symbol": original_symbol,
                "direction": direction,
                "order_id": entry_order.id,
                "filled_price": filled_price,
                "size": actual_amount,
                "size_usdt": size,
                "leverage": leverage,
                "sl_order_id": sl_order_id,
                "tp_order_ids": tp_order_ids,
            }

            # Register SL/TP metadata with the position tracker so that the
            # position manager's risk overlays can enforce them in paper mode.
            try:
                tracker = await self._position_manager.get_position(symbol)
                if tracker is not None:
                    if stop_loss > 0:
                        tracker.stop_loss = stop_loss
                    if tp_levels:
                        tracker.take_profit = sorted(tp_levels)
                    # Set trailing TP distance based on entry→last-TP range
                    if tp_levels:
                        settings = Settings.get_settings()
                        if getattr(getattr(settings, "risk", None), "enable_trailing_tp", True):
                            trailing_pct = getattr(
                                settings.risk, "trailing_tp_distance_pct", 0.5
                            )
                            last_tp = tp_levels[-1]
                            # trailing distance = trailing_pct % of the last TP price
                            tracker.trailing_tp_distance = last_tp * (trailing_pct / 100.0)
            except Exception as exc:
                logger.debug("Could not register SL/TP with position tracker: {}", exc)

            # Persist the active position and orders to the database so that the
            # reconciler can restore state on restart.
            await self._persist_active_position(
                symbol=symbol,
                exchange_id=entry_order.id,
                strategy=strategy,
                direction=direction,
                amount=actual_amount,
                entry_price=filled_price,
                leverage=leverage,
                stop_loss=stop_loss if stop_loss > 0 else None,
                tp_levels=tp_levels,
            )
            # SL is already persisted immediately after placement to prevent
            # watchdog race conditions. Only persist Take Profits here.
            for tp_oid, tp_price in zip(tp_order_ids, tp_levels):
                await self._persist_active_order(
                    exchange_id=tp_oid,
                    symbol=symbol,
                    strategy=strategy,
                    order_type="take_profit",
                    side="sell" if direction == "long" else "buy",
                    amount=actual_amount,
                    take_profit=tp_price,
                )

            logger.info(
                "Trade executed: {} — order_id={} sl={} tps={}",
                symbol,
                entry_order.id,
                sl_order_id,
                tp_order_ids,
            )
            return result

        except Exception as exc:
            logger.error("Trade execution failed for {}: {}", symbol, exc)
            return {"success": False, "symbol": symbol, "error": str(exc)}

    async def modify_position(
        self,
        position_id: str,
        params: Dict[str, Any],
    ) -> bool:
        """Modify an existing position's SL or TP levels.

        Args:
            position_id: Symbol used as position identifier.
            params: Dict with optional ``stop_loss`` and ``take_profit`` keys.

        Returns:
            ``True`` if any modification was applied.
        """
        modified = False
        try:
            if "stop_loss" in params:
                updated = await self._position_manager.update_stop_loss(
                    position_id, params["stop_loss"]
                )
                modified = modified or updated
            if "take_profit" in params:
                updated = await self._position_manager.update_take_profit(
                    position_id, params["take_profit"]
                )
                modified = modified or updated
            logger.info("Position {} modified: {} applied={}", position_id, params, modified)

            # Persist the updated SL/TP to the database
            if modified:
                await self._update_active_position_sltp(
                    symbol=position_id,
                    stop_loss=params.get("stop_loss"),
                    tp_levels=params.get("take_profit"),
                )
        except Exception as exc:
            logger.error("Failed to modify position {}: {}", position_id, exc)
        return modified

    async def close_position(self, position_id: str, reason: str = "") -> dict:
        """Close an open position.

        Args:
            position_id: Symbol used as position identifier.
            reason: Human-readable closure reason.

        Returns:
            Close result dict.
        """
        try:
            result = await self._position_manager.close_position(position_id, reason)
            logger.info("Position {} closed: reason='{}' result={}", position_id, reason, result)
            # Remove the persisted active position record now that the trade is closed
            if result.get("closed"):
                await self._delete_active_position(position_id)
            return result
        except Exception as exc:
            logger.error("Failed to close position {}: {}", position_id, exc)
            return {"symbol": position_id, "closed": False, "error": str(exc)}

    async def close_all_positions(self, reason: str = "") -> List[dict]:
        """Close all open positions.

        Args:
            reason: Human-readable closure reason.

        Returns:
            List of close result dicts.
        """
        logger.warning("Closing all positions: reason='{}'", reason)
        try:
            return await self._position_manager.close_all_positions(reason)
        except Exception as exc:
            logger.error("Failed to close all positions: {}", exc)
            return []

    async def execute_partial_close(
        self, symbol: str, percentage: float, reason: str = ""
    ) -> dict:
        """Close a percentage of an open position.

        After closing, records the partial close in the position manager.  If
        this was triggered by a take-profit hit (``reason`` contains
        ``"take_profit"``), break-even stop-loss activation is also attempted.

        Args:
            symbol: Trading pair symbol.
            percentage: Fraction of the position to close (0 < percentage ≤ 1).
            reason: Human-readable reason for the partial close.

        Returns:
            Result dict with keys ``symbol``, ``closed_amount``, ``pnl``,
            ``order_id``, and ``success``.
        """
        if percentage <= 0 or percentage > 1:
            return {
                "symbol": symbol,
                "success": False,
                "error": f"percentage {percentage} out of range (0, 1]",
            }
        try:
            tracker = await self._position_manager.get_position(symbol)
            if tracker is None:
                return {"symbol": symbol, "success": False, "error": "position not found"}

            pos_amount = tracker.position.amount
            close_amount = pos_amount * percentage

            # Get current price for P&L estimation
            try:
                ticker = await self._exchange.get_ticker(symbol)
                current_price = ticker.last
            except Exception:
                current_price = tracker.position.current_price or tracker.position.entry_price

            result = await self._position_manager.reduce_position(symbol, close_amount)
            if not result.get("reduced"):
                return {
                    "symbol": symbol,
                    "success": False,
                    "error": result.get("reason", "reduce_position failed"),
                }

            # Estimate realised P&L from this partial close
            entry = tracker.position.entry_price
            side = tracker.position.side
            from exchange.base_exchange import PositionSide  # type: ignore[import]

            if side == PositionSide.LONG:
                pnl = (current_price - entry) * close_amount
            else:
                pnl = (entry - current_price) * close_amount

            await self._position_manager.record_partial_close(
                symbol, close_amount, current_price, pnl
            )

            # Activate break-even SL after a TP-triggered partial close
            if "take_profit" in reason.lower():
                await self._position_manager.activate_break_even(symbol)

            logger.info(
                "Partial close executed: {} percentage={:.0%} amount={} pnl={:.4f} reason='{}'",
                symbol,
                percentage,
                close_amount,
                pnl,
                reason,
            )
            return {
                "symbol": symbol,
                "success": True,
                "closed_amount": close_amount,
                "pnl": pnl,
                "order_id": result.get("order_id"),
            }
        except Exception as exc:
            logger.error("execute_partial_close failed for {}: {}", symbol, exc)
            return {"symbol": symbol, "success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def cancel_stale_entry_orders(
        self,
        max_age_minutes: int = 30,
        symbol: Optional[str] = None,
    ) -> int:
        """Cancel open limit entry orders that have been unfilled for too long.

        Orphaned entry orders are dangerous: a limit order placed hours ago
        can trigger unexpectedly during a sudden price move.  This method
        fetches all open orders (optionally filtered by *symbol*) and cancels
        any that are:

        * Limit (not stop-loss / take-profit trigger) orders, AND
        * Older than *max_age_minutes* minutes.

        Args:
            max_age_minutes: Age threshold in minutes (default 30).
            symbol: Optional symbol filter; ``None`` checks all symbols.

        Returns:
            Number of orders cancelled.
        """
        cancelled = 0
        try:
            now_ms = time.time() * 1000
            cutoff_ms = max_age_minutes * 60 * 1000
            open_orders = await self._exchange.get_open_orders(symbol)
            for order in open_orders:
                # Only target plain limit entry orders, not SL/TP triggers
                is_limit = (
                    hasattr(order, "type") and
                    str(getattr(order, "type", "")).lower() == "limit"
                )
                order_ts = getattr(order, "timestamp", 0) or 0
                age_ms = now_ms - order_ts
                if is_limit and order_ts > 0 and age_ms > cutoff_ms:
                    try:
                        await self._exchange.cancel_order(order.id, order.symbol)
                        logger.warning(
                            "Cancelled stale entry limit order: id={} symbol={} age={:.1f}min",
                            order.id,
                            order.symbol,
                            age_ms / 60000,
                        )
                        cancelled += 1
                    except Exception as cancel_exc:
                        logger.debug(
                            "Could not cancel stale order {} for {}: {}",
                            order.id,
                            order.symbol,
                            cancel_exc,
                        )
        except Exception as exc:
            logger.warning("cancel_stale_entry_orders failed: {}", exc)
        return cancelled

    async def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: float = 15.0,
    ) -> Optional[Any]:
        """Poll for order fill confirmation with timeout.

        Attempts WebSocket-driven fill confirmation first (if the exchange
        supports it and WS data is fresh), then falls back to REST polling.

        Args:
            order_id: Exchange order ID to monitor.
            symbol: Trading symbol.
            timeout: Maximum seconds to wait (default 15).

        Returns:
            The filled Order if status == CLOSED (fully filled), a partially-
            filled Order if partially filled after timeout, or None if the order
            could not be confirmed (cancelled or error).
        """
        from exchange.base_exchange import OrderStatus

        # --- WebSocket-first path ---
        if (
            hasattr(self._exchange, "register_fill_waiter")
            and hasattr(self._exchange, "is_ws_data_fresh")
            and self._exchange.is_ws_data_fresh()
        ):
            try:
                future = self._exchange.register_fill_waiter(order_id)
                _ws_start = time.time()
                ws_result = await asyncio.wait_for(future, timeout=timeout)
                if ws_result:
                    try:
                        order = await self._exchange.get_order(order_id, symbol)
                        if order.status == OrderStatus.CLOSED:
                            logger.info(
                                "Fill confirmed via WebSocket for {} in {:.2f}s",
                                order_id,
                                time.time() - _ws_start,
                            )
                            return order
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                logger.debug(
                    "WS fill timeout for {} — falling back to REST polling", order_id
                )
            except Exception as exc:
                logger.debug(
                    "WS fill error for {}: {} — falling back to REST", order_id, exc
                )

        # --- Fallback: REST polling ---
        deadline = time.time() + timeout
        poll_interval = 1.0
        last_order = None

        while time.time() < deadline:
            try:
                order = await self._exchange.get_order(order_id, symbol)
                last_order = order
                if order.status == OrderStatus.CLOSED:
                    return order
                if order.status in (
                    getattr(OrderStatus, "CANCELED", None),
                    getattr(OrderStatus, "CANCELLED", None),
                    getattr(OrderStatus, "EXPIRED", None),
                ):
                    logger.warning(
                        "_wait_for_fill: order {} for {} was {}", order_id, symbol, order.status
                    )
                    return None
            except Exception as exc:
                logger.debug("_wait_for_fill poll error for {}: {}", order_id, exc)
            await asyncio.sleep(poll_interval)

        # Timeout reached
        if last_order is not None:
            partially_filled = last_order.filled > 0
            if partially_filled:
                logger.warning(
                    "_wait_for_fill timeout for {} on {}: partial fill {:.4f}/{:.4f}",
                    order_id, symbol, last_order.filled, last_order.amount,
                )
                # Cancel the unfilled remainder
                try:
                    await self._exchange.cancel_order(order_id, symbol)
                    logger.info("Cancelled unfilled remainder of partial order {}", order_id)
                except Exception as exc:
                    logger.debug("Could not cancel partial order {}: {}", order_id, exc)
                return last_order
            else:
                logger.warning(
                    "_wait_for_fill timeout for {} on {}: order still OPEN — cancelling",
                    order_id, symbol,
                )
                try:
                    await self._exchange.cancel_order(order_id, symbol)
                except Exception as exc:
                    logger.debug("Could not cancel open order {}: {}", order_id, exc)
                return None

        return None

    async def _execute_maker_entry(
        self,
        symbol: str,
        side: "Any",
        amount: float,
        current_price: float,
        strategy: str,
        max_chase_attempts: int = 5,
        chase_interval_seconds: float = 2.0,
        max_total_wait_seconds: float = 15.0,
        fallback_to_market: bool = True,
    ) -> "Optional[Any]":
        """Execute entry using Post-Only limit orders to capture maker rebates.

        Places a limit order at the best bid (for buys) or best ask (for sells).
        If not filled within chase_interval_seconds, cancels and replaces at the
        new best bid/ask. After max_chase_attempts or max_total_wait_seconds,
        falls back to a market order if fallback_to_market is True.

        Returns the filled Order, or None if all attempts failed.
        """
        import time as _time

        from exchange.base_exchange import OrderSide as _OS  # type: ignore[import]

        start_time = _time.time()

        for attempt in range(max_chase_attempts):
            elapsed = _time.time() - start_time
            if elapsed >= max_total_wait_seconds:
                break

            try:
                orderbook = self._get_local_or_rest_orderbook_sync(symbol)
                if orderbook is None:
                    orderbook = await self._exchange.get_orderbook(symbol, limit=5)

                bids = orderbook.get("bids", []) if orderbook else []
                asks = orderbook.get("asks", []) if orderbook else []
                if not bids or not asks:
                    break

                if side == _OS.BUY:
                    limit_price = float(bids[0][0])
                else:
                    limit_price = float(asks[0][0])

                # Apply exchange price precision
                if hasattr(self._exchange, "_client") and self._exchange._client:
                    try:
                        limit_price = float(
                            self._exchange._client.price_to_precision(symbol, limit_price)
                        )
                    except Exception:
                        pass

                # Place post-only limit order
                order = await self._order_manager.place_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=limit_price,
                    strategy=strategy,
                    postOnly=True,
                )

                logger.debug(
                    "Maker entry attempt {}/{}: {} {} @ {:.4f}",
                    attempt + 1, max_chase_attempts, getattr(side, "value", side), symbol, limit_price,
                )

                # Wait for fill
                filled_order = await self._wait_for_fill(
                    order.id, symbol, timeout=chase_interval_seconds
                )

                if filled_order is not None:
                    from exchange.base_exchange import OrderStatus as _OStatus
                    if filled_order.status == _OStatus.CLOSED:
                        logger.info(
                            "Maker entry filled: {} @ {:.4f} (saved taker fees, attempt {})",
                            symbol, filled_order.price or limit_price, attempt + 1,
                        )
                        return filled_order

                # Not filled - cancel and retry at new price
                try:
                    await self._exchange.cancel_order(order.id, symbol)
                except Exception:
                    pass

            except Exception as exc:
                logger.debug("Maker entry attempt {} failed: {}", attempt + 1, exc)

        # Fallback to market order
        if fallback_to_market:
            logger.info(
                "Maker entry failed after {} attempts - falling back to market order",
                max_chase_attempts,
            )
            return await self._order_manager.place_market_order(
                symbol=symbol, side=side, amount=amount, strategy=strategy
            )

        return None

    def _split_order(
        self,
        amount: float,
        max_chunk: float,
        min_order_size: float = 0.0,
    ) -> List[float]:
        """Split *amount* into chunks of at most *max_chunk*.

        If the final remaining chunk would be smaller than *min_order_size*,
        it is merged into the previous chunk rather than returned as a tiny
        invalid fraction.

        Args:
            amount: Total order size.
            max_chunk: Maximum size per chunk.
            min_order_size: Exchange minimum order size. Any remainder smaller
                than this value is folded into the preceding chunk.

        Returns:
            List of chunk sizes that sum to *amount*.
        """
        if max_chunk <= 0 or amount <= max_chunk:
            return [amount]
        chunks: List[float] = []
        remaining = amount
        while remaining > 0:
            chunk = min(max_chunk, remaining)
            chunks.append(round(chunk, 8))
            remaining = round(remaining - chunk, 8)
        # Merge a sub-minimum final chunk into the previous one.
        # Note: the merged chunk may exceed max_chunk, but that is
        # intentional — a slightly oversized chunk is preferable to
        # submitting an invalid sub-minimum order.
        if min_order_size > 0 and len(chunks) > 1 and chunks[-1] < min_order_size:
            chunks[-2] = round(chunks[-2] + chunks[-1], 8)
            chunks.pop()
        return chunks

    async def _calculate_optimal_entry(self, symbol: str, direction: str) -> float:
        """Return the optimal limit entry price for *symbol*.

        Uses the local cached order book first (zero latency) and falls back
        to a REST fetch when the cache is absent or stale.

        Args:
            symbol: Trading symbol.
            direction: ``"long"`` or ``"short"``.

        Returns:
            Optimal limit price, or 0.0 on failure.
        """
        try:
            orderbook = self._get_local_or_rest_orderbook_sync(symbol)
            if orderbook is None:
                orderbook = await self._exchange.get_orderbook(symbol, limit=5)
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if not bids or not asks:
                return 0.0
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            mid = (best_bid + best_ask) / 2.0
            side = "buy" if direction == "long" else "sell"
            return self._optimizer.calculate_optimal_limit_price(symbol, side, spread, mid)
        except Exception as exc:
            logger.error("Failed to calculate optimal entry for {}: {}", symbol, exc)
            return 0.0

    def _get_local_or_rest_orderbook_sync(self, symbol: str) -> Optional[dict]:
        """Return a fresh local book for *symbol*, or ``None`` if unavailable/stale.

        When the local book is fresh (< 2 s old) it is returned immediately
        without any network round-trip.  The caller is responsible for
        performing a REST fallback when ``None`` is returned.
        """
        if self._local_orderbook_manager is not None:
            return self._local_orderbook_manager.get_book(symbol)
        return None

    # ------------------------------------------------------------------
    # Database persistence helpers
    # ------------------------------------------------------------------

    async def _persist_active_position(
        self,
        symbol: str,
        exchange_id: str,
        strategy: str,
        direction: str,
        amount: float,
        entry_price: float,
        leverage: int,
        stop_loss: Optional[float],
        tp_levels: List[float],
    ) -> None:
        """Upsert an :class:`~data.storage.models.ActivePosition` record."""
        try:
            from datetime import datetime, timezone

            from sqlalchemy import select

            from data.storage.models import ActivePosition, get_async_session

            now = datetime.now(tz=timezone.utc)
            async with get_async_session() as session:
                result = await session.execute(
                    select(ActivePosition).where(ActivePosition.symbol == symbol)
                )
                record = result.scalar_one_or_none()
                if record is None:
                    record = ActivePosition(
                        exchange_id=exchange_id,
                        symbol=symbol,
                        strategy_name=strategy,
                        side="long" if direction == "long" else "short",
                        amount=amount,
                        entry_price=entry_price,
                        leverage=leverage,
                        stop_loss=stop_loss,
                        take_profit=tp_levels if tp_levels else None,
                        opened_at=now,
                        updated_at=now,
                    )
                    session.add(record)
                else:
                    record.exchange_id = exchange_id
                    record.strategy_name = strategy
                    record.side = "long" if direction == "long" else "short"
                    record.amount = amount
                    record.entry_price = entry_price
                    record.leverage = leverage
                    record.stop_loss = stop_loss
                    record.take_profit = tp_levels if tp_levels else None
                    record.updated_at = now
        except Exception as exc:
            logger.debug(
                "Could not persist active position for {} (strategy={} side={}): {}",
                symbol,
                strategy,
                "long" if direction == "long" else "short",
                exc,
            )

    async def _persist_active_order(
        self,
        exchange_id: str,
        symbol: str,
        strategy: str,
        order_type: str,
        side: str,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        entry_price: Optional[float] = None,
    ) -> None:
        """Insert an :class:`~data.storage.models.ActiveOrder` record."""
        try:
            from sqlalchemy import select

            from data.storage.models import ActiveOrder, get_async_session

            async with get_async_session() as session:
                result = await session.execute(
                    select(ActiveOrder).where(ActiveOrder.exchange_id == exchange_id)
                )
                if result.scalar_one_or_none() is None:
                    record = ActiveOrder(
                        exchange_id=exchange_id,
                        symbol=symbol,
                        strategy_name=strategy,
                        order_type=order_type,
                        side=side,
                        amount=amount,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    session.add(record)
        except Exception as exc:
            logger.debug(
                "Could not persist active order {} ({} {} for {}): {}",
                exchange_id,
                order_type,
                side,
                symbol,
                exc,
            )

    async def _update_active_position_sltp(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        tp_levels=None,
    ) -> None:
        """Update SL / TP on an existing :class:`~data.storage.models.ActivePosition`."""
        try:
            from datetime import datetime, timezone

            from sqlalchemy import select

            from data.storage.models import ActivePosition, get_async_session

            async with get_async_session() as session:
                result = await session.execute(
                    select(ActivePosition).where(ActivePosition.symbol == symbol)
                )
                record = result.scalar_one_or_none()
                if record is not None:
                    if stop_loss is not None:
                        record.stop_loss = stop_loss
                    if tp_levels is not None:
                        levels = tp_levels if isinstance(tp_levels, list) else [tp_levels]
                        record.take_profit = sorted(float(v) for v in levels if v is not None)
                    record.updated_at = datetime.now(tz=timezone.utc)
        except Exception as exc:
            logger.debug(
                "Could not update active position SL/TP for {} (sl={} tp={}): {}",
                symbol,
                stop_loss,
                tp_levels,
                exc,
            )

    async def _delete_active_position(self, symbol: str) -> None:
        """Remove the :class:`~data.storage.models.ActivePosition` row for *symbol*."""
        try:
            from sqlalchemy import delete

            from data.storage.models import ActiveOrder, ActivePosition, get_async_session

            async with get_async_session() as session:
                await session.execute(
                    delete(ActiveOrder).where(ActiveOrder.symbol == symbol)
                )
                await session.execute(
                    delete(ActivePosition).where(ActivePosition.symbol == symbol)
                )
        except Exception as exc:
            logger.debug(
                "Could not delete active position/order records for {} ({}): {}",
                symbol,
                type(exc).__name__,
                exc,
            )

    def _get_telegram_alerter(self) -> Optional[Any]:
        """Lazily initialise and return the :class:`~monitoring.alerting.TelegramAlerter`."""
        if self._telegram_alerter is None:
            try:
                from monitoring.alerting import TelegramAlerter
                self._telegram_alerter = TelegramAlerter()
            except Exception as exc:
                logger.debug("TelegramAlerter unavailable: {}", exc)
        return self._telegram_alerter

    async def _send_order_alert(self, entry_order: Any, signal: dict) -> None:
        """Send a Telegram order alert after a successful entry order.

        Errors are swallowed so they never interrupt the trading flow.
        """
        try:
            alerter = self._get_telegram_alerter()
            if alerter is None:
                return

            from config.settings import Settings
            settings = Settings.get_settings()
            mode = getattr(settings, "trading_mode", "paper")

            order_dict = {
                "symbol": signal.get("symbol", ""),
                "direction": signal.get("direction", ""),
                "leverage": signal.get("leverage", 1),
                "price": entry_order.price or 0.0,
                "amount": entry_order.amount or 0.0,
                "strategy": signal.get("strategy", ""),
            }

            # Try to fetch current free margin
            free_margin = 0.0
            try:
                balance = await self._exchange.get_balance()
                free_margin = float(balance.usdt_free) if hasattr(balance, "usdt_free") else 0.0
            except Exception:
                pass

            await alerter.send_order_alert(order_dict, margin_remaining=free_margin, mode=mode)
        except Exception as exc:
            logger.debug("_send_order_alert failed: {}", exc)

    # ------------------------------------------------------------------
    # Advanced Execution API
    # ------------------------------------------------------------------

    def get_latency_summary(self) -> Dict:
        """Get latency monitoring summary.

        Returns:
            Dict with latency statistics, or empty dict if disabled
        """
        if self._latency_monitor:
            return self._latency_monitor.get_summary()
        return {}

    def get_execution_quality_report(self) -> Dict:
        """Get comprehensive execution quality report.

        Returns:
            Dict with quality statistics, or empty dict if disabled
        """
        if self._execution_quality_analyzer:
            return self._execution_quality_analyzer.generate_comprehensive_report()
        return {}

    async def evaluate_smart_exits(self, positions: List) -> List:
        """Evaluate smart exit signals for open positions.

        Args:
            positions: List of open positions

        Returns:
            List of exit signals
        """
        if self._smart_exit_engine:
            return await self._smart_exit_engine.evaluate_exits(positions)
        return []

    def get_anti_gaming_summary(self, symbol: str) -> Dict:
        """Get anti-gaming protection summary for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with protection status, or empty dict if disabled
        """
        if self._anti_gaming:
            return self._anti_gaming.get_protection_summary(symbol)
        return {}

    async def execute_with_adaptive_algorithm(
        self,
        symbol: str,
        side: str,
        amount: float,
        algorithm: str = "adaptive_twap",
        **kwargs
    ) -> Dict:
        """Execute order using adaptive execution algorithm.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            amount: Amount to execute
            algorithm: "implementation_shortfall", "adaptive_twap", "adaptive_vwap", "iceberg", or "sniper"
            **kwargs: Algorithm-specific parameters

        Returns:
            Dict with execution result
        """
        if not self._adaptive_engine:
            return {"success": False, "error": "Advanced execution disabled"}

        if algorithm == "implementation_shortfall":
            result = await self._adaptive_engine.execute_implementation_shortfall(
                symbol=symbol,
                side=side,
                total_amount=amount,
                **kwargs
            )
        elif algorithm == "adaptive_twap":
            result = await self._adaptive_engine.execute_adaptive_twap(
                symbol=symbol,
                side=side,
                total_amount=amount,
                **kwargs
            )
        elif algorithm == "adaptive_vwap":
            result = await self._adaptive_engine.execute_adaptive_vwap(
                symbol=symbol,
                side=side,
                total_amount=amount,
                **kwargs
            )
        elif algorithm == "iceberg":
            result = await self._adaptive_engine.execute_iceberg(
                symbol=symbol,
                side=side,
                total_amount=amount,
                **kwargs
            )
        elif algorithm == "sniper":
            result = await self._adaptive_engine.execute_sniper(
                symbol=symbol,
                side=side,
                amount=amount,
                **kwargs
            )
        else:
            return {"success": False, "error": f"Unknown algorithm: {algorithm}"}

        # Convert ExecutionResult to dict
        return {
            "success": result.success,
            "filled_amount": result.filled_amount,
            "average_price": result.average_price,
            "total_slices": result.total_slices,
            "completed_slices": result.completed_slices,
            "total_time_seconds": result.total_time_seconds,
            "error": result.error
        }

    # ------------------------------------------------------------------
    # Reconciliation (Rust ↔ Exchange REST)
    # ------------------------------------------------------------------

    async def reconcile_positions_with_rust(
        self,
        rust_positions: Dict[str, Any],
        tolerance_pct: float = 0.01,
    ) -> Dict[str, Any]:
        """Compare Rust-reported positions against exchange REST state.

        Fetches the live position list via REST and compares it with the
        ``rust_positions`` dict (keyed by symbol) that the Rust engine last
        reported via ZeroMQ telemetry.  Emits a warning for any discrepancy
        larger than ``tolerance_pct``.

        Args:
            rust_positions: Dict mapping symbol → position payload, typically
                sourced from ``GateIOClient.get_positions_from_rust()`` or the
                ``EventDrivenEngine._state_cache``.
            tolerance_pct: Maximum relative size deviation considered acceptable
                (default 1 %).

        Returns:
            Dict with keys:
              ``reconciled`` (list of matching symbols),
              ``discrepancies`` (list of dicts with symbol, rust_size, exchange_size,
              diff_pct),
              ``exchange_only`` (symbols in REST but not in Rust positions),
              ``rust_only`` (symbols in Rust but not in REST).
        """
        result: Dict[str, Any] = {
            "reconciled": [],
            "discrepancies": [],
            "exchange_only": [],
            "rust_only": [],
        }

        try:
            rest_positions = await self._exchange.get_positions()
        except Exception as exc:
            logger.warning("reconcile_positions_with_rust: REST call failed: {}", exc)
            return result

        rest_by_symbol: Dict[str, Any] = {
            p.symbol: p for p in rest_positions if p.size != 0
        }

        # Check all Rust-reported positions
        for symbol, rust_pos in rust_positions.items():
            rust_size = float(rust_pos.get("size", rust_pos.get("amount", 0)))
            if symbol in rest_by_symbol:
                rest_size = float(rest_by_symbol[symbol].size)
                if rest_size == 0 and rust_size == 0:
                    result["reconciled"].append(symbol)
                    continue
                denom = max(abs(rest_size), abs(rust_size), 1e-12)
                diff_pct = abs(rest_size - rust_size) / denom
                if diff_pct > tolerance_pct:
                    result["discrepancies"].append({
                        "symbol": symbol,
                        "rust_size": rust_size,
                        "exchange_size": rest_size,
                        "diff_pct": round(diff_pct * 100, 4),
                    })
                    logger.warning(
                        "Position discrepancy for {}: Rust={:.4f}, REST={:.4f} ({:.2f}%)",
                        symbol, rust_size, rest_size, diff_pct * 100,
                    )
                else:
                    result["reconciled"].append(symbol)
            else:
                result["rust_only"].append(symbol)
                logger.warning(
                    "Rust reports open position for {} but REST shows none.", symbol
                )

        # Check REST positions not known to Rust
        for symbol in rest_by_symbol:
            if symbol not in rust_positions:
                result["exchange_only"].append(symbol)
                logger.warning(
                    "REST reports open position for {} but Rust has no record.", symbol
                )

        return result

