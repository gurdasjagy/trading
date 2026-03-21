"""Pre-flight safety checks for live trading readiness.

Run :meth:`LiveReadinessChecker.run_all_checks` before allowing
``TRADING_MODE=live`` to proceed.  All checks must pass; any failure
is surfaced as a human-readable error message in the returned list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from exchange.base_exchange import BaseExchange


class LiveReadinessChecker:
    """Validates that the system is ready to trade live.

    Performs API key validation, balance checks, Telegram alert tests,
    trading-pair availability checks, and risk-settings sanity checks
    before allowing live trading to proceed.
    """

    #: Minimum USDT free balance required to start live trading.
    _MIN_BALANCE_USDT: float = 100.0
    #: Maximum allowed ``max_position_size_pct`` for live mode.
    _MAX_POSITION_SIZE_PCT: float = 15.0
    #: Maximum allowed ``max_leverage`` for live mode.
    _MAX_LEVERAGE: int = 20

    async def run_all_checks(
        self,
        settings: "Settings",
        exchange: "BaseExchange",
    ) -> Tuple[bool, List[str]]:
        """Run all live-readiness checks and return a pass/fail summary.

        Args:
            settings: Application settings instance.
            exchange: Fully-initialised exchange client.

        Returns:
            A tuple of ``(passed: bool, errors: list[str])``.  When
            ``passed`` is ``True`` the errors list is empty.
        """
        errors: List[str] = []

        # 1. API key validation — place and immediately cancel a tiny limit order
        try:
            pair = settings.exchange.trading_pairs[0]
            ticker = await exchange.get_ticker(pair)
            test_price = ticker.last * 0.5  # Far from market, won't fill
            order = await exchange.create_limit_order(pair, "buy", 1, test_price)
            await exchange.cancel_order(order.id, pair)
            logger.info("[LiveCheck] API key validation passed (order placed and cancelled)")
        except Exception as exc:
            errors.append(f"API key test failed: {exc}")
            logger.error("[LiveCheck] API key validation failed: {}", exc)

        # 2. Balance check — minimum $100 for live trading
        try:
            balance = await exchange.get_balance()
            usdt_free = float(
                balance.usdt_free if hasattr(balance, "usdt_free") else 0.0
            )
            if usdt_free < self._MIN_BALANCE_USDT:
                errors.append(
                    f"Insufficient balance: ${usdt_free:.2f} "
                    f"(min ${self._MIN_BALANCE_USDT:.0f})"
                )
            else:
                logger.info("[LiveCheck] Balance check passed: ${:.2f} free", usdt_free)
        except Exception as exc:
            errors.append(f"Balance check failed: {exc}")
            logger.error("[LiveCheck] Balance check error: {}", exc)

        # 3. Telegram alert test
        if getattr(getattr(settings, "monitoring", None), "enable_telegram_alerts", False):
            try:
                from monitoring.alerting import AlertManager

                alert_mgr = AlertManager(settings)
                success = await alert_mgr.send_alert(
                    "🔧 Live trading readiness test — ignore this message"
                )
                if not success:
                    errors.append("Telegram alerts not working")
                else:
                    logger.info("[LiveCheck] Telegram alert test passed")
            except Exception as exc:
                errors.append(f"Telegram alert test failed: {exc}")
                logger.error("[LiveCheck] Telegram alert test error: {}", exc)

        # 4. Check all trading pairs are valid on exchange
        try:
            markets: dict = {}
            if hasattr(exchange, "_client") and hasattr(exchange._client, "markets"):
                markets = exchange._client.markets or {}
            for pair in settings.exchange.trading_pairs:
                swap = f"{pair}:USDT" if ":" not in pair else pair
                if swap not in markets and pair not in markets:
                    errors.append(f"Trading pair {pair} not available on exchange")
                    logger.warning("[LiveCheck] Pair {} not found in exchange markets", pair)
                else:
                    logger.debug("[LiveCheck] Pair {} available on exchange", pair)
        except Exception as exc:
            logger.warning("[LiveCheck] Market availability check error: {}", exc)

        # 5. Risk settings sanity check
        risk = getattr(settings, "risk", None)
        if risk is not None:
            if getattr(risk, "max_position_size_pct", 0) > self._MAX_POSITION_SIZE_PCT:
                errors.append(
                    f"max_position_size_pct="
                    f"{risk.max_position_size_pct}% too high for live "
                    f"(max {self._MAX_POSITION_SIZE_PCT}%)"
                )
            if getattr(settings.exchange, "max_leverage", 0) > self._MAX_LEVERAGE:
                errors.append(
                    f"max_leverage={settings.exchange.max_leverage}x too high for live "
                    f"(max {self._MAX_LEVERAGE}x)"
                )

        # 6. Walk-forward validation for ML strategies (only in live mode)
        if settings.trading_mode == "live":
            try:
                # Get strategy manager from engine
                from core.engine import TradingEngine
                engine = getattr(TradingEngine, "_instance", None)
                if engine is not None and hasattr(engine, "strategy_manager"):
                    strategy_manager = engine.strategy_manager
                    
                    # Get enabled strategies
                    enabled_strategies = [
                        name for name, strat in strategy_manager._strategies.items()
                        if strat.enabled and hasattr(strat, "model") and strat.model is not None
                    ]
                    
                    if enabled_strategies:
                        logger.info(
                            "[LiveCheck] Validating {} ML strategies for live trading",
                            len(enabled_strategies),
                        )
                        
                        for strategy_name in enabled_strategies:
                            try:
                                passed, reasons = await strategy_manager.validate_strategy_for_live(
                                    strategy_name
                                )
                                if not passed:
                                    error_msg = (
                                        f"Strategy {strategy_name!r} failed walk-forward validation: "
                                        f"{', '.join(reasons)}"
                                    )
                                    errors.append(error_msg)
                                    logger.error("[LiveCheck] {}", error_msg)
                                else:
                                    logger.info(
                                        "[LiveCheck] Strategy {} passed walk-forward validation",
                                        strategy_name,
                                    )
                            except Exception as exc:
                                error_msg = f"Strategy {strategy_name!r} validation error: {exc}"
                                errors.append(error_msg)
                                logger.error("[LiveCheck] {}", error_msg)
                    else:
                        logger.info("[LiveCheck] No ML strategies to validate")
                else:
                    logger.warning("[LiveCheck] Strategy manager not available — skipping validation")
            except Exception as exc:
                logger.error("[LiveCheck] Walk-forward validation check error: {}", exc)

        passed = len(errors) == 0
        if passed:
            logger.info("[LiveCheck] All live readiness checks passed")
        else:
            logger.error(
                "[LiveCheck] {} check(s) failed: {}", len(errors), errors
            )
        return passed, errors
