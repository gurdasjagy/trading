"""Pre-flight check system that runs before the trading loop starts.

Verifies exchange connectivity, configured trading pairs, API key permissions,
and risk parameter sanity.  Blocks live trading if any critical check fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from exchange.base_exchange import BaseExchange
    from monitoring.alerting import AlertManager


class PreflightCheckResult:
    """Result of a single pre-flight check."""

    def __init__(self, name: str, passed: bool, message: str, critical: bool = True) -> None:
        self.name = name
        self.passed = passed
        self.message = message
        self.critical = critical

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else ("❌ FAIL" if self.critical else "⚠️ WARN")
        return f"{status} [{self.name}]: {self.message}"


class PreflightChecker:
    """Runs pre-flight checks before the trading loop starts.

    Usage::

        checker = PreflightChecker(
            exchange=engine.exchange,
            settings=engine.settings,
            alert_manager=engine.alert_manager,
        )
        ok = await checker.run_all()
        if not ok:
            raise RuntimeError("Pre-flight checks failed — refusing to start live trading")
    """

    def __init__(
        self,
        exchange: "BaseExchange",
        settings: "Settings",
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        self._exchange = exchange
        self._settings = settings
        self._alert_manager = alert_manager
        self._results: List[PreflightCheckResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_all(self) -> bool:
        """Run all pre-flight checks.

        Returns:
            True if all critical checks passed; False if any critical check failed.
        """
        self._results.clear()
        logger.info("🚦 Running pre-flight checks…")

        await self._check_exchange_connectivity()
        await self._check_trading_pairs()
        await self._check_risk_parameters()
        await self._check_paper_trading_balance()

        # Only check API permissions for non-paper modes
        is_paper = getattr(self._settings, "is_paper_trading", False)
        if not is_paper:
            await self._check_api_permissions()

        # Log all results
        for result in self._results:
            logger.info("{}", result)

        critical_failures = [r for r in self._results if not r.passed and r.critical]
        warnings = [r for r in self._results if not r.passed and not r.critical]

        if warnings:
            logger.warning(
                "Pre-flight warnings: {}",
                "; ".join(r.message for r in warnings),
            )

        if critical_failures:
            failure_msg = "❌ Pre-flight checks FAILED:\n" + "\n".join(
                f"  • {r}" for r in critical_failures
            )
            logger.error(failure_msg)
            if self._alert_manager is not None:
                try:
                    await self._alert_manager.send_alert(failure_msg, level="critical")
                except Exception as exc:
                    logger.debug("Failed to send pre-flight alert: {}", exc)
            return False

        logger.info(
            "✅ All pre-flight checks passed ({} checks, {} warnings).",
            len(self._results),
            len(warnings),
        )
        return True

    def get_summary(self) -> Dict:
        """Return a summary dict of all check results."""
        return {
            "total": len(self._results),
            "passed": sum(1 for r in self._results if r.passed),
            "failed": sum(1 for r in self._results if not r.passed and r.critical),
            "warnings": sum(1 for r in self._results if not r.passed and not r.critical),
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "critical": r.critical,
                    "message": r.message,
                }
                for r in self._results
            ],
        }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _check_exchange_connectivity(self) -> None:
        """Verify exchange connectivity: fetch balance, ticker, and order book."""
        exchange_cfg = getattr(self._settings, "exchange", None)
        trading_pairs: List[str] = getattr(exchange_cfg, "trading_pairs", [])
        test_symbol = trading_pairs[0] if trading_pairs else "BTC/USDT"

        # Check balance
        try:
            balance = await self._exchange.get_balance()
            self._results.append(
                PreflightCheckResult(
                    "exchange_balance",
                    True,
                    f"Balance fetched: USDT free={balance.usdt_free:.2f}",
                )
            )
        except Exception as exc:
            self._results.append(
                PreflightCheckResult(
                    "exchange_balance",
                    False,
                    f"Cannot fetch balance: {exc}",
                )
            )

        # Check ticker
        try:
            ticker = await self._exchange.get_ticker(test_symbol)
            self._results.append(
                PreflightCheckResult(
                    "exchange_ticker",
                    True,
                    f"Ticker OK for {test_symbol}: last={ticker.last}",
                )
            )
        except Exception as exc:
            self._results.append(
                PreflightCheckResult(
                    "exchange_ticker",
                    False,
                    f"Cannot fetch ticker for {test_symbol}: {exc}",
                )
            )

        # Check order book
        try:
            ob = await self._exchange.get_orderbook(test_symbol, limit=5)
            bids = ob.get("bids", [])
            self._results.append(
                PreflightCheckResult(
                    "exchange_orderbook",
                    True,
                    f"Order book OK for {test_symbol}: {len(bids)} bids",
                )
            )
        except Exception as exc:
            self._results.append(
                PreflightCheckResult(
                    "exchange_orderbook",
                    False,
                    f"Cannot fetch order book for {test_symbol}: {exc}",
                    critical=False,
                )
            )

    async def _check_trading_pairs(self) -> None:
        """Verify all configured trading pairs exist on the exchange."""
        exchange_cfg = getattr(self._settings, "exchange", None)
        trading_pairs: List[str] = getattr(exchange_cfg, "trading_pairs", [])

        if not trading_pairs:
            self._results.append(
                PreflightCheckResult(
                    "trading_pairs",
                    False,
                    "No trading pairs configured",
                )
            )
            return

        try:
            markets = await self._exchange.get_markets()
            if not markets:
                self._results.append(
                    PreflightCheckResult(
                        "trading_pairs",
                        False,
                        "Could not load exchange markets",
                        critical=False,
                    )
                )
                return

            missing: List[str] = []
            for pair in trading_pairs:
                # Check both spot and swap forms
                swap_pair = f"{pair}:{pair.split('/')[-1]}" if ":" not in pair else pair
                if pair not in markets and swap_pair not in markets:
                    # For precious metals, the mapping makes them available via XAUT
                    from exchange.ccxt_exchange import CcxtExchange
                    pm_mapping = CcxtExchange.PRECIOUS_METALS_MAPPING
                    mapped = pm_mapping.get(pair)
                    if mapped:
                        mapped_swap = f"{mapped}:{mapped.split('/')[-1]}"
                        if mapped not in markets and mapped_swap not in markets:
                            missing.append(f"{pair} (mapped to {mapped}, not found)")
                    else:
                        missing.append(pair)

            if missing:
                self._results.append(
                    PreflightCheckResult(
                        "trading_pairs",
                        False,
                        f"Pairs not found on exchange: {', '.join(missing)}",
                        critical=False,  # warn but don't block — may still work via mapping
                    )
                )
            else:
                self._results.append(
                    PreflightCheckResult(
                        "trading_pairs",
                        True,
                        f"All {len(trading_pairs)} trading pairs verified on exchange",
                    )
                )
        except Exception as exc:
            self._results.append(
                PreflightCheckResult(
                    "trading_pairs",
                    False,
                    f"Could not verify trading pairs: {exc}",
                    critical=False,
                )
            )

    async def _check_api_permissions(self) -> None:
        """Check API key has sufficient permissions (read balance, read positions)."""
        try:
            await self._exchange.get_positions()
            self._results.append(
                PreflightCheckResult(
                    "api_permissions",
                    True,
                    "API key has position read permission",
                )
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "permission" in exc_str or "auth" in exc_str or "key" in exc_str:
                self._results.append(
                    PreflightCheckResult(
                        "api_permissions",
                        False,
                        f"API key permission error: {exc}",
                    )
                )
            else:
                self._results.append(
                    PreflightCheckResult(
                        "api_permissions",
                        True,
                        f"API key seems OK (positions fetch: {exc})",
                        critical=False,
                    )
                )

    async def _check_risk_parameters(self) -> None:
        """Validate that risk parameters are sane."""
        risk_cfg = getattr(self._settings, "risk", None)
        if risk_cfg is None:
            self._results.append(
                PreflightCheckResult(
                    "risk_parameters",
                    False,
                    "No risk configuration found",
                )
            )
            return

        issues: List[str] = []

        sl_pct = getattr(risk_cfg, "default_stop_loss_pct", 0)
        if sl_pct <= 0:
            issues.append(f"default_stop_loss_pct must be > 0, got {sl_pct}")

        max_pos = getattr(risk_cfg, "max_position_size_pct", 0)
        if max_pos <= 0:
            issues.append(f"max_position_size_pct must be > 0, got {max_pos}")

        max_daily_loss = getattr(risk_cfg, "max_daily_loss_pct", 0)
        if max_daily_loss <= 0:
            issues.append(f"max_daily_loss_pct must be > 0, got {max_daily_loss}")

        if issues:
            self._results.append(
                PreflightCheckResult(
                    "risk_parameters",
                    False,
                    "Insane risk parameters: " + "; ".join(issues),
                )
            )
        else:
            self._results.append(
                PreflightCheckResult(
                    "risk_parameters",
                    True,
                    f"Risk parameters OK: SL={sl_pct}% max_pos={max_pos}% daily_loss={max_daily_loss}%",
                )
            )

    async def _check_paper_trading_balance(self) -> None:
        """Verify paper trading balance is set when in paper mode."""
        is_paper = getattr(self._settings, "is_paper_trading", False)
        if not is_paper:
            return

        paper_balance = getattr(
            getattr(self._settings, "exchange", None), "paper_trading_balance", None
        )
        if paper_balance is None:
            paper_balance = getattr(self._settings, "paper_trading_balance", None)

        if paper_balance is not None and float(paper_balance) > 0:
            self._results.append(
                PreflightCheckResult(
                    "paper_balance",
                    True,
                    f"Paper trading balance set: {paper_balance} USDT",
                )
            )
        else:
            self._results.append(
                PreflightCheckResult(
                    "paper_balance",
                    False,
                    "Paper trading balance not set or zero",
                    critical=False,
                )
            )
