#!/usr/bin/env python3
"""
Main entry point for the Crypto Trading Bot.

Usage:
  python main.py run                 # Start live/paper trading
  python main.py backtest            # Run backtests
  python main.py optimize            # Optimize strategy parameters
  python main.py dashboard           # Start dashboard only
  python main.py setup               # Initial setup wizard
  python main.py generate-keys       # Generate encryption keys
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure the package root is on sys.path so bare imports work.
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Also add the parent so that `crypto_trading_bot.*` package-prefixed imports
# inside existing __init__.py files resolve correctly.
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from config.logging_config import configure_logging  # noqa: E402
from config.settings import Settings  # noqa: E402

# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-trading-bot",
        description="Crypto Trading Bot — AI-powered perpetual futures trading",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level from settings",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Start live/paper trading")
    run_parser.add_argument(
        "--mode",
        choices=["live", "paper", "testnet"],
        default=None,
        help="Override trading_mode from settings",
    )

    # ── backtest ──────────────────────────────────────────────────────────────
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest")
    bt_parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    bt_parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: 1 year ago)",
    )
    bt_parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    bt_parser.add_argument(
        "--from",
        dest="from_date",
        default=None,
        metavar="FROM",
        help="Start date YYYY-MM-DD (alias for --start)",
    )
    bt_parser.add_argument(
        "--to",
        dest="to_date",
        default=None,
        metavar="TO",
        help="End date YYYY-MM-DD (alias for --end)",
    )
    bt_parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial capital in USDT (default: 10000)",
    )
    bt_parser.add_argument(
        "--strategy",
        default="momentum",
        help="Strategy name to backtest (default: momentum)",
    )
    bt_parser.add_argument(
        "--timeframe",
        default="1h",
        help="OHLCV timeframe (default: 1h)",
    )

    # ── optimize ─────────────────────────────────────────────────────────────
    opt_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    opt_parser.add_argument("--symbol", default="BTC/USDT")
    opt_parser.add_argument("--start", default=None)
    opt_parser.add_argument("--end", default=None)
    opt_parser.add_argument(
        "--method",
        choices=["grid", "random"],
        default="random",
        help="Optimisation method (default: random)",
    )
    opt_parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of random trials (default: 50)",
    )
    opt_parser.add_argument(
        "--strategy",
        default="technical_breakout",
        help="Strategy name to optimise (default: technical_breakout)",
    )

    # ── dashboard ─────────────────────────────────────────────────────────────
    dash_parser = subparsers.add_parser("dashboard", help="Start the monitoring dashboard")
    dash_parser.add_argument("--host", default="0.0.0.0")
    dash_parser.add_argument("--port", type=int, default=None)

    # ── setup ─────────────────────────────────────────────────────────────────
    subparsers.add_parser("setup", help="Interactive initial setup wizard")

    # ── generate-keys ─────────────────────────────────────────────────────────
    subparsers.add_parser("generate-keys", help="Generate encryption and secret keys")

    return parser


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


async def _cmd_run(args: argparse.Namespace, settings: Settings) -> None:
    """Start the trading engine."""
    if args.mode:
        # Override trading_mode without reloading settings from env
        object.__setattr__(settings, "trading_mode", args.mode)

    mode = settings.trading_mode
    logger.info("Trading mode: {}", mode)

    # Determine which engines to run
    run_forex = mode.startswith("forex_")
    run_both = (
        not run_forex
        and mode in ("live", "paper", "testnet")
        and getattr(settings, "enable_forex_trading", False)
    )

    engines = []

    if run_forex:
        # Pure forex mode — only run the forex engine
        from core.forex_engine import ForexTradingEngine
        engines.append(ForexTradingEngine(settings))
    elif run_both:
        # Dual mode: crypto + forex simultaneously
        from core.engine import TradingEngine
        from core.forex_engine import ForexTradingEngine
        engines.append(TradingEngine(settings))
        engines.append(ForexTradingEngine(settings))
        logger.info("Dual mode: running crypto + forex engines simultaneously")
    else:
        # Pure crypto mode
        from core.engine import TradingEngine
        engines.append(TradingEngine(settings))

    async def _run_engine(eng: Any) -> None:
        try:
            await eng.start()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — shutting down {}", type(eng).__name__)
            await eng.stop()
        except Exception as exc:
            logger.critical("Fatal error in {}: {}", type(eng).__name__, exc)
            try:
                await eng.stop()
            except Exception:
                pass
            raise

    try:
        if len(engines) == 1:
            await _run_engine(engines[0])
        else:
            # Run multiple engines concurrently
            await asyncio.gather(*[_run_engine(e) for e in engines])
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping all engines")
        for eng in engines:
            try:
                await eng.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


async def _cmd_backtest(args: argparse.Namespace, settings: Settings) -> None:
    """Run a backtest using :class:`BacktestEngine` and :class:`BacktestReport`."""
    from backtest.backtest_engine import BacktestEngine
    from backtest.backtest_report import BacktestReport

    # --from/--to take priority over --start/--end when both are supplied
    start_str = getattr(args, "from_date", None) or args.start
    end_str = getattr(args, "to_date", None) or args.end

    end_date = datetime.strptime(end_str, "%Y-%m-%d") if end_str else datetime.now(tz=timezone.utc)
    start_date = (
        datetime.strptime(start_str, "%Y-%m-%d") if start_str else end_date - timedelta(days=365)
    )

    timeframe = getattr(args, "timeframe", "1h") or "1h"
    strategy = _load_strategy(args.strategy, settings, symbols=[args.symbol])

    logger.info(
        "Running backtest: {} on {} {} ({} → {})",
        args.strategy,
        args.symbol,
        timeframe,
        start_date.date(),
        end_date.date(),
    )

    engine = BacktestEngine()
    result = await engine.run(
        strategy=strategy,
        symbol=args.symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
    )

    report = BacktestReport()
    paths = report.generate(result)

    logger.info(
        "Backtest complete — return: {:.2f}%, Sharpe: {:.3f}, max DD: {:.2f}%",
        result.total_return_pct,
        result.sharpe_ratio,
        result.max_drawdown_pct,
    )
    if paths.get("json_path"):
        logger.info("Report saved to {}", paths["json_path"])
    if paths.get("chart_path"):
        logger.info("Equity chart saved to {}", paths["chart_path"])


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


async def _cmd_optimize(args: argparse.Namespace, settings: Settings) -> None:
    """Run strategy parameter optimisation."""
    from backtest.backtester import Backtester

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now(tz=timezone.utc)
    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d") if args.start else end_date - timedelta(days=180)
    )

    strategy_class = _load_strategy_class(args.strategy, settings)
    # Default parameter ranges for optimisation — strategy-specific ranges
    # should be passed programmatically when integrating with your pipeline.
    param_ranges = _default_param_ranges(args.strategy)

    backtester = Backtester()
    logger.info(
        "Optimising {} on {} ({} → {}) using {} search",
        args.strategy,
        args.symbol,
        start_date.date(),
        end_date.date(),
        args.method,
    )

    if args.method == "random":
        best = await backtester.optimize(
            strategy_class=strategy_class,
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            param_ranges=param_ranges,
            method="random",
        )
    else:
        best = await backtester.optimize(
            strategy_class=strategy_class,
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            param_ranges=param_ranges,
            method="grid",
        )

    logger.info("Best parameters: {}", best.get("params", {}))
    logger.info("Best metrics: {}", best.get("metrics", {}))


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------


async def _cmd_dashboard(args: argparse.Namespace, settings: Settings) -> None:
    """Start only the monitoring dashboard."""
    from monitoring.dashboard import TradingDashboard

    port = args.port or settings.monitoring.dashboard_port
    dashboard = TradingDashboard(settings=settings)
    logger.info("Starting dashboard on {}:{}", args.host, port)
    try:
        await dashboard.run(host=args.host, port=port)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as exc:
        logger.error("Dashboard error: {}", exc)
        raise


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def _cmd_setup() -> None:
    """Interactive setup wizard."""
    package_dir = Path(__file__).parent
    env_example = package_dir / ".env.example"
    env_file = package_dir / ".env"

    print("\n🤖  Crypto Trading Bot — Setup Wizard\n" + "─" * 45)

    # Create .env from example if missing
    if not env_file.exists():
        if env_example.exists():
            import shutil

            shutil.copy(env_example, env_file)
            print(f"✅  Created {env_file} from .env.example")
            print("    → Edit .env and add your API keys before running the bot.")
        else:
            print(f"⚠️   No .env.example found. Create {env_file} manually.")
    else:
        print(f"ℹ️   {env_file} already exists — skipping copy.")

    # Run Alembic migrations
    alembic_ini = package_dir / "alembic.ini"
    if alembic_ini.exists():
        print("\n🔄  Running database migrations (alembic upgrade head)…")
        try:
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                cwd=str(package_dir),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode == 0:
                print("✅  Migrations applied successfully.")
            else:
                print(f"⚠️   Alembic returned exit code {result.returncode}:")
                if result.stderr:
                    print(result.stderr)
        except FileNotFoundError:
            print("⚠️   alembic not found. Install with: pip install alembic")
        except subprocess.TimeoutExpired:
            print("⚠️   Migration timed out.")
    else:
        print("ℹ️   No alembic.ini found — skipping migrations.")

    print(
        "\n📋  Next steps:\n"
        "  1. Edit .env with your exchange API keys\n"
        "  2. Run `python main.py generate-keys` to create encryption keys\n"
        "  3. Run `python main.py run` to start paper trading\n"
        "  4. Run `python main.py dashboard` to view the monitoring UI\n"
    )


# ---------------------------------------------------------------------------
# generate-keys
# ---------------------------------------------------------------------------


def _cmd_generate_keys() -> None:
    """Generate a Fernet encryption key and a random secret key."""
    from utils.encryption import KeyManager

    fernet_key = KeyManager.generate_key()
    secret_key = secrets.token_urlsafe(32)

    print("\n🔑  Generated Keys\n" + "─" * 45)
    print(f"ENCRYPTION_KEY={fernet_key}")
    print(f"SECRET_KEY={secret_key}")
    print(
        "\n⚠️  Add these to your .env file and keep them secret.\n"
        "   Changing ENCRYPTION_KEY will invalidate encrypted API keys.\n"
    )


# ---------------------------------------------------------------------------
# Strategy loader helpers
# ---------------------------------------------------------------------------


def _load_strategy(name: str, settings: Settings, symbols: list | None = None):
    """Return an instantiated strategy object by name.

    Args:
        name: Strategy name as registered in the strategy map.
        settings: Application settings (used by some strategies).
        symbols: Optional list of symbols to pass to the strategy constructor.
            Defaults to ``["BTC/USDT"]`` for strategies that require it.
    """
    cls = _load_strategy_class(name, settings)
    _symbols = symbols or ["BTC/USDT"]
    try:
        return cls(symbols=_symbols)
    except TypeError:
        # Strategy doesn't accept a 'symbols' kwarg — try bare instantiation
        return cls()


def _load_strategy_class(name: str, settings: Settings):
    """Return the strategy class for the given *name*.

    Used primarily for backtesting (``python main.py backtest --strategy <name>``).
    Live and paper trading always use the intelligent :class:`StrategySelector`
    — see ``crypto_trading_bot/strategy/strategy_selector.py``.
    """
    strategy_map = {
        # ── Original strategies ──────────────────────────────────────────────
        "momentum": "strategy.strategies.momentum.MomentumStrategy",
        "technical_breakout": "strategy.strategies.technical_breakout.TechnicalBreakoutStrategy",
        "scalping": "strategy.strategies.scalping.ScalpingStrategy",
        "dca": "strategy.strategies.dca_strategy.DCAStrategy",
        "grid": "strategy.strategies.grid_trading.GridTradingStrategy",
        "ai_adaptive": "strategy.strategies.ai_adaptive.AIAdaptiveStrategy",
        "news_momentum": "strategy.strategies.news_momentum.NewsMomentumStrategy",
        "sentiment_reversal": "strategy.strategies.sentiment_reversal.SentimentReversalStrategy",
        "whale_follower": "strategy.strategies.whale_follower.WhaleFollowerStrategy",
        "funding_rate_arb": "strategy.strategies.funding_rate_arb.FundingRateArbStrategy",
        "market_making": "strategy.strategies.market_making.MarketMakingStrategy",
        "liquidation_hunter": "strategy.strategies.liquidation_hunter.LiquidationHunterStrategy",
        "smart_money_flow": "strategy.strategies.smart_money_flow.SmartMoneyFlowStrategy",
        # ── New strategies (31) ──────────────────────────────────────────────
        "bollinger_squeeze": "strategy.strategies.bollinger_squeeze.BollingerSqueezeStrategy",
        "vwap_deviation": "strategy.strategies.vwap_deviation.VWAPDeviationStrategy",
        "ichimoku_cloud": "strategy.strategies.ichimoku_cloud.IchimokuCloudStrategy",
        "fibonacci_retracement": "strategy.strategies.fibonacci_retracement.FibonacciRetracementStrategy",
        "order_flow_imbalance": "strategy.strategies.order_flow_imbalance.OrderFlowImbalanceStrategy",
        "volume_profile": "strategy.strategies.volume_profile.VolumeProfileStrategy",
        "rsi_divergence": "strategy.strategies.rsi_divergence.RSIDivergenceStrategy",
        "macd_crossover": "strategy.strategies.macd_crossover.MACDCrossoverStrategy",
        "ema_ribbon": "strategy.strategies.ema_ribbon.EMARibbonStrategy",
        "supertrend": "strategy.strategies.supertrend.SupertrendStrategy",
        "keltner_channel": "strategy.strategies.keltner_channel.KeltnerChannelStrategy",
        "donchian_breakout": "strategy.strategies.donchian_breakout.DonchianBreakoutStrategy",
        "parabolic_sar": "strategy.strategies.parabolic_sar.ParabolicSARStrategy",
        "stochastic_rsi": "strategy.strategies.stochastic_rsi.StochasticRSIStrategy",
        "williams_r": "strategy.strategies.williams_r.WilliamsRStrategy",
        "adx_trend": "strategy.strategies.adx_trend.ADXTrendStrategy",
        "pivot_point": "strategy.strategies.pivot_point.PivotPointStrategy",
        "harmonic_pattern": "strategy.strategies.harmonic_pattern.HarmonicPatternStrategy",
        "elliott_wave": "strategy.strategies.elliott_wave.ElliottWaveStrategy",
        "supply_demand_zone": "strategy.strategies.supply_demand_zone.SupplyDemandZoneStrategy",
        "market_structure_break": "strategy.strategies.market_structure_break.MarketStructureBreakStrategy",
        "fair_value_gap": "strategy.strategies.fair_value_gap.FairValueGapStrategy",
        "order_block": "strategy.strategies.order_block.OrderBlockStrategy",
        "accumulation_distribution": "strategy.strategies.accumulation_distribution.AccDistStrategy",
        "on_chain_momentum": "strategy.strategies.on_chain_momentum.OnChainMomentumStrategy",
        "correlation_divergence": "strategy.strategies.correlation_divergence.CorrelationDivergenceStrategy",
        "volatility_breakout": "strategy.strategies.volatility_breakout.VolatilityBreakoutStrategy",
        "time_based": "strategy.strategies.time_based.TimeBasedStrategy",
        "mtf_confluence": "strategy.strategies.multi_timeframe_confluence.MTFConfluenceStrategy",
        "range_breakout": "strategy.strategies.range_breakout.RangeBreakoutStrategy",
        "momentum_divergence": "strategy.strategies.momentum_divergence.MomentumDivergenceStrategy",
    }

    dotted_path = strategy_map.get(name)
    if dotted_path is None:
        logger.warning("Unknown strategy '{}' — using base strategy", name)
        from strategy.base_strategy import BaseStrategy

        return BaseStrategy

    module_path, class_name = dotted_path.rsplit(".", 1)
    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not load strategy {!r}: {} — using base strategy", name, exc)
        from strategy.base_strategy import BaseStrategy

        return BaseStrategy


def _default_param_ranges(strategy_name: str) -> dict:
    """Return sensible default parameter ranges for common strategies."""
    defaults: dict = {
        "technical_breakout": {
            "lookback_period": [10, 20, 30, 50],
            "breakout_pct": [0.5, 1.0, 1.5, 2.0],
        },
        "scalping": {
            "rsi_period": [7, 14, 21],
            "rsi_overbought": [65, 70, 75],
        },
        "dca": {
            "interval_hours": [1, 4, 8, 24],
            "investment_pct": [0.05, 0.10, 0.15],
        },
    }
    return defaults.get(strategy_name, {"dummy_param": [1, 2, 3]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load settings (may fail if .env is missing — that's fine for setup/generate-keys)
    settings = None
    if args.command not in ("setup", "generate-keys"):
        try:
            settings = Settings.get_settings()
        except Exception as exc:
            print(f"❌  Failed to load settings: {exc}", file=sys.stderr)
            print("    Run `python main.py setup` first.", file=sys.stderr)
            sys.exit(1)

        log_level = args.log_level or settings.log_level
        # Enable JSON logging when LOG_FORMAT=json env var is set, or when running in Docker
        import os as _os
        _log_format = _os.environ.get("LOG_FORMAT", "text").lower()
        _in_docker = _os.path.exists("/.dockerenv") or _os.environ.get("DOCKER", "").lower() == "true"
        _use_json = _log_format == "json" or _in_docker
        _colorize = not _use_json
        configure_logging(
            log_level=log_level,
            log_file="data/logs/trading.log",
            json_logs=_use_json,
            colorize=_colorize,
        )
    else:
        # Minimal logging for setup / key generation
        configure_logging(log_level=args.log_level or "INFO")

    # Dispatch subcommand
    try:
        if args.command == "run":
            asyncio.run(_cmd_run(args, settings))
        elif args.command == "backtest":
            asyncio.run(_cmd_backtest(args, settings))
        elif args.command == "optimize":
            asyncio.run(_cmd_optimize(args, settings))
        elif args.command == "dashboard":
            asyncio.run(_cmd_dashboard(args, settings))
        elif args.command == "setup":
            _cmd_setup()
        elif args.command == "generate-keys":
            _cmd_generate_keys()
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.critical("Unhandled exception: {}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
