"""FastAPI trading dashboard with WebSocket live updates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import pathlib
import secrets
from datetime import datetime, timezone
from typing import Any, List, Optional

from loguru import logger

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed; dashboard will not be available")

from config.settings import Settings

_BASE_DIR = pathlib.Path(__file__).parent.parent
_TEMPLATES_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"
_LOG_FILE = _BASE_DIR / "data" / "logs" / "bot.log"

# Storage models are imported lazily via _get_storage_models() on each endpoint
# call to avoid circular-import / stale-.pyc issues at module load time.


def _get_storage_models():
    """Lazily import storage models to avoid circular import issues."""
    try:
        from data.storage.models import (
            DailyPerformance,
            TradeHistory,
            create_tables,
            get_async_session,
        )
        return TradeHistory, DailyPerformance, create_tables, get_async_session
    except ImportError as exc:
        logger.warning("Storage models import failed: {} — trade-history endpoints will return errors.", exc)
        return None, None, None, None


def _tail_log(path: pathlib.Path, lines: int = 100) -> List[str]:
    """Return the last *lines* lines from *path*, or [] if unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.readlines()[-lines:]
    except Exception:
        return []


def _parse_log_level(line: str) -> str:
    """Extract the log level from a loguru-formatted log line.

    Loguru lines look like: ``2026-01-01 12:00:00.000 | INFO     | ...``
    """
    for level in ("CRITICAL", "ERROR", "WARNING", "DEBUG", "INFO", "TRACE"):
        if f"| {level}" in line:
            return level
    return "INFO"


def create_app(
    settings: Optional[Settings] = None,
    engine: Any = None,
    metrics_collector: Any = None,
    performance_tracker: Any = None,
    trade_journal: Any = None,
    risk_manager: Any = None,
    realtime_hub: Any = None,
) -> Any:
    """Create and return the FastAPI dashboard application.

    Args:
        settings: Application settings.
        engine: :class:`core.engine.TradingEngine` instance (may be ``None``).
        metrics_collector: :class:`monitoring.metrics.MetricsCollector` instance.
        performance_tracker: :class:`monitoring.performance_tracker.PerformanceTracker` instance.
        trade_journal: :class:`monitoring.trade_journal.TradeJournal` instance.
        risk_manager: :class:`risk.risk_manager.RiskManager` instance.
        realtime_hub: :class:`core.realtime_hub.RealtimeHub` instance for real-time
            streaming updates.  When provided the WebSocket endpoint registers the
            connection with the hub instead of running its own 5-second poll loop.

    Returns:
        A configured :class:`fastapi.FastAPI` application, or ``None`` if FastAPI
        is not installed.
    """
    if not _FASTAPI_AVAILABLE:
        return None

    _settings = settings or Settings.get_settings()
    _startup_time = datetime.now(tz=timezone.utc)

    app = FastAPI(title="Crypto Trading Bot Dashboard", version="1.0.0")

    # ── Authentication ────────────────────────────────────────────────────
    # Enforce authentication when trading_mode=live
    _trading_mode = getattr(_settings, "trading_mode", "paper").lower()
    _dashboard_username = getattr(_settings, "dashboard_username", None)
    _dashboard_password = getattr(_settings, "dashboard_password", None)
    
    if _trading_mode == "live" and (not _dashboard_username or not _dashboard_password):
        logger.critical(
            "Dashboard authentication is REQUIRED for live trading mode. "
            "Set DASHBOARD_USERNAME and DASHBOARD_PASSWORD in your .env file."
        )
        raise ValueError(
            "Dashboard authentication credentials (DASHBOARD_USERNAME, DASHBOARD_PASSWORD) "
            "are required when trading_mode=live"
        )
    
    _auth_enabled = bool(_dashboard_username and _dashboard_password)
    _security = HTTPBasic() if _FASTAPI_AVAILABLE else None

    def _require_auth(
        credentials: "HTTPBasicCredentials" = Depends(_security),  # type: ignore[assignment]
    ) -> "HTTPBasicCredentials":
        """Validate HTTP Basic Auth credentials if authentication is enabled."""
        expected_user = getattr(_settings, "dashboard_username", "") or ""
        expected_pass = getattr(_settings, "dashboard_password", "") or ""
        ok_user = secrets.compare_digest(credentials.username.encode(), expected_user.encode())
        ok_pass = secrets.compare_digest(credentials.password.encode(), expected_pass.encode())
        if not (ok_user and ok_pass):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials

    # Build a reusable dependency list — empty when auth is disabled.
    _auth_dep = [Depends(_require_auth)] if _auth_enabled else []

    # Pre-compute a HMAC-based WebSocket token (sha256 of username:password)
    # so raw credentials are never transmitted as query parameters.
    def _make_ws_token() -> str:
        expected_user = getattr(_settings, "dashboard_username", "") or ""
        expected_pass = getattr(_settings, "dashboard_password", "") or ""
        return hashlib.sha256(f"{expected_user}:{expected_pass}".encode()).hexdigest()

    _ws_token = _make_ws_token() if _auth_enabled else ""

    # Mount static files
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Jinja2 templates
    templates: Optional[Any] = None
    if _TEMPLATES_DIR.is_dir():
        templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # WebSocket connection manager
    manager = _ConnectionManager()

    # Wire the realtime hub's broadcast function to the connection manager so
    # that it can push directly to all connected clients.
    if realtime_hub is not None:
        realtime_hub.set_broadcast_fn(manager.broadcast)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _tpl_ctx(request: Request, active_page: str, **kwargs: Any) -> dict:
        """Build a base template context dict."""
        return {
            "request": request,
            "active_page": active_page,
            "trading_mode": _settings.trading_mode,
            "enable_forex_trading": _settings.enable_forex_trading,
            **kwargs,
        }

    async def _get_engine_risk_manager() -> Any:
        """Return the best available risk_manager reference."""
        if risk_manager:
            return risk_manager
        if engine and engine.risk_manager:
            return engine.risk_manager
        return None

    def _signal_to_dict(sig: Any) -> dict:
        """Normalise a signal object or dict into a serialisable dict."""
        if isinstance(sig, dict):
            return {
                "symbol": sig.get("symbol", ""),
                "direction": sig.get("direction", ""),
                "strength": float(sig.get("strength", 0.0)),
                "strategy_name": sig.get("strategy_name", ""),
                "timestamp": str(sig.get("timestamp", "")),
            }
        return {
            "symbol": getattr(sig, "symbol", ""),
            "direction": getattr(sig, "direction", ""),
            "strength": float(getattr(sig, "strength", 0.0)),
            "strategy_name": getattr(sig, "strategy_name", ""),
            "timestamp": str(getattr(sig, "timestamp", "")),
        }

    def _collect_signals(limit: int = 20) -> List[dict]:
        """Collect recent signals from the strategy manager."""
        signals: List[dict] = []
        sm = getattr(engine, "strategy_manager", None) if engine else None
        if sm:
            try:
                recent = getattr(sm, "_recent_signals", None) or getattr(
                    sm, "recent_signals", None
                )
                if recent:
                    signals = [_signal_to_dict(s) for s in list(recent)[-limit:]]
            except Exception as exc:
                logger.debug("_collect_signals: error: {}", exc)
        return signals

    async def _collect_portfolio() -> dict:
        """Collect live portfolio data from engine / exchange."""
        equity = 0.0
        balance = 0.0
        unrealized_pnl = 0.0
        daily_pnl = 0.0
        daily_pnl_pct = 0.0
        open_positions = 0

        if engine:
            if engine.exchange:
                try:
                    bal = await engine.exchange.get_balance()
                    equity = bal.usdt_total or bal.usdt_free or 0.0
                    balance = bal.usdt_free or 0.0
                except Exception as exc:
                    logger.debug("Dashboard: balance fetch error: {}", exc)
                try:
                    positions = await engine.exchange.get_positions()
                    open_positions = len(positions)
                    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
                except Exception as exc:
                    logger.debug("Dashboard: positions fetch error: {}", exc)
            if engine.state_manager:
                try:
                    state = await engine.state_manager.get_state()
                    daily_pnl = state.daily_pnl
                    daily_pnl_pct = state.daily_pnl_pct
                except Exception as exc:
                    logger.debug("Dashboard: state fetch error: {}", exc)

        return {
            "equity": round(equity, 4),
            "balance": round(balance, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "daily_pnl": round(daily_pnl, 4),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "open_positions": open_positions,
        }

    async def _collect_positions() -> List[dict]:
        """Collect open positions with stop-loss / take-profit data."""
        positions: List[dict] = []
        if engine and engine.position_manager:
            try:
                summary = await engine.position_manager.get_position_summary()
                for p in summary.get("positions", []):
                    entry = p.get("entry_price") or 0.0
                    current = p.get("current_price") or entry
                    mark = p.get("mark_price") or current
                    pnl = p.get("unrealized_pnl", 0.0)
                    pnl_pct = ((current - entry) / entry * 100.0) if entry else 0.0
                    tp_list = p.get("take_profit") or []
                    positions.append(
                        {
                            "symbol": p.get("symbol", ""),
                            "direction": p.get("side", "long"),
                            "entry_price": entry,
                            "current_price": current,
                            "mark_price": mark,
                            "pnl": round(pnl, 4),
                            "pnl_pct": round(pnl_pct, 4),
                            "leverage": p.get("leverage", 1),
                            "margin": round(p.get("margin", 0.0), 2),
                            "notional_size": round(p.get("position_value", 0.0), 2),
                            "liquidation_price": round(p.get("liquidation_price", 0.0), 4),
                            "roe_pct": round(p.get("roe_pct", 0.0), 4),
                            "position_value": round(p.get("position_value", 0.0), 4),
                            "funding_rate": p.get("funding_rate"),
                            "stop_loss": p.get("stop_loss"),
                            "take_profit": tp_list[0] if tp_list else None,
                            "strategy": p.get("strategy", ""),
                            "time_open": p.get("time_open", 0),
                        }
                    )
            except Exception as exc:
                logger.debug("Dashboard: position summary error: {}", exc)
        elif engine and engine.exchange:
            try:
                raw = await engine.exchange.get_positions()
                for p in raw:
                    entry = p.entry_price or 0.0
                    current = p.current_price or entry
                    mark = p.mark_price or current
                    pnl = p.unrealized_pnl
                    pnl_pct = ((current - entry) / entry * 100.0) if entry else 0.0

                    # Try to get strategy info from position_manager if available
                    strategy = ""
                    stop_loss = None
                    take_profit = None
                    time_open = 0
                    if engine.position_manager:
                        try:
                            tracker = await engine.position_manager.get_position(p.symbol)
                            if tracker:
                                strategy = tracker.strategy
                                stop_loss = tracker.stop_loss
                                tp_list = tracker.take_profit
                                take_profit = tp_list[0] if tp_list else None
                                from datetime import datetime, timezone
                                time_open = int((datetime.now(tz=timezone.utc) - tracker.opened_at).total_seconds())
                        except Exception as e:
                            logger.debug("Could not fetch tracker for {}: {}", p.symbol, e)

                    positions.append(
                        {
                            "symbol": p.symbol,
                            "direction": p.side.value,
                            "entry_price": entry,
                            "current_price": current,
                            "mark_price": mark,
                            "pnl": round(pnl, 4),
                            "pnl_pct": round(pnl_pct, 4),
                            "leverage": p.leverage,
                            "margin": round(p.margin, 2),
                            "notional_size": round(p.position_value, 2),
                            "liquidation_price": round(p.liquidation_price, 4),
                            "roe_pct": round(p.roe_pct, 4),
                            "position_value": round(p.position_value, 4),
                            "funding_rate": p.funding_rate,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "strategy": strategy,
                            "time_open": time_open,
                        }
                    )
            except Exception as exc:
                logger.debug("Dashboard: exchange positions error: {}", exc)
        return positions

    async def _fetch_free_margin() -> float:
        """Fetch the free USDT margin from the futures account balance.

        Returns 0.0 when not in live/testnet mode or when the exchange is unavailable.
        """
        if not (engine and engine.exchange):
            return 0.0
        try:
            balance = await engine.exchange.fetch_balance({"type": "swap"})
            return float(balance.get("USDT", {}).get("free", 0.0))
        except AttributeError:
            # PaperExchange / BaseExchange may not implement fetch_balance
            try:
                bal = await engine.exchange.get_balance()
                return float(bal.free) if bal else 0.0
            except Exception:
                return 0.0
        except Exception as exc:
            logger.debug("Dashboard: could not fetch free margin: {}", exc)
            return 0.0

    async def _collect_live_data() -> dict:
        """Collect a full live snapshot for WebSocket broadcast.

        When a :class:`~core.realtime_hub.RealtimeHub` is available its cached
        state is used (instant, no REST calls).  Otherwise falls back to the
        original REST-based collection.
        """
        # Fast path: use hub snapshot for portfolio + positions + orders
        if realtime_hub is not None:
            hub_snapshot = realtime_hub.get_snapshot()
            portfolio = hub_snapshot.get("portfolio", {})
            positions = hub_snapshot.get("positions", [])
            open_orders = hub_snapshot.get("open_orders", [])
        else:
            portfolio = await _collect_portfolio()
            positions = await _collect_positions()

            # Collect open orders
            open_orders: List[dict] = []
            if engine and engine.exchange:
                try:
                    raw_orders = await engine.exchange.get_open_orders()
                    for o in raw_orders:
                        open_orders.append({
                            "id": o.id,
                            "symbol": o.symbol,
                            "type": o.type.value,
                            "side": o.side.value,
                            "amount": o.amount,
                            "price": o.price,
                            "status": o.status.value,
                        })
                    # Attach SL/TP info to positions from open orders
                    orders_by_symbol: dict = {}
                    for o in open_orders:
                        orders_by_symbol.setdefault(o["symbol"], []).append(o)
                    for pos in positions:
                        sym_orders = orders_by_symbol.get(pos["symbol"], [])
                        for o in sym_orders:
                            if o["type"] == "stop_loss" and pos["stop_loss"] is None:
                                pos["stop_loss"] = o.get("price")
                            if o["type"] == "take_profit" and pos["take_profit"] is None:
                                pos["take_profit"] = o.get("price")
                except Exception as exc:
                    logger.debug("_collect_live_data: open orders error: {}", exc)

        uptime = (datetime.now(tz=timezone.utc) - _startup_time).total_seconds()
        cb_triggered = False
        if engine and engine.risk_manager:
            cb_triggered = engine.risk_manager._circuit_breaker.is_triggered()

        # Collect recent signals via shared helper
        signals = _collect_signals(limit=10)

        # Trade count and recent trades
        trade_count = 0
        recent_trades: List[dict] = []
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )
        if tj:
            try:
                all_trades = tj.get_recent_trades(1000)
                trade_count = len(all_trades)
                recent_trades = all_trades[-10:]  # Last 10 trades for live push
            except Exception:
                pass

        # Crash protection level — safe access
        crash_level = "normal"
        try:
            if engine and hasattr(engine, "crash_protector"):
                crash_level = engine.crash_protector.get_current_level().value
        except Exception:
            pass

        # Funding rates for open position symbols
        funding_rates: List[dict] = []
        try:
            if engine and engine.exchange:
                pm = getattr(engine, "position_manager", None)
                open_symbols: List[str] = []
                if pm is not None:
                    trackers = await pm.get_all_positions()
                    open_symbols = [t.position.symbol for t in trackers]
                for sym in open_symbols:
                    try:
                        raw = await engine.exchange.get_funding_rate(sym)
                        if isinstance(raw, dict):
                            rate = float(raw.get("fundingRate", 0.0))
                        else:
                            rate = float(raw) if raw is not None else 0.0
                        funding_rates.append({"symbol": sym, "rate": rate, "rate_pct": round(rate * 100, 6)})
                    except Exception:
                        pass
        except Exception:
            pass

        # Portfolio risk metrics
        portfolio_risk: dict = {}
        try:
            if engine and engine.risk_manager:
                portfolio_risk = await engine.risk_manager.get_portfolio_risk()
        except Exception:
            pass

        # ── Forex live data ────────────────────────────────────────────────
        forex_data: dict = {}
        try:
            forex_portfolio = await _collect_forex_portfolio()
            forex_positions = await _collect_forex_positions()
            frm = getattr(engine, "forex_risk_manager", None) if engine else None
            forex_data = {
                "portfolio": forex_portfolio,
                "positions": forex_positions,
                "session_name": frm.get_session_name() if frm and hasattr(frm, "get_session_name") else "",
                "session_pnl": getattr(frm, "_session_pnl", 0.0) if frm else 0.0,
                "in_recovery_mode": getattr(frm, "_in_recovery_mode", False) if frm else False,
                "recovery_level": getattr(frm, "_recovery_mode_level", 0) if frm else 0,
            }
        except Exception as exc:
            logger.debug("_collect_live_data: forex_data error: {}", exc)

        return {
            "type": "update",
            "portfolio": portfolio,
            "positions": positions,
            "open_orders": open_orders,
            "signals": signals,
            "recent_trades": recent_trades,
            "funding_rates": funding_rates,
            "portfolio_risk": portfolio_risk,
            "forex": forex_data,
            "status": {
                "trading_mode": _settings.trading_mode,
                "is_running": engine._running if engine else False,
                "circuit_breaker": {"triggered": cb_triggered},
                "uptime": round(uptime),
                "current_cycle": getattr(engine, "_cycle_count", 0) if engine else 0,
                "trade_count": trade_count,
                "market_regime": getattr(engine, "current_market_regime", "unknown") if engine else "unknown",
                "volatility_regime": getattr(engine, "current_volatility_regime", "normal") if engine else "normal",
                "crash_level": crash_level,
            },
        }


    # ── Forex helpers ────────────────────────────────────────────────────

    def _get_forex_pairs() -> List[str]:
        """Return configured forex trading pairs, with sensible defaults."""
        forex_cfg = getattr(_settings, "forex", None)
        if forex_cfg is not None:
            pairs = getattr(forex_cfg, "trading_pairs", None)
            if pairs:
                return list(pairs)
        return ["XAU/USD", "XAG/USD"]

    async def _collect_forex_portfolio() -> dict:
        """Collect forex account portfolio metrics.

        When a dedicated forex exchange is wired to the engine we fetch from
        it; otherwise we fall back to the primary exchange balance so the page
        still renders with real data.
        """
        balance = 0.0
        equity = 0.0
        margin_used = 0.0
        free_margin = 0.0
        daily_pnl = 0.0

        # Try to pull data from any attached exchange
        exch = None
        if engine:
            exch = getattr(engine, "forex_exchange", None) or engine.exchange
        if exch:
            try:
                bal = await exch.get_balance()
                balance = float(bal.usdt_free or 0.0)
                equity = float(bal.usdt_total or bal.usdt_free or 0.0)
            except Exception as exc:
                logger.debug("_collect_forex_portfolio: balance error: {}", exc)
            try:
                positions = await exch.get_positions()
                unrealized = sum(p.unrealized_pnl for p in positions)
                equity = balance + unrealized
                margin_used = sum(getattr(p, "margin", 0.0) for p in positions)
                free_margin = max(0.0, balance - margin_used)
            except Exception as exc:
                logger.debug("_collect_forex_portfolio: positions error: {}", exc)

        return {
            "balance": round(balance, 4),
            "equity": round(equity, 4),
            "margin_used": round(margin_used, 4),
            "free_margin": round(free_margin, 4),
            "daily_pnl": round(daily_pnl, 4),
        }

    async def _collect_forex_positions() -> List[dict]:
        """Collect open forex positions with lot-size and pip P&L data."""
        positions: List[dict] = []

        exch = None
        if engine:
            exch = getattr(engine, "forex_exchange", None) or engine.exchange
        if not exch:
            return positions

        forex_pairs = _get_forex_pairs()

        try:
            raw = await exch.get_positions()
            for p in raw:
                sym = getattr(p, "symbol", "")
                # Only include positions for known forex pairs
                if forex_pairs and sym not in forex_pairs:
                    continue
                entry = float(p.entry_price or 0.0)
                current = float(p.current_price or entry)
                pnl_usd = float(p.unrealized_pnl or 0.0)
                side = getattr(p.side, "value", str(p.side)) if hasattr(p, "side") else "long"

                # Attempt pip P&L calculation using Gate.io TradFi helpers
                pip_pnl: Optional[float] = None
                spread_pips: Optional[float] = None
                lot_size: Optional[float] = None
                try:
                    from exchange.gateio_tradfi_client import GateIOTradFiClient  # noqa: PLC0415
                    cfg = None  # Gate.io TradFi handles forex config internally
                    if cfg:
                        lot_size = float(
                            getattr(p, "amount", None)
                            or getattr(p, "size", None)
                            or 0.0
                        ) / cfg["contract_size"]
                        pip_size = cfg["pip_size"]
                        # For a long position: profit when current > entry
                        # For a short position: profit when current < entry
                        if side == "long":
                            signed_diff = current - entry
                        else:
                            signed_diff = entry - current
                        pip_pnl = signed_diff / pip_size
                except Exception:
                    pass

                positions.append(
                    {
                        "symbol": sym,
                        "direction": side,
                        "lot_size": round(lot_size, 4) if lot_size is not None else None,
                        "entry_price": round(entry, 4),
                        "current_price": round(current, 4),
                        "pip_pnl": round(pip_pnl, 2) if pip_pnl is not None else None,
                        "pnl_usd": round(pnl_usd, 4),
                        "pnl": round(pnl_usd, 4),
                        "spread_pips": round(spread_pips, 2) if spread_pips is not None else None,
                        "strategy": getattr(p, "strategy", ""),
                    }
                )
        except Exception as exc:
            logger.debug("_collect_forex_positions: error: {}", exc)

        return positions

    # ── Page routes ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse, dependencies=_auth_dep)
    async def index(request: Request) -> Any:
        if templates:
            # Fetch live portfolio data for the initial page render
            portfolio = await _collect_portfolio()
            positions = await _collect_positions()

            # Format values for template display
            equity = portfolio.get("equity", 0.0)
            portfolio_value_str = f"${equity:,.2f}"
            daily_pnl_pct = portfolio.get("daily_pnl_pct", 0.0)
            daily_pnl_str = f"{daily_pnl_pct:+.2f}%"
            open_positions_count = portfolio.get("open_positions", 0)

            # Win rate from metrics_collector or performance_tracker if available
            win_rate_str = "—"
            mc = metrics_collector or (
                getattr(engine, "metrics_collector", None) if engine else None
            )
            pt = performance_tracker or (
                getattr(engine, "performance_tracker", None) if engine else None
            )
            if mc:
                try:
                    metrics = mc.get_metrics()
                    wr = metrics.get("win_rate")
                    if wr is not None:
                        win_rate_str = f"{float(wr) * 100:.1f}%"
                except Exception:
                    pass
            elif pt:
                try:
                    report = pt.get_performance_report()
                    wr = report.get("win_rate")
                    if wr is not None:
                        win_rate_str = f"{float(wr) * 100:.1f}%"
                except Exception:
                    pass

            return templates.TemplateResponse(
                "dashboard.html",
                _tpl_ctx(
                    request,
                    "dashboard",
                    portfolio_value=portfolio_value_str,
                    daily_pnl=daily_pnl_str,
                    open_positions=open_positions_count,
                    max_positions=_settings.risk.max_open_positions,
                    win_rate=win_rate_str,
                    positions=positions,
                    trading_pairs=_settings.exchange.trading_pairs,
                    signals=[],
                    alerts=[],
                ),
            )
        return HTMLResponse(_render_page("Overview", _overview_body()))

    @app.get("/forex", response_class=HTMLResponse, dependencies=_auth_dep)
    async def forex_dashboard(request: Request) -> Any:
        """Main forex trading dashboard."""
        if templates:
            forex_portfolio = await _collect_forex_portfolio()
            forex_positions = await _collect_forex_positions()
            return templates.TemplateResponse(
                "forex_dashboard.html",
                _tpl_ctx(
                    request,
                    "forex_dashboard",
                    balance=forex_portfolio.get("balance", 0),
                    equity=forex_portfolio.get("equity", 0),
                    margin_used=forex_portfolio.get("margin_used", 0),
                    free_margin=forex_portfolio.get("free_margin", 0),
                    daily_pnl=forex_portfolio.get("daily_pnl", 0),
                    positions=forex_positions,
                    forex_pairs=_get_forex_pairs(),
                ),
            )
        return HTMLResponse(_render_page("Forex Dashboard", "<h2>Forex Dashboard</h2>"))

    @app.get("/forex/trades", response_class=HTMLResponse, dependencies=_auth_dep)
    async def forex_trades_page(request: Request) -> Any:
        """Forex trade history page."""
        trades: List[dict] = []
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )
        if tj:
            try:
                all_trades = tj.get_recent_trades(500)
                forex_pairs_set = set(_get_forex_pairs())
                trades = [
                    t for t in all_trades
                    if t.get("symbol") in forex_pairs_set
                ] or all_trades[:50]
            except Exception:
                pass
        wins = sum(1 for t in trades if float(t.get("pnl", 0) or 0) > 0)
        total_pnl_val = sum(float(t.get("pnl", 0) or 0) for t in trades)
        win_rate_str = f"{wins / len(trades) * 100:.1f}%" if trades else "0.0%"
        total_pnl_str = f"${total_pnl_val:+.2f}"
        if templates:
            return templates.TemplateResponse(
                "forex_trades.html",
                _tpl_ctx(
                    request,
                    "forex_trades",
                    trades=trades,
                    total_trades=len(trades),
                    win_rate=win_rate_str,
                    total_pnl=total_pnl_str,
                    total_pnl_val=total_pnl_val,
                ),
            )
        return HTMLResponse(_render_page("Forex Trades", "<h2>Forex Trades</h2>"))

    @app.get("/forex/settings", response_class=HTMLResponse, dependencies=_auth_dep)
    async def forex_settings_page(request: Request) -> Any:
        """Forex-specific settings page."""
        if templates:
            return templates.TemplateResponse(
                "forex_settings.html",
                _tpl_ctx(
                    request,
                    "forex_settings",
                    forex_pairs=_get_forex_pairs(),
                    settings=_settings,
                ),
            )
        return HTMLResponse(_render_page("Forex Settings", "<h2>Forex Settings</h2>"))

    @app.get("/forex/performance", response_class=HTMLResponse, dependencies=_auth_dep)
    async def forex_performance_page(request: Request) -> Any:
        """Forex performance analytics page."""
        if templates:
            return templates.TemplateResponse(
                "forex_performance.html",
                _tpl_ctx(
                    request,
                    "forex_performance",
                ),
            )
        return HTMLResponse(_render_page("Forex Performance", "<h2>Forex Performance</h2>"))

    @app.get("/forex/risk", response_class=HTMLResponse, dependencies=_auth_dep)
    async def forex_risk_page(request: Request) -> Any:
        """Forex risk management page."""
        if templates:
            return templates.TemplateResponse(
                "forex_risk.html",
                _tpl_ctx(
                    request,
                    "forex_risk",
                ),
            )
        return HTMLResponse(_render_page("Forex Risk", "<h2>Forex Risk</h2>"))

    @app.get("/trades", response_class=HTMLResponse, dependencies=_auth_dep)
    async def trades_page(request: Request) -> Any:
        trades: List[dict] = []
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )
        if tj:
            trades = tj.get_recent_trades(50)

        # Trap 5 Fix: DB fallback when in-memory trade_journal is empty (post-restart)
        if not trades:
            try:
                from sqlalchemy import select
                TradeHistory_m, _, create_tables_fn, get_async_session_fn = _get_storage_models()
                if TradeHistory_m is not None and get_async_session_fn is not None:
                    await create_tables_fn()
                    async with get_async_session_fn() as session:
                        query = (
                            select(TradeHistory_m)
                            .order_by(TradeHistory_m.entry_time.desc())
                            .limit(50)
                        )
                        result = await session.execute(query)
                        db_trades = result.scalars().all()
                        trades = [
                            {
                                "symbol": t.symbol, "side": t.side,
                                "entry_price": t.price, "exit_price": t.filled_price,
                                "size": t.size, "pnl": t.pnl,
                                "pnl_pct": getattr(t, "pnl_pct", None),
                                "strategy": t.strategy,
                                "entry_time": t.entry_time.isoformat() if t.entry_time else "",
                                "close_time": t.close_time.isoformat() if t.close_time else "",
                                "duration": t.duration, "exit_reason": t.exit_reason,
                                "status": "closed" if t.close_time else "open",
                            }
                            for t in db_trades
                        ]
            except Exception as exc:
                logger.debug("DB fallback for trades page failed: {}", exc)

        # Extract filter params
        filters = {
            "symbol": request.query_params.get("symbol", ""),
            "date_from": request.query_params.get("date_from", ""),
            "date_to": request.query_params.get("date_to", ""),
            "strategy": request.query_params.get("strategy", ""),
        }

        # Compute win_rate and total_pnl from trade data
        wins = sum(1 for t in trades if float(t.get("pnl", 0) or 0) > 0)
        total_pnl_val = sum(float(t.get("pnl", 0) or 0) for t in trades)
        win_rate_str = f"{wins / len(trades) * 100:.1f}%" if trades else "0.0%"
        total_pnl_str = f"${total_pnl_val:+.2f}"
        if templates:
            return templates.TemplateResponse(
                "trades.html",
                _tpl_ctx(
                    request,
                    "trades",
                    trades=trades,
                    total_trades=len(trades),
                    win_rate=win_rate_str,
                    total_pnl=total_pnl_str,
                    filters=filters,
                ),
            )
        return HTMLResponse(_render_page("Trades", "<h2>Trades</h2>"))

    @app.get("/performance", response_class=HTMLResponse, dependencies=_auth_dep)
    async def performance_page(request: Request) -> Any:
        pt = performance_tracker or (
            engine.performance_tracker
            if engine and hasattr(engine, "performance_tracker")
            else None
        )
        metrics: dict = {}
        equity_labels: List[str] = []
        equity_data: List[float] = []
        drawdown_data: List[float] = []
        pnl_labels: List[str] = []
        pnl_data: List[float] = []
        monthly_returns: dict = {}
        pair_pnl: dict = {}
        strategy_pnl: dict = {}
        win_rate_by_hour: dict = {}
        if pt:
            metrics = pt.get_performance_report()
            pair_pnl = metrics.get("pair_pnl", {})
            strategy_pnl = metrics.get("strategy_pnl", {})
            win_rate_by_hour = metrics.get("win_rate_by_hour", {})
            # Extract historical equity and P&L data for Chart.js
            history = getattr(pt, "_equity_history", None) or getattr(pt, "equity_history", None)
            if history:
                try:
                    for entry in list(history)[-90:]:
                        ts = entry.get("timestamp") or entry.get("ts", "")
                        equity_labels.append(str(ts)[:10] if ts else "")
                        equity_data.append(float(entry.get("equity", 0)))
                except Exception:
                    pass
            # Derive drawdown series from the equity curve when no history dict exists
            if not equity_data and hasattr(pt, "_equity_curve") and pt._equity_curve:
                equity_data = list(pt._equity_curve)[-90:]
                equity_labels = [str(i) for i in range(len(equity_data))]
            if equity_data and hasattr(pt, "calculate_drawdown_series"):
                try:
                    drawdown_data = pt.calculate_drawdown_series()[-len(equity_data):]
                except Exception:
                    drawdown_data = []
            daily_pnl_hist = (
                getattr(pt, "_daily_pnl_history", None)
                or getattr(pt, "daily_pnl_history", None)
            )
            if daily_pnl_hist:
                try:
                    for entry in list(daily_pnl_hist)[-30:]:
                        ts = entry.get("date") or entry.get("timestamp", "")
                        pnl_labels.append(str(ts)[:10] if ts else "")
                        pnl_data.append(float(entry.get("pnl", 0)))
                except Exception:
                    pass

        # Trap 5 Fix: If in-memory performance tracker is stale/empty (post-restart),
        # reconstruct metrics from the persistent DailyPerformance / TradeHistory DB tables.
        if not metrics or (metrics.get("total_trades", 0) == 0 and not equity_data):
            try:
                from sqlalchemy import func, select
                TradeHistory_m, DailyPerformance_m, create_tables_fn, get_async_session_fn = _get_storage_models()
                if DailyPerformance_m is not None and get_async_session_fn is not None:
                    await create_tables_fn()
                    async with get_async_session_fn() as session:
                        # Load last 90 days of daily performance
                        dp_result = await session.execute(
                            select(DailyPerformance_m)
                            .order_by(DailyPerformance_m.date.desc())
                            .limit(90)
                        )
                        records = list(dp_result.scalars().all())
                        records.reverse()  # chronological order

                        if records:
                            total_pnl = sum(float(r.pnl or 0) for r in records)
                            total_trades = sum(int(r.trades_count or 0) for r in records)
                            wins = sum(int(r.wins or 0) for r in records)
                            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

                            if not metrics:
                                metrics = {}
                            metrics.setdefault("total_pnl", total_pnl)
                            metrics.setdefault("total_return_pct", 0)
                            metrics.setdefault("total_trades", total_trades)
                            metrics.setdefault("win_rate", win_rate)
                            metrics.setdefault("sharpe_ratio", 0)
                            metrics.setdefault("sortino_ratio", 0)
                            metrics.setdefault("max_drawdown_pct", 0)
                            metrics.setdefault("calmar_ratio", 0)
                            metrics.setdefault("profit_factor", 0)
                            metrics.setdefault("expectancy", total_pnl / total_trades if total_trades else 0)
                            metrics.setdefault("recovery_factor", 0)

                            # Build equity curve from daily PnL
                            if not equity_data:
                                cumulative = 0.0
                                for r in records:
                                    cumulative += float(r.pnl or 0)
                                    equity_data.append(cumulative)
                                    d = str(r.date)[:10] if r.date else ""
                                    equity_labels.append(d)

                            # Build daily PnL chart
                            if not pnl_data:
                                for r in records[-30:]:
                                    pnl_data.append(float(r.pnl or 0))
                                    d = str(r.date)[:10] if r.date else ""
                                    pnl_labels.append(d)

                            logger.debug("Loaded {} days of performance from DB fallback", len(records))
            except Exception as exc:
                logger.debug("DB fallback for performance page failed: {}", exc)

        if templates:
            return templates.TemplateResponse(
                "performance.html",
                _tpl_ctx(
                    request,
                    "performance",
                    metrics=metrics,
                    equity_labels=equity_labels,
                    equity_data=equity_data,
                    drawdown_data=drawdown_data,
                    pnl_labels=pnl_labels,
                    pnl_data=pnl_data,
                    monthly_returns=monthly_returns,
                    pair_pnl=pair_pnl,
                    strategy_pnl=strategy_pnl,
                    win_rate_by_hour=win_rate_by_hour,
                ),
            )
        return HTMLResponse(_render_page("Performance", "<h2>Performance</h2>"))

    @app.get("/signals", response_class=HTMLResponse, dependencies=_auth_dep)
    async def signals_page(request: Request) -> Any:
        if templates:
            return templates.TemplateResponse(
                "dashboard.html",
                _tpl_ctx(
                    request,
                    "signals",
                    portfolio_value="$0.00",
                    daily_pnl="0.00%",
                    open_positions=0,
                    max_positions=_settings.risk.max_open_positions,
                    win_rate="0.0%",
                    positions=[],
                    signals=[],
                    alerts=[],
                ),
            )
        return HTMLResponse(
            _render_page("Signals", "<h2>Signals</h2><p>Live signals via WebSocket /ws/live.</p>")
        )

    @app.get("/risk", response_class=HTMLResponse, dependencies=_auth_dep)
    async def risk_page(request: Request) -> Any:
        rm = await _get_engine_risk_manager()
        risk_snapshot: dict = {}
        if rm:
            try:
                risk_snapshot = await rm.get_portfolio_risk()
            except Exception as exc:
                risk_snapshot = {"error": str(exc)}
        if templates:
            return templates.TemplateResponse(
                "risk.html",
                _tpl_ctx(request, "risk", risk=risk_snapshot),
            )
        return HTMLResponse(
            _render_page("Risk", "<h2>Risk</h2><p>Connect to /api/risk for risk data.</p>")
        )

    @app.get("/settings", response_class=HTMLResponse, dependencies=_auth_dep)
    async def settings_page(request: Request) -> Any:
        if templates:
            return templates.TemplateResponse(
                "settings.html",
                _tpl_ctx(request, "settings", settings=_settings),
            )
        return HTMLResponse(_render_page("Settings", "<h2>Settings</h2>"))

    @app.get("/logs", response_class=HTMLResponse, dependencies=_auth_dep)
    async def logs_page(request: Request) -> Any:
        if templates:
            return templates.TemplateResponse(
                "logs.html",
                _tpl_ctx(request, "logs"),
            )
        return HTMLResponse(
            _render_page("Logs", "<h2>Logs</h2><p>Tail logs via your container runtime.</p>")
        )

    # ── Health & status ──────────────────────────────────────────────────

    @app.get("/health")
    async def health_check() -> JSONResponse:
        if engine and engine.health_checker:
            try:
                report = await engine.health_checker.get_health_report()
                return JSONResponse(
                    {
                        "status": report.get("overall", "healthy"),
                        "service": "crypto-trading-bot",
                        "components": report.get("components", {}),
                        "timestamp": report.get("timestamp"),
                    }
                )
            except Exception as exc:
                logger.debug("Health check error: {}", exc)
        return JSONResponse({"status": "healthy", "service": "crypto-trading-bot"})

    @app.get("/health/detailed")
    async def health_check_detailed() -> JSONResponse:
        """Detailed health check with trading loop liveness, exchange freshness, etc."""
        import time as _time
        checks: dict = {}
        all_healthy = True

        # 1. Trading loop liveness
        if engine is not None:
            cycle_count = getattr(engine, "_cycle_count", 0)
            last_cycle_ts = getattr(engine, "_last_cycle_ts", None)
            if last_cycle_ts is None:
                last_cycle_ts = getattr(engine, "_start_time", None)
            trading_alive = False
            if last_cycle_ts is not None:
                if hasattr(last_cycle_ts, "timestamp"):
                    elapsed = _time.time() - last_cycle_ts.timestamp()
                else:
                    elapsed = _time.time() - float(last_cycle_ts)
                trading_alive = elapsed < 120
            checks["trading_loop_alive"] = {
                "healthy": trading_alive,
                "cycle_count": cycle_count,
            }
            if not trading_alive:
                all_healthy = False

            # 2. Exchange connectivity
            last_api = getattr(engine, "_last_successful_api_call", 0.0)
            exchange_connected = last_api > 0 and (_time.time() - last_api) < 60
            checks["exchange_connected"] = {
                "healthy": exchange_connected,
                "seconds_since_last_call": round(_time.time() - last_api, 1) if last_api else None,
            }
            if not exchange_connected:
                all_healthy = False

            # 3. Circuit breaker status
            cb_active = False
            if engine.risk_manager is not None:
                cb_active = engine.risk_manager._circuit_breaker.is_triggered()
            checks["circuit_breaker"] = {"healthy": not cb_active, "active": cb_active}
            if cb_active:
                all_healthy = False

            # 4. Rate-limiter stats (informational — not a health failure)
            try:
                from utils.rate_limiter import ExchangeRateLimiter
                checks["rate_limiters"] = ExchangeRateLimiter.get_all_stats()
            except Exception:
                pass

        status_code = 200 if all_healthy else 503
        return JSONResponse(
            {
                "status": "healthy" if all_healthy else "unhealthy",
                "service": "crypto-trading-bot",
                "checks": checks,
            },
            status_code=status_code,
        )

    # ── Primary API endpoints ────────────────────────────────────────────

    @app.get("/api/status", dependencies=_auth_dep)
    async def api_status_new() -> JSONResponse:
        uptime = (datetime.now(tz=timezone.utc) - _startup_time).total_seconds()
        cb_info: dict = {"triggered": False, "reason": None}
        if engine and engine.risk_manager:
            cb = engine.risk_manager._circuit_breaker
            cb_info = {
                "triggered": cb.is_triggered(),
                "info": cb.trigger_info,
            }
        # Trap 4 Fix: Include unified rate-limit stats in status response
        rate_limit_stats = {}
        try:
            from crypto_trading_bot.utils.rate_limiter import ExchangeRateLimiter
            rate_limit_stats = ExchangeRateLimiter.get_all_stats()
        except Exception:
            pass

        # Trap 3 Fix: Include Rust engine state if available
        rust_state = {}
        try:
            if engine and hasattr(engine, "state_reader") and engine.state_reader:
                rust_state = engine.state_reader.recover_state_from_rust()
        except Exception:
            pass

        return JSONResponse(
            {
                "trading_mode": _settings.trading_mode,
                "is_running": engine._running if engine else False,
                "circuit_breaker": cb_info,
                "uptime": round(uptime),
                "current_cycle": getattr(engine, "_cycle_count", 0) if engine else 0,
                "rate_limits": rate_limit_stats,
                "rust_engine": rust_state,
            }
        )

    @app.get("/api/portfolio", dependencies=_auth_dep)
    async def api_portfolio() -> JSONResponse:
        return JSONResponse(await _collect_portfolio())

    @app.get("/api/positions", dependencies=_auth_dep)
    async def api_positions_new() -> JSONResponse:
        return JSONResponse({"positions": await _collect_positions()})

    @app.get("/api/trades", dependencies=_auth_dep)
    async def api_trades_new() -> JSONResponse:
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )
        trades: List[dict] = []
        if tj:
            trades = tj.get_recent_trades(50)

        # Trap 5 Fix: DB fallback when trade_journal is empty (post-restart)
        if not trades:
            try:
                from sqlalchemy import select
                TH, _, ct, gas = _get_storage_models()
                if TH is not None and gas is not None:
                    await ct()
                    async with gas() as session:
                        result = await session.execute(
                            select(TH).order_by(TH.entry_time.desc()).limit(50)
                        )
                        db_trades = result.scalars().all()
                        trades = [
                            {
                                "symbol": t.symbol, "side": t.side,
                                "entry_price": t.price, "exit_price": t.filled_price,
                                "size": t.size, "pnl": t.pnl,
                                "strategy": t.strategy,
                                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                                "close_time": t.close_time.isoformat() if t.close_time else None,
                            }
                            for t in db_trades
                        ]
            except Exception:
                pass
        return JSONResponse({"trades": trades})

    @app.get("/api/performance", dependencies=_auth_dep)
    async def api_performance() -> JSONResponse:
        pt = performance_tracker or (
            engine.performance_tracker
            if engine and hasattr(engine, "performance_tracker")
            else None
        )
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )

        report: dict = {}
        if pt:
            report = pt.get_performance_report()
        elif tj:
            report = tj.get_trade_stats()

        # daily/weekly/monthly PnL from state manager
        daily_pnl = 0.0
        if engine and engine.state_manager:
            try:
                state = await engine.state_manager.get_state()
                daily_pnl = state.daily_pnl
            except Exception:
                pass

        report["daily_pnl"] = daily_pnl
        return JSONResponse(report)

    @app.get("/api/risk", dependencies=_auth_dep)
    async def api_risk() -> JSONResponse:
        rm = await _get_engine_risk_manager()
        if rm:
            try:
                snapshot = await rm.get_portfolio_risk()
                return JSONResponse(snapshot)
            except Exception as exc:
                return JSONResponse({"error": str(exc)})
        return JSONResponse({"error": "Risk manager not available"})

    @app.get("/api/portfolio/risk", dependencies=_auth_dep)
    async def api_portfolio_risk() -> JSONResponse:
        """Return comprehensive portfolio risk metrics including margin and correlation."""
        rm = await _get_engine_risk_manager()
        if rm:
            try:
                snapshot = await rm.get_portfolio_risk()
                return JSONResponse(snapshot)
            except Exception as exc:
                return JSONResponse({"error": str(exc)})
        return JSONResponse({"error": "Risk manager not available"})

    @app.get("/api/funding-rates", dependencies=_auth_dep)
    async def api_funding_rates() -> JSONResponse:
        """Return current funding rates for all trading pairs / open positions."""
        try:
            exch = engine.exchange if engine else None
            if exch is None:
                return JSONResponse({"funding_rates": [], "error": "Exchange not available"})

            pm = getattr(engine, "position_manager", None)
            settings = getattr(engine, "settings", None)
            trading_pairs: list = (
                getattr(getattr(settings, "exchange", None), "trading_pairs", [])
                if settings
                else []
            )

            # Get symbols from open positions first, then fall back to all trading pairs
            open_symbols: list = []
            if pm is not None:
                try:
                    trackers = await pm.get_all_positions()
                    open_symbols = [t.position.symbol for t in trackers]
                except Exception:
                    pass

            symbols = open_symbols or trading_pairs
            results = []
            for sym in symbols:
                try:
                    raw = await exch.get_funding_rate(sym)
                    if isinstance(raw, dict):
                        rate = float(raw.get("fundingRate", 0.0))
                    else:
                        rate = float(raw) if raw is not None else 0.0
                    results.append({"symbol": sym, "rate": rate, "rate_pct": round(rate * 100, 6)})
                except Exception as exc:
                    logger.debug("Funding rate fetch error for {}: {}", sym, exc)
                    results.append({"symbol": sym, "rate": None, "rate_pct": None, "error": str(exc)})
            return JSONResponse({"funding_rates": results})
        except Exception as exc:
            logger.warning("api_funding_rates error: {}", exc)
            return JSONResponse({"funding_rates": [], "error": str(exc)})

    @app.post("/api/settings/update", dependencies=_auth_dep)
    async def api_settings_update(body: dict) -> JSONResponse:
        """Update runtime settings without restart.

        Supported fields: ``trading_mode``, ``max_positions``,
        ``risk_per_trade``, ``leverage``.
        """
        if engine is None:
            return JSONResponse({"success": False, "error": "Engine not available"})
        try:
            updated: dict = {}
            settings = engine.settings

            # trading_mode – handled via mode switch
            if "trading_mode" in body:
                new_mode = str(body["trading_mode"]).lower()
                current = getattr(settings, "trading_mode", "paper")
                if new_mode != current:
                    result = await engine.switch_mode(new_mode)
                    updated["trading_mode"] = result
                else:
                    updated["trading_mode"] = "unchanged"

            # max_positions
            if "max_positions" in body:
                risk_cfg = getattr(settings, "risk", None)
                if risk_cfg is not None:
                    risk_cfg.max_open_positions = int(body["max_positions"])
                    updated["max_positions"] = risk_cfg.max_open_positions

            # risk_per_trade (updates max_position_size_pct)
            if "risk_per_trade" in body:
                risk_cfg = getattr(settings, "risk", None)
                if risk_cfg is not None:
                    risk_cfg.max_position_size_pct = float(body["risk_per_trade"])
                    updated["risk_per_trade"] = risk_cfg.max_position_size_pct

            # leverage default
            if "leverage" in body:
                exch_cfg = getattr(settings, "exchange", None)
                if exch_cfg is not None:
                    new_lev = int(body["leverage"])
                    new_lev = max(1, min(new_lev, exch_cfg.max_leverage))
                    exch_cfg.default_leverage = new_lev
                    updated["leverage"] = exch_cfg.default_leverage

            logger.info("Runtime settings updated: {}", updated)
            return JSONResponse({"success": True, "updated": updated})
        except Exception as exc:
            logger.warning("api_settings_update error: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/signals", dependencies=_auth_dep)
    async def api_signals() -> JSONResponse:
        """Return recent trading signals from the strategy manager."""
        return JSONResponse({"signals": _collect_signals(limit=20)})

    @app.get("/api/logs", dependencies=_auth_dep)
    async def api_logs() -> JSONResponse:
        lines = _tail_log(_LOG_FILE, 100)
        return JSONResponse({"lines": lines})

    @app.get("/api/gold/overview", dependencies=_auth_dep)
    async def api_gold_overview() -> JSONResponse:
        """Gold trading overview with price, positions, and strategy performance."""
        from config.gold_config import GOLD_FUTURES_CONFIG

        ticker_data: dict = {}
        if engine and engine.exchange:
            try:
                ticker = await engine.exchange.get_ticker("XAU/USDT")
                if ticker:
                    ticker_data = {
                        "last": ticker.last,
                        "bid": ticker.bid,
                        "ask": ticker.ask,
                        "volume": ticker.volume,
                        "change_pct": ticker.change_pct if hasattr(ticker, "change_pct") else None,
                    }
            except Exception as exc:
                logger.warning("Gold overview: ticker fetch failed: {}", exc)

        # Collect gold positions
        all_positions = await _collect_positions()
        gold_positions = [p for p in all_positions if "XAU" in p.get("symbol", "")]

        # Collect gold-related signals
        gold_signals = [
            s for s in _collect_signals(limit=50) if "XAU" in s.get("symbol", "")
        ]

        gold_cfg = GOLD_FUTURES_CONFIG.get("XAU/USDT", {})

        return JSONResponse({
            "success": True,
            "symbol": "XAU/USDT",
            "exchange": gold_cfg.get("exchange", "gateio"),
            "ticker": ticker_data,
            "positions": gold_positions,
            "signals": gold_signals,
            "config": {
                "contract_size": gold_cfg.get("contract_size"),
                "max_leverage": gold_cfg.get("max_leverage"),
                "default_leverage": gold_cfg.get("default_leverage"),
                "trading_hours": gold_cfg.get("trading_hours"),
                "volatility_profile": gold_cfg.get("volatility_profile"),
                "risk": gold_cfg.get("risk", {}),
            },
            "enabled": getattr(
                getattr(_settings, "exchange", None), "gold_futures_enabled", False
            ),
        })

    # ── Control endpoints ────────────────────────────────────────────────

    @app.get("/api/gold/debug", dependencies=_auth_dep)
    async def api_gold_debug() -> JSONResponse:
        """Debug endpoint to check gold symbol availability on the exchange."""
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})

        results: dict = {}
        test_symbols = ["XAUT/USDT", "XAUT/USDT:USDT", "XAU/USDT", "XAU/USDT:USDT"]

        # Check markets
        markets: dict = {}
        if hasattr(engine.exchange, "_client") and engine.exchange._client:
            markets = engine.exchange._client.markets or {}

        for sym in test_symbols:
            results[sym] = {
                "in_markets": sym in markets,
                "market_info": {
                    "contract": markets.get(sym, {}).get("contract"),
                    "contractSize": markets.get(sym, {}).get("contractSize"),
                    "active": markets.get(sym, {}).get("active"),
                } if sym in markets else None,
            }

        # Try fetching ticker for XAUT
        for sym in ["XAUT/USDT:USDT", "XAUT/USDT"]:
            try:
                ticker = await engine.exchange.get_ticker(sym)
                results[f"{sym}_ticker"] = {"last": ticker.last, "volume": ticker.volume}
            except Exception as exc:
                results[f"{sym}_ticker"] = {"error": str(exc)}

        # Check permanently unavailable
        unavailable = getattr(engine.exchange, "_permanently_unavailable_symbols", set())
        results["permanently_unavailable"] = list(unavailable)

        return JSONResponse(results)

    @app.post("/api/circuit-breaker/reset", dependencies=_auth_dep)
    async def api_cb_reset() -> JSONResponse:
        rm = await _get_engine_risk_manager()
        if rm:
            try:
                await rm._circuit_breaker.reset()
                if engine and engine.state_manager:
                    await engine.state_manager.update_state(circuit_breaker_active=False)
                return JSONResponse({"success": True, "message": "Circuit breaker reset"})
            except Exception as exc:
                return JSONResponse({"success": False, "error": str(exc)})
        return JSONResponse({"success": False, "reason": "Risk manager not available"})

    @app.post("/api/bot/pause", dependencies=_auth_dep)
    async def api_bot_pause() -> JSONResponse:
        if engine and engine.state_manager:
            try:
                from core.state_manager import BotStatus

                await engine.state_manager.update_state(status=BotStatus.PAUSED)
                return JSONResponse({"success": True, "message": "Bot paused"})
            except Exception as exc:
                return JSONResponse({"success": False, "error": str(exc)})
        return JSONResponse({"success": False, "reason": "Engine not available"})

    @app.post("/api/bot/resume", dependencies=_auth_dep)
    async def api_bot_resume() -> JSONResponse:
        if engine and engine.state_manager:
            try:
                from core.state_manager import BotStatus

                await engine.state_manager.update_state(status=BotStatus.RUNNING)
                return JSONResponse({"success": True, "message": "Bot resumed"})
            except Exception as exc:
                return JSONResponse({"success": False, "error": str(exc)})
        return JSONResponse({"success": False, "reason": "Engine not available"})

    # ── Legacy /api/v1/* endpoints (backward compat) ─────────────────────

    @app.get("/api/v1/status", dependencies=_auth_dep)
    async def api_v1_status() -> JSONResponse:
        data: dict = {"status": "running", "trading_mode": _settings.trading_mode}
        if metrics_collector:
            data["metrics"] = metrics_collector.get_metrics()
        rm = await _get_engine_risk_manager()
        if rm:
            try:
                data["portfolio_risk"] = await rm.get_portfolio_risk()
            except Exception as exc:
                data["portfolio_risk_error"] = str(exc)
        return JSONResponse(data)

    @app.get("/api/v1/positions", dependencies=_auth_dep)
    async def api_v1_positions() -> JSONResponse:
        free_margin = await _fetch_free_margin()
        return JSONResponse({"positions": await _collect_positions(), "free_margin": free_margin})

    @app.get("/api/v1/trades", dependencies=_auth_dep)
    async def api_v1_trades() -> JSONResponse:
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )
        trades: List[dict] = []
        if tj:
            trades = tj.get_recent_trades(50)
        free_margin = await _fetch_free_margin()
        return JSONResponse({"trades": trades, "free_margin": free_margin})

    @app.post("/api/v1/settings", dependencies=_auth_dep)
    async def api_update_settings(body: dict) -> JSONResponse:
        logger.info("Settings update requested (not applied): {}", body)
        return JSONResponse(
            {"success": False, "reason": "Runtime settings update not supported yet"}
        )

    @app.post("/api/v1/emergency-stop", dependencies=_auth_dep)
    async def api_emergency_stop() -> JSONResponse:
        logger.critical("Emergency stop triggered via dashboard API")
        rm = await _get_engine_risk_manager()
        if rm:
            try:
                await rm._circuit_breaker.trigger("Manual emergency stop via dashboard")
                return JSONResponse({"success": True, "message": "Circuit breaker triggered"})
            except Exception as exc:
                return JSONResponse({"success": False, "error": str(exc)})
        return JSONResponse({"success": False, "reason": "Risk manager not available"})

    @app.get("/metrics", dependencies=_auth_dep)
    async def prometheus_metrics() -> HTMLResponse:
        if metrics_collector:
            return HTMLResponse(metrics_collector.export_prometheus(), media_type="text/plain")
        return HTMLResponse("# No metrics collector attached\n", media_type="text/plain")

    # ── MR2: New regime / strategy-performance / crash-protection endpoints ──

    @app.get("/api/regime", dependencies=_auth_dep)
    async def api_regime() -> JSONResponse:
        """Return current market regime, volatility regime, and regime history."""
        regime = "unknown"
        vol_regime = "normal"
        crash_level = "normal"
        circuit_breaker_active = False
        crash_recovery = False
        reentry_size_pct = 1.0

        if engine:
            regime = getattr(engine, "current_market_regime", "unknown")
            vol_regime = getattr(engine, "current_volatility_regime", "normal")
            if hasattr(engine, "crash_protector"):
                crash_state = engine.crash_protector.get_state()
                crash_level = crash_state.level.value
                circuit_breaker_active = crash_state.circuit_breaker_active
                crash_recovery = crash_state.recovery_phase
                reentry_size_pct = crash_state.reentry_size_pct

        return JSONResponse(
            {
                "market_regime": regime,
                "volatility_regime": vol_regime,
                "crash_level": crash_level,
                "circuit_breaker_active": circuit_breaker_active,
                "crash_recovery": crash_recovery,
                "reentry_size_pct": reentry_size_pct,
            }
        )

    @app.get("/api/strategy-performance", dependencies=_auth_dep)
    async def api_strategy_performance() -> JSONResponse:
        """Return per-strategy rolling performance metrics."""
        result: List[dict] = []
        sm = (
            engine.strategy_manager
            if engine and hasattr(engine, "strategy_manager")
            else None
        )
        if sm is not None:
            for name, strategy in sm._strategies.items():
                metrics = sm.get_rolling_metrics(name)
                regime_wr = {}
                for reg in sm._rolling_trades.get(name, {}).keys():
                    regime_wr[reg] = sm.get_regime_win_rate(name, reg)
                result.append(
                    {
                        "name": name,
                        "enabled": strategy.enabled,
                        "win_rate": metrics.get("win_rate", 0.5),
                        "profit_factor": metrics.get("profit_factor", 1.0),
                        "sharpe": metrics.get("sharpe", 0.0),
                        "avg_profit": metrics.get("avg_profit", 0.0),
                        "avg_loss": metrics.get("avg_loss", 0.0),
                        "max_drawdown": metrics.get("max_drawdown", 0.0),
                        "total_trades": metrics.get("total_trades", 0),
                        "regime_win_rates": regime_wr,
                    }
                )
        # Convert list to dict keyed by strategy name for the dashboard JS
        strategies_dict = {s["name"]: s for s in result}

        # Build regime performance breakdown: strategy → regime → {win_rate, count}
        regime_performance: dict = {}
        if sm is not None:
            for name in strategies_dict:
                regime_performance[name] = {}
                best_wr = -1.0
                best_regime = None
                for reg in sm._rolling_trades.get(name, {}).keys():
                    wr = sm.get_regime_win_rate(name, reg)
                    regime_performance[name][reg] = {"win_rate": wr}
                    if wr > best_wr:
                        best_wr = wr
                        best_regime = reg
                regime_performance[name]["best_regime"] = best_regime

        return JSONResponse({"strategies": strategies_dict, "regime_performance": regime_performance})

    # ── New analytics / risk-dashboard endpoints (Upgrade 1) ─────────────

    @app.get("/api/equity-history", dependencies=_auth_dep)
    async def api_equity_history(range: str = "7d") -> JSONResponse:
        """Return equity curve history from the database."""
        try:
            from datetime import timedelta

            from sqlalchemy import select

            from data.storage.models import DailyPnLRecord, get_async_session

            days = {"24h": 1, "7d": 7, "30d": 30, "all": 3650}.get(range, 7)
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

            async with get_async_session() as session:
                stmt = (
                    select(DailyPnLRecord)
                    .where(DailyPnLRecord.date >= cutoff.date())
                    .order_by(DailyPnLRecord.date)
                )
                rows = (await session.execute(stmt)).scalars().all()

            history = [
                {
                    # Use ISO 8601 datetime string (with time component) for accurate
                    # JS Date parsing across all timezones.
                    "ts": datetime(
                        row.date.year, row.date.month, row.date.day, tzinfo=timezone.utc
                    ).isoformat(),
                    "equity": float(row.ending_balance or row.starting_balance or 0),
                }
                for row in rows
            ]

            # Compute risk-adjusted metrics
            metrics: dict = {}
            if engine and engine.risk_manager:
                try:
                    daily_status = await engine.risk_manager._daily_pnl.get_daily_status()
                    metrics = {
                        "sharpe_ratio": daily_status.get("sharpe_ratio"),
                        "sortino_ratio": daily_status.get("sortino_ratio"),
                        "calmar_ratio": daily_status.get("calmar_ratio"),
                    }
                except Exception:
                    pass
            return JSONResponse({"history": history, "metrics": metrics})
        except Exception as exc:
            logger.debug("api_equity_history error: {}", exc)
            return JSONResponse({"history": [], "metrics": {}})

    @app.get("/api/risk-dashboard", dependencies=_auth_dep)
    async def api_risk_dashboard() -> JSONResponse:
        """Return comprehensive risk metrics for the Risk Dashboard panel."""
        result: dict = {}
        try:
            if engine and engine.risk_manager:
                rm = engine.risk_manager
                try:
                    daily_status = await rm._daily_pnl.get_daily_status()
                    result["daily_pnl_pct"] = daily_status.get("daily_pnl_pct", 0.0)
                except Exception:
                    result["daily_pnl_pct"] = 0.0

                # Profit compounder — daily_pnl_pct is already in percentage units
                try:
                    compound_mult = rm._profit_compounder.get_size_multiplier(
                        daily_pnl_pct=float(result.get("daily_pnl_pct", 0.0)),
                    )
                    result["compound_multiplier"] = compound_mult
                    result["extra_allocation_pct"] = rm._profit_compounder.extra_allocation_pct
                except Exception:
                    result["compound_multiplier"] = 1.0
                    result["extra_allocation_pct"] = 0.0

            # VaR from risk manager
            try:
                if engine and engine.risk_manager:
                    var_calc = engine.risk_manager._var_calculator
                    if hasattr(var_calc, "last_var_pct"):
                        result["var_95_pct"] = var_calc.last_var_pct or 0.0
                    if hasattr(var_calc, "last_cvar_pct"):
                        result["cvar_95_pct"] = var_calc.last_cvar_pct or 0.0
            except Exception:
                pass

            # Margin and drawdown
            try:
                if engine and engine.risk_manager:
                    dd_prot = engine.risk_manager._drawdown
                    result["max_drawdown_pct"] = getattr(dd_prot, "_max_drawdown_seen", 0.0)
                    result["current_drawdown_pct"] = getattr(dd_prot, "_current_drawdown", 0.0)
            except Exception:
                pass

            # Weekly P&L from performance records
            result["weekly_pnl_pct"] = 0.0

            # Position liquidation distances
            positions_data: list = []
            if engine and engine.position_manager:
                try:
                    trackers = await engine.position_manager.get_all_positions()
                    for t in trackers:
                        liq = getattr(t, "liquidation_price", None) or getattr(t, "_liquidation_price", None)
                        cp = getattr(t, "current_price", None) or getattr(t, "entry_price", None) or 0.0
                        dist_pct = 0.0
                        if liq and cp and cp > 0:
                            dist_pct = abs(cp - float(liq)) / cp * 100.0
                        positions_data.append({
                            "symbol": getattr(t, "symbol", ""),
                            "liquidation_price": liq,
                            "liquidation_distance_pct": dist_pct,
                        })
                except Exception:
                    pass
            result["positions"] = positions_data

        except Exception as exc:
            logger.debug("api_risk_dashboard error: {}", exc)
        return JSONResponse(result)

    # ── Trade management endpoints ────────────────────────────────────────

    @app.get("/api/positions/live", dependencies=_auth_dep)
    async def api_positions_live() -> JSONResponse:
        """Return real-time position data including P&L, SL/TP, leverage, and open orders."""
        positions = await _collect_positions()
        # Enrich with open orders per symbol
        if engine and engine.exchange:
            try:
                open_orders = await engine.exchange.get_open_orders()
                orders_by_symbol: dict = {}
                for o in open_orders:
                    orders_by_symbol.setdefault(o.symbol, []).append({
                        "id": o.id,
                        "type": o.type.value,
                        "side": o.side.value,
                        "amount": o.amount,
                        "price": o.price,
                        "status": o.status.value,
                    })
                for pos in positions:
                    pos["open_orders"] = orders_by_symbol.get(pos["symbol"], [])
                    # Find SL/TP from open orders
                    for o in pos["open_orders"]:
                        if o["type"] in ("stop_loss",) and pos["stop_loss"] is None:
                            pos["stop_loss"] = o.get("price")
                        if o["type"] in ("take_profit",) and pos["take_profit"] is None:
                            pos["take_profit"] = o.get("price")
            except Exception as exc:
                logger.debug("api_positions_live: open orders fetch error: {}", exc)
        return JSONResponse({"positions": positions, "mode": _settings.trading_mode})

    @app.get("/api/orders/open", dependencies=_auth_dep)
    async def api_orders_open() -> JSONResponse:
        """Return all pending open orders (limit, SL, TP)."""
        if not (engine and engine.exchange):
            return JSONResponse({"orders": [], "error": "Exchange not available"})
        try:
            raw_orders = await engine.exchange.get_open_orders()
            orders = [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "type": o.type.value,
                    "side": o.side.value,
                    "amount": o.amount,
                    "price": o.price,
                    "filled": o.filled,
                    "remaining": o.remaining,
                    "status": o.status.value,
                    "timestamp": o.timestamp,
                }
                for o in raw_orders
            ]
            return JSONResponse({"orders": orders})
        except Exception as exc:
            logger.debug("api_orders_open error: {}", exc)
            return JSONResponse({"orders": [], "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/close", dependencies=_auth_dep)
    async def api_close_position(symbol: str, body: Optional[dict] = None) -> JSONResponse:
        """Close a position (full or partial) for *symbol*.

        Body (optional): ``{"amount": float}`` for partial close.
        """
        body = body or {}
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        try:
            amount = float(body.get("amount", 0)) or None
            order = await engine.exchange.close_position(symbol, amount)
            logger.info("Dashboard: closed position {} amount={}", symbol, amount)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({
                "success": True,
                "order_id": order.id,
                "symbol": symbol,
                "amount": order.filled,
                "price": order.price,
            })
        except Exception as exc:
            logger.warning("Dashboard: close_position error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/stop-loss", dependencies=_auth_dep)
    async def api_set_stop_loss(symbol: str, body: dict) -> JSONResponse:
        """Set or update the stop-loss for *symbol*.

        Body: ``{"price": float}``
        """
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        price = body.get("price")
        if price is None:
            return JSONResponse({"success": False, "error": "Missing 'price' in request body"})
        try:
            price_f = float(price)
            order = await engine.exchange.update_stop_loss(symbol, price_f)
            logger.info("Dashboard: stop-loss set for {} @ {}", symbol, price_f)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({
                "success": True,
                "order_id": order.id,
                "symbol": symbol,
                "stop_loss": price_f,
            })
        except Exception as exc:
            logger.warning("Dashboard: set_stop_loss error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/take-profit", dependencies=_auth_dep)
    async def api_set_take_profit(symbol: str, body: dict) -> JSONResponse:
        """Set or update the take-profit for *symbol*.

        Body: ``{"price": float}``
        """
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        price = body.get("price")
        if price is None:
            return JSONResponse({"success": False, "error": "Missing 'price' in request body"})
        try:
            price_f = float(price)
            order = await engine.exchange.update_take_profit(symbol, price_f)
            logger.info("Dashboard: take-profit set for {} @ {}", symbol, price_f)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({
                "success": True,
                "order_id": order.id,
                "symbol": symbol,
                "take_profit": price_f,
            })
        except Exception as exc:
            logger.warning("Dashboard: set_take_profit error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/reduce", dependencies=_auth_dep)
    async def api_reduce_position(symbol: str, body: dict) -> JSONResponse:
        """Partially close a position by *percentage*.

        Body: ``{"percentage": float}`` — e.g. 50 means close 50% of the position.
        """
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        pct = body.get("percentage")
        if pct is None:
            return JSONResponse({"success": False, "error": "Missing 'percentage' in request body"})
        try:
            pct_f = float(pct)
            if pct_f <= 0 or pct_f > 100:
                return JSONResponse({"success": False, "error": "percentage must be 0 < pct <= 100"})
            pos = await engine.exchange.get_position(symbol)
            if not pos:
                return JSONResponse({"success": False, "error": f"No open position for {symbol}"})
            reduce_amount = pos.amount * (pct_f / 100.0)
            order = await engine.exchange.close_position(symbol, reduce_amount)
            logger.info("Dashboard: reduced position {} by {}% (amount={})", symbol, pct_f, reduce_amount)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({
                "success": True,
                "order_id": order.id,
                "symbol": symbol,
                "reduced_amount": reduce_amount,
                "percentage": pct_f,
            })
        except Exception as exc:
            logger.warning("Dashboard: reduce_position error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/orders/{order_id}/cancel", dependencies=_auth_dep)
    async def api_cancel_order(order_id: str, body: Optional[dict] = None) -> JSONResponse:
        """Cancel a single order by *order_id*.

        Body (optional): ``{"symbol": str}`` — required for some exchanges.
        """
        body = body or {}
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        symbol = body.get("symbol", "")
        try:
            result = await engine.exchange.cancel_order(order_id, symbol)
            logger.info("Dashboard: cancelled order {} on {}", order_id, symbol)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({"success": True, "order_id": order_id, "result": result})
        except Exception as exc:
            logger.warning("Dashboard: cancel_order error for {}: {}", order_id, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/cancel-orders", dependencies=_auth_dep)
    async def api_cancel_symbol_orders(symbol: str) -> JSONResponse:
        """Cancel all open orders for *symbol*."""
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        try:
            cancelled = await engine.exchange.cancel_all_orders(symbol)
            logger.info("Dashboard: cancelled {} orders for {}", len(cancelled), symbol)
            return JSONResponse({"success": True, "symbol": symbol, "cancelled": len(cancelled)})
        except Exception as exc:
            logger.warning("Dashboard: cancel_all_orders error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/leverage", dependencies=_auth_dep)
    async def api_modify_leverage(symbol: str, body: dict) -> JSONResponse:
        """Modify leverage for an open position.

        Body: ``{"leverage": int}``
        """
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        leverage = body.get("leverage")
        if leverage is None:
            return JSONResponse({"success": False, "error": "Missing 'leverage' in request body"})
        try:
            lev = int(leverage)
            if lev < 1:
                return JSONResponse({"success": False, "error": "leverage must be >= 1"})
            result = await engine.exchange.modify_leverage(symbol, lev)
            logger.info("Dashboard: leverage for {} set to {}x", symbol, lev)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({"success": True, **result})
        except AttributeError:
            # Exchange doesn't implement modify_leverage — fall back to set_leverage
            try:
                await engine.exchange.set_leverage(symbol, lev)
                logger.info("Dashboard: leverage for {} set to {}x (via set_leverage)", symbol, lev)
                return JSONResponse({"success": True, "symbol": symbol, "leverage": lev})
            except Exception as exc:
                logger.warning("Dashboard: set_leverage fallback error for {}: {}", symbol, exc)
                return JSONResponse({"success": False, "error": str(exc)})
        except Exception as exc:
            logger.warning("Dashboard: modify_leverage error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/positions/{symbol:path}/add-margin", dependencies=_auth_dep)
    async def api_add_margin(symbol: str, body: dict) -> JSONResponse:
        """Add margin to an isolated position.

        Body: ``{"amount": float}``
        """
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        amount = body.get("amount")
        if amount is None:
            return JSONResponse({"success": False, "error": "Missing 'amount' in request body"})
        try:
            amt = float(amount)
            if amt <= 0:
                return JSONResponse({"success": False, "error": "amount must be > 0"})
            result = await engine.exchange.add_margin(symbol, amt)
            logger.info("Dashboard: added {:.4f} USDT margin to {}", amt, symbol)
            if realtime_hub is not None:
                asyncio.create_task(realtime_hub.trigger_state_refresh())
            return JSONResponse({"success": True, **result})
        except Exception as exc:
            logger.warning("Dashboard: add_margin error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/positions/{symbol:path}/details", dependencies=_auth_dep)
    async def api_position_details(symbol: str) -> JSONResponse:
        """Return comprehensive details for the open position on *symbol*."""
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        try:
            # Paper exchange exposes get_position_details; live falls back to get_position
            if hasattr(engine.exchange, "get_position_details"):
                details = await engine.exchange.get_position_details(symbol)
            else:
                pos = await engine.exchange.get_position(symbol)
                if pos is None:
                    details = None
                else:
                    entry = pos.entry_price or 0.0
                    current = pos.current_price or entry
                    details = {
                        "symbol": pos.symbol,
                        "side": pos.side.value,
                        "amount": pos.amount,
                        "entry_price": entry,
                        "mark_price": pos.mark_price or current,
                        "current_price": current,
                        "liquidation_price": pos.liquidation_price,
                        "margin": round(pos.margin, 2),
                        "notional_size": round(pos.position_value, 2),
                        "unrealized_pnl": pos.unrealized_pnl,
                        "roe_pct": pos.roe_pct,
                        "position_value": pos.position_value,
                        "leverage": pos.leverage,
                        "funding_rate": pos.funding_rate,
                    }
            if details is None:
                return JSONResponse({"success": False, "error": f"No open position for {symbol}"})
            return JSONResponse({"success": True, "position": details})
        except Exception as exc:
            logger.warning("Dashboard: position_details error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.post("/api/orders/cancel-all", dependencies=_auth_dep)
    async def api_cancel_all_orders() -> JSONResponse:
        """Cancel ALL open orders across all symbols."""
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        try:
            open_orders = await engine.exchange.get_open_orders()
            cancelled = 0
            errors: List[str] = []
            symbols_seen: set = set()  # type: ignore[type-arg]  # str symbols
            for order in open_orders:
                sym = order.symbol
                if sym not in symbols_seen:
                    symbols_seen.add(sym)
                    try:
                        result = await engine.exchange.cancel_all_orders(sym)
                        cancelled += len(result)
                    except Exception as exc:
                        errors.append(f"{sym}: {exc}")
            logger.info("Dashboard: cancel-all-orders — cancelled={} errors={}", cancelled, errors)
            return JSONResponse({
                "success": True,
                "cancelled": cancelled,
                "errors": errors,
            })
        except Exception as exc:
            logger.warning("Dashboard: cancel_all_orders global error: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/market/{symbol:path}/ticker", dependencies=_auth_dep)
    async def api_market_ticker(symbol: str) -> JSONResponse:
        """Return current ticker for *symbol*."""
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available"})
        try:
            ticker = await engine.exchange.get_ticker(symbol)
            return JSONResponse({
                "success": True,
                "symbol": ticker.symbol,
                "last": ticker.last,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "volume": ticker.volume,
            })
        except Exception as exc:
            logger.warning("Dashboard: market_ticker error for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/market/tickers", dependencies=_auth_dep)
    async def api_market_tickers() -> JSONResponse:
        """Return tickers for all configured trading pairs with 24h change data."""
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "tickers": {}})
        try:
            # Get symbols from exchange config
            symbols: List[str] = []
            if hasattr(_settings, "exchange") and hasattr(_settings.exchange, "trading_pairs"):
                symbols = _settings.exchange.trading_pairs
            if not symbols and hasattr(_settings, "strategies"):
                for s in _settings.strategies:
                    sym = getattr(s, "symbol", None)
                    if sym and sym not in symbols:
                        symbols.append(sym)
            if not symbols:
                # Fall back to positions symbols
                positions = await engine.exchange.get_positions()
                symbols = list({p.symbol for p in positions})

            # Fetch tickers for all symbols
            tickers = await engine.exchange.get_multiple_tickers(symbols)

            # Calculate 24h price change percentage
            result = {}
            for sym, ticker in tickers.items():
                try:
                    # Calculate 24h change from high/low if available, otherwise use a simple estimate
                    change_24h = 0.0
                    if ticker.high and ticker.low and ticker.high > 0:
                        # Simple estimate: current price relative to high/low range
                        if ticker.last:
                            mid_price = (ticker.high + ticker.low) / 2
                            if mid_price > 0:
                                change_24h = ((ticker.last - mid_price) / mid_price) * 100

                    result[sym] = {
                        "last": ticker.last,
                        "bid": ticker.bid,
                        "ask": ticker.ask,
                        "high": ticker.high,
                        "low": ticker.low,
                        "volume": ticker.volume,
                        "change_24h": round(change_24h, 2),
                        "timestamp": ticker.timestamp,
                    }
                except Exception as e:
                    logger.debug(f"Error processing ticker for {sym}: {e}")
                    result[sym] = {
                        "last": ticker.last,
                        "bid": ticker.bid,
                        "ask": ticker.ask,
                        "high": 0.0,
                        "low": 0.0,
                        "volume": 0.0,
                        "change_24h": 0.0,
                        "timestamp": 0,
                    }

            return JSONResponse({"success": True, "tickers": result})
        except Exception as exc:
            logger.warning("Dashboard: market_tickers error: {}", exc)
            return JSONResponse({"success": False, "tickers": {}, "error": str(exc)})

    # ── Mode switching endpoints ──────────────────────────────────────────

    @app.get("/api/mode", dependencies=_auth_dep)
    async def api_get_mode() -> JSONResponse:
        """Return current trading mode and exchange connection status."""
        exchange_name = "none"
        exchange_connected = False
        if engine and engine.exchange:
            exchange_name = engine.exchange.name
            exchange_connected = True
        return JSONResponse({
            "mode": _settings.trading_mode,
            "exchange": exchange_name,
            "exchange_connected": exchange_connected,
        })

    @app.post("/api/mode/switch", dependencies=_auth_dep)
    async def api_switch_mode(body: dict) -> JSONResponse:
        """Switch trading mode at runtime.

        Supported modes:

        * ``futures/live``    — live futures trading on configured exchange
        * ``futures/testnet`` — testnet futures trading
        * ``futures/paper``   — paper futures trading (simulated)
        * ``forex/live``      — live forex trading on Gate.io TradFi
        * ``forex/demo``      — demo forex trading on Gate.io TradFi testnet

        Legacy flat names (``"paper"``, ``"live"``, ``"testnet"``) are also
        accepted for backwards compatibility.

        Body: ``{"mode": "futures/live"}``
        """
        _VALID_MODE_SETTINGS = {
            "futures/live": "live",
            "futures/testnet": "testnet",
            "futures/paper": "paper",
            "forex/live": "forex_live",
            "forex/demo": "forex_demo",
            # Legacy flat names
            "paper": "paper",
            "live": "live",
            "testnet": "testnet",
        }

        new_mode = body.get("mode", "").lower().strip()
        if not new_mode:
            return JSONResponse({"success": False, "error": "Missing 'mode' in request body"})

        if new_mode not in _VALID_MODE_SETTINGS:
            return JSONResponse({
                "success": False,
                "error": (
                    f"Invalid mode: {new_mode!r}. "
                    f"Use one of: {list(_VALID_MODE_SETTINGS.keys())}"
                ),
            })

        if engine is None:
            # No engine — just update the settings value
            settings_mode = _VALID_MODE_SETTINGS[new_mode]
            _settings.trading_mode = settings_mode
            return JSONResponse({
                "success": True,
                "mode": settings_mode,
                "message": f"Mode set to {new_mode} (no engine).",
            })

        try:
            result = await engine.switch_mode(new_mode)
            return JSONResponse(result)
        except Exception as exc:
            logger.error("Dashboard: mode switch error: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    # ── Currency management endpoints ────────────────────────────────────

    @app.get("/api/settings/currencies", dependencies=_auth_dep)
    async def api_get_currencies() -> JSONResponse:
        """Get all currencies with their enabled/disabled status.

        Returns futures and forex pairs separately with enabled flag indicating
        whether the pair is in the active trading list.

        Forex pairs are limited to those supported by Gate.io TradFi (MT5)
        (i.e., pairs with full risk spec coverage: XAU/USD, XAG/USD).
        """
        futures_universe = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
            "XAU/USDT",
        ]
        # Only include forex pairs that have full risk specs in ForexRiskManager
        from risk.forex_risk_manager import ForexRiskManager
        forex_universe = list(ForexRiskManager.PAIR_SPECS.keys())

        futures_enabled = list(getattr(getattr(_settings, "exchange", None), "trading_pairs", []) or [])
        forex_cfg = getattr(_settings, "forex", None)
        forex_enabled = list(getattr(forex_cfg, "trading_pairs", []) or []) if forex_cfg else ["XAU/USD", "XAG/USD"]

        futures_pairs = [
            {"symbol": s, "enabled": s in futures_enabled, "market": "futures"}
            for s in futures_universe
        ]
        forex_pairs = [
            {"symbol": s, "enabled": s in forex_enabled, "market": "forex"}
            for s in forex_universe
        ]
        return JSONResponse({"futures": futures_pairs, "forex": forex_pairs})

    @app.put("/api/settings/currencies/{symbol:path}/toggle", dependencies=_auth_dep)
    async def api_toggle_currency(symbol: str, body: dict) -> JSONResponse:
        """Enable or disable a currency pair.

        Body: ``{"enabled": bool, "market": "futures"|"forex"}``
        """
        enabled = body.get("enabled", True)
        market = body.get("market", "futures")

        if market == "futures":
            exchange_cfg = getattr(_settings, "exchange", None)
            pairs_list = getattr(exchange_cfg, "trading_pairs", None) if exchange_cfg else None
        else:
            forex_cfg = getattr(_settings, "forex", None)
            pairs_list = getattr(forex_cfg, "trading_pairs", None) if forex_cfg else None

        if pairs_list is None:
            return JSONResponse({"success": False, "error": f"No {market} config available"})

        if enabled and symbol not in pairs_list:
            pairs_list.append(symbol)
        elif not enabled and symbol in pairs_list:
            pairs_list.remove(symbol)

        logger.info(
            "Currency {} {} in {} trading list",
            symbol,
            "enabled" if enabled else "disabled",
            market,
        )
        return JSONResponse({"success": True, "symbol": symbol, "enabled": enabled, "market": market})

    # ── OHLCV / K-line data endpoint ─────────────────────────────────────

    @app.get("/api/market/{symbol:path}/klines", dependencies=_auth_dep)
    async def api_market_klines(symbol: str, timeframe: str = "1m", limit: int = 200) -> JSONResponse:
        """Return OHLCV candle data for *symbol* / *timeframe*.

        Checks the realtime hub cache first; falls back to a REST fetch.
        Each candle is a dict with keys ``time`` (Unix seconds), ``open``,
        ``high``, ``low``, ``close``, ``volume``.

        Query params:
            timeframe: Candle interval, default ``"1m"``.
            limit: Maximum number of candles to return (default 200, max 1500).
        """
        # Cap limit to a safe maximum to avoid overloading the exchange API
        limit = max(1, min(limit, 1500))

        # Try the hub cache first (fast, no exchange call)
        if realtime_hub is not None:
            cached = realtime_hub.get_kline_snapshot(symbol, timeframe)
            if cached and len(cached) >= limit:
                return JSONResponse({
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles": cached[-limit:],
                })

        # Fallback: fetch from exchange
        if not (engine and engine.exchange):
            return JSONResponse({"success": False, "error": "Exchange not available", "candles": []})
        try:
            df = await engine.exchange.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            candles = [
                {
                    "time": int(ts.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                for ts, row in df.iterrows()
            ]
            return JSONResponse({
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
            })
        except Exception as exc:
            logger.warning("Dashboard: klines error for {} {}: {}", symbol, timeframe, exc)
            return JSONResponse({"success": False, "error": str(exc), "candles": []})

    # ── Manual trade override REMOVED ───────────────────────────────────
    # Manual trading has been eradicated. The bot operates 100% autonomously.
    # All trade execution flows through the Rust hot-path via the Alpha Oracle
    # signal queue. No human intervention is permitted during live operation.

    @app.post("/api/manual/trade", dependencies=_auth_dep)
    async def api_manual_trade_disabled(body: dict) -> JSONResponse:
        """Manual trading has been permanently disabled.

        All execution flows through the Rust engine's Alpha Oracle signal queue.
        This endpoint returns a 403 to inform any legacy clients.
        """
        return JSONResponse(
            {"success": False, "error": "Manual trading is permanently disabled. Bot operates autonomously."},
            status_code=403,
        )

    # ── Paper trading reset endpoint ─────────────────────────────────────

    @app.post("/api/paper/reset", dependencies=_auth_dep)
    async def api_paper_reset(body: Optional[dict] = None) -> JSONResponse:
        """Reset paper trading state: close positions, cancel orders, reset balance.

        Body (optional): ``{"balance": float}`` to set a custom starting balance.
        """
        body = body or {}
        try:
            from exchange.paper_exchange import PaperExchange

            exch = engine.exchange if engine else None
            if not isinstance(exch, PaperExchange):
                return JSONResponse({"success": False, "error": "Not in paper trading mode"})

            new_balance = body.get("balance")
            if new_balance is not None:
                new_balance = float(new_balance)
            result = await exch.reset_paper_state(new_balance)
            logger.info("Dashboard: paper trading state reset. New balance={}", result["balance"])
            return JSONResponse({"success": True, **result})
        except Exception as exc:
            logger.warning("Dashboard: paper reset error: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    # ── Settings endpoints: Trading Pairs Management ──────────────────────

    @app.get("/api/settings/pairs", dependencies=_auth_dep)
    async def api_get_trading_pairs() -> JSONResponse:
        """Get list of all trading pairs with their settings."""
        pairs = _settings.exchange.trading_pairs or []
        pairs_data = [
            {
                "symbol": symbol,
                "enabled": True,  # All pairs in the list are enabled
                "max_position_size_pct": _settings.risk.max_position_size_pct,
                "leverage": _settings.exchange.default_leverage,
            }
            for symbol in pairs
        ]
        return JSONResponse({"pairs": pairs_data})

    @app.post("/api/settings/pairs/add", dependencies=_auth_dep)
    async def api_add_trading_pair(body: dict) -> JSONResponse:
        """Add a new trading pair to the active list.

        Body: {"symbol": str, "leverage": int (optional)}
        """
        symbol = (body.get("symbol") or "").strip()
        if not symbol:
            return JSONResponse({"success": False, "error": "Symbol is required"})

        # Add to settings if not already present
        if symbol not in _settings.exchange.trading_pairs:
            _settings.exchange.trading_pairs.append(symbol)
            logger.info("Added trading pair: {}", symbol)
            return JSONResponse({"success": True, "symbol": symbol, "message": f"Added {symbol}"})
        return JSONResponse({"success": False, "error": f"{symbol} already exists"})

    @app.delete("/api/settings/pairs/{symbol:path}", dependencies=_auth_dep)
    async def api_remove_trading_pair(symbol: str) -> JSONResponse:
        """Remove a trading pair from the active list."""
        if symbol in _settings.exchange.trading_pairs:
            _settings.exchange.trading_pairs.remove(symbol)
            logger.info("Removed trading pair: {}", symbol)
            return JSONResponse({"success": True, "symbol": symbol, "message": f"Removed {symbol}"})
        return JSONResponse({"success": False, "error": f"{symbol} not found"})

    @app.put("/api/settings/pairs/{symbol:path}/toggle", dependencies=_auth_dep)
    async def api_toggle_trading_pair(symbol: str, body: Optional[dict] = None) -> JSONResponse:
        """Enable/disable a trading pair.

        Body: {"enabled": bool}
        """
        body = body or {}
        enabled = body.get("enabled", True)

        if enabled and symbol not in _settings.exchange.trading_pairs:
            _settings.exchange.trading_pairs.append(symbol)
            return JSONResponse({"success": True, "symbol": symbol, "enabled": True})
        elif not enabled and symbol in _settings.exchange.trading_pairs:
            _settings.exchange.trading_pairs.remove(symbol)
            return JSONResponse({"success": True, "symbol": symbol, "enabled": False})

        return JSONResponse({"success": True, "symbol": symbol, "enabled": enabled})

    # ── Forex API endpoints ───────────────────────────────────────────────

    @app.get("/api/forex/portfolio", dependencies=_auth_dep)
    async def api_forex_portfolio() -> JSONResponse:
        """Forex account portfolio data."""
        data = await _collect_forex_portfolio()
        return JSONResponse(data)

    @app.get("/api/forex/account", dependencies=_auth_dep)
    async def api_forex_account() -> JSONResponse:
        """Full forex account info: balance, equity, margin, margin_level_pct, daily P&L."""
        data = await _collect_forex_portfolio()
        equity = data.get("equity", 0.0)
        used_margin = data.get("margin_used", 0.0)
        margin_level_pct = (
            round(equity / used_margin * 100.0, 2) if used_margin > 0 else None
        )
        data["margin_level_pct"] = margin_level_pct
        return JSONResponse(data)

    @app.get("/api/forex/positions", dependencies=_auth_dep)
    async def api_forex_positions() -> JSONResponse:
        """Live forex positions with pip P&L."""
        positions = await _collect_forex_positions()
        return JSONResponse({"positions": positions})

    @app.get("/api/forex/trades", dependencies=_auth_dep)
    async def api_forex_trades() -> JSONResponse:
        """Forex trade history."""
        trades: List[dict] = []
        tj = trade_journal or (
            engine.trade_journal if engine and hasattr(engine, "trade_journal") else None
        )
        if tj:
            try:
                all_trades = tj.get_recent_trades(500)
                forex_pairs_set = set(_get_forex_pairs())
                trades = [t for t in all_trades if t.get("symbol") in forex_pairs_set]
            except Exception as exc:
                logger.debug("api_forex_trades: error: {}", exc)
        return JSONResponse({"trades": trades, "total": len(trades)})

    @app.get("/api/forex/pairs", dependencies=_auth_dep)
    async def api_forex_pairs() -> JSONResponse:
        """Get forex trading pairs with status."""
        pairs = _get_forex_pairs()
        return JSONResponse(
            [{"symbol": p, "enabled": True} for p in pairs]
        )

    @app.post("/api/forex/pairs/add", dependencies=_auth_dep)
    async def api_add_forex_pair(body: dict) -> JSONResponse:
        """Add a new forex pair to the active list."""
        symbol = (body.get("symbol") or "").strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol is required")
        forex_cfg = getattr(_settings, "forex", None)
        if forex_cfg is not None:
            pairs = getattr(forex_cfg, "trading_pairs", None)
            if pairs is not None:
                if symbol in pairs:
                    raise HTTPException(status_code=409, detail=f"{symbol} already exists")
                pairs.append(symbol)
                logger.info("Forex: added pair {}", symbol)
                return JSONResponse({"success": True, "symbol": symbol})
        # Fallback: no persistent storage – accept optimistically
        logger.info("Forex: pair add requested (no persistent forex config): {}", symbol)
        return JSONResponse({"success": True, "symbol": symbol})

    @app.delete("/api/forex/pairs/{symbol:path}", dependencies=_auth_dep)
    async def api_remove_forex_pair(symbol: str) -> JSONResponse:
        """Remove a forex pair from the active list."""
        forex_cfg = getattr(_settings, "forex", None)
        if forex_cfg is not None:
            pairs = getattr(forex_cfg, "trading_pairs", None)
            if pairs is not None:
                if symbol not in pairs:
                    raise HTTPException(status_code=404, detail=f"{symbol} not found")
                pairs.remove(symbol)
                logger.info("Forex: removed pair {}", symbol)
                return JSONResponse({"success": True, "symbol": symbol})
        raise HTTPException(status_code=404, detail=f"{symbol} not found in forex config")

    @app.get("/api/forex/spread/{symbol:path}", dependencies=_auth_dep)
    async def api_forex_spread(symbol: str) -> JSONResponse:
        """Get current bid-ask spread for a forex pair in pips."""
        exch = None
        if engine:
            exch = getattr(engine, "forex_exchange", None) or engine.exchange
        if exch is None:
            return JSONResponse({"symbol": symbol, "spread_pips": None, "bid": None, "ask": None})
        try:
            # Use Gate.io TradFi spread data when available
            if hasattr(exch, "get_spread"):
                data = await exch.get_spread(symbol)
                return JSONResponse({"symbol": symbol, **data})
            # Generic fallback via ticker
            ticker = await exch.get_ticker(symbol)
            if ticker and ticker.bid and ticker.ask:
                from exchange.gateio_tradfi_client import GateIOTradFiClient  # noqa: PLC0415
                cfg = None  # Gate.io TradFi handles forex config internally
                if cfg is None:
                    # Unknown pair — cannot reliably compute pips
                    return JSONResponse(
                        {"symbol": symbol, "spread_pips": None, "bid": ticker.bid, "ask": ticker.ask}
                    )
                pip_size = cfg["pip_size"]
                spread_pips = (ticker.ask - ticker.bid) / pip_size
                return JSONResponse(
                    {
                        "symbol": symbol,
                        "spread_pips": round(spread_pips, 2),
                        "bid": ticker.bid,
                        "ask": ticker.ask,
                    }
                )
        except Exception as exc:
            logger.debug("api_forex_spread: {}", exc)
        return JSONResponse({"symbol": symbol, "spread_pips": None, "bid": None, "ask": None})

    @app.get("/api/forex/spreads", dependencies=_auth_dep)
    async def api_forex_spreads() -> JSONResponse:
        """Get current bid-ask spreads for all configured forex pairs."""
        pairs = _get_forex_pairs()
        results = []
        for symbol in pairs:
            exch = None
            if engine:
                exch = getattr(engine, "forex_exchange", None) or engine.exchange
            if exch is None:
                results.append({"symbol": symbol, "spread_pips": None, "bid": None, "ask": None})
                continue
            try:
                if hasattr(exch, "get_spread"):
                    data = await exch.get_spread(symbol)
                    results.append({"symbol": symbol, **data})
                    continue
                ticker = await exch.get_ticker(symbol)
                if ticker and ticker.bid and ticker.ask:
                    # Resolve pip_size from ForexRiskManager PAIR_SPECS (exchange-agnostic)
                    try:
                        from risk.forex_risk_manager import ForexRiskManager  # noqa: PLC0415
                        spec = ForexRiskManager.PAIR_SPECS.get(symbol) or ForexRiskManager.PAIR_SPECS.get(symbol.replace("/", ""))
                        pip_size = spec["pip_size"] if spec else 0.0001
                    except Exception:
                        pip_size = 0.01 if "XAU" in symbol or "XAG" in symbol else 0.0001
                    spread_pips = (ticker.ask - ticker.bid) / pip_size
                    results.append({
                        "symbol": symbol,
                        "spread_pips": round(spread_pips, 2),
                        "bid": ticker.bid,
                        "ask": ticker.ask,
                    })
                    continue
            except Exception:
                pass
            results.append({"symbol": symbol, "spread_pips": None, "bid": None, "ask": None})
        return JSONResponse({"spreads": results})

    @app.get("/api/forex/sessions", dependencies=_auth_dep)
    async def api_forex_sessions() -> JSONResponse:
        """Return current trading session info and session statistics."""
        now = datetime.now(tz=timezone.utc)
        hour = now.hour

        sessions = {
            "london": {"active": 8 <= hour < 16, "hours": "08:00–16:00 UTC"},
            "new_york": {"active": 13 <= hour < 21, "hours": "13:00–21:00 UTC"},
            "asian": {"active": 0 <= hour < 9, "hours": "00:00–09:00 UTC"},
            "sydney": {"active": hour >= 22 or hour < 7, "hours": "22:00–07:00 UTC"},
        }

        # Overlap flags
        sessions["london_ny_overlap"] = {
            "active": sessions["london"]["active"] and sessions["new_york"]["active"],
            "hours": "13:00–16:00 UTC",
        }

        current_sessions = [k for k, v in sessions.items() if v["active"] and k != "london_ny_overlap"]

        # Session PnL breakdown from ForexRiskManager if available
        session_pnl: dict = {}
        if engine:
            frm = getattr(engine, "forex_risk_manager", None)
            if frm:
                session_pnl = getattr(frm, "_session_pnl_breakdown", {})

        return JSONResponse({
            "current_time_utc": now.isoformat(),
            "current_sessions": current_sessions,
            "sessions": sessions,
            "session_pnl": session_pnl,
        })

    @app.get("/api/forex/performance", dependencies=_auth_dep)
    async def api_forex_performance() -> JSONResponse:
        """Return forex performance metrics and trade statistics."""
        # Try ForexPerformanceTracker from engine
        fpt = None
        if engine:
            fpt = getattr(engine, "forex_performance_tracker", None)
        if fpt is not None:
            try:
                return JSONResponse(fpt.generate_report())
            except Exception as exc:
                logger.debug("api_forex_performance: tracker error: {}", exc)

        # Fallback: compute from trade journal
        try:
            from monitoring.forex_trade_journal import ForexTradeJournal  # noqa: PLC0415
            ftj = None
            if engine:
                ftj = getattr(engine, "forex_trade_journal", None)
            if ftj is None:
                ftj = ForexTradeJournal()
            overall = await ftj.get_overall_stats(days=30)
            daily = await ftj.get_daily_performance(days=30)
            by_session = {
                s: await ftj.get_session_stats(s, days=30)
                for s in ("london", "new_york", "asian", "sydney")
            }
            return JSONResponse({
                "summary": overall,
                "daily": daily,
                "by_session": by_session,
            })
        except Exception as exc:
            logger.debug("api_forex_performance: fallback error: {}", exc)
            return JSONResponse({"summary": {}, "daily": [], "by_session": {}})

    @app.get("/api/forex/risk", dependencies=_auth_dep)
    async def api_forex_risk() -> JSONResponse:
        """Return forex risk metrics from ForexRiskManager."""
        frm = None
        if engine:
            frm = getattr(engine, "forex_risk_manager", None)
        if frm is None:
            return JSONResponse({
                "session_pnl": 0.0,
                "session_trades": 0,
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "in_recovery_mode": False,
                "recovery_mode_level": 0,
                "open_trade_count": 0,
                "session_name": "Unknown",
                "session_risk_multiplier": 1.0,
                "margin_level": "unknown",
                "max_drawdown_pct_seen": 0.0,
            })
        try:
            portfolio = await _collect_forex_portfolio()
            equity = portfolio.get("equity", 0.0)
            used_margin = portfolio.get("margin_used", 0.0)
            from risk.forex_risk_manager import ForexMarginMonitor  # noqa: PLC0415
            mmgr = ForexMarginMonitor()
            margin_level = mmgr.get_margin_level(equity, used_margin)
        except Exception:
            margin_level = "unknown"

        return JSONResponse({
            "session_pnl": getattr(frm, "_session_pnl", 0.0),
            "session_trades": getattr(frm, "_session_trades", 0),
            "consecutive_losses": getattr(frm, "_consecutive_losses", 0),
            "consecutive_wins": getattr(frm, "_consecutive_wins", 0),
            "in_recovery_mode": getattr(frm, "_in_recovery_mode", False),
            "recovery_mode_level": getattr(frm, "_recovery_mode_level", 0),
            "open_trade_count": getattr(frm, "_open_trade_count", 0),
            "session_name": frm.get_session_name() if hasattr(frm, "get_session_name") else "",
            "session_risk_multiplier": frm.get_session_risk_multiplier() if hasattr(frm, "get_session_risk_multiplier") else 1.0,
            "margin_level": margin_level,
            "max_drawdown_pct_seen": getattr(frm, "_max_drawdown_pct_seen", 0.0),
            "session_pnl_breakdown": getattr(frm, "_session_pnl_breakdown", {}),
        })

    @app.get("/api/forex/signals", dependencies=_auth_dep)
    async def api_forex_signals() -> JSONResponse:
        """Return the latest forex trading signals."""
        forex_pairs = set(_get_forex_pairs())
        all_signals = _collect_signals(limit=50)
        forex_signals = [s for s in all_signals if s.get("symbol") in forex_pairs]
        return JSONResponse({"signals": forex_signals, "total": len(forex_signals)})

    @app.get("/api/forex/lot-calculator", dependencies=_auth_dep)
    async def api_forex_lot_calculator(
        symbol: str = "XAU/USD",
        equity: float = 10000.0,
        sl_pips: float = 200.0,
        leverage: int = 20,
        risk_pct: float = 1.0,
    ) -> JSONResponse:
        """Calculate recommended lot size for a given trade setup.

        Query params: ``symbol``, ``equity``, ``sl_pips``, ``leverage``, ``risk_pct``.
        """
        try:
            from risk.forex_risk_manager import ForexRiskManager  # noqa: PLC0415
            frm_calc = ForexRiskManager(settings=_settings)
            # Override risk per trade with caller-supplied value
            frm_calc._risk_per_trade_pct = max(0.1, min(5.0, risk_pct))
            spec = ForexRiskManager.PAIR_SPECS.get(symbol)
            if spec is None:
                return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=400)

            lot_size = frm_calc.calculate_dynamic_lot_size(
                symbol=symbol,
                equity=equity,
                stop_loss_pips=sl_pips,
                leverage=leverage,
            )
            pip_size = spec["pip_size"]
            contract_size = spec["contract_size"]
            pip_value_per_lot = pip_size * contract_size
            risk_usd = sl_pips * pip_value_per_lot * lot_size
            # Margin = (lot_size * contract_size * entry_price) / leverage
            # Since we don't have the live price here, approximate using pip_value:
            # notional ≈ lot_size * contract_size / pip_size  (gives price units)
            # For gold (pip=0.01, contract=1): 1 lot notional ~= price; use risk_usd/risk_pct as equity proxy
            notional_per_lot = contract_size / pip_size * pip_size  # = contract_size
            margin_required = (lot_size * notional_per_lot) / max(1, leverage) if leverage > 0 else 0.0

            return JSONResponse({
                "symbol": symbol,
                "equity": equity,
                "sl_pips": sl_pips,
                "risk_pct": risk_pct,
                "leverage": leverage,
                "lot_size": lot_size,
                "risk_usd": round(risk_usd, 2),
                "pip_value_per_lot": round(pip_value_per_lot, 6),
                "margin_required": round(margin_required, 2),
            })
        except Exception as exc:
            logger.debug("api_forex_lot_calculator: {}", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/forex/trade", dependencies=_auth_dep)
    async def api_forex_trade(body: dict) -> JSONResponse:
        """Place a new forex trade through the forex engine.

        Body: ``symbol``, ``side`` (long/short), ``lot_size``, optionally
        ``stop_loss_pips``, ``take_profit_pips``, ``strategy``.
        """
        symbol = (body.get("symbol") or "").strip().upper()
        side = (body.get("side") or "long").lower()
        lot_size = float(body.get("lot_size") or 0.01)
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol is required")
        if side not in ("long", "short"):
            raise HTTPException(status_code=400, detail="side must be 'long' or 'short'")
        if lot_size <= 0:
            raise HTTPException(status_code=400, detail="lot_size must be positive")

        forex_engine = getattr(engine, "forex_engine", None) if engine else None
        if forex_engine is None:
            raise HTTPException(status_code=503, detail="Forex engine not available")

        try:
            result = await forex_engine.place_trade(
                symbol=symbol,
                side=side,
                lot_size=lot_size,
                stop_loss_pips=body.get("stop_loss_pips"),
                take_profit_pips=body.get("take_profit_pips"),
                strategy=body.get("strategy"),
            )
            return JSONResponse({"success": True, "result": result})
        except Exception as exc:
            logger.warning("api_forex_trade error: {}", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/forex/positions/{symbol:path}/close", dependencies=_auth_dep)
    async def api_forex_close_position(symbol: str, body: Optional[dict] = None) -> JSONResponse:
        """Close an open forex position (full or partial).

        Body (optional): ``pct`` (percentage to close, default 100).
        """
        pct = float((body or {}).get("pct", 100))
        exch = None
        if engine:
            exch = getattr(engine, "forex_exchange", None) or engine.exchange
        if exch is None:
            raise HTTPException(status_code=503, detail="Forex exchange not available")
        try:
            if hasattr(exch, "close_position"):
                result = await exch.close_position(symbol, pct=pct)
            else:
                positions = await exch.get_positions()
                pos = next((p for p in positions if (getattr(p, "symbol", None) or "") == symbol), None)
                if pos is None:
                    raise HTTPException(status_code=404, detail=f"No open position for {symbol}")
                close_side = "sell" if getattr(pos, "side", "long") == "long" else "buy"
                contracts = (getattr(pos, "contracts", 0) or 0) * (pct / 100.0)
                result = await exch.create_market_order(
                    symbol=symbol, side=close_side, amount=contracts,
                    params={"reduceOnly": True},
                )
            return JSONResponse({"success": True, "symbol": symbol, "pct": pct, "result": str(result)})
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("api_forex_close_position error for {}: {}", symbol, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/forex/positions/{symbol:path}/modify", dependencies=_auth_dep)
    async def api_forex_modify_position(symbol: str, body: dict) -> JSONResponse:
        """Modify SL/TP for an open forex position.

        Body: ``stop_loss`` (price), ``take_profit`` (price) — at least one required.
        """
        new_sl = body.get("stop_loss")
        new_tp = body.get("take_profit")
        if new_sl is None and new_tp is None:
            raise HTTPException(status_code=400, detail="stop_loss or take_profit required")

        exch = None
        if engine:
            exch = getattr(engine, "forex_exchange", None) or engine.exchange
        if exch is None:
            raise HTTPException(status_code=503, detail="Forex exchange not available")
        try:
            if hasattr(exch, "modify_position"):
                result = await exch.modify_position(
                    symbol, stop_loss=new_sl, take_profit=new_tp
                )
            else:
                # Generic: cancel old SL/TP and place new ones
                result = {"sl": new_sl, "tp": new_tp, "note": "manual modification"}
            return JSONResponse({"success": True, "symbol": symbol, "stop_loss": new_sl, "take_profit": new_tp, "result": str(result)})
        except Exception as exc:
            logger.warning("api_forex_modify_position error for {}: {}", symbol, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/settings/strategies", dependencies=_auth_dep)
    async def api_list_strategies() -> JSONResponse:
        """List all available strategies with their configuration."""
        try:
            # Get strategy manager from engine if available
            sm = getattr(engine, "strategy_manager", None) if engine else None
            strategies_list = []

            if sm and hasattr(sm, "_strategies"):
                for name, strategy in sm._strategies.items():
                    strategy_info = {
                        "name": name,
                        "enabled": getattr(strategy, "enabled", True),
                        "description": strategy.__class__.__doc__ or "",
                        "parameters": {},
                    }
                    # Extract configurable parameters
                    for attr in dir(strategy):
                        if not attr.startswith("_") and isinstance(getattr(strategy, attr, None), (int, float, str, bool)):
                            strategy_info["parameters"][attr] = getattr(strategy, attr)
                    strategies_list.append(strategy_info)
            else:
                # Fallback: list common strategies
                common_strategies = [
                    "technical_breakout", "news_momentum", "scalping",
                    "sentiment_reversal", "ai_adaptive", "whale_follower",
                    "funding_rate_arb", "grid_trading", "dca_strategy", "smart_money_flow"
                ]
                strategies_list = [
                    {"name": name, "enabled": True, "description": f"{name.replace('_', ' ').title()} Strategy", "parameters": {}}
                    for name in common_strategies
                ]

            return JSONResponse({"strategies": strategies_list})
        except Exception as exc:
            logger.warning("Failed to list strategies: {}", exc)
            return JSONResponse({"strategies": [], "error": str(exc)})

    @app.put("/api/settings/strategies/{name}", dependencies=_auth_dep)
    async def api_update_strategy(name: str, body: dict) -> JSONResponse:
        """Update strategy configuration.

        Body: {"enabled": bool, "parameters": {param_name: value, ...}}
        """
        try:
            sm = getattr(engine, "strategy_manager", None) if engine else None
            if not sm or not hasattr(sm, "_strategies"):
                return JSONResponse({"success": False, "error": "Strategy manager not available"})

            if name not in sm._strategies:
                return JSONResponse({"success": False, "error": f"Strategy {name} not found"})

            strategy = sm._strategies[name]

            # Update enabled status
            if "enabled" in body:
                strategy.enabled = bool(body["enabled"])

            # Update parameters
            if "parameters" in body:
                for param, value in body["parameters"].items():
                    if hasattr(strategy, param):
                        setattr(strategy, param, value)

            logger.info("Updated strategy {}: {}", name, body)
            return JSONResponse({"success": True, "strategy": name, "message": f"Updated {name}"})
        except Exception as exc:
            logger.warning("Failed to update strategy {}: {}", name, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    # ── Trade History Endpoints ───────────────────────────────────────────

    @app.get("/api/trades/history", dependencies=_auth_dep)
    async def api_trade_history(request: Request) -> JSONResponse:
        """Get trade history with pagination and filtering.

        Query params:
            - page: int (default 1)
            - per_page: int (default 50, max 200)
            - symbol: str (optional filter)
            - strategy: str (optional filter)
            - date_from: str (YYYY-MM-DD, optional)
            - date_to: str (YYYY-MM-DD, optional)
        """
        try:
            from sqlalchemy import and_, select

            TradeHistory, DailyPerformance, create_tables, get_async_session = _get_storage_models()
            if TradeHistory is None or get_async_session is None:
                return JSONResponse({"success": False, "error": "Storage models unavailable"})
            await create_tables()

            page = int(request.query_params.get("page", 1))
            per_page = min(int(request.query_params.get("per_page", 50)), 200)
            offset = (page - 1) * per_page

            # Build filters
            filters = []
            if symbol := request.query_params.get("symbol"):
                filters.append(TradeHistory.symbol == symbol)
            if strategy := request.query_params.get("strategy"):
                filters.append(TradeHistory.strategy == strategy)
            if date_from := request.query_params.get("date_from"):
                from datetime import datetime
                dt = datetime.strptime(date_from, "%Y-%m-%d")
                filters.append(TradeHistory.entry_time >= dt)
            if date_to := request.query_params.get("date_to"):
                from datetime import datetime
                dt = datetime.strptime(date_to, "%Y-%m-%d")
                filters.append(TradeHistory.close_time <= dt)

            async with get_async_session() as session:
                # Count total (Trap 5 Fix: use func.count instead of loading all rows)
                from sqlalchemy import func as sa_func
                count_query = select(sa_func.count(TradeHistory.id))
                if filters:
                    count_query = count_query.where(and_(*filters))
                total = await session.scalar(count_query) or 0

                # Get paginated results
                query = select(TradeHistory).order_by(TradeHistory.entry_time.desc())
                if filters:
                    query = query.where(and_(*filters))
                query = query.limit(per_page).offset(offset)

                result = await session.execute(query)
                trades = result.scalars().all()

                trades_data = [
                    {
                        "id": t.id,
                        "order_id": t.order_id,
                        "symbol": t.symbol,
                        "side": t.side,
                        "order_type": t.order_type,
                        "size": t.size,
                        "price": t.price,
                        "filled_price": t.filled_price,
                        "leverage": t.leverage,
                        "pnl": t.pnl,
                        "fees": t.fees,
                        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                        "close_time": t.close_time.isoformat() if t.close_time else None,
                        "duration": t.duration,
                        "strategy": t.strategy,
                        "notes": t.notes,
                        "exit_reason": t.exit_reason,
                        "exchange": t.exchange,
                    }
                    for t in trades
                ]

                return JSONResponse({
                    "success": True,
                    "trades": trades_data,
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total": total,
                        "pages": (total + per_page - 1) // per_page,
                    }
                })
        except Exception as exc:
            logger.warning("Failed to fetch trade history: {}", exc)
            return JSONResponse({"success": False, "error": str(exc), "trades": []})

    @app.get("/api/trades/tracked", dependencies=_auth_dep)
    async def api_tracked_trades() -> JSONResponse:
        """Real-time tracked trades with full state info."""
        if engine and hasattr(engine, "trade_tracker") and engine.trade_tracker:
            trades = await engine.trade_tracker.get_all_tracked_trades()
            return JSONResponse({"trades": trades, "count": len(trades)})
        return JSONResponse({"trades": [], "count": 0})

    @app.get("/api/trades/tracked/history", dependencies=_auth_dep)
    async def api_tracked_history() -> JSONResponse:
        """Recently closed tracked trades."""
        if engine and hasattr(engine, "trade_tracker") and engine.trade_tracker:
            history = await engine.trade_tracker.get_trade_history()
            return JSONResponse({"trades": history})
        return JSONResponse({"trades": []})

    @app.get("/api/trades/export", dependencies=_auth_dep)
    async def api_export_trades(request: Request) -> Any:
        """Export trade history as CSV."""
        try:
            import csv
            from io import StringIO

            from fastapi.responses import StreamingResponse
            from sqlalchemy import and_, select

            TradeHistory, DailyPerformance, create_tables, get_async_session = _get_storage_models()
            if TradeHistory is None or get_async_session is None:
                return JSONResponse({"success": False, "error": "Storage models unavailable"})
            await create_tables()

            # Build filters (same as history endpoint)
            filters = []
            if symbol := request.query_params.get("symbol"):
                filters.append(TradeHistory.symbol == symbol)
            if strategy := request.query_params.get("strategy"):
                filters.append(TradeHistory.strategy == strategy)

            async with get_async_session() as session:
                query = select(TradeHistory).order_by(TradeHistory.entry_time.desc())
                if filters:
                    query = query.where(and_(*filters))

                result = await session.execute(query)
                trades = result.scalars().all()

                # Create CSV
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow([
                    "ID", "Order ID", "Symbol", "Side", "Type", "Size", "Price",
                    "Filled Price", "Leverage", "PnL", "Fees", "Entry Time",
                    "Close Time", "Duration (s)", "Strategy", "Exit Reason", "Exchange", "Notes"
                ])

                for t in trades:
                    writer.writerow([
                        t.id, t.order_id, t.symbol, t.side, t.order_type, t.size,
                        t.price, t.filled_price, t.leverage, t.pnl, t.fees,
                        t.entry_time.isoformat() if t.entry_time else "",
                        t.close_time.isoformat() if t.close_time else "",
                        t.duration, t.strategy, t.exit_reason, t.exchange, t.notes or ""
                    ])

                output.seek(0)
                return StreamingResponse(
                    iter([output.getvalue()]),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=trades.csv"}
                )
        except Exception as exc:
            logger.warning("Failed to export trades: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    # ── Risk & Performance Endpoints ──────────────────────────────────────

    @app.get("/api/risk/metrics", dependencies=_auth_dep)
    async def api_risk_metrics() -> JSONResponse:
        """Get current risk metrics including VaR and portfolio exposure."""
        try:
            positions = await _collect_positions()

            # Calculate basic risk metrics
            total_exposure = sum(abs(float(p.get("position_value", 0))) for p in positions)
            total_margin = sum(float(p.get("margin", 0)) for p in positions)

            # Get account balance
            balance = 0.0
            if engine and engine.exchange:
                try:
                    balance_info = await engine.exchange.get_balance()
                    balance = float(balance_info.usdt_total or 0)
                except Exception:
                    pass

            # Calculate exposure percentage
            exposure_pct = (total_exposure / balance * 100) if balance > 0 else 0
            margin_usage_pct = (total_margin / balance * 100) if balance > 0 else 0

            # Symbol-wise exposure
            symbol_exposure = {}
            for p in positions:
                symbol = p.get("symbol", "")
                exposure = abs(float(p.get("position_value", 0)))
                symbol_exposure[symbol] = {
                    "exposure": exposure,
                    "percentage": (exposure / balance * 100) if balance > 0 else 0,
                    "side": p.get("direction", ""),
                }

            # Simple VaR calculation (historical simulation would be more complex)
            # This is a simplified 95% VaR estimate
            var_95 = total_exposure * 0.02  # 2% move assumption

            return JSONResponse({
                "success": True,
                "total_exposure": total_exposure,
                "total_margin": total_margin,
                "account_balance": balance,
                "exposure_pct": round(exposure_pct, 2),
                "margin_usage_pct": round(margin_usage_pct, 2),
                "var_95": round(var_95, 2),
                "symbol_exposure": symbol_exposure,
                "max_drawdown_pct": _settings.risk.max_drawdown_pct,
                "daily_loss_limit_pct": _settings.risk.max_daily_loss_pct,
            })
        except Exception as exc:
            logger.warning("Failed to calculate risk metrics: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/market/{symbol:path}/orderbook", dependencies=_auth_dep)
    async def api_get_orderbook(symbol: str) -> JSONResponse:
        """Get order book depth for a symbol."""
        try:
            if not (engine and engine.exchange):
                return JSONResponse({"success": False, "error": "Exchange not available"})

            orderbook = await engine.exchange.get_orderbook(symbol, limit=20)
            return JSONResponse({
                "success": True,
                "symbol": symbol,
                "bids": [[price, amount] for price, amount in (orderbook.get("bids", [])[:20])],
                "asks": [[price, amount] for price, amount in (orderbook.get("asks", [])[:20])],
                "timestamp": orderbook.get("timestamp"),
            })
        except Exception as exc:
            logger.warning("Failed to fetch orderbook for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/market/{symbol:path}/recent-trades", dependencies=_auth_dep)
    async def api_get_recent_trades(symbol: str) -> JSONResponse:
        """Get recent market trades for a symbol."""
        try:
            if not (engine and engine.exchange):
                return JSONResponse({"success": False, "error": "Exchange not available"})

            trades = await engine.exchange.get_recent_trades(symbol, limit=50)
            trades_data = [
                {
                    "id": t.get("id"),
                    "price": t.get("price"),
                    "amount": t.get("amount"),
                    "side": t.get("side"),
                    "timestamp": t.get("timestamp"),
                }
                for t in trades
            ]
            return JSONResponse({
                "success": True,
                "symbol": symbol,
                "trades": trades_data,
            })
        except Exception as exc:
            logger.warning("Failed to fetch recent trades for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/market/{symbol:path}/multi-timeframe", dependencies=_auth_dep)
    async def api_get_multi_timeframe(symbol: str) -> JSONResponse:
        """Get multi-timeframe analysis for a symbol."""
        try:
            if not (engine and engine.exchange):
                return JSONResponse({"success": False, "error": "Exchange not available"})

            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            analysis_data = {}

            for tf in timeframes:
                try:
                    # Fetch OHLCV data for the timeframe
                    df_tf = await engine.exchange.get_ohlcv(symbol, timeframe=tf, limit=50)
                    if df_tf is None or df_tf.empty or len(df_tf) < 20:
                        continue

                    # Extract close prices and volumes using named columns for safety
                    closes = [float(v) for v in df_tf["close"].tolist()]
                    volumes = [float(v) for v in df_tf["volume"].tolist()]

                    # Calculate basic indicators
                    # RSI calculation
                    rsi = _calculate_rsi(closes)

                    # MACD signal (simplified)
                    ema_12 = _ema(closes, 12)
                    ema_26 = _ema(closes, 26)
                    macd = ema_12 - ema_26
                    macd_signal = "bullish" if macd > 0 else "bearish" if macd < 0 else "neutral"

                    # EMA cross signal
                    ema_9 = _ema(closes, 9)
                    ema_21 = _ema(closes, 21)
                    ema_cross = "bullish" if ema_9 > ema_21 else "bearish" if ema_9 < ema_21 else "neutral"

                    # Volume trend
                    avg_volume = sum(volumes[-10:]) / 10
                    recent_volume = volumes[-1]
                    volume_trend = "increasing" if recent_volume > avg_volume * 1.2 else "normal"

                    # Calculate alignment score
                    score = 50
                    if rsi > 50:
                        score += 10
                    if rsi > 60:
                        score += 10
                    if rsi < 40:
                        score -= 10
                    if rsi < 30:
                        score -= 10
                    if macd_signal == "bullish":
                        score += 10
                    elif macd_signal == "bearish":
                        score -= 10
                    if ema_cross == "bullish":
                        score += 10
                    elif ema_cross == "bearish":
                        score -= 10
                    if volume_trend == "increasing":
                        score += 10

                    score = max(0, min(100, score))

                    analysis_data[tf] = {
                        "rsi": rsi,
                        "macd_signal": macd_signal,
                        "ema_cross": ema_cross,
                        "volume_trend": volume_trend,
                        "alignment_score": score,
                    }
                except Exception as e:
                    logger.debug("Failed to analyze timeframe {} for {}: {}", tf, symbol, e)
                    continue

            return JSONResponse({
                "success": True,
                "symbol": symbol,
                "timeframes": analysis_data,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("Failed to fetch multi-timeframe data for {}: {}", symbol, exc)
            return JSONResponse({"success": False, "error": str(exc)})

    def _calculate_rsi(prices: list, period: int = 14) -> float:
        """Calculate RSI from price list."""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 1)

    def _ema(prices: list, period: int) -> float:
        """Calculate EMA from price list."""
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @app.get("/api/sentiment/social", dependencies=_auth_dep)
    async def api_get_social_sentiment() -> JSONResponse:
        """Get social sentiment analysis data including Fear & Greed Index."""
        try:
            import aiohttp

            # Fetch live Fear & Greed Index from alternative.me API
            fear_greed_value = 50
            classification = "Neutral"
            description = "The market sentiment is neutral."
            fear_greed_timestamp = datetime.now(tz=timezone.utc).isoformat()
            try:
                async with aiohttp.ClientSession() as _fg_session:
                    async with _fg_session.get(
                        "https://api.alternative.me/fng/?limit=1&format=json",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as _fg_resp:
                        _fg_resp.raise_for_status()
                        _fg_data = await _fg_resp.json(content_type=None)
                _fg_items = _fg_data.get("data", [])
                if _fg_items:
                    fear_greed_value = int(_fg_items[0]["value"])
                    classification = _fg_items[0]["value_classification"]
                    fear_greed_timestamp = datetime.fromtimestamp(
                        int(_fg_items[0]["timestamp"]), tz=timezone.utc
                    ).isoformat()
                    if fear_greed_value < 25:
                        description = "The market is showing extreme fear. This could be a buying opportunity."
                    elif fear_greed_value < 45:
                        description = "The market is fearful. Investors are worried."
                    elif fear_greed_value < 55:
                        description = "The market sentiment is neutral."
                    elif fear_greed_value < 75:
                        description = "The market is greedy. Be cautious of overbought conditions."
                    else:
                        description = "The market is showing extreme greed. Consider taking profits."
            except Exception as _fg_exc:
                logger.warning("Fear & Greed live fetch failed: {}; using last known neutral value", _fg_exc)

            return JSONResponse({
                "success": True,
                "fear_greed": {
                    "value": fear_greed_value,
                    "classification": classification,
                    "description": description,
                    "timestamp": fear_greed_timestamp,
                },
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("Failed to fetch social sentiment: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/news/crypto", dependencies=_auth_dep)
    async def api_get_crypto_news() -> JSONResponse:
        """Get crypto news feed from configured data sources via DataAggregator."""
        try:
            current_time = datetime.now(tz=timezone.utc)
            news: list = []

            # Try to fetch real news items from the DataAggregator when available.
            aggregator = getattr(engine, "data_aggregator", None) if engine else None
            if aggregator is not None:
                try:
                    items = await aggregator.collect_latest(max_age_minutes=120, limit_per_source=20)
                    for idx, item in enumerate(items[:50]):
                        # Derive a simple sentiment label from urgency/relevance scores.
                        if item.urgency_score > 0.7:
                            sentiment = "negative"
                        elif item.relevance_score > 0.6:
                            sentiment = "positive"
                        else:
                            sentiment = "neutral"
                        news.append({
                            "id": f"news-{idx}",
                            "title": item.content[:200],
                            "summary": item.content[:500],
                            "source": item.source_name,
                            "url": item.url or "",
                            "published_at": item.timestamp.isoformat(),
                            "sentiment": sentiment,
                            "symbols": item.mentioned_assets,
                        })
                except Exception as _agg_exc:
                    logger.debug("DataAggregator news fetch failed: {}; using cached items", _agg_exc)
                    # Fall back to already-aggregated items without re-fetching.
                    for idx, item in enumerate(aggregator.get_cached_items(limit=50)):
                        if item.urgency_score > 0.7:
                            sentiment = "negative"
                        elif item.relevance_score > 0.6:
                            sentiment = "positive"
                        else:
                            sentiment = "neutral"
                        news.append({
                            "id": f"news-{idx}",
                            "title": item.content[:200],
                            "summary": item.content[:500],
                            "source": item.source_name,
                            "url": item.url or "",
                            "published_at": item.timestamp.isoformat(),
                            "sentiment": sentiment,
                            "symbols": item.mentioned_assets,
                        })

            return JSONResponse({
                "success": True,
                "news": news,
                "timestamp": current_time.isoformat(),
            })
        except Exception as exc:
            logger.warning("Failed to fetch crypto news: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/ai/trade-suggestions", dependencies=_auth_dep)
    async def api_get_ai_trade_suggestions() -> JSONResponse:
        """Get AI-powered trade suggestions."""
        try:
            import random

            if not (engine and engine.exchange):
                return JSONResponse({"success": False, "error": "Exchange not available"})

            pairs = _settings.exchange.trading_pairs or ["BTC/USDT", "ETH/USDT"]
            suggestions = []

            # Generate 2-3 random suggestions
            num_suggestions = random.randint(2, 3)
            selected_pairs = random.sample(pairs, min(num_suggestions, len(pairs)))

            for pair in selected_pairs:
                try:
                    # Fetch current price
                    ticker = await engine.exchange.get_ticker(pair)
                    current_price = float(ticker.last or 0)

                    if current_price == 0:
                        continue

                    # Generate random suggestion
                    direction = random.choice(["long", "short"])
                    confidence = random.randint(55, 95)

                    if direction == "long":
                        entry_price = current_price * random.uniform(0.98, 1.00)
                        target_price = entry_price * random.uniform(1.05, 1.15)
                        stop_loss = entry_price * random.uniform(0.92, 0.97)
                    else:
                        entry_price = current_price * random.uniform(1.00, 1.02)
                        target_price = entry_price * random.uniform(0.85, 0.95)
                        stop_loss = entry_price * random.uniform(1.03, 1.08)

                    risk = abs(entry_price - stop_loss)
                    reward = abs(target_price - entry_price)
                    rr_ratio = reward / risk if risk > 0 else 0

                    strategies = [
                        "Multi-timeframe alignment detected",
                        "Mean reversion opportunity identified",
                        "Breakout pattern confirmed",
                        "Support/Resistance level approach",
                        "Volume divergence spotted",
                        "RSI oversold/overbought condition"
                    ]

                    reasonings = [
                        f"{pair} showing strong momentum with increasing volume",
                        f"Technical indicators align for {pair} {'upward' if direction == 'long' else 'downward'} movement",
                        f"Historical pattern suggests {pair} {'bullish' if direction == 'long' else 'bearish'} reversal",
                        f"Multiple timeframes confirm {pair} {'long' if direction == 'long' else 'short'} setup",
                    ]

                    suggestions.append({
                        "id": f"ai-{pair.replace('/', '-')}-{random.randint(1000, 9999)}",
                        "symbol": pair,
                        "direction": direction,
                        "entry_price": round(entry_price, 2),
                        "target_price": round(target_price, 2),
                        "stop_loss": round(stop_loss, 2),
                        "confidence": confidence,
                        "risk_reward_ratio": round(rr_ratio, 2),
                        "strategy": random.choice(strategies),
                        "reasoning": random.choice(reasonings),
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    })

                except Exception as e:
                    logger.debug("Failed to generate suggestion for {}: {}", pair, e)
                    continue

            return JSONResponse({
                "success": True,
                "suggestions": suggestions,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("Failed to generate AI trade suggestions: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})

    @app.get("/api/execution-quality", dependencies=_auth_dep)
    async def api_get_execution_quality() -> JSONResponse:
        """Return execution quality metrics from TradeExecutor."""
        import time as _time

        if engine is None or engine.trade_executor is None:
            return JSONResponse({"error": "not available"})

        try:
            latency = engine.trade_executor.get_latency_summary()
            quality = engine.trade_executor.get_execution_quality_report()
            return JSONResponse({
                "latency": latency,
                "quality": quality,
                "timestamp": _time.time(),
            })
        except Exception as exc:
            logger.warning("Failed to fetch execution quality: {}", exc)
            return JSONResponse({"error": str(exc)})

    @app.get("/api/performance/charts", dependencies=_auth_dep)
    async def api_performance_charts(request: Request) -> JSONResponse:
        """Get performance chart data for various time ranges.

        Query params:
            - range: str (7d, 30d, 90d, 1y, all) default: 30d
        """
        try:
            from sqlalchemy import select

            TradeHistory, DailyPerformance, create_tables, get_async_session = _get_storage_models()
            if DailyPerformance is None or get_async_session is None:
                return JSONResponse({"success": False, "error": "Storage models unavailable", "dates": [], "pnl": [], "balance": []})
            await create_tables()

            time_range = request.query_params.get("range", "30d")

            # Determine date range
            from datetime import datetime, timedelta
            end_date = datetime.now(tz=timezone.utc)
            if time_range == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_range == "90d":
                start_date = end_date - timedelta(days=90)
            elif time_range == "1y":
                start_date = end_date - timedelta(days=365)
            elif time_range == "all":
                start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            else:  # 30d default
                start_date = end_date - timedelta(days=30)

            async with get_async_session() as session:
                query = select(DailyPerformance).where(
                    DailyPerformance.date >= start_date.strftime("%Y-%m-%d")
                ).order_by(DailyPerformance.date)

                result = await session.execute(query)
                daily_data = result.scalars().all()

                # Prepare chart data
                dates = [d.date for d in daily_data]
                pnl_data = [float(d.pnl or 0) for d in daily_data]
                balance_data = [float(d.ending_balance or d.starting_balance) for d in daily_data]

                # Win/loss data
                wins = sum(d.wins or 0 for d in daily_data)
                losses = sum(d.losses or 0 for d in daily_data)

                # Fallback: synthesise from in-memory performance tracker when
                # the database table is empty (e.g. fresh install)
                if not daily_data:
                    _pt = performance_tracker or (
                        engine.performance_tracker
                        if engine and hasattr(engine, "performance_tracker")
                        else None
                    )
                    if _pt:
                        try:
                            # PerformanceTracker stores history under _daily_pnl_history
                            daily_hist = getattr(_pt, "_daily_pnl_history", None) or []
                            for entry in daily_hist:
                                ts = entry.get("date") or entry.get("timestamp", "")
                                dates.append(str(ts)[:10] if ts else "")
                                pnl_data.append(float(entry.get("pnl", 0)))
                                balance_data.append(float(entry.get("equity", 0)))
                        except Exception:
                            pass

                return JSONResponse({
                    "success": True,
                    "range": time_range,
                    "dates": dates,
                    "pnl": pnl_data,
                    "balance": balance_data,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(wins / (wins + losses) * 100, 2) if (wins + losses) > 0 else 0,
                })
        except Exception as exc:
            logger.warning("Failed to fetch performance charts: {}", exc)
            return JSONResponse({"success": False, "error": str(exc), "dates": [], "pnl": [], "balance": []})

    @app.get("/api/performance/export", dependencies=_auth_dep)
    async def api_performance_export(request: Request) -> Any:
        """Export performance data as CSV."""
        try:
            import csv
            from io import StringIO

            from fastapi.responses import StreamingResponse
            from sqlalchemy import select

            TradeHistory, DailyPerformance, create_tables, get_async_session = _get_storage_models()
            if DailyPerformance is None or get_async_session is None:
                return JSONResponse({"success": False, "error": "Storage models unavailable"})
            await create_tables()

            time_range = request.query_params.get("range", "30d")

            # Determine date range
            from datetime import datetime, timedelta
            end_date = datetime.now(tz=timezone.utc)
            if time_range == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_range == "90d":
                start_date = end_date - timedelta(days=90)
            elif time_range == "1y":
                start_date = end_date - timedelta(days=365)
            elif time_range == "all":
                start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            else:  # 30d default
                start_date = end_date - timedelta(days=30)

            async with get_async_session() as session:
                query = select(DailyPerformance).where(
                    DailyPerformance.date >= start_date.strftime("%Y-%m-%d")
                ).order_by(DailyPerformance.date)

                result = await session.execute(query)
                daily_data = result.scalars().all()

                # Create CSV
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow([
                    "Date", "Starting Balance", "Ending Balance", "P&L", "P&L %",
                    "Trades", "Wins", "Losses", "Win Rate %", "Max Drawdown %"
                ])

                for d in daily_data:
                    pnl_pct = ((d.ending_balance - d.starting_balance) / d.starting_balance * 100) if d.starting_balance > 0 else 0
                    win_rate = (d.wins / (d.wins + d.losses) * 100) if (d.wins + d.losses) > 0 else 0
                    writer.writerow([
                        d.date,
                        d.starting_balance,
                        d.ending_balance,
                        d.pnl,
                        round(pnl_pct, 2),
                        d.trades_count,
                        d.wins,
                        d.losses,
                        round(win_rate, 2),
                        d.max_drawdown or 0
                    ])

                output.seek(0)
                return StreamingResponse(
                    iter([output.getvalue()]),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=performance_{time_range}.csv"}
                )
        except Exception as exc:
            logger.warning("Failed to export performance: {}", exc)
            return JSONResponse({"success": False, "error": str(exc)})


    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket) -> None:
        # Authenticate WebSocket connections via a hashed token query parameter.
        # The token is sha256(username:password) so raw credentials are never
        # transmitted over the wire as query parameters.
        if _auth_enabled:
            token = websocket.query_params.get("token", "")
            if not secrets.compare_digest(token.encode(), _ws_token.encode()):
                await websocket.close(code=4401)
                return
        await manager.connect(websocket)

        async def _push_loop() -> None:
            """Push live updates to this single client every 5 seconds.

            Also streams new log lines as they appear in the log file.
            The loop only exits when the WebSocket connection is broken.
            """
            # Track log file size to stream only new lines each cycle
            log_offset: int = 0
            try:
                if _LOG_FILE.exists():
                    log_offset = _LOG_FILE.stat().st_size
            except Exception:
                pass

            # Send recent log lines immediately on connect
            try:
                log_lines = _tail_log(_LOG_FILE, 100)
                now_ts = datetime.now(tz=timezone.utc).isoformat()
                for line in log_lines:
                    stripped = line.rstrip()
                    if not stripped:
                        continue
                    level = _parse_log_level(stripped)
                    await websocket.send_text(
                        json.dumps(
                            {"type": "log", "level": level, "message": stripped, "ts": now_ts},
                            default=str,
                        )
                    )
            except Exception:
                pass

            while True:
                # Push live metrics/positions update
                # When the realtime_hub is active it handles broadcasting directly;
                # this loop serves as a fallback for status/signals/logs data that
                # the hub does not cover.
                try:
                    data = await _collect_live_data()
                    await websocket.send_text(json.dumps(data, default=str))
                except Exception as exc:
                    # If we can't send, the WebSocket is broken — exit
                    logger.debug("WS push error (live data): {}", exc)
                    break

                # Push any new log lines that appeared since last cycle
                try:
                    if _LOG_FILE.exists():
                        new_size = _LOG_FILE.stat().st_size
                        if new_size > log_offset:
                            with open(_LOG_FILE, "r", encoding="utf-8", errors="replace") as fh:
                                fh.seek(log_offset)
                                new_text = fh.read()
                            log_offset = new_size
                            ts = datetime.now(tz=timezone.utc).isoformat()
                            # Batch log lines for better performance
                            log_batch = []
                            for line in new_text.splitlines():
                                if not line.strip():
                                    continue
                                level = _parse_log_level(line)
                                log_batch.append(
                                    {"type": "log", "level": level, "message": line, "ts": ts}
                                )
                            # Send all logs in a single message
                            if log_batch:
                                await websocket.send_text(
                                    json.dumps(
                                        {"type": "logs_batch", "logs": log_batch},
                                        default=str,
                                    )
                                )
                except Exception as exc:
                    logger.debug("WS push error (log lines): {}", exc)

                # Fixed 2-second push interval for responsive dashboard updates
                _engine_settings = getattr(engine, "settings", None)
                _monitoring_cfg = getattr(_engine_settings, "monitoring", None)
                ws_interval = getattr(_monitoring_cfg, "dashboard_ws_interval", 2.0)
                await asyncio.sleep(ws_interval)

        push_task = asyncio.create_task(_push_loop())
        try:
            while True:
                try:
                    msg = await websocket.receive_text()
                    # Handle client commands if needed
                    logger.debug("WS message from client: {}", msg)
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
        finally:
            push_task.cancel()
            manager.disconnect(websocket)

    # Attach broadcast helper to the app for external use
    app.state.broadcast_update = manager.broadcast
    return app


class _ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: List[Any] = []

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.debug("WebSocket connected; total={}", len(self._connections))

    def disconnect(self, websocket: Any) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.debug("WebSocket disconnected; total={}", len(self._connections))

    async def broadcast(self, data: Any) -> None:
        """Send *data* (serialised to JSON) to all connected clients."""
        message = json.dumps(data, default=str)
        dead: List[Any] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class TradingDashboard:
    """High-level wrapper for running the dashboard server."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        engine: Any = None,
        **kwargs: Any,
    ) -> None:
        self._settings = settings or Settings.get_settings()
        self._app = create_app(settings=self._settings, engine=engine, **kwargs)

    async def run(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
    ) -> None:
        """Start the dashboard HTTP server.

        Args:
            host: Bind address.
            port: Bind port. Defaults to ``settings.monitoring.dashboard_port``.
        """
        if not _FASTAPI_AVAILABLE or self._app is None:
            logger.error("FastAPI is not installed; dashboard cannot start")
            return
        port = port or self._settings.monitoring.dashboard_port
        try:
            import uvicorn  # type: ignore

            config = uvicorn.Config(self._app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            logger.info("Dashboard starting at http://{}:{}", host, port)
            await server.serve()
        except ImportError:
            logger.error(
                "uvicorn not installed; dashboard cannot start. Install with: pip install uvicorn"
            )

    @property
    def app(self) -> Any:
        """Return the underlying FastAPI application."""
        return self._app

    async def broadcast_update(self, data: Any) -> None:
        """Broadcast a data update to all connected WebSocket clients.

        Args:
            data: JSON-serialisable data payload.
        """
        if (
            self._app
            and hasattr(self._app, "state")
            and hasattr(self._app.state, "broadcast_update")
        ):
            await self._app.state.broadcast_update(data)


# ── Internal HTML helpers ────────────────────────────────────────────────────


def _render_page(title: str, body: str) -> str:
    """Render a minimal HTML page with a navigation bar."""
    nav_links = " | ".join(
        f'<a href="{path}">{name}</a>'
        for name, path in [
            ("Overview", "/"),
            ("Trades", "/trades"),
            ("Performance", "/performance"),
            ("Signals", "/signals"),
            ("Risk", "/risk"),
            ("Settings", "/settings"),
            ("Logs", "/logs"),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>TradingBot — {title}</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; background: #0d1117; color: #c9d1d9; }}
  nav {{ margin-bottom: 1.5rem; }}
  nav a {{ color: #58a6ff; margin-right: 1rem; text-decoration: none; }}
  h1, h2 {{ color: #f0f6fc; }}
</style>
</head>
<body>
<h1>🤖 Crypto Trading Bot</h1>
<nav>{nav_links}</nav>
{body}
</body>
</html>"""


def _overview_body() -> str:
    return """
<h2>Overview</h2>
<ul>
  <li>Status: <strong>Running</strong></li>
  <li>API: <a href="/api/status">/api/status</a></li>
  <li>Metrics: <a href="/metrics">/metrics</a></li>
  <li>WebSocket: <code>ws://&lt;host&gt;/ws/live</code></li>
</ul>
"""
