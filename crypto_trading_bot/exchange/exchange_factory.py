"""Exchange factory — creates the appropriate exchange instance from settings.

Usage::

    from exchange.exchange_factory import create_exchange
    from config.settings import Settings

    settings = Settings()
    exchange = await create_exchange(settings)   # PaperExchange or CcxtExchange
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .base_exchange import BaseExchange
from .ccxt_exchange import SUPPORTED_EXCHANGES, CcxtExchange
from .exness_client import ExnessForexClient
from .exness_paper_exchange import ExnessPaperExchange
from .forex_paper_exchange import ForexPaperExchange
from .gateio_client import GateIOClient
from .gateio_tradfi_client import GateIOTradFiClient
from .paper_exchange import PaperExchange

# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_exchange(settings: Any) -> BaseExchange:
    """Return a configured exchange instance based on *settings*.

    * ``TRADING_MODE=paper``                   → :class:`~.paper_exchange.PaperExchange`
    * ``TRADING_MODE=live``                    → :class:`~.ccxt_exchange.CcxtExchange`
    * ``TRADING_MODE=testnet``                 → :class:`~.ccxt_exchange.CcxtExchange` (testnet)
    * ``TRADING_MODE=forex_live``              → :class:`~.gateio_tradfi_client.GateIOTradFiClient`
    * ``TRADING_MODE=forex_demo``              → :class:`~.gateio_tradfi_client.GateIOTradFiClient` (testnet)
    * ``TRADING_MODE=forex_paper``             → :class:`~.forex_paper_exchange.ForexPaperExchange`
    * ``TRADING_MODE=forex_exness_live``       → :class:`~.exness_client.ExnessForexClient` (DEPRECATED)
    * ``TRADING_MODE=forex_exness_demo``       → :class:`~.exness_client.ExnessForexClient` (demo, DEPRECATED)
    * ``TRADING_MODE=forex_exness_paper``      → :class:`~.exness_paper_exchange.ExnessPaperExchange` (DEPRECATED)

    **NOTE**: ``forex_exness_*`` modes are deprecated in favor of Gate.io TradFi.
    Exness MT5 is heavily restricted in India. Use ``forex_live`` (Gate.io TradFi REST) instead.

    For **live / testnet mode** the following settings are required:

    * ``PRIMARY_EXCHANGE`` (or ``settings.exchange.primary_exchange``) — one of
      ``mexc``, ``gateio``, ``bingx``, ``bitget``.
    * A valid API key/secret for the chosen exchange.  Credentials are read in
      this priority order:

      1. Generic ``EXCHANGE_API_KEY`` / ``EXCHANGE_SECRET`` / ``EXCHANGE_PASSPHRASE``.
      2. Exchange-specific ``{NAME}_API_KEY`` / ``{NAME}_SECRET_KEY`` /
         ``{NAME}_PASSPHRASE``.

    For **forex_live / forex_demo** modes (Gate.io TradFi REST):

    * ``GATEIO_API_KEY`` / ``GATEIO_SECRET_KEY`` (or generic ``EXCHANGE_API_KEY`` / ``EXCHANGE_SECRET``)

    For **forex_exness_live / forex_exness_demo / forex_exness_paper** modes (DEPRECATED):

    * ``EXNESS_LOGIN`` / ``EXNESS_PASSWORD`` / ``EXNESS_SERVER`` / ``EXNESS_ACCOUNT_TYPE``

    Args:
        settings: A :class:`~config.settings.Settings` instance.

    Returns:
        An uninitialised :class:`~.base_exchange.BaseExchange`.  Call
        ``await exchange.connect()`` before using it.

    Raises:
        ValueError: When live mode is requested but API credentials are missing,
            or an unsupported exchange name is given.
    """
    mode: str = getattr(settings, "trading_mode", "paper").lower()
    exchange_id: str = _get_primary_exchange(settings)

    if mode == "paper":
        logger.info(
            "Exchange factory: paper mode — creating PaperExchange (exchange={})", exchange_id
        )
        price_feed = _build_price_feed(settings, exchange_id)
        balance = float(getattr(settings, "paper_trading_balance", 10_000.0))
        return PaperExchange(
            starting_balance=balance,
            price_exchange=price_feed,
        )

    if mode in ("live", "testnet"):
        is_testnet = mode == "testnet" or getattr(
            getattr(settings, "exchange", None), "use_testnet", False
        )

        # Strict validation for testnet mode
        if is_testnet and mode == "testnet":
            # Check for Gate.io testnet keys specifically
            if exchange_id == "gateio":
                tn_key = getattr(settings, "gateio_testnet_api_key", None) or ""
                tn_secret = getattr(settings, "gateio_testnet_secret_key", None) or ""
                if not tn_key or not tn_secret:
                    raise ValueError(
                        "Testnet mode is enabled but GATEIO_TESTNET_API_KEY or "
                        "GATEIO_TESTNET_SECRET_KEY is missing. "
                        "Did you forget to remove the '#' in your .env file?"
                    )
            # Check for generic testnet keys
            else:
                tn_key = getattr(settings, "exchange_testnet_api_key", None) or ""
                tn_secret = getattr(settings, "exchange_testnet_api_secret", None) or ""
                if not tn_key or not tn_secret:
                    raise ValueError(
                        "Testnet mode is enabled but EXCHANGE_TESTNET_API_KEY or "
                        "EXCHANGE_TESTNET_API_SECRET is missing. "
                        "Did you forget to remove the '#' in your .env file?"
                    )

        logger.info(
            "Exchange factory: {} mode — creating {} (exchange={}, testnet={})",
            mode,
            "GateIOClient" if exchange_id == "gateio" else "CcxtExchange",
            exchange_id,
            is_testnet,
        )
        api_key, secret, passphrase = _resolve_credentials(settings, exchange_id, is_testnet)
        if exchange_id == "gateio":
            return GateIOClient(
                api_key=api_key,
                secret_key=secret,
                passphrase=passphrase,
                testnet=is_testnet,
            )
        else:
            return CcxtExchange(
                exchange_id=exchange_id,
                api_key=api_key,
                secret_key=secret,
                passphrase=passphrase,
                testnet=is_testnet,
            )

    if mode in ("forex_live", "forex_demo"):
        # Default forex mode uses Gate.io TradFi REST API
        is_demo = mode == "forex_demo"
        logger.info(
            "Exchange factory: {} mode — creating GateIOTradFiClient (testnet={})",
            mode,
            is_demo,
        )
        api_key, secret, _ = _resolve_gateio_credentials(settings, is_demo)
        return GateIOTradFiClient(
            api_key=api_key,
            secret_key=secret,
            testnet=is_demo,
        )

    if mode == "forex_paper":
        logger.info("Exchange factory: forex_paper mode — creating ForexPaperExchange")
        price_feed = _build_price_feed(settings, exchange_id)
        balance = float(getattr(settings, "paper_trading_balance", 10_000.0))
        return ForexPaperExchange(
            starting_balance=balance,
            price_exchange=price_feed,
        )

    if mode in ("forex_exness_live", "forex_exness_demo"):
        # DEPRECATED: Exness MT5 is heavily restricted in India
        logger.warning(
            "⚠️  forex_exness_* modes are DEPRECATED. Exness MT5 is restricted in India. "
            "Use 'forex_live' (Gate.io TradFi) instead for better availability."
        )
        is_demo = mode == "forex_exness_demo"
        logger.info(
            "Exchange factory: {} mode — creating ExnessForexClient (testnet={})",
            mode,
            is_demo,
        )
        login, password, server, account_type = _resolve_exness_credentials(settings, is_demo)
        return ExnessForexClient(
            login=login,
            password=password,
            server=server,
            account_type=account_type,
            testnet=is_demo,
        )

    if mode == "forex_exness_paper":
        # DEPRECATED: Exness paper trading
        logger.warning(
            "⚠️  forex_exness_paper mode is DEPRECATED. "
            "Use 'paper' mode with Gate.io price feed instead."
        )
        logger.info("Exchange factory: forex_exness_paper mode — creating ExnessPaperExchange")
        price_feed = _build_price_feed(settings, exchange_id)
        balance = float(getattr(settings, "paper_trading_balance", 10_000.0))
        return ExnessPaperExchange(
            starting_balance=balance,
            price_exchange=price_feed,
        )

    raise ValueError(
        f"Unsupported trading mode: {mode!r}. "
        "Use 'paper', 'live', 'testnet', 'forex_live', 'forex_demo', 'forex_paper', "
        "'forex_exness_live' (deprecated), "
        "'forex_exness_demo' (deprecated), or 'forex_exness_paper' (deprecated)."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_primary_exchange(settings: Any) -> str:
    """Resolve the primary exchange name from settings."""
    # Prefer top-level settings.primary_exchange (maps to PRIMARY_EXCHANGE env var)
    name: str = getattr(settings, "primary_exchange", None) or ""
    if not name:
        # Fall back to nested ExchangeConfig
        exchange_cfg = getattr(settings, "exchange", None)
        name = getattr(exchange_cfg, "primary_exchange", "mexc") or "mexc"
    name = name.lower().strip()
    if name not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Unsupported exchange: {name!r}. Supported: {SUPPORTED_EXCHANGES}")
    return name


def _resolve_credentials(
    settings: Any, exchange_id: str, testnet: bool = False
) -> tuple[str, str, str]:
    """Return (api_key, secret, passphrase) for *exchange_id*.

    When *testnet* is ``True``, this method first checks for generic
    ``EXCHANGE_TESTNET_API_KEY`` / ``EXCHANGE_TESTNET_API_SECRET`` /
    ``EXCHANGE_TESTNET_PASSWORD`` credentials, then falls back to
    exchange-specific testnet keys (e.g. ``GATEIO_TESTNET_API_KEY``), and
    finally falls back to the standard credential lookup.

    Raises:
        ValueError: If api_key or secret is missing in live/testnet mode.
    """
    # 0. Generic testnet credentials — highest priority when testnet mode is active
    if testnet:
        tn_key = getattr(settings, "exchange_testnet_api_key", None) or ""
        tn_secret = getattr(settings, "exchange_testnet_api_secret", None) or ""
        tn_pass = getattr(settings, "exchange_testnet_password", None) or ""
        if tn_key and tn_secret:
            logger.debug(
                "Credentials resolved for testnet exchange={} via EXCHANGE_TESTNET_* (key={}...)",
                exchange_id, tn_key[:4],
            )
            return tn_key, tn_secret, tn_pass

    # Testnet-specific credentials (Gate.io sandbox)
    if testnet and exchange_id == "gateio":
        tn_key = getattr(settings, "gateio_testnet_api_key", None) or ""
        tn_secret = getattr(settings, "gateio_testnet_secret_key", None) or ""
        if tn_key and tn_secret:
            logger.debug(
                "Credentials resolved for testnet exchange={} (key={}...)",
                exchange_id, tn_key[:4],
            )
            return tn_key, tn_secret, ""

    # 1. Generic credentials take highest priority
    api_key: str = getattr(settings, "exchange_api_key", None) or ""
    secret: str = getattr(settings, "exchange_secret", None) or ""
    passphrase: str = getattr(settings, "exchange_passphrase", None) or ""

    # 2. Fall back to exchange-specific credentials
    if not api_key:
        api_key = getattr(settings, f"{exchange_id}_api_key", None) or ""
    if not secret:
        secret = getattr(settings, f"{exchange_id}_secret_key", None) or ""
    if not passphrase and exchange_id == "bitget":
        passphrase = getattr(settings, "bitget_passphrase", None) or ""

    if not api_key:
        raise ValueError(
            f"Live trading requires an API key for {exchange_id!r}. "
            f"Set EXCHANGE_API_KEY or {exchange_id.upper()}_API_KEY in your environment."
        )
    if not secret:
        raise ValueError(
            f"Live trading requires an API secret for {exchange_id!r}. "
            f"Set EXCHANGE_SECRET or {exchange_id.upper()}_SECRET_KEY in your environment."
        )

    logger.debug("Credentials resolved for exchange={} (key={}...)", exchange_id, api_key[:4])
    return api_key, secret, passphrase


def _build_price_feed(settings: Any, exchange_id: str) -> CcxtExchange | GateIOClient | None:
    """Build a read-only CCXT price-feed client for paper trading.

    Returns *None* if no API credentials are available (public endpoints only
    are still usable for price data on most exchanges, but ``ccxt`` requires at
    least empty strings as credentials).
    """
    testnet: bool = getattr(getattr(settings, "exchange", None), "use_testnet", False)
    try:
        api_key, secret, passphrase = _resolve_credentials(settings, exchange_id, testnet)
    except ValueError:
        # No credentials available — construct with empty strings so public
        # endpoints (tickers, OHLCV) still work.
        logger.info(
            "No API credentials found for {}; price feed will use public endpoints only",
            exchange_id,
        )
        api_key, secret, passphrase = "", "", ""

    if exchange_id == "gateio":
        return GateIOClient(
            api_key=api_key,
            secret_key=secret,
            passphrase=passphrase,
            testnet=testnet,
        )
    else:
        return CcxtExchange(
            exchange_id=exchange_id,
            api_key=api_key,
            secret_key=secret,
            passphrase=passphrase,
            testnet=testnet,
        )


def _resolve_gateio_credentials(
    settings: Any, testnet: bool = False
) -> tuple[str, str, str]:
    """Return (api_key, secret, passphrase) for Gate.io TradFi forex trading.

    Credential lookup order:

    1. ``GATEIO_TESTNET_API_KEY`` / ``GATEIO_TESTNET_SECRET_KEY`` (when *testnet* is ``True``).
    2. Generic ``EXCHANGE_API_KEY`` / ``EXCHANGE_SECRET``.
    3. ``GATEIO_API_KEY`` / ``GATEIO_SECRET_KEY``.

    Raises:
        ValueError: If api_key or secret is missing.
    """
    # 1. Gate.io testnet credentials
    if testnet:
        tn_key = getattr(settings, "gateio_testnet_api_key", None) or ""
        tn_secret = getattr(settings, "gateio_testnet_secret_key", None) or ""
        if tn_key and tn_secret:
            logger.debug(
                "Gate.io TradFi credentials resolved via GATEIO_TESTNET_* (key={}...)", tn_key[:4]
            )
            return tn_key, tn_secret, ""

    # 2. Generic exchange credentials
    api_key: str = getattr(settings, "exchange_api_key", None) or ""
    secret: str = getattr(settings, "exchange_secret", None) or ""

    # 3. Gate.io-specific credentials
    if not api_key:
        api_key = getattr(settings, "gateio_api_key", None) or ""
    if not secret:
        secret = getattr(settings, "gateio_secret_key", None) or ""

    if not api_key:
        raise ValueError(
            "Gate.io TradFi forex trading requires an API key. "
            "Set GATEIO_API_KEY or EXCHANGE_API_KEY in your environment."
        )
    if not secret:
        raise ValueError(
            "Gate.io TradFi forex trading requires an API secret. "
            "Set GATEIO_SECRET_KEY or EXCHANGE_SECRET in your environment."
        )

    logger.debug("Gate.io TradFi credentials resolved (key={}...)", api_key[:4])
    return api_key, secret, ""


def _resolve_exness_credentials(
    settings: Any, testnet: bool = False
) -> tuple[str, str, str, str]:
    """Return (login, password, server, account_type) for Exness forex trading.

    Credential lookup order:

    1. ``EXNESS_DEMO_LOGIN`` / ``EXNESS_DEMO_PASSWORD`` (when *testnet* is ``True``).
    2. ``EXNESS_LOGIN`` / ``EXNESS_PASSWORD`` / ``EXNESS_SERVER`` / ``EXNESS_ACCOUNT_TYPE``.

    Raises:
        ValueError: If login or password is missing.
    """
    # 1. Exness demo credentials
    if testnet:
        login = getattr(settings, "exness_demo_login", None) or ""
        password = getattr(settings, "exness_demo_password", None) or ""
        server = getattr(settings, "exness_demo_server", "Exness-MT5Demo")
        account_type = getattr(settings, "exness_demo_account_type", "Raw Spread")
        if login and password:
            logger.debug("Exness demo credentials resolved (login={}...)", login[:4])
            return login, password, server, account_type

    # 2. Exness live credentials
    login = getattr(settings, "exness_login", None) or ""
    password = getattr(settings, "exness_password", None) or ""
    server = getattr(settings, "exness_server", "Exness-MT5Real")
    account_type = getattr(settings, "exness_account_type", "Raw Spread")

    if not login:
        raise ValueError(
            "Exness forex trading requires a login. "
            "Set EXNESS_LOGIN or EXNESS_DEMO_LOGIN in your environment."
        )
    if not password:
        raise ValueError(
            "Exness forex trading requires a password. "
            "Set EXNESS_PASSWORD or EXNESS_DEMO_PASSWORD in your environment."
        )

    logger.debug("Exness credentials resolved (login={}...)", login[:4])
    return login, password, server, account_type


def _resolve_gateio_tradfi_mt5_credentials(
    settings: Any, testnet: bool = False
) -> tuple[str, str, str, str]:
    """Return (login, password, server, account_type) for Gate.io TradFi MT5 trading.

    Credential lookup order:

    1. ``GATEIO_TRADFI_DEMO_LOGIN`` / ``GATEIO_TRADFI_DEMO_PASSWORD`` (when *testnet* is ``True``).
    2. ``GATEIO_TRADFI_LOGIN`` / ``GATEIO_TRADFI_PASSWORD`` / ``GATEIO_TRADFI_SERVER`` / ``GATEIO_TRADFI_ACCOUNT_TYPE``.

    Raises:
        ValueError: If login or password is missing.
    """
    # 1. Gate.io TradFi demo credentials
    if testnet:
        login = getattr(settings, "gateio_tradfi_demo_login", None) or ""
        password = getattr(settings, "gateio_tradfi_demo_password", None) or ""
        server = getattr(settings, "gateio_tradfi_demo_server", "GateIO-TradFi-Demo")
        account_type = getattr(settings, "gateio_tradfi_demo_account_type", "Standard")
        if login and password:
            logger.debug("Gate.io TradFi demo MT5 credentials resolved (login={}...)", login[:4])
            return login, password, server, account_type

    # 2. Gate.io TradFi live credentials
    login = getattr(settings, "gateio_tradfi_login", None) or ""
    password = getattr(settings, "gateio_tradfi_password", None) or ""
    server = getattr(settings, "gateio_tradfi_server", "GateIO-TradFi")
    account_type = getattr(settings, "gateio_tradfi_account_type", "Standard")

    if not login:
        raise ValueError(
            "Gate.io TradFi MT5 trading requires a login. "
            "Set GATEIO_TRADFI_LOGIN or GATEIO_TRADFI_DEMO_LOGIN in your environment. "
            "Get your MT5 account at: https://www.gate.io/tradfi"
        )
    if not password:
        raise ValueError(
            "Gate.io TradFi MT5 trading requires a password. "
            "Set GATEIO_TRADFI_PASSWORD or GATEIO_TRADFI_DEMO_PASSWORD in your environment."
        )

    logger.debug("Gate.io TradFi MT5 credentials resolved (login={}..., server={})", login[:4], server)
    return login, password, server, account_type
