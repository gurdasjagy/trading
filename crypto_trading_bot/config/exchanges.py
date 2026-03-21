"""Exchange-specific API and WebSocket endpoint configurations."""

from __future__ import annotations

from typing import Dict, Optional, Type

from pydantic import BaseModel


class ExchangeAPIConfig(BaseModel):
    """Base model for exchange API credentials and limits."""

    api_key: str = ""
    secret_key: str = ""
    passphrase: Optional[str] = None
    testnet: bool = False
    rate_limit_requests_per_second: int = 10


class MEXCConfig(ExchangeAPIConfig):
    """MEXC-specific endpoints."""

    base_url: str = "https://api.mexc.com"
    futures_base_url: str = "https://contract.mexc.com"
    ws_url: str = "wss://wbs.mexc.com/ws"
    ws_futures_url: str = "wss://contract.mexc.com/ws"


class GateIOConfig(ExchangeAPIConfig):
    """Gate.io-specific endpoints."""

    base_url: str = "https://api.gateio.ws"
    ws_url: str = "wss://api.gateio.ws/ws/v4/"


class BingXConfig(ExchangeAPIConfig):
    """BingX-specific endpoints."""

    base_url: str = "https://open-api.bingx.com"
    ws_url: str = "wss://open-api-ws.bingx.com/market"


class BitgetConfig(ExchangeAPIConfig):
    """Bitget-specific endpoints."""

    base_url: str = "https://api.bitget.com"
    ws_url: str = "wss://ws.bitget.com/v2/ws/public"


# Maps exchange name → config class for dynamic instantiation.
SUPPORTED_EXCHANGES: Dict[str, Type[ExchangeAPIConfig]] = {
    "mexc": MEXCConfig,
    "gateio": GateIOConfig,
    "bingx": BingXConfig,
    "bitget": BitgetConfig,
}


def get_exchange_config(exchange_name: str, **kwargs) -> ExchangeAPIConfig:
    """Instantiate and return the config for the named exchange.

    Args:
        exchange_name: One of the keys in ``SUPPORTED_EXCHANGES``.
        **kwargs: Field overrides forwarded to the config constructor.

    Returns:
        A populated exchange config object.

    Raises:
        ValueError: If *exchange_name* is not in ``SUPPORTED_EXCHANGES``.
    """
    name = exchange_name.lower()
    if name not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Unsupported exchange '{exchange_name}'. " f"Choose from: {list(SUPPORTED_EXCHANGES)}"
        )
    config_cls = SUPPORTED_EXCHANGES[name]
    return config_cls(**kwargs)
