"""Bybit TradFi forex client for XAU/USD and XAG/USD trading.

Extends :class:`~.ccxt_exchange.CcxtExchange` with forex-specific helpers:

* Lot-size calculation from USDT margin amount.
* Pip-value calculation for position sizing.
* Margin calculation.
* Spread monitoring (bid-ask spread expressed in pips).
* Forex market order placement with optional stop-loss / take-profit in pips.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base_exchange import OrderSide, Ticker
from .ccxt_exchange import CcxtExchange


class BybitForexClient(CcxtExchange):
    """Bybit TradFi forex client for XAU/USD, XAG/USD trading.

    Bybit supports precious-metal forex pairs (XAU/USD, XAG/USD) through
    their TradFi product line.  This client adds forex-specific helpers on
    top of the generic :class:`~.ccxt_exchange.CcxtExchange` implementation.

    Args:
        api_key: Bybit API key.
        secret_key: Bybit API secret.
        passphrase: Unused for Bybit; kept for interface consistency.
        testnet: When *True*, routes requests to the Bybit testnet (sandbox).
    """

    EXCHANGE_NAME = "bybit"

    # ------------------------------------------------------------------
    # Forex pair configurations
    # ------------------------------------------------------------------

    #: Per-pair metadata used by the forex helper methods.
    FOREX_PAIRS: Dict[str, Dict[str, Any]] = {
        "XAU/USD": {
            "contract_size": 1,       # 1 oz per lot unit
            "pip_size": 0.01,         # 1 pip = $0.01 for gold
            "min_lot": 0.01,
            "max_lot": 100.0,
            "pip_value_per_lot": 0.01,
        },
        "XAG/USD": {
            "contract_size": 5000,    # 5000 oz per lot unit
            "pip_size": 0.001,        # 1 pip = $0.001 for silver
            "min_lot": 0.01,
            "max_lot": 100.0,
            "pip_value_per_lot": 0.05,
        },
    }

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str = "",
        testnet: bool = False,
    ) -> None:
        super().__init__("bybit", api_key, secret_key, passphrase, testnet)
        # Override to linear sub-type for TradFi forex instruments
        self._default_options = {"defaultType": "swap", "defaultSubType": "linear"}

    # ------------------------------------------------------------------
    # Forex helpers
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        symbol: str,
        usdt_amount: float,
        leverage: int,
        current_price: float,
    ) -> float:
        """Calculate the number of lots from a USDT margin amount.

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.
            usdt_amount: Available margin in USDT.
            leverage: Leverage multiplier to apply.
            current_price: Current market price of the pair.

        Returns:
            Lot size rounded to the minimum lot increment, clamped to
            the pair's allowed lot-size range.

        Raises:
            ValueError: If *symbol* is not in :attr:`FOREX_PAIRS`.
        """
        config = self._get_pair_config(symbol)
        margin_per_lot = config["contract_size"] * current_price / leverage
        raw_lots = usdt_amount / margin_per_lot
        min_lot = config["min_lot"]
        # Round to the nearest min_lot increment
        lots = round(raw_lots / min_lot) * min_lot
        lots = max(min_lot, lots)
        lots = min(lots, config["max_lot"])
        return lots

    def calculate_pip_value(self, symbol: str, lot_size: float) -> float:
        """Return the monetary value of a single pip for *lot_size* lots.

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.
            lot_size: Number of lots held.

        Returns:
            Pip value in USD.

        Raises:
            ValueError: If *symbol* is not in :attr:`FOREX_PAIRS`.
        """
        config = self._get_pair_config(symbol)
        return lot_size * config["pip_value_per_lot"] * config["contract_size"]

    def calculate_margin_required(
        self,
        symbol: str,
        lot_size: float,
        price: float,
        leverage: int,
    ) -> float:
        """Return the margin required (in USD) to hold *lot_size* lots.

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.
            lot_size: Position size in lots.
            price: Current market price of the pair.
            leverage: Leverage multiplier applied to the position.

        Returns:
            Required margin in USD.

        Raises:
            ValueError: If *symbol* is not in :attr:`FOREX_PAIRS`.
        """
        config = self._get_pair_config(symbol)
        return (lot_size * config["contract_size"] * price) / leverage

    async def get_spread(self, symbol: str) -> Dict[str, float]:
        """Return the current bid-ask spread expressed in pips.

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.

        Returns:
            A dict with keys ``"spread_pips"``, ``"bid"``, and ``"ask"``.

        Raises:
            ValueError: If *symbol* is not in :attr:`FOREX_PAIRS`.
        """
        config = self._get_pair_config(symbol)
        ticker: Ticker = await self.get_ticker(symbol)
        spread_pips = (ticker.ask - ticker.bid) / config["pip_size"]
        return {"spread_pips": spread_pips, "bid": ticker.bid, "ask": ticker.ask}

    async def create_forex_order(
        self,
        symbol: str,
        side: OrderSide,
        lot_size: float,
        leverage: int,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a forex market order with optional stop-loss / take-profit.

        Stop-loss and take-profit distances are specified in *pips* and
        converted to absolute prices using the current market mid-price.

        Args:
            symbol: Forex pair, e.g. ``"XAU/USD"``.
            side: :attr:`~.base_exchange.OrderSide.BUY` or
                :attr:`~.base_exchange.OrderSide.SELL`.
            lot_size: Position size in lots.
            leverage: Leverage multiplier for the trade.
            sl_pips: Stop-loss distance in pips (positive number).
            tp_pips: Take-profit distance in pips (positive number).

        Returns:
            The normalised order dict returned by
            :meth:`~.ccxt_exchange.CcxtExchange.create_market_order`.

        Raises:
            ValueError: If *symbol* is not in :attr:`FOREX_PAIRS`.
        """
        config = self._get_pair_config(symbol)
        amount = lot_size * config["contract_size"]

        await self.set_leverage(symbol, leverage)

        ticker: Ticker = await self.get_ticker(symbol)
        params: Dict[str, Any] = {"leverage": leverage}

        pip = config["pip_size"]
        if sl_pips is not None:
            if side == OrderSide.BUY:
                params["stopLoss"] = ticker.last - (sl_pips * pip)
            else:
                params["stopLoss"] = ticker.last + (sl_pips * pip)

        if tp_pips is not None:
            if side == OrderSide.BUY:
                params["takeProfit"] = ticker.last + (tp_pips * pip)
            else:
                params["takeProfit"] = ticker.last - (tp_pips * pip)

        return await self.create_market_order(symbol, side, amount, params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_pair_config(self, symbol: str) -> Dict[str, Any]:
        """Return the pair config for *symbol* or raise :exc:`ValueError`."""
        config = self.FOREX_PAIRS.get(symbol)
        if config is None:
            raise ValueError(
                f"Unknown forex pair: {symbol!r}. "
                f"Supported pairs: {list(self.FOREX_PAIRS.keys())}"
            )
        return config
