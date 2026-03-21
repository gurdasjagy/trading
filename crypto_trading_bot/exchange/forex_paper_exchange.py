"""Forex paper exchange — extends PaperExchange with forex-specific features.

Provides lot-based position sizing, pip calculations, and realistic
spread simulation for forex and precious metals paper trading.
"""

from __future__ import annotations

from typing import Dict

from .paper_exchange import PaperExchange


class ForexPaperExchange(PaperExchange):
    """Paper exchange with forex-specific features.

    Extends :class:`~.paper_exchange.PaperExchange` with:

    * Lot-size based position sizing (micro/mini/standard lots).
    * Pip-based P&L calculation.
    * Spread simulation using realistic bid/ask spreads.
    * Symbol normalization for forex pairs (XAU_USDT → XAU/USDT etc.).
    """

    EXCHANGE_NAME = "forex_paper"

    # Pip sizes per symbol (Gate.io TradFi underscore format or slash format)
    FOREX_SPECS: Dict[str, Dict[str, float]] = {
        "XAU_USDT":  {"pip_size": 0.01,   "contract_size": 1.0,      "typical_spread_pips": 3.0},
        "XAU/USDT":  {"pip_size": 0.01,   "contract_size": 1.0,      "typical_spread_pips": 3.0},
        "XAU/USD":   {"pip_size": 0.01,   "contract_size": 1.0,      "typical_spread_pips": 3.0},
        "XAUUSD":    {"pip_size": 0.01,   "contract_size": 1.0,      "typical_spread_pips": 3.0},
        "XAG_USDT":  {"pip_size": 0.001,  "contract_size": 5000.0,   "typical_spread_pips": 5.0},
        "XAG/USDT":  {"pip_size": 0.001,  "contract_size": 5000.0,   "typical_spread_pips": 5.0},
        "XAG/USD":   {"pip_size": 0.001,  "contract_size": 5000.0,   "typical_spread_pips": 5.0},
        "EUR_USDT":  {"pip_size": 0.0001, "contract_size": 100000.0, "typical_spread_pips": 0.5},
        "GBP_USDT":  {"pip_size": 0.0001, "contract_size": 100000.0, "typical_spread_pips": 0.8},
        "JPY_USDT":  {"pip_size": 0.01,   "contract_size": 100000.0, "typical_spread_pips": 0.7},
        "AUD_USDT":  {"pip_size": 0.0001, "contract_size": 100000.0, "typical_spread_pips": 0.7},
        "CAD_USDT":  {"pip_size": 0.0001, "contract_size": 100000.0, "typical_spread_pips": 0.9},
        "CHF_USDT":  {"pip_size": 0.0001, "contract_size": 100000.0, "typical_spread_pips": 0.9},
        "NZD_USDT":  {"pip_size": 0.0001, "contract_size": 100000.0, "typical_spread_pips": 1.0},
    }

    @property
    def name(self) -> str:
        return "Forex Paper"

    def _get_forex_spec(self, symbol: str) -> Dict[str, float]:
        """Return forex spec for *symbol*, falling back to gold defaults."""
        return self.FOREX_SPECS.get(symbol, {
            "pip_size": 0.01,
            "contract_size": 1.0,
            "typical_spread_pips": 5.0,
        })

    def pips_to_price(self, symbol: str, pips: float) -> float:
        """Convert a pip count to a price movement for *symbol*.

        Args:
            symbol: Trading symbol (e.g. ``"XAU_USDT"``).
            pips: Number of pips.

        Returns:
            Price movement in price units.
        """
        spec = self._get_forex_spec(symbol)
        return pips * spec["pip_size"]

    def price_to_pips(self, symbol: str, price_movement: float) -> float:
        """Convert a price movement to pips for *symbol*.

        Args:
            symbol: Trading symbol.
            price_movement: Absolute price change.

        Returns:
            Equivalent pip count.
        """
        spec = self._get_forex_spec(symbol)
        if spec["pip_size"] == 0:
            return 0.0
        return price_movement / spec["pip_size"]

    def calculate_lot_pnl(
        self,
        symbol: str,
        lot_size: float,
        entry_price: float,
        exit_price: float,
        direction: str,
    ) -> float:
        """Calculate pip-based P&L for a forex position.

        Args:
            symbol: Trading symbol.
            lot_size: Position size in lots (0.01 = micro lot).
            entry_price: Average entry price.
            exit_price: Exit price.
            direction: ``"long"`` or ``"short"``.

        Returns:
            P&L in USDT.
        """
        spec = self._get_forex_spec(symbol)
        pip_size = spec["pip_size"]
        contract_size = spec["contract_size"]

        if direction == "long":
            pip_diff = (exit_price - entry_price) / pip_size
        else:
            pip_diff = (entry_price - exit_price) / pip_size

        # P&L = pips × pip_value × lot_size
        # pip_value = pip_size × contract_size (in quote currency = USDT)
        pip_value = pip_size * contract_size
        return round(pip_diff * pip_value * lot_size, 4)

    def simulate_spread(self, symbol: str, mid_price: float, side: str) -> float:
        """Apply bid/ask spread simulation to a mid price.

        Buys are filled at ask (mid + half spread), sells at bid (mid - half spread).

        Args:
            symbol: Trading symbol.
            mid_price: Mid market price.
            side: ``"buy"`` or ``"sell"``.

        Returns:
            Fill price with spread applied.
        """
        spec = self._get_forex_spec(symbol)
        half_spread = (spec["typical_spread_pips"] / 2.0) * spec["pip_size"]
        if side == "buy":
            return mid_price + half_spread
        return mid_price - half_spread
