"""Cross-exchange arbitrage strategy — captures price discrepancies across exchanges.

Compares live prices across all configured exchanges (MEXC, Gate.io, BingX, Bitget)
and generates a signal when the spread exceeds the combined round-trip fee cost.

The strategy is *directional*: it goes long on the cheapest exchange and short on
the most expensive one, then closes both legs when the spread collapses.

.. note::
    This strategy requires two separate exchange connections (one per leg).  When
    running in paper mode both legs are simulated via the price feed.  In live mode
    the engine must supply exchange clients for each leg via
    :meth:`set_exchange_clients`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class CrossExchangeArbStrategy(BaseStrategy):
    """Generates signals when cross-exchange price spread exceeds combined fees.

    Logic
    -----
    1. Fetch the latest mid-price for each symbol from all registered exchanges.
    2. Identify the exchange with the lowest ask (*buy leg*) and the exchange
       with the highest bid (*sell leg*).
    3. Calculate the gross spread: ``(best_bid - best_ask) / best_ask``.
    4. Subtract the estimated round-trip fee (``2 × fee_pct``).
    5. If ``net_spread > min_spread_threshold``, emit a ``long`` signal with
       metadata describing both legs.

    The strategy emits at most one signal per symbol per price update to avoid
    signal flooding when the spread persists.

    Args:
        symbols: List of symbols to monitor (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        timeframe: Unused by this strategy (kept for API compatibility).
        enabled: Whether the strategy is active.
        min_spread_pct: Minimum net spread (after fees) to trigger a signal,
            expressed as a fraction (e.g. ``0.001`` for 0.1 %).
        fee_pct: Estimated one-way taker fee per exchange (default 0.05 %).
        cooldown_seconds: Minimum seconds between signals for the same symbol.
    """

    #: Supported exchanges — must match keys in the exchange registry (informational)
    SUPPORTED_EXCHANGES = ("mexc", "gateio", "bingx", "bitget")
    #: Fraction of min_spread_pct at which a position is considered worth closing
    _CLOSE_THRESHOLD_MULTIPLIER: float = 0.5

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        enabled: bool = True,
        min_spread_pct: float = 0.001,
        fee_pct: float = 0.0005,
        cooldown_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            name="cross_exchange_arb",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._min_spread = min_spread_pct
        self._fee_pct = fee_pct
        self._cooldown = cooldown_seconds
        # exchange_name → {symbol: {"bid": float, "ask": float}}
        self._prices: Dict[str, Dict[str, Dict[str, float]]] = {}
        # exchange clients injected at runtime: exchange_name → BaseExchange instance
        self._exchange_clients: Dict[str, Any] = {}
        # Last signal timestamp per symbol to enforce cooldown
        self._last_signal: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_exchange_clients(self, clients: Dict[str, Any]) -> None:
        """Register exchange clients used to fetch live prices.

        Args:
            clients: Mapping of exchange name → BaseExchange (or compatible) client.
        """
        self._exchange_clients = dict(clients)
        logger.info(
            "CrossExchangeArbStrategy: registered {} exchange client(s): {}",
            len(clients),
            list(clients.keys()),
        )

    def update_price(
        self,
        exchange: str,
        symbol: str,
        bid: float,
        ask: float,
    ) -> None:
        """Inject a live price snapshot for *symbol* from *exchange*.

        This method is called by the engine's price-feed loop whenever a new
        ticker arrives.

        Args:
            exchange: Exchange name, e.g. ``"mexc"``.
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            bid: Best bid price.
            ask: Best ask price.
        """
        if exchange not in self._prices:
            self._prices[exchange] = {}
        self._prices[exchange][symbol] = {"bid": bid, "ask": ask}

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def generate_signal(
        self, symbol: str, ohlcv: Any, market_data: Optional[Dict] = None
    ) -> Optional[Signal]:
        """Return an arb signal if a profitable cross-exchange spread exists.

        If live exchange clients are registered, prices are refreshed before
        computing the spread.  Otherwise cached prices from :meth:`update_price`
        are used.

        Args:
            symbol: Symbol to evaluate.
            ohlcv: OHLCV DataFrame (not used by this strategy).
            market_data: Optional additional market context.

        Returns:
            A :class:`~strategy.base_strategy.Signal` when an opportunity
            exists, or ``None`` otherwise.
        """
        if not self._enabled:
            return None

        # Refresh prices from live clients if available
        if self._exchange_clients:
            await self._refresh_prices(symbol)

        opportunity = self._find_best_spread(symbol)
        if opportunity is None:
            return None

        buy_exchange, sell_exchange, gross_spread, buy_ask, sell_bid = opportunity
        net_spread = gross_spread - 2 * self._fee_pct

        if net_spread < self._min_spread:
            return None

        # Enforce per-symbol cooldown
        import time
        now = time.monotonic()
        if now - self._last_signal.get(symbol, 0.0) < self._cooldown:
            return None
        self._last_signal[symbol] = now

        reasoning = (
            f"Arb opportunity: buy on {buy_exchange} @ {buy_ask:.6f}, "
            f"sell on {sell_exchange} @ {sell_bid:.6f}. "
            f"Gross spread={gross_spread:.4%} net={net_spread:.4%}"
        )
        logger.info(
            "CrossExchangeArb signal: {} {} (buy={} ask={:.4f}, sell={} bid={:.4f}, "
            "net_spread={:.4%})",
            symbol,
            "long",
            buy_exchange,
            buy_ask,
            sell_exchange,
            sell_bid,
            net_spread,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            strength=min(1.0, net_spread / self._min_spread),
            confidence=0.7,
            strategy_name=self._name,
            reasoning=reasoning,
        )

    def should_close(
        self, symbol: str, position: Any, market_data: Optional[Dict] = None
    ) -> bool:
        """Close the position when the spread has collapsed below fees.

        Args:
            symbol: Market symbol.
            position: Current position tracker.
            market_data: Optional market context.

        Returns:
            ``True`` if the spread has collapsed below the minimum threshold.
        """
        opportunity = self._find_best_spread(symbol)
        if opportunity is None:
            # No spread data — keep position open
            return False
        _buy_ex, _sell_ex, gross_spread, _buy_ask, _sell_bid = opportunity
        net_spread = gross_spread - 2 * self._fee_pct
        if net_spread < self._min_spread * self._CLOSE_THRESHOLD_MULTIPLIER:
            logger.info(
                "CrossExchangeArb: closing {} — spread collapsed to {:.4%}",
                symbol,
                net_spread,
            )
            return True
        return False

    def calculate_parameters(
        self, symbol: str, ohlcv: Any
    ) -> Dict[str, Any]:
        """Return current strategy parameters for *symbol*."""
        return {
            "min_spread_pct": self._min_spread,
            "fee_pct": self._fee_pct,
            "tracked_exchanges": list(self._prices.keys()),
            "cooldown_seconds": self._cooldown,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_best_spread(
        self, symbol: str
    ) -> Optional[Tuple[str, str, float, float, float]]:
        """Return the best (buy_exchange, sell_exchange, spread, buy_ask, sell_bid).

        Scans all exchanges that have a cached price for *symbol* and identifies
        the lowest ask and highest bid.  Returns ``None`` when fewer than 2
        exchanges have price data for the symbol.
        """
        quotes: List[Tuple[str, float, float]] = []  # (exchange, bid, ask)
        for exchange, symbols in self._prices.items():
            if symbol in symbols:
                entry = symbols[symbol]
                bid = entry.get("bid", 0.0)
                ask = entry.get("ask", 0.0)
                if bid > 0 and ask > 0:
                    quotes.append((exchange, bid, ask))

        if len(quotes) < 2:
            return None

        # Find exchange with lowest ask (buy leg) and highest bid (sell leg)
        buy_exchange, _bid_buy, buy_ask = min(quotes, key=lambda x: x[2])
        sell_exchange, sell_bid, _ask_sell = max(quotes, key=lambda x: x[1])

        if buy_exchange == sell_exchange:
            return None

        if buy_ask <= 0:
            return None

        gross_spread = (sell_bid - buy_ask) / buy_ask
        return buy_exchange, sell_exchange, gross_spread, buy_ask, sell_bid

    async def _refresh_prices(self, symbol: str) -> None:
        """Fetch latest prices from all registered exchange clients."""
        tasks = []
        for name, client in self._exchange_clients.items():
            tasks.append(self._fetch_one(name, symbol, client))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_one(self, exchange: str, symbol: str, client: Any) -> None:
        """Fetch a single ticker and store it in the price cache."""
        try:
            ticker = await client.get_ticker(symbol)
            self.update_price(
                exchange=exchange,
                symbol=symbol,
                bid=ticker.bid,
                ask=ticker.ask,
            )
        except Exception as exc:
            logger.debug(
                "CrossExchangeArb: could not fetch {} ticker for {}: {}",
                exchange,
                symbol,
                exc,
            )
