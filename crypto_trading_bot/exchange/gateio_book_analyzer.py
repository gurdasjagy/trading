"""Gate.io order book analyzer — deep liquidity analysis for optimal execution.

.. deprecated::
    This module is superseded by the **Synthetic L3 Microstructure Engine**
    (``rust_engine/src/microstructure.rs``).

    The Rust engine provides equivalent and more sophisticated analysis:
    * ``BookPressureAnalyzer`` — depth-change gradient, spoofing, absorption
    * ``SyntheticQueueTracker`` — per-level queue-position estimation
    * ``MicrostructureEngine`` — composite edge score

    All computations are in the Rust hot path (< 10 µs per evaluation).
    This Python class is retained for backward compatibility only.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "exchange.gateio_book_analyzer.GateioBookAnalyzer is deprecated. "
    "Use the Rust MicrostructureEngine (rust_engine/src/microstructure.rs) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from exchange.base_exchange import BaseExchange

try:
    from rust_trading_engine.orderbook import RustBookAnalyzer, RustOrderBook
    _USE_RUST_ANALYZER = True
except ImportError:
    _USE_RUST_ANALYZER = False


class GateioBookAnalyzer:
    """Analyzes the Gate.io order book to guide execution decisions.

    Provides:
    - Imbalance signal: bid/ask depth ratio → directional pressure
    - Spread measurement in basis-points
    - Large-order level identification (iceberg / whale support/resistance)
    - Optimal entry/exit prices that minimize total cost
    - Market-impact simulation: VWAP fill price for a given size
    """

    # A level is "large" when its notional exceeds this multiple of the average
    # (i.e. the level is 5× the mean notional across visible book levels — a
    # common heuristic for iceberg/whale walls in futures markets).
    _LARGE_ORDER_MULTIPLIER: float = 5.0
    # Number of top levels used for depth calculations
    _DEPTH_LEVELS: int = 10
    # Fraction inside the spread used for optimal price calculation (10 %)
    _OPTIMAL_PRICE_SPREAD_FRACTION: float = 0.1
    # Minimum price tick to avoid division/rounding issues
    _MIN_PRICE_TICK: float = 1e-8

    def __init__(self, exchange: "BaseExchange") -> None:
        self._exchange = exchange

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_book(
        self,
        symbol: str,
        depth: int = 20,
        rust_book: Optional["RustOrderBook"] = None,
    ) -> Dict[str, Any]:
        """Fetch the Gate.io order book and return rich execution metrics.

        Returns a dict with keys:
            imbalance       : float in [-1, +1]  (+1 = fully bid-heavy)
            spread_bps      : float              (best spread in basis points)
            bid_depth_usdt  : float              (total bid liquidity, top 10 levels)
            ask_depth_usdt  : float              (total ask liquidity, top 10 levels)
            large_bid_levels: list[dict]         (price, size, notional)
            large_ask_levels: list[dict]         (price, size, notional)
            optimal_buy_price : float            (price that minimises taker cost)
            optimal_sell_price: float            (price that maximises taker credit)
            best_bid        : float
            best_ask        : float
            mid_price       : float

        When *rust_book* is provided and the Rust engine is available the REST
        fetch is skipped entirely — analysis runs fully in-process on the
        pre-populated :class:`~rust_trading_engine.orderbook.RustOrderBook`.
        """
        # Fast path: Rust book available — skip REST entirely.
        if _USE_RUST_ANALYZER and rust_book is not None:
            try:
                result = RustBookAnalyzer.analyze(rust_book, depth)
                return result
            except Exception as exc:
                logger.debug(
                    "BookAnalyzer: Rust analysis failed for {} ({}), falling back to REST",
                    symbol,
                    exc,
                )

        # Original Python implementation (REST fetch + Python analysis).
        try:
            book = await self._exchange.get_orderbook(symbol, limit=depth)
        except Exception as exc:
            logger.warning("BookAnalyzer: failed to fetch book for {} — {}", symbol, exc)
            return self._empty_result()

        bids: List[List[float]] = book.get("bids", [])
        asks: List[List[float]] = book.get("asks", [])

        if not bids or not asks:
            return self._empty_result()

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid_price) * 10_000.0 if mid_price > 0 else 0.0

        # Depth in USDT for top _DEPTH_LEVELS levels
        bid_levels = bids[: self._DEPTH_LEVELS]
        ask_levels = asks[: self._DEPTH_LEVELS]

        bid_depth_usdt = sum(float(p) * float(q) for p, q in bid_levels)
        ask_depth_usdt = sum(float(p) * float(q) for p, q in ask_levels)

        total_depth = bid_depth_usdt + ask_depth_usdt
        imbalance = (
            (bid_depth_usdt - ask_depth_usdt) / total_depth if total_depth > 0 else 0.0
        )

        # Identify large orders
        large_bid_levels = self._find_large_levels(bid_levels)
        large_ask_levels = self._find_large_levels(ask_levels)

        # Optimal prices (just inside the spread — best possible limit fill)
        tick = (best_ask - best_bid) / 2.0
        optimal_buy_price = best_bid + max(
            tick * self._OPTIMAL_PRICE_SPREAD_FRACTION, self._MIN_PRICE_TICK
        )
        optimal_sell_price = best_ask - max(
            tick * self._OPTIMAL_PRICE_SPREAD_FRACTION, self._MIN_PRICE_TICK
        )

        return {
            "imbalance": round(imbalance, 4),
            "spread_bps": round(spread_bps, 4),
            "bid_depth_usdt": round(bid_depth_usdt, 2),
            "ask_depth_usdt": round(ask_depth_usdt, 2),
            "large_bid_levels": large_bid_levels,
            "large_ask_levels": large_ask_levels,
            "optimal_buy_price": round(optimal_buy_price, 8),
            "optimal_sell_price": round(optimal_sell_price, 8),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": round(mid_price, 8),
        }

    def calculate_market_impact(
        self,
        symbol: str,
        side: str,
        amount_contracts: float,
        book: Optional[Dict[str, Any]] = None,
        rust_book: Optional["RustOrderBook"] = None,
    ) -> float:
        """Estimate VWAP fill price for *amount_contracts* contracts.

        Walks the cached book (or a provided snapshot) and returns the
        volume-weighted average price.  If the book is too thin the last
        available level price is used for the remainder.

        When *rust_book* is provided and the Rust engine is available the
        computation runs fully in Rust (no Python loops).

        Args:
            symbol: trading symbol (used for logging)
            side: "buy" or "sell"
            amount_contracts: order size in contracts
            book: pre-fetched book dict from analyze_book(); if None a
                  synchronous fallback returning 0.0 is used (caller
                  should prefer passing the book).
            rust_book: optional Rust-backed order book for the fast path.

        Returns:
            VWAP fill price (0.0 if book is unavailable).
        """
        # Fast path: delegate to Rust
        if _USE_RUST_ANALYZER and rust_book is not None:
            try:
                return RustBookAnalyzer.calculate_market_impact(rust_book, side, amount_contracts)
            except Exception as exc:
                logger.debug(
                    "BookAnalyzer: Rust market-impact calc failed for {} ({}), falling back",
                    symbol,
                    exc,
                )

        if book is None:
            logger.debug(
                "BookAnalyzer.calculate_market_impact: no book provided for {}", symbol
            )
            return 0.0

        raw_bids: List[List[float]] = book.get("bids", [])
        raw_asks: List[List[float]] = book.get("asks", [])
        levels = raw_asks if side.lower() == "buy" else raw_bids

        if not levels:
            return 0.0

        remaining = amount_contracts
        total_cost = 0.0
        last_price = float(levels[-1][0])

        for price_raw, qty_raw in levels:
            price = float(price_raw)
            qty = float(qty_raw)
            fill = min(remaining, qty)
            total_cost += fill * price
            remaining -= fill
            last_price = price
            if remaining <= 0:
                break

        if remaining > 0:
            # Not enough depth — fill remainder at last level price
            total_cost += remaining * last_price

        return total_cost / amount_contracts if amount_contracts > 0 else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_large_levels(
        self, levels: List[List[float]]
    ) -> List[Dict[str, float]]:
        """Return levels whose notional is > _LARGE_ORDER_MULTIPLIER × average."""
        if not levels:
            return []
        notionals = [float(p) * float(q) for p, q in levels]
        avg = sum(notionals) / len(notionals) if notionals else 0.0
        threshold = avg * self._LARGE_ORDER_MULTIPLIER
        result = []
        for (p, q), notional in zip(levels, notionals):
            if notional >= threshold:
                result.append(
                    {
                        "price": float(p),
                        "size": float(q),
                        "notional_usdt": round(notional, 2),
                    }
                )
        return result

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "imbalance": 0.0,
            "spread_bps": 0.0,
            "bid_depth_usdt": 0.0,
            "ask_depth_usdt": 0.0,
            "large_bid_levels": [],
            "large_ask_levels": [],
            "optimal_buy_price": 0.0,
            "optimal_sell_price": 0.0,
            "best_bid": 0.0,
            "best_ask": 0.0,
            "mid_price": 0.0,
        }
