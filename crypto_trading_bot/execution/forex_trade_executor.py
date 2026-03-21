"""Forex Trade Executor — end-to-end order lifecycle for forex/gold trading."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

from exchange.base_exchange import OrderSide, PositionSide
from risk.forex_risk_manager import ForexRiskManager, ForexTradeApproval

# ---------------------------------------------------------------------------
# ForexTradeExecutor
# ---------------------------------------------------------------------------


class ForexTradeExecutor:
    """Handles the complete lifecycle of a forex trade.

    * Lot-size calculation via :class:`ForexRiskManager`.
    * Pip-based SL/TP placement.
    * Spread-aware entry (waits for spread to narrow if too wide).
    * Partial close at TP1/TP2/TP3 (25 % / 50 % / 25 %).
    * Break-even stop-loss after TP1 is reached.
    * Trailing stop in pips.
    * Position reconciliation on startup.
    """

    # Partial close percentages at each TP level
    PARTIAL_CLOSE_TP1_PCT = 0.25   # close 25% at TP1
    PARTIAL_CLOSE_TP2_PCT = 0.50   # close 50% at TP2
    PARTIAL_CLOSE_TP3_PCT = 0.25   # close remaining 25% at TP3

    # TP level multipliers: fraction of full TP distance for each partial-close level
    TP1_MULTIPLIER = 0.33  # one-third of total TP distance
    TP2_MULTIPLIER = 0.66  # two-thirds of total TP distance

    # Maximum retries when spread is too wide
    MAX_SPREAD_RETRIES = 5
    SPREAD_RETRY_DELAY = 10.0  # seconds

    # Default pip values for gold
    GOLD_PIP_SIZE = 0.01

    # Trailing stop: activated after TP1 hit; trails at this fraction of the SL distance
    DEFAULT_TRAIL_SL_PIPS = 200.0
    TRAIL_SL_FACTOR = 0.5

    def __init__(self, exchange: Any, risk_manager: ForexRiskManager) -> None:
        self._exchange = exchange
        self._risk_manager = risk_manager
        # Track active trades: symbol → trade state
        self._active_trades: Dict[str, Dict[str, Any]] = {}
        self._tp1_hit: Dict[str, bool] = {}
        self._trailing_stop_active: Dict[str, bool] = {}
        self._trailing_stop_price: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def execute_forex_trade(
        self,
        signal: Dict[str, Any],
        equity: float,
    ) -> Optional[Dict[str, Any]]:
        """Execute a forex trade from a strategy signal.

        Args:
            signal: Signal dict with keys: symbol, direction, strength, confidence.
            equity: Current account equity in USDT.

        Returns:
            Trade result dict or ``None`` if trade was not placed.
        """
        symbol = signal.get("symbol", "XAU/USDT")
        direction = signal.get("direction", "long")

        logger.info("ForexTradeExecutor: evaluating {} {} signal", direction, symbol)

        # 1. Get current ticker for spread check
        ticker = await self._exchange.get_ticker(symbol)
        if ticker.last == 0:
            logger.warning("ForexTradeExecutor: zero price for {} — skipping", symbol)
            return None

        # 2. Estimate spread
        spread_pips = self._estimate_spread_pips(symbol, ticker.bid, ticker.ask)

        # 3. Risk manager validation and lot sizing
        ohlcv = await self._exchange.get_ohlcv(symbol, "15m", 50)
        from strategy.base_strategy import BaseStrategy
        atr = BaseStrategy._calculate_atr(ohlcv, 14) if len(ohlcv) >= 15 else 0.0

        leverage = getattr(
            getattr(self._exchange, "_settings", None),
            "forex",
            None,
        )
        leverage_val = getattr(leverage, "default_leverage", 20) if leverage else 20

        approval: ForexTradeApproval = self._risk_manager.validate_forex_trade(
            symbol=symbol,
            direction=direction,
            equity=equity,
            current_price=ticker.last,
            spread_pips=spread_pips,
            atr=atr,
            leverage=leverage_val,
        )

        if not approval.approved:
            logger.info(
                "ForexTradeExecutor: trade REJECTED — {}", approval.rejection_reason
            )
            return {"status": "rejected", "reason": approval.rejection_reason}

        # 4. Spread-aware entry: wait for spread to narrow if too wide
        entry_price = await self._wait_for_acceptable_spread(symbol, spread_pips, approval)
        if entry_price is None:
            return {"status": "rejected", "reason": "Spread remained too wide — entry skipped"}

        # 5. Set leverage
        try:
            await self._exchange.set_leverage(symbol, approval.leverage)
        except Exception as e:
            logger.warning("ForexTradeExecutor: failed to set leverage — {}", e)

        # 6. Place entry order
        order_side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        size = approval.lot_size  # lot size (Gate.io TradFi uses contracts)

        logger.info(
            "ForexTradeExecutor: placing {} {} at ~{} (lot={} SL={} TP={})",
            direction, symbol, round(entry_price, 2),
            approval.lot_size, approval.stop_loss_price, approval.take_profit_price,
        )

        try:
            entry_order = await self._exchange.create_market_order(
                symbol=symbol,
                side=order_side,
                amount=size,
                params={"leverage": approval.leverage},
            )
        except Exception as e:
            logger.error("ForexTradeExecutor: entry order failed — {}", e)
            return {"status": "error", "reason": str(e)}

        fill_price = entry_order.filled if entry_order.filled > 0 else entry_price

        # 7. Place SL/TP orders
        await self._place_sl_tp(symbol, direction, size, approval, fill_price)

        # 8. Track trade state
        trade_id = entry_order.id
        self._active_trades[symbol] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": fill_price,
            "lot_size": size,
            "sl_price": approval.stop_loss_price,
            "tp1_price": approval.take_profit_price,
            "tp2_price": None,
            "tp3_price": approval.take_profit_price,
            "sl_pips": approval.stop_loss_pips,
            "tp_pips": approval.take_profit_pips,
            "opened_at": time.time(),
            "approval": approval,
        }
        self._tp1_hit[symbol] = False
        self._trailing_stop_active[symbol] = False

        self._risk_manager.increment_open_trades()

        logger.info(
            "ForexTradeExecutor: trade opened — {} {} fill={} lot={} SL={} TP={}",
            direction, symbol, round(fill_price, 2), size,
            approval.stop_loss_price, approval.take_profit_price,
        )

        return {
            "status": "opened",
            "symbol": symbol,
            "direction": direction,
            "fill_price": fill_price,
            "lot_size": size,
            "stop_loss": approval.stop_loss_price,
            "take_profit": approval.take_profit_price,
            "order_id": trade_id,
        }

    # ------------------------------------------------------------------
    # SL/TP placement
    # ------------------------------------------------------------------

    async def _place_sl_tp(
        self,
        symbol: str,
        direction: str,
        size: float,
        approval: ForexTradeApproval,
        fill_price: float,
    ) -> None:
        """Place stop-loss and take-profit orders."""
        close_side = OrderSide.SELL if direction == "long" else OrderSide.BUY

        # Calculate TP levels for partial close
        if direction == "long":
            tp1 = fill_price + (approval.take_profit_pips * self.TP1_MULTIPLIER * self.GOLD_PIP_SIZE)
            tp2 = fill_price + (approval.take_profit_pips * self.TP2_MULTIPLIER * self.GOLD_PIP_SIZE)
            tp3 = approval.take_profit_price
        else:
            tp1 = fill_price - (approval.take_profit_pips * self.TP1_MULTIPLIER * self.GOLD_PIP_SIZE)
            tp2 = fill_price - (approval.take_profit_pips * self.TP2_MULTIPLIER * self.GOLD_PIP_SIZE)
            tp3 = approval.take_profit_price

        # Update stored TP levels
        if symbol in self._active_trades:
            self._active_trades[symbol]["tp1_price"] = tp1
            self._active_trades[symbol]["tp2_price"] = tp2
            self._active_trades[symbol]["tp3_price"] = tp3

        # Place SL
        try:
            await self._exchange.create_stop_loss_order(
                symbol=symbol,
                side=close_side,
                amount=size,
                stop_price=approval.stop_loss_price,
            )
            logger.debug("SL placed at {} for {}", approval.stop_loss_price, symbol)
        except Exception as e:
            logger.error("Failed to place SL for {}: {}", symbol, e)

        # Place TP3 (full close at final TP)
        try:
            await self._exchange.create_take_profit_order(
                symbol=symbol,
                side=close_side,
                amount=size,
                tp_price=tp3,
            )
            logger.debug("TP3 placed at {} for {}", tp3, symbol)
        except Exception as e:
            logger.error("Failed to place TP for {}: {}", symbol, e)

    # ------------------------------------------------------------------
    # Position management (break-even, trailing, partial close)
    # ------------------------------------------------------------------

    async def update_position_management(self, symbol: str) -> None:
        """Check and update SL/TP for an open position.

        Should be called periodically (every 30 seconds) while a trade is active.
        Implements break-even after TP1, trailing stop activation.
        """
        if symbol not in self._active_trades:
            return

        trade = self._active_trades[symbol]
        ticker = await self._exchange.get_ticker(symbol)
        current_price = ticker.last
        if current_price == 0:
            return

        direction = trade["direction"]
        entry_price = trade["entry_price"]
        pip_size = self.GOLD_PIP_SIZE

        # Check TP1 hit → activate break-even SL
        tp1 = trade.get("tp1_price")
        if tp1 and not self._tp1_hit.get(symbol, False):
            tp1_hit = (direction == "long" and current_price >= tp1) or \
                      (direction == "short" and current_price <= tp1)
            if tp1_hit:
                self._tp1_hit[symbol] = True
                # Move SL to break-even
                breakeven_sl = entry_price + (2 * pip_size) if direction == "long" \
                    else entry_price - (2 * pip_size)
                await self._move_stop_loss(symbol, direction, trade["lot_size"], breakeven_sl)
                logger.info(
                    "Break-even SL activated for {} at {:.2f}", symbol, breakeven_sl
                )

                # Partially close at TP1
                partial_size = round(trade["lot_size"] * self.PARTIAL_CLOSE_TP1_PCT, 2)
                if partial_size > 0:
                    await self._partial_close(symbol, direction, partial_size)
                    trade["lot_size"] = round(trade["lot_size"] - partial_size, 2)

        # Trailing stop in pips (activate after TP1 hit)
        if self._tp1_hit.get(symbol, False):
            trail_pips = trade.get("sl_pips", self.DEFAULT_TRAIL_SL_PIPS) * self.TRAIL_SL_FACTOR
            if direction == "long":
                new_trail_sl = current_price - (trail_pips * pip_size)
                current_sl = self._trailing_stop_price.get(symbol, trade["sl_price"])
                if new_trail_sl > current_sl:
                    self._trailing_stop_price[symbol] = new_trail_sl
                    await self._move_stop_loss(symbol, direction, trade["lot_size"], new_trail_sl)
                    logger.debug("Trailing SL updated to {:.2f} for {}", new_trail_sl, symbol)
            else:
                new_trail_sl = current_price + (trail_pips * pip_size)
                current_sl = self._trailing_stop_price.get(symbol, trade["sl_price"])
                if new_trail_sl < current_sl:
                    self._trailing_stop_price[symbol] = new_trail_sl
                    await self._move_stop_loss(symbol, direction, trade["lot_size"], new_trail_sl)
                    logger.debug("Trailing SL updated to {:.2f} for {}", new_trail_sl, symbol)

    async def _move_stop_loss(
        self, symbol: str, direction: str, size: float, new_sl: float
    ) -> None:
        """Cancel existing SL and place a new one."""
        try:
            # Cancel all existing SL orders for symbol
            await self._exchange.cancel_all_orders(symbol)
            # Place new SL
            close_side = OrderSide.SELL if direction == "long" else OrderSide.BUY
            await self._exchange.create_stop_loss_order(
                symbol=symbol,
                side=close_side,
                amount=size,
                stop_price=new_sl,
            )
        except Exception as e:
            logger.warning("Failed to move SL for {}: {}", symbol, e)

    async def _partial_close(self, symbol: str, direction: str, size: float) -> None:
        """Partially close a position."""
        close_side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        try:
            await self._exchange.create_market_order(
                symbol=symbol,
                side=close_side,
                amount=size,
                params={"reduceOnly": True},
            )
            logger.info("Partial close {:.2f} lots of {} completed", size, symbol)
        except Exception as e:
            logger.warning("Partial close failed for {}: {}", symbol, e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_spread_pips(
        self, symbol: str, bid: float, ask: float
    ) -> float:
        """Estimate current spread in pips."""
        if bid <= 0 or ask <= 0:
            return 5.0  # safe default
        spread_price = ask - bid
        # Gold pip = 0.01
        pip_size = self.GOLD_PIP_SIZE
        return spread_price / pip_size

    async def _wait_for_acceptable_spread(
        self,
        symbol: str,
        current_spread_pips: float,
        approval: ForexTradeApproval,
    ) -> Optional[float]:
        """Wait for spread to narrow if it is too wide.

        Returns the current price when spread is acceptable, or ``None`` if
        the maximum retries are exhausted.
        """
        max_spread = ForexRiskManager.PAIR_SPECS.get(
            symbol, {}).get("max_acceptable_spread_pips", 10.0)

        for attempt in range(self.MAX_SPREAD_RETRIES):
            if current_spread_pips <= max_spread:
                ticker = await self._exchange.get_ticker(symbol)
                return ticker.last
            logger.info(
                "Spread too wide for {} ({:.1f} > {:.1f} pips) — waiting {}s (attempt {}/{})",
                symbol, current_spread_pips, max_spread,
                self.SPREAD_RETRY_DELAY, attempt + 1, self.MAX_SPREAD_RETRIES,
            )
            await asyncio.sleep(self.SPREAD_RETRY_DELAY)
            ticker = await self._exchange.get_ticker(symbol)
            current_spread_pips = self._estimate_spread_pips(symbol, ticker.bid, ticker.ask)

        return None  # spread remained too wide

    async def reconcile_positions(self) -> None:
        """Reconcile active trades with exchange positions on startup."""
        try:
            positions = await self._exchange.get_positions()
            for pos in positions:
                if pos.symbol not in self._active_trades:
                    logger.info(
                        "ForexTradeExecutor: found untracked position {} {} {}",
                        pos.symbol, pos.side.value, pos.amount
                    )
                    self._active_trades[pos.symbol] = {
                        "symbol": pos.symbol,
                        "direction": "long" if pos.side == PositionSide.LONG else "short",
                        "entry_price": pos.entry_price,
                        "lot_size": pos.amount,
                        "sl_price": 0.0,
                        "tp1_price": 0.0,
                        "tp3_price": 0.0,
                        "sl_pips": 200.0,
                        "tp_pips": 400.0,
                        "opened_at": time.time(),
                    }
            logger.info(
                "ForexTradeExecutor: reconciled {} positions", len(positions)
            )
        except Exception as e:
            logger.warning("ForexTradeExecutor: reconciliation failed — {}", e)

    def close_trade(self, symbol: str) -> None:
        """Mark a trade as closed and update risk manager."""
        if symbol in self._active_trades:
            del self._active_trades[symbol]
        if symbol in self._tp1_hit:
            del self._tp1_hit[symbol]
        if symbol in self._trailing_stop_price:
            del self._trailing_stop_price[symbol]
        self._risk_manager.decrement_open_trades()

    @property
    def active_trades(self) -> Dict[str, Dict[str, Any]]:
        """Return currently active trades."""
        return dict(self._active_trades)
