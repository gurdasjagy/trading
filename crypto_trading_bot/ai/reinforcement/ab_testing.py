"""A/B testing framework for strategy selection.

Randomly assigns 10 % of signals to a "challenger" strategy selection method
and compares its performance against the incumbent over 100 trades.  When the
challenger outperforms the incumbent by > 5 % Sharpe, it is promoted.

All test results are logged for offline analysis.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


class ABTestingFramework:
    """Compare two strategy selection methods (incumbent vs challenger).

    Terminology
    -----------
    *Incumbent*: The current production selection method.
    *Challenger*: The experimental selection method under test.

    The framework routes ``challenger_fraction`` of signal generations to the
    challenger.  After ``min_trades`` trades on each side, a promotion check
    is run.  If the challenger's Sharpe ratio exceeds the incumbent's by more
    than ``promotion_threshold_pct`` percent, the challenger is promoted to
    become the new incumbent.

    Args:
        incumbent_fn: Callable that returns a list of selected strategy names.
        challenger_fn: Callable for the experimental selection method.
        challenger_fraction: Fraction of calls routed to the challenger
            (default 0.10 = 10 %).
        min_trades: Minimum number of trades per variant before promotion is
            evaluated (default 100).
        promotion_threshold_pct: Minimum % Sharpe improvement required to
            promote the challenger (default 5 %).
        max_history: Maximum number of trade outcomes to keep per variant.
    """

    def __init__(
        self,
        incumbent_fn: Optional[Callable] = None,
        challenger_fn: Optional[Callable] = None,
        challenger_fraction: float = 0.10,
        min_trades: int = 100,
        promotion_threshold_pct: float = 5.0,
        max_history: int = 500,
    ) -> None:
        self.challenger_fraction = challenger_fraction
        self.min_trades = min_trades
        self.promotion_threshold_pct = promotion_threshold_pct

        self._incumbent_fn: Optional[Callable] = incumbent_fn
        self._challenger_fn: Optional[Callable] = challenger_fn

        # Trade outcome histories: deque of pnl_pct values
        self._incumbent_trades: deque = deque(maxlen=max_history)
        self._challenger_trades: deque = deque(maxlen=max_history)

        # Running log of all A/B test events
        self._log: List[Dict[str, Any]] = []

        # Promotion history
        self._promotions: List[Dict[str, Any]] = []

        # Current cycle start timestamp
        self._cycle_start_ts: float = time.time()
        self._promotion_count: int = 0

        logger.info(
            "ABTestingFramework initialised: challenger_fraction={:.0%}, "
            "min_trades={}, promotion_threshold={:.1f}%",
            challenger_fraction,
            min_trades,
            promotion_threshold_pct,
        )

    # ------------------------------------------------------------------
    # Selection routing
    # ------------------------------------------------------------------

    def should_use_challenger(self) -> bool:
        """Return True if this invocation should use the challenger."""
        return random.random() < self.challenger_fraction

    def select(self, *args: Any, **kwargs: Any) -> Tuple[List[str], str]:
        """Route to incumbent or challenger and return (selections, variant).

        Args:
            *args, **kwargs: Forwarded to the chosen selection callable.

        Returns:
            Tuple of (selected_strategy_names, variant) where variant is
            ``"challenger"`` or ``"incumbent"``.
        """
        use_challenger = self.should_use_challenger()
        variant = "challenger" if use_challenger else "incumbent"
        fn = self._challenger_fn if use_challenger else self._incumbent_fn

        if fn is None:
            return [], variant

        try:
            result = fn(*args, **kwargs)
            # Support both plain lists and (names, metadata) tuples
            if isinstance(result, tuple):
                names, meta = result
            else:
                names, meta = result, {}
        except Exception as exc:
            logger.error("ABTest {} selection failed: {}", variant, exc)
            names, meta = [], {}

        self._log.append(
            {
                "timestamp": time.time(),
                "variant": variant,
                "selected": names,
                "metadata": meta,
            }
        )
        return names, variant

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(self, variant: str, pnl_pct: float) -> None:
        """Record a trade outcome for the given variant.

        Args:
            variant: ``"incumbent"`` or ``"challenger"``.
            pnl_pct: Profit/loss percentage for the trade.
        """
        if variant == "challenger":
            self._challenger_trades.append(pnl_pct)
        else:
            self._incumbent_trades.append(pnl_pct)

        self._log.append(
            {
                "timestamp": time.time(),
                "event": "outcome",
                "variant": variant,
                "pnl_pct": pnl_pct,
            }
        )

        # Check promotion criteria
        self._maybe_promote()

    # ------------------------------------------------------------------
    # Promotion check
    # ------------------------------------------------------------------

    def _maybe_promote(self) -> bool:
        """Check if the challenger should be promoted; return True if promoted."""
        n_inc = len(self._incumbent_trades)
        n_cha = len(self._challenger_trades)

        if n_inc < self.min_trades or n_cha < self.min_trades:
            return False

        inc_sharpe = self._sharpe(list(self._incumbent_trades))
        cha_sharpe = self._sharpe(list(self._challenger_trades))

        pct_improvement = (
            ((cha_sharpe - inc_sharpe) / (abs(inc_sharpe) + 1e-9)) * 100.0
        )

        logger.info(
            "ABTest check: incumbent_sharpe={:.3f}, challenger_sharpe={:.3f}, "
            "improvement={:.1f}%",
            inc_sharpe,
            cha_sharpe,
            pct_improvement,
        )

        if pct_improvement > self.promotion_threshold_pct:
            self._promote()
            return True
        return False

    def _promote(self) -> None:
        """Promote challenger to incumbent."""
        record = {
            "timestamp": time.time(),
            "promotion_number": self._promotion_count + 1,
            "incumbent_sharpe": self._sharpe(list(self._incumbent_trades)),
            "challenger_sharpe": self._sharpe(list(self._challenger_trades)),
            "incumbent_trades": len(self._incumbent_trades),
            "challenger_trades": len(self._challenger_trades),
        }
        self._promotions.append(record)
        self._promotion_count += 1

        # Swap functions
        self._incumbent_fn, self._challenger_fn = (
            self._challenger_fn,
            self._incumbent_fn,
        )

        # Reset histories for the next cycle
        self._incumbent_trades.clear()
        self._challenger_trades.clear()
        self._cycle_start_ts = time.time()

        logger.info(
            "ABTest: challenger PROMOTED to incumbent (promotion #{})",
            self._promotion_count,
        )
        self._log.append({"timestamp": time.time(), "event": "promotion", **record})

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """Return a summary of the current A/B test state."""
        return {
            "incumbent_trades": len(self._incumbent_trades),
            "challenger_trades": len(self._challenger_trades),
            "incumbent_sharpe": self._sharpe(list(self._incumbent_trades)),
            "challenger_sharpe": self._sharpe(list(self._challenger_trades)),
            "promotions": self._promotions,
            "promotion_count": self._promotion_count,
            "cycle_start": self._cycle_start_ts,
            "log_events": len(self._log),
        }

    def get_full_log(self) -> List[Dict[str, Any]]:
        """Return the complete event log (selection + outcome events)."""
        return list(self._log)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: List[float], annualisation: float = 252.0) -> float:
        """Compute annualised Sharpe ratio from a list of returns."""
        if len(returns) < 2:
            return 0.0
        import numpy as np

        arr = np.array(returns, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-9:
            return 0.0
        return (mean / std) * math.sqrt(annualisation)
