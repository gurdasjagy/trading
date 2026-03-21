"""Anomaly detector for price, volume, and order-book data."""

from typing import Dict, List

import pandas as pd
from loguru import logger

# Z-score thresholds for flagging anomalies
_PRICE_Z_THRESHOLD: float = 3.0
_VOLUME_Z_THRESHOLD: float = 3.5
_ORDERBOOK_IMBALANCE_THRESHOLD: float = 0.75  # bid/ask imbalance ratio


class AnomalyDetector:
    """Detects statistical anomalies in price, volume, and order-book data.

    Uses Z-score analysis and ratio-based heuristics to flag unusual market
    activity that may precede significant price moves.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_price_anomaly(self, prices: List[float]) -> List[Dict]:
        """Detect abnormal price movements using Z-score analysis.

        Args:
            prices: Ordered list of close prices (oldest first).

        Returns:
            List of anomaly dicts with keys ``"index"``, ``"price"``,
            ``"z_score"``, and ``"direction"`` (``"up"`` or ``"down"``).
        """
        if len(prices) < 10:
            return []
        try:
            s = pd.Series(prices, dtype=float)
            returns = s.pct_change().dropna()

            mean = float(returns.mean())
            std = float(returns.std())
            if std == 0:
                return []

            anomalies: List[Dict] = []
            for i, ret in enumerate(returns):
                z = (float(ret) - mean) / std
                if abs(z) >= _PRICE_Z_THRESHOLD:
                    anomalies.append(
                        {
                            "index": i + 1,  # +1 because pct_change drops first row
                            "price": float(prices[i + 1]),
                            "z_score": round(z, 3),
                            "direction": "up" if z > 0 else "down",
                        }
                    )
            return anomalies
        except Exception as exc:
            logger.warning(f"AnomalyDetector.detect_price_anomaly error: {exc}")
            return []

    def detect_volume_anomaly(self, volumes: List[float]) -> List[Dict]:
        """Detect abnormal volume spikes using Z-score analysis.

        Args:
            volumes: Ordered list of volume values (oldest first).

        Returns:
            List of anomaly dicts with keys ``"index"``, ``"volume"``,
            and ``"z_score"``.
        """
        if len(volumes) < 10:
            return []
        try:
            s = pd.Series(volumes, dtype=float)
            mean = float(s.mean())
            std = float(s.std())
            if std == 0:
                return []

            anomalies: List[Dict] = []
            for i, vol in enumerate(volumes):
                z = (float(vol) - mean) / std
                if z >= _VOLUME_Z_THRESHOLD:  # volume spikes are always positive
                    anomalies.append(
                        {
                            "index": i,
                            "volume": float(vol),
                            "z_score": round(z, 3),
                        }
                    )
            return anomalies
        except Exception as exc:
            logger.warning(f"AnomalyDetector.detect_volume_anomaly error: {exc}")
            return []

    def detect_order_book_anomaly(self, orderbook: Dict) -> Dict:
        """Detect imbalances and walls in an order-book snapshot.

        Args:
            orderbook: Dict with ``"bids"`` and ``"asks"`` keys, each a list
                       of ``[price, size]`` pairs ordered from best to worst.

        Returns:
            Dict with keys:
            - ``"imbalance_ratio"``: bid_volume / (bid_volume + ask_volume)
            - ``"direction"``: ``"buy_pressure"``, ``"sell_pressure"``, or ``"balanced"``
            - ``"anomaly_detected"``: bool
            - ``"bid_wall"`` / ``"ask_wall"``: largest individual level size detected
        """
        result: Dict = {
            "imbalance_ratio": 0.5,
            "direction": "balanced",
            "anomaly_detected": False,
            "bid_wall": 0.0,
            "ask_wall": 0.0,
        }
        try:
            bids: List = orderbook.get("bids", [])
            asks: List = orderbook.get("asks", [])
            if not bids or not asks:
                return result

            top_n = min(20, len(bids), len(asks))
            bid_vol = sum(float(b[1]) for b in bids[:top_n])
            ask_vol = sum(float(a[1]) for a in asks[:top_n])
            total = bid_vol + ask_vol

            if total == 0:
                return result

            ratio = bid_vol / total
            result["imbalance_ratio"] = round(ratio, 4)

            bid_wall = max((float(b[1]) for b in bids[:top_n]), default=0.0)
            ask_wall = max((float(a[1]) for a in asks[:top_n]), default=0.0)
            result["bid_wall"] = round(bid_wall, 4)
            result["ask_wall"] = round(ask_wall, 4)

            if ratio >= _ORDERBOOK_IMBALANCE_THRESHOLD:
                result["direction"] = "buy_pressure"
                result["anomaly_detected"] = True
            elif ratio <= (1 - _ORDERBOOK_IMBALANCE_THRESHOLD):
                result["direction"] = "sell_pressure"
                result["anomaly_detected"] = True

            return result
        except Exception as exc:
            logger.warning(f"AnomalyDetector.detect_order_book_anomaly error: {exc}")
            return result

    def score_anomaly(self, data: Dict) -> float:
        """Compute a composite anomaly score from a pre-computed anomaly result dict.

        Accepts the output of any of the ``detect_*`` methods (or a merged dict)
        and returns a severity score in [0, 1].

        Args:
            data: Dict output from one of the detection methods, or a merged
                  dict containing keys from multiple methods.

        Returns:
            Float in [0.0, 1.0]; higher = more anomalous.
        """
        try:
            score = 0.0

            # Z-score contribution (price / volume anomalies)
            z = abs(float(data.get("z_score", 0.0)))
            if z > 0:
                score += min(1.0, z / 6.0) * 0.5  # max contribution 0.5

            # Order-book imbalance contribution
            ratio = float(data.get("imbalance_ratio", 0.5))
            imbalance = abs(ratio - 0.5) * 2  # 0 = balanced, 1 = fully one-sided
            score += imbalance * 0.3

            # Boolean anomaly flag
            if data.get("anomaly_detected", False):
                score += 0.2

            return min(1.0, round(score, 4))
        except Exception as exc:
            logger.warning(f"AnomalyDetector.score_anomaly error: {exc}")
            return 0.0
