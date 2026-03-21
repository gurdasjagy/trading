"""Volume profile analyzer — point of control and value area."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


class VolumeProfileAnalyzer:
    """Analyses the volume distribution across price levels.

    The volume profile bins OHLCV data into price buckets and calculates
    the volume traded at each price level.  The key outputs are:

    * **POC** — Point of Control (highest volume node).
    * **Value Area** — the price range that contains 70 % of total volume.
    * **High Volume Nodes** — price levels with above-average volume.
    """

    def calculate_profile(self, ohlcv: pd.DataFrame, bins: int = 50) -> Dict[str, object]:
        """Build the volume profile.

        Returns a dict with:
        * ``"price_levels"`` — list of price mid-points for each bin
        * ``"volumes"`` — corresponding aggregated volume
        * ``"bins"`` — number of bins used
        * ``"price_min"`` — lowest price in the dataset
        * ``"price_max"`` — highest price in the dataset
        """
        if ohlcv.empty:
            return {"price_levels": [], "volumes": [], "bins": bins}

        price_min = float(ohlcv["low"].min())
        price_max = float(ohlcv["high"].max())
        price_range = price_max - price_min
        if price_range == 0:
            return {
                "price_levels": [price_min],
                "volumes": [float(ohlcv["volume"].sum())],
                "bins": 1,
                "price_min": price_min,
                "price_max": price_max,
            }

        bin_size = price_range / bins
        price_levels = [price_min + (i + 0.5) * bin_size for i in range(bins)]
        volumes = [0.0] * bins

        for _, row in ohlcv.iterrows():
            candle_low = float(row["low"])
            candle_high = float(row["high"])
            vol = float(row["volume"])
            candle_range = candle_high - candle_low
            for i, level in enumerate(price_levels):
                bin_low = price_min + i * bin_size
                bin_high = bin_low + bin_size
                overlap = max(0.0, min(candle_high, bin_high) - max(candle_low, bin_low))
                if candle_range > 0:
                    volumes[i] += vol * (overlap / candle_range)
                else:
                    volumes[i] += vol / bins

        return {
            "price_levels": price_levels,
            "volumes": volumes,
            "bins": bins,
            "price_min": price_min,
            "price_max": price_max,
        }

    def find_poc(self, profile: Dict[str, object]) -> float:
        """Return the Point of Control (price level with the most volume)."""
        price_levels = profile.get("price_levels", [])
        volumes = profile.get("volumes", [])
        if not price_levels or not volumes:
            return 0.0
        max_idx = int(max(range(len(volumes)), key=lambda i: volumes[i]))  # type: ignore[arg-type]
        return float(price_levels[max_idx])  # type: ignore[index]

    def find_value_area(
        self, profile: Dict[str, object], value_area_pct: float = 0.70
    ) -> Tuple[float, float]:
        """Return the (Value Area Low, Value Area High) price range.

        The value area encompasses *value_area_pct* of the total volume,
        expanding from the POC outward.
        """
        price_levels: List[float] = profile.get("price_levels", [])  # type: ignore[assignment]
        volumes: List[float] = profile.get("volumes", [])  # type: ignore[assignment]
        if not price_levels or not volumes:
            return 0.0, 0.0

        total_vol = sum(volumes)
        target = total_vol * value_area_pct
        poc_idx = int(max(range(len(volumes)), key=lambda i: volumes[i]))

        accumulated = volumes[poc_idx]
        lo_idx = poc_idx
        hi_idx = poc_idx

        while accumulated < target:
            expand_lo = lo_idx > 0
            expand_hi = hi_idx < len(volumes) - 1
            if not expand_lo and not expand_hi:
                break
            lo_gain = volumes[lo_idx - 1] if expand_lo else -1
            hi_gain = volumes[hi_idx + 1] if expand_hi else -1
            if lo_gain >= hi_gain and expand_lo:
                lo_idx -= 1
                accumulated += volumes[lo_idx]
            elif expand_hi:
                hi_idx += 1
                accumulated += volumes[hi_idx]
            else:
                break

        return float(price_levels[lo_idx]), float(price_levels[hi_idx])

    def detect_high_volume_nodes(
        self, profile: Dict[str, object], threshold_multiplier: float = 1.5
    ) -> List[float]:
        """Return price levels with above-average volume.

        A node is considered "high volume" if its volume exceeds
        *threshold_multiplier* × average volume across all bins.
        """
        price_levels: List[float] = profile.get("price_levels", [])  # type: ignore[assignment]
        volumes: List[float] = profile.get("volumes", [])  # type: ignore[assignment]
        if not price_levels or not volumes:
            return []
        avg_vol = sum(volumes) / len(volumes)
        threshold = avg_vol * threshold_multiplier
        return [float(price_levels[i]) for i, v in enumerate(volumes) if v >= threshold]

    # ------------------------------------------------------------------
    # Stop-loss / take-profit integration helpers
    # ------------------------------------------------------------------

    def get_stop_loss_level(
        self,
        profile: Dict[str, object],
        entry: float,
        direction: str,
        buffer_pct: float = 0.001,
    ) -> float:
        """Return a dynamic stop-loss level derived from volume profile levels.

        For **longs**, the stop is placed just below the Value Area Low (VAL).
        For **shorts**, the stop is placed just above the Value Area High (VAH).
        A small buffer (``buffer_pct``) is applied to avoid premature fills.

        Args:
            profile: Volume profile dict from :meth:`calculate_profile`.
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.
            buffer_pct: Buffer fraction applied outside the value area boundary.

        Returns:
            Stop-loss price, or a 2 % fallback when no profile data is present.
        """
        val, vah = self.find_value_area(profile)
        if val == 0.0 and vah == 0.0:
            return entry * (0.98 if direction == "long" else 1.02)

        if direction == "long":
            stop = val * (1.0 - buffer_pct)
        else:
            stop = vah * (1.0 + buffer_pct)
        return max(stop, 0.0)

    def get_take_profit_levels(
        self,
        profile: Dict[str, object],
        entry: float,
        direction: str,
    ) -> List[float]:
        """Return take-profit levels derived from volume profile analysis.

        The Point of Control (POC), Value Area High, and Value Area Low act
        as natural magnet / resistance / support levels for TP placement.

        For **longs**: TP levels are POC (if above entry), VAH, and above-VAH
        high-volume nodes.  For **shorts**: TP levels are POC (if below entry),
        VAL, and below-VAL high-volume nodes.

        Args:
            profile: Volume profile dict from :meth:`calculate_profile`.
            entry: Entry price.
            direction: ``"long"`` or ``"short"``.

        Returns:
            Sorted list of take-profit prices (ascending for longs, descending
            for shorts), empty list when insufficient data.
        """
        poc = self.find_poc(profile)
        val, vah = self.find_value_area(profile)
        hvns = self.detect_high_volume_nodes(profile)

        if poc == 0.0:
            return []

        tp_prices: List[float] = []
        if direction == "long":
            if poc > entry:
                tp_prices.append(poc)
            if vah > entry:
                tp_prices.append(vah)
            tp_prices.extend(n for n in hvns if n > max(vah, entry))
            tp_prices = sorted(set(tp_prices))
        else:
            if poc < entry:
                tp_prices.append(poc)
            if val < entry:
                tp_prices.append(val)
            tp_prices.extend(n for n in hvns if n < min(val, entry))
            tp_prices = sorted(set(tp_prices), reverse=True)

        return tp_prices

    def get_mean_reversion_entry(
        self,
        profile: Dict[str, object],
        current_price: float,
        direction: str,
    ) -> float:
        """Return a mean-reversion entry price using POC as a magnet.

        When price is away from the Point of Control, the POC acts as a
        gravitational target.  This method returns the POC price directly as
        the ideal entry for a mean-reversion trade.

        Args:
            profile: Volume profile dict.
            current_price: Latest market price.
            direction: ``"long"`` or ``"short"``.

        Returns:
            POC price if it represents a valid mean-reversion entry, else
            ``current_price``.
        """
        poc = self.find_poc(profile)
        if poc == 0.0:
            return current_price
        # Only use POC as entry if price is on the correct side
        if direction == "long" and current_price < poc:
            return poc
        if direction == "short" and current_price > poc:
            return poc
        return current_price
