"""Liquidation Magnet Strategy — trades toward large liquidation clusters."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from strategy.base_strategy import BaseStrategy, Signal


class LiquidationMagnetStrategy(BaseStrategy):
    """Identifies price levels with concentrated liquidations and generates signals toward them.
    
    Strategy Logic:
    - Fetches liquidation cluster data from LiquidationHeatmapSource
    - Identifies clusters >$50M in size
    - Generates signals toward the largest cluster within 5% of current price
    - Confidence scales with cluster size (larger clusters = higher confidence)
    - Stop-loss placed at 1.5× ATR from entry
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        enabled: bool = True,
        min_cluster_size_usd: float = 50_000_000.0,
        max_distance_pct: float = 0.05,
    ) -> None:
        super().__init__(
            name="liquidation_magnet",
            symbols=symbols,
            timeframe=timeframe,
            enabled=enabled,
        )
        self._min_cluster_size = min_cluster_size_usd
        self._max_distance_pct = max_distance_pct
        self._liquidation_source = None  # Injected externally

    def set_liquidation_source(self, source: Any) -> None:
        """Inject the LiquidationHeatmapSource instance."""
        self._liquidation_source = source

    async def generate_signal(self, symbol: str) -> Signal:
        try:
            # Fetch current price
            ohlcv = await self._get_ohlcv(symbol, limit=50)
            if len(ohlcv) < 20:
                return self._neutral_signal(symbol, "Insufficient OHLCV data")
            
            current_price = float(ohlcv["close"].iloc[-1])
            atr = self._calculate_atr(ohlcv)
            
            # Fetch liquidation clusters
            if self._liquidation_source is None:
                return self._neutral_signal(symbol, "Liquidation source not configured")
            
            base_symbol = symbol.split("/")[0]
            clusters = await self._liquidation_source.fetch_liquidation_clusters(base_symbol)
            
            if not clusters:
                return self._neutral_signal(symbol, "No liquidation clusters found")
            
            # Filter clusters within max_distance_pct of current price
            nearby_clusters = [
                c for c in clusters
                if abs(c["price"] - current_price) / current_price <= self._max_distance_pct
                and c["size_usd"] >= self._min_cluster_size
            ]
            
            if not nearby_clusters:
                return self._neutral_signal(
                    symbol,
                    f"No large clusters within {self._max_distance_pct*100:.0f}% of price"
                )
            
            # Find largest cluster
            largest_cluster = max(nearby_clusters, key=lambda c: c["size_usd"])
            cluster_price = largest_cluster["price"]
            cluster_size = largest_cluster["size_usd"]
            cluster_side = largest_cluster["side"]
            
            # Determine direction: trade toward the cluster
            if cluster_price > current_price:
                direction = "long"
                stop_loss = current_price - atr * 1.5
                take_profit = cluster_price
            else:
                direction = "short"
                stop_loss = current_price + atr * 1.5
                take_profit = cluster_price
            
            # Confidence scales with cluster size (50M = 0.6, 200M+ = 0.85)
            base_confidence = 0.6
            size_bonus = min(0.25, (cluster_size - 50_000_000) / 600_000_000)
            confidence = base_confidence + size_bonus
            
            # Strength based on distance to cluster
            distance_pct = abs(cluster_price - current_price) / current_price
            strength = min(1.0, 1.0 - (distance_pct / self._max_distance_pct))
            
            return Signal(
                symbol=symbol,
                direction=direction,
                strength=round(strength, 3),
                confidence=round(confidence, 3),
                strategy_name=self.name,
                reasoning=(
                    f"Liquidation magnet: ${cluster_size/1e6:.0f}M cluster at "
                    f"${cluster_price:.2f} ({cluster_side}), "
                    f"{distance_pct*100:.1f}% from current ${current_price:.2f}"
                ),
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=2,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] generate_signal error for {symbol}: {exc}")
            return self._neutral_signal(symbol, f"Error: {exc}")

    async def should_close(self, position: Any, data: Dict[str, Any]) -> bool:
        """Close if price reaches the liquidation cluster target."""
        symbol = getattr(position, "symbol", None) or data.get("symbol", "")
        ohlcv = await self._get_ohlcv(symbol, limit=5)
        if ohlcv.empty:
            return False
        
        current_price = float(ohlcv["close"].iloc[-1])
        entry_price = float(getattr(position, "entry_price", 0.0))
        side = str(getattr(position, "side", "long")).lower()
        
        # Close if we've reached the cluster (5% move in target direction)
        if side == "long" and current_price >= entry_price * 1.05:
            return True
        if side == "short" and current_price <= entry_price * 0.95:
            return True
        
        return False

    async def calculate_parameters(self, symbol: str, direction: str) -> Dict[str, Any]:
        ohlcv = await self._get_ohlcv(symbol, limit=50)
        atr = self._calculate_atr(ohlcv)
        last_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 0.0
        sl_pct = (atr * 1.5 / last_price) if last_price > 0 else 0.03
        
        return {
            "position_size_pct": 0.04,
            "stop_loss_pct": min(0.05, sl_pct),
            "take_profit_pct": min(0.08, sl_pct * 2.0),
            "leverage": 2,
        }
