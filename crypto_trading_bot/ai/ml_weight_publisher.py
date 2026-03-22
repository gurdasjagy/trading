import mmap
import os
import struct
import time
from collections import deque
from typing import Dict, Any, Optional

class StrategyPerformanceTracker:
    """Tracks strategy performance over a rolling 24h window.
    
    Computes momentum_weight vs mean_reversion_weight based on which
    strategies performed better in recent history.
    """
    
    def __init__(self, window_hours: int = 24):
        self.window_seconds = window_hours * 3600
        # Per-symbol performance history: {symbol_id: deque[(timestamp, return, strategy_type)]}
        self.performance_history: Dict[int, deque] = {}
        
    def record_trade(self, symbol_id: int, return_pct: float, strategy_type: str, timestamp: float = None):
        """Record a trade result.
        
        Args:
            symbol_id: Symbol ID (1=BTC, 2=ETH, etc.)
            return_pct: Trade return as percentage (e.g., 0.05 = 5%)
            strategy_type: "momentum" or "mean_reversion"
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()
            
        if symbol_id not in self.performance_history:
            self.performance_history[symbol_id] = deque(maxlen=1000)
            
        self.performance_history[symbol_id].append((timestamp, return_pct, strategy_type))
        
    def get_weights(self, symbol_id: int) -> tuple[float, float]:
        """Calculate momentum_weight and mean_reversion_weight for a symbol.
        
        Returns:
            (momentum_weight, mean_reversion_weight) tuple
        """
        if symbol_id not in self.performance_history:
            return (1.0, 0.0)  # Default to momentum
            
        history = self.performance_history[symbol_id]
        if not history:
            return (1.0, 0.0)
            
        # Filter to last 24 hours
        cutoff = time.time() - self.window_seconds
        recent = [(ts, ret, stype) for ts, ret, stype in history if ts >= cutoff]
        
        if not recent:
            return (1.0, 0.0)
            
        # Calculate average returns by strategy type
        momentum_returns = [ret for _, ret, stype in recent if stype == "momentum"]
        mean_rev_returns = [ret for _, ret, stype in recent if stype == "mean_reversion"]
        
        avg_momentum = sum(momentum_returns) / len(momentum_returns) if momentum_returns else 0.0
        avg_mean_rev = sum(mean_rev_returns) / len(mean_rev_returns) if mean_rev_returns else 0.0
        
        # Normalize to weights (0.0 to 1.0)
        total = abs(avg_momentum) + abs(avg_mean_rev)
        if total > 0.0:
            momentum_weight = max(0.0, avg_momentum) / total
            mean_rev_weight = max(0.0, avg_mean_rev) / total
        else:
            momentum_weight = 1.0
            mean_rev_weight = 0.0
            
        return (momentum_weight, mean_rev_weight)

class MlWeightPublisher:
    """
    Publishes calibrated ML weights to a shared memory segment for the Rust execution engine.
    Uses a seqlock pattern for lock-free reads on the Rust side.
    """
    
    SHM_SIZE = 65536
    MAGIC_BYTES = 0x4D4C5F5747485453
    MAX_SYMBOLS = 1024

    def __init__(self, shm_path: str = "/dev/shm/ml_weights", regime_shm_path: str = "/dev/shm/regime_weights"):
        self.shm_path = shm_path
        self.regime_shm_path = regime_shm_path
        self.model_version = 0
        self.performance_tracker = StrategyPerformanceTracker()
        self._init_shm()
        self._init_regime_shm()
        
    def _init_shm(self):
        if not os.path.exists(self.shm_path):
            with open(self.shm_path, "wb") as f:
                f.write(b'\x00' * self.SHM_SIZE)
                
        fd = os.open(self.shm_path, os.O_RDWR)
        self.mmap = mmap.mmap(fd, self.SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        
        # Write magic bytes initially
        struct.pack_into("<Q", self.mmap, 8, self.MAGIC_BYTES)
        
    def _init_regime_shm(self):
        """Initialize regime weights shared memory for reading."""
        try:
            if os.path.exists(self.regime_shm_path):
                fd = os.open(self.regime_shm_path, os.O_RDONLY)
                self.regime_mmap = mmap.mmap(fd, 65536, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
            else:
                self.regime_mmap = None
        except Exception as e:
            print(f"Failed to init regime SHM: {e}")
            self.regime_mmap = None
            
    def _read_regime_state(self) -> Optional[Dict[str, Any]]:
        """Read current regime state from shared memory using seqlock pattern.
        
        Returns:
            Dict with regime data or None on error
        """
        if self.regime_mmap is None:
            return None
            
        try:
            # Seqlock read loop
            max_retries = 10
            for _ in range(max_retries):
                seq_before = struct.unpack_from("<I", self.regime_mmap, 0)[0]
                if seq_before % 2 != 0:
                    continue  # Writer is active, retry
                    
                # Read data
                magic = struct.unpack_from("<Q", self.regime_mmap, 4)[0]
                if magic != 0x5245474D5F574754:  # "REGM_WGT"
                    return None
                    
                volatility_regime = struct.unpack_from("<I", self.regime_mmap, 12)[0]
                
                seq_after = struct.unpack_from("<I", self.regime_mmap, 0)[0]
                if seq_before == seq_after:
                    # Successful read
                    regime_map = {0: "Low", 1: "Normal", 2: "High", 3: "Extreme"}
                    return {
                        "volatility_regime": regime_map.get(volatility_regime, "Normal")
                    }
            return None
        except Exception as e:
            print(f"Failed to read regime state: {e}")
            return None
        
    def publish(self, symbol_weights: Dict[int, Dict[str, float]]):
        """
        Publish new weights with performance tracking and regime-aware adjustments.
        
        symbol_weights is a dict mapping symbol_id (int) to a dict of weights:
        {
            'momentum_weight': float,
            'mean_reversion_weight': float,
            'volatility_weight': float,
            'confidence_floor': float,
            'max_position_scale': float
        }
        """
        # Task 13: Apply strategy performance tracking
        adjusted_weights = {}
        for symbol_id, weights in symbol_weights.items():
            # Get performance-based weights
            perf_momentum, perf_mean_rev = self.performance_tracker.get_weights(symbol_id)
            
            # Blend with input weights (70% performance, 30% input)
            momentum_weight = weights.get('momentum_weight', 1.0) * 0.3 + perf_momentum * 0.7
            mean_reversion_weight = weights.get('mean_reversion_weight', 0.0) * 0.3 + perf_mean_rev * 0.7
            
            adjusted_weights[symbol_id] = {
                'momentum_weight': momentum_weight,
                'mean_reversion_weight': mean_reversion_weight,
                'volatility_weight': weights.get('volatility_weight', 1.0),
                'confidence_floor': weights.get('confidence_floor', 0.0),
                'max_position_scale': weights.get('max_position_scale', 1.0)
            }
        
        # Task 14: Apply regime-aware weight adjustments
        regime_state = self._read_regime_state()
        if regime_state:
            vol_regime = regime_state.get("volatility_regime", "Normal")
            
            for symbol_id in adjusted_weights:
                if vol_regime == "High":
                    # High volatility: increase confidence floor, reduce position scale
                    adjusted_weights[symbol_id]['confidence_floor'] = max(
                        adjusted_weights[symbol_id]['confidence_floor'], 0.6
                    )
                    adjusted_weights[symbol_id]['max_position_scale'] *= 0.5
                elif vol_regime == "Extreme":
                    # Extreme volatility: very conservative
                    adjusted_weights[symbol_id]['confidence_floor'] = max(
                        adjusted_weights[symbol_id]['confidence_floor'], 0.8
                    )
                    adjusted_weights[symbol_id]['max_position_scale'] *= 0.25
        
        self.model_version += 1
        num_symbols = min(len(adjusted_weights), self.MAX_SYMBOLS)
        
        # Seqlock write start (odd)
        seq = struct.unpack_into("<I", self.mmap, 0)[0]
        struct.pack_into("<I", self.mmap, 0, seq + 1)
        
        # Memory barrier equivalent in python is not strictly needed for mmap 
        # but we ensure the struct packing happens sequentially.
        
        struct.pack_into("<Q", self.mmap, 16, self.model_version)
        struct.pack_into("<I", self.mmap, 24, num_symbols)
        
        offset = 32
        for symbol_id, weights in list(adjusted_weights.items())[:self.MAX_SYMBOLS]:
            # SymbolWeight: symbol_id (H), pad (H), 5x float (5f) = 24 bytes
            struct.pack_into(
                "<HHfffff", 
                self.mmap, 
                offset,
                symbol_id,
                0, # pad
                weights.get('momentum_weight', 1.0),
                weights.get('mean_reversion_weight', 0.0),
                weights.get('volatility_weight', 1.0),
                weights.get('confidence_floor', 0.0),
                weights.get('max_position_scale', 1.0)
            )
            offset += 24
            
        # Seqlock write end (even)
        struct.pack_into("<I", self.mmap, 0, seq + 2)

if __name__ == "__main__":
    # Test publisher
    pub = MlWeightPublisher()
    test_weights = {
        1: { # BTC_USDT
            'momentum_weight': 0.8,
            'mean_reversion_weight': 0.2,
            'volatility_weight': 1.2,
            'confidence_floor': 0.4,
            'max_position_scale': 2.0
        },
        2: { # ETH_USDT
            'momentum_weight': 0.5,
            'mean_reversion_weight': 0.6,
            'volatility_weight': 0.8,
            'confidence_floor': 0.3,
            'max_position_scale': 1.5
        }
    }
    while True:
        pub.publish(test_weights)
        print(f"Published model version {pub.model_version}")
        time.sleep(60)
