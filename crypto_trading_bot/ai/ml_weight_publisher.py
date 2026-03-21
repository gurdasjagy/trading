import mmap
import os
import struct
import time
from typing import Dict, Any

class MlWeightPublisher:
    """
    Publishes calibrated ML weights to a shared memory segment for the Rust execution engine.
    Uses a seqlock pattern for lock-free reads on the Rust side.
    """
    
    SHM_SIZE = 65536
    MAGIC_BYTES = 0x4D4C5F5747485453
    MAX_SYMBOLS = 1024

    def __init__(self, shm_path: str = "/dev/shm/ml_weights"):
        self.shm_path = shm_path
        self.model_version = 0
        self._init_shm()
        
    def _init_shm(self):
        if not os.path.exists(self.shm_path):
            with open(self.shm_path, "wb") as f:
                f.write(b'\x00' * self.SHM_SIZE)
                
        fd = os.open(self.shm_path, os.O_RDWR)
        self.mmap = mmap.mmap(fd, self.SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        
        # Write magic bytes initially
        struct.pack_into("<Q", self.mmap, 8, self.MAGIC_BYTES)
        
    def publish(self, symbol_weights: Dict[int, Dict[str, float]]):
        """
        Publish new weights.
        symbol_weights is a dict mapping symbol_id (int) to a dict of weights:
        {
            'momentum_weight': float,
            'mean_reversion_weight': float,
            'volatility_weight': float,
            'confidence_floor': float,
            'max_position_scale': float
        }
        """
        self.model_version += 1
        num_symbols = min(len(symbol_weights), self.MAX_SYMBOLS)
        
        # Seqlock write start (odd)
        seq = struct.unpack_into("<I", self.mmap, 0)[0]
        struct.pack_into("<I", self.mmap, 0, seq + 1)
        
        # Memory barrier equivalent in python is not strictly needed for mmap 
        # but we ensure the struct packing happens sequentially.
        
        struct.pack_into("<Q", self.mmap, 16, self.model_version)
        struct.pack_into("<I", self.mmap, 24, num_symbols)
        
        offset = 32
        for symbol_id, weights in list(symbol_weights.items())[:self.MAX_SYMBOLS]:
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
