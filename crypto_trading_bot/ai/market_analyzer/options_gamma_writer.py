"""Options Gamma Exposure Writer for Deribit Options Data.

Fetches BTC and ETH options data from Deribit public API, calculates gamma flip
levels, and writes to /dev/shm/gamma_exposure in binary format for the Rust engine.

Architecture:
    Deribit API → GammaExposureWriter → /dev/shm/gamma_exposure → Rust gamma_shm.rs

The gamma flip level is calculated as:
    gamma_flip = sum(strike * gamma * open_interest) / sum(gamma * open_interest)

This represents the price level where market maker gamma exposure flips from
positive to negative, creating a support/resistance zone.
"""

import asyncio
import mmap
import os
import struct
import time
from typing import Dict, Optional

import aiohttp
from loguru import logger


class GammaExposureWriter:
    """Fetches options data from Deribit and writes gamma flip levels to SHM.
    
    Binary format (matches gamma_shm.rs):
        - seqlock: u32 (4 bytes)
        - magic: u64 (8 bytes) = 0x47414D4D415F5348 ("GAMMA_SH")
        - btc_flip: f64 (8 bytes)
        - eth_flip: f64 (8 bytes)
        - padding: 4 bytes
        Total: 32 bytes
    """
    
    SHM_SIZE = 65536
    MAGIC_BYTES = 0x47414D4D415F5348  # "GAMMA_SH" in hex
    DERIBIT_API_BASE = "https://www.deribit.com/api/v2"
    
    def __init__(self, shm_path: str = "/dev/shm/gamma_exposure"):
        self.shm_path = shm_path
        self.seqlock = 0
        self._init_shm()
        
    def _init_shm(self):
        """Initialize shared memory file."""
        if not os.path.exists(self.shm_path):
            with open(self.shm_path, "wb") as f:
                f.write(b'\x00' * self.SHM_SIZE)
                
        fd = os.open(self.shm_path, os.O_RDWR)
        self.mmap = mmap.mmap(fd, self.SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        
        # Write magic bytes initially
        struct.pack_into("<Q", self.mmap, 4, self.MAGIC_BYTES)
        logger.info(f"GammaExposureWriter initialized at {self.shm_path}")
        
    async def fetch_options_data(self, currency: str) -> Optional[Dict]:
        """Fetch options book summary from Deribit public API.
        
        Args:
            currency: "BTC" or "ETH"
            
        Returns:
            Dict with options data or None on error
        """
        url = f"{self.DERIBIT_API_BASE}/public/get_book_summary_by_currency"
        params = {
            "currency": currency,
            "kind": "option"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("result", [])
                    else:
                        logger.warning(f"Deribit API returned status {resp.status} for {currency}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Deribit API timeout for {currency}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch {currency} options data: {e}")
            return None
            
    def calculate_gamma_flip(self, options_data: list) -> float:
        """Calculate gamma flip level from options data.
        
        Formula:
            gamma_flip = sum(strike * gamma * open_interest) / sum(gamma * open_interest)
            
        Args:
            options_data: List of option instruments from Deribit
            
        Returns:
            Gamma flip price level (float)
        """
        if not options_data:
            return 0.0
            
        numerator = 0.0
        denominator = 0.0
        
        for instrument in options_data:
            try:
                # Extract strike price from instrument name (e.g., "BTC-31DEC21-50000-C")
                parts = instrument.get("instrument_name", "").split("-")
                if len(parts) < 4:
                    continue
                    
                strike = float(parts[2])
                
                # Get gamma and open interest from greeks
                greeks = instrument.get("greeks", {})
                gamma = greeks.get("gamma", 0.0)
                open_interest = instrument.get("open_interest", 0.0)
                
                if gamma != 0.0 and open_interest > 0.0:
                    numerator += strike * gamma * open_interest
                    denominator += gamma * open_interest
                    
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed instrument: {e}")
                continue
                
        if denominator > 0.0:
            return numerator / denominator
        else:
            return 0.0
            
    async def update_and_publish(self):
        """Fetch latest options data and write gamma flip levels to SHM."""
        # Fetch BTC and ETH options data
        btc_data = await self.fetch_options_data("BTC")
        eth_data = await self.fetch_options_data("ETH")
        
        # Calculate gamma flip levels
        btc_flip = self.calculate_gamma_flip(btc_data) if btc_data else 0.0
        eth_flip = self.calculate_gamma_flip(eth_data) if eth_data else 0.0
        
        # Write to shared memory with seqlock
        self.seqlock += 1
        
        # Seqlock write start (odd)
        struct.pack_into("<I", self.mmap, 0, self.seqlock)
        
        # Write data
        struct.pack_into("<Q", self.mmap, 4, self.MAGIC_BYTES)
        struct.pack_into("<d", self.mmap, 12, btc_flip)
        struct.pack_into("<d", self.mmap, 20, eth_flip)
        
        # Seqlock write end (even)
        self.seqlock += 1
        struct.pack_into("<I", self.mmap, 0, self.seqlock)
        
        logger.info(f"Gamma exposure updated: BTC flip={btc_flip:.2f}, ETH flip={eth_flip:.2f}")
        
    def close(self):
        """Close shared memory mapping."""
        if hasattr(self, 'mmap'):
            self.mmap.close()


async def main():
    """Test the gamma exposure writer."""
    writer = GammaExposureWriter()
    
    try:
        while True:
            await writer.update_and_publish()
            await asyncio.sleep(300)  # Update every 5 minutes
    except KeyboardInterrupt:
        logger.info("Gamma exposure writer stopped")
    finally:
        writer.close()


if __name__ == "__main__":
    asyncio.run(main())
