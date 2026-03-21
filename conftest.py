"""Root conftest.py — ensure project source directories are on sys.path."""
import sys
import os

# Make both the project root and the crypto_trading_bot package's parent paths
# available so that ``from utils.xxx import ...`` and
# ``from crypto_trading_bot.xxx import ...`` both resolve correctly.
_ROOT = os.path.dirname(__file__)
for _p in [
    _ROOT,
    os.path.join(_ROOT, "crypto_trading_bot"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
