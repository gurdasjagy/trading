"""Dependency verification script.

Run from the crypto_trading_bot directory:
    python scripts/check_deps.py

Attempts to import every third-party package used by the codebase and reports
which imports succeed or fail, so missing packages can be identified quickly.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

# ---------------------------------------------------------------------------
# (package_to_import, pip_package_name, description)
# ---------------------------------------------------------------------------
_CHECKS: List[Tuple[str, str, str]] = [
    # Exchange connectivity
    ("ccxt", "ccxt>=4.2.0", "Exchange connectivity"),
    ("aiohttp", "aiohttp>=3.9.0", "Async HTTP client"),
    ("websockets", "websockets>=12.0", "WebSocket support"),
    # Web framework
    ("fastapi", "fastapi>=0.109.0", "Web framework"),
    ("uvicorn", "uvicorn[standard]>=0.27.0", "ASGI server"),
    ("jinja2", "jinja2>=3.1.0", "HTML templates"),
    ("multipart", "python-multipart>=0.0.6", "Form data parsing"),
    ("socketio", "python-socketio>=5.11.0", "Socket.IO"),
    ("httpx", "httpx>=0.26.0", "Async HTTP client"),
    # Config
    ("dotenv", "python-dotenv>=1.0.0", "Environment variables"),
    ("pydantic", "pydantic>=2.5.0", "Data validation"),
    ("pydantic_settings", "pydantic-settings>=2.1.0", "Settings management"),
    # Database
    ("sqlalchemy", "sqlalchemy>=2.0.0", "ORM"),
    ("alembic", "alembic>=1.13.0", "DB migrations"),
    # Caching / queue
    ("redis", "redis>=5.0.0", "Redis client"),
    ("celery", "celery>=5.3.0", "Task queue"),
    # Data processing
    ("pandas", "pandas>=2.1.0", "Data analysis"),
    ("polars", "polars>=0.20.0", "Fast DataFrames"),
    ("numpy", "numpy>=1.26.0", "Numerical computing"),
    ("scipy", "scipy>=1.12.0", "Scientific computing"),
    ("pyarrow", "pyarrow>=15.0.0", "Apache Arrow"),
    ("orjson", "orjson>=3.9.0", "Fast JSON"),
    ("matplotlib", "matplotlib>=3.8.0", "Plotting"),
    # Technical analysis
    ("pandas_ta", "pandas-ta>=0.3.14", "Technical indicators"),
    # Machine learning
    ("sklearn", "scikit-learn>=1.4.0", "Machine learning"),
    ("transformers", "transformers>=4.36.0", "HuggingFace Transformers"),
    ("torch", "torch>=2.6.0", "PyTorch (AI/LSTM/Transformer models)"),
    ("optuna", "optuna>=3.4.0", "Hyperparameter optimisation (backtest)"),
    # AI / LLM
    ("openai", "openai>=1.10.0", "OpenAI API"),
    ("anthropic", "anthropic>=0.18.0", "Anthropic API"),
    # Sentiment
    ("vaderSentiment", "vaderSentiment>=3.3.2", "VADER sentiment"),
    ("textblob", "textblob>=0.18.0", "TextBlob sentiment"),
    ("nltk", "nltk>=3.8.0", "Natural language toolkit"),
    # Data sources
    ("telethon", "telethon>=1.34.0", "Telegram client"),
    ("tweepy", "tweepy>=4.14.0", "Twitter/X API"),
    ("praw", "praw>=7.7.0", "Reddit API"),
    ("feedparser", "feedparser>=6.0.0", "RSS/Atom feeds"),
    ("bs4", "beautifulsoup4>=4.12.0", "HTML parsing"),
    ("pytrends", "pytrends>=4.9.0", "Google Trends (google_trends.py)"),
    (
        "googleapiclient",
        "google-api-python-client>=2.100.0",
        "Google API client (youtube_monitor.py)",
    ),
    # Async utilities
    ("aiofiles", "aiofiles>=23.2.0", "Async file I/O (aggregator.py)"),
    # Scheduling
    ("apscheduler", "apscheduler>=3.10.0", "Task scheduler"),
    # Monitoring
    ("prometheus_client", "prometheus-client>=0.19.0", "Prometheus metrics"),
    # Security
    ("cryptography", "cryptography>=42.0.0", "Cryptographic operations"),
    # Logging / CLI
    ("loguru", "loguru>=0.7.0", "Structured logging"),
    ("rich", "rich>=13.7.0", "Rich terminal output"),
    ("click", "click>=8.1.0", "CLI framework"),
]


def _check() -> int:
    """Run all import checks and print a summary table.

    Returns:
        Exit code — 0 if all imports succeeded, 1 if any failed.
    """
    ok: List[Tuple[str, str]] = []
    fail: List[Tuple[str, str, str]] = []

    for import_name, pip_spec, description in _CHECKS:
        try:
            __import__(import_name)
            ok.append((import_name, description))
        except ImportError as exc:
            fail.append((pip_spec, description, str(exc)))

    # ── Summary ──────────────────────────────────────────────────────────
    width = 70
    print("=" * width)
    print(f"{'DEPENDENCY CHECK':^{width}}")
    print("=" * width)

    if ok:
        print(f"\n✅  {len(ok)} packages imported successfully:\n")
        for name, desc in ok:
            print(f"   {name:<30} {desc}")

    if fail:
        print(f"\n❌  {len(fail)} packages MISSING — install with pip:\n")
        for pip_spec, desc, err in fail:
            print(f"   pip install \"{pip_spec}\"")
            print(f"      ({desc})")
            print(f"      Error: {err}\n")

        print("-" * width)
        print("Fix all missing packages with:")
        print(f"   pip install {' '.join(repr(s) for s, _, _ in fail)}")
        print("=" * width)
        return 1

    print("\n✅  All dependency checks passed.")
    print("=" * width)
    return 0


if __name__ == "__main__":
    sys.exit(_check())
