#!/usr/bin/env python3
"""download_models.py — Download the HuggingFace sentiment transformer model.

Downloads:
    cardiffnlp/twitter-roberta-base-sentiment-latest

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import sys

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _check_transformers() -> None:
    """Ensure the transformers library is installed."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'transformers' package is not installed.\n"
            "Install with: pip install transformers[sentencepiece]",
            file=sys.stderr,
        )
        sys.exit(1)


def download_sentiment_model(model_name: str = _MODEL_NAME) -> bool:
    """Download *model_name* from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        True if the download succeeded, False otherwise.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        print(f"Downloading tokenizer: {model_name} …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("  Tokenizer downloaded ✓")

        print(f"Downloading model: {model_name} …")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("  Model downloaded ✓")

        # Quick smoke-test
        import torch  # type: ignore

        inputs = tokenizer("Bitcoin is mooning!", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"  Smoke-test passed — logits shape: {tuple(outputs.logits.shape)} ✓")
        return True

    except Exception as exc:
        print(f"ERROR: Download failed — {exc}", file=sys.stderr)
        return False


def main() -> None:
    """Entry point: verify environment and download model."""
    print()
    print("=" * 60)
    print("  CryptoBot — Sentiment Model Downloader")
    print("=" * 60)
    print()

    _check_transformers()

    success = download_sentiment_model(_MODEL_NAME)

    if success:
        print()
        print(f"✅  Model '{_MODEL_NAME}' is ready.")
        print("    It will be loaded automatically by the AI sentiment module.")
    else:
        print()
        print("❌  Model download failed. Check your internet connection and try again.")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
