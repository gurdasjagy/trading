#!/usr/bin/env python3
"""generate_keys.py — Generate cryptographic keys for the .env configuration.

Usage:
    python scripts/generate_keys.py
"""

from __future__ import annotations

import secrets
import sys


def generate_fernet_key() -> str:
    """Generate a URL-safe base64-encoded Fernet encryption key.

    Returns:
        Fernet key as a decoded string.
    """
    try:
        from cryptography.fernet import Fernet  # type: ignore

        return Fernet.generate_key().decode()
    except ImportError:
        print(
            "WARNING: 'cryptography' package not installed.\n"
            "Install with: pip install cryptography\n"
            "Falling back to a random 32-byte hex string (not a valid Fernet key).",
            file=sys.stderr,
        )
        return secrets.token_hex(32)


def generate_secret_key() -> str:
    """Generate a cryptographically secure 32-byte random hex secret key.

    Returns:
        64-character hexadecimal string.
    """
    return secrets.token_hex(32)


def main() -> None:
    """Print newly generated keys to stdout with clear labels."""
    print()
    print("=" * 60)
    print("  CryptoBot Key Generator")
    print("=" * 60)
    print()

    fernet_key = generate_fernet_key()
    secret_key = generate_secret_key()

    print("Fernet Encryption Key (ENCRYPTION_KEY):")
    print(f"  {fernet_key}")
    print()
    print("Secret Key (SECRET_KEY):")
    print(f"  {secret_key}")
    print()
    print("-" * 60)
    print("Add these lines to your .env file:")
    print()
    print(f"ENCRYPTION_KEY={fernet_key}")
    print(f"SECRET_KEY={secret_key}")
    print()
    print("⚠️  Keep these keys secret — never commit them to version control.")
    print()


if __name__ == "__main__":
    main()
