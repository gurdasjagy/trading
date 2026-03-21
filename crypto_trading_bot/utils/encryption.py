"""API key encryption/decryption using the Fernet symmetric scheme."""

import base64
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from loguru import logger


class KeyManager:
    """Handles encryption of sensitive data (API keys) using Fernet."""

    def __init__(self, encryption_key: Optional[str] = None) -> None:
        if encryption_key:
            raw = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
            self._fernet: Optional[Fernet] = Fernet(raw)
        else:
            self._fernet = None

    @staticmethod
    def generate_key() -> str:
        """Generate and return a new URL-safe base64-encoded Fernet key."""
        return Fernet.generate_key().decode()

    def encrypt(self, data: str) -> str:
        """
        Encrypt *data* and return a base64-encoded ciphertext string.

        Raises:
            RuntimeError: if no encryption key was provided at construction.
        """
        if self._fernet is None:
            raise RuntimeError("No encryption key configured.")
        ciphertext = self._fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(ciphertext).decode()

    def decrypt(self, data: str) -> str:
        """
        Decrypt a base64-encoded ciphertext and return the plaintext string.

        Raises:
            RuntimeError: if no encryption key was provided at construction.
            ValueError: if decryption fails (invalid token / wrong key).
        """
        if self._fernet is None:
            raise RuntimeError("No encryption key configured.")
        try:
            raw_ciphertext = base64.urlsafe_b64decode(data.encode())
            return self._fernet.decrypt(raw_ciphertext).decode()
        except InvalidToken as exc:
            logger.error("Decryption failed — invalid token or wrong key.")
            raise ValueError("Decryption failed: invalid token or wrong key.") from exc

    def encrypt_api_keys(self, keys: dict) -> dict:
        """Encrypt every string value in *keys* and return a new dict."""
        encrypted: dict = {}
        for k, v in keys.items():
            if isinstance(v, str):
                encrypted[k] = self.encrypt(v)
            else:
                encrypted[k] = v
        return encrypted

    def decrypt_api_keys(self, encrypted: dict) -> dict:
        """Decrypt every string value in *encrypted* and return a new dict."""
        decrypted: dict = {}
        for k, v in encrypted.items():
            if isinstance(v, str):
                try:
                    decrypted[k] = self.decrypt(v)
                except ValueError:
                    logger.warning(f"Could not decrypt key {k!r} — leaving as-is.")
                    decrypted[k] = v
            else:
                decrypted[k] = v
        return decrypted
