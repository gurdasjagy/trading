"""Logging configuration using loguru with rich formatting and file rotation."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{message}"
)

LOG_FORMAT_PLAIN = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | " "{name}:{function}:{line} | {message}"
)

LOG_FORMAT_JSON = (
    '{{"time":"{time:YYYY-MM-DD HH:mm:ss.SSS}",'
    '"level":"{level}",'
    '"name":"{name}",'
    '"function":"{function}",'
    '"line":{line},'
    '"message":"{message}"}}'
)


def configure_logging(
    log_level: str = "INFO",
    log_file: str = "data/logs/bot.log",
    json_logs: bool = False,
    colorize: bool = True,
) -> None:
    """Configure the global loguru logger for the trading bot.

    Removes loguru's default handler and adds:
    - A stderr handler with rich coloured (or plain) formatting.
    - A rotating file handler (10 MB rotation, 7-day retention, gzip compression).
    - A separate error-only log file for quick error triage.

    Args:
        log_level: Minimum log level to capture (e.g. ``"INFO"``).
        log_file: Path to the primary rotating log file.
        json_logs: When *True*, write structured JSON to the file handler.
        colorize: When *True*, apply ANSI colour codes to the stderr handler.
    """
    # Remove every existing handler (including loguru's default stderr sink).
    logger.remove()

    # ── stderr handler ────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=LOG_FORMAT if colorize else LOG_FORMAT_PLAIN,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
        enqueue=False,
    )

    # ── Primary rotating file handler ─────────────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=log_level.upper(),
        format=LOG_FORMAT_JSON if json_logs else LOG_FORMAT_PLAIN,
        rotation="00:00",   # Rotate daily at midnight
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        encoding="utf-8",
    )

    # ── Error-only file handler ───────────────────────────────────────────────
    error_log_path = log_path.parent / "error.log"
    logger.add(
        str(error_log_path),
        level="ERROR",
        format=LOG_FORMAT_PLAIN,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        encoding="utf-8",
    )

    logger.info(
        "Logging configured: level={} file={} json={}",
        log_level.upper(),
        log_file,
        json_logs,
    )
