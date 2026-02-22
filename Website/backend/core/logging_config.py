"""
Logging configuration.
Sets up a structured console logger compatible with uvicorn's log format.
"""

import logging
import sys

from backend.core.config import settings


def configure_logging() -> None:
    """Configure root logger with level from settings."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (use module __name__)."""
    return logging.getLogger(name)
