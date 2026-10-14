"""Structured logging setup used by every entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: Optional[str | Path] = None) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Safe to call more than once; existing handlers installed by this function are replaced.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_wrestleai", False):
            root.removeHandler(handler)
    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console._wrestleai = True  # type: ignore[attr-defined]
    root.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._wrestleai = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)
    root.setLevel(level if isinstance(level, int) else level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (thin wrapper kept for a single import point)."""
    return logging.getLogger(name)
