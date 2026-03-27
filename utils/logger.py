from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure a clean, library-friendly logger.

    - Uses stdlib logging (no external deps).
    - Safe to call multiple times (idempotent-ish).
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    root.addHandler(handler)
    root.setLevel(level.upper())

