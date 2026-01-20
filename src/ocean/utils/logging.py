from __future__ import annotations

import logging
import os


def configure_logging(level: str | int | None = None) -> None:
    """Configure a sane default logger.

    - Respects LOG_LEVEL env var.
    - Prints time, level, module and message.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
