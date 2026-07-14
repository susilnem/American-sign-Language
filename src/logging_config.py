"""Shared logging setup: colored console + rotating file under logs/."""

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

_configured = False


def configure_logging(level="INFO"):
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    logger.add(LOG_FILE, level=level, rotation="1 MB", retention=3)
