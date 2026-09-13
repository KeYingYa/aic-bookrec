"""Shared logging setup for the book recommendation system."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple


def setup_logging(log_prefix: str = "training",
                  log_dir: Path | str | None = None) -> Tuple[logging.Logger, str]:
    """Configure logging with file and console handlers.

    Args:
        log_prefix: Prefix for the log filename (e.g. "mf", "cbmf").
        log_dir: Directory for log files. Defaults to project logs/ directory.

    Returns:
        Tuple of (logger, log_file_path).
    """
    if log_dir is None:
        from src.config.paths import LOG_DIR
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(log_dir / f"{log_prefix}_{timestamp}.log")

    # Prevent adding duplicate handlers on re-import
    logger = logging.getLogger(log_prefix)
    if logger.handlers:
        return logger, log_file

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file
