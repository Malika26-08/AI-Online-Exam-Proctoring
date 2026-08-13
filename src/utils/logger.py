"""
Logging Utility for Online Exam Proctoring Pipeline.
"""

import logging
import sys
from pathlib import Path
from src.config import RESULTS_DIR


def get_logger(name: str = "exam_proctoring", log_file: str = "app.log") -> logging.Logger:
    """Configures and returns a logger instance with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    # Formatting
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(RESULTS_DIR / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        console_handler.setFormatter(formatter)

    return logger
