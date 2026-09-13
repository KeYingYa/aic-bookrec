"""Configuration module for the book recommendation system."""

from src.config.paths import (
    PROJECT_ROOT, DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR, PLOT_DIR,
    SUBMISSION_DIR, REPORTS_DIR, INTER_REEVALUATION_CSV, ITEM_CSV,
    USER_INTER_MATRIX, BOOK_VEC, ID_MAPPINGS, INTER2USER, INTER2BOOK,
    INTER2RATING,
)
from src.config.base import BaseConfig
from src.config.mf_config import MFConfig
from src.config.cbmf_config import CBMFConfig

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "MODEL_DIR", "OUTPUT_DIR", "LOG_DIR",
    "PLOT_DIR", "SUBMISSION_DIR", "REPORTS_DIR", "INTER_REEVALUATION_CSV",
    "ITEM_CSV", "USER_INTER_MATRIX", "BOOK_VEC", "ID_MAPPINGS", "INTER2USER",
    "INTER2BOOK", "INTER2RATING", "BaseConfig", "MFConfig", "CBMFConfig",
]
