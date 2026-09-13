"""Centralized path configuration.

All file paths are resolved relative to the project root.
Override via environment variables (e.g. BOOKREC_DATA_DIR).
"""
import os
from pathlib import Path

# Project root: 3 levels up from this file (src/config/paths.py -> .../ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory constants
DATA_DIR = Path(os.environ.get("BOOKREC_DATA_DIR", PROJECT_ROOT / "DataLoad"))
MODEL_DIR = Path(os.environ.get("BOOKREC_MODEL_DIR", PROJECT_ROOT / "MyModule"))
OUTPUT_DIR = Path(os.environ.get("BOOKREC_OUTPUT_DIR", PROJECT_ROOT / "models"))
LOG_DIR = Path(os.environ.get("BOOKREC_LOG_DIR", PROJECT_ROOT / "logs"))
PLOT_DIR = Path(os.environ.get("BOOKREC_PLOT_DIR", PROJECT_ROOT / "plots"))
SUBMISSION_DIR = Path(os.environ.get("BOOKREC_SUBMISSION_DIR", PROJECT_ROOT / "submissions"))
REPORTS_DIR = Path(os.environ.get("BOOKREC_REPORTS_DIR", PROJECT_ROOT / "reports"))

# Ensure output directories exist
for _d in (LOG_DIR, PLOT_DIR, OUTPUT_DIR, SUBMISSION_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Specific data files
INTER_REEVALUATION_CSV = DATA_DIR / "inter_reevaluation.csv"
ITEM_CSV = DATA_DIR / "item.csv"
USER_CSV = DATA_DIR / "user.csv"
USER_INTER_MATRIX = DATA_DIR / "user_inter_matrix.parquet"
BOOK_VEC = DATA_DIR / "book_vec.parquet"
USER_VEC = DATA_DIR / "user_vec.parquet"

# Mapping / intermediate files
ID_MAPPINGS = PROJECT_ROOT / "id_mappings.json"
INTER2USER = DATA_DIR / "inter2user.json"
INTER2BOOK = DATA_DIR / "inter2book.json"
INTER2RATING = DATA_DIR / "inter2rating.json"
