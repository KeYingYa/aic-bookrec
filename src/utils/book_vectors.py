"""Book vector loading utilities (unified from duplicated CBMF.py copies)."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import BOOK_VEC

logger = logging.getLogger(__name__)


def _parse_vector(vector_data: Any) -> np.ndarray:
    """Parse a single book vector from various formats.

    Handles: str ("[0.1, 0.2, ...]"), list, np.ndarray, etc.
    """
    if isinstance(vector_data, str):
        vector_str = vector_data.strip("[]")
        return np.fromstring(vector_str, sep=",", dtype=np.float32)
    elif isinstance(vector_data, list):
        return np.array(vector_data, dtype=np.float32)
    else:
        return np.array(vector_data, dtype=np.float32)


def load_book_vectors(
    file_path: Path | str | None = None,
) -> tuple[np.ndarray, dict[Any, int]]:
    """Load book vectors from a parquet file.

    Args:
        file_path: Path to book_vec.parquet. Defaults to DataLoad/book_vec.parquet.

    Returns:
        Tuple of (book_vectors: np.ndarray shape (n_books, dim),
                  book_id_to_index: dict mapping original book_id -> 0-based index).
    """
    if file_path is None:
        file_path = BOOK_VEC

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Book vectors file not found: {file_path}")

    df = pd.read_parquet(file_path)
    logger.info("Book vector data loaded: %s", df.shape)

    book_vectors_list = []
    book_ids = []

    for _, row in df.iterrows():
        book_ids.append(row["book_id"])
        book_vectors_list.append(_parse_vector(row["book_vector"]))

    book_vectors = np.array(book_vectors_list, dtype=np.float32)
    book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

    logger.info("Book vectors shape: %s, dim: %d", book_vectors.shape,
                book_vectors.shape[1] if book_vectors.ndim > 1 else 0)
    return book_vectors, book_id_to_index
