"""Vector similarity utilities.

Refactored from similarity.py — no more hardcoded paths.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import BOOK_VEC


def load_vec_mapping(file_path: Path | str | None = None,
                     id_col: str = "book_id",
                     vec_col: str = "book_vector") -> dict[Any, np.ndarray]:
    """Load an id→vector mapping from a parquet file.

    Args:
        file_path: Path to the parquet file.
        id_col: Name of the ID column.
        vec_col: Name of the vector column.

    Returns:
        Dict mapping each id to its numpy vector.
    """
    if file_path is None:
        file_path = BOOK_VEC
    df = pd.read_parquet(file_path)
    return dict(zip(df[id_col], df[vec_col]))


def cosine_similarity_by_id(id1: Any, id2: Any,
                            id2vec: dict[Any, np.ndarray]) -> float:
    """Compute cosine similarity between two items by their IDs.

    Args:
        id1: First item ID.
        id2: Second item ID.
        id2vec: Dict mapping item ID → vector.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero.
    """
    v1 = id2vec[id1]
    v2 = id2vec[id2]
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))
