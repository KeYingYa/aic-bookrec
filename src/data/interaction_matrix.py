"""Interaction matrix construction and ID mappings.

Refactored from Data.py.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import (
    INTER2USER, INTER2BOOK, INTER2RATING, ITEM_CSV, USER_INTER_MATRIX,
    ID_MAPPINGS, DATA_DIR,
)


def build_interaction_matrix(
    inter2user_path: Path | str | None = None,
    inter2book_path: Path | str | None = None,
    inter2rating_path: Path | str | None = None,
    item_csv_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build the user-item rating matrix from interaction JSON files.

    Creates a wide-format DataFrame (users x books) with ratings filled
    where interactions exist, NaN elsewhere.

    Args:
        inter2user_path: Path to inter2user.json.
        inter2book_path: Path to inter2book.json.
        inter2rating_path: Path to inter2rating.json.
        item_csv_path: Path to item.csv (for full book list).
        output_path: Where to save user_inter_matrix.parquet.

    Returns:
        The user-item interaction matrix DataFrame.
    """
    if inter2user_path is None:
        inter2user_path = INTER2USER
    if inter2book_path is None:
        inter2book_path = INTER2BOOK
    if inter2rating_path is None:
        inter2rating_path = INTER2RATING
    if item_csv_path is None:
        item_csv_path = ITEM_CSV
    if output_path is None:
        output_path = USER_INTER_MATRIX

    inter2user_path = Path(inter2user_path)
    inter2book_path = Path(inter2book_path)
    inter2rating_path = Path(inter2rating_path)
    item_csv_path = Path(item_csv_path)
    output_path = Path(output_path)

    # Load interaction mappings
    inter2user = json.loads(inter2user_path.read_text())
    inter2book = json.loads(inter2book_path.read_text())
    inter2rating = json.loads(inter2rating_path.read_text())

    inter_df = pd.DataFrame({
        "user_id": pd.Series(inter2user, dtype="int"),
        "book_id": pd.Series(inter2book, dtype="int"),
        "rating": pd.Series(inter2rating, dtype="float"),
    })

    # Full book set from item.csv (try common Chinese encodings)
    item_df = None
    for enc in ("gbk", "gb2312", "utf-8", "ansi"):
        try:
            item_df = pd.read_csv(item_csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if item_df is None:
        raise ValueError(f"Failed to read {item_csv_path}: encoding error")
    all_books = item_df["book_id"].sort_values().unique()

    # Build skeleton: all users x all books, init to NaN
    users = inter_df["user_id"].unique()
    skeleton = pd.DataFrame(
        index=pd.Index(users, name="user_id"),
        columns=pd.Index(all_books, name="book_id"),
        dtype="float",
    )

    # Fill ratings (max per user-book pair)
    g = inter_df.groupby(["user_id", "book_id"])["rating"].max()
    skeleton.update(g.unstack(level="book_id"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton.to_parquet(output_path)
    print(f"Interaction matrix saved to {output_path}. "
          f"Shape: {skeleton.shape}")

    return skeleton


def build_id_mappings(
    matrix: pd.DataFrame,
    output_path: Path | str | None = None,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Build user↔index and book↔index mappings.

    Args:
        matrix: The user-item interaction matrix (users=index, books=columns).
        output_path: Where to save id_mappings.json.
        max_users: Optional cap on number of users to include (None = all).

    Returns:
        Dict with keys: index_to_user, user_to_index, index_to_item, item_to_index.
    """
    if output_path is None:
        output_path = ID_MAPPINGS

    user_ids = list(matrix.index)
    book_ids = list(matrix.columns)

    if max_users is not None:
        user_ids = user_ids[:max_users]

    index_to_user = {str(i): int(uid) for i, uid in enumerate(user_ids)}
    user_to_index = {str(uid): i for i, uid in enumerate(user_ids)}
    index_to_item = {str(i): int(bid) for i, bid in enumerate(book_ids)}
    item_to_index = {str(bid): i for i, bid in enumerate(book_ids)}

    mappings = {
        "index_to_user": index_to_user,
        "user_to_index": user_to_index,
        "index_to_item": index_to_item,
        "item_to_index": item_to_index,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    print(f"ID mappings saved to {output_path}. "
          f"Users: {len(user_ids)}, Books: {len(book_ids)}")
    return mappings
