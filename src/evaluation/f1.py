"""F1 evaluation for recommendation submissions.

Refactored from F1.py. No more hardcoded absolute paths.
"""

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from src.config.paths import INTER_REEVALUATION_CSV


def _f1(sub: pd.DataFrame, val: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for aligned user-item DataFrames.

    Both DataFrames must have columns ['user_id', 'book_id'].

    Returns:
        Tuple of (precision, recall, f1_score).
    """
    if sub.empty or val.empty:
        return 0.0, 0.0, 0.0

    mrg = pd.merge(sub, val, on="user_id", suffixes=("_p", "_t"))
    tp = (mrg["book_id_p"] == mrg["book_id_t"]).sum()
    fp = len(sub) - tp
    fn = len(val) - tp

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(inp: Union[str, pd.DataFrame],
             *,
             validation_path: Path | str | None = None,
             user_col: str = "user_id",
             book_col: str = "book_id") -> dict:
    """Evaluate a submission file or DataFrame against validation data.

    Args:
        inp: CSV file path or DataFrame with user_id and book_id columns.
        validation_path: Path to validation CSV. Defaults to DataLoad version.
        user_col: Name of the user column in inp.
        book_col: Name of the book column in inp.

    Returns:
        Dict with keys: precision, recall, f1_score, is_valid, message.
    """
    if validation_path is None:
        validation_path = INTER_REEVALUATION_CSV

    # Load submission
    try:
        sub = pd.read_csv(inp) if isinstance(inp, str) else inp.copy()
        sub = sub[[user_col, book_col]].rename(
            columns={user_col: "user_id", book_col: "book_id"})
    except Exception as e:
        return {"precision": 0, "recall": 0, "f1_score": 0,
                "is_valid": False, "message": f"Failed to read submission: {e}"}

    # Load validation (try common Chinese encodings)
    val = None
    for enc in ("gbk", "gb2312", "utf-8", "ansi"):
        try:
            val = (pd.read_csv(validation_path, encoding=enc)
                   .rename(columns={user_col: "user_id", book_col: "book_id"})
                   [["user_id", "book_id"]]
                   .drop_duplicates())
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if val is None:
        return {"precision": 0, "recall": 0, "f1_score": 0,
                "is_valid": False, "message": "Failed to read validation: encoding error"}

    p, r, f = _f1(sub, val)
    return {"precision": p, "recall": r, "f1_score": f,
            "is_valid": f > 0.0055, "message": "Evaluation complete"}


def evaluate_list(pairs: List[Tuple[int, int]],
                  validation_path: Path | str | None = None) -> dict:
    """Evaluate a list of (user_id, book_id) pairs.

    Args:
        pairs: List of (user_id, book_id) tuples.
        validation_path: Path to validation CSV.

    Returns:
        Evaluation dict.
    """
    if not pairs:
        return {"precision": 0, "recall": 0, "f1_score": 0,
                "is_valid": False, "message": "Submission is empty"}

    sub = pd.DataFrame(pairs, columns=["user_id", "book_id"])
    users = sub["user_id"].unique().tolist()

    if validation_path is None:
        validation_path = INTER_REEVALUATION_CSV

    val = None
    for enc in ("gbk", "gb2312", "utf-8", "ansi"):
        try:
            val = (pd.read_csv(validation_path, encoding=enc)
                   [["user_id", "book_id"]]
                   .drop_duplicates()
                   .query("user_id in @users"))
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if val is None:
        return {"precision": 0, "recall": 0, "f1_score": 0,
                "is_valid": False, "message": "Failed to read validation: encoding error"}

    p, r, f = _f1(sub, val)
    return {"precision": p, "recall": r, "f1_score": f,
            "is_valid": f > 0.0055,
            "message": f"{len(pairs)} pairs evaluated"}


def evaluate_single(u: int, b: int,
                    validation_path: Path | str | None = None) -> dict:
    """Evaluate a single (user_id, book_id) pair."""
    return evaluate_list([(u, b)], validation_path=validation_path)


def evaluate_triple(u1: int, b1: int, u2: int, b2: int, u3: int, b3: int,
                    validation_path: Path | str | None = None) -> dict:
    """Evaluate three (user_id, book_id) pairs."""
    return evaluate_list([(u1, b1), (u2, b2), (u3, b3)],
                         validation_path=validation_path)
