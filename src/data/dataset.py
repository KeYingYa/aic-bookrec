"""Dataset loading and train/validation splitting.

Unified implementation (was duplicated in MF.py and CBMF.py).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.base import BaseConfig
from src.config.paths import ID_MAPPINGS, INTER_REEVALUATION_CSV
from src.data.preprocessing import compute_ratings, read_interactions_raw

logger = logging.getLogger(__name__)


def split_one_user(g: pd.DataFrame, n_keep: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-last-N-out split for a single user's interactions.

    Keeps the last `n_keep` interactions as validation, the rest as training.

    Args:
        g: Sorted DataFrame of one user's interactions.
        n_keep: Number of last interactions to hold out for validation.

    Returns:
        Tuple of (train_df, valid_df).
    """
    if len(g) < n_keep + 1:
        return g, g.iloc[:0]
    valid = g.tail(n_keep)
    train = g.drop(valid.index)
    return train, valid


def timed_records(reference_date: str = "2025-11-02") -> pd.DataFrame:
    """Long-format interactions with **real borrow times**.

    Returns columns ``user, item, rating, borrow_ts`` using contiguous 0-based
    indices that match the wide matrix and ``id_mappings.json``, so the result
    can be fed straight to the trainers.
    """
    raw = compute_ratings(read_interactions_raw(INTER_REEVALUATION_CSV), reference_date)
    mappings = json.loads(Path(ID_MAPPINGS).read_text(encoding="utf-8"))
    user_to_index = {int(k): int(v) for k, v in mappings["user_to_index"].items()}
    item_to_index = {int(k): int(v) for k, v in mappings["item_to_index"].items()}

    frame = pd.DataFrame({
        "user": raw["user_id"].map(user_to_index),
        "item": raw["book_id"].map(item_to_index),
        "rating": raw["rating"].astype(float),
        "borrow_ts": pd.to_datetime(raw["\u501f\u9605\u65f6\u95f4"], errors="coerce"),
    })
    missing = int(frame["user"].isna().sum() + frame["item"].isna().sum())
    if missing:
        logger.warning("%d interactions reference ids outside id_mappings; dropped",
                       missing)
    frame = frame.dropna(subset=["user", "item", "borrow_ts"])
    frame["user"] = frame["user"].astype(int)
    frame["item"] = frame["item"].astype(int)
    return frame.sort_values(["user", "borrow_ts"], kind="stable").reset_index(drop=True)


def load_interaction_data(
    config: type[BaseConfig],
    file_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load interaction data and split into train/validation sets **by borrow time**.

    FIXED. The previous implementation built its ``time`` column like this::

        df_reset.columns = range(df_reset.shape[1])   # book ids discarded here
        "time": stacked.index.get_level_values(1)     # ...so this is a position

    which made ``sort_values(["user", "time"])`` sort by the book's *position* in
    the matrix. Book ids run in acquisition order, so "leave-last-%d-out" actually
    withheld each user's N largest book ids. The validation set was arbitrary and
    every validation loss the project reported was meaningless.

    The borrow timestamps were in ``DataLoad/inter_reevaluation.csv`` the whole
    time; they are now used, mapped onto the matrix index space through
    ``id_mappings.json``.

    Args:
        config: Configuration class (BaseConfig subclass).
        file_path: Path to the wide parquet matrix. Used only for the index space
            (its shape defines the number of users and items).

    Returns:
        Tuple of (train_data, val_data, df_reset) where:
        - train_data: shape (N, 3) ``[user_idx, item_idx, rating]``
        - val_data: shape (M, 3), the ``n_keep_val`` latest borrows per user
        - df_reset: the wide user x item matrix
    """
    if file_path is None:
        file_path = config.DATA["file_path"]

    file_path = Path(file_path)
    n_keep = config.TRAINING.get("n_keep_val", 2)

    logger.info("Loading wide matrix for the index space: %s", file_path)
    df = pd.read_parquet(file_path)
    df_reset = df.reset_index(drop=True)
    logger.info("Wide matrix: %s", df_reset.shape)

    records = timed_records()
    logger.info("Timed interactions: %d rows over %d users",
                len(records), records["user"].nunique())

    # Rank each user's borrows by real time: 0 is the most recent.
    position_from_end = records.groupby("user").cumcount(ascending=False)
    sizes = records.groupby("user")["user"].transform("size")
    # Users with too few borrows keep everything in training rather than
    # producing an empty training set.
    is_validation = (position_from_end < n_keep) & (sizes >= n_keep + 1)

    train_df = records.loc[~is_validation]
    val_df = records.loc[is_validation]
    logger.info("Split by borrow time -- train %d, val %d (leave-last-%d-out)",
                len(train_df), len(val_df), n_keep)
    if not val_df.empty:
        logger.info("Train borrows up to %s, validation targets from %s",
                    train_df["borrow_ts"].max().date(), val_df["borrow_ts"].min().date())

    train_data = train_df[["user", "item", "rating"]].to_numpy()
    val_data = val_df[["user", "item", "rating"]].to_numpy()
    return train_data, val_data, df_reset
