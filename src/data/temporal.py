"""Temporal splitting on the real borrow timestamps.

The original ``src/data/dataset.py`` built its "time" column from the *column
position* of the wide user x book matrix::

    df_reset.columns = range(df_reset.shape[1])      # book ids discarded
    "time": stacked.index.get_level_values(1)        # ...so this is a position

so the advertised "leave-last-N-out" split actually withheld the two largest
book ids per user. Book ids are assigned roughly by acquisition order, which
makes the resulting validation set arbitrary and the reported validation RMSE
meaningless.

This module uses the borrow timestamps that were in the data all along
(2020-01-02 .. 2024-12-30) and offers the two splits that matter here:

``temporal_split``
    Train on everything before a cut-off; the target is each user's **first**
    borrow at or after it. This mirrors the real task -- predict the next borrow
    -- and is the split every number in ``FIXES.md`` is measured on.

``leave_last_one_out``
    Each user's final borrow is the target. Useful as a second opinion, but it
    lets a user's other late borrows leak into training through the recency
    signal, so it is the more optimistic of the two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import INTER_REEVALUATION_CSV, ITEM_CSV

logger = logging.getLogger(__name__)

__all__ = [
    "Interactions",
    "load_interactions",
    "load_books",
    "temporal_split",
    "leave_last_one_out",
    "CUTOFFS",
]

#: Cut-offs used for model selection. Selecting on a single cut-off is fitting
#: that cut-off: item-item CF scores 0.016 / 0.005 / 0.002 on three different
#: ones, so every configuration is scored on all of them and ranked by the mean.
CUTOFFS: tuple[str, ...] = ("2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01")

_RAW_COLUMNS = ["inter_id", "user_id", "book_id", "borrow", "return",
                "renew_time", "renew_count"]


def load_interactions(path: Path | str | None = None) -> pd.DataFrame:
    """Load the borrow log with parsed timestamps.

    The file is GBK-encoded with Chinese column names. Columns returned:
    ``inter_id, user_id, book_id, borrow_ts, renew_count``.
    """
    path = Path(path) if path is not None else INTER_REEVALUATION_CSV
    frame = pd.read_csv(path, encoding="gbk")
    frame.columns = _RAW_COLUMNS

    frame["borrow_ts"] = pd.to_datetime(frame["borrow"], errors="coerce")
    n_bad = int(frame["borrow_ts"].isna().sum())
    if n_bad:
        logger.warning("dropping %d rows with unparseable borrow time", n_bad)
        frame = frame.dropna(subset=["borrow_ts"])

    frame["renew_count"] = pd.to_numeric(frame["renew_count"], errors="coerce").fillna(0)
    frame = frame[["inter_id", "user_id", "book_id", "borrow_ts", "renew_count"]]
    frame = frame.sort_values(["user_id", "borrow_ts", "inter_id"]).reset_index(drop=True)
    logger.info("loaded %d interactions, %d users, %d books, %s .. %s",
                len(frame), frame["user_id"].nunique(), frame["book_id"].nunique(),
                frame["borrow_ts"].min().date(), frame["borrow_ts"].max().date())
    return frame


def load_books(path: Path | str | None = None) -> pd.DataFrame:
    """Load book metadata (``book_id, title, author, publisher, cat1, cat2``)."""
    path = Path(path) if path is not None else ITEM_CSV
    frame = pd.read_csv(path, encoding="gbk")
    frame.columns = ["book_id", "title", "author", "publisher", "cat1", "cat2"]
    return frame


@dataclass
class Interactions:
    """A dataset with contiguous 0-based user and book indices."""

    frame: pd.DataFrame
    users: np.ndarray
    books: np.ndarray

    def __post_init__(self) -> None:
        self._user_index = {int(u): i for i, u in enumerate(self.users)}
        self._book_index = {int(b): i for i, b in enumerate(self.books)}

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_books(self) -> int:
        return len(self.books)

    def user_row(self, user_ids) -> np.ndarray:
        """Map original user ids to matrix rows."""
        return np.array([self._user_index[int(u)] for u in user_ids], dtype=np.int64)

    def book_col(self, book_ids) -> np.ndarray:
        """Map original book ids to matrix columns."""
        return np.array([self._book_index[int(b)] for b in book_ids], dtype=np.int64)

    def book_ids_of(self, columns) -> np.ndarray:
        """Map matrix columns back to original book ids."""
        return self.books[np.asarray(columns, dtype=np.int64)]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Interactions":
        users = np.sort(frame["user_id"].unique())
        books = np.sort(frame["book_id"].unique())
        return cls(frame=frame, users=users, books=books)


def temporal_split(frame: pd.DataFrame, cutoff: str,
                   require_history: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train before ``cutoff``; target is each user's first borrow at/after it.

    Args:
        frame: the interaction log.
        cutoff: ISO date string.
        require_history: when True (the default) users with no borrow before the
            cut-off are dropped from the test set. Such a user has no signal at
            all, so including them only dilutes the metric with a constant miss:
            at the 2023-01-01 cut-off, 226 of 736 users are cold and they move
            the measured accuracy from 0.0196 to 0.0136 without any model
            changing. Warm-user accuracy is the right quantity for *choosing*
            between models; the all-user accuracy is the right estimate of what
            a submission will score, since the real task gives every user a
            history. ``cli/validate.py`` reports both.

    Already-borrowed books stay in the candidate set: on these cut-offs about
    11% of next borrows are re-borrows (17% under leave-last-one-out), so masking
    them would discard roughly one correct answer in nine.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train = frame[frame["borrow_ts"] < cutoff_ts]
    after = frame[frame["borrow_ts"] >= cutoff_ts]
    test = after.groupby("user_id", sort=True).head(1)
    if require_history:
        test = test[test["user_id"].isin(train["user_id"].unique())]
    logger.info("cutoff %s -> train %d, test %d users", cutoff, len(train), len(test))
    return train.reset_index(drop=True), test.reset_index(drop=True)


def leave_last_one_out(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each user's final borrow is the target, the rest is training."""
    test = frame.groupby("user_id", sort=True).tail(1)
    train = frame.drop(test.index)
    train = train[train["user_id"].isin(test["user_id"].unique())]
    logger.info("leave-last-one-out -> train %d, test %d users", len(train), len(test))
    return train.reset_index(drop=True), test.reset_index(drop=True)
