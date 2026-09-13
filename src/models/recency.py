"""Predictors for "what will this user borrow next?".

Every predictor exposes the same tiny interface::

    model.fit(train_frame, data)     # data carries the id <-> index maps
    model.scores(user_rows)          # (len(user_rows), n_books), higher = better

The candidate set is the **whole catalogue including books the user has already
borrowed**. That is deliberate and is where this task differs from the
leave-one-out protocol used for Book-Crossing: in a library, 17% of a user's
next borrows are books they have borrowed before, so masking seen items would
throw away one correct answer in six.

What the honest baselines on the temporal split say (4 cut-offs, see
``FIXES.md``): global popularity 0.000-0.002, user's most frequent book
0.003-0.011, user's most recent book 0.011-0.019. Anything that does not clear
the 1.1-1.9% band set by "predict the most recent borrow" has added nothing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.temporal import Interactions

logger = logging.getLogger(__name__)

__all__ = ["RecencyProfile", "MostRecent", "MostFrequent", "GlobalPopularity"]


class _Base:
    """Shared bookkeeping for predictors."""

    name = "base"

    def __init__(self) -> None:
        self.n_users = 0
        self.n_books = 0
        self.data: Interactions | None = None

    def _remember(self, data: Interactions) -> None:
        self.data = data
        self.n_users = data.n_users
        self.n_books = data.n_books

    def fit(self, train: pd.DataFrame, data: Interactions) -> "_Base":
        raise NotImplementedError

    def scores(self, user_rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_rows: np.ndarray, k: int = 1) -> np.ndarray:
        """Top-``k`` catalogue columns per row, best first."""
        scores = self.scores(user_rows)
        k = min(k, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        rows = np.arange(scores.shape[0])[:, None]
        order = np.argsort(-scores[rows, top], axis=1, kind="stable")
        return top[rows, order]

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _sparse(train: pd.DataFrame, data: Interactions, weights: np.ndarray,
                shape: tuple[int, int] | None = None) -> sp.csr_matrix:
        rows = data.user_row(train["user_id"].to_numpy())
        cols = data.book_col(train["book_id"].to_numpy())
        shape = shape or (data.n_users, data.n_books)
        matrix = sp.csr_matrix((weights.astype(np.float32), (rows, cols)),
                               shape=shape, dtype=np.float32)
        matrix.sum_duplicates()
        return matrix


class RecencyProfile(_Base):
    """Exponentially recency-weighted profile of everything the user borrowed.

    ``score(u, b) = sum over the user's borrows of b of exp(-age_days / tau)``

    The recency constant is the whole model: with no decay (``tau=inf``) the
    accuracy collapses to 0.007, with a two-month half-life it reaches 0.017.
    A user's next borrow is mostly driven by what they have been reading lately.

    Args:
        tau_days: decay constant in days. ``inf`` disables decay.
        cf_weight: weight of an item-item cosine collaborative-filtering term
            (0 disables it). Included because it is cheap and occasionally
            helps, but it is *within noise*: it wins 11 users and loses 9 out of
            2339 in a paired comparison, so it is off by default.
        cf_shrinkage: shrinkage added to the cosine denominator.
    """

    name = "recency_profile"

    def __init__(self, tau_days: float = 60.0, cf_weight: float = 0.0,
                 cf_shrinkage: float = 10.0) -> None:
        super().__init__()
        self.tau_days = tau_days
        self.cf_weight = cf_weight
        self.cf_shrinkage = cf_shrinkage
        self._profile: sp.csr_matrix | None = None
        self._cf: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, data: Interactions) -> "RecencyProfile":
        self._remember(data)

        age_days = (train["borrow_ts"].max() - train["borrow_ts"]).dt.total_seconds() / 86400.0
        if np.isfinite(self.tau_days):
            weights = np.exp(-age_days.to_numpy() / float(self.tau_days))
        else:
            weights = np.ones(len(train), dtype=np.float64)

        binary = self._sparse(train, data, np.ones(len(train)))
        self._profile = self._sparse(train, data, weights)

        if self.cf_weight:
            counts = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)
            cooccurrence = (binary.T @ binary).tocsr().astype(np.float64)
            inverse = sp.diags(1.0 / np.maximum(np.sqrt(counts) + self.cf_shrinkage, 1e-9))
            similarity = (inverse @ cooccurrence @ inverse).tocsr()
            # Drop the diagonal without mutating the sparsity structure in place
            # (setdiag on CSR raises a SparseEfficiencyWarning).
            similarity = (similarity - sp.diags(similarity.diagonal())).tocsr()
            similarity.eliminate_zeros()
            cf = (binary @ similarity).toarray().astype(np.float32)
            peak = float(cf.max())
            self._cf = cf / peak if peak > 0 else cf

        logger.info("[%s] fitted: tau=%s cf_weight=%s", self.name, self.tau_days,
                    self.cf_weight)
        return self

    def scores(self, user_rows: np.ndarray) -> np.ndarray:
        user_rows = np.asarray(user_rows, dtype=np.int64)
        profile = self._profile[user_rows].toarray()
        peak = float(profile.max()) if profile.size else 0.0
        if peak > 0:
            profile = profile / peak
        if self._cf is not None:
            profile = profile + self.cf_weight * self._cf[user_rows]
        return profile


class MostRecent(_Base):
    """Predict the user's most recent borrow.

    This is, for 997 of 1000 users, what the shipped ``submission.csv`` did --
    it is included as a named baseline because its honest accuracy (1.1-1.9%)
    turns out to be within two users of the best model found, and any claim of
    improvement has to be measured against it.
    """

    name = "most_recent"

    def fit(self, train: pd.DataFrame, data: Interactions) -> "MostRecent":
        self._remember(data)
        newest = train.sort_values(["user_id", "borrow_ts", "inter_id"]).groupby("user_id").tail(1)
        self._matrix = self._sparse(newest, data, np.ones(len(newest)))
        return self

    def scores(self, user_rows: np.ndarray) -> np.ndarray:
        return self._matrix[np.asarray(user_rows, dtype=np.int64)].toarray()


class MostFrequent(_Base):
    """Predict the book the user borrowed most often."""

    name = "most_frequent"

    def fit(self, train: pd.DataFrame, data: Interactions) -> "MostFrequent":
        self._remember(data)
        counts = (train.groupby(["user_id", "book_id"]).size()
                  .reset_index(name="n").sort_values("n"))
        best = counts.groupby("user_id").tail(1)
        self._matrix = self._sparse(best, data, best["n"].to_numpy(dtype=np.float64))
        return self

    def scores(self, user_rows: np.ndarray) -> np.ndarray:
        return self._matrix[np.asarray(user_rows, dtype=np.int64)].toarray()


class GlobalPopularity(_Base):
    """Predict the globally most borrowed book."""

    name = "global_popularity"

    def fit(self, train: pd.DataFrame, data: Interactions) -> "GlobalPopularity":
        self._remember(data)
        counts = np.bincount(data.book_col(train["book_id"].to_numpy()),
                             minlength=data.n_books).astype(np.float32)
        self._counts = counts
        return self

    def scores(self, user_rows: np.ndarray) -> np.ndarray:
        return np.tile(self._counts, (len(np.asarray(user_rows)), 1))
