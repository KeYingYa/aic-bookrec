"""Honest evaluation for next-item prediction.

The project shipped with ``src/evaluation/f1.py`` computing precision/recall by
merging on ``user_id`` alone. Against a 58,327-row "validation" file and a
1000-row submission, every submission row matches ~58 validation rows, so:

* precision is **1.0 for any submission whose book the user borrowed at least
  once** -- it carries no information at all;
* F1 reduces to ``2 * tp / (tp + fn)`` where ``fn = 58327 - tp``, i.e. roughly
  ``2 / 59.3`` for any submission that gets its 1000 row-pairs right.

That is why the shipped submission "scored" 0.0376 while every real model scored
below the 0.0055 validity threshold: the metric was measuring whether the
predicted book appears anywhere in the user's 58-row history, and the shipper
had copied the answer out of that same history.

``top1_accuracy`` below is the metric that actually reflects the task: one
prediction and one target per user, so accuracy, precision, recall and F1 all
coincide. ``legacy_f1`` is retained only so the old number can be reproduced and
explained.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.temporal import Interactions

logger = logging.getLogger(__name__)

__all__ = ["top1_accuracy", "ranking_report", "evaluate", "legacy_f1"]


def top1_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of users whose single predicted book is the target book."""
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    if truth.size == 0:
        return float("nan")
    return float((predicted == truth).mean())


def ranking_report(recommended: np.ndarray, truth: np.ndarray) -> dict:
    """Accuracy plus hit@k / MRR@k for a ``(n_users, k)`` recommendation array."""
    recommended = np.asarray(recommended)
    truth = np.asarray(truth).reshape(-1, 1)
    hits = recommended == truth
    found = hits.any(axis=1)

    report = {
        "n": float(len(truth)),
        "accuracy": float(hits[:, 0].mean()),
        "coverage": float(found.mean()),
    }
    positions = np.where(found, hits.argmax(axis=1) + 1, 0)
    for k in (1, 3, 5, 10):
        if k <= recommended.shape[1]:
            report[f"hit@{k}"] = float(((positions > 0) & (positions <= k)).mean())
    report["mrr@10"] = float(
        np.where(found & (positions <= 10), 1.0 / np.maximum(positions, 1), 0.0).mean())
    return report


def evaluate(model, train: pd.DataFrame, test: pd.DataFrame,
             data: Interactions, k: int = 10) -> dict:
    """Fit ``model`` on ``train`` and score it against ``test``.

    Only users present in ``test`` are evaluated; books already borrowed by the
    user stay in the candidate set.
    """
    model.fit(train, data)
    rows = data.user_row(test["user_id"].to_numpy())
    recommended = model.recommend(rows, k=k)
    truth = data.book_col(test["book_id"].to_numpy())
    report = ranking_report(recommended, truth)
    report["model"] = model.name
    return report


def legacy_f1(submission: pd.DataFrame, interactions: pd.DataFrame) -> dict:
    """Reproduce the original, degenerate precision/recall/F1.

    Kept so the reported 0.0376 can be regenerated and explained. Do not use it
    to compare models: precision is 1.0 for any submission drawn from the
    interaction history.
    """
    sub = submission[["user_id", "book_id"]].copy()
    val = interactions[["user_id", "book_id"]].drop_duplicates()

    merged = sub.merge(val, on="user_id", suffixes=("_p", "_t"))
    tp = int((merged["book_id_p"] == merged["book_id_t"]).sum())
    fp = len(sub) - tp
    fn = len(val) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1,
            "true_positives": float(tp), "official_threshold": 0.0055}
