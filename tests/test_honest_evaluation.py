"""Tests for the honest evaluation path.

These run on a small synthetic borrow log -- no raw data, no network, no
SentenceTransformer -- so they verify the *logic* that the fixes depend on:

* the temporal split trains strictly before the cut-off;
* it is not the old fake "hold the largest book ids" split;
* re-borrowed books stay in the candidate set (17% of next borrows are repeats);
* the shipped F1 metric is degenerate, which is why it was replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.temporal import Interactions, temporal_split
from src.evaluation.nextitem import legacy_f1, ranking_report, top1_accuracy
from src.models.recency import GlobalPopularity, MostFrequent, MostRecent, RecencyProfile


def synthetic_frame(n_users: int = 60, per_user: int = 10) -> pd.DataFrame:
    """Users borrow from a small pool, roughly monthly, with some repeats."""
    rng = np.random.default_rng(0)
    rows = []
    inter_id = 0
    for user in range(n_users):
        pool = rng.choice(40, size=6, replace=False)
        when = pd.Timestamp("2023-01-01")
        for step in range(per_user):
            book = int(pool[step % len(pool)])
            when = when + pd.Timedelta(days=int(rng.integers(20, 45)))
            rows.append((inter_id, user, book, when))
            inter_id += 1
    return pd.DataFrame(rows, columns=["inter_id", "user_id", "book_id", "borrow_ts"])


@pytest.fixture
def data() -> Interactions:
    return Interactions.from_frame(synthetic_frame())


def test_temporal_split_never_trains_on_the_future():
    frame = synthetic_frame()
    train, test = temporal_split(frame, "2023-09-01")
    assert len(test) > 0
    assert train["borrow_ts"].max() < pd.Timestamp("2023-09-01")
    assert test["borrow_ts"].min() >= pd.Timestamp("2023-09-01")


def test_temporal_split_takes_the_first_borrow_after_the_cutoff():
    frame = synthetic_frame()
    train, test = temporal_split(frame, "2023-09-01")
    after = frame[frame["borrow_ts"] >= pd.Timestamp("2023-09-01")]
    expected = after.sort_values("borrow_ts").groupby("user_id").head(1)
    merged = test.merge(expected, on=["user_id", "book_id"])
    assert len(merged) == len(test)


def test_temporal_split_is_one_row_per_user():
    train, test = temporal_split(synthetic_frame(), "2023-09-01")
    assert test["user_id"].is_unique


def test_temporal_split_excludes_users_without_history():
    frame = synthetic_frame()
    _, warm = temporal_split(frame, "2023-09-01", require_history=True)
    _, every = temporal_split(frame, "2023-09-01", require_history=False)
    assert len(warm) <= len(every)
    train = frame[frame["borrow_ts"] < pd.Timestamp("2023-09-01")]
    assert set(warm["user_id"]) <= set(train["user_id"])


def test_seen_books_remain_candidates():
    """Unlike a leave-one-out protocol, re-borrows must stay predictable."""
    frame = synthetic_frame()
    train, test = temporal_split(frame, "2023-09-01")
    history = train.groupby("user_id")["book_id"].agg(set)
    repeats = sum(int(b) in history.get(int(u), set())
                  for u, b in zip(test["user_id"], test["book_id"]))
    # The synthetic log is deliberately repetitive, so repeats must be common.
    assert repeats > 0

    data = Interactions.from_frame(frame)
    rows = data.user_row(test["user_id"].to_numpy())
    prediction = MostRecent().fit(train, data).recommend(rows, k=1)[:, 0]
    assert prediction.shape == (len(test),)


def test_recency_weighting_prefers_recent_items():
    """Mechanism, not a performance claim: tau decides what dominates.

    A performance claim belongs on the real data (where decay measurably helps:
    0.0166 with tau=60d against 0.0069 with no decay). Here the synthetic case is
    contrived so the two regimes are unambiguous: one book borrowed three times
    long ago, one book borrowed once yesterday.
    """
    when = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({
        "inter_id": [0, 1, 2, 3],
        "user_id": [0, 0, 0, 0],
        "book_id": [1, 1, 1, 2],
        "borrow_ts": [when, when + pd.Timedelta(days=1),
                      when + pd.Timedelta(days=2),
                      when + pd.Timedelta(days=300)],
    })
    data = Interactions.from_frame(frame)
    rows = np.array([0])

    recent = RecencyProfile(tau_days=10).fit(frame, data).recommend(rows, k=1)[0, 0]
    frequent = RecencyProfile(tau_days=np.inf).fit(frame, data).recommend(rows, k=1)[0, 0]
    assert data.book_ids_of([recent])[0] == 2      # the newest borrow wins
    assert data.book_ids_of([frequent])[0] == 1    # the most frequent book wins


def test_predictors_return_valid_columns():
    frame = synthetic_frame()
    data = Interactions.from_frame(frame)
    rows = data.user_row(np.array([0, 1, 2]))
    for model in (RecencyProfile(tau_days=60), MostRecent(), MostFrequent(),
                  GlobalPopularity()):
        prediction = model.fit(frame, data).recommend(rows, k=3)
        assert prediction.shape == (3, 3)
        assert prediction.min() >= 0 and prediction.max() < data.n_books


def test_ranking_report_counts_the_first_hit():
    # k must not exceed the number of recommended columns.
    recommended = np.array([[5, 1, 4], [2, 3, 9], [9, 8, 7]])
    truth = np.array([1, 2, 7])
    report = ranking_report(recommended, truth)
    assert report["accuracy"] == pytest.approx(1 / 3)      # only user 1 at rank 1
    assert report["hit@3"] == pytest.approx(1.0)           # all three found
    assert report["coverage"] == pytest.approx(1.0)
    assert report["mrr@10"] == pytest.approx((1 / 2 + 1 + 1 / 3) / 3)


def test_legacy_f1_is_degenerate():
    """Precision is 1.0 for any prediction drawn from the user's own history."""
    frame = synthetic_frame()
    prediction = frame.groupby("user_id").head(1)[["user_id", "book_id"]]
    report = legacy_f1(prediction, frame)
    assert report["precision"] == pytest.approx(1.0)
    # ...and it says nothing about whether the prediction is correct.
    assert report["f1_score"] > 0
    wrong = frame.groupby("user_id").tail(1)[["user_id", "book_id"]]
    assert legacy_f1(wrong, frame)["precision"] == pytest.approx(1.0)


def test_top1_accuracy_handles_empty_input():
    assert np.isnan(top1_accuracy(np.array([]), np.array([])))
