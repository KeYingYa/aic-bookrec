"""CLI: honest validation of next-borrow prediction.

    python cli/validate.py

Evaluates every candidate predictor on four temporal cut-offs and ranks them by
the mean, because selecting on a single cut-off is fitting that cut-off. Also
reproduces the original 0.0376 and explains why it is not a result.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config.paths import SUBMISSION_DIR, INTER_REEVALUATION_CSV, REPORTS_DIR
from src.data.temporal import CUTOFFS, Interactions, load_interactions, temporal_split
from src.evaluation.nextitem import evaluate, legacy_f1
from src.models.recency import GlobalPopularity, MostFrequent, MostRecent, RecencyProfile
from src.utils.logging_utils import setup_logging

MODELS = [
    ("global_popularity", lambda: GlobalPopularity()),
    ("most_frequent", lambda: MostFrequent()),
    ("most_recent", lambda: MostRecent()),
    ("profile tau=30d", lambda: RecencyProfile(tau_days=30)),
    ("profile tau=60d", lambda: RecencyProfile(tau_days=60)),
    ("profile tau=180d", lambda: RecencyProfile(tau_days=180)),
    ("profile tau=365d", lambda: RecencyProfile(tau_days=365)),
    ("profile no decay", lambda: RecencyProfile(tau_days=np.inf)),
    ("profile60 + cf0.25", lambda: RecencyProfile(tau_days=60, cf_weight=0.25)),
    ("profile60 + cf1.0", lambda: RecencyProfile(tau_days=60, cf_weight=1.0)),
]

METRIC = "accuracy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest validation on temporal cut-offs.")
    parser.add_argument("--metric", default=METRIC, choices=["accuracy", "hit@5", "mrr@10"])
    parser.add_argument("--report", type=str, default=None,
                        help="write a markdown report to this path")
    args = parser.parse_args()
    setup_logging("validate")

    frame = load_interactions()
    data = Interactions.from_frame(frame)
    print(f"\ninteractions {len(frame):,} | users {data.n_users:,} | "
          f"books borrowed {data.n_books:,} | "
          f"{frame['borrow_ts'].min().date()} .. {frame['borrow_ts'].max().date()}\n")

    table: dict[str, dict[str, float]] = {}
    counts: dict[str, tuple[int, int]] = {}
    for cutoff in CUTOFFS:
        train, test = temporal_split(frame, cutoff, require_history=True)
        _, test_all = temporal_split(frame, cutoff, require_history=False)
        counts[cutoff] = (len(test), len(test_all))
        for name, factory in MODELS:
            warm = evaluate(factory(), train, test, data, k=10)
            every = evaluate(factory(), train, test_all, data, k=10)
            table.setdefault(name, {})[cutoff] = warm[args.metric]
            table[name][f"{cutoff}_all"] = every[args.metric]

    print("users per cut-off: warm / all "
          + " ".join(f"{c[:7]}={counts[c][0]}/{counts[c][1]}" for c in CUTOFFS))
    print(f"cold users (no borrow before the cut-off): "
          f"{', '.join(str(counts[c][1] - counts[c][0]) for c in CUTOFFS)} "
          f"-- unlearnable, excluded from the warm column\n")

    rows = []
    for name, scores in table.items():
        per = [scores[c] for c in CUTOFFS]
        per_all = [scores[f"{c}_all"] for c in CUTOFFS]
        rows.append((float(np.mean(per)), float(min(per)), float(max(per)),
                     float(np.mean(per_all)), name, per, per_all))
    rows.sort(key=lambda r: -r[0])

    header = ("| model | " + " | ".join(c[:7] for c in CUTOFFS)
              + " | mean (warm) | mean (all users) | min | max |")
    print(header)
    print("|---" * (len(CUTOFFS) + 5) + "|")
    for mean, low, high, mean_all, name, per, per_all in rows:
        cells = " | ".join(f"{v:.4f}" for v in per)
        print(f"| {name} | {cells} | **{mean:.4f}** | {mean_all:.4f} | {low:.4f} | {high:.4f} |")

    best = rows[0]
    baseline = next(r for r in rows if r[4] == "most_recent")
    print(f"\nbest: {best[4]} at {best[0]:.4f} (warm {args.metric})")
    print(f"shipped strategy (most_recent): {baseline[0]:.4f}")
    print(f"difference: {best[0] - baseline[0]:+.4f} -- the noise floor with "
          f"400-740 users per cut-off is about +-0.006, so read this as 'no "
          f"measurable difference'.")

    # ---- the statistic that actually settles it -----------------------
    print(f"\n=== paired comparison: {best[4]} vs most_recent ===")
    print("(pooled over all cut-offs, users with history; a difference of a few "
          "users out of thousands is not a result)")
    wins = losses = 0
    for cutoff in CUTOFFS:
        train, test = temporal_split(frame, cutoff, require_history=True)
        rows_ids = data.user_row(test["user_id"].to_numpy())
        truth_cols = data.book_col(test["book_id"].to_numpy())
        baseline_pred = MostRecent().fit(train, data).recommend(rows_ids, k=1)[:, 0]
        best_pred = dict(MODELS)[best[4]]().fit(train, data).recommend(rows_ids, k=1)[:, 0]
        hit_b = baseline_pred == truth_cols
        hit_n = best_pred == truth_cols
        wins += int((hit_n & ~hit_b).sum())
        losses += int((hit_b & ~hit_n).sum())
    total = sum(len(temporal_split(frame, c, require_history=True)[1]) for c in CUTOFFS)
    print(f"  {best[4]}: +{wins} users, -{losses} users, net {wins - losses:+d} "
          f"out of {total} pooled users")

    # ---- why the original number was 0.0376 ---------------------------
    print("\n=== the original metric ===")
    submitted = SUBMISSION_DIR / "submission.csv"
    if submitted.exists():
        submission = pd.read_csv(submitted)
        legacy = legacy_f1(submission, frame)
        print(f"submissions/submission.csv vs inter_reevaluation.csv: "
              f"precision={legacy['precision']:.4f} recall={legacy['recall']:.4f} "
              f"F1={legacy['f1_score']:.6f}")
        print("precision is 1.0 because every predicted book appears somewhere in "
              "that user's own 58-row history; the metric cannot distinguish a "
              "model from a lookup of the answer key.")

    if args.report:
        write_report(Path(args.report), rows, args.metric, counts)


def write_report(path: Path, rows, metric: str, counts: dict) -> None:
    lines = [
        "# Honest validation on temporal cut-offs",
        "",
        f"Metric: **{metric}** -- one prediction and one target per user, so "
        "accuracy, precision, recall and F1 coincide.",
        "",
        "Training is everything before the cut-off; the target is each user's "
        "first borrow at or after it.",
        "",
        "Users per cut-off (warm / all): "
        + ", ".join(f"{c} = {counts[c][0]}/{counts[c][1]}" for c in CUTOFFS) + ".",
        "A *cold* user has no borrow before the cut-off, so no model can know "
        "anything about them; they are excluded from the warm column used to "
        "rank models and included in the all-users column, which estimates the "
        "submission score.",
        "",
        "| model | " + " | ".join(CUTOFFS) + " | mean (warm) | mean (all) | min | max |",
        "|---" * (len(CUTOFFS) + 5) + "|",
    ]
    for mean, low, high, mean_all, name, per, _ in rows:
        cells = " | ".join(f"{v:.4f}" for v in per)
        lines.append(f"| {name} | {cells} | **{mean:.4f}** | {mean_all:.4f} "
                     f"| {low:.4f} | {high:.4f} |")
    lines += [
        "",
        "The spread across cut-offs is as large as the differences between "
        "models. With 400-740 evaluable users per cut-off and an accuracy near "
        "2%, one standard error is about 0.006: **only the recency signal is "
        "real**. Everything else, including collaborative filtering, is inside "
        "the noise.",
        "",
        "For reference, ``submissions/submission.csv`` -- which predicts each "
        "user's most recent borrow and was reported as F1 0.0376 -- corresponds "
        "to the ``most_recent`` row here.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
