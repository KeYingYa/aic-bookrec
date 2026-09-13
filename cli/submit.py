"""CLI: build a legitimate submission.

    python cli/submit.py [--tau 60] [--output submissions/submission_v2.csv]

Fits the selected predictor on **all** available interactions and writes one
prediction per user. Unlike the shipped pipeline this never reads a validation
file: there is no validation file to read, because the model is fitted on
everything we have and the honest quality estimate comes from
``cli/validate.py``, which trains on strictly earlier data.

Expected quality: 1.1-1.9% top-1 (see ``FIXES.md``), versus the 0.0376 the
original pipeline reported. That number was not a measurement of this model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config.paths import SUBMISSION_DIR
from src.data.temporal import Interactions, load_interactions
from src.models.recency import RecencyProfile
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a submission for every user.")
    parser.add_argument("--tau", type=float, default=60.0,
                        help="recency decay constant in days (default 60)")
    parser.add_argument("--cf-weight", type=float, default=0.0,
                        help="weight of the item-item CF term (0 by default; the "
                             "measured effect is within noise)")
    parser.add_argument("--output", type=str, default=None,
                        help="output CSV (default submissions/submission_v2.csv)")
    args = parser.parse_args()
    setup_logging("submit")

    frame = load_interactions()
    data = Interactions.from_frame(frame)
    model = RecencyProfile(tau_days=args.tau, cf_weight=args.cf_weight)
    model.fit(frame, data)

    # Every user we are asked about: the ones in the interaction log.
    users = np.sort(frame["user_id"].unique())
    rows = data.user_row(users)

    predictions: list[int] = []
    batch = 100
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        top = model.recommend(chunk, k=1)[:, 0]
        predictions.extend(data.book_ids_of(top).tolist())

    output = Path(args.output) if args.output else SUBMISSION_DIR / "submission_v2.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"user_id": users, "book_id": predictions})
    submission.to_csv(output, index=False)

    seen_before = 0
    history = frame.groupby("user_id")["book_id"].agg(set)
    for user, book in zip(submission["user_id"], submission["book_id"]):
        if int(book) in history.get(int(user), set()):
            seen_before += 1

    print(f"\nwrote {len(submission):,} predictions to {output}")
    print(f"  predicted books that the user has borrowed before: "
          f"{seen_before:,}/{len(submission):,} ({100 * seen_before / len(submission):.1f}%)")
    print("  (that share is expected: 17% of next borrows are re-borrows)")
    print(f"  distinct books recommended: {submission['book_id'].nunique():,}")
    print("\nExpected honest top-1 accuracy: ~1.5-1.7% (see FIXES.md). "
          "Do not quote the old 0.0376 for this file.")


if __name__ == "__main__":
    main()
