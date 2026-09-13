"""CLI: Evaluate a submission file."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.f1 import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a recommendation submission against validation data.")
    parser.add_argument("submission", type=str,
                        help="Path to submission CSV file.")
    parser.add_argument("--validation", type=str, default=None,
                        help="Path to validation CSV (default: DataLoad/inter_reevaluation.csv).")
    parser.add_argument("--user-col", type=str, default="user_id",
                        help="User column name in submission.")
    parser.add_argument("--book-col", type=str, default="book_id",
                        help="Book column name in submission.")
    args = parser.parse_args()

    result = evaluate(
        args.submission,
        validation_path=args.validation,
        user_col=args.user_col,
        book_col=args.book_col,
    )

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Precision: {result['precision']:.6f}")
    print(f"  Recall:    {result['recall']:.6f}")
    print(f"  F1 Score:  {result['f1_score']:.8f}")
    print(f"  Valid:     {result['is_valid']}")
    print(f"  Message:   {result['message']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
