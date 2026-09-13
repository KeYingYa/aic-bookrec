"""CLI: Data preprocessing — compute ratings and generate book vectors."""

import argparse
import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import add_rating, tovec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess data: compute ratings and generate book vectors.")
    parser.add_argument("--skip-rating", action="store_true",
                        help="Skip rating computation.")
    parser.add_argument("--skip-tovec", action="store_true",
                        help="Skip book vector generation.")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Path to inter_reevaluation.csv.")
    parser.add_argument("--item-csv", type=str, default=None,
                        help="Path to item.csv.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to SentenceTransformer model directory.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for vector encoding.")
    args = parser.parse_args()

    if not args.skip_rating:
        print("Step 1/2: Computing ratings...")
        add_rating(input_csv=args.input_csv)
    else:
        print("Skipping rating computation.")

    if not args.skip_tovec:
        print("Step 2/2: Generating book vectors...")
        tovec(csv_path=args.item_csv, model_dir=args.model_dir,
              batch_size=args.batch_size)
    else:
        print("Skipping book vector generation.")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
