"""CLI: Build interaction matrix and ID mappings."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.interaction_matrix import build_interaction_matrix, build_id_mappings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build user-item interaction matrix and ID mappings.")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Cap on number of users (None = all users).")
    parser.add_argument("--output-matrix", type=str, default=None,
                        help="Output path for user_inter_matrix.parquet.")
    parser.add_argument("--output-mappings", type=str, default=None,
                        help="Output path for id_mappings.json.")
    args = parser.parse_args()

    print("Step 1/2: Building interaction matrix...")
    matrix = build_interaction_matrix(output_path=args.output_matrix)
    print(f"  Matrix shape: {matrix.shape}")

    print("Step 2/2: Building ID mappings...")
    mappings = build_id_mappings(matrix, output_path=args.output_mappings,
                                  max_users=args.max_users)
    print(f"  Users: {len(mappings['index_to_user'])}, "
          f"Books: {len(mappings['index_to_item'])}")
    print("Build complete.")


if __name__ == "__main__":
    main()
