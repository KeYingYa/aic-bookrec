"""CLI: Generate recommendations from trained models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_utils import setup_logging
from src.evaluation.inference import (
    generate_mf_recommendations,
    generate_cbmf_recommendations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate book recommendations from trained models.")
    parser.add_argument("--model", choices=["mf", "cbmf"], required=True,
                        help="Which model to use for inference.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (.pth file).")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Number of recommendations per user.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path.")
    parser.add_argument("--mappings", type=str, default=None,
                        help="Path to id_mappings.json.")
    args = parser.parse_args()

    setup_logging(f"inference_{args.model}")

    if args.model == "mf":
        output = generate_mf_recommendations(
            model_path=args.model_path or "models/fin_model.pth",
            mapping_file=args.mappings,
            top_k=args.top_k,
            output_file=args.output,
        )
    else:
        output = generate_cbmf_recommendations(
            model_path=args.model_path or "models/cb_mf_model.pth",
            mapping_file=args.mappings,
            n_recommend=args.top_k,
            output_file=args.output,
        )
        print(f"Recommendations generated for {len(output)} users.")

    print(f"Inference complete. Output: {output}")


if __name__ == "__main__":
    main()
