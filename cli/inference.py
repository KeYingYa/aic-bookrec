"""CLI: Generate recommendations from trained models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_utils import setup_logging
from src.config.paths import OUTPUT_DIR
from src.evaluation.inference import (
    generate_mf_recommendations,
    generate_cbmf_recommendations,
)

USEFUL_MODEL_HINT = """\
--model-path is required: there is no safe default.

The hard-coded default used to be 'models/fin_model.pth'. That checkpoint
records only 5 training epochs (train loss 8.15, val RMSE 3.10); the better
trained ones are mf_v2.pth / mf_250_new.pth (250 epochs) or mf_128.pth.

More importantly, on the honest temporal split (`python cli/validate.py`) the
latent-factor models score about 0.002 top-1, below the 0.0196 of simply
predicting each user's most recent borrow. Prefer `python cli/submit.py`, which
trains the recency model on all data and writes a legitimate submission.
See FIXES.md.
"""


def _require_checkpoint(explicit: str | None) -> str:
    """Resolve --model-path, refusing to guess."""
    if explicit:
        return explicit
    available = sorted(p.name for p in OUTPUT_DIR.glob("*.pth"))
    listing = "\n  ".join(available) if available else "(none)"
    raise SystemExit(f"{USEFUL_MODEL_HINT}\nCheckpoints in {OUTPUT_DIR}:\n  {listing}\n")


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
            model_path=_require_checkpoint(args.model_path),
            mapping_file=args.mappings,
            top_k=args.top_k,
            output_file=args.output,
        )
    else:
        output = generate_cbmf_recommendations(
            model_path=_require_checkpoint(args.model_path),
            mapping_file=args.mappings,
            n_recommend=args.top_k,
            output_file=args.output,
        )
        print(f"Recommendations generated for {len(output)} users.")

    print(f"Inference complete. Output: {output}")


if __name__ == "__main__":
    main()
