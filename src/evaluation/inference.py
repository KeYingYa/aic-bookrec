"""Recommendation inference — generate and save predictions.

Unified from MF.py generate_recommendations_for_all_users() and
CBMF.py batch_recommend().

Fixes applied:
- Removed os.system('python tem2.py') dead call
- Removed int(real_book_id)+1 off-by-one
- Uses id_mappings.json item_to_index/index_to_item
- Uses weights_only=True for checkpoint loading
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config.base import BaseConfig
from src.config.mf_config import MFConfig
from src.config.cbmf_config import CBMFConfig

# Legacy aliases for old checkpoint compatibility
# Old checkpoints serialized classes as __main__.MyConfig / __main__.ContentMFConfig
import sys as _sys
_sys.modules["__main__"].__dict__.setdefault("MyConfig", MFConfig)
_sys.modules["__main__"].__dict__.setdefault("ContentMFConfig", CBMFConfig)
from src.config.paths import ID_MAPPINGS, BOOK_VEC, SUBMISSION_DIR
from src.models.mf import MatrixFactorization
from src.models.cbmf import ContentBasedMatrixFactorization
from src.utils.book_vectors import load_book_vectors

logger = logging.getLogger(__name__)


def _load_checkpoint_safe(model_path, map_location="cpu") -> dict:
    """Load checkpoint handling both legacy and new formats."""
    model_path = str(model_path)
    try:
        return torch.load(model_path, map_location=map_location, weights_only=True)
    except Exception:
        logger.warning(
            "Legacy checkpoint (%s). Loading with weights_only=False. "
            "Re-save to enable safe loading.", model_path)
        return torch.load(model_path, map_location=map_location, weights_only=False)


def _load_mappings(mapping_file: Path | str | None = None) -> dict:
    """Load ID mappings from JSON.

    Returns a dict with: user_to_index, index_to_user, item_to_index, index_to_item.
    """
    if mapping_file is None:
        mapping_file = ID_MAPPINGS
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_real_user_ids(mappings: dict | None = None) -> list:
    """Extract all real user IDs from the mappings."""
    if mappings is None:
        mappings = _load_mappings()
    if "index_to_user" in mappings:
        return list(mappings["index_to_user"].values())
    if "user_to_index" in mappings:
        return list(mappings["user_to_index"].keys())
    raise KeyError("Mappings missing 'index_to_user' or 'user_to_index'")


def _find_indices(real_ids: list, mappings: dict) -> list:
    """Map real user IDs to internal indices."""
    inner = mappings["index_to_user"]
    return [int(k) for real_id in real_ids
            for k, v in inner.items() if v == real_id]


# ---- MF Inference ----

def generate_mf_recommendations(
    model_path: Path | str = "models_output/fin_model.pth",
    mapping_file: Path | str | None = None,
    top_k: int = 1,
    output_file: Path | str | None = None,
) -> str:
    """Generate top-K recommendations for all users using MF model.

    Args:
        model_path: Path to the trained MF model checkpoint.
        mapping_file: Path to id_mappings.json.
        top_k: Number of recommendations per user.
        output_file: Path for output CSV. Defaults to submissions/submission_mf.csv.

    Returns:
        Path to the generated submission CSV.
    """
    if mapping_file is None:
        mapping_file = ID_MAPPINGS
    if output_file is None:
        output_file = SUBMISSION_DIR / "submission_mf.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load checkpoint
    checkpoint = _load_checkpoint_safe(model_path, device)
    logger.info("Model checkpoint loaded from %s", model_path)

    # Load mappings
    mappings = _load_mappings(mapping_file)
    # (MF path)
    user_to_index = {int(k): int(v) for k, v in mappings.get("user_to_index", {}).items()}
    index_to_user = {int(k): int(v) for k, v in mappings.get("index_to_user", {}).items()}
    item_to_index = {int(k): int(v) for k, v in mappings.get("item_to_index", {}).items()}
    index_to_item = {int(k): int(v) for k, v in mappings.get("index_to_item", {}).items()}

    logger.info("User mappings: %d users, Item mappings: %d items",
                len(index_to_user), len(index_to_item))

    # Reconstruct model
    config = checkpoint.get("config", MFConfig)
    n_users = checkpoint["n_users"]
    n_items = checkpoint["n_items"]

    model = MatrixFactorization(n_users, n_items, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Generate recommendations
    total = 0
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("user_id,book_id\n")

        for user_idx in tqdm(range(n_users), desc="Generating recommendations"):
            real_user_id = index_to_user.get(user_idx)
            if real_user_id is None:
                continue

            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            all_scores = []

            # Batch-score all items
            batch_size = 8000
            for start in range(0, n_items, batch_size):
                end = min(start + batch_size, n_items)
                batch_items = torch.arange(start, end, dtype=torch.long, device=device)
                batch_users = user_tensor.repeat(len(batch_items))
                with torch.no_grad():
                    batch_scores = model(batch_users, batch_items)
                for j, score in enumerate(batch_scores.cpu().tolist()):
                    all_scores.append((start + j, score))

            # Sort and take top-K
            all_scores.sort(key=lambda x: x[1], reverse=True)
            for item_idx, score in all_scores[:top_k]:
                real_book_id = index_to_item.get(item_idx)
                if real_book_id is not None:
                    f.write(f"{real_user_id},{real_book_id}\n")
                    total += 1

    logger.info("MF recommendations saved to %s (%d rows)", output_file, total)
    return str(output_file)


# ---- CBMF Inference ----

def generate_cbmf_recommendations(
    model_path: Path | str = "models_output/cb_mf_model.pth",
    book_vector_path: Path | str | None = None,
    mapping_file: Path | str | None = None,
    n_recommend: int = 1,
    output_file: Path | str | None = None,
) -> dict:
    """Batch-generate recommendations using the Content-Based MF model.

    Args:
        model_path: Path to the trained CBMF checkpoint.
        book_vector_path: Path to book_vec.parquet.
        mapping_file: Path to id_mappings.json.
        n_recommend: Number of recommendations per user.
        output_file: Path for output CSV. Defaults to submissions/submission_cbmf.csv.

    Returns:
        Dict mapping real_user_id -> list of (book_id, score) tuples.
    """
    if book_vector_path is None:
        book_vector_path = BOOK_VEC
    if mapping_file is None:
        mapping_file = ID_MAPPINGS
    if output_file is None:
        output_file = SUBMISSION_DIR / "submission_cbmf.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load checkpoint
    checkpoint = _load_checkpoint_safe(model_path, device)
    logger.info("Model checkpoint loaded from %s", model_path)

    # Load book vectors
    book_vectors, book_id_to_index = load_book_vectors(book_vector_path)

    # Reconstruct model
    config = checkpoint.get("config", CBMFConfig)
    n_users = checkpoint["n_users"]
    n_items = checkpoint["n_items"]
    book_vectors_tensor = torch.FloatTensor(book_vectors).to(device)

    model = ContentBasedMatrixFactorization(n_users, n_items, book_vectors_tensor, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load mappings and prepare user list
    mappings = _load_mappings(mapping_file)
    index_to_book_id = {idx: bid for bid, idx in book_id_to_index.items()}
    all_item_indices = torch.arange(n_items, dtype=torch.long, device=device)
    real_user_ids = _get_real_user_ids(mappings)
    user_indices = _find_indices(real_user_ids, mappings)

    results = {}
    logger.info("Starting batch recommendation for %d users...", len(user_indices))

    with torch.no_grad():
        for real_uid, user_idx in tqdm(
            zip(real_user_ids, user_indices), total=len(user_indices),
            desc="Batch processing users"
        ):
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            user_batch = user_tensor.repeat(n_items)
            scores = model(user_batch, all_item_indices)

            scores_np = scores.cpu().numpy()
            top_indices = scores_np.argsort()[::-1][:n_recommend]

            user_recs = [
                (index_to_book_id.get(int(idx), f"book_{int(idx+1)}"),
                 round(float(scores_np[idx]), 4))
                for idx in top_indices
            ]
            results[real_uid] = user_recs

    # Write submission CSV
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("user_id,book_id\n")
        for user_id, recs in results.items():
            for book_id, _ in recs:
                f.write(f"{user_id},{book_id}\n")
                total += 1

    logger.info("CBMF recommendations saved to %s (%d rows)", output_file, total)
    return results
