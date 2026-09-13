"""Preprocessing: rating computation and book vector generation.

Refactored from DataLoader.py.
"""

import json
from pathlib import Path

import pandas as pd

from src.config.paths import (
    INTER_REEVALUATION_CSV, ITEM_CSV, BOOK_VEC, INTER2USER,
    INTER2BOOK, INTER2RATING, MODEL_DIR, DATA_DIR,
)


def add_rating(input_csv: Path | str | None = None,
               output_dir: Path | str | None = None,
               reference_date: str = "2025-11-02") -> pd.DataFrame:
    """Compute implicit ratings from borrowing records and save mappings.

    Rating formula:
        score = 0.8 * normalized_borrow_duration + 0.2 * normalized_renewal_count
        rating = score * 9 + 1  (scaled to [1, 10])

    Args:
        input_csv: Path to inter_reevaluation.csv. Defaults to DataLoad version.
        output_dir: Directory for output JSON files. Defaults to DataLoad/.
        reference_date: Fixed reference date for 'now' (prevents non-deterministic
                        ratings for unreturned books across runs).

    Returns:
        DataFrame with added 'rating' column.
    """
    if input_csv is None:
        input_csv = INTER_REEVALUATION_CSV
    if output_dir is None:
        output_dir = DATA_DIR

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = None
    for enc in ("gbk", "gb2312", "utf-8", "ansi"):
        try:
            df = pd.read_csv(input_csv, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if df is None:
        raise ValueError(f"Failed to read {input_csv}: encoding error")

    # Time-based rating computation
    borrow_time = pd.to_datetime(df["借阅时间"])
    return_time = pd.to_datetime(df["还书时间"], format="%Y-%m-%d%H:%M:%S")
    now_time = pd.Timestamp(reference_date)
    actual_duration = (return_time - borrow_time).fillna(now_time - borrow_time)
    time_days = actual_duration.dt.total_seconds() / 86400

    dur = time_days.clip(lower=0, upper=365)
    dur_norm = (dur - dur.min()) / (dur.max() - dur.min() + 1e-8)

    reborrow = df["续借次数"].clip(upper=5)
    reborrow_norm = (reborrow - reborrow.min()) / (reborrow.max() - reborrow.min() + 1e-8)

    score = 0.8 * dur_norm + 0.2 * reborrow_norm
    df["rating"] = (score * 9 + 1).round(3)

    # Build 1-to-1 mappings
    inter2user = dict(zip(df["inter_id"], df["user_id"]))
    inter2book = dict(zip(df["inter_id"], df["book_id"]))
    inter2rating = dict(zip(df["inter_id"], df["rating"]))

    (output_dir / "inter2user.json").write_text(
        json.dumps(inter2user, ensure_ascii=False, indent=2))
    (output_dir / "inter2book.json").write_text(
        json.dumps(inter2book, ensure_ascii=False, indent=2))
    (output_dir / "inter2rating.json").write_text(
        json.dumps(inter2rating, ensure_ascii=False, indent=2))

    print(f"Rating computation complete. {len(df)} records processed.")
    return df


def tovec(csv_path: Path | str | None = None,
          model_dir: Path | str | None = None,
          save_path: Path | str | None = None,
          id_col: str = "book_id",
          batch_size: int = 64) -> pd.DataFrame | None:
    """Generate book embeddings using a SentenceTransformer model.

    Concatenates all non-ID columns as text, then encodes with the model.

    Args:
        csv_path: Path to item metadata CSV. Defaults to DataLoad/item.csv.
        model_dir: Path to SentenceTransformer model directory.
        save_path: Output path for book_vec.parquet.
        id_col: Column name for the identifier.
        batch_size: Encoding batch size.

    Returns:
        DataFrame with [id_col, book_vector] columns, or None on error.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence_transformers not installed. Skipping tovec().")
        return None

    if csv_path is None:
        csv_path = ITEM_CSV
    if model_dir is None:
        model_dir = MODEL_DIR
    if save_path is None:
        save_path = BOOK_VEC

    csv_file = Path(csv_path)
    save_file = Path(save_path)
    model_path = Path(model_dir)

    # Ensure model directory structure
    pooling_dir = model_path / "1_Pooling"
    if not pooling_dir.exists():
        pooling_dir.mkdir(parents=True, exist_ok=True)
        pooling_config = {
            "word_embedding_dimension": 512,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        }
        with open(pooling_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(pooling_config, f, indent=2)

    modules_file = model_path / "modules.json"
    if not modules_file.exists():
        modules_config = [
            {"type": "sentence_transformers.models.Transformer",
             "path": ".", "tokenizer_name": "."},
            {"type": "sentence_transformers.models.Pooling",
             "word_embedding_dimension": 512, "pooling_mode": "mean"},
        ]
        with open(modules_file, "w", encoding="utf-8") as f:
            json.dump(modules_config, f, indent=2)

    model = SentenceTransformer(str(model_dir))

    # Read item CSV (try common Chinese encodings)
    df = None
    for enc in ("gbk", "gb2312", "utf-8", "ansi"):
        try:
            df = pd.read_csv(csv_file, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if df is None:
        raise ValueError(f"Failed to read {csv_file}: encoding error")
    text_cols = [c for c in df.columns if c != id_col]
    df["text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    embs = model.encode(df["text"].tolist(), normalize_embeddings=True,
                        batch_size=batch_size)
    df["book_vector"] = list(embs)

    out_df = df[[id_col, "book_vector"]]
    save_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(save_file, index=False)

    print(f"Book vectors saved to {save_file}. Shape: {out_df.shape}, "
          f"dim: {embs.shape[1]}")
    return out_df
