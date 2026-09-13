"""Dataset loading and train/validation splitting.

Unified implementation (was duplicated in MF.py and CBMF.py).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.base import BaseConfig

logger = logging.getLogger(__name__)


def split_one_user(g: pd.DataFrame, n_keep: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-last-N-out split for a single user's interactions.

    Keeps the last `n_keep` interactions as validation, the rest as training.

    Args:
        g: Sorted DataFrame of one user's interactions.
        n_keep: Number of last interactions to hold out for validation.

    Returns:
        Tuple of (train_df, valid_df).
    """
    if len(g) < n_keep + 1:
        return g, g.iloc[:0]
    valid = g.tail(n_keep)
    train = g.drop(valid.index)
    return train, valid


def load_interaction_data(
    config: type[BaseConfig],
    file_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load interaction data and split into train/validation sets.

    Reads a wide-format user-item matrix, converts to long format,
    and applies a leave-last-N-out split per user.

    Args:
        config: Configuration class (BaseConfig subclass).
        file_path: Path to the parquet file. Defaults to config.DATA['file_path'].

    Returns:
        Tuple of (train_data, val_data, df_reset) where:
        - train_data: shape (N, 3) numpy array [user_idx, item_idx, rating]
        - val_data: shape (M, 3) numpy array
        - df_reset: the wide-format DataFrame with reset index
    """
    if file_path is None:
        file_path = config.DATA["file_path"]

    file_path = Path(file_path)
    n_keep = config.TRAINING.get("n_keep_val", 2)

    logger.info("Loading interaction file: %s", file_path)
    df = pd.read_parquet(file_path)
    logger.info("Interaction data loaded: %s", df.shape)

    df_reset = df.reset_index(drop=True)
    df_reset.columns = range(df_reset.shape[1])

    # Vectorized wide-to-long conversion
    logger.info("Converting interaction matrix to long format...")
    stacked = df_reset.stack(dropna=True)
    records = pd.DataFrame({
        "user": stacked.index.get_level_values(0).astype(int),
        "item": stacked.index.get_level_values(1).astype(int),
        "rating": stacked.values.astype(float),
        "time": stacked.index.get_level_values(1).astype(int),
    }).reset_index(drop=True)

    logger.info("Valid rating count: %d", len(records))

    # Sort by user then time
    records = records.sort_values(["user", "time"])

    # Leave-last-N-out split
    logger.info("Performing leave-last-%d-out split...", n_keep)
    chunks = records.groupby("user", group_keys=False).apply(
        lambda g: split_one_user(g, n_keep=n_keep)
    )
    train_df = pd.concat([c[0] for c in chunks]).reset_index(drop=True)
    val_df = pd.concat([c[1] for c in chunks]).reset_index(drop=True)

    train_data = train_df[["user", "item", "rating"]].values
    val_data = val_df[["user", "item", "rating"]].values

    logger.info("Split complete — train: %d, val: %d", len(train_data), len(val_data))
    return train_data, val_data, df_reset
