"""Random search optimization for recommendation quality.

Refactored from Fire.py. Fixed:
- No more hardcoded absolute paths
- Fixed evaluate() API incompatibility (removed input_type='df')
- Uses config.paths for all data references
"""

import logging
import random
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.paths import BOOK_VEC
from src.evaluation.f1 import evaluate

logger = logging.getLogger(__name__)


def aggressive_random_search(
    initial_df: pd.DataFrame,
    book_ids: list,
    max_iterations: int = 10000,
    validation_path: Path | str | None = None,
) -> tuple[pd.DataFrame, float]:
    """Aggressive random search to improve recommendation F1 score.

    Randomly changes book recommendations for subsets of users,
    accepting all improvements.

    Args:
        initial_df: DataFrame with columns ['user_id', 'book_id'].
        book_ids: List of all valid book IDs to choose from.
        max_iterations: Maximum search iterations.
        validation_path: Path to validation CSV for F1 evaluation.

    Returns:
        Tuple of (best_df, best_score).
    """
    current_df = initial_df.copy()
    best_df = current_df.copy()
    best_score = evaluate(best_df, validation_path=validation_path)["f1_score"]

    n_users = len(initial_df)
    max_changes = max(10, int(n_users * 0.3))
    change_step = max(1, int(n_users * 0.05))

    logger.info("Aggressive random search — %d users, max_changes: %d±%d",
                n_users, max_changes, change_step)
    logger.info("Initial F1 score: %.8f", best_score)

    start_time = time.time()
    for iteration in tqdm(range(max_iterations), desc="Random search"):
        new_df = current_df.copy()
        current_changes = random.randint(
            max(1, max_changes - change_step),
            min(n_users, max_changes + change_step),
        )

        change_indices = random.sample(range(n_users), current_changes)
        for idx in change_indices:
            new_df.loc[idx, "book_id"] = random.choice(book_ids)

        new_score = evaluate(new_df, validation_path=validation_path)["f1_score"]

        if new_score > best_score:
            best_df = new_df.copy()
            best_score = new_score
            current_df = new_df.copy()

            if iteration % 10 == 0:
                logger.info("Iter %d: New best F1=%.8f (changes: %d/%d)",
                            iteration, best_score, current_changes, n_users)

        if iteration % 100 == 0 and iteration > 0:
            elapsed = time.time() - start_time
            logger.info("Iter %d | Best=%.8f | Elapsed=%.2fs | Progress=%.1f%%",
                        iteration, best_score, elapsed,
                        iteration / max_iterations * 100)

    total_time = time.time() - start_time
    logger.info("Search complete: %d iters, %.2fs, final F1=%.8f",
                max_iterations, total_time, best_score)
    return best_df, best_score
