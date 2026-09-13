"""CLI: Train recommendation models (MF or CBMF)."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_utils import setup_logging
from src.config.mf_config import MFConfig
from src.config.cbmf_config import CBMFConfig
from src.config.paths import USER_INTER_MATRIX, BOOK_VEC
from src.data.dataset import load_interaction_data
from src.models.mf import MatrixFactorization
from src.models.cbmf import ContentBasedMatrixFactorization
from src.training.mf_trainer import MFTrainer
from src.training.cbmf_trainer import CBMFTrainer
from src.utils.book_vectors import load_book_vectors


def train_mf(config_overrides: dict | None = None) -> None:
    """Train the standard Matrix Factorization model."""
    logger, _ = setup_logging("mf")
    config = MFConfig
    if config_overrides:
        config.merge_overrides(config_overrides)

    config.print_config()

    train_data, val_data, df = load_interaction_data(config, file_path=USER_INTER_MATRIX)
    n_users, n_items = df.shape
    logger.info("Creating MF model: %d users, %d items", n_users, n_items)

    model = MatrixFactorization(n_users, n_items, config)
    trainer = MFTrainer(model, config)

    logger.info("Starting MF training...")
    trainer.train_model(train_data, val_data)
    trainer.plot_loss()
    logger.info("MF training complete.")


def train_cbmf(config_overrides: dict | None = None) -> None:
    """Train the Content-Based Matrix Factorization model."""
    logger, _ = setup_logging("cbmf")
    config = CBMFConfig
    if config_overrides:
        config.merge_overrides(config_overrides)

    config.print_config()

    train_data, val_data, df = load_interaction_data(config, file_path=USER_INTER_MATRIX)
    n_users, n_items = df.shape

    book_vectors_np, book_id_to_index = load_book_vectors(BOOK_VEC)
    if book_vectors_np is None:
        logger.error("Failed to load book vectors")
        return

    n_items = min(n_items, len(book_vectors_np))
    config.MODEL["book_vector_dim"] = book_vectors_np.shape[1]
    logger.info("Creating CBMF model: %d users, %d items, book_dim=%d",
                n_users, n_items, book_vectors_np.shape[1])

    model = ContentBasedMatrixFactorization(n_users, n_items,
                                            book_vectors_np, config)
    trainer = CBMFTrainer(model, config)

    logger.info("Starting CBMF training...")
    trainer.train_model(train_data, val_data)
    trainer.plot_loss()
    logger.info("CBMF training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a recommendation model.")
    parser.add_argument("--model", choices=["mf", "cbmf"], required=True,
                        help="Which model to train.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides.setdefault("TRAINING", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("TRAINING", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("TRAINING", {})["learning_rate"] = args.lr

    if not overrides:
        overrides = None

    if args.model == "mf":
        train_mf(overrides)
    else:
        train_cbmf(overrides)


if __name__ == "__main__":
    main()
