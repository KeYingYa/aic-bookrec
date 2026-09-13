"""Base trainer with shared training, evaluation, and checkpointing logic.

Extracted from duplicated code in MF.py (PyTorchMF) and CBMF.py (ContentBasedMF).
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config.base import BaseConfig

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Abstract trainer for recommender models.

    Subclasses override _configure_optimizer, _configure_scheduler, and
    optionally _on_epoch_end for model-specific behavior.
    """

    def __init__(self, model: nn.Module, config: type[BaseConfig]):
        self.config = config
        self.device = config.get_device()
        self.model = model.to(self.device)

        self.optimizer = self._configure_optimizer()
        self.criterion = nn.MSELoss()
        self.scheduler = self._configure_scheduler()

        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    # ---- Subclass hooks ----

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer. Override in subclasses."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.TRAINING["learning_rate"],
            weight_decay=self.config.TRAINING["weight_decay"],
        )

    def _configure_scheduler(self):
        """Create the LR scheduler. Override in subclasses."""
        if not self.config.SCHEDULER.get("enabled", True):
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.SCHEDULER["factor"],
            patience=self.config.SCHEDULER["patience"],
            min_lr=self.config.SCHEDULER["min_lr"],
        )

    # ---- Training ----

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch.

        Returns:
            Average training loss for this epoch.
        """
        self.model.train()
        epoch_loss = 0.0

        for batch_users, batch_items, batch_ratings in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_users, batch_items)
            loss = self.criterion(predictions, batch_ratings)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def train_model(self, train_data: np.ndarray,
                    val_data: np.ndarray | None = None) -> dict[str, float]:
        """Full training loop.

        Args:
            train_data: shape (N, 3) [user_idx, item_idx, rating]
            val_data: optional validation data, same shape.

        Returns:
            Dict with final training metrics.
        """
        epochs = self.config.TRAINING["epochs"]
        batch_size = self.config.TRAINING["batch_size"]
        early_stopping = self.config.EARLY_STOPPING

        # Prepare DataLoader
        users = torch.tensor(train_data[:, 0], dtype=torch.long, device=self.device)
        items = torch.tensor(train_data[:, 1], dtype=torch.long, device=self.device)
        ratings = torch.tensor(train_data[:, 2], dtype=torch.float32, device=self.device)
        dataloader = DataLoader(
            TensorDataset(users, items, ratings),
            batch_size=batch_size, shuffle=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            self.loss_history.append(avg_loss)

            # Validation
            val_loss = None
            if val_data is not None and len(val_data) > 0:
                val_loss = self.evaluate(val_data)
                self.val_loss_history.append(val_loss)

            # Scheduler step
            if self.scheduler is not None:
                monitor = val_loss if val_loss is not None else avg_loss
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitor)
                else:
                    self.scheduler.step()

            # Early stopping
            if early_stopping.get("enabled", False) and val_loss is not None:
                if val_loss < best_val_loss - early_stopping.get("delta", 0.001):
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping.get("patience", 5):
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

            # Logging
            if (epoch + 1) % self.config.LOGGING.get("log_interval", 1) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %03d/%d | TrainLoss=%.4f | ValLoss=%s | LR=%.2e",
                    epoch + 1, epochs, avg_loss,
                    f"{val_loss:.4f}" if val_loss is not None else "N/A",
                    lr,
                )

        logger.info("Training complete! Final train loss: %.4f", self.loss_history[-1])
        if self.config.LOGGING.get("save_model", True):
            self.save_checkpoint()

        return {"final_train_loss": self.loss_history[-1]}

    # ---- Evaluation ----

    def evaluate(self, test_data: np.ndarray) -> float:
        """Compute RMSE on test data.

        Args:
            test_data: shape (N, 3) [user_idx, item_idx, rating]

        Returns:
            RMSE value.
        """
        self.model.eval()
        with torch.no_grad():
            users = torch.tensor(test_data[:, 0], dtype=torch.long, device=self.device)
            items = torch.tensor(test_data[:, 1], dtype=torch.long, device=self.device)
            ratings = torch.tensor(test_data[:, 2], dtype=torch.float32, device=self.device)

            predictions = self.model(users, items)
            rmse = torch.sqrt(self.criterion(predictions, ratings))
            return rmse.item()

    # ---- Checkpointing ----

    def save_checkpoint(self, filepath: Path | str | None = None) -> None:
        """Save model checkpoint with metadata.

        Args:
            filepath: Target path. Defaults to config.LOGGING['model_path'].
        """
        if filepath is None:
            filepath = self.config.LOGGING["model_path"]

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "val_loss_history": self.val_loss_history,
            "config": self.config,
            "n_users": self.model.n_users,
            "n_items": self.model.n_items,
        }

        # Model-specific metadata
        if hasattr(self.model, "n_factors"):
            checkpoint["n_factors"] = self.model.n_factors
        if hasattr(self.model, "book_vector_dim"):
            checkpoint["book_vector_dim"] = self.model.book_vector_dim

        torch.save(checkpoint, str(filepath))
        logger.info("Checkpoint saved to %s", filepath)

    def load_checkpoint(self, filepath: Path | str) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            filepath: Path to .pth file.

        Returns:
            The checkpoint dict.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        try:
            checkpoint = torch.load(str(filepath), map_location=self.device,
                                    weights_only=True)
        except Exception:
            logger.warning("Legacy checkpoint, loading with weights_only=False")
            checkpoint = torch.load(str(filepath), map_location=self.device,
                                    weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Checkpoint loaded from %s", filepath)
        return checkpoint

    # ---- Plotting ----

    def plot_loss(self, save_dir: Path | str | None = None) -> None:
        """Plot training and validation loss curves.

        Args:
            save_dir: Directory for plot images. Defaults to plots/.
        """
        if not self.config.LOGGING.get("plot_loss", True):
            return

        if save_dir is None:
            save_dir = Path("plots")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure(figsize=(15, 6))

        # Subplot 1: Training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, "b-", linewidth=2, label="Training Loss")
        plt.title("Training Loss Curve", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if self.loss_history:
            final = self.loss_history[-1]
            plt.annotate(f"Final: {final:.4f}", xy=(len(self.loss_history) - 1, final),
                         xytext=(10, 10), textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Subplot 2: Validation or detailed train loss
        plt.subplot(1, 2, 2)
        if self.val_loss_history:
            plt.plot(self.val_loss_history, "r-", linewidth=2, label="Validation RMSE")
            plt.title("Validation RMSE Curve", fontsize=14, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.legend()
            if self.val_loss_history:
                final_rmse = self.val_loss_history[-1]
                plt.annotate(f"Final: {final_rmse:.4f}",
                             xy=(len(self.val_loss_history) - 1, final_rmse),
                             xytext=(10, 10), textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        else:
            plt.plot(self.loss_history, "g-", linewidth=2, alpha=0.7,
                     label="Training Loss (Detailed)")
            plt.title("Training Loss (Last 50 Epochs)", fontsize=14, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            if len(self.loss_history) > 50:
                plt.xlim(len(self.loss_history) - 50, len(self.loss_history) - 1)

        plt.tight_layout()

        plot_file = save_dir / f"loss_curve_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Loss plot saved to %s", plot_file)
