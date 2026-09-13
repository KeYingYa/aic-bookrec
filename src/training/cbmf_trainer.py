"""Content-Based Matrix Factorization trainer.

Refactored from CBMF.py ContentBasedMF class.
"""

import signal
import logging

import torch
import torch.nn as nn

from src.config.cbmf_config import CBMFConfig
from src.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class CBMFTrainer(BaseTrainer):
    """Trainer for the Content-Based Matrix Factorization model."""

    def __init__(self, model, config: type[CBMFConfig] | None = None):
        if config is None:
            config = CBMFConfig
        super().__init__(model, config)

        # Register interrupt handler for graceful save on Ctrl+C
        self._register_interrupt_handler()

    def _register_interrupt_handler(self) -> None:
        """Save model on SIGINT (Ctrl+C)."""
        def handler(sig, frame):
            print("\nSaving model before exit...")
            self.save_checkpoint()
            print("Model saved. Exiting.")
            exit(0)
        signal.signal(signal.SIGINT, handler)

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.TRAINING["learning_rate"],
            weight_decay=self.config.TRAINING["weight_decay"],
        )

    def _configure_scheduler(self):
        if not self.config.SCHEDULER.get("enabled", True):
            return None
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=80,
            gamma=0.5,
        )
