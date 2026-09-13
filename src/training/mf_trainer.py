"""Matrix Factorization trainer.

Refactored from MF.py PyTorchMF class.
"""

import logging

import torch

from src.config.mf_config import MFConfig
from src.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MFTrainer(BaseTrainer):
    """Trainer for the standard Matrix Factorization model."""

    def __init__(self, model, config: type[MFConfig] | None = None):
        if config is None:
            config = MFConfig
        super().__init__(model, config)

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.TRAINING["learning_rate"],
            weight_decay=self.config.TRAINING["weight_decay"],
            betas=self.config.OPTIMIZER["betas"],
            eps=self.config.OPTIMIZER["eps"],
            amsgrad=self.config.OPTIMIZER["amsgrad"],
        )

    def _configure_scheduler(self):
        if not self.config.SCHEDULER.get("enabled", True):
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.SCHEDULER["factor"],
            patience=self.config.SCHEDULER["patience"],
            min_lr=self.config.SCHEDULER["min_lr"],
            verbose=False,
        )
