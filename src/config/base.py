"""Base configuration for all recommender models.

All model-specific configs inherit from this class and override sections as needed.
"""

import logging
from typing import ClassVar

import torch

logger = logging.getLogger(__name__)


class BaseConfig:
    """Shared configuration sections used by all recommendation models."""

    # Model architecture
    MODEL: ClassVar[dict] = {
        "n_factors": 20,
    }

    # Training hyperparameters
    TRAINING: ClassVar[dict] = {
        "epochs": 250,
        "batch_size": 2048,
        "learning_rate": 2e-4,
        "weight_decay": 0.001,
        "random_state": 42,
        "n_keep_val": 2,  # items held out per user for validation (leave-last-one-out)
    }

    # Optimizer settings
    OPTIMIZER: ClassVar[dict] = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    }

    # Data paths and constraints
    DATA: ClassVar[dict] = {
        "file_path": "DataLoad/user_inter_matrix.parquet",
        "rating_range": (1, 10),
        "min_user_ratings": 1,
        "min_item_ratings": 1,
    }

    # Early stopping
    EARLY_STOPPING: ClassVar[dict] = {
        "enabled": False,
        "patience": 5,
        "delta": 0.001,
        "monitor": "loss",
    }

    # Device
    DEVICE: ClassVar[dict] = {
        "use_gpu": True,
        "device_id": 0,
    }

    # Logging and output
    LOGGING: ClassVar[dict] = {
        "verbose": True,
        "log_interval": 100,
        "save_model": True,
        "model_path": "models/best_model.pth",
        "plot_loss": True,
    }

    # Learning rate scheduler
    SCHEDULER: ClassVar[dict] = {
        "enabled": True,
        "factor": 0.5,
        "patience": 10,
        "min_lr": 1e-6,
    }

    # Evaluation
    EVALUATION: ClassVar[dict] = {
        "top_k": 1,
        "metrics": ["rmse", "mae"],
        "verbose": True,
    }

    # ---- Class methods ----

    @classmethod
    def get_device(cls) -> torch.device:
        """Resolve the PyTorch device (CPU or CUDA)."""
        if cls.DEVICE["use_gpu"] and torch.cuda.is_available():
            return torch.device(f"cuda:{cls.DEVICE['device_id']}")
        return torch.device("cpu")

    @classmethod
    def print_config(cls) -> None:
        """Log the current configuration."""
        logger.info("=" * 50)
        logger.info("%s Configuration", cls.__name__)
        logger.info("=" * 50)

        for section_name in sorted(cls.__dict__):
            if section_name.startswith("_"):
                continue
            section = getattr(cls, section_name)
            if not isinstance(section, dict):
                continue
            logger.info("[%s]", section_name)
            for key, value in section.items():
                logger.info("  %s: %s", key, value)

        logger.info("Device: %s", cls.get_device())
        logger.info("=" * 50)

    @classmethod
    def merge_overrides(cls, overrides: dict | None) -> None:
        """Merge a dict of section overrides into class variables.

        Args:
            overrides: Dict of the form {SECTION_NAME: {key: value, ...}, ...}
        """
        if overrides is None:
            return
        for section_name, section_overrides in overrides.items():
            if hasattr(cls, section_name):
                current = getattr(cls, section_name)
                if isinstance(current, dict) and isinstance(section_overrides, dict):
                    current.update(section_overrides)
