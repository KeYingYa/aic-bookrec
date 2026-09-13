"""Matrix Factorization (MF) model configuration.

Inherits from BaseConfig and overrides MF-specific hyperparameters.
"""

from src.config.base import BaseConfig


class MFConfig(BaseConfig):
    """Configuration for the standard Matrix Factorization model."""

    MODEL = {
        **BaseConfig.MODEL,
        "n_factors": 20,
    }

    TRAINING = {
        **BaseConfig.TRAINING,
        "epochs": 250,
        "batch_size": 2048,
        "learning_rate": 2e-4,
        "weight_decay": 0.001,
        "random_state": 42,
        "n_keep_val": 2,
    }

    OPTIMIZER = {
        **BaseConfig.OPTIMIZER,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": True,
    }

    DATA = {
        **BaseConfig.DATA,
        "file_path": "DataLoad/user_inter_matrix.parquet",
    }

    EARLY_STOPPING = {
        **BaseConfig.EARLY_STOPPING,
        "enabled": False,
    }

    SCHEDULER = {
        **BaseConfig.SCHEDULER,
        "enabled": True,
        "factor": 0.5,
        "patience": 10,
        "min_lr": 1e-6,
    }

    LOGGING = {
        **BaseConfig.LOGGING,
        "log_interval": 100,
        "model_path": "models/fin_model.pth",
        "plot_loss": True,
    }
