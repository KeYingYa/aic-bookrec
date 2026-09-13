"""Content-Based Matrix Factorization (CBMF) model configuration.

Inherits from BaseConfig and adds content-specific sections.
"""

from src.config.base import BaseConfig


class CBMFConfig(BaseConfig):
    """Configuration for the Content-Based Matrix Factorization model."""

    MODEL = {
        **BaseConfig.MODEL,
        "n_factors": 1024,
        "book_vector_dim": 512,
    }

    TRAINING = {
        **BaseConfig.TRAINING,
        "epochs": 150,
        "batch_size": 4096,
        "learning_rate": 1e-2,
        "weight_decay": 4e-5,
        "random_state": 42,
        "n_keep_val": 2,
    }

    OPTIMIZER = {
        **BaseConfig.OPTIMIZER,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    }

    DATA = {
        **BaseConfig.DATA,
        "file_path": "DataLoad/user_inter_matrix.parquet",
        "rating_range": (1, 10),
    }

    CONTENT = {
        "book_vector_path": "DataLoad/book_vec.parquet",
    }

    EARLY_STOPPING = {
        **BaseConfig.EARLY_STOPPING,
        "enabled": True,
        "patience": 3,
        "delta": 0.001,
        "monitor": "loss",
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
        "log_interval": 10,
        "model_path": "models/cb_mf_model.pth",
        "plot_loss": True,
    }
