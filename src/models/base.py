"""Abstract base class for recommender models."""

import torch
import torch.nn as nn

from src.config.base import BaseConfig


class BaseRecommender(nn.Module):
    """Abstract recommender model with common initialization patterns."""

    def __init__(self, n_users: int, n_items: int, config: type[BaseConfig] | None = None):
        """Initialize the recommender.

        Args:
            n_users: Number of users (embedding vocabulary size).
            n_items: Number of items (embedding vocabulary size).
            config: Configuration class. If None, uses BaseConfig.
        """
        super().__init__()
        self.config = config if config is not None else BaseConfig
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = self.config.MODEL.get("n_factors", 20)

    def _init_weights(self) -> None:
        """Initialize embedding weights. Override in subclasses."""
        pass

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Forward pass. Must be overridden.

        Args:
            user: Tensor of user indices, shape (batch_size,).
            item: Tensor of item indices, shape (batch_size,).

        Returns:
            Tensor of predicted ratings, shape (batch_size,).
        """
        raise NotImplementedError
