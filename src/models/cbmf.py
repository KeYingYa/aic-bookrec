"""Content-Based Matrix Factorization model.

Uses book content vectors to initialize item latent factors via a learned
mapping network, then performs collaborative filtering.

Refactored from CBMF.py ContentBasedMatrixFactorization class.
FIX: Removed unused self.interaction_net MLP (was defined but never used in forward).
"""

import torch
import torch.nn as nn

from src.config.cbmf_config import CBMFConfig
from src.models.base import BaseRecommender


class ContentBasedMatrixFactorization(BaseRecommender):
    """Matrix factorization where item factors are derived from content vectors.

    A content_mapping MLP transforms raw book vectors (e.g. 512-dim) into
    the latent factor space (e.g. 1024-dim). Item factors are trainable
    Parameters initialized from this mapping.
    """

    def __init__(self, n_users: int, n_items: int,
                 book_vectors: torch.Tensor,
                 config: type[CBMFConfig] | None = None):
        config = config if config is not None else CBMFConfig
        super().__init__(n_users, n_items, config)

        device = config.get_device()
        self.book_vector_dim = book_vectors.shape[1]

        # User embedding
        self.user_factors = nn.Embedding(n_users, self.latent_dim)

        # Content → latent space mapping
        self.content_mapping = nn.Sequential(
            nn.Linear(self.book_vector_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim),
        )

        # Initialize item factors from book vectors (trainable)
        book_vectors = book_vectors.to(device)
        with torch.no_grad():
            init_item_latent = self.content_mapping(book_vectors)
        self.item_factors = nn.Parameter(init_item_latent)

        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Parameter(torch.zeros(n_items, 1))
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        self.to(device)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize user embeddings and biases."""
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.zeros_(self.item_bias)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Compute predicted ratings via element-wise product of latent factors.

        Args:
            user: shape (batch_size,)
            item: shape (batch_size,)

        Returns:
            Predicted ratings, shape (batch_size,)
        """
        u_vec = self.user_factors(user)             # (B, latent_dim)
        u_b = self.user_bias(user).squeeze()         # (B,)
        i_vec = self.item_factors[item]              # (B, latent_dim)
        i_b = self.item_bias[item].squeeze()         # (B,)

        score = (u_vec * i_vec).sum(dim=1)            # element-wise product
        return self.global_bias + u_b + i_b + score
