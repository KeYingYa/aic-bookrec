"""Standard Matrix Factorization model with neural scoring head.

Refactored from MF.py MatrixFactorization class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.mf_config import MFConfig
from src.models.base import BaseRecommender


class MatrixFactorization(BaseRecommender):
    """Matrix Factorization with user/item embeddings and a small MLP score head.

    Implicit features (dot product, cosine similarity) are concatenated
    with the raw embeddings before passing through the score_head MLP.
    """

    def __init__(self, n_users: int, n_items: int,
                 config: type[MFConfig] | None = None):
        super().__init__(n_users, n_items, config if config is not None else MFConfig)
        self.n_factors = self.config.MODEL["n_factors"]

        # Embedding layers
        self.user_factors = nn.Embedding(n_users, self.n_factors)
        self.item_factors = nn.Embedding(n_items, self.n_factors)

        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        # Score head: [user_vec, item_vec, dot_product, cosine_sim] -> prediction
        total_features = 2 * self.n_factors + 2
        self.score_head = nn.Sequential(
            nn.Linear(total_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding and bias weights."""
        self.user_factors.weight.data.uniform_(0, 0.1)
        self.item_factors.weight.data.uniform_(0, 0.1)
        self.user_bias.weight.data.uniform_(0, 0.01)
        self.item_bias.weight.data.uniform_(0, 0.01)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Compute predicted ratings for user-item pairs.

        Args:
            user: shape (batch_size,)
            item: shape (batch_size,)

        Returns:
            Predicted ratings, shape (batch_size,)
        """
        # Embeddings
        user_vec = self.user_factors(user)   # (B, n_factors)
        item_vec = self.item_factors(item)   # (B, n_factors)

        # Implicit features
        dot_product = torch.sum(user_vec * item_vec, dim=1, keepdim=True)  # (B, 1)
        cosine_sim = F.cosine_similarity(user_vec, item_vec, dim=1).unsqueeze(1)

        # Concatenate features
        feat = torch.cat([user_vec, item_vec, dot_product, cosine_sim], dim=1)

        # Score from MLP
        score = self.score_head(feat).squeeze(1)  # (B,)

        # Biases
        u_bias = self.user_bias(user).squeeze()
        i_bias = self.item_bias(item).squeeze()

        return self.global_bias + u_bias + i_bias + score
