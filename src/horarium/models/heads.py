"""Policy and value heads scoring every (course, period) pair from the encoder's embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["ActionHead", "ValueHead"]


class ActionHead(nn.Module):
    """Scores (course, period) actions as a scaled dot product between two projections.

    A full MLP over every concatenated pair would cost `n_courses * n_periods` forward passes;
    projecting each side once and taking the product costs one pass each and scales to the
    Erlangen instances. The graph embedding modulates the course side (FiLM), so global state
    still reaches every action.
    """

    def __init__(self, embedding_dim: int, period_features: int, hidden: int = 64) -> None:
        """Build the period encoder, the two projections and the FiLM modulation.

        Args:
            embedding_dim: Width of the encoder's node and graph embeddings.
            period_features: Width of the per-period feature vector.
            hidden: Width of the projected course/period space the dot product is taken in.
        """
        super().__init__()
        self.hidden = hidden
        self.period_mlp = nn.Sequential(
            nn.Linear(period_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.film = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.course_proj = nn.Linear(embedding_dim, hidden)
        self.period_proj = nn.Linear(hidden, hidden)
        self.course_bias = nn.Linear(embedding_dim, 1)
        self.period_bias = nn.Linear(hidden, 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        period_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score every (course, period) action.

        Args:
            node_embeddings: Per-course embeddings, shape `(batch, n_courses, embedding_dim)`.
            graph_embedding: Per-instance embedding, shape `(batch, embedding_dim)`.
            period_features: Per-period features, shape `(batch, n_periods, period_features)`.

        Returns:
            Flat `(batch, n_courses * n_periods)` logits, row-major over (course, period).
        """
        scale, shift = self.film(graph_embedding).chunk(2, dim=-1)
        courses = node_embeddings * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        periods = self.period_mlp(period_features)

        scores = torch.einsum(
            "bnh,bph->bnp", self.course_proj(courses), self.period_proj(periods)
        ) / math.sqrt(self.hidden)
        scores = scores + self.course_bias(courses) + self.period_bias(periods).transpose(1, 2)
        logits: torch.Tensor = scores.flatten(start_dim=1)
        return logits


class ValueHead(nn.Module):
    """State value from the graph embedding."""

    def __init__(self, embedding_dim: int, hidden: int = 64) -> None:
        """Build the value MLP.

        Args:
            embedding_dim: Width of the encoder's graph embedding.
            hidden: Width of the MLP's hidden layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """Estimate the state value from the graph embedding.

        Args:
            graph_embedding: Per-instance embedding, shape `(batch, embedding_dim)`.

        Returns:
            State values, shape `(batch,)`.
        """
        values: torch.Tensor = self.mlp(graph_embedding)
        return values.squeeze(-1)
