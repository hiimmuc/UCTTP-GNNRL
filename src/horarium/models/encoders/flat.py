"""Structure-free encoder: an MLP per course plus pooling. The E3 control arm, not a stub."""

from __future__ import annotations

import torch
from torch import nn

from .base import EncoderInput

__all__ = ["FlatEncoder"]


class FlatEncoder(nn.Module):
    """Per-course MLP, mean and max pooled into a graph embedding alongside the global features.

    It sees exactly the features the GNN sees, minus the conflict structure: `edge_index` and
    `edge_attr` are deliberately unused, so any gap between this and the GNN is attributable to
    the structure rather than to the inputs.
    """

    def __init__(
        self,
        node_features: int,
        global_features: int,
        edge_features: int = 0,  # noqa: ARG002 - part of the shared encoder ctor, unused here
        embedding_dim: int = 64,
    ) -> None:
        """Build the per-node and pooling MLPs.

        Args:
            node_features: Width of the per-course node feature vector.
            global_features: Width of the per-instance global feature vector.
            edge_features: Edge feature width; accepted for a uniform constructor, unused.
            embedding_dim: Width of every embedding this encoder produces.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim + global_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, batch: EncoderInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed every course independently, then pool into one graph embedding.

        Args:
            batch: The observations to encode.

        Returns:
            Node embeddings of shape `(batch, n_courses, dim)` and a graph embedding of shape
            `(batch, dim)`.
        """
        node_embeddings = self.node_mlp(batch.nodes)
        pooled = torch.cat(
            [node_embeddings.mean(dim=1), node_embeddings.amax(dim=1), batch.globals.squeeze(1)],
            dim=-1,
        )
        return node_embeddings, self.graph_mlp(pooled)
