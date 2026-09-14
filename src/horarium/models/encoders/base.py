"""The Encoder protocol every state encoder implements, and the batch it consumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

__all__ = ["Encoder", "EncoderInput"]


@dataclass(frozen=True, slots=True)
class EncoderInput:
    """One batch of observations of a single instance.

    `edge_index` and `edge_attr` describe the conflict graph, which is fixed for an instance and
    therefore shared across the batch. A structure-free encoder ignores both; that is the whole
    difference between the flat control arm and the GNN.
    """

    nodes: torch.Tensor
    """(batch, n_courses, node_features)"""
    globals: torch.Tensor
    """(batch, 1, global_features)"""
    edge_index: torch.Tensor
    """(2, n_edges), int64, both directions present"""
    edge_attr: torch.Tensor
    """(n_edges, edge_features)"""


@runtime_checkable
class Encoder(Protocol):
    """Turns node and global features into per-course embeddings plus one graph embedding."""

    embedding_dim: int

    def __call__(self, batch: EncoderInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of observations.

        Args:
            batch: The observations to encode.

        Returns:
            Node embeddings of shape `(batch, n_courses, dim)` and a graph embedding of shape
            `(batch, dim)`.
        """
        ...
