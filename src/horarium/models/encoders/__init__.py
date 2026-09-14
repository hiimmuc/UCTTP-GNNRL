"""Interchangeable state encoders.

Add one by implementing the `Encoder` protocol and adding a single line to `ENCODERS`. Nothing
in the environment, the agent, the trainer or the solver needs to change.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

from .base import Encoder, EncoderInput
from .flat import FlatEncoder
from .gnn import CONVOLUTIONS, GNNEncoder

__all__ = ["ENCODERS", "Encoder", "EncoderInput", "EncoderSpec", "FlatEncoder", "GNNEncoder"]

#: Every encoder the CLIs and checkpoints can name. Each GNN entry fixes one graph-convolution.
ENCODERS: dict[str, Callable[..., Encoder]] = {
    "flat": FlatEncoder,
    **{name: functools.partial(GNNEncoder, conv=name) for name in CONVOLUTIONS},
}


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Which encoder to build and how wide. Recorded in checkpoints so they can be reloaded."""

    name: str
    embedding_dim: int = 64

    def build(self, *, node_features: int, global_features: int, edge_features: int) -> Encoder:
        """Construct the encoder for an environment's feature widths.

        Args:
            node_features: Width of the per-course node feature vector.
            global_features: Width of the per-instance global feature vector.
            edge_features: Width of the per-edge feature vector.

        Returns:
            A freshly constructed encoder.

        Raises:
            KeyError: No encoder is registered under this spec's name.
        """
        if self.name not in ENCODERS:
            msg = f"unknown encoder {self.name!r}; registered: {sorted(ENCODERS)}"
            raise KeyError(msg)
        return ENCODERS[self.name](
            node_features=node_features,
            global_features=global_features,
            edge_features=edge_features,
            embedding_dim=self.embedding_dim,
        )
