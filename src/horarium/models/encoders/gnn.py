"""Message-passing encoders over the conflict graph: the structure-aware arm of the study.

One `GNNEncoder` wraps an input projection, a stack of graph-convolution layers and the same
pooled graph head as `FlatEncoder`, so anything it gains over the flat control arm is
attributable to the convolution rather than to the inputs or the head. The convolution is
chosen by name; `CONVOLUTIONS` lists the layers, and `encoders.ENCODERS` registers one entry
per layer (`gcn`, `sage`, `gat`, `gin`, `mpnn`).

The conflict graph is fixed for an instance, so `edge_index` and `edge_attr` are shared across
the batch: node states carry the `(batch, n_courses, dim)` axis, edges do not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from .base import EncoderInput

__all__ = ["CONVOLUTIONS", "ConvName", "GNNEncoder"]

ConvName = Literal["gcn", "sage", "gat", "gin", "mpnn"]


def _incoming_degree(dst: torch.Tensor, n_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    """Count the edges pointing at each node.

    Args:
        dst: Destination node of every directed edge, shape `(n_edges,)`.
        n_nodes: Number of nodes in the graph.
        dtype: Floating dtype for the returned counts.

    Returns:
        Per-node in-degree, shape `(n_nodes,)`.
    """
    degree = torch.zeros(n_nodes, dtype=dtype, device=dst.device)
    degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=dtype))
    return degree


def _scatter_sum(messages: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Sum every message into its destination node.

    Args:
        messages: Per-edge messages, shape `(batch, n_edges, dim)`.
        dst: Destination node of every edge, shape `(n_edges,)`.
        n_nodes: Number of nodes to scatter into.

    Returns:
        Per-node sums, shape `(batch, n_nodes, dim)`.
    """
    batch, _, dim = messages.shape
    out = messages.new_zeros(batch, n_nodes, dim)
    index = dst.view(1, -1, 1).expand(batch, -1, dim)
    out.scatter_add_(1, index, messages)
    return out


def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Softmax each score over the edges sharing its destination node.

    Args:
        scores: Per-edge, per-head attention scores, shape `(batch, n_edges, heads)`.
        dst: Destination node of every edge, shape `(n_edges,)`.
        n_nodes: Number of nodes.

    Returns:
        Normalised weights of the same shape as `scores`, each destination's edges summing to 1.
    """
    batch, _, heads = scores.shape
    index = dst.view(1, -1, 1).expand(batch, -1, heads)
    maxes = scores.new_full((batch, n_nodes, heads), float("-inf"))
    maxes.scatter_reduce_(1, index, scores, reduce="amax", include_self=False)
    maxes = torch.where(maxes.isinf(), torch.zeros_like(maxes), maxes)
    weights = (scores - maxes.gather(1, index)).exp()
    denom = weights.new_zeros(batch, n_nodes, heads)
    denom.scatter_add_(1, index, weights)
    return weights / denom.gather(1, index).clamp_min(1e-16)


class GCNConv(nn.Module):
    """Kipf-Welling graph convolution: symmetric-normalised mean of transformed neighbours."""

    def __init__(self, dim: int, edge_features: int) -> None:  # noqa: ARG002 - uniform ctor
        """Build the linear map applied before propagation.

        Args:
            dim: Width of the node state, unchanged by the layer.
            edge_features: Edge feature width; unused, GCN ignores edge attributes.
        """
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, batch: EncoderInput) -> torch.Tensor:
        """Propagate one GCN step.

        Args:
            h: Node states, shape `(batch, n_nodes, dim)`.
            batch: The encoder input, read for `edge_index`.

        Returns:
            Updated node states of the same shape as `h`.
        """
        src, dst = batch.edge_index[0], batch.edge_index[1]
        n_nodes = h.shape[1]
        transformed = self.linear(h)
        degree = _incoming_degree(dst, n_nodes, h.dtype) + 1.0
        norm = (degree[src] * degree[dst]).rsqrt()
        messages = transformed.index_select(1, src) * norm.view(1, -1, 1)
        aggregated = _scatter_sum(messages, dst, n_nodes)
        updated: torch.Tensor = aggregated + transformed / degree.view(1, n_nodes, 1)
        return updated


class SAGEConv(nn.Module):
    """GraphSAGE with mean aggregation: concatenate self and neighbour mean, then project."""

    def __init__(self, dim: int, edge_features: int) -> None:  # noqa: ARG002 - uniform ctor
        """Build the projection over the concatenated self/neighbour state.

        Args:
            dim: Width of the node state, unchanged by the layer.
            edge_features: Edge feature width; unused.
        """
        super().__init__()
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, h: torch.Tensor, batch: EncoderInput) -> torch.Tensor:
        """Propagate one GraphSAGE step.

        Args:
            h: Node states, shape `(batch, n_nodes, dim)`.
            batch: The encoder input, read for `edge_index`.

        Returns:
            Updated node states of the same shape as `h`.
        """
        src, dst = batch.edge_index[0], batch.edge_index[1]
        n_nodes = h.shape[1]
        neighbour_sum = _scatter_sum(h.index_select(1, src), dst, n_nodes)
        degree = _incoming_degree(dst, n_nodes, h.dtype).clamp_min(1.0)
        neighbour_mean = neighbour_sum / degree.view(1, n_nodes, 1)
        updated: torch.Tensor = self.linear(torch.cat([h, neighbour_mean], dim=-1))
        return updated


class GINConv(nn.Module):
    """Graph Isomorphism Network: sum neighbours, add a learnable share of self, then an MLP."""

    def __init__(self, dim: int, edge_features: int) -> None:  # noqa: ARG002 - uniform ctor
        """Build the update MLP and the self-weight epsilon.

        Args:
            dim: Width of the node state, unchanged by the layer.
            edge_features: Edge feature width; unused.
        """
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, batch: EncoderInput) -> torch.Tensor:
        """Propagate one GIN step.

        Args:
            h: Node states, shape `(batch, n_nodes, dim)`.
            batch: The encoder input, read for `edge_index`.

        Returns:
            Updated node states of the same shape as `h`.
        """
        src, dst = batch.edge_index[0], batch.edge_index[1]
        neighbour_sum = _scatter_sum(h.index_select(1, src), dst, h.shape[1])
        updated: torch.Tensor = self.mlp((1.0 + self.eps) * h + neighbour_sum)
        return updated


class GATConv(nn.Module):
    """Multi-head graph attention: neighbour weights from a learned score over endpoint pairs."""

    def __init__(self, dim: int, edge_features: int, heads: int = 4) -> None:  # noqa: ARG002
        """Build the per-head projection and the additive attention vectors.

        Args:
            dim: Width of the node state; must be divisible by `heads`.
            edge_features: Edge feature width; unused by this variant.
            heads: Number of attention heads; their outputs are concatenated back to `dim`.

        Raises:
            ValueError: `dim` is not divisible by `heads`.
        """
        super().__init__()
        if dim % heads:
            msg = f"embedding_dim {dim} must be divisible by the {heads} attention heads"
            raise ValueError(msg)
        self.heads = heads
        self.head_dim = dim // heads
        self.linear = nn.Linear(dim, dim)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, h: torch.Tensor, batch: EncoderInput) -> torch.Tensor:
        """Propagate one graph-attention step.

        Args:
            h: Node states, shape `(batch, n_nodes, dim)`.
            batch: The encoder input, read for `edge_index`.

        Returns:
            Updated node states of the same shape as `h`.
        """
        src, dst = batch.edge_index[0], batch.edge_index[1]
        size, n_nodes, _ = h.shape
        transformed = self.linear(h).view(size, n_nodes, self.heads, self.head_dim)
        src_score = (transformed * self.attn_src).sum(-1).index_select(1, src)
        dst_score = (transformed * self.attn_dst).sum(-1).index_select(1, dst)
        weights = _segment_softmax(self.leaky_relu(src_score + dst_score), dst, n_nodes)
        messages = transformed.index_select(1, src) * weights.unsqueeze(-1)
        index = dst.view(1, -1, 1, 1).expand(size, -1, self.heads, self.head_dim)
        out = messages.new_zeros(size, n_nodes, self.heads, self.head_dim)
        out.scatter_add_(1, index, messages)
        updated: torch.Tensor = out.reshape(size, n_nodes, self.heads * self.head_dim)
        return updated


class MPNNConv(nn.Module):
    """Edge-conditioned message passing: the only variant that reads `edge_attr`."""

    def __init__(self, dim: int, edge_features: int) -> None:
        """Build the message and update MLPs.

        Args:
            dim: Width of the node state, unchanged by the layer.
            edge_features: Width of the per-edge feature vector fed into every message.
        """
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_features, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.update_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, batch: EncoderInput) -> torch.Tensor:
        """Propagate one edge-conditioned message-passing step.

        Args:
            h: Node states, shape `(batch, n_nodes, dim)`.
            batch: The encoder input, read for `edge_index` and `edge_attr`.

        Returns:
            Updated node states of the same shape as `h`.
        """
        src, dst = batch.edge_index[0], batch.edge_index[1]
        n_nodes = h.shape[1]
        edge_attr = batch.edge_attr.unsqueeze(0).expand(h.shape[0], -1, -1)
        pairs = torch.cat([h.index_select(1, src), h.index_select(1, dst), edge_attr], dim=-1)
        messages = _scatter_sum(self.message_mlp(pairs), dst, n_nodes)
        degree = _incoming_degree(dst, n_nodes, h.dtype).clamp_min(1.0)
        updated: torch.Tensor = self.update_mlp(
            torch.cat([h, messages / degree.view(1, n_nodes, 1)], dim=-1)
        )
        return updated


#: Every graph-convolution layer, keyed by the name it is registered under in `ENCODERS`.
CONVOLUTIONS: dict[ConvName, Callable[[int, int], nn.Module]] = {
    "gcn": GCNConv,
    "sage": SAGEConv,
    "gat": GATConv,
    "gin": GINConv,
    "mpnn": MPNNConv,
}


class GNNEncoder(nn.Module):
    """Input projection, a stack of graph-convolution layers, then a pooled graph head.

    Each layer is a residual block: `h <- LayerNorm(h + GELU(conv(h)))`. The graph head is the
    same mean/max pool over the global features that `FlatEncoder` uses, so the only difference
    between the two encoders is that this one propagates over `edge_index`.
    """

    def __init__(  # noqa: PLR0913 - feature widths plus two architecture knobs
        self,
        node_features: int,
        global_features: int,
        edge_features: int,
        embedding_dim: int = 64,
        *,
        conv: ConvName = "mpnn",
        num_layers: int = 2,
    ) -> None:
        """Build the projection, the convolution stack and the graph MLP.

        Args:
            node_features: Width of the per-course node feature vector.
            global_features: Width of the per-instance global feature vector.
            edge_features: Width of the per-edge feature vector.
            embedding_dim: Width of every embedding this encoder produces.
            conv: Which graph-convolution layer to stack; a key of `CONVOLUTIONS`.
            num_layers: How many convolution layers to stack.

        Raises:
            KeyError: `conv` names no registered convolution.
        """
        super().__init__()
        if conv not in CONVOLUTIONS:
            msg = f"unknown gnn conv {conv!r}; registered: {sorted(CONVOLUTIONS)}"
            raise KeyError(msg)
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(node_features, embedding_dim)
        self.convs = nn.ModuleList(
            CONVOLUTIONS[conv](embedding_dim, edge_features) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(num_layers))
        self.graph_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim + global_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, batch: EncoderInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of observations of one instance.

        Args:
            batch: The node, global and conflict-graph features to encode.

        Returns:
            Node embeddings of shape `(batch, n_courses, dim)` and a graph embedding of shape
            `(batch, dim)`.
        """
        h = self.input_proj(batch.nodes)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + torch.nn.functional.gelu(conv(h, batch)))
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1), batch.globals.squeeze(1)], dim=-1)
        return h, self.graph_mlp(pooled)
