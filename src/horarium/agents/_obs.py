"""Turning a Gymnasium observation into the tensors the actor-critic consumes.

Training and inference both walk an episode the same way -- reset, read the observation, encode
it, act -- so the observation-to-tensor step lives here once rather than in `ppo` and `rollout`
separately.
"""

from __future__ import annotations

import numpy as np
import torch

from horarium.envs.base import TimetablingEnv
from horarium.models.encoders import EncoderInput

__all__ = ["edge_tensors", "encoder_input", "period_batch"]


def edge_tensors(env: TimetablingEnv, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The instance's conflict-graph edge index and edge features, on `device`.

    Both are fixed for an instance, so they are built once per episode and shared across steps.

    Args:
        env: The environment whose instance's conflict graph is wanted.
        device: Device the tensors are created on.

    Returns:
        The `(2, n_edges)` edge index and the `(n_edges, edge_features)` edge features.
    """
    edge_index = torch.as_tensor(env.features.edge_index, device=device)
    edge_attr = torch.as_tensor(env.features.edges.values, device=device)
    return edge_index, edge_attr


def encoder_input(
    observation: dict[str, object],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> EncoderInput:
    """Wrap one observation and the shared edge tensors as a batch of size one.

    Args:
        observation: An environment observation with `"nodes"` and `"globals"` keys.
        edge_index: The conflict-graph edge index, already on `device`.
        edge_attr: The conflict-graph edge features, already on `device`.
        device: Device the node and global tensors are created on.

    Returns:
        A batch-of-one `EncoderInput`.
    """
    return EncoderInput(
        nodes=torch.as_tensor(np.asarray(observation["nodes"]), device=device).unsqueeze(0),
        globals=torch.as_tensor(np.asarray(observation["globals"]), device=device).unsqueeze(0),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def period_batch(observation: dict[str, object], device: torch.device) -> torch.Tensor:
    """Period features as a `(1, n_periods, period_features)` batch.

    Args:
        observation: An environment observation with a `"periods"` key.
        device: Device the tensor is created on.

    Returns:
        The period features, batched to size one.
    """
    return torch.as_tensor(np.asarray(observation["periods"]), device=device).unsqueeze(0)
