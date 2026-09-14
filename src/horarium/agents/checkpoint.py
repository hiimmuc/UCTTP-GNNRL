"""Saving and restoring a trained agent, with the shapes needed to rebuild it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from horarium.envs.base import TimetablingEnv
from horarium.models.encoders import EncoderSpec

__all__ = ["Checkpoint", "CheckpointError", "load_agent", "save_agent"]

FORMAT_VERSION = 2


class CheckpointError(RuntimeError):
    """A checkpoint is unreadable, or does not fit the environment it is being loaded against."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything needed to rebuild an agent, plus what it was trained on."""

    format_version: int
    encoder: str
    embedding_dim: int
    node_features: int
    global_features: int
    edge_features: int
    period_features: int
    n_actions: int
    instance_name: str
    formulation: str
    seed: int


def save_agent(
    path: str | Path, agent: torch.nn.Module, env: TimetablingEnv, spec: EncoderSpec, seed: int
) -> Checkpoint:
    """Write the agent's weights and the shapes required to reconstruct it.

    Args:
        path: Destination path.
        agent: The trained network to save.
        env: The environment it was trained on, used only for its shapes.
        spec: Which encoder `agent` uses and how wide.
        seed: The training seed, recorded for provenance.

    Returns:
        The metadata written alongside the weights.
    """
    metadata = Checkpoint(
        format_version=FORMAT_VERSION,
        encoder=spec.name,
        embedding_dim=spec.embedding_dim,
        node_features=env.node_feature_dim,
        global_features=env.global_feature_dim,
        edge_features=env.edge_feature_dim,
        period_features=env.period_feature_dim,
        n_actions=env.n_actions,
        instance_name=env.instance.name,
        formulation=env.formulation.name,
        seed=seed,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": asdict(metadata), "state_dict": agent.state_dict()}, destination)
    return metadata


def load_agent(path: str | Path, env: TimetablingEnv) -> tuple[torch.nn.Module, Checkpoint]:
    """Rebuild the agent from a checkpoint and check it fits this environment.

    The shapes are compared rather than trusted: loading a policy trained on another instance
    would otherwise produce silently meaningless actions.

    Args:
        path: Path to the checkpoint file.
        env: The environment to load the agent into; its shapes are checked against the
            checkpoint's.

    Returns:
        The rebuilt agent, in eval mode, and its checkpoint metadata.

    Raises:
        CheckpointError: The file is not a checkpoint, its format is unknown, or its shapes
            disagree with the environment.
    """
    from horarium.agents.ppo import ActorCritic  # noqa: PLC0415 - avoids an import cycle

    source = Path(path)
    if not source.exists():
        msg = f"no checkpoint at {source}"
        raise CheckpointError(msg)
    payload: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=True)
    if "metadata" not in payload or "state_dict" not in payload:
        msg = f"{source} is not a horarium checkpoint"
        raise CheckpointError(msg)

    metadata = Checkpoint(**payload["metadata"])
    if metadata.format_version != FORMAT_VERSION:
        msg = f"{source} has format version {metadata.format_version}, expected {FORMAT_VERSION}"
        raise CheckpointError(msg)

    mismatched = [
        f"{label}: checkpoint {saved} vs environment {live}"
        for label, saved, live in (
            ("node features", metadata.node_features, env.node_feature_dim),
            ("global features", metadata.global_features, env.global_feature_dim),
            ("edge features", metadata.edge_features, env.edge_feature_dim),
            ("period features", metadata.period_features, env.period_feature_dim),
            ("actions", metadata.n_actions, env.n_actions),
        )
        if saved != live
    ]
    if mismatched:
        listed = "; ".join(mismatched)
        msg = (
            f"{source} was trained on {metadata.instance_name!r} and does not fit "
            f"{env.instance.name!r}: {listed}"
        )
        raise CheckpointError(msg)

    spec = EncoderSpec(name=metadata.encoder, embedding_dim=metadata.embedding_dim)
    agent = ActorCritic(
        spec.build(
            node_features=metadata.node_features,
            global_features=metadata.global_features,
            edge_features=metadata.edge_features,
        ),
        metadata.period_features,
    )
    agent.load_state_dict(payload["state_dict"])
    agent.eval()
    return agent, metadata
