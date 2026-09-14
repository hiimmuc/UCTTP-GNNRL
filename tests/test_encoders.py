"""The GNN encoders: they must use the conflict structure, train, and run on the GPU."""

from __future__ import annotations

import pytest
import torch

from horarium.agents.ppo import PPOConfig, resolve_device, train
from horarium.agents.rollout import rollout
from horarium.envs.construct import ConstructEnv
from horarium.models.encoders import ENCODERS, EncoderSpec
from horarium.models.encoders.base import EncoderInput
from horarium.models.encoders.gnn import CONVOLUTIONS, GNNEncoder
from horarium.problem.formulations import FORMULATIONS
from horarium.problem.instance import Instance

GNN_NAMES = sorted(CONVOLUTIONS)


def _batch(env: ConstructEnv, *, drop_edges: bool = False) -> EncoderInput:
    observation, _ = env.reset(seed=0)
    edge_index = torch.as_tensor(env.features.edge_index)
    edge_attr = torch.as_tensor(env.features.edges.values)
    if drop_edges:
        edge_index = edge_index[:, :0]
        edge_attr = edge_attr[:0]
    return EncoderInput(
        nodes=torch.as_tensor(observation["nodes"]).unsqueeze(0),
        globals=torch.as_tensor(observation["globals"]).unsqueeze(0),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


@pytest.mark.parametrize("conv", GNN_NAMES)
def test_gnn_output_depends_on_the_conflict_structure(toy: Instance, conv: str) -> None:
    """Every GNN variant must actually propagate: removing the edges must change its output."""
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    encoder = GNNEncoder(
        node_features=env.node_feature_dim,
        global_features=env.global_feature_dim,
        edge_features=env.edge_feature_dim,
        embedding_dim=32,
        conv=conv,  # type: ignore[arg-type]
    )
    encoder.eval()
    with torch.no_grad():
        nodes_with, _ = encoder(_batch(env))
        nodes_without, _ = encoder(_batch(env, drop_edges=True))
    assert not torch.allclose(nodes_with, nodes_without)


@pytest.mark.parametrize("conv", GNN_NAMES)
def test_gradients_reach_every_gnn_parameter(toy: Instance, conv: str) -> None:
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    encoder = GNNEncoder(
        node_features=env.node_feature_dim,
        global_features=env.global_feature_dim,
        edge_features=env.edge_feature_dim,
        embedding_dim=32,
        conv=conv,  # type: ignore[arg-type]
    )
    nodes, graph = encoder(_batch(env))
    (nodes.pow(2).mean() + graph.pow(2).mean()).backward()
    missing = [name for name, p in encoder.named_parameters() if p.grad is None]
    assert missing == []


def test_flat_encoder_ignores_the_conflict_structure(toy: Instance) -> None:
    """The control arm: the same probe must leave the flat encoder's output unchanged."""
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    encoder = EncoderSpec(name="flat", embedding_dim=32).build(
        node_features=env.node_feature_dim,
        global_features=env.global_feature_dim,
        edge_features=env.edge_feature_dim,
    )
    assert isinstance(encoder, torch.nn.Module)
    encoder.eval()
    with torch.no_grad():
        nodes_with, _ = encoder(_batch(env))
        nodes_without, _ = encoder(_batch(env, drop_edges=True))
    assert torch.allclose(nodes_with, nodes_without)


def test_every_gnn_name_is_registered() -> None:
    for name in GNN_NAMES:
        assert name in ENCODERS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gnn_trains_on_the_gpu(toy: Instance) -> None:
    """A short PPO run on CUDA must complete and produce a usable policy."""
    assert resolve_device("cuda").type == "cuda"
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    report = train(
        env,
        lambda: EncoderSpec(name="mpnn", embedding_dim=32).build(
            node_features=env.node_feature_dim,
            global_features=env.global_feature_dim,
            edge_features=env.edge_feature_dim,
        ),
        PPOConfig(total_steps=512, rollout_steps=256, seed=0, device="cuda", minibatches=2),
    )
    assert report.agent is not None
    assert next(report.agent.parameters()).is_cuda
    assert rollout(env, report.agent, greedy=True, seed=0).stats.steps > 0
