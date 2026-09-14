"""The inference half of the pipeline: checkpoint, reload, roll out, write, validate."""

from pathlib import Path

import pytest
import torch

from horarium.agents.checkpoint import CheckpointError, load_agent, save_agent
from horarium.agents.ppo import PPOConfig, resolve_device, train
from horarium.agents.rollout import rollout, solve
from horarium.data.solution_io import read_solution, write_solution
from horarium.envs.construct import ConstructEnv
from horarium.eval.validate import compare, default_validator_path, run_validator
from horarium.models.encoders import ENCODERS, EncoderSpec
from horarium.problem.cost import cost
from horarium.problem.formulations import FORMULATIONS
from horarium.problem.instance import Instance

SPEC = EncoderSpec(name="flat", embedding_dim=32)
needs_validator = pytest.mark.skipif(
    not default_validator_path().exists(), reason="run scripts/setup.sh to compile it"
)


@pytest.fixture
def trained(toy: Instance, tmp_path: Path) -> tuple[Path, ConstructEnv]:
    """A genuinely trained toy policy, checkpointed to disk."""
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    report = train(
        env,
        lambda: SPEC.build(
            node_features=env.node_feature_dim,
            global_features=env.global_feature_dim,
            edge_features=env.edge_feature_dim,
        ),
        PPOConfig(total_steps=1536, rollout_steps=256, seed=0, device="cpu", minibatches=2),
    )
    assert report.agent is not None
    path = tmp_path / "policy.pt"
    save_agent(path, report.agent, env, SPEC, seed=0)
    return path, env


def test_checkpoint_round_trips(trained: tuple[Path, ConstructEnv]) -> None:
    path, env = trained
    agent, metadata = load_agent(path, env)
    assert metadata.encoder == "flat"
    assert metadata.instance_name == env.instance.name
    assert metadata.n_actions == env.n_actions
    assert not agent.training


def test_reloaded_policy_is_identical(trained: tuple[Path, ConstructEnv]) -> None:
    """A greedy rollout is deterministic, so reloading must reproduce the same timetable."""
    path, env = trained
    first, _ = load_agent(path, env)
    second, _ = load_agent(path, env)
    assert rollout(env, first, greedy=True, seed=0).solution.lectures() == (
        rollout(env, second, greedy=True, seed=0).solution.lectures()
    )


def test_checkpoint_from_another_instance_is_refused(
    trained: tuple[Path, ConstructEnv], comp01: Instance
) -> None:
    """Loading a mismatched policy would produce silently meaningless actions."""
    path, _ = trained
    other = ConstructEnv(comp01, FORMULATIONS["UD2"])
    with pytest.raises(CheckpointError, match="does not fit"):
        load_agent(path, other)


def test_missing_checkpoint_raises(toy: Instance, tmp_path: Path) -> None:
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_agent(tmp_path / "absent.pt", env)


def test_corrupt_checkpoint_raises(toy: Instance, tmp_path: Path) -> None:
    path = tmp_path / "junk.pt"
    torch.save({"something": "else"}, path)
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    with pytest.raises(CheckpointError, match="not a horarium checkpoint"):
        load_agent(path, env)


def test_solve_produces_a_feasible_timetable(trained: tuple[Path, ConstructEnv]) -> None:
    path, env = trained
    agent, _ = load_agent(path, env)
    result = solve(env, agent, restarts=8, seed=0)
    assert result.best is not None
    assert result.best.stats.feasible
    assert result.feasibility_rate > 0


@needs_validator
def test_written_solution_agrees_with_the_validator(
    trained: tuple[Path, ConstructEnv], tmp_path: Path
) -> None:
    """The end of the pipeline: what inference writes must survive the reference validator."""
    path, env = trained
    agent, _ = load_agent(path, env)
    result = solve(env, agent, restarts=8, seed=0)
    assert result.best is not None

    written = tmp_path / "inference.sol"
    write_solution(env.instance, result.best.solution, written)
    breakdown = cost(env.instance, result.best.solution, env.formulation)
    report = run_validator("UD2", default_toy_path(), written)
    assert compare(breakdown, report) == []
    assert report.violations == 0
    assert report.warnings == 0

    reread = read_solution(env.instance, written)
    assert reread.lectures() == result.best.solution.lectures()


def default_toy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "raw" / "Test" / "toy.ectt"


def test_every_registered_encoder_builds_and_runs(toy: Instance) -> None:
    """The extension point: adding an encoder must need nothing but a line in ENCODERS."""
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    for name in ENCODERS:
        report = train(
            env,
            lambda name=name: EncoderSpec(name=name, embedding_dim=16).build(  # type: ignore[misc]
                node_features=env.node_feature_dim,
                global_features=env.global_feature_dim,
                edge_features=env.edge_feature_dim,
            ),
            PPOConfig(total_steps=256, rollout_steps=128, seed=0, device="cpu", minibatches=2),
        )
        assert report.agent is not None
        assert rollout(env, report.agent, greedy=True, seed=0).stats.steps > 0


def test_requesting_an_unusable_device_fails_loudly() -> None:
    """Silently running on the CPU after asking for CUDA is the failure this guards against."""
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cannot use CUDA"):
            resolve_device("cuda")
    else:
        assert resolve_device("cuda").type == "cuda"
