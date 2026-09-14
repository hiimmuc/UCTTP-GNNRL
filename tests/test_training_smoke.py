"""A short PPO run must go end to end and produce a feasible timetable on the toy instance."""

from horarium.agents.ppo import PPOConfig, TrainingReport, train
from horarium.envs.construct import ConstructEnv
from horarium.models.encoders.flat import FlatEncoder
from horarium.problem.formulations import FORMULATIONS
from horarium.problem.instance import Instance


def _run(toy: Instance, seed: int, steps: int = 1536) -> TrainingReport:
    env = ConstructEnv(toy, FORMULATIONS["UD2"])

    def build_encoder() -> FlatEncoder:
        return FlatEncoder(
            node_features=env.node_feature_dim,
            global_features=env.global_feature_dim,
            edge_features=env.edge_feature_dim,
            embedding_dim=32,
        )

    config = PPOConfig(total_steps=steps, rollout_steps=256, seed=seed, device="cpu", minibatches=2)
    return train(env, build_encoder, config)


def test_ppo_reaches_feasibility_on_toy(toy: Instance) -> None:
    report = _run(toy, seed=0)
    assert report.episodes > 0
    assert report.recent_feasibility_rate == 1.0
    assert report.best_stats is not None
    assert report.best_stats.feasible


def test_training_is_deterministic_given_a_seed(toy: Instance) -> None:
    """Same seed, same config, same result -- the reproducibility the project requires."""
    first = _run(toy, seed=3, steps=768)
    second = _run(toy, seed=3, steps=768)
    assert first.best_cost == second.best_cost
    assert first.episodes == second.episodes
    assert first.feasible_episodes == second.feasible_episodes


def test_different_seeds_give_different_runs(toy: Instance) -> None:
    """Guards against a seed that is accepted and then quietly ignored."""
    assert _run(toy, seed=1, steps=768).history != _run(toy, seed=2, steps=768).history
