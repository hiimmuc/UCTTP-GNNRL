"""Masked actions must be provably feasible: no rollout may ever break a hard constraint."""

import random

import numpy as np
import numpy.typing as npt
import pytest

from horarium.envs.construct import ConstructEnv
from horarium.problem.cost import cost
from horarium.problem.formulations import FORMULATIONS, Hard
from horarium.problem.instance import Instance
from horarium.problem.solution import EMPTY, Solution

EPISODES = 8


def _rollout(env: ConstructEnv, rng: random.Random) -> dict[str, object]:
    observation, _ = env.reset(seed=rng.randrange(10_000))
    info: dict[str, object] = {}
    while True:
        legal = np.flatnonzero(observation["action_mask"])
        if legal.size == 0:
            break
        observation, _reward, terminated, _truncated, info = env.step(int(rng.choice(legal)))
        if terminated:
            break
    return info


@pytest.mark.parametrize("formulation", sorted(FORMULATIONS))
def test_masked_rollouts_never_violate_a_hard_constraint(
    comp01: Instance, formulation: str
) -> None:
    """The only hard cost a masked rollout may incur is unplaced lectures at a dead end."""
    rule = FORMULATIONS[formulation]
    env = ConstructEnv(comp01, rule)
    rng = random.Random(f"mask/{formulation}")  # noqa: S311
    for _ in range(EPISODES):
        _rollout(env, rng)
        breakdown = cost(comp01, env.solution, rule)
        for constraint, violations in breakdown.hard.items():
            if constraint is not Hard.LECTURES:
                assert violations == 0, f"{constraint} violated under masking"


def _reference_mask(env: ConstructEnv) -> npt.NDArray[np.bool_]:
    """The mask recomputed from the solution, straight from the definition of each constraint.

    `ConstructEnv` maintains the real mask incrementally; this is the slow, obviously-correct
    version it has to keep agreeing with.

    Args:
        env: The environment to read the current solution from.

    Returns:
        A boolean array of shape `(n_courses, n_periods)`.
    """
    instance, solution = env.instance, env.solution
    mask = np.zeros((instance.n_courses, instance.n_periods), dtype=np.bool_)
    for c, course in enumerate(instance.courses):
        if solution.n_scheduled[c] >= course.n_lectures:
            continue
        for p in instance.available_periods[c]:
            legal = (
                solution.room_at[c][p] == EMPTY
                and not (instance.conflicts[c] & solution.courses_at_period[p])
                and env._free_room(c, p) is not None
            )
            mask[c, p] = legal
    return mask


@pytest.mark.parametrize("fixture", ["toy", "comp01"])
def test_the_incremental_mask_matches_a_from_scratch_recomputation(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    """Every step of a full episode, the maintained counters must agree with the definition."""
    instance: Instance = request.getfixturevalue(fixture)
    env = ConstructEnv(instance, FORMULATIONS["UD2"])
    observation, _ = env.reset(seed=0)
    rng = random.Random(fixture)  # noqa: S311
    for step in range(400):
        mask = observation["action_mask"].reshape(instance.n_courses, instance.n_periods)
        np.testing.assert_array_equal(mask, _reference_mask(env), err_msg=f"step {step}")
        legal = np.flatnonzero(observation["action_mask"])
        if legal.size == 0:
            break
        observation, _r, terminated, _t, _i = env.step(int(rng.choice(legal)))
        if terminated:
            break


def test_taking_a_masked_action_raises(toy: Instance) -> None:
    env = ConstructEnv(toy, FORMULATIONS["UD2"])
    observation, _ = env.reset(seed=0)
    illegal = int(np.flatnonzero(~observation["action_mask"].astype(bool))[0])
    with pytest.raises(ValueError, match="is not legal"):
        env.step(illegal)


def test_reward_sums_to_the_negative_final_cost(toy: Instance) -> None:
    """Steps are charged the exact soft-cost increase, so the return is minus the cost.

    Offset by the cost of the empty timetable, which is not zero: an unscheduled course already
    owes its whole MinWorkingDays cost. The offset is constant, so it cannot reorder solutions.
    """
    rule = FORMULATIONS["UD2"]
    empty = cost(toy, Solution(toy), rule).total
    env = ConstructEnv(toy, rule)
    observation, _ = env.reset(seed=1)
    rng = random.Random(2)  # noqa: S311
    earned = 0.0
    while True:
        legal = np.flatnonzero(observation["action_mask"])
        if legal.size == 0:
            break
        observation, reward, terminated, _t, info = env.step(int(rng.choice(legal)))
        earned += reward
        if terminated:
            stats = info["episode_stats"]
            if not stats.dead_end:
                expected = -(stats.total_cost - empty) * env.reward_scale
                assert earned == pytest.approx(expected)
            break
