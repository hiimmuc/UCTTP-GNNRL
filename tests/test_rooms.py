"""The stage-2 room oracle: optimal per period, never worse than the rule it replaces."""

from __future__ import annotations

import itertools
import random

import pytest

from horarium.envs.construct import ConstructEnv
from horarium.problem.cost import cost
from horarium.problem.formulations import FORMULATIONS, UD1, UD2, Formulation, Hard, Soft
from horarium.problem.instance import Instance
from horarium.problem.solution import EMPTY, Solution
from horarium.solvers.rooms import RoomAssignmentError, assign_rooms, reassign_rooms

FORMULATION_NAMES = sorted(FORMULATIONS)


def _random_feasible_periods(instance: Instance, seed: int) -> Solution:
    """Build a solution by taking random legal actions, which is period-feasible by construction.

    Args:
        instance: The instance to build for.
        seed: Seed for the action choice.

    Returns:
        The solution the random rollout reached, complete or not.
    """
    rng = random.Random(seed)  # noqa: S311 - reproducible test fixtures, not cryptography
    env = ConstructEnv(instance, UD2)
    env.reset(seed=seed)
    while True:
        legal = [i for i, ok in enumerate(env.action_mask()) if ok]
        if not legal:
            return env.solution
        _obs, _reward, terminated, _truncated, _info = env.step(rng.choice(legal))
        if terminated:
            return env.solution


def _periods_of(solution: Solution) -> list[list[int]]:
    """Extract the period assignment, discarding the rooms.

    Args:
        solution: The solution to read.

    Returns:
        For each course, the periods holding one of its lectures.
    """
    return [[p for p, room in enumerate(row) if room != EMPTY] for row in solution.room_at]


def _brute_force(instance: Instance, courses: list[int], formulation: Formulation) -> int:
    """Cheapest assignment of one period's courses to distinct rooms, by exhaustive search.

    Args:
        instance: The instance being solved.
        courses: The courses sharing a period.
        formulation: Which components are active and at what weight.

    Returns:
        The lowest achievable weighted cost for that period.
    """
    weight = formulation.weights.get(Soft.ROOM_CAPACITY, 0)
    best = None
    for rooms in itertools.permutations(range(instance.n_rooms), len(courses)):
        total = sum(
            weight * max(0, instance.courses[c].n_students - instance.rooms[r].capacity)
            for c, r in zip(courses, rooms, strict=True)
        )
        if best is None or total < best:
            best = total
    assert best is not None
    return best


@pytest.mark.parametrize("seed", range(3))
def test_a_period_is_solved_exactly_when_stability_is_inactive(comp01: Instance, seed: int) -> None:
    """Without RoomStability the periods are independent, so each must hit the brute-force best."""
    solution = _random_feasible_periods(comp01, seed)
    assigned = assign_rooms(comp01, _periods_of(solution), UD1)
    weight = UD1.weights[Soft.ROOM_CAPACITY]
    for period in range(comp01.n_periods):
        courses = sorted(assigned.courses_at_period[period])
        if not 0 < len(courses) <= 4:  # keep the permutation search tractable
            continue
        got = sum(
            weight
            * max(
                0, comp01.courses[c].n_students - comp01.rooms[assigned.room_at[c][period]].capacity
            )
            for c in courses
        )
        assert got == _brute_force(comp01, courses, UD1)


@pytest.mark.parametrize("formulation_name", FORMULATION_NAMES)
def test_the_oracle_never_loses_to_the_placeholder_rule(
    comp01: Instance, formulation_name: str
) -> None:
    """The rule it replaces is one feasible assignment, so the oracle must match or beat it."""
    formulation = FORMULATIONS[formulation_name]
    solution = _random_feasible_periods(comp01, seed=0)
    before = cost(comp01, solution, formulation)
    after = cost(comp01, reassign_rooms(solution, formulation), formulation)
    assert after.total <= before.total


@pytest.mark.parametrize("seed", range(4))
def test_reassigning_is_an_improvement_operator_on_every_instance(
    sample: Instance, seed: int
) -> None:
    """Warm-started descent is monotone, so `reassign_rooms` must never return something worse.

    A cold start does not have this property: solving capacity first can give back more in
    RoomStability than the placeholder rule's "reuse a room you already have" tie-break won.
    """
    solution = _random_feasible_periods(sample, seed)
    before = cost(sample, solution, UD2).total
    assert cost(sample, reassign_rooms(solution, UD2), UD2).total <= before


def test_the_oracle_keeps_every_lecture_in_its_period(sample: Instance) -> None:
    """Stage 2 may move a lecture between rooms, never between periods."""
    solution = _random_feasible_periods(sample, seed=0)
    assigned = reassign_rooms(solution, UD2)
    assert _periods_of(assigned) == _periods_of(solution)
    assert assigned.n_scheduled == solution.n_scheduled


def test_the_oracle_never_double_books_a_room(sample: Instance) -> None:
    """RoomOccupation is what makes each period an assignment rather than a free choice."""
    assigned = reassign_rooms(_random_feasible_periods(sample, seed=0), UD2)
    assert cost(sample, assigned, UD2).hard[Hard.ROOM_OCCUPATION] == 0
    for room_loads in assigned.room_period_load:
        assert max(room_loads, default=0) <= 1


def test_sweeping_converges_so_more_sweeps_change_nothing(comp01: Instance) -> None:
    """Block-coordinate descent reaches a fixed point; running longer must not move it."""
    periods = _periods_of(_random_feasible_periods(comp01, seed=1))
    settled = cost(comp01, assign_rooms(comp01, periods, UD2, sweeps=64), UD2).total
    assert cost(comp01, assign_rooms(comp01, periods, UD2, sweeps=128), UD2).total == settled


def test_more_sweeps_never_cost_more(comp01: Instance) -> None:
    """Each sweep re-solves periods exactly, so cost is monotone non-increasing in `sweeps`."""
    periods = _periods_of(_random_feasible_periods(comp01, seed=2))
    totals = [
        cost(comp01, assign_rooms(comp01, periods, UD2, sweeps=n), UD2).total for n in range(5)
    ]
    assert totals == sorted(totals, reverse=True)


def test_reassigning_an_oracle_solution_is_idempotent(comp01: Instance) -> None:
    """A fixed point of the sweep stays fixed when fed back in."""
    once = reassign_rooms(_random_feasible_periods(comp01, seed=3), UD2)
    assert cost(comp01, reassign_rooms(once, UD2), UD2).total == cost(comp01, once, UD2).total


def test_a_period_with_more_lectures_than_rooms_is_rejected(comp01: Instance) -> None:
    """The caller gave an impossible period assignment and must be told, not handed a bad table."""
    crammed = [[0] if course <= comp01.n_rooms else [] for course in range(comp01.n_courses)]
    with pytest.raises(RoomAssignmentError, match="rooms"):
        assign_rooms(comp01, crammed, UD2)


def test_under_ud4_forbidden_rooms_are_refused_rather_than_charged(comp01: Instance) -> None:
    """UD4 makes room constraints hard, so the oracle must never spend a forbidden pairing."""
    assigned = reassign_rooms(_random_feasible_periods(comp01, seed=0), FORMULATIONS["UD4"])
    for course, row in enumerate(assigned.room_at):
        for room in row:
            if room != EMPTY:
                assert room not in comp01.forbidden_rooms[course]
