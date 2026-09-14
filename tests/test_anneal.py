"""Simulated annealing: must stay hard-feasible throughout and never hand back something worse."""

from __future__ import annotations

import pytest

from horarium.problem.cost import cost
from horarium.problem.formulations import FORMULATIONS, UD2, UD4
from horarium.problem.instance import Instance
from horarium.problem.solution import Solution
from horarium.solvers.anneal import AnnealConfig, anneal
from horarium.solvers.construction import dsatur, random_order
from horarium.solvers.rooms import assign_rooms

FORMULATION_NAMES = sorted(FORMULATIONS)
#: Enough iterations to exercise both moves and a couple of room re-solves, cheap on every
#: sample instance, and small enough that the suite stays fast.
_QUICK = AnnealConfig(iterations=300, reassign_every=50)


def _feasible_complete(instance: Instance, seed: int = 0) -> Solution:
    """A complete, hard-feasible timetable to anneal, built the same way a real caller would.

    Args:
        instance: The instance to build a timetable for.
        seed: Seed for the constructor.

    Returns:
        A complete solution under UD2.
    """
    construction = dsatur(instance, UD2, seed=seed, restarts=8)
    if not construction.complete:  # a few sample instances need the fallback constructor
        construction = random_order(instance, UD2, seed=seed, restarts=8)
    if not construction.complete:
        # comp05 resists every current constructor (see test_construction.py's coverage notes);
        # annealing needs a complete start, so there is nothing this suite can exercise on it.
        pytest.skip(f"no constructor reaches a complete timetable on {instance.name}")
    return assign_rooms(instance, construction.periods_of_course, UD2)


def test_annealing_never_returns_something_worse(sample: Instance) -> None:
    """The whole point of tracking `best` separately from `current` is this guarantee."""
    start = _feasible_complete(sample)
    before = cost(sample, start, UD2).total
    result = anneal(start, UD2, _QUICK)
    assert result.best_cost <= before


def test_the_result_stays_hard_feasible(sample: Instance) -> None:
    """Every move is legality-checked before being proposed, so this should never trip."""
    start = _feasible_complete(sample)
    result = anneal(start, UD2, _QUICK)
    assert cost(sample, result.solution, UD2).violations == 0


def test_the_result_is_still_complete(sample: Instance) -> None:
    """Annealing only relocates existing lectures; it must never drop or duplicate one."""
    start = _feasible_complete(sample)
    result = anneal(start, UD2, _QUICK)
    assert result.solution.is_complete
    assert sum(result.solution.n_scheduled) == sum(start.n_scheduled)


def test_the_reported_cost_matches_the_returned_solution(sample: Instance) -> None:
    """`best_cost` is tracked incrementally; it must agree with a full recomputation."""
    start = _feasible_complete(sample)
    result = anneal(start, UD2, _QUICK)
    assert result.best_cost == cost(sample, result.solution, UD2).total


def test_the_caller_s_solution_is_not_mutated(sample: Instance) -> None:
    """`anneal` copies its input, so the caller's timetable must be unaffected."""
    start = _feasible_complete(sample)
    before = start.copy()
    anneal(start, UD2, _QUICK)
    assert start.room_at == before.room_at


def test_the_same_seed_gives_the_same_result(comp01: Instance) -> None:
    """Reproducibility: every source of randomness in the search is seeded."""
    start = _feasible_complete(comp01)
    config = AnnealConfig(seed=3, iterations=500, reassign_every=100)
    first = anneal(start, UD2, config)
    second = anneal(start, UD2, config)
    assert first.solution.room_at == second.solution.room_at
    assert first.best_cost == second.best_cost


def test_more_iterations_never_cost_more(comp01: Instance) -> None:
    """`best` only ever improves, so cost is monotone non-increasing in the iteration budget."""
    start = _feasible_complete(comp01)
    costs = [
        anneal(start, UD2, AnnealConfig(seed=0, iterations=n, reassign_every=50)).best_cost
        for n in (0, 100, 300, 600)
    ]
    assert costs == sorted(costs, reverse=True)


def test_annealing_a_deliberately_bad_solution_improves_it(comp01: Instance) -> None:
    """The clearest end-to-end signal: a poor start should end up cheaper than it began."""
    poor = _feasible_complete(comp01, seed=0)
    result = anneal(poor, UD2, AnnealConfig(seed=0, iterations=4000, reassign_every=200))
    assert result.best_cost < result.initial_cost


def test_a_time_limit_of_zero_still_returns_a_valid_result(sample: Instance) -> None:
    """`time_limit=0` should behave like `iterations=0`: no moves, but a well-formed result."""
    start = _feasible_complete(sample)
    result = anneal(start, UD2, AnnealConfig(iterations=1, time_limit=0.0))
    assert result.iterations == 0
    assert result.accepted == 0
    assert result.best_cost == result.initial_cost


def test_an_incomplete_solution_is_rejected(comp01: Instance) -> None:
    """Starting from a partial timetable is a caller error, not something to accept silently."""
    empty = Solution(comp01)
    with pytest.raises(ValueError, match="complete"):
        anneal(empty, UD2, _QUICK)


def test_under_ud4_a_forbidden_room_is_never_introduced(comp01: Instance) -> None:
    """UD4 makes RoomConstraints hard; a relocated lecture must not land in a banned room."""
    construction = dsatur(comp01, UD4, seed=0, restarts=8)
    start = assign_rooms(comp01, construction.periods_of_course, UD4)
    result = anneal(start, UD4, _QUICK)
    for course, row in enumerate(result.solution.room_at):
        for room in row:
            if room >= 0:
                assert room not in comp01.forbidden_rooms[course]


@pytest.mark.parametrize("formulation_name", FORMULATION_NAMES)
def test_it_runs_under_every_formulation(comp01: Instance, formulation_name: str) -> None:
    """Every formulation has a different soft-cost mix; the search must not assume UD2's."""
    formulation = FORMULATIONS[formulation_name]
    construction = dsatur(comp01, formulation, seed=0, restarts=8)
    start = assign_rooms(comp01, construction.periods_of_course, formulation)
    result = anneal(start, formulation, _QUICK)
    assert cost(comp01, result.solution, formulation).violations == 0
    assert result.best_cost <= result.initial_cost
