"""The constructive baselines: what they produce must be a legal timetable, not just a table."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from horarium.problem.cost import cost
from horarium.problem.formulations import UD2, Hard
from horarium.problem.instance import Instance
from horarium.solvers.construction import Construction, dsatur, random_order
from horarium.solvers.rooms import assign_rooms

Constructor = Callable[..., Construction]

#: Both constructors differ only in which course they pick next, so every hard-feasibility
#: invariant below has to hold for either one.
CONSTRUCTORS = [dsatur, random_order]
constructors = pytest.mark.parametrize("build", CONSTRUCTORS, ids=lambda f: f.__name__)


@constructors
def test_a_complete_construction_is_a_hard_feasible_timetable(
    sample: Instance, build: Constructor
) -> None:
    """Feasibility is the whole point of a baseline: incomplete is allowed, illegal is not."""
    construction = build(sample, UD2, restarts=8)
    if construction.complete:
        solution = assign_rooms(sample, construction.periods_of_course, UD2)
        assert cost(sample, solution, UD2).violations == 0


@constructors
def test_no_course_takes_the_same_period_twice(sample: Instance, build: Constructor) -> None:
    """Two lectures of one course in one period is a Conflicts violation with itself."""
    for periods in build(sample, UD2, restarts=4).periods_of_course:
        assert len(periods) == len(set(periods))


@constructors
def test_no_course_is_scheduled_when_it_is_unavailable(
    sample: Instance, build: Constructor
) -> None:
    """Availability is hard in every formulation."""
    for course, periods in enumerate(build(sample, UD2, restarts=4).periods_of_course):
        assert set(periods) <= set(sample.available_periods[course])


@constructors
def test_conflicting_courses_never_share_a_period(sample: Instance, build: Constructor) -> None:
    """The colouring constraint itself, checked against the instance rather than the counters."""
    at_period: dict[int, set[int]] = {}
    for course, periods in enumerate(build(sample, UD2, restarts=4).periods_of_course):
        for period in periods:
            at_period.setdefault(period, set()).add(course)
    for courses in at_period.values():
        for course in courses:
            assert not (sample.conflicts[course] & (courses - {course}))


@constructors
def test_a_period_never_holds_more_lectures_than_there_are_rooms(
    sample: Instance, build: Constructor
) -> None:
    """RoomOccupation is what bounds the colour classes; exceeding it makes rooming impossible."""
    load: dict[int, int] = {}
    for periods in build(sample, UD2, restarts=4).periods_of_course:
        for period in periods:
            load[period] = load.get(period, 0) + 1
    assert max(load.values(), default=0) <= sample.n_rooms


@constructors
def test_the_same_seed_gives_the_same_construction(comp01: Instance, build: Constructor) -> None:
    """Restarts explore only because the seed changes, so a fixed seed must be reproducible."""
    first = build(comp01, UD2, seed=7, restarts=3)
    assert build(comp01, UD2, seed=7, restarts=3).periods_of_course == first.periods_of_course


@constructors
def test_more_restarts_never_place_fewer_lectures(comp01: Instance, build: Constructor) -> None:
    """Restarts keep the best attempt, so the count is monotone in the budget."""
    counts = [build(comp01, UD2, seed=0, restarts=n).placed for n in (1, 2, 4, 8)]
    assert counts == sorted(counts)


@constructors
@pytest.mark.parametrize("restarts", [0, 1])
def test_a_degenerate_restart_budget_still_returns_one_attempt(
    toy: Instance, build: Constructor, restarts: int
) -> None:
    """`restarts=0` is a caller mistake, not a reason to return nothing."""
    assert build(toy, UD2, restarts=restarts).placed > 0


def test_a_different_seed_gives_a_different_random_order(comp01: Instance) -> None:
    """If the seed did not change the order, restarts would explore nothing."""
    first = random_order(comp01, UD2, seed=0, restarts=1)
    assert (
        random_order(comp01, UD2, seed=1, restarts=1).periods_of_course != first.periods_of_course
    )


def test_dsatur_solves_the_instance_the_policy_found_easiest(comp01: Instance) -> None:
    """comp01 is the smallest ITC-2007 instance; a baseline that cannot do it is not a baseline."""
    construction = dsatur(comp01, UD2, restarts=8)
    assert construction.complete
    solution = assign_rooms(comp01, construction.periods_of_course, UD2)
    assert cost(comp01, solution, UD2).hard[Hard.CONFLICTS] == 0


def test_saturation_ordering_is_not_merely_a_random_ordering(comp01: Instance) -> None:
    """The two constructors share everything but the selection rule, so they must diverge."""
    ordered = dsatur(comp01, UD2, seed=0, restarts=1)
    shuffled = random_order(comp01, UD2, seed=0, restarts=1)
    assert ordered.periods_of_course != shuffled.periods_of_course
