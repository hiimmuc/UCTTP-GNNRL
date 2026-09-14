"""Exact and incremental cost evaluation, transcribed from the reference validator.

Every component is computed by a scoped helper covering the smallest unit the component is
defined over: a course, a (course, day), or a (curriculum, day). `cost` sums those helpers over
everything; `delta` sums them over only the units a move touches, before and after. Both use the
same helpers, so a full evaluation and an incremental one cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from horarium.problem.formulations import Formulation, Hard, Soft
from horarium.problem.instance import Instance
from horarium.problem.solution import EMPTY, Move, Solution

__all__ = ["CostBreakdown", "cost", "delta", "soft_cost_upper_bound"]


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Hard violation counts and weighted soft costs, keyed exactly as the validator prints them."""

    formulation: str
    hard: dict[Hard, int]
    soft: dict[Soft, int]

    @property
    def violations(self) -> int:
        """Total hard violations."""
        return sum(self.hard.values())

    @property
    def total(self) -> int:
        """Total weighted soft cost."""
        return sum(self.soft.values())

    @property
    def feasible(self) -> bool:
        """True when no hard constraint is violated."""
        return self.violations == 0


def cost(instance: Instance, solution: Solution, formulation: Formulation) -> CostBreakdown:
    """Evaluate a solution exactly, component by component.

    Args:
        instance: The instance the solution is for.
        solution: The solution to evaluate.
        formulation: Which components are active and at what weight.

    Returns:
        Hard violation counts and weighted soft costs.
    """
    courses = range(instance.n_courses)
    curricula = range(instance.n_curricula)
    days = range(instance.days)
    raw = {
        Soft.ROOM_CAPACITY: _sum(_room_capacity_course(instance, solution, c) for c in courses),
        Soft.MIN_WORKING_DAYS: _sum(
            _min_working_days_course(instance, solution, c) for c in courses
        ),
        Soft.ROOM_STABILITY: _sum(_room_stability_course(solution, c) for c in courses),
        Soft.ROOM_CONSTRAINTS: _sum(
            _room_constraints_course(instance, solution, c) for c in courses
        ),
        Soft.DOUBLE_LECTURES: _sum(
            _double_lectures_course_day(instance, solution, c, d) for c in courses for d in days
        ),
        Soft.ISOLATED_LECTURES: _sum(
            _isolated_curriculum_day(instance, solution, q, d) for q in curricula for d in days
        ),
        Soft.CURRICULUM_COMPACTNESS: _sum(
            _windows_curriculum_day(instance, solution, q, d) for q in curricula for d in days
        ),
        Soft.STUDENT_LOAD: _sum(
            _student_load_curriculum_day(instance, solution, q, d) for q in curricula for d in days
        ),
        Soft.TRAVEL_DISTANCE: _sum(
            _travel_curriculum_day(instance, solution, q, d) for q in curricula for d in days
        ),
    }
    hard = {
        Hard.LECTURES: _sum(_lectures_course(instance, solution, c) for c in courses),
        Hard.CONFLICTS: _sum(
            _conflicts_period(instance, solution, p) for p in range(instance.n_periods)
        ),
        Hard.AVAILABILITY: _sum(_availability_course(instance, solution, c) for c in courses),
        Hard.ROOM_OCCUPATION: _sum(
            max(0, load - 1) for row in solution.room_period_load for load in row
        ),
    }
    if formulation.room_constraints_are_hard:
        hard[Hard.ROOM_CONSTRAINTS] = raw[Soft.ROOM_CONSTRAINTS]
    return CostBreakdown(
        formulation=formulation.name,
        hard={key: hard[key] for key in formulation.hard},
        soft={key: raw[key] * weight for key, weight in formulation.weights.items()},
    )


def delta(instance: Instance, solution: Solution, move: Move, formulation: Formulation) -> float:
    """Change in weighted soft cost if `move` were applied. The solution is left unchanged.

    Only the course, the days and the curricula the move touches are re-evaluated, so this is
    independent of instance size. Hard violations are not included; use `cost` for those.

    Args:
        instance: The instance the solution is for.
        solution: The solution to evaluate; unchanged on return.
        move: The hypothetical move.
        formulation: Which components are active and at what weight.

    Returns:
        `cost(after) - cost(before)` for the weighted soft total, without applying `move`.
    """
    previous_room = solution.room_at[move.course][move.period]
    if previous_room == move.room:
        return 0.0
    before = _scoped_soft_cost(instance, solution, move, formulation)
    solution.apply(move)
    after = _scoped_soft_cost(instance, solution, move, formulation)
    solution.undo(move, previous_room)
    return float(after - before)


def _scoped_soft_cost(
    instance: Instance, solution: Solution, move: Move, formulation: Formulation
) -> int:
    """Weighted soft cost of just the units a move can change.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        move: The move whose affected scope should be evaluated.
        formulation: Which components are active and at what weight.

    Returns:
        The weighted soft cost restricted to the scope `move` can affect.
    """
    course, day = move.course, instance.day_of(move.period)
    return sum(
        weight * _SCOPED[component](instance, solution, course, day)
        for component, weight in formulation.weights.items()
    )


def soft_cost_upper_bound(instance: Instance, formulation: Formulation) -> int:
    """A weighted soft cost no solution of this instance can exceed.

    Loose but sound, and computed in one pass. Its purpose is to size the penalty a
    constructive environment charges for failing to place a lecture: set that penalty to this
    bound and completing a timetable always beats abandoning one, whatever the soft costs.

    Args:
        instance: The instance to bound.
        formulation: Which components are active and at what weight.

    Returns:
        A weighted soft cost no solution of `instance` under `formulation` can exceed.
    """
    days, ppd = instance.days, instance.periods_per_day
    n_q, lectures = instance.n_curricula, instance.total_lectures
    # Two courses of one curriculum conflict, so a curriculum has at most one lecture per period.
    bounds = {
        Soft.ROOM_CAPACITY: sum(c.n_lectures * c.n_students for c in instance.courses),
        Soft.MIN_WORKING_DAYS: sum(c.min_working_days for c in instance.courses),
        Soft.ISOLATED_LECTURES: sum(
            course.n_lectures * len(instance.curricula_of_course[index])
            for index, course in enumerate(instance.courses)
        ),
        Soft.CURRICULUM_COMPACTNESS: n_q * days * max(0, ppd - 2),
        Soft.ROOM_STABILITY: sum(max(0, c.n_lectures - 1) for c in instance.courses),
        Soft.DOUBLE_LECTURES: lectures,
        Soft.ROOM_CONSTRAINTS: lectures,
        Soft.STUDENT_LOAD: n_q * days * max(instance.min_daily_lectures, ppd),
        Soft.TRAVEL_DISTANCE: sum(
            days * max(0, ppd - 1) * len(members) ** 2 for members in instance.courses_of_curriculum
        ),
    }
    return sum(bounds[key] * weight for key, weight in formulation.weights.items())


# --------------------------------------------------------------------- soft components


def _room_capacity_course(instance: Instance, solution: Solution, course: int) -> int:
    """Seats short, summed over every lecture of the course.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        course: Course index.

    Returns:
        The raw (unweighted) RoomCapacity cost for this course.
    """
    students = instance.courses[course].n_students
    return sum(
        max(0, students - instance.rooms[room].capacity)
        for room in solution.room_at[course]
        if room != EMPTY
    )


def _min_working_days_course(instance: Instance, solution: Solution, course: int) -> int:
    """Working days short of the course's minimum.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        course: Course index.

    Returns:
        The raw (unweighted) MinWorkingDays cost for this course.
    """
    return max(0, instance.courses[course].min_working_days - solution.working_days[course])


def _room_stability_course(solution: Solution, course: int) -> int:
    """Distinct rooms beyond the first.

    Args:
        solution: The current solution.
        course: Course index.

    Returns:
        The raw (unweighted) RoomStability cost for this course.
    """
    return max(0, len(solution.rooms_of_course[course]) - 1)


def _room_constraints_course(instance: Instance, solution: Solution, course: int) -> int:
    """Lectures sitting in a room the course is forbidden from.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        course: Course index.

    Returns:
        The raw (unweighted) RoomConstraints cost for this course.
    """
    banned = instance.forbidden_rooms[course]
    return sum(1 for room in solution.room_at[course] if room != EMPTY and room in banned)


def _double_lectures_course_day(
    instance: Instance, solution: Solution, course: int, day: int
) -> int:
    """Lectures of a double-lecture course that lack an adjacent same-room lecture that day.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        course: Course index.
        day: Day index.

    Returns:
        The raw (unweighted) DoubleLectures cost for this course on this day.
    """
    if not instance.courses[course].double_lectures:
        return 0
    if solution.course_daily_lectures[course][day] < 2:  # noqa: PLR2004
        return 0
    row = solution.room_at[course]
    first, last = _day_bounds(instance, day)
    return sum(
        1
        for p in range(first, last + 1)
        if row[p] != EMPTY
        and (p == last or row[p + 1] != row[p])
        and (p == first or row[p - 1] != row[p])
    )


def _isolated_curriculum_day(instance: Instance, solution: Solution, q: int, day: int) -> int:
    """Lectures of a curriculum with no adjacent lecture of the same curriculum that day.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        q: Curriculum index.
        day: Day index.

    Returns:
        The raw (unweighted) IsolatedLectures cost for this curriculum on this day.
    """
    load = solution.curriculum_period_lectures[q]
    first, last = _day_bounds(instance, day)
    return sum(
        load[p]
        for p in range(first, last + 1)
        if load[p] > 0 and (p == first or load[p - 1] == 0) and (p == last or load[p + 1] == 0)
    )


def _windows_curriculum_day(instance: Instance, solution: Solution, q: int, day: int) -> int:
    """Empty periods between a curriculum's first and last lecture of the day.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        q: Curriculum index.
        day: Day index.

    Returns:
        The raw (unweighted) CurriculumCompactness cost for this curriculum on this day.
    """
    if solution.curriculum_daily_lectures[q][day] < 2:  # noqa: PLR2004
        return 0
    load = solution.curriculum_period_lectures[q]
    first, last = _day_bounds(instance, day)
    while load[first] == 0:
        first += 1
    while load[last] == 0:
        last -= 1
    return sum(1 for p in range(first + 1, last) if load[p] == 0)


def _student_load_curriculum_day(instance: Instance, solution: Solution, q: int, day: int) -> int:
    """Daily lectures outside the instance's min/max band, on days the curriculum is taught.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        q: Curriculum index.
        day: Day index.

    Returns:
        The raw (unweighted) StudentLoad cost for this curriculum on this day.
    """
    taught = solution.curriculum_daily_lectures[q][day]
    if taught == 0:
        return 0
    if taught < instance.min_daily_lectures:
        return instance.min_daily_lectures - taught
    return max(0, taught - instance.max_daily_lectures)


def _travel_curriculum_day(instance: Instance, solution: Solution, q: int, day: int) -> int:
    """Consecutive same-curriculum lecture pairs that sit in different buildings.

    Ordered pairs, and a course following itself counts, exactly as the validator does it.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        q: Curriculum index.
        day: Day index.

    Returns:
        The raw (unweighted) TravelDistance cost for this curriculum on this day.
    """
    members = instance.courses_of_curriculum[q]
    load = solution.curriculum_period_lectures[q]
    first, last = _day_bounds(instance, day)
    building = [room.building for room in instance.rooms]
    total = 0
    for p in range(first, last):
        if load[p] == 0:
            continue
        here = [solution.room_at[c][p] for c in members]
        there = [solution.room_at[c][p + 1] for c in members]
        total += sum(
            1
            for r1 in here
            if r1 != EMPTY
            for r2 in there
            if r2 != EMPTY and building[r1] != building[r2]
        )
    return total


# --------------------------------------------------------------------- hard constraints


def _lectures_course(instance: Instance, solution: Solution, course: int) -> int:
    """Distance between the lectures scheduled and the lectures required.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        course: Course index.

    Returns:
        The Lectures hard-violation count for this course.
    """
    return abs(solution.n_scheduled[course] - instance.courses[course].n_lectures)


def _availability_course(instance: Instance, solution: Solution, course: int) -> int:
    """Lectures placed in periods the course is unavailable for.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        course: Course index.

    Returns:
        The Availability hard-violation count for this course.
    """
    blocked = instance.unavailable[course]
    return sum(1 for p in blocked if solution.room_at[course][p] != EMPTY)


def _conflicts_period(instance: Instance, solution: Solution, period: int) -> int:
    """Unordered pairs of conflicting courses both taught in this period.

    Args:
        instance: The instance the solution is for.
        solution: The current solution.
        period: Flattened period index.

    Returns:
        The Conflicts hard-violation count for this period.
    """
    present = solution.courses_at_period[period]
    return sum(len(instance.conflicts[c] & present) for c in present) // 2


# --------------------------------------------------------------------- helpers


def _day_bounds(instance: Instance, day: int) -> tuple[int, int]:
    """First and last flattened period index of a day, both inclusive.

    Args:
        instance: The instance defining the day/period structure.
        day: Day index.

    Returns:
        The (first, last) flattened period indices of the day.
    """
    first = day * instance.periods_per_day
    return first, first + instance.periods_per_day - 1


def _sum(values: Iterable[int]) -> int:
    """Sum an iterable of ints; named so the component tables above stay readable.

    Args:
        values: The values to sum.

    Returns:
        Their sum, or 0 for an empty iterable.
    """
    return sum(values)


# Raw value of each component over the units a single (course, day) cell can affect. `cost`
# sums the same helpers over every unit; `delta` sums them over these scopes only.
_SCOPED: dict[Soft, Callable[[Instance, Solution, int, int], int]] = {
    Soft.ROOM_CAPACITY: lambda i, s, c, _d: _room_capacity_course(i, s, c),
    Soft.MIN_WORKING_DAYS: lambda i, s, c, _d: _min_working_days_course(i, s, c),
    Soft.ROOM_STABILITY: lambda _i, s, c, _d: _room_stability_course(s, c),
    Soft.ROOM_CONSTRAINTS: lambda i, s, c, _d: _room_constraints_course(i, s, c),
    Soft.DOUBLE_LECTURES: _double_lectures_course_day,
    Soft.ISOLATED_LECTURES: lambda i, s, c, d: _sum(
        _isolated_curriculum_day(i, s, q, d) for q in i.curricula_of_course[c]
    ),
    Soft.CURRICULUM_COMPACTNESS: lambda i, s, c, d: _sum(
        _windows_curriculum_day(i, s, q, d) for q in i.curricula_of_course[c]
    ),
    Soft.STUDENT_LOAD: lambda i, s, c, d: _sum(
        _student_load_curriculum_day(i, s, q, d) for q in i.curricula_of_course[c]
    ),
    Soft.TRAVEL_DISTANCE: lambda i, s, c, d: _sum(
        _travel_curriculum_day(i, s, q, d) for q in i.curricula_of_course[c]
    ),
}
