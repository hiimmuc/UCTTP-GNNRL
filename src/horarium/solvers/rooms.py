"""Stage 2 of the decomposition: choose rooms for a period assignment that is already fixed.

Once every lecture has a period, the rooms no longer interact through the hard constraints:
`RoomOccupation` only says two lectures in the same period need different rooms, so each period
is an independent bipartite assignment of its courses to the rooms. `RoomCapacity` and
`RoomConstraints` are both sums over individual lectures, so for those components a period's
optimum is found exactly by the Hungarian algorithm, and solving every period optimally is
globally optimal.

`RoomStability` is the one component that couples periods -- it charges a course for each
distinct room beyond the first -- so with it active the problem stops decomposing. Holding every
other period fixed, though, the marginal stability cost of putting course `c` in room `r` during
period `p` is `0` if `c` already uses `r` somewhere else and the weight otherwise. That is the
whole of the varying part, so one period's re-assignment is still an exact Hungarian step, and
sweeping over the periods is block-coordinate descent: every sweep is monotone non-increasing and
the fixed point is an assignment no single-period change can improve. A local optimum, not a
global one, and `assign_rooms` says so rather than overclaiming.

Because the descent is monotone *from wherever it starts*, `reassign_rooms` warm-starts from the
rooms the caller already had. That makes it a true improvement operator: it can never return
something worse than it was given, which a cold start cannot promise once stability is in play.

This replaces the "cheapest free room" rule in `ConstructEnv`, which was a placeholder standing
in for exactly this. Note that it is deliberately formulation-aware: under UD2 `RoomConstraints`
carries no weight and is not hard, so a course sitting in a room its file lists as unwanted is
free, and restricting the search to `permitted_rooms` -- as the placeholder rule did -- only
throws away zero-cost options.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from horarium.problem.formulations import Formulation, Soft
from horarium.problem.instance import Instance
from horarium.problem.solution import EMPTY, Solution

__all__ = [
    "RoomAssignmentError",
    "assign_rooms",
    "cheapest_room",
    "reassign_rooms",
    "usable_rooms",
]

#: Cost charged for a pairing the formulation forbids outright. Large enough that the assignment
#: never chooses one unless it has no alternative, small enough to keep the matrix finite so
#: `linear_sum_assignment` still returns and we can report which pairing was impossible.
_FORBIDDEN = 1e9


class RoomAssignmentError(ValueError):
    """No room assignment exists for the given periods.

    Either some period holds more lectures than there are rooms, or the formulation makes room
    constraints hard and the permitted rooms cannot cover the period's courses.
    """


def usable_rooms(instance: Instance, formulation: Formulation) -> tuple[tuple[int, ...], ...]:
    """Rooms each course may occupy without violating a hard constraint.

    A room a course merely dislikes is chargeable at worst under every formulation but UD4,
    where `RoomConstraints` is hard and such a room is off-limits outright. Restricting the
    candidate set to `permitted_rooms` unconditionally throws away legal, sometimes free,
    placements everywhere else.

    Args:
        instance: The instance being solved.
        formulation: Which components are active and at what weight.

    Returns:
        For each course, the rooms it may be greedily assigned to.
    """
    if formulation.room_constraints_are_hard:
        return instance.permitted_rooms
    every_room = tuple(range(instance.n_rooms))
    return (every_room,) * instance.n_courses


def cheapest_room(  # noqa: PLR0913 - one argument per fact the choice genuinely depends on
    instance: Instance,
    solution: Solution,
    course: int,
    period: int,
    formulation: Formulation,
    *,
    candidates: Sequence[int] | None = None,
) -> int | None:
    """The cheapest usable room free for `course` in `period`, chosen greedily.

    Ties break on whether `course` already occupies the room elsewhere, then on room index, so
    the choice is deterministic given the state. This is the fast rule a constructive method
    uses to keep every partial state cost-evaluable; `assign_rooms`/`reassign_rooms` is the exact
    oracle that replaces it once a whole period assignment is fixed and worth optimising properly.

    Args:
        instance: The instance being solved.
        solution: The current solution.
        course: Course index.
        period: Flattened period index.
        formulation: Which components are active and at what weight.
        candidates: Rooms to consider, typically `usable_rooms(instance, formulation)[course]`.
            Recomputed on the fly if omitted, which is wasteful when called in a loop.

    Returns:
        The chosen room index, or `None` if nothing usable is free.
    """
    rooms = (
        candidates
        if candidates is not None
        else (
            instance.permitted_rooms[course]
            if formulation.room_constraints_are_hard
            else range(instance.n_rooms)
        )
    )
    students = instance.courses[course].n_students
    already = solution.rooms_of_course[course]
    banned = instance.forbidden_rooms[course]
    penalty = float(formulation.weights.get(Soft.ROOM_CONSTRAINTS, 0))
    best: tuple[float, int, int] | None = None
    for room in rooms:
        if solution.room_period_load[room][period]:
            continue
        key = (
            max(0, students - instance.rooms[room].capacity) + (penalty if room in banned else 0.0),
            0 if room in already else 1,
            room,
        )
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


def _pair_costs(instance: Instance, formulation: Formulation) -> npt.NDArray[np.float64]:
    """Weighted cost of putting one lecture of each course in each room, stability aside.

    Args:
        instance: The instance being solved.
        formulation: Which components are active and at what weight.

    Returns:
        A `(n_courses, n_rooms)` array of per-lecture costs.
    """
    students = np.fromiter(
        (course.n_students for course in instance.courses), dtype=np.int64, count=instance.n_courses
    )[:, None]
    capacities = np.fromiter(
        (room.capacity for room in instance.rooms), dtype=np.int64, count=instance.n_rooms
    )[None, :]
    weight = formulation.weights.get(Soft.ROOM_CAPACITY, 0)
    costs: npt.NDArray[np.float64] = float(weight) * np.maximum(0, students - capacities).astype(
        np.float64
    )

    penalty = (
        _FORBIDDEN
        if formulation.room_constraints_are_hard
        else float(formulation.weights.get(Soft.ROOM_CONSTRAINTS, 0))
    )
    if penalty:
        for course, banned in enumerate(instance.forbidden_rooms):
            if banned:
                costs[course, list(banned)] += penalty
    return costs


def _assign_period(
    block: npt.NDArray[np.float64], courses: Sequence[int], period: int
) -> list[int]:
    """Solve one period's bipartite assignment exactly.

    Args:
        block: A `(len(courses), n_rooms)` cost array for this period.
        courses: The courses holding a lecture in this period, in `block`'s row order.
        period: The period being solved, used only for error messages.

    Returns:
        The chosen room for each course, in `courses` order.

    Raises:
        RoomAssignmentError: The period has more lectures than rooms, or no permitted room is
            left for some course.
    """
    if block.shape[0] > block.shape[1]:
        msg = (
            f"period {period} holds {block.shape[0]} lectures but the instance has "
            f"{block.shape[1]} rooms"
        )
        raise RoomAssignmentError(msg)
    rows, columns = linear_sum_assignment(block)
    if float(block[rows, columns].max(initial=0.0)) >= _FORBIDDEN:
        blocked = [
            courses[int(r)] for r, c in zip(rows, columns, strict=True) if block[r, c] >= _FORBIDDEN
        ]
        msg = f"period {period}: no permitted room left for course(s) {blocked}"
        raise RoomAssignmentError(msg)
    chosen = [0] * len(courses)
    for row, column in zip(rows, columns, strict=True):
        chosen[int(row)] = int(column)
    return chosen


def _release(counter: Counter[int], room: int) -> None:
    """Drop one use of `room`, removing the key entirely when it falls to zero.

    `Counter` keeps zero-valued keys, and the stability term tests membership, so a room left at
    zero would keep looking "already used". `EMPTY` means the cell held nothing and is ignored.

    Args:
        counter: The course's room multiset.
        room: The room to release one use of, or `EMPTY` for none.
    """
    if room == EMPTY:
        return
    if counter[room] <= 1:
        del counter[room]
    else:
        counter[room] -= 1


@dataclass(slots=True)
class _Descent:
    """One block-coordinate descent over the periods, and the state it walks.

    `rooms` and `used` are the two halves of the same assignment and are always updated together:
    `rooms[c, p]` is the room course `c` occupies in period `p`, and `used[c]` counts how many
    lectures of `c` sit in each room, which is what the stability term reads.
    """

    at_period: list[list[int]]
    pair_costs: npt.NDArray[np.float64]
    stability: float
    rooms: npt.NDArray[np.int64]
    used: list[Counter[int]]

    def run(self, sweeps: int) -> None:
        """Sweep until nothing moves, or until `sweeps` is spent.

        Without stability the periods are independent, so one sweep is already optimal and any
        further pass would only reproduce it.

        Args:
            sweeps: Maximum passes when stability couples the periods.
        """
        for _ in range(max(1, sweeps) if self.stability else 1):
            if not self._sweep():
                break

    def _sweep(self) -> bool:
        """Re-solve every period exactly against the rooms the other periods currently use.

        Returns:
            Whether any course changed room, i.e. whether another sweep could still help.
        """
        moved = False
        for period, courses in enumerate(self.at_period):
            if not courses:
                continue
            for course in courses:  # this period's own room must not count as "already used"
                _release(self.used[course], int(self.rooms[course, period]))

            block = self._block(courses)
            for course, room in zip(courses, _assign_period(block, courses, period), strict=True):
                moved |= room != int(self.rooms[course, period])
                self.rooms[course, period] = room
                self.used[course][room] += 1
        return moved

    def _block(self, courses: Sequence[int]) -> npt.NDArray[np.float64]:
        """Cost of every (course, room) pairing for one period, stability included.

        Args:
            courses: The courses holding a lecture in the period.

        Returns:
            A `(len(courses), n_rooms)` cost array.
        """
        block = self.pair_costs[courses]
        if not self.stability:
            return block
        block = block + self.stability
        for row, course in enumerate(courses):
            if self.used[course]:
                block[row, list(self.used[course])] -= self.stability
        return block


def _courses_at_period(
    periods_of_course: Sequence[Sequence[int]], n_periods: int
) -> list[list[int]]:
    """Invert the period assignment.

    Args:
        periods_of_course: For each course, the periods holding one of its lectures.
        n_periods: Number of periods in the instance.

    Returns:
        For each period, the courses holding a lecture in it.
    """
    at_period: list[list[int]] = [[] for _ in range(n_periods)]
    for course, periods in enumerate(periods_of_course):
        for period in periods:
            at_period[period].append(course)
    return at_period


def _build(
    instance: Instance, periods_of_course: Sequence[Sequence[int]], rooms: npt.NDArray[np.int64]
) -> Solution:
    """Turn a finished room table into a `Solution`.

    Args:
        instance: The instance being solved.
        periods_of_course: For each course, the periods holding one of its lectures.
        rooms: The `(n_courses, n_periods)` room table.

    Returns:
        A solution placing every listed lecture in its chosen room.
    """
    solution = Solution(instance)
    for course, periods in enumerate(periods_of_course):
        for period in periods:
            solution.place(course, period, int(rooms[course, period]))
    return solution


def assign_rooms(
    instance: Instance,
    periods_of_course: Sequence[Sequence[int]],
    formulation: Formulation,
    *,
    sweeps: int = 8,
) -> Solution:
    """Choose rooms for a fixed period assignment, from scratch.

    Exact for the components that decompose over lectures (`RoomCapacity`, `RoomConstraints`).
    When `RoomStability` carries weight the periods couple, and the result is instead a local
    optimum under single-period re-assignment.

    Args:
        instance: The instance being solved.
        periods_of_course: For each course, the periods holding one of its lectures. A course
            may not appear twice in the same period.
        formulation: Which components are active and at what weight.
        sweeps: Maximum stability-improving passes over the periods.

    Returns:
        A `Solution` placing every listed lecture in its chosen room.

    Raises:
        RoomAssignmentError: Some period admits no valid assignment.
    """
    descent = _Descent(
        at_period=_courses_at_period(periods_of_course, instance.n_periods),
        pair_costs=_pair_costs(instance, formulation),
        stability=float(formulation.weights.get(Soft.ROOM_STABILITY, 0)),
        rooms=np.full((instance.n_courses, instance.n_periods), EMPTY, dtype=np.int64),
        used=[Counter() for _ in range(instance.n_courses)],
    )
    descent.run(sweeps)
    return _build(instance, periods_of_course, descent.rooms)


def reassign_rooms(solution: Solution, formulation: Formulation, *, sweeps: int = 8) -> Solution:
    """Improve the rooms of an existing solution, keeping every lecture in its period.

    Warm-starts from the rooms `solution` already uses. The descent is monotone from its starting
    point, so the result is never worse than what was passed in.

    Args:
        solution: The solution whose period assignment should be kept.
        formulation: Which components are active and at what weight.
        sweeps: Maximum stability-improving passes over the periods.

    Returns:
        A new `Solution` with the same periods and rooms at least as good.

    Raises:
        RoomAssignmentError: Some period admits no valid assignment.
    """
    instance = solution.instance
    periods_of_course = [
        [period for period, room in enumerate(row) if room != EMPTY] for row in solution.room_at
    ]
    descent = _Descent(
        at_period=_courses_at_period(periods_of_course, instance.n_periods),
        pair_costs=_pair_costs(instance, formulation),
        stability=float(formulation.weights.get(Soft.ROOM_STABILITY, 0)),
        rooms=np.array(solution.room_at, dtype=np.int64),
        used=[Counter(dict(seen)) for seen in solution.rooms_of_course],
    )
    descent.run(sweeps)
    return _build(instance, periods_of_course, descent.rooms)
