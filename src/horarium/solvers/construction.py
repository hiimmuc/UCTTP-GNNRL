"""Constructive baselines: the heuristics any learned constructor has to beat.

Assigning lectures to periods is graph colouring with bounded colour classes. The courses are
the vertices, a conflict is an edge, the periods are the colours, and a period can hold at most
`n_rooms` lectures because every lecture needs its own room. Courses needing several lectures
need several distinct colours.

`dsatur` colours the most constrained vertex first -- classically the one whose neighbours already
use the most colours, which here is equivalently the one with the fewest periods still open. That
ordering is the standard strong baseline for graph colouring, and the point of having it is that
the RL policy's whole job is choosing a construction order. If DSATUR reaches feasibility on the
instances the policy cannot, the policy is not learning a useful ordering.

`random_order` is the control for that claim. It differs from `dsatur` in exactly one thing --
which course it picks next -- and is otherwise the same code taking the same periods, so the gap
between the two is attributable to the ordering rule alone. A policy that cannot beat it has
learned nothing about ordering.

Getting stuck is not fatal: `repair` tries a depth-1 ejection, relocating the lectures that block
a stuck course rather than giving up on it. Randomised tie-breaking makes restarts explore
different orders.

Rooms are not chosen here at all. These produce a period assignment, and `rooms.assign_rooms`
turns it into a timetable -- the same stage-2 oracle the learned method uses, so the comparison
is like for like.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from horarium.problem.formulations import Formulation
from horarium.problem.instance import Instance

__all__ = ["Construction", "Order", "dsatur", "random_order"]


class Order(StrEnum):
    """Which course to colour next -- the only thing separating the two constructors."""

    SATURATION = "saturation"
    """Fewest periods still open, i.e. DSATUR's most-constrained-first rule."""

    RANDOM = "random"
    """Uniformly at random among the courses still owed lectures."""


@dataclass(frozen=True, slots=True)
class Construction:
    """A period assignment and how much of the instance it managed to place."""

    periods_of_course: list[list[int]]
    placed: int
    required: int

    @property
    def complete(self) -> bool:
        """True when every lecture found a period."""
        return self.placed == self.required


class _Constructor:
    """Incremental colouring state: which (course, period) pairs are still legal, and why not.

    Mirrors the counters `ConstructEnv` keeps, but reads the room limit as "at most `n_rooms`
    lectures share a period" rather than restricting each course to its `permitted_rooms`. Under
    every formulation but UD4 an unwanted room is free, so the tighter rule only discards
    zero-cost options.
    """

    def __init__(self, instance: Instance, rng: random.Random) -> None:
        """Set up the counters for an empty timetable.

        Args:
            instance: The instance to construct for.
            rng: Source of randomness for tie-breaking between equally good choices.
        """
        self.instance = instance
        self.rng = rng
        n_courses, n_periods = instance.n_courses, instance.n_periods

        self.available = np.zeros((n_courses, n_periods), dtype=np.bool_)
        for course, periods in enumerate(instance.available_periods):
            self.available[course, list(periods)] = True
        self.conflicts_of = tuple(
            np.fromiter(sorted(peers), dtype=np.intp, count=len(peers))
            for peers in instance.conflicts
        )
        self.degree = np.fromiter(
            (len(peers) for peers in instance.conflicts), dtype=np.int32, count=n_courses
        )
        self.occupied = np.zeros((n_courses, n_periods), dtype=np.bool_)
        self.blocked = np.zeros((n_courses, n_periods), dtype=np.int32)
        self.load = np.zeros(n_periods, dtype=np.int32)
        self.remaining = np.fromiter(
            (course.n_lectures for course in instance.courses), dtype=np.int32, count=n_courses
        )
        self.chosen: list[list[int]] = [[] for _ in range(n_courses)]

    def open_periods(self, course: int) -> npt.NDArray[np.bool_]:
        """Periods `course` could still take a lecture in.

        Args:
            course: Course index.

        Returns:
            A boolean mask over periods.
        """
        mask: npt.NDArray[np.bool_] = (
            self.available[course]
            & ~self.occupied[course]
            & (self.blocked[course] == 0)
            & (self.load < self.instance.n_rooms)
        )
        return mask

    def place(self, course: int, period: int) -> None:
        """Colour one lecture of `course` with `period`.

        Args:
            course: Course index.
            period: Flattened period index.
        """
        self.occupied[course, period] = True
        self.blocked[self.conflicts_of[course], period] += 1
        self.load[period] += 1
        self.remaining[course] -= 1
        self.chosen[course].append(period)

    def unplace(self, course: int, period: int) -> None:
        """Undo `place`.

        Args:
            course: Course index.
            period: Flattened period index.
        """
        self.occupied[course, period] = False
        self.blocked[self.conflicts_of[course], period] -= 1
        self.load[period] -= 1
        self.remaining[course] += 1
        self.chosen[course].remove(period)

    def next_course(self, order: Order) -> int | None:
        """The next course to colour, under the given ordering rule.

        For `Order.SATURATION`, ties break on lectures still owed, then conflict degree, then at
        random, so restarts explore genuinely different orders rather than re-running one
        deterministic pass. The count is taken for every course at once: doing it course by
        course costs a full `(n_courses, n_periods)` pass per placement, which dominates the run
        on the big families.

        Args:
            order: Which selection rule to apply.

        Returns:
            A course index, or `None` when every lecture is placed.
        """
        unfinished = self.remaining > 0
        if not unfinished.any():
            return None
        if order is Order.RANDOM:
            return int(self.rng.choice(np.flatnonzero(unfinished).tolist()))
        still_open = (
            self.available
            & ~self.occupied
            & (self.blocked == 0)
            & (self.load < self.instance.n_rooms)
        )
        counts = np.where(unfinished, still_open.sum(axis=1), np.iinfo(np.int64).max)
        candidates = np.flatnonzero(counts == counts.min())
        return min(
            (int(c) for c in candidates),
            key=lambda c: (-int(self.remaining[c]), -int(self.degree[c]), self.rng.random()),
        )

    def best_period(self, course: int) -> int | None:
        """Where to put the next lecture of `course`.

        Prefers a day the course does not use yet, which is what `MinWorkingDays` charges for,
        then the emptiest period, so room capacity stays available for courses picked later.

        Args:
            course: Course index.

        Returns:
            A period index, or `None` if the course is stuck.
        """
        options = np.flatnonzero(self.open_periods(course))
        if options.size == 0:
            return None
        per_day = self.instance.periods_per_day
        days_used = {period // per_day for period in self.chosen[course]}
        return min(
            (int(p) for p in options),
            key=lambda p: (p // per_day in days_used, int(self.load[p]), self.rng.random()),
        )

    def repair(self, course: int) -> bool:
        """Free a period for a stuck `course` by relocating whatever blocks it.

        Looks for the period blocked by the fewest conflicting lectures and tries to move each of
        them somewhere else. Depth 1: the evicted lectures must find room without evicting anyone
        in turn, which keeps the repair cheap and guarantees it terminates.

        Args:
            course: The course that has no period left.

        Returns:
            Whether a period was freed.
        """
        room = self.available[course] & ~self.occupied[course] & (self.load < self.instance.n_rooms)
        for period in sorted(np.flatnonzero(room), key=lambda p: int(self.blocked[course, p])):
            blockers = [
                (int(peer), int(period))
                for peer in self.conflicts_of[course]
                if self.occupied[peer, period]
            ]
            if self._relocate(blockers):
                return True
        return False

    def _relocate(self, blockers: list[tuple[int, int]]) -> bool:
        """Move every blocking lecture elsewhere, rolling back entirely if one cannot move.

        Args:
            blockers: The (course, period) lectures standing in the way.

        Returns:
            Whether all of them moved.
        """
        moved: list[tuple[int, int, int]] = []
        for peer, period in blockers:
            self.unplace(peer, period)
            target = self.best_period(peer)
            if target is None:
                self.place(peer, period)
                for course, was, now in reversed(moved):
                    self.unplace(course, now)
                    self.place(course, was)
                return False
            self.place(peer, target)
            moved.append((peer, period, target))
        return True

    def finish(self) -> Construction:
        """Package the colouring reached so far.

        Returns:
            The period assignment and how many lectures it placed.
        """
        return Construction(
            periods_of_course=[sorted(periods) for periods in self.chosen],
            placed=sum(len(periods) for periods in self.chosen),
            required=self.instance.total_lectures,
        )


def _attempt(instance: Instance, seed: int, order: Order) -> Construction:
    """Colour the whole instance once.

    Args:
        instance: The instance to construct for.
        seed: Seed for tie-breaking.
        order: Which course-selection rule to use.

    Returns:
        The construction this pass reached, complete or stuck.
    """
    state = _Constructor(instance, random.Random(seed))  # noqa: S311 - tie-breaking, not crypto
    while True:
        course = state.next_course(order)
        if course is None:
            break
        period = state.best_period(course)
        if period is None:
            if not state.repair(course):
                break
            period = state.best_period(course)
            if period is None:
                break
        state.place(course, period)
    return state.finish()


def construct(
    instance: Instance, order: Order, *, seed: int = 0, restarts: int = 1
) -> Construction:
    """Assign every lecture a period, retrying until one attempt places them all.

    Args:
        instance: The instance to construct for.
        order: Which course-selection rule to use.
        seed: Seed for tie-breaking.
        restarts: Attempts to make, keeping the one that placed the most lectures. Restarts stop
            early as soon as one is complete.

    Returns:
        The best construction found.
    """
    best = _attempt(instance, seed, order)
    for extra in range(1, max(1, restarts)):
        if best.complete:
            break
        found = _attempt(instance, seed + extra, order)
        if found.placed > best.placed:
            best = found
    return best


def dsatur(
    instance: Instance,
    formulation: Formulation | None = None,  # noqa: ARG001 - shared constructor signature
    *,
    seed: int = 0,
    restarts: int = 1,
) -> Construction:
    """Assign every lecture a period, most-constrained course first.

    Args:
        instance: The instance to construct for.
        formulation: Accepted so every constructor shares one signature; the period assignment
            does not depend on the soft weights.
        seed: Seed for tie-breaking.
        restarts: Attempts to make, keeping the one that placed the most lectures.

    Returns:
        The best construction found.
    """
    return construct(instance, Order.SATURATION, seed=seed, restarts=restarts)


def random_order(
    instance: Instance,
    formulation: Formulation | None = None,  # noqa: ARG001 - shared constructor signature
    *,
    seed: int = 0,
    restarts: int = 1,
) -> Construction:
    """Assign every lecture a period, choosing the next course uniformly at random.

    The control for `dsatur`: identical in every respect except which course it picks next, so
    any gap between them measures the ordering rule and nothing else.

    Args:
        instance: The instance to construct for.
        formulation: Accepted so every constructor shares one signature; the period assignment
            does not depend on the soft weights.
        seed: Seed for the course order.
        restarts: Attempts to make, keeping the one that placed the most lectures.

    Returns:
        The best construction found.
    """
    return construct(instance, Order.RANDOM, seed=seed, restarts=restarts)
