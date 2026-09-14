"""Randomised solution shapes used to stress every cost component during validation and tests."""

from __future__ import annotations

import random
from collections.abc import Callable

from horarium.problem.instance import Instance
from horarium.problem.solution import Solution

__all__ = ["SHAPES", "random_solution"]


def _scattered(instance: Instance, solution: Solution, rng: random.Random) -> None:
    """Lectures anywhere, rooms anywhere: exercises every hard violation and soft cost at once.

    Args:
        instance: The instance to generate a solution for.
        solution: Empty solution to fill in place.
        rng: Source of randomness.
    """
    for c, course in enumerate(instance.courses):
        for p in rng.sample(range(instance.n_periods), min(course.n_lectures, instance.n_periods)):
            solution.place(c, p, rng.randrange(instance.n_rooms))


def _clustered(instance: Instance, solution: Solution, rng: random.Random) -> None:
    """Everything squeezed into the first two days: heavy windows, student load and conflicts.

    Args:
        instance: The instance to generate a solution for.
        solution: Empty solution to fill in place.
        rng: Source of randomness.
    """
    horizon = min(instance.n_periods, 2 * instance.periods_per_day)
    for c, course in enumerate(instance.courses):
        for p in rng.sample(range(horizon), min(course.n_lectures, horizon)):
            solution.place(c, p, rng.randrange(instance.n_rooms))


def _one_room(instance: Instance, solution: Solution, rng: random.Random) -> None:
    """Each course pinned to a single room: room stability zero, capacity and travel stressed.

    Args:
        instance: The instance to generate a solution for.
        solution: Empty solution to fill in place.
        rng: Source of randomness.
    """
    for c, course in enumerate(instance.courses):
        room = rng.randrange(instance.n_rooms)
        for p in rng.sample(range(instance.n_periods), min(course.n_lectures, instance.n_periods)):
            solution.place(c, p, room)


def _adjacent_pairs(instance: Instance, solution: Solution, rng: random.Random) -> None:
    """Lectures placed in same-room adjacent pairs: the shape double-lecture costs care about.

    Args:
        instance: The instance to generate a solution for.
        solution: Empty solution to fill in place.
        rng: Source of randomness.
    """
    for c, course in enumerate(instance.courses):
        room = rng.randrange(instance.n_rooms)
        remaining = course.n_lectures
        free = set(range(instance.n_periods))
        while remaining and free:
            start = rng.choice(sorted(free))
            run = [start] if remaining == 1 else [start, start + 1]
            if any(p not in free or instance.day_of(p) != instance.day_of(start) for p in run):
                free.discard(start)
                continue
            for p in run:
                solution.place(c, p, room)
                free.discard(p)
            remaining -= len(run)


def _partial(instance: Instance, solution: Solution, rng: random.Random) -> None:
    """Half the courses left unscheduled and some over-scheduled: exercises the Lectures count.

    Args:
        instance: The instance to generate a solution for.
        solution: Empty solution to fill in place.
        rng: Source of randomness.
    """
    for c, course in enumerate(instance.courses):
        wanted = course.n_lectures + rng.choice([-course.n_lectures, -1, 0, 1])
        wanted = max(0, min(wanted, instance.n_periods))
        for p in rng.sample(range(instance.n_periods), wanted):
            solution.place(c, p, rng.randrange(instance.n_rooms))


#: Named solution shapes. Add one here and every agreement test picks it up.
SHAPES: dict[str, Callable[[Instance, Solution, random.Random], None]] = {
    "scattered": _scattered,
    "clustered": _clustered,
    "one_room": _one_room,
    "adjacent_pairs": _adjacent_pairs,
    "partial": _partial,
}


def random_solution(instance: Instance, shape: str, seed: int) -> Solution:
    """Build a random solution of the named shape. Not feasible in general; that is the point.

    Args:
        instance: The instance to generate a solution for.
        shape: A key of `SHAPES`.
        seed: Seed for the random number generator.

    Returns:
        A solution built by the named shape generator.

    Raises:
        KeyError: The shape is not in SHAPES.
    """
    solution = Solution(instance)
    SHAPES[shape](instance, solution, random.Random(seed))  # noqa: S311 - not cryptographic
    return solution
