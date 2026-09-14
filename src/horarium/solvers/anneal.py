"""Simulated annealing: the improvement baseline a learned local-search policy has to beat.

Takes an already-complete, hard-feasible timetable -- from `dsatur`, `random_order`, or an RL
solve -- and improves it by relocating and swapping lectures between periods, accepting a
worsening move with probability `exp(-delta / T)` and cooling `T` geometrically. Two moves:

- **relocate**: move one lecture to a different period.
- **swap**: exchange the periods of two lectures.

Both are standard neighbourhoods for this kind of scheduling problem, and both are generated so
that every intermediate state stays hard-feasible -- a move is only ever proposed once a legal
period and a free room have been found for it -- so acceptance never needs a repair step. Cost is
tracked incrementally via `problem.cost.delta`, exact per the tests in `test_cost.py`, so
thousands of iterations cost a handful of scoped re-evaluations rather than one full pass each.
Rejection is just as cheap: every proposal records the exact `Solution.undo` calls that reverse
it, so backing out of a rejected move costs the same handful of cell writes as making it, rather
than a full copy of the timetable.

Rooms are chosen by the same greedy rule `ConstructEnv` uses (`cheapest_room`), which keeps every
intermediate state cost-evaluable without paying for an exact re-solve on every move. The exact
stage-2 oracle (`reassign_rooms`) is run periodically and once more at the end, so the room
assignment the search reports is never worse than what the greedy rule alone would have found.

This exists to answer one question the sweep cannot: whether the learned policy is doing more
than a generic local search would from the same start. Running it from a policy's solution and
from a construction baseline's solution at equal wall-clock is the ablation that answers it.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from horarium.problem.cost import cost, delta
from horarium.problem.formulations import Formulation
from horarium.problem.instance import Instance
from horarium.problem.solution import EMPTY, Move, Solution
from horarium.solvers.rooms import RoomAssignmentError, cheapest_room, reassign_rooms, usable_rooms

#: One applied move and the room it overwrote, in application order -- enough to undo the move
#: exactly via `Solution.undo`, without copying the timetable to take a snapshot.
_Trail = list[tuple[Move, int]]
#: What a successful proposal returns: its exact cost delta and how to undo it.
_Proposal = tuple[float, _Trail]

__all__ = ["AnnealConfig", "AnnealResult", "anneal"]

#: Trial moves used only to estimate a starting temperature; never applied.
_CALIBRATION_TRIALS = 64
#: Target acceptance probability for a "typical" worsening move at the starting temperature.
_TARGET_ACCEPTANCE = 0.5
#: Share of proposals that are relocate rather than swap: relocate is cheaper and finds most
#: improvements; swap exists for the pairs that each want the other's slot.
_RELOCATE_SHARE = 0.75
#: A swap needs two lectures to choose from.
_MIN_LECTURES_FOR_SWAP = 2


@dataclass(frozen=True, slots=True)
class AnnealConfig:
    """Hyperparameters for one `anneal` run."""

    seed: int = 0
    iterations: int = 20_000
    """Ceiling on iterations. The loop also stops early once `time_limit` elapses, so this is a
    ceiling rather than a target when a time budget is given."""
    time_limit: float | None = None
    """Wall-clock budget in seconds, or `None` to run the full `iterations`. This is what lets
    two searches from different starting points be compared at equal cost."""
    cooling: float = 0.9995
    """Geometric decay applied to the temperature after every iteration."""
    reassign_every: int = 200
    """Accepted moves between calls to the exact stage-2 room oracle. Room choices during the
    search are greedy so the loop stays fast; this periodically recovers the optimum a full room
    re-solve would have found."""


@dataclass(frozen=True, slots=True)
class AnnealResult:
    """What one annealing run achieved."""

    solution: Solution
    initial_cost: float
    best_cost: float
    iterations: int
    accepted: int
    seconds: float


class _Search:
    """Mutable state for one run: the working solution and the exact cost it is tracked against.

    `current_cost` is maintained incrementally from `delta`, not recomputed from scratch, which
    is what keeps an iteration cheap. It is resynchronised with a full `cost` evaluation whenever
    `reassign_rooms` changes the solution outside the moves this class makes itself.
    """

    def __init__(self, solution: Solution, formulation: Formulation, rng: random.Random) -> None:
        """Start a search from an existing complete, feasible solution.

        Args:
            solution: The solution to improve. Copied; the caller's object is untouched.
            formulation: Which components are active and at what weight.
            rng: Source of randomness for move proposals and the Metropolis criterion.

        Raises:
            ValueError: `solution` is not complete. Annealing only ever relocates lectures that
                are already placed, so an incomplete solution would stay incomplete.
        """
        if not solution.is_complete:
            msg = "anneal requires a complete timetable; it only relocates existing lectures"
            raise ValueError(msg)
        self.instance: Instance = solution.instance
        self.formulation = formulation
        self.rng = rng
        self.current = solution.copy()
        self.current_cost = float(cost(self.instance, self.current, formulation).total)
        self.usable = usable_rooms(self.instance, formulation)
        self.best = self.current.copy()
        self.best_cost = self.current_cost

    def _room_for(self, course: int, period: int) -> int | None:
        """The greedy room choice for `course` in `period`, restricted to usable rooms.

        Args:
            course: Course index.
            period: Flattened period index.

        Returns:
            A room index, or `None` if nothing usable is free.
        """
        return cheapest_room(
            self.instance,
            self.current,
            course,
            period,
            self.formulation,
            candidates=self.usable[course],
        )

    def _open_period(self, course: int, period: int, excusing: int | None) -> bool:
        """Whether `course` could hold a lecture in `period`, ignoring one course's presence.

        Args:
            course: Course to place.
            period: Candidate period.
            excusing: A course to treat as absent from `period` -- the peer this move is about to
                vacate it, in a swap.

        Returns:
            Whether the period is open for `course` once conflicts and self-occupancy are ruled
            out.
        """
        if period not in self.instance.available_periods[course]:
            return False
        if self.current.room_at[course][period] != EMPTY:
            return False
        occupants = self.current.courses_at_period[period]
        blocking = self.instance.conflicts[course] & occupants
        if excusing is not None:
            blocking = blocking - {excusing}
        return not blocking

    def _apply(self, move: Move, previous_room: int, trail: _Trail) -> float:
        """Apply one move, recording it on `trail` for an exact later undo, and return its delta.

        Args:
            move: The move to apply.
            previous_room: The room `move`'s cell holds beforehand -- what undoing it restores.
            trail: Appended in place with `(move, previous_room)`.

        Returns:
            The exact cost delta the move causes.
        """
        change = delta(self.instance, self.current, move, self.formulation)
        self.current.apply(move)
        trail.append((move, previous_room))
        return change

    def _undo(self, trail: _Trail) -> None:
        """Reverse every move on `trail`, most recent first.

        Args:
            trail: Moves to undo, in the order `_apply` recorded them.
        """
        for move, previous_room in reversed(trail):
            self.current.undo(move, previous_room)

    def relocate(self) -> _Proposal | None:
        """Propose moving one random lecture to a different, legal, room-available period.

        Returns:
            The move's exact cost delta and how to undo it, or `None` if the chosen lecture has
            nowhere legal to go and nothing was changed.
        """
        lectures = self.current.lectures()
        course, period, room = lectures[self.rng.randrange(len(lectures))]
        periods = [
            p
            for p in self.instance.available_periods[course]
            if p != period and self._open_period(course, p, excusing=None)
        ]
        self.rng.shuffle(periods)

        trail: _Trail = []
        total = self._apply(Move(course=course, period=period, room=EMPTY), room, trail)
        for target in periods:
            new_room = self._room_for(course, target)
            if new_room is None:
                continue
            total += self._apply(Move(course=course, period=target, room=new_room), EMPTY, trail)
            return total, trail
        self._undo(trail)  # no legal target: put the lecture straight back
        return None

    def swap(self) -> _Proposal | None:
        """Propose exchanging the periods of two random lectures.

        Returns:
            The move's exact cost delta and how to undo it, or `None` if the two lectures cannot
            legally swap and nothing was changed.
        """
        lectures = self.current.lectures()
        if len(lectures) < _MIN_LECTURES_FOR_SWAP:  # pragma: no cover - true of every sample
            return None
        (course_a, period_a, room_a), (course_b, period_b, room_b) = self.rng.sample(lectures, 2)
        if period_a == period_b or course_a == course_b:
            return None
        if not self._open_period(course_a, period_b, excusing=course_b):
            return None
        if not self._open_period(course_b, period_a, excusing=course_a):
            return None

        trail: _Trail = []
        total = self._apply(Move(course=course_a, period=period_a, room=EMPTY), room_a, trail)
        total += self._apply(Move(course=course_b, period=period_b, room=EMPTY), room_b, trail)

        room_at_b = self._room_for(course_a, period_b)
        room_at_a = self._room_for(course_b, period_a) if room_at_b is not None else None
        if room_at_b is None or room_at_a is None:
            self._undo(trail)
            return None

        total += self._apply(Move(course=course_a, period=period_b, room=room_at_b), EMPTY, trail)
        total += self._apply(Move(course=course_b, period=period_a, room=room_at_a), EMPTY, trail)
        return total, trail

    def propose(self) -> _Proposal | None:
        """Try one move, relocate three times as often as swap.

        Relocate alone cannot exchange two courses that each want the other's slot, which is
        exactly what swap is for, but it is the cheaper move and finds most improvements, so it
        gets most of the budget.

        Returns:
            The accepted proposal's delta and undo trail, or `None` if no legal move was found.
        """
        return self.relocate() if self.rng.random() < _RELOCATE_SHARE else self.swap()

    def calibrate(self) -> float:
        """Estimate a starting temperature from the scale of a typical worsening move.

        Samples `_CALIBRATION_TRIALS` proposals and immediately undoes every one, so this leaves
        no trace regardless of a proposal's sign. `T0` is set so that a move worsening cost by
        the mean magnitude seen would be accepted with probability `_TARGET_ACCEPTANCE`. A
        degenerate instance with no legal moves at all falls back to `1.0`, which only matters in
        that it never fires the acceptance test again.

        Returns:
            The starting temperature.
        """
        magnitudes = []
        for _ in range(_CALIBRATION_TRIALS):
            proposal = self.propose()
            if proposal is None:
                continue
            change, trail = proposal
            magnitudes.append(abs(change))
            self._undo(trail)
        if not magnitudes:
            return 1.0
        typical = sum(magnitudes) / len(magnitudes)
        return typical / math.log(1 / _TARGET_ACCEPTANCE) if typical else 1.0

    def step(self, temperature: float) -> bool:
        """One Metropolis iteration: propose a move, accept or reject it, track the best found.

        Rejecting a move undoes it exactly rather than restoring a saved copy of the timetable,
        which is what keeps an iteration's cost independent of the instance's size.

        Args:
            temperature: The current annealing temperature.

        Returns:
            Whether a move was proposed and accepted.
        """
        proposal = self.propose()
        if proposal is None:
            return False
        change, trail = proposal
        accept = change <= 0 or self.rng.random() < math.exp(-change / temperature)
        if not accept:
            self._undo(trail)
            return False
        self.current_cost += change
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best = self.current.copy()
        return True

    def polish_rooms(self) -> None:
        """Re-solve rooms exactly and resynchronise the tracked cost.

        `reassign_rooms` is monotone, so this can only lower `current_cost`; a full `cost` call
        is still needed afterwards because the incremental tracking only ever accounts for the
        moves this class made, not a room-wide re-solve.
        """
        try:
            self.current = reassign_rooms(self.current, self.formulation)
        except RoomAssignmentError:  # pragma: no cover - the greedy rule already found rooms
            return
        self.current_cost = float(cost(self.instance, self.current, self.formulation).total)
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best = self.current.copy()


def anneal(
    solution: Solution, formulation: Formulation, config: AnnealConfig | None = None
) -> AnnealResult:
    """Improve a complete, feasible timetable by simulated annealing over its period assignment.

    Args:
        solution: The timetable to improve. Must be complete; copied, so the caller's object is
            left untouched.
        formulation: Which components are active and at what weight.
        config: Hyperparameters for the run, or `None` for `AnnealConfig()`'s defaults.

    Returns:
        The best timetable found and how the search got there.
    """
    config = config or AnnealConfig()
    started = time.perf_counter()
    rng = random.Random(config.seed)  # noqa: S311 - move proposals, not cryptography
    search = _Search(solution, formulation, rng)
    initial_cost = search.current_cost
    temperature = search.calibrate()

    accepted = 0
    iteration = 0
    while iteration < config.iterations and (
        config.time_limit is None or time.perf_counter() - started < config.time_limit
    ):
        iteration += 1
        if search.step(temperature):
            accepted += 1
            if accepted % config.reassign_every == 0:
                search.polish_rooms()
        temperature *= config.cooling

    # The search may have wandered away from its best point since the last periodic polish
    # (that is the whole purpose of accepting worsening moves), so the final oracle pass has to
    # target `best` explicitly rather than whatever `current` happens to be when the loop ends.
    search.current, search.current_cost = search.best, search.best_cost
    search.polish_rooms()
    return AnnealResult(
        solution=search.best,
        initial_cost=initial_cost,
        best_cost=search.best_cost,
        iterations=iteration,
        accepted=accepted,
        seconds=time.perf_counter() - started,
    )
