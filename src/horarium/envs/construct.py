"""Constructive environment: place one lecture per step until the timetable is complete."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from horarium.problem.cost import cost, delta, soft_cost_upper_bound
from horarium.problem.formulations import Formulation
from horarium.problem.instance import Instance
from horarium.problem.solution import Move
from horarium.solvers.rooms import cheapest_room, usable_rooms

from .base import EpisodeStats, TimetablingEnv

__all__ = ["ConstructEnv"]


class ConstructEnv(TimetablingEnv):
    """One step places one lecture of one course into one period.

    The action is a flat `(course, period)` index, exactly the space the mask is defined over.
    A room is chosen for the placement by a deterministic rule -- the cheapest permitted room
    that is free, preferring one the course already uses -- which is the placeholder the
    stage-2 room MIP replaces. That keeps the action space the size the problem calls for
    while every state remains a real, cost-evaluable partial solution.

    Reward is the negative change in weighted soft cost, divided by `soft_cost_upper_bound` so
    an episode's soft-cost term lies in [-1, 0]. The empty timetable is not free -- a course
    with no lectures placed already owes its full MinWorkingDays cost -- so an episode's return
    is the negative final cost offset by that constant, which leaves the ordering over solutions
    unchanged. A dead end -- no legal action while lectures remain -- ends the episode and is
    charged the same bound per unplaced lecture, so abandoning a lecture costs a full 1.0 and
    completing always beats dead-ending.
    """

    def __init__(self, instance: Instance, formulation: Formulation) -> None:
        """Set up the environment.

        Args:
            instance: The instance to build an environment for.
            formulation: Which cost components are active and at what weight.
        """
        bound = float(soft_cost_upper_bound(instance, formulation))
        self.reward_scale = 1.0 / bound
        self.dead_end_penalty = bound

        n_courses, n_periods = instance.n_courses, instance.n_periods
        self._available = np.zeros((n_courses, n_periods), dtype=np.bool_)
        for c, periods in enumerate(instance.available_periods):
            self._available[c, list(periods)] = True
        self._conflicts_of = tuple(
            np.fromiter(sorted(peers), dtype=np.intp, count=len(peers))
            for peers in instance.conflicts
        )
        # A room a course is merely "unwanted" in is only off-limits where the formulation makes
        # RoomConstraints hard, which is UD4 alone. Everywhere else it is chargeable at worst, and
        # under UD2 it is free -- so restricting the mask to `permitted_rooms`, as this env used
        # to, threw away legal placements. On Erlangen that discarded ~70% of the rooms and is a
        # large part of why those instances dead-ended.
        self._usable_rooms = usable_rooms(instance, formulation)
        self._courses_of_room = tuple(
            np.fromiter(
                (c for c, rooms in enumerate(self._usable_rooms) if room in set(rooms)),
                dtype=np.intp,
            )
            for room in range(instance.n_rooms)
        )
        self._n_permitted = np.fromiter(
            (len(rooms) for rooms in self._usable_rooms), dtype=np.int32, count=n_courses
        )
        self._required = np.fromiter(
            (course.n_lectures for course in instance.courses), dtype=np.int32, count=n_courses
        )
        super().__init__(instance, formulation)
        self._reset_tracking()

    @property
    def n_actions(self) -> int:
        """One action per (course, period) pair.

        Returns:
            `n_courses * n_periods`.
        """
        return self.instance.n_courses * self.instance.n_periods

    def decode(self, action: int) -> tuple[int, int]:
        """Split a flat action into its course and period.

        Args:
            action: A flat action index.

        Returns:
            The (course, period) pair the action encodes.
        """
        return divmod(action, self.instance.n_periods)

    def _reset_tracking(self) -> None:
        """Restore the mask's incremental counters to their empty-timetable values."""
        shape = (self.instance.n_courses, self.instance.n_periods)
        self._occupied = np.zeros(shape, dtype=np.bool_)
        self._blocked = np.zeros(shape, dtype=np.int32)
        self._free_rooms = np.repeat(self._n_permitted[:, None], shape[1], axis=1)
        self._unplaced = self._required > 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh episode, resetting the mask counters alongside the solution.

        Args:
            seed: Seed for the episode's random number generator.
            options: Unused; part of the Gymnasium `reset` signature.

        Returns:
            The initial observation and an empty info dict.
        """
        self._reset_tracking()
        return super().reset(seed=seed, options=options)

    def _compute_action_mask(self) -> npt.NDArray[np.bool_]:
        """Legal (course, period) pairs: every hard constraint still holds after placing there.

        The four conditions are read off counters that `_take` maintains, so the whole mask is
        a handful of array comparisons rather than a loop over every (course, period, room).
        `_blocked[c, p]` counts conflicting courses already sitting in `p`, and
        `_free_rooms[c, p]` counts permitted rooms of `c` still free there.

        Returns:
            A boolean array of shape `(n_actions,)`.
        """
        legal = self._available & ~self._occupied
        legal &= self._blocked == 0
        legal &= self._free_rooms > 0
        legal &= self._unplaced[:, None]
        return legal.reshape(-1)

    def step(
        self, action: int | np.int64
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Place one lecture and return the usual Gymnasium tuple.

        Args:
            action: A legal, flat `(course, period)` action index.

        Returns:
            The (observation, reward, terminated, truncated, info) tuple. `info` carries
            `"episode_stats"` on the terminal step.

        Raises:
            ValueError: The action is masked out, which means the caller ignored the mask.
        """
        index = int(action)
        if not self.action_mask()[index]:
            course, period = self.decode(index)
            msg = f"action {index} (course {course}, period {period}) is not legal"
            raise ValueError(msg)

        reward = self._take(index)
        self.steps += 1
        remaining = self.solution.is_complete
        dead_end = not remaining and not self.action_mask().any()
        terminated = remaining or dead_end
        if dead_end:
            unplaced = sum(
                course.n_lectures - self.solution.n_scheduled[c]
                for c, course in enumerate(self.instance.courses)
            )
            reward -= self.dead_end_penalty * unplaced * self.reward_scale
        info: dict[str, Any] = {}
        if terminated:
            info["episode_stats"] = self.episode_stats(dead_end=dead_end)
        return self.observe(), reward, terminated, False, info

    def _take(self, action: int) -> float:
        """Place the lecture and charge the exact soft-cost increase the placement causes.

        Args:
            action: A legal, flat `(course, period)` action index.

        Returns:
            The reward for this placement.
        """
        course, period = self.decode(action)
        room = self._free_room(course, period)
        if room is None:  # pragma: no cover - the mask already excluded this
            msg = f"no free room for course {course} in period {period}"
            raise ValueError(msg)
        move = Move(course=course, period=period, room=room)
        increase = delta(self.instance, self.solution, move, self.formulation)
        self.solution.apply(move)

        self._occupied[course, period] = True
        self._blocked[self._conflicts_of[course], period] += 1
        self._free_rooms[self._courses_of_room[room], period] -= 1
        if self.solution.n_scheduled[course] >= self._required[course]:
            self._unplaced[course] = False

        self.invalidate_mask()
        return -increase * self.reward_scale

    def _free_room(self, course: int, period: int) -> int | None:
        """Cheapest usable room free in this period, preferring one the course already uses.

        Args:
            course: Course index.
            period: Flattened period index.

        Returns:
            The chosen room index, or `None` if nothing usable is free.
        """
        return cheapest_room(
            self.instance,
            self.solution,
            course,
            period,
            self.formulation,
            candidates=self._usable_rooms[course],
        )

    def _legal_period_counts(self) -> list[int]:
        """Legal periods per course, read straight off the current mask.

        Returns:
            One count per course, in course order.
        """
        mask = self.action_mask().reshape(self.instance.n_courses, self.instance.n_periods)
        counts: list[int] = mask.sum(axis=1).tolist()
        return counts

    def episode_stats(self, *, dead_end: bool) -> EpisodeStats:
        """Evaluate the finished timetable exactly, not from accumulated rewards.

        Args:
            dead_end: Whether the episode ended because no legal action remained.

        Returns:
            What the episode achieved.
        """
        breakdown = cost(self.instance, self.solution, self.formulation)
        return EpisodeStats(
            placed=sum(self.solution.n_scheduled),
            required=self.instance.total_lectures,
            steps=self.steps,
            dead_end=dead_end,
            total_cost=float(breakdown.total),
            violations=breakdown.violations,
        )
