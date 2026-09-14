"""Shared machinery for the timetabling environments: observations, masking, bookkeeping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from horarium.graph.features import InstanceFeatures, encode_instance
from horarium.problem.formulations import Formulation
from horarium.problem.instance import Instance
from horarium.problem.solution import Solution

__all__ = ["DYNAMIC_NODE_FEATURES", "PERIOD_FEATURES", "EpisodeStats", "TimetablingEnv"]

Array = npt.NDArray[np.float32]

#: Per-course state that changes as an episode runs, appended to the static node features.
DYNAMIC_NODE_FEATURES = (
    "placed_fraction",
    "remaining_fraction",
    "working_days_fraction",
    "legal_periods_fraction",
    "is_complete",
)

#: Per-period state, the other half of every (course, period) action.
PERIOD_FEATURES = (
    "day_position",
    "slot_position",
    "room_occupancy",
    "curriculum_occupancy",
    "is_first_slot_of_day",
    "is_last_slot_of_day",
)


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """What an episode achieved, reported in `info` on the terminal step."""

    placed: int
    required: int
    steps: int
    dead_end: bool
    total_cost: float
    violations: int

    @property
    def feasible(self) -> bool:
        """True when every lecture was placed and no hard constraint is violated."""
        return self.placed == self.required and self.violations == 0


class TimetablingEnv(gym.Env[dict[str, Any], np.int64], ABC):
    """Base for timetabling environments.

    Subclasses decide what an action means and when an episode ends; this class owns the
    instance, the static features, the observation layout and the episode counters. The
    action space is always flat with a boolean mask, because the set of legal actions changes
    every step and depends on the instance.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 - Gymnasium declares this mutable

    def __init__(self, instance: Instance, formulation: Formulation) -> None:
        """Precompute the static features and declare the observation and action spaces.

        Args:
            instance: The instance to build an environment for.
            formulation: Which cost components are active and at what weight.
        """
        self.instance = instance
        self.formulation = formulation
        self.features: InstanceFeatures = encode_instance(instance)
        self.solution = Solution(instance)
        self.steps = 0
        self._mask: npt.NDArray[np.bool_] | None = None

        n_courses, n_periods = instance.n_courses, instance.n_periods
        self.node_feature_dim = self.features.nodes.values.shape[1] + len(DYNAMIC_NODE_FEATURES)
        self.period_feature_dim = len(PERIOD_FEATURES)
        self.global_feature_dim = int(self.features.globals.values.shape[1])
        self.edge_feature_dim = int(self.features.edges.values.shape[1])
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict(
            {
                "nodes": _box(n_courses, self.node_feature_dim),
                "periods": _box(n_periods, self.period_feature_dim),
                "globals": _box(1, self.global_feature_dim),
                "action_mask": spaces.MultiBinary(self.n_actions),
            }
        )

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Size of the flat action space.

        Returns:
            The number of discrete actions.
        """

    def action_mask(self) -> npt.NDArray[np.bool_]:
        """True at every action that is legal right now.

        An action is legal only if taking it cannot violate a hard constraint, so a masked
        rollout can never produce an infeasible solution. Recomputing the mask dominates the
        step cost, so it is cached and invalidated by `invalidate_mask` whenever the solution
        changes.

        Returns:
            A boolean array of shape `(n_actions,)`.
        """
        if self._mask is None:
            self._mask = self._compute_action_mask()
        return self._mask

    def invalidate_mask(self) -> None:
        """Drop the cached mask. Call after any change to the solution."""
        self._mask = None

    @abstractmethod
    def _compute_action_mask(self) -> npt.NDArray[np.bool_]:
        """Build the legality mask from scratch.

        Returns:
            A boolean array of shape `(n_actions,)`.
        """

    @abstractmethod
    def _take(self, action: int) -> float:
        """Apply an action and return its reward.

        Args:
            action: A legal, flat action index.

        Returns:
            The reward for taking this action.
        """

    @abstractmethod
    def episode_stats(self, *, dead_end: bool) -> EpisodeStats:
        """Summarise the finished episode.

        Args:
            dead_end: Whether the episode ended because no legal action remained.

        Returns:
            What the episode achieved.
        """

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh episode on an empty timetable.

        Args:
            seed: Seed for the episode's random number generator.
            options: Unused; part of the Gymnasium `reset` signature.

        Returns:
            The initial observation and an empty info dict.
        """
        del options
        super().reset(seed=seed)
        self.solution = Solution(self.instance)
        self.steps = 0
        self.invalidate_mask()
        return self.observe(), {}

    def observe(self) -> dict[str, Any]:
        """Assemble the observation: static features concatenated with the live episode state.

        Returns:
            A dict with keys "nodes", "periods", "globals" and "action_mask".
        """
        return {
            "nodes": np.concatenate(
                [self.features.nodes.values, self._dynamic_node_features()], axis=1
            ),
            "periods": self._period_features(),
            "globals": self.features.globals.values,
            "action_mask": self.action_mask(),
        }

    def _dynamic_node_features(self) -> Array:
        """Per-course progress, all scaled into [0, 1].

        Returns:
            Array of shape `(n_courses, len(DYNAMIC_NODE_FEATURES))`.
        """
        instance, solution = self.instance, self.solution
        rows = np.zeros((instance.n_courses, len(DYNAMIC_NODE_FEATURES)), dtype=np.float32)
        legal = self._legal_period_counts()
        for c, course in enumerate(instance.courses):
            placed = solution.n_scheduled[c]
            rows[c] = (
                placed / course.n_lectures,
                (course.n_lectures - placed) / course.n_lectures,
                solution.working_days[c] / instance.days,
                legal[c] / instance.n_periods,
                float(placed >= course.n_lectures),
            )
        return rows

    def _period_features(self) -> Array:
        """Per-period position and how contended it currently is.

        Returns:
            Array of shape `(n_periods, len(PERIOD_FEATURES))`.
        """
        instance, solution = self.instance, self.solution
        rows = np.zeros((instance.n_periods, len(PERIOD_FEATURES)), dtype=np.float32)
        curriculum_span = max(1, instance.n_curricula)
        for p in range(instance.n_periods):
            slot = instance.slot_of(p)
            rows[p] = (
                instance.day_of(p) / max(1, instance.days - 1),
                slot / max(1, instance.periods_per_day - 1),
                len(solution.courses_at_period[p]) / instance.n_rooms,
                solution.busy_curricula_at_period[p] / curriculum_span,
                float(slot == 0),
                float(slot == instance.periods_per_day - 1),
            )
        return rows

    @abstractmethod
    def _legal_period_counts(self) -> list[int]:
        """How many periods each course could still legally use.

        Returns:
            One count per course, in course order.
        """


def _box(rows: int, columns: int) -> spaces.Box:
    """Unbounded float32 observation block.

    Args:
        rows: Number of rows in the block.
        columns: Number of columns in the block.

    Returns:
        A Gymnasium `Box` space of shape `(rows, columns)`.
    """
    return spaces.Box(low=-np.inf, high=np.inf, shape=(rows, columns), dtype=np.float32)
