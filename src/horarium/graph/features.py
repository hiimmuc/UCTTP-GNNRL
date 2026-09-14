"""Node, edge and global feature vectors, with per-instance normalization.

Each feature is declared once, as a spec carrying its own extractor. The matrices and
`docs/feature_dictionary.md` are both generated from those declarations, so the documentation
cannot drift from the code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from horarium.graph.build import TEACHER, build_conflict_graph
from horarium.problem.instance import Instance

__all__ = [
    "EDGE_FEATURES",
    "GLOBAL_FEATURES",
    "NODE_FEATURES",
    "FeatureMatrix",
    "InstanceFeatures",
    "encode_instance",
    "feature_dictionary_markdown",
]

Array = npt.NDArray[np.float32]
normalization = Literal["zscore", "log1p", "raw"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Feature(Generic[T]):
    """One named feature: how to compute it, how to scale it, and what it means."""

    name: str
    normalization: normalization
    description: str
    extract: T


NodeExtractor = Callable[[Instance, int], float]
EdgeExtractor = Callable[[Instance, int, int, int], float]
GlobalExtractor = Callable[[Instance, "nx.Graph[int]"], float]


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A feature block kept in both raw and scaled form; only `values` goes into a model."""

    names: tuple[str, ...]
    raw: Array
    values: Array


@dataclass(frozen=True, slots=True)
class InstanceFeatures:
    """Everything an encoder needs for one instance."""

    instance_name: str
    nodes: FeatureMatrix
    edges: FeatureMatrix
    globals: FeatureMatrix
    edge_index: npt.NDArray[np.int64]

    @property
    def n_nodes(self) -> int:
        """Number of course nodes."""
        return int(self.nodes.values.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of directed edges; each conflict appears once in each direction."""
        return int(self.edge_index.shape[1])


def _teacher_load(instance: Instance, course: int) -> float:
    """Total lectures taught by this course's teacher, across all their courses.

    Args:
        instance: The instance the course belongs to.
        course: Course index.

    Returns:
        Sum of `n_lectures` over every course taught by this course's teacher.
    """
    peers = instance.courses_of_teacher[instance.courses[course].teacher]
    return float(sum(instance.courses[p].n_lectures for p in peers))


def _largest_permitted_room(instance: Instance, course: int) -> float:
    """Capacity of the biggest room the course is allowed to use.

    Args:
        instance: The instance the course belongs to.
        course: Course index.

    Returns:
        The largest capacity among `instance.permitted_rooms[course]`, or 0 if none.
    """
    return float(
        max((instance.rooms[r].capacity for r in instance.permitted_rooms[course]), default=0)
    )


def _curriculum_load(instance: Instance, course: int) -> float:
    """Total lectures demanded by every curriculum this course belongs to.

    Args:
        instance: The instance the course belongs to.
        course: Course index.

    Returns:
        Sum of `n_lectures` over every course in every curriculum this course belongs to.
    """
    return float(
        sum(
            instance.courses[member].n_lectures
            for q in instance.curricula_of_course[course]
            for member in instance.courses_of_curriculum[q]
        )
    )


NODE_FEATURES: tuple[Feature[NodeExtractor], ...] = (
    Feature(
        "n_lectures",
        "zscore",
        "Lectures the course must schedule.",
        lambda i, c: float(i.courses[c].n_lectures),
    ),
    Feature(
        "min_working_days",
        "zscore",
        "Distinct days the lectures should spread over.",
        lambda i, c: float(i.courses[c].min_working_days),
    ),
    Feature(
        "n_students",
        "zscore",
        "Students attending, used by the room-capacity cost.",
        lambda i, c: float(i.courses[c].n_students),
    ),
    Feature(
        "double_lectures",
        "raw",
        "1 if the course prefers adjacent same-room pairs.",
        lambda i, c: float(i.courses[c].double_lectures),
    ),
    Feature(
        "availability_ratio",
        "raw",
        "Fraction of periods the course may be taught in.",
        lambda i, c: len(i.available_periods[c]) / i.n_periods,
    ),
    Feature(
        "permitted_room_ratio",
        "raw",
        "Fraction of rooms the course is allowed to use.",
        lambda i, c: len(i.permitted_rooms[c]) / i.n_rooms,
    ),
    Feature(
        "n_curricula",
        "zscore",
        "Curricula the course belongs to.",
        lambda i, c: float(len(i.curricula_of_course[c])),
    ),
    Feature(
        "conflict_degree",
        "zscore",
        "Courses that may not share a period with this one.",
        lambda i, c: float(len(i.conflicts[c])),
    ),
    Feature(
        "conflict_ratio",
        "raw",
        "Conflict degree over the largest degree possible.",
        lambda i, c: len(i.conflicts[c]) / max(1, i.n_courses - 1),
    ),
    Feature(
        "teacher_load",
        "zscore",
        "Lectures the course's teacher must give in total.",
        _teacher_load,
    ),
    Feature(
        "largest_permitted_room",
        "zscore",
        "Capacity of the biggest allowed room.",
        _largest_permitted_room,
    ),
    Feature(
        "capacity_slack",
        "zscore",
        "Largest allowed room capacity minus students.",
        lambda i, c: _largest_permitted_room(i, c) - i.courses[c].n_students,
    ),
    Feature(
        "capacity_feasible",
        "raw",
        "1 if some allowed room can seat every student.",
        lambda i, c: float(_largest_permitted_room(i, c) >= i.courses[c].n_students),
    ),
    Feature(
        "lecture_pressure",
        "raw",
        "Lectures divided by the periods still available.",
        lambda i, c: i.courses[c].n_lectures / max(1, len(i.available_periods[c])),
    ),
    Feature(
        "min_working_days_ratio",
        "raw",
        "Required working days over the days available.",
        lambda i, c: i.courses[c].min_working_days / i.days,
    ),
    Feature(
        "curriculum_load",
        "zscore",
        "Lectures demanded by all curricula containing it.",
        _curriculum_load,
    ),
)

EDGE_FEATURES: tuple[Feature[EdgeExtractor], ...] = (
    Feature(
        "shared_curricula",
        "zscore",
        "Curricula both endpoints belong to.",
        lambda i, u, v, shared: float(shared),
    ),
    Feature(
        "same_teacher",
        "raw",
        "1 if the two courses share a teacher.",
        lambda i, u, v, shared: float(i.courses[u].teacher == i.courses[v].teacher),
    ),
    Feature(
        "combined_lectures",
        "zscore",
        "Lectures the two endpoints need between them.",
        lambda i, u, v, shared: float(i.courses[u].n_lectures + i.courses[v].n_lectures),
    ),
    Feature(
        "period_overlap",
        "raw",
        "Periods available to both, over all periods.",
        lambda i, u, v, shared: (
            len(set(i.available_periods[u]) & set(i.available_periods[v])) / i.n_periods
        ),
    ),
    Feature(
        "room_overlap",
        "raw",
        "Rooms permitted to both, over all rooms.",
        lambda i, u, v, shared: (
            len(set(i.permitted_rooms[u]) & set(i.permitted_rooms[v])) / i.n_rooms
        ),
    ),
    Feature(
        "min_endpoint_slack",
        "raw",
        "Smaller endpoint's availability ratio.",
        lambda i, u, v, shared: (
            min(len(i.available_periods[u]), len(i.available_periods[v])) / i.n_periods
        ),
    ),
)

GLOBAL_FEATURES: tuple[Feature[GlobalExtractor], ...] = (
    Feature("n_courses", "log1p", "Courses in the instance.", lambda i, g: float(i.n_courses)),
    Feature("n_rooms", "log1p", "Rooms available.", lambda i, g: float(i.n_rooms)),
    Feature("n_curricula", "log1p", "Curricula declared.", lambda i, g: float(i.n_curricula)),
    Feature("days", "raw", "Teaching days per week.", lambda i, g: float(i.days)),
    Feature("periods_per_day", "raw", "Timeslots per day.", lambda i, g: float(i.periods_per_day)),
    Feature("n_periods", "log1p", "Total timeslots.", lambda i, g: float(i.n_periods)),
    Feature("total_lectures", "log1p", "Lectures to place.", lambda i, g: float(i.total_lectures)),
    Feature(
        "room_period_density",
        "raw",
        "Lectures over room-period slots.",
        lambda i, g: i.total_lectures / (i.n_rooms * i.n_periods),
    ),
    Feature(
        "mean_conflict_degree",
        "log1p",
        "Average conflict-graph degree.",
        lambda i, g: 2 * g.number_of_edges() / max(1, i.n_courses),
    ),
    Feature(
        "conflict_density",
        "raw",
        "Conflict edges over all possible pairs.",
        lambda i, g: nx.density(g),
    ),
    Feature(
        "teacher_edge_fraction",
        "raw",
        "Conflict edges that a shared teacher explains.",
        lambda i, g: (
            sum(1 for _, _, d in g.edges(data=True) if TEACHER in d["sources"])
            / max(1, g.number_of_edges())
        ),
    ),
    Feature(
        "unavailability_density",
        "raw",
        "Blocked course-periods over all course-periods.",
        lambda i, g: sum(len(s) for s in i.unavailable) / (i.n_courses * i.n_periods),
    ),
    Feature(
        "room_constraint_density",
        "raw",
        "Forbidden course-rooms over all course-rooms.",
        lambda i, g: sum(len(s) for s in i.forbidden_rooms) / (i.n_courses * i.n_rooms),
    ),
    Feature(
        "mean_curriculum_size",
        "log1p",
        "Courses per curriculum.",
        lambda i, g: sum(len(q.course_ids) for q in i.curricula) / max(1, i.n_curricula),
    ),
    Feature(
        "mean_lectures_per_course",
        "raw",
        "Lectures per course.",
        lambda i, g: i.total_lectures / i.n_courses,
    ),
    Feature(
        "min_daily_lectures",
        "raw",
        "Lower end of the student-load band.",
        lambda i, g: float(i.min_daily_lectures),
    ),
    Feature(
        "max_daily_lectures",
        "raw",
        "Upper end of the student-load band.",
        lambda i, g: float(i.max_daily_lectures),
    ),
    Feature(
        "double_lecture_fraction",
        "raw",
        "Courses that prefer adjacent pairs.",
        lambda i, g: sum(1 for c in i.courses if c.double_lectures) / i.n_courses,
    ),
)


def _scale(raw: Array, normalizations: tuple[normalization, ...]) -> Array:
    """Apply each column's declared normalization.

    A z-scored column with zero variance becomes all zeros rather than NaN, which is the
    only sane reading of "this feature is constant across this instance".

    Args:
        raw: Feature matrix, one column per feature.
        normalizations: The normalization to apply to each column, in column order.

    Returns:
        A new matrix of the same shape with each column normalised.
    """
    scaled = raw.copy()
    for column, how in enumerate(normalizations):
        values = scaled[:, column]
        if how == "zscore":
            spread = float(values.std())
            scaled[:, column] = (
                (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
            )
        elif how == "log1p":
            scaled[:, column] = np.log1p(np.maximum(values, 0.0))
    return scaled


def encode_instance(instance: Instance, *, include_teacher: bool = True) -> InstanceFeatures:
    """Build the conflict graph and every feature block for one instance.

    Node and edge features are z-scored within this instance; globals are log-scaled or raw,
    since there is only one global row to standardise against.

    Args:
        instance: The instance to encode.
        include_teacher: Whether a shared teacher also creates a conflict edge.

    Returns:
        Node, edge and global feature blocks plus the directed edge index.
    """
    graph = build_conflict_graph(instance, include_teacher=include_teacher)

    node_raw = np.array(
        [[f.extract(instance, c) for f in NODE_FEATURES] for c in range(instance.n_courses)],
        dtype=np.float32,
    ).reshape(instance.n_courses, len(NODE_FEATURES))

    pairs = sorted((u, v) for u, v in graph.edges())
    edge_raw = np.array(
        [
            [
                f.extract(instance, u, v, graph.edges[u, v]["shared_curricula"])
                for f in EDGE_FEATURES
            ]
            for u, v in pairs
        ],
        dtype=np.float32,
    ).reshape(len(pairs), len(EDGE_FEATURES))

    global_raw = np.array([[f.extract(instance, graph) for f in GLOBAL_FEATURES]], dtype=np.float32)

    # Undirected conflicts, emitted in both directions so message passing sees each edge twice.
    both_ways = np.array([[u, v] for u, v in pairs] + [[v, u] for u, v in pairs], dtype=np.int64)
    edge_index = both_ways.T.reshape(2, 2 * len(pairs))
    edge_values = _scale(edge_raw, tuple(f.normalization for f in EDGE_FEATURES))

    return InstanceFeatures(
        instance_name=instance.name,
        nodes=FeatureMatrix(
            names=tuple(f.name for f in NODE_FEATURES),
            raw=node_raw,
            values=_scale(node_raw, tuple(f.normalization for f in NODE_FEATURES)),
        ),
        edges=FeatureMatrix(
            names=tuple(f.name for f in EDGE_FEATURES),
            raw=np.concatenate([edge_raw, edge_raw]),
            values=np.concatenate([edge_values, edge_values]),
        ),
        globals=FeatureMatrix(
            names=tuple(f.name for f in GLOBAL_FEATURES),
            raw=global_raw,
            values=_scale(global_raw, tuple(f.normalization for f in GLOBAL_FEATURES)),
        ),
        edge_index=edge_index,
    )


def feature_dictionary_markdown() -> str:
    """Render the feature declarations above as a Markdown reference table.

    Returns:
        The complete Markdown document, ending in a newline.
    """
    sections: list[tuple[str, tuple[Feature[Any], ...]]] = [
        ("Node features (one row per course)", NODE_FEATURES),
        ("Edge features (one row per directed conflict edge)", EDGE_FEATURES),
        ("Global features (one row per instance)", GLOBAL_FEATURES),
    ]
    lines = [
        "# Feature dictionary",
        "",
        "Generated by `python -m horarium.cli.prepare`. Do not edit by hand.",
        "",
        "All features are `float32`. `zscore` is standardised within the instance, so a "
        "constant column becomes zeros rather than NaN. `log1p` is `log(1 + max(x, 0))`. "
        "`raw` values are already bounded ratios or flags.",
    ]
    for title, features in sections:
        lines += ["", f"## {title}", "", "| name | normalization | meaning |", "|---|---|---|"]
        lines += [f"| `{f.name}` | `{f.normalization}` | {f.description} |" for f in features]
    return "\n".join([*lines, ""])
