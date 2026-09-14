"""Immutable problem instance for curriculum-based course timetabling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["Course", "Curriculum", "Instance", "Room"]


@dataclass(frozen=True, slots=True)
class Course:
    """A course: a fixed number of lectures taught by one teacher to one student group."""

    id: str
    teacher: str
    n_lectures: int
    min_working_days: int
    n_students: int
    double_lectures: bool


@dataclass(frozen=True, slots=True)
class Room:
    """A room with a seat capacity, sited in a building (used only by the travel cost)."""

    id: str
    capacity: int
    building: int


@dataclass(frozen=True, slots=True)
class Curriculum:
    """A group of courses a cohort of students attends; no two may share a period."""

    id: str
    course_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Instance:
    """One timetabling instance, with the lookup tables every solver and env needs.

    Periods are flattened and 0-indexed: `period = day * periods_per_day + slot`.
    Courses, rooms and curricula are 0-indexed in the order they appear in the file.

    `unavailable[c]` holds periods course `c` may not use. `forbidden_rooms[c]` holds rooms it
    may not use, so an empty set means every room is permitted; `permitted_rooms[c]` is the
    complement, already materialised for action masking.

    `conflicts[c]` holds the courses that may not share a period with `c`, because they share a
    curriculum or a teacher. It is a set of courses, not of reasons: two courses conflicting for
    several reasons still conflict once.
    """

    name: str
    courses: tuple[Course, ...]
    rooms: tuple[Room, ...]
    curricula: tuple[Curriculum, ...]
    days: int
    periods_per_day: int
    min_daily_lectures: int
    max_daily_lectures: int
    unavailable: tuple[frozenset[int], ...]
    forbidden_rooms: tuple[frozenset[int], ...]

    course_index: Mapping[str, int] = field(init=False, compare=False, repr=False)
    room_index: Mapping[str, int] = field(init=False, compare=False, repr=False)
    curriculum_index: Mapping[str, int] = field(init=False, compare=False, repr=False)
    courses_of_teacher: Mapping[str, tuple[int, ...]] = field(init=False, compare=False, repr=False)
    courses_of_curriculum: tuple[tuple[int, ...], ...] = field(
        init=False, compare=False, repr=False
    )
    curricula_of_course: tuple[tuple[int, ...], ...] = field(init=False, compare=False, repr=False)
    conflicts: tuple[frozenset[int], ...] = field(init=False, compare=False, repr=False)
    permitted_rooms: tuple[tuple[int, ...], ...] = field(init=False, compare=False, repr=False)
    available_periods: tuple[tuple[int, ...], ...] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Materialise the derived lookup tables.

        Raises:
            ValueError: A curriculum names a course that does not exist.
        """
        set_ = object.__setattr__
        course_index = {c.id: i for i, c in enumerate(self.courses)}
        set_(self, "course_index", course_index)
        set_(self, "room_index", {r.id: i for i, r in enumerate(self.rooms)})
        set_(self, "curriculum_index", {q.id: i for i, q in enumerate(self.curricula)})

        teachers: dict[str, list[int]] = {}
        for i, course in enumerate(self.courses):
            teachers.setdefault(course.teacher, []).append(i)
        set_(self, "courses_of_teacher", {t: tuple(v) for t, v in teachers.items()})

        members: list[tuple[int, ...]] = []
        belongs: list[list[int]] = [[] for _ in self.courses]
        for q_index, curriculum in enumerate(self.curricula):
            row = []
            for course_id in curriculum.course_ids:
                if course_id not in course_index:
                    msg = f"curriculum {curriculum.id!r} names unknown course {course_id!r}"
                    raise ValueError(msg)
                c_index = course_index[course_id]
                row.append(c_index)
                belongs[c_index].append(q_index)
            members.append(tuple(row))
        set_(self, "courses_of_curriculum", tuple(members))
        set_(self, "curricula_of_course", tuple(tuple(v) for v in belongs))

        adjacency: list[set[int]] = [set() for _ in self.courses]
        for group in (*members, *teachers.values()):
            for a in group:
                adjacency[a].update(group)
        for a, neighbours in enumerate(adjacency):
            neighbours.discard(a)
        set_(self, "conflicts", tuple(frozenset(n) for n in adjacency))

        all_rooms = range(len(self.rooms))
        set_(
            self,
            "permitted_rooms",
            tuple(
                tuple(r for r in all_rooms if r not in banned) for banned in self.forbidden_rooms
            ),
        )
        all_periods = range(self.n_periods)
        set_(
            self,
            "available_periods",
            tuple(
                tuple(p for p in all_periods if p not in blocked) for blocked in self.unavailable
            ),
        )

    @property
    def n_courses(self) -> int:
        """Number of courses."""
        return len(self.courses)

    @property
    def n_rooms(self) -> int:
        """Number of rooms."""
        return len(self.rooms)

    @property
    def n_curricula(self) -> int:
        """Number of curricula."""
        return len(self.curricula)

    @property
    def n_periods(self) -> int:
        """Total number of periods, `days * periods_per_day`."""
        return self.days * self.periods_per_day

    @property
    def total_lectures(self) -> int:
        """Number of lectures that must be scheduled across all courses."""
        return sum(c.n_lectures for c in self.courses)

    def day_of(self, period: int) -> int:
        """Day index containing a flattened period index.

        Args:
            period: Flattened, 0-indexed period.

        Returns:
            The 0-indexed day the period falls on.
        """
        return period // self.periods_per_day

    def slot_of(self, period: int) -> int:
        """Position of a flattened period index within its day.

        Args:
            period: Flattened, 0-indexed period.

        Returns:
            The 0-indexed slot within the day.
        """
        return period % self.periods_per_day
