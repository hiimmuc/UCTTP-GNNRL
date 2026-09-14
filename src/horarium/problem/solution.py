"""A timetable assignment plus the incremental counters that make delta evaluation cheap."""

from __future__ import annotations

from dataclasses import dataclass

from horarium.problem.instance import Instance

__all__ = ["EMPTY", "Move", "Solution"]

#: Sentinel room index for "this course has no lecture in this period".
EMPTY = -1


@dataclass(frozen=True, slots=True)
class Move:
    """Set the content of one (course, period) cell to `room`, or to EMPTY to clear it."""

    course: int
    period: int
    room: int

    def __post_init__(self) -> None:
        """Reject negative indices other than the EMPTY sentinel.

        Raises:
            ValueError: Course or period is negative, or room is neither EMPTY nor >= 0.
        """
        if self.course < 0 or self.period < 0 or self.room < EMPTY:
            msg = f"invalid move {self!r}"
            raise ValueError(msg)


class Solution:
    """A (course, period) -> room table with the counters every cost component needs.

    Every counter is maintained by `place` and `clear`; nothing recomputes them from scratch.
    Mutating `room_at` directly desynchronises the solution and is not supported.
    """

    __slots__ = (
        "busy_curricula_at_period",
        "course_daily_lectures",
        "courses_at_period",
        "curriculum_daily_lectures",
        "curriculum_period_lectures",
        "instance",
        "n_scheduled",
        "room_at",
        "room_period_load",
        "rooms_of_course",
        "working_days",
    )

    def __init__(self, instance: Instance) -> None:
        """Create an empty timetable for `instance`.

        Args:
            instance: The instance to build the timetable for.
        """
        n_c, n_p, n_r = instance.n_courses, instance.n_periods, instance.n_rooms
        n_q, n_d = instance.n_curricula, instance.days
        self.instance = instance
        self.room_at: list[list[int]] = [[EMPTY] * n_p for _ in range(n_c)]
        self.room_period_load: list[list[int]] = [[0] * n_p for _ in range(n_r)]
        self.course_daily_lectures: list[list[int]] = [[0] * n_d for _ in range(n_c)]
        self.curriculum_period_lectures: list[list[int]] = [[0] * n_p for _ in range(n_q)]
        self.curriculum_daily_lectures: list[list[int]] = [[0] * n_d for _ in range(n_q)]
        self.rooms_of_course: list[dict[int, int]] = [{} for _ in range(n_c)]
        self.courses_at_period: list[set[int]] = [set() for _ in range(n_p)]
        self.busy_curricula_at_period: list[int] = [0] * n_p
        self.working_days: list[int] = [0] * n_c
        self.n_scheduled: list[int] = [0] * n_c

    def place(self, course: int, period: int, room: int) -> None:
        """Put a lecture of `course` in `period` and `room`.

        Args:
            course: Course index.
            period: Flattened period index.
            room: Room index.

        Raises:
            ValueError: The cell is already occupied.
        """
        if self.room_at[course][period] != EMPTY:
            msg = f"course {course} already has a lecture in period {period}"
            raise ValueError(msg)
        instance = self.instance
        day = period // instance.periods_per_day

        self.room_at[course][period] = room
        self.room_period_load[room][period] += 1
        self.rooms_of_course[course][room] = self.rooms_of_course[course].get(room, 0) + 1
        self.courses_at_period[period].add(course)
        self.n_scheduled[course] += 1
        if self.course_daily_lectures[course][day] == 0:
            self.working_days[course] += 1
        self.course_daily_lectures[course][day] += 1
        for q in instance.curricula_of_course[course]:
            if self.curriculum_period_lectures[q][period] == 0:
                self.busy_curricula_at_period[period] += 1
            self.curriculum_period_lectures[q][period] += 1
            self.curriculum_daily_lectures[q][day] += 1

    def clear(self, course: int, period: int) -> int:
        """Remove the lecture of `course` in `period` and return the room it used.

        Args:
            course: Course index.
            period: Flattened period index.

        Returns:
            The room index the cleared cell used.

        Raises:
            ValueError: The cell is already empty.
        """
        room = self.room_at[course][period]
        if room == EMPTY:
            msg = f"course {course} has no lecture in period {period}"
            raise ValueError(msg)
        instance = self.instance
        day = period // instance.periods_per_day

        self.room_at[course][period] = EMPTY
        self.room_period_load[room][period] -= 1
        remaining = self.rooms_of_course[course][room] - 1
        if remaining:
            self.rooms_of_course[course][room] = remaining
        else:
            del self.rooms_of_course[course][room]
        self.courses_at_period[period].discard(course)
        self.n_scheduled[course] -= 1
        self.course_daily_lectures[course][day] -= 1
        if self.course_daily_lectures[course][day] == 0:
            self.working_days[course] -= 1
        for q in instance.curricula_of_course[course]:
            self.curriculum_period_lectures[q][period] -= 1
            self.curriculum_daily_lectures[q][day] -= 1
            if self.curriculum_period_lectures[q][period] == 0:
                self.busy_curricula_at_period[period] -= 1
        return room

    def apply(self, move: Move) -> None:
        """Overwrite one cell: clear whatever is there, then place the move's room if any.

        Args:
            move: The cell to overwrite and the room to put there.
        """
        if self.room_at[move.course][move.period] != EMPTY:
            self.clear(move.course, move.period)
        if move.room != EMPTY:
            self.place(move.course, move.period, move.room)

    def undo(self, move: Move, previous_room: int) -> None:
        """Reverse `apply`, restoring the room that occupied the cell beforehand.

        Args:
            move: The move that was applied.
            previous_room: The room `move`'s cell held before it was applied.
        """
        self.apply(Move(course=move.course, period=move.period, room=previous_room))

    def copy(self) -> Solution:
        """Deep copy of the assignment and every counter.

        Returns:
            An independent copy that shares no mutable state with this solution.
        """
        clone = Solution(self.instance)
        clone.room_at = [row[:] for row in self.room_at]
        clone.room_period_load = [row[:] for row in self.room_period_load]
        clone.course_daily_lectures = [row[:] for row in self.course_daily_lectures]
        clone.curriculum_period_lectures = [row[:] for row in self.curriculum_period_lectures]
        clone.curriculum_daily_lectures = [row[:] for row in self.curriculum_daily_lectures]
        clone.rooms_of_course = [dict(d) for d in self.rooms_of_course]
        clone.courses_at_period = [set(s) for s in self.courses_at_period]
        clone.busy_curricula_at_period = self.busy_curricula_at_period[:]
        clone.working_days = self.working_days[:]
        clone.n_scheduled = self.n_scheduled[:]
        return clone

    @property
    def is_complete(self) -> bool:
        """True when every course has exactly the number of lectures it requires."""
        return all(
            self.n_scheduled[c] == course.n_lectures
            for c, course in enumerate(self.instance.courses)
        )

    def lectures(self) -> list[tuple[int, int, int]]:
        """Every scheduled lecture as (course, period, room), in course then period order.

        Returns:
            One (course, period, room) tuple per scheduled lecture.
        """
        return [
            (course, period, room)
            for course, row in enumerate(self.room_at)
            for period, room in enumerate(row)
            if room != EMPTY
        ]
