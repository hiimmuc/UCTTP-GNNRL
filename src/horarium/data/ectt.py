"""Reader and writer for the .ectt curriculum-based course timetabling format."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn

from horarium.problem.instance import Course, Curriculum, Instance, Room

__all__ = ["EcttParseError", "read_ectt", "write_ectt"]


class EcttParseError(ValueError):
    """The file does not follow the .ectt grammar, or its declared counts do not hold."""


class _Tokens:
    """Whitespace-separated token cursor, mirroring how the reference C++ validator reads."""

    def __init__(self, text: str, source: Path) -> None:
        self._tokens = text.split()
        self._at = 0
        self.source = source

    def fail(self, message: str) -> NoReturn:
        """Raise a parse error naming the file being read.

        Args:
            message: What went wrong, without the file name (added automatically).

        Raises:
            EcttParseError: Always.
        """
        raise EcttParseError(f"{self.source}: {message}")

    def take(self) -> str:
        """Consume and return the next token.

        Returns:
            The next token.

        Raises:
            EcttParseError: The stream is exhausted.
        """
        if self._at >= len(self._tokens):
            self.fail("file ended while more tokens were expected")
        token = self._tokens[self._at]
        self._at += 1
        return token

    def take_int(self) -> int:
        """Consume the next token as an integer.

        Returns:
            The token parsed as an integer.

        Raises:
            EcttParseError: The token is not an integer.
        """
        token = self.take()
        if not _is_int(token):
            self.fail(f"expected an integer, got {token!r}")
        return int(token)

    def expect(self, literal: str) -> None:
        """Consume the next token and require it to equal `literal`.

        Args:
            literal: The exact token expected next.

        Raises:
            EcttParseError: The token differs.
        """
        token = self.take()
        if token != literal:
            self.fail(f"expected {literal!r}, got {token!r}")

    def remaining(self) -> int:
        """Number of tokens not yet consumed.

        Returns:
            The count of unconsumed tokens.
        """
        return len(self._tokens) - self._at


def read_ectt(path: str | Path) -> Instance:
    """Parse an .ectt file into an Instance.

    Section order, the declared header counts and the trailing `END.` marker are all enforced.
    The room-constraint section lists (course, room) pairs the course may *not* use, so it
    populates `Instance.forbidden_rooms`.

    Args:
        path: Path to the .ectt file.

    Returns:
        The parsed instance.

    Raises:
        EcttParseError: The grammar, the counts or the referenced ids are wrong.
    """
    source = Path(path)
    tok = _Tokens(source.read_text(), source)

    tok.expect("Name:")
    name = tok.take()
    tok.expect("Courses:")
    n_courses = tok.take_int()
    tok.expect("Rooms:")
    n_rooms = tok.take_int()
    tok.expect("Days:")
    days = tok.take_int()
    tok.expect("Periods_per_day:")
    periods_per_day = tok.take_int()
    tok.expect("Curricula:")
    n_curricula = tok.take_int()
    tok.expect("Min_Max_Daily_Lectures:")
    min_daily = tok.take_int()
    max_daily = tok.take_int()
    tok.expect("UnavailabilityConstraints:")
    n_unavailability = tok.take_int()
    tok.expect("RoomConstraints:")
    n_room_constraints = tok.take_int()

    tok.expect("COURSES:")
    courses = tuple(
        Course(
            id=tok.take(),
            teacher=tok.take(),
            n_lectures=tok.take_int(),
            min_working_days=tok.take_int(),
            n_students=tok.take_int(),
            double_lectures=_flag(tok, tok.take_int()),
        )
        for _ in range(n_courses)
    )

    tok.expect("ROOMS:")
    rooms = tuple(
        Room(id=tok.take(), capacity=tok.take_int(), building=tok.take_int())
        for _ in range(n_rooms)
    )

    tok.expect("CURRICULA:")
    curricula = []
    for _ in range(n_curricula):
        curriculum_id = tok.take()
        size = tok.take_int()
        curricula.append(
            Curriculum(id=curriculum_id, course_ids=tuple(tok.take() for _ in range(size)))
        )

    course_index = _index_of(tok, (c.id for c in courses), "course")
    room_index = _index_of(tok, (r.id for r in rooms), "room")
    for curriculum in curricula:
        for course_id in curriculum.course_ids:
            if course_id not in course_index:
                tok.fail(f"curriculum {curriculum.id!r} names unknown course {course_id!r}")
    n_periods = days * periods_per_day

    unavailable = _read_unavailability(tok, n_unavailability, course_index, days, periods_per_day)
    forbidden = _read_room_constraints(tok, n_room_constraints, course_index, room_index)

    tok.expect("END.")
    if tok.remaining():
        tok.fail(f"{tok.remaining()} unexpected tokens after 'END.'")
    if n_periods <= 0:
        tok.fail(f"days * periods_per_day must be positive, got {days} * {periods_per_day}")

    return Instance(
        name=name,
        courses=courses,
        rooms=rooms,
        curricula=tuple(curricula),
        days=days,
        periods_per_day=periods_per_day,
        min_daily_lectures=min_daily,
        max_daily_lectures=max_daily,
        unavailable=unavailable,
        forbidden_rooms=forbidden,
    )


def write_ectt(instance: Instance, path: str | Path) -> None:
    """Write an Instance back out in .ectt form.

    The result re-reads to an equal Instance; it is not byte-identical to the input, because
    constraint rows are emitted in course order with periods and rooms sorted.

    Args:
        instance: The instance to write.
        path: Destination path.
    """
    n_unavailability = sum(len(s) for s in instance.unavailable)
    n_room_constraints = sum(len(s) for s in instance.forbidden_rooms)

    lines = [
        f"Name: {instance.name}",
        f"Courses: {instance.n_courses}",
        f"Rooms: {instance.n_rooms}",
        f"Days: {instance.days}",
        f"Periods_per_day: {instance.periods_per_day}",
        f"Curricula: {instance.n_curricula}",
        f"Min_Max_Daily_Lectures: {instance.min_daily_lectures} {instance.max_daily_lectures}",
        f"UnavailabilityConstraints: {n_unavailability}",
        f"RoomConstraints: {n_room_constraints}",
        "",
        "COURSES:",
    ]
    lines += [
        f"{c.id} {c.teacher} {c.n_lectures} {c.min_working_days} {c.n_students} "
        f"{int(c.double_lectures)}"
        for c in instance.courses
    ]
    lines += ["", "ROOMS:"]
    lines += [f"{r.id} {r.capacity} {r.building}" for r in instance.rooms]
    lines += ["", "CURRICULA:"]
    lines += [f"{q.id} {len(q.course_ids)} " + " ".join(q.course_ids) for q in instance.curricula]
    lines += ["", "UNAVAILABILITY_CONSTRAINTS:"]
    lines += [
        f"{course.id} {period // instance.periods_per_day} {period % instance.periods_per_day}"
        for course, periods in zip(instance.courses, instance.unavailable, strict=True)
        for period in sorted(periods)
    ]
    lines += ["", "ROOM_CONSTRAINTS:"]
    lines += [
        f"{course.id} {instance.rooms[room].id}"
        for course, banned in zip(instance.courses, instance.forbidden_rooms, strict=True)
        for room in sorted(banned)
    ]
    lines += ["", "END.", ""]
    Path(path).write_text("\n".join(lines))


def _read_unavailability(
    tok: _Tokens,
    count: int,
    course_index: dict[str, int],
    days: int,
    periods_per_day: int,
) -> tuple[frozenset[int], ...]:
    """Read the unavailability section into per-course sets of flattened period indices.

    Args:
        tok: Token cursor positioned just before the section.
        count: Declared number of rows to read.
        course_index: Course id to index mapping.
        days: Number of days in the instance.
        periods_per_day: Number of periods per day in the instance.

    Returns:
        Per-course frozensets of flattened period indices the course may not use.

    Raises:
        EcttParseError: A row names an unknown course or an out-of-range period.
    """
    tok.expect("UNAVAILABILITY_CONSTRAINTS:")
    blocked: list[set[int]] = [set() for _ in course_index]
    for _ in range(count):
        course_id, day, slot = tok.take(), tok.take_int(), tok.take_int()
        if course_id not in course_index:
            tok.fail(f"unavailability names unknown course {course_id!r}")
        if not 0 <= day < days or not 0 <= slot < periods_per_day:
            tok.fail(f"unavailability period ({day}, {slot}) out of range for {course_id!r}")
        blocked[course_index[course_id]].add(day * periods_per_day + slot)
    return tuple(frozenset(s) for s in blocked)


def _read_room_constraints(
    tok: _Tokens,
    count: int,
    course_index: dict[str, int],
    room_index: dict[str, int],
) -> tuple[frozenset[int], ...]:
    """Read the room-constraint section into per-course sets of *forbidden* room indices.

    Args:
        tok: Token cursor positioned just before the section.
        count: Declared number of rows to read.
        course_index: Course id to index mapping.
        room_index: Room id to index mapping.

    Returns:
        Per-course frozensets of room indices the course may not use.

    Raises:
        EcttParseError: A row names an unknown course or room.
    """
    tok.expect("ROOM_CONSTRAINTS:")
    banned: list[set[int]] = [set() for _ in course_index]
    for _ in range(count):
        course_id, room_id = tok.take(), tok.take()
        if course_id not in course_index:
            tok.fail(f"room constraint names unknown course {course_id!r}")
        if room_id not in room_index:
            tok.fail(f"room constraint names unknown room {room_id!r}")
        banned[course_index[course_id]].add(room_index[room_id])
    return tuple(frozenset(s) for s in banned)


def _flag(tok: _Tokens, value: int) -> bool:
    """Interpret a 0/1 token as a boolean.

    Args:
        tok: Token cursor, used only to report a parse error with the file name attached.
        value: The integer value read.

    Returns:
        `True` for 1, `False` for 0.

    Raises:
        EcttParseError: The value is neither 0 nor 1.
    """
    if value not in (0, 1):
        tok.fail(f"expected a 0/1 flag, got {value}")
    return bool(value)


def _is_int(token: str) -> bool:
    """True when the token parses as a signed decimal integer.

    Args:
        token: The token to check.

    Returns:
        Whether `int(token)` would succeed.
    """
    return token.lstrip("-").isdigit()


def _index_of(tok: _Tokens, ids: Iterable[str], kind: str) -> dict[str, int]:
    """Map entity id to position, rejecting duplicates.

    Args:
        tok: Token cursor, used only to report a parse error with the file name attached.
        ids: Entity ids, in file order.
        kind: Entity kind, used only in the error message (e.g. "course").

    Returns:
        Mapping from id to its 0-indexed position in `ids`.

    Raises:
        EcttParseError: Two entities share an id.
    """
    index: dict[str, int] = {}
    for position, entity_id in enumerate(ids):
        if entity_id in index:
            tok.fail(f"duplicate {kind} id {entity_id!r}")
        index[entity_id] = position
    return index
