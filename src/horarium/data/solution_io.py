"""Reader and writer for the one-lecture-per-line .sol solution format."""

from __future__ import annotations

from pathlib import Path

from horarium.problem.instance import Instance
from horarium.problem.solution import Solution

__all__ = ["SolutionParseError", "read_solution", "write_solution"]


class SolutionParseError(ValueError):
    """A solution line is malformed or names something the instance does not contain."""


def write_solution(instance: Instance, solution: Solution, path: str | Path) -> None:
    """Write `<course_id> <room_id> <day> <period>` per scheduled lecture, in course order.

    Args:
        instance: The instance the solution is for, used to translate indices to ids.
        solution: The solution to write.
        path: Destination path.
    """
    lines = [
        f"{instance.courses[course].id} {instance.rooms[room].id} "
        f"{instance.day_of(period)} {instance.slot_of(period)}"
        for course, period, room in solution.lectures()
    ]
    Path(path).write_text("\n".join([*lines, ""]))


def read_solution(instance: Instance, path: str | Path) -> Solution:
    """Parse a .sol file against an instance.

    Unlike the reference validator, which warns and skips, every malformed or unknown entry
    raises: a silently dropped lecture would make a comparison meaningless.

    Args:
        instance: The instance the solution must be checked against.
        path: Path to the .sol file.

    Returns:
        The parsed solution.

    Raises:
        SolutionParseError: A line has the wrong arity, names an unknown id, or repeats a cell.
    """
    source = Path(path)
    solution = Solution(instance)
    for number, raw in enumerate(source.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:  # noqa: PLR2004
            msg = f"{source}:{number}: expected 4 fields, got {len(fields)}"
            raise SolutionParseError(msg)
        course_id, room_id, day_text, slot_text = fields
        if course_id not in instance.course_index:
            msg = f"{source}:{number}: unknown course {course_id!r}"
            raise SolutionParseError(msg)
        if room_id not in instance.room_index:
            msg = f"{source}:{number}: unknown room {room_id!r}"
            raise SolutionParseError(msg)
        day, slot = _int(day_text, source, number), _int(slot_text, source, number)
        if not 0 <= day < instance.days or not 0 <= slot < instance.periods_per_day:
            msg = f"{source}:{number}: period ({day}, {slot}) outside the horizon"
            raise SolutionParseError(msg)
        course = instance.course_index[course_id]
        period = day * instance.periods_per_day + slot
        if solution.room_at[course][period] != -1:
            msg = f"{source}:{number}: repeated entry for {course_id!r} at ({day}, {slot})"
            raise SolutionParseError(msg)
        solution.place(course, period, instance.room_index[room_id])
    return solution


def _int(text: str, source: Path, number: int) -> int:
    """Parse an integer field.

    Args:
        text: The field to parse.
        source: File the field was read from, used only in the error message.
        number: Line number the field was read from, used only in the error message.

    Returns:
        The field parsed as an integer.

    Raises:
        SolutionParseError: The field is not an integer.
    """
    if not text.lstrip("-").isdigit():
        msg = f"{source}:{number}: expected an integer, got {text!r}"
        raise SolutionParseError(msg)
    return int(text)
