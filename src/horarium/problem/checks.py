"""Structural validation of an Instance, separating hard contract violations from oddities."""

from __future__ import annotations

from dataclasses import dataclass

from horarium.problem.instance import Instance

__all__ = ["CheckResult", "InstanceCheckError", "assert_valid", "check_instance"]


class InstanceCheckError(ValueError):
    """An instance violates a structural invariant the rest of the codebase relies on."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of `check_instance`.

    `errors` mean the instance is malformed or hard-infeasible before any solver runs.
    `warnings` mean it is well formed but unusual; several appear in the public corpus, so
    they must never be promoted to errors.
    """

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when there are no errors. Warnings do not affect this."""
        return not self.errors


def check_instance(instance: Instance) -> CheckResult:
    """Check dimensions, referential integrity, index ranges and per-course feasibility.

    Args:
        instance: The instance to validate.

    Returns:
        The errors and warnings found; empty tuples mean a clean instance.
    """
    errors = [*_dimension_errors(instance), *_id_errors(instance), *_course_errors(instance)]
    capacity_slots = instance.n_rooms * instance.n_periods
    if instance.total_lectures > capacity_slots:
        errors.append(
            f"{instance.total_lectures} lectures exceed the {capacity_slots} room-period slots"
        )
    return CheckResult(errors=tuple(errors), warnings=tuple(_warnings(instance)))


def assert_valid(instance: Instance) -> None:
    """Raise if `check_instance` reports any error; warnings are ignored.

    Args:
        instance: The instance to validate.

    Raises:
        InstanceCheckError: The instance is malformed or hard-infeasible.
    """
    result = check_instance(instance)
    if not result.ok:
        listed = "\n  - ".join(result.errors)
        msg = f"instance {instance.name!r} failed {len(result.errors)} check(s):\n  - {listed}"
        raise InstanceCheckError(msg)


def _dimension_errors(instance: Instance) -> list[str]:
    """Header dimensions must be positive and mutually consistent.

    Args:
        instance: The instance to check.

    Returns:
        One message per violated dimension constraint.
    """
    errors = []
    for label, value in (
        ("days", instance.days),
        ("periods_per_day", instance.periods_per_day),
        ("courses", instance.n_courses),
        ("rooms", instance.n_rooms),
    ):
        if value < 1:
            errors.append(f"{label} must be at least 1, got {value}")
    if instance.min_daily_lectures > instance.max_daily_lectures:
        errors.append(
            f"min_daily_lectures {instance.min_daily_lectures} exceeds "
            f"max_daily_lectures {instance.max_daily_lectures}"
        )
    return errors


def _id_errors(instance: Instance) -> list[str]:
    """Ids must be unique per entity kind, and curricula must name existing courses.

    Args:
        instance: The instance to check.

    Returns:
        One message per duplicate id or dangling curriculum reference.
    """
    errors = []
    for kind, ids in (
        ("course", [c.id for c in instance.courses]),
        ("room", [r.id for r in instance.rooms]),
        ("curriculum", [q.id for q in instance.curricula]),
    ):
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            errors.append(f"duplicate {kind} ids: {duplicates}")
    for curriculum in instance.curricula:
        unknown = sorted(set(curriculum.course_ids) - instance.course_index.keys())
        if unknown:
            errors.append(f"curriculum {curriculum.id!r} names unknown courses {unknown}")
        if len(set(curriculum.course_ids)) != len(curriculum.course_ids):
            errors.append(f"curriculum {curriculum.id!r} lists a course more than once")
    return errors


def _course_errors(instance: Instance) -> list[str]:
    """Every course must have at least one lecture and enough legal places to put them.

    Args:
        instance: The instance to check.

    Returns:
        One message per infeasible course.
    """
    errors = []
    for index, course in enumerate(instance.courses):
        if course.n_lectures < 1:
            errors.append(f"course {course.id!r} has {course.n_lectures} lectures")
        if any(not 0 <= p < instance.n_periods for p in instance.unavailable[index]):
            errors.append(f"course {course.id!r} is unavailable in a period outside the horizon")
        if any(not 0 <= r < instance.n_rooms for r in instance.forbidden_rooms[index]):
            errors.append(f"course {course.id!r} forbids a room index outside the room list")
        available = len(instance.available_periods[index])
        if course.n_lectures > available:
            errors.append(
                f"course {course.id!r} needs {course.n_lectures} periods "
                f"but only {available} are available"
            )
        if not instance.permitted_rooms[index]:
            errors.append(f"course {course.id!r} is forbidden from every room")
    return errors


def _warnings(instance: Instance) -> list[str]:
    """Well-formed oddities that occur in the public corpus and must not block a run.

    Args:
        instance: The instance to check.

    Returns:
        One message per oddity found; never treated as an error.
    """
    warnings = []
    for index, course in enumerate(instance.courses):
        cap = max((instance.rooms[r].capacity for r in instance.permitted_rooms[index]), default=0)
        if course.min_working_days > min(course.n_lectures, instance.days):
            warnings.append(
                f"course {course.id!r} wants {course.min_working_days} working days but has "
                f"{course.n_lectures} lectures over {instance.days} days; its MinWorkingDays "
                f"cost can never reach zero"
            )
        if course.n_students > cap:
            warnings.append(
                f"course {course.id!r} has {course.n_students} students but its largest "
                f"permitted room seats {cap}; its RoomCapacity cost can never reach zero"
            )
        if not instance.curricula_of_course[index]:
            warnings.append(f"course {course.id!r} belongs to no curriculum")
    warnings.extend(
        f"room {room.id!r} has capacity 0" for room in instance.rooms if room.capacity == 0
    )
    return warnings
