import re
from pathlib import Path

import pytest

from horarium.data.ectt import EcttParseError, read_ectt, write_ectt
from horarium.problem.instance import Instance

MINIMAL = """\
Name: Tiny
Courses: 2
Rooms: 2
Days: 2
Periods_per_day: 2
Curricula: 1
Min_Max_Daily_Lectures: 1 2
UnavailabilityConstraints: 1
RoomConstraints: 1

COURSES:
c1 t1 2 2 10 1
c2 t2 1 1 5 0

ROOMS:
r1 20 0
r2 5 1

CURRICULA:
q1 2 c1 c2

UNAVAILABILITY_CONSTRAINTS:
c1 0 1

ROOM_CONSTRAINTS:
c2 r1

END.
"""


@pytest.fixture
def minimal(tmp_path: Path) -> Instance:
    path = tmp_path / "tiny.ectt"
    path.write_text(MINIMAL)
    return read_ectt(path)


def test_minimal_fields(minimal: Instance) -> None:
    assert minimal.name == "Tiny"
    assert minimal.n_periods == 4
    assert minimal.total_lectures == 3
    assert minimal.courses[0].double_lectures is True
    assert minimal.courses[1].double_lectures is False
    assert minimal.rooms[1].building == 1


def test_unavailability_flattens_day_and_slot(minimal: Instance) -> None:
    assert minimal.unavailable[0] == {1}
    assert minimal.available_periods[0] == (0, 2, 3)


def test_room_constraints_are_forbidden_not_allowed(minimal: Instance) -> None:
    """ROOM_CONSTRAINTS lists rooms a course may NOT use (validator.cc CostsOnRoomConstraints)."""
    assert minimal.forbidden_rooms[1] == {0}
    assert minimal.permitted_rooms[1] == (1,)
    assert minimal.forbidden_rooms[0] == frozenset()
    assert minimal.permitted_rooms[0] == (0, 1)


def test_derived_index_tables(minimal: Instance) -> None:
    assert minimal.course_index == {"c1": 0, "c2": 1}
    assert minimal.courses_of_curriculum == ((0, 1),)
    assert minimal.curricula_of_course == ((0,), (0,))
    assert minimal.courses_of_teacher == {"t1": (0,), "t2": (1,)}


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        (lambda s: s.replace("Rooms:", "Room:"), "expected 'Rooms:'"),
        (lambda s: s.replace("Courses: 2", "Courses: 3"), "expected an integer"),
        (lambda s: s.replace("c1 t1 2 2 10 1", "c1 t1 2 2 10 2"), "0/1 flag"),
        (lambda s: s.replace("q1 2 c1 c2", "q1 2 c1 cX"), "unknown course"),
        (lambda s: s.replace("c1 0 1", "c1 0 9"), "out of range"),
        (lambda s: s.replace("c2 r1", "c2 rX"), "unknown room"),
        (lambda s: s.replace("c2 t2 1 1 5 0", "c1 t2 1 1 5 0"), "duplicate course id"),
        (lambda s: s.replace("END.", "END.\nextra"), "unexpected tokens"),
        (lambda s: s.replace("\nEND.\n", "\n"), "file ended"),
    ],
)
def test_malformed_files_raise(tmp_path: Path, corruption: object, message: str) -> None:
    path = tmp_path / "broken.ectt"
    path.write_text(corruption(MINIMAL))  # type: ignore[operator]
    with pytest.raises(EcttParseError, match=re.escape(message)):
        read_ectt(path)


def test_every_instance_parses(sample: Instance) -> None:
    assert sample.n_courses > 0
    assert sample.n_rooms > 0
    assert sample.n_periods > 0


def test_round_trip_is_equal(sample: Instance, tmp_path: Path) -> None:
    out = tmp_path / "round_trip.ectt"
    write_ectt(sample, out)
    assert read_ectt(out) == sample


def test_parsed_counts_match_declared_header(sample_path: Path, sample: Instance) -> None:
    """The header counts are read, not trusted: verify they equal what the sections contain."""
    tokens = sample_path.read_text().split()
    declared = {
        key: int(tokens[tokens.index(key) + 1])
        for key in ("Courses:", "Rooms:", "Curricula:", "Days:", "Periods_per_day:")
    }
    assert declared["Courses:"] == sample.n_courses
    assert declared["Rooms:"] == sample.n_rooms
    assert declared["Curricula:"] == sample.n_curricula
    assert declared["Days:"] == sample.days
    assert declared["Periods_per_day:"] == sample.periods_per_day

    # The header counts *rows*, and EA10, test2 and test3 repeat some; the repeats are
    # idempotent, so the parsed sets can only be smaller. Row counts themselves are enforced
    # structurally: a wrong count desynchronises the reader and trips the next section label.
    unavailability = int(tokens[tokens.index("UnavailabilityConstraints:") + 1])
    room_constraints = int(tokens[tokens.index("RoomConstraints:") + 1])
    assert sum(len(s) for s in sample.unavailable) <= unavailability
    assert sum(len(s) for s in sample.forbidden_rooms) <= room_constraints


DUPLICATED_ROWS = {"EA10": 27, "test2": 4, "test3": 1}


def test_known_instances_repeat_constraint_rows(sample_path: Path, sample: Instance) -> None:
    """Three shipped instances declare more constraint rows than they hold distinct pairs."""
    tokens = sample_path.read_text().split()
    declared = int(tokens[tokens.index("UnavailabilityConstraints:") + 1]) + int(
        tokens[tokens.index("RoomConstraints:") + 1]
    )
    distinct = sum(len(s) for s in sample.unavailable) + sum(len(s) for s in sample.forbidden_rooms)
    assert declared - distinct == DUPLICATED_ROWS.get(sample_path.stem, 0)
