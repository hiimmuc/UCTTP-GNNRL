import dataclasses

from horarium.problem.checks import assert_valid, check_instance
from horarium.problem.instance import Instance


def test_public_corpus_has_no_errors(sample: Instance) -> None:
    result = check_instance(sample)
    assert result.errors == ()


def test_capacity_zero_and_orphan_courses_are_warnings_not_errors(sample: Instance) -> None:
    """Both occur in the shipped corpus, so promoting them to errors would reject real data."""
    result = check_instance(sample)
    assert result.ok


def test_course_forbidden_from_every_room_is_an_error(toy: Instance) -> None:
    broken = dataclasses.replace(
        toy, forbidden_rooms=(frozenset(range(toy.n_rooms)), *toy.forbidden_rooms[1:])
    )
    errors = check_instance(broken).errors
    assert any("forbidden from every room" in e for e in errors)


def test_more_lectures_than_available_periods_is_an_error(toy: Instance) -> None:
    broken = dataclasses.replace(
        toy, unavailable=(frozenset(range(toy.n_periods)), *toy.unavailable[1:])
    )
    errors = check_instance(broken).errors
    assert any("are available" in e for e in errors)


def test_assert_valid_passes_on_real_instances(toy: Instance) -> None:
    assert_valid(toy)
