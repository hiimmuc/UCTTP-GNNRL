"""`Solution`: its incrementally maintained counters, and the `.sol` reader/writer."""

from pathlib import Path

import pytest

from horarium.data.solution_io import SolutionParseError, read_solution, write_solution
from horarium.problem.cost import cost
from horarium.problem.formulations import FORMULATIONS
from horarium.problem.instance import Instance
from horarium.problem.random_solutions import SHAPES, random_solution
from horarium.problem.solution import EMPTY, Move, Solution


def _recount(solution: Solution, instance: Instance) -> None:
    """Rebuild every counter from room_at alone and require it to match what was maintained."""
    fresh = Solution(instance)
    for course, period, room in solution.lectures():
        fresh.place(course, period, room)
    assert fresh.room_period_load == solution.room_period_load
    assert fresh.course_daily_lectures == solution.course_daily_lectures
    assert fresh.curriculum_period_lectures == solution.curriculum_period_lectures
    assert fresh.curriculum_daily_lectures == solution.curriculum_daily_lectures
    assert fresh.rooms_of_course == solution.rooms_of_course
    assert fresh.working_days == solution.working_days
    assert fresh.n_scheduled == solution.n_scheduled
    assert fresh.courses_at_period == solution.courses_at_period


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_counters_survive_construction(toy: Instance, shape: str) -> None:
    _recount(random_solution(toy, shape, seed=2), toy)


def test_counters_survive_a_churn_of_moves(comp01: Instance) -> None:
    solution = random_solution(comp01, "scattered", seed=4)
    for step, (course, period, _room) in enumerate(solution.lectures()[:40]):
        solution.apply(Move(course, period, step % comp01.n_rooms))
    for course, period, _room in solution.lectures()[:20]:
        solution.apply(Move(course, period, EMPTY))
    _recount(solution, comp01)


def test_place_twice_in_one_cell_raises(toy: Instance) -> None:
    solution = Solution(toy)
    solution.place(0, 0, 0)
    with pytest.raises(ValueError, match="already has a lecture"):
        solution.place(0, 0, 1)


def test_clear_an_empty_cell_raises(toy: Instance) -> None:
    with pytest.raises(ValueError, match="no lecture"):
        Solution(toy).clear(0, 0)


def test_copy_is_independent(toy: Instance) -> None:
    original = random_solution(toy, "scattered", seed=6)
    clone = original.copy()
    before = cost(toy, original, FORMULATIONS["UD2"]).total
    course, period, _room = clone.lectures()[0]
    clone.clear(course, period)
    assert cost(toy, original, FORMULATIONS["UD2"]).total == before
    assert original.room_at != clone.room_at


def test_sol_round_trip_preserves_every_lecture(sample: Instance, tmp_path: Path) -> None:
    original = random_solution(sample, "scattered", seed=8)
    path = tmp_path / "solution.sol"
    write_solution(sample, original, path)
    assert read_solution(sample, path).lectures() == original.lectures()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("nope rA 0 0", "unknown course"),
        ("SceCosC nope 0 0", "unknown room"),
        ("SceCosC rA 0", "expected 4 fields"),
        ("SceCosC rA 0 99", "outside the horizon"),
        ("SceCosC rA x 0", "expected an integer"),
    ],
)
def test_malformed_solutions_raise_rather_than_warn(
    toy: Instance, tmp_path: Path, line: str, message: str
) -> None:
    """The reference validator warns and skips; silently dropping a lecture would hide a bug."""
    path = tmp_path / "broken.sol"
    path.write_text(line + "\n")
    with pytest.raises(SolutionParseError, match=message):
        read_solution(toy, path)


def test_repeated_cell_in_a_sol_file_raises(toy: Instance, tmp_path: Path) -> None:
    path = tmp_path / "repeat.sol"
    path.write_text("SceCosC rA 0 0\nSceCosC rB 0 0\n")
    with pytest.raises(SolutionParseError, match="repeated entry"):
        read_solution(toy, path)
