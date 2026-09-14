"""The Phase 2 gate for `problem/cost.py`.

Agreement with the reference validator, a correct `delta`, and a soft-cost bound that actually
bounds. These run over the sample corpus; `python -m horarium.cli.verify_cost` runs the same
comparison over all 61 instances as the blocking gate.
"""

import random
from pathlib import Path

import pytest

from horarium.data.solution_io import write_solution
from horarium.eval.validate import compare, default_validator_path, run_validator
from horarium.problem.cost import cost, delta, soft_cost_upper_bound
from horarium.problem.formulations import FORMULATIONS
from horarium.problem.instance import Instance
from horarium.problem.random_solutions import SHAPES, random_solution
from horarium.problem.solution import EMPTY, Move

needs_validator = pytest.mark.skipif(
    not default_validator_path().exists(), reason="run scripts/setup.sh to compile it"
)
MOVES_PER_CASE = 12


#: Three solution shapes with distinct cost profiles: spread out, all in one room, paired.
COST_SHAPES = ["scattered", "one_room", "adjacent_pairs"]


@needs_validator
@pytest.mark.parametrize("shape", COST_SHAPES)
@pytest.mark.parametrize("formulation", sorted(FORMULATIONS))
def test_cost_matches_validator(
    sample: Instance, sample_path: Path, shape: str, formulation: str, tmp_path: Path
) -> None:
    solution = random_solution(sample, shape, seed=11)
    written = tmp_path / "candidate.sol"
    write_solution(sample, solution, written)

    ours = cost(sample, solution, FORMULATIONS[formulation])
    theirs = run_validator(formulation, sample_path, written)
    assert compare(ours, theirs) == []


@needs_validator
def test_validator_reports_the_components_our_formulation_table_declares(
    sample: Instance, sample_path: Path, tmp_path: Path
) -> None:
    """Guards the transcription in formulations.py against the validator's own print order."""
    solution = random_solution(sample, "scattered", seed=5)
    written = tmp_path / "candidate.sol"
    write_solution(sample, solution, written)
    for name, formulation in FORMULATIONS.items():
        report = run_validator(name, sample_path, written)
        assert set(report.soft) == set(formulation.weights), name
        assert set(report.hard) == set(formulation.hard), name


@pytest.mark.parametrize("shape", COST_SHAPES)
@pytest.mark.parametrize("formulation", sorted(FORMULATIONS))
def test_delta_matches_recomputed_cost(sample: Instance, shape: str, formulation: str) -> None:
    rule = FORMULATIONS[formulation]
    solution = random_solution(sample, shape, seed=3)
    rng = random.Random(f"{sample.name}/{shape}/{formulation}")  # noqa: S311
    base = cost(sample, solution, rule).total

    for _ in range(MOVES_PER_CASE):
        move = Move(
            course=rng.randrange(sample.n_courses),
            period=rng.randrange(sample.n_periods),
            room=rng.choice([EMPTY, *range(sample.n_rooms)]),
        )
        predicted = delta(sample, solution, move, rule)
        assert cost(sample, solution, rule).total == base, "delta must not mutate the solution"

        previous = solution.room_at[move.course][move.period]
        solution.apply(move)
        assert cost(sample, solution, rule).total == base + predicted
        solution.undo(move, previous)
        assert cost(sample, solution, rule).total == base


def test_delta_of_a_no_op_move_is_zero(toy: Instance) -> None:
    solution = random_solution(toy, "scattered", seed=1)
    course, period, room = solution.lectures()[0]
    assert delta(toy, solution, Move(course, period, room), FORMULATIONS["UD2"]) == 0.0


@pytest.mark.parametrize("formulation", sorted(FORMULATIONS))
def test_soft_cost_bound_is_never_exceeded(sample: Instance, formulation: str) -> None:
    """The env divides rewards by this bound, so a solution may never cost more than it."""
    rule = FORMULATIONS[formulation]
    bound = soft_cost_upper_bound(sample, rule)
    assert bound > 0
    for shape in SHAPES:
        assert cost(sample, random_solution(sample, shape, seed=13), rule).total <= bound
