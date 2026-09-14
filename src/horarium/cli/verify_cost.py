"""`python -m horarium.cli.verify_cost`: blocking gate — our cost must equal the validator's."""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

from horarium.data.ectt import read_ectt
from horarium.data.solution_io import write_solution
from horarium.eval.validate import compare, run_validator
from horarium.problem.cost import cost, delta
from horarium.problem.formulations import FORMULATIONS, Formulation
from horarium.problem.instance import Instance
from horarium.problem.random_solutions import SHAPES, random_solution
from horarium.problem.solution import EMPTY, Move, Solution

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Compare `cost` against the validator on every instance, shape and formulation.

    Also property-checks `delta` against a full recomputation, and prints the offending
    components rather than summarising them away.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: non-zero on any disagreement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--moves", type=int, default=8, help="delta probes per instance/shape")
    args = parser.parse_args(argv)

    paths = sorted(args.raw_dir.rglob("*.ectt"))
    if not paths:
        print(
            f"no instances under {args.raw_dir}; run `python -m horarium.cli.prepare` first",
            file=sys.stderr,
        )
        return 1

    compared = mismatched = probes = wrong_deltas = 0
    with tempfile.TemporaryDirectory() as workspace:
        written = Path(workspace) / "candidate.sol"
        for path in paths:
            instance = read_ectt(path)
            for shape in sorted(SHAPES):
                solution = random_solution(instance, shape, seed=args.seed)
                write_solution(instance, solution, written)
                for name, formulation in FORMULATIONS.items():
                    ours = cost(instance, solution, formulation)
                    problems = compare(ours, run_validator(name, path, written))
                    compared += 1
                    if problems:
                        mismatched += 1
                        print(f"MISMATCH {path.stem} {shape} {name}", file=sys.stderr)
                        for problem in problems:
                            print(f"  - {problem}", file=sys.stderr)
                    checked, wrong = _probe_delta(
                        instance, solution, formulation, args.seed, args.moves
                    )
                    probes += checked
                    wrong_deltas += wrong
            print(f"  {path.parent.name}/{path.stem}: ok", flush=True)

    print(f"\n{compared} cost comparisons, {mismatched} mismatched")
    print(f"{probes} delta probes, {wrong_deltas} wrong")
    if mismatched or wrong_deltas:
        print("E0 FAILED - nothing downstream may be trusted until this agrees", file=sys.stderr)
        return 1
    print("E0 PASSED")
    return 0


def _probe_delta(
    instance: Instance,
    solution: Solution,
    formulation: Formulation,
    seed: int,
    moves: int,
) -> tuple[int, int]:
    """Check `delta` against a full recomputation.

    Args:
        instance: The instance the solution is for.
        solution: The current solution; unchanged on return.
        formulation: Which components are active and at what weight.
        seed: Seed for the moves probed.
        moves: Number of moves to probe.

    Returns:
        A (probes made, probes wrong) pair.
    """
    rng = random.Random(f"{instance.name}/{formulation.name}/{seed}")  # noqa: S311
    base = cost(instance, solution, formulation).total
    wrong = 0
    for _ in range(moves):
        move = Move(
            course=rng.randrange(instance.n_courses),
            period=rng.randrange(instance.n_periods),
            room=rng.choice([EMPTY, *range(instance.n_rooms)]),
        )
        predicted = delta(instance, solution, move, formulation)
        previous = solution.room_at[move.course][move.period]
        solution.apply(move)
        actual = cost(instance, solution, formulation).total
        solution.undo(move, previous)
        if actual != base + predicted:
            wrong += 1
            print(
                f"DELTA MISMATCH {instance.name} {formulation.name} {move}: "
                f"base={base} delta={predicted} actual={actual}",
                file=sys.stderr,
            )
    return moves, wrong


if __name__ == "__main__":
    raise SystemExit(main())
