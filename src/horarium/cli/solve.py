"""`python -m horarium.cli.solve`: run a trained policy, write a .sol, and check it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from horarium.agents.checkpoint import CheckpointError, load_agent
from horarium.agents.ppo import resolve_device
from horarium.agents.rollout import solve
from horarium.data.ectt import read_ectt
from horarium.data.solution_io import write_solution
from horarium.envs.construct import ConstructEnv
from horarium.eval.validate import ValidatorError, compare, default_validator_path, run_validator
from horarium.problem.cost import CostBreakdown, cost
from horarium.problem.formulations import FORMULATIONS

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Produce a timetable with a trained policy and report what it costs.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: 0 on a validated feasible timetable, 1 if no attempt was feasible or the
        validator disagrees, 2 if the checkpoint cannot be used for this instance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formulation", choices=sorted(FORMULATIONS), default="UD2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument(
        "--greedy", action="store_true", help="always take the best legal action, never sample"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, help="where to write the .sol file")
    parser.add_argument(
        "--no-validate", action="store_true", help="skip the reference validator cross-check"
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    instance = read_ectt(args.instance)
    formulation = FORMULATIONS[args.formulation]
    env = ConstructEnv(instance, formulation)
    try:
        agent, metadata = load_agent(args.checkpoint, env)
    except CheckpointError as failure:
        print(f"cannot load {args.checkpoint}: {failure}", file=sys.stderr)
        return 2
    agent.to(device)

    if metadata.formulation != args.formulation:
        print(f"note: checkpoint was trained on {metadata.formulation}, solving {args.formulation}")

    result = solve(env, agent, restarts=args.restarts, greedy=args.greedy, seed=args.seed)
    summary: dict[str, object] = {
        "instance": instance.name,
        "formulation": args.formulation,
        "encoder": metadata.encoder,
        "checkpoint": str(args.checkpoint),
        "attempts": result.attempts,
        "feasible_attempts": result.feasible,
        "feasibility_rate": round(result.feasibility_rate, 4),
    }

    if result.best is None:
        summary["status"] = "no feasible timetable found"
        print(json.dumps(summary, indent=2))
        return 1

    breakdown = cost(instance, result.best.solution, formulation)
    destination = args.out or Path("experiments/solutions") / (
        f"{args.instance.stem}-{args.formulation}-seed{args.seed}.sol"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_solution(instance, result.best.solution, destination)

    summary |= {
        "status": "feasible",
        "total_cost": breakdown.total,
        "components": {key.value: value for key, value in breakdown.soft.items()},
        "violations": breakdown.violations,
        "solution": str(destination),
    }

    if not args.no_validate:
        summary["validated"] = _cross_check(args, destination, breakdown, summary)
        if summary["validated"] is False:
            print(json.dumps(summary, indent=2))
            return 1

    print(json.dumps(summary, indent=2))
    return 0


def _cross_check(
    args: argparse.Namespace,
    solution_path: Path,
    breakdown: CostBreakdown,
    summary: dict[str, object],
) -> bool | None:
    """Compare our cost against the reference validator.

    Args:
        args: Parsed command-line arguments; `formulation` and `instance` are read from it.
        solution_path: Path to the .sol file to validate.
        breakdown: Our own cost breakdown for the solution.
        summary: Result dict, updated in place with a `"validator"` entry.

    Returns:
        `True` if the validator agrees, `False` if it disagrees or fails, `None` if the
        validator binary is not built.
    """
    if not default_validator_path().exists():
        summary["validator"] = "not built; run scripts/setup.sh"
        return None
    try:
        report = run_validator(args.formulation, args.instance, solution_path)
    except ValidatorError as failure:
        summary["validator"] = str(failure)
        return False
    problems = compare(breakdown, report)
    if problems:
        summary["validator"] = problems
        return False
    summary["validator"] = "agrees"
    return True


if __name__ == "__main__":
    raise SystemExit(main())
