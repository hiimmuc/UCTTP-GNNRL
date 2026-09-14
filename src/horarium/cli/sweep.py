"""`python -m horarium.cli.sweep`: run every arm on every instance and log what each achieved.

An arm is either an encoder, trained then solved in its own `horarium.cli.train` /
`horarium.cli.solve` subprocess so a crash or a CUDA out-of-memory on one instance is recorded
and the sweep moves on, or a non-learned baseline, which runs in-process in milliseconds. Both
kinds go through the same stage-2 room oracle and produce the same record, so the baselines are
rows in the results table rather than a separate study.

The step budget is scaled per instance to buy a fixed number of episodes -- see `_budget` -- and
every arm can be run under several seeds, because a ranking from a single seed is not testable.
Re-running skips (instance, arm, seed) triples already logged with `status == "ok"`, so an
interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from horarium.data.ectt import read_ectt
from horarium.data.solution_io import write_solution
from horarium.eval.experiment import RunRecord, load_runs, write_run
from horarium.models.encoders import ENCODERS
from horarium.problem.cost import cost
from horarium.problem.formulations import FORMULATIONS
from horarium.problem.instance import Instance
from horarium.solvers import RoomAssignmentError, assign_rooms, dsatur, random_order

__all__ = ["main"]

DEFAULT_LOG = Path("experiments/runs.jsonl")
DEFAULT_HISTORY = Path("experiments/history.jsonl")
DEFAULT_CHECKPOINTS = Path("experiments/sweep")

#: Non-learned arms, run in-process because they take milliseconds. They share the stage-2 room
#: oracle with the learned arms, so the only thing that differs is how periods were chosen.
BASELINES = {"dsatur": dsatur, "random_order": random_order}


def _budget(instance: Instance, episodes: int, max_steps: int) -> int:
    """How many env steps to give this instance, so every instance gets the same episode count.

    An episode places one lecture per step, so a complete episode is exactly `total_lectures`
    steps. A flat step budget therefore trains small instances for an order of magnitude more
    episodes than large ones -- 628 on comp01 against 45 on UUMCAS_A131 -- which is not a
    comparison of encoders but of how many lectures an instance happens to have.

    The cap keeps the largest families from dominating the wall-clock; where it binds the run is
    short of the episode target, and `train_steps_per_episode` on the record shows by how much.

    Args:
        instance: The instance about to be trained on.
        episodes: Episodes to aim for.
        max_steps: Ceiling on the step budget regardless.

    Returns:
        The step budget for this instance.
    """
    return min(max_steps, max(1, episodes * instance.total_lectures))


def _run_json(argv: list[str], timeout: float) -> tuple[dict[str, Any] | None, str]:
    """Run a CLI subprocess and parse the JSON object it prints last.

    Args:
        argv: The command to run.
        timeout: Seconds before the subprocess is killed.

    Returns:
        The parsed JSON summary and an empty string when the subprocess printed one (an exit
        code of 1 still counts -- `solve` exits 1 on an honest "no feasible timetable"); `None`
        and an error tag when it crashed without printing a summary.
    """
    try:
        done = subprocess.run(  # noqa: S603 - argv is built here from our own CLI names
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    try:
        parsed: dict[str, Any] = json.loads(done.stdout[done.stdout.index("{") :])
    except (ValueError, json.JSONDecodeError):
        parsed = {}
    if parsed:
        return parsed, ""
    tail = (done.stderr or done.stdout).strip().splitlines()
    reason = tail[-1] if tail else f"exit {done.returncode}"
    tag = "oom" if "out of memory" in reason.lower() else "error"
    return None, f"{tag}: {reason[:200]}"


def _instances(root: Path, families: list[str] | None) -> list[Path]:
    """Every `.ectt` file under `root`, optionally restricted to some family folders.

    Args:
        root: Directory to search.
        families: Family folder names to keep, or `None` for all.

    Returns:
        Sorted instance paths.
    """
    paths = sorted(root.rglob("*.ectt"))
    if families:
        keep = set(families)
        paths = [p for p in paths if p.parent.name in keep]
    return paths


def _baseline_record(
    instance_path: Path, arm: str, seed: int, args: argparse.Namespace
) -> RunRecord:
    """Run one non-learned arm and package it as a run record.

    Constructs a period assignment, hands it to the same stage-2 oracle the learned arms use,
    and reports the same fields, so the baseline sits in the results table as another row rather
    than as a footnote.

    Args:
        instance_path: The `.ectt` file to solve.
        arm: Which baseline to run; a key of `BASELINES`.
        seed: Seed for the constructor's tie-breaking.
        args: Parsed sweep arguments, for the formulation and restart budget.

    Returns:
        The finished record, feasible or not.
    """
    instance = read_ectt(instance_path)
    formulation = FORMULATIONS[args.formulation]
    started = time.perf_counter()
    construction = BASELINES[arm](instance, formulation, seed=seed, restarts=args.restarts)

    record = RunRecord(
        instance=instance_path.stem,
        instance_file=str(instance_path),
        family=instance_path.parent.name,
        encoder=arm,
        seed=seed,
        device="cpu",
        status="ok",
    )
    if not construction.complete:
        return replace(
            record,
            solve_status=f"placed {construction.placed}/{construction.required}",
            solve_seconds=round(time.perf_counter() - started, 2),
        )
    try:
        solution = assign_rooms(instance, construction.periods_of_course, formulation)
    except RoomAssignmentError as failure:
        return replace(record, solve_status="no room assignment", error=str(failure)[:200])

    breakdown = cost(instance, solution, formulation)
    destination = args.checkpoints / f"{instance_path.stem}-{args.formulation}-{arm}-seed{seed}.sol"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_solution(instance, solution, destination)
    return replace(
        record,
        solve_status="feasible" if breakdown.feasible else "hard violations",
        solve_feasibility_rate=1.0 if breakdown.feasible else 0.0,
        solve_total_cost=float(breakdown.total) if breakdown.feasible else None,
        solve_components={key.value: value for key, value in breakdown.soft.items()},
        solve_seconds=round(time.perf_counter() - started, 2),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the sweep's command-line interface.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="per-update training curves, one line per run; used by the report's learning curves",
    )
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--encoders", nargs="+", default=sorted(ENCODERS))
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=sorted(BASELINES),
        help="non-learned arms to run alongside the encoders; pass none to skip them",
    )
    parser.add_argument("--families", nargs="+", help="restrict to these family folders")
    parser.add_argument("--formulation", default="UD2")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="one run per seed per arm; several seeds are what makes a ranking testable",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="episodes to give each instance; the step budget is scaled per instance to match",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300_000,
        help="ceiling on the per-instance step budget, so the biggest families stay affordable",
    )
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-timeout", type=float, default=3600.0)
    parser.add_argument("--solve-timeout", type=float, default=900.0)
    parser.add_argument("--redo", action="store_true", help="re-run pairs already logged ok")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every arm on every instance and append one `RunRecord` per run to the log.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: 0 if every run finished with `status == "ok"`, 1 otherwise.
    """
    args = _parser().parse_args(argv)

    instances = _instances(args.data, args.families)
    if not instances:
        print(f"no .ectt instances under {args.data}", file=sys.stderr)
        return 1

    done = {
        (r.instance_file, r.encoder, r.seed)
        for r in load_runs(args.log)
        if r.status == "ok" or (r.status.startswith(("oom", "timeout")) and not args.redo)
    }
    arms = [*args.baselines, *args.encoders]
    pairs = [
        (inst, arm, seed)
        for seed in args.seeds
        for arm in arms
        for inst in instances
        if args.redo or (str(inst), arm, seed) not in done
    ]
    total = len(pairs)
    print(
        f"{total} runs to do ({len(instances)} instances x {len(arms)} arms "
        f"x {len(args.seeds)} seeds); arms: {', '.join(arms)}"
    )

    failures = 0
    for index, (instance, encoder, seed) in enumerate(pairs, start=1):
        stem = instance.stem
        checkpoint = args.checkpoints / f"{stem}-{args.formulation}-{encoder}-seed{seed}.pt"
        family = instance.parent.name
        print(f"[{index}/{total}] {family}/{stem} :: {encoder} :: seed {seed}", flush=True)

        if encoder in BASELINES:
            record = _baseline_record(instance, encoder, seed, args)
            write_run(args.log, record)
            cost_text = (
                f"cost {record.solve_total_cost:g}"
                if record.solve_total_cost is not None
                else record.solve_status
            )
            print(f"    {encoder} :: {cost_text} in {record.solve_seconds:.2f}s", flush=True)
            continue

        total_steps = _budget(read_ectt(instance), args.episodes, args.max_steps)
        started = time.perf_counter()
        train_summary, train_error = _run_json(
            [
                sys.executable,
                "-m",
                "horarium.cli.train",
                "--instance",
                str(instance),
                "--formulation",
                args.formulation,
                "--encoder",
                encoder,
                "--seed",
                str(seed),
                "--total-steps",
                str(total_steps),
                "--rollout-steps",
                str(args.rollout_steps),
                "--embedding-dim",
                str(args.embedding_dim),
                "--device",
                args.device,
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(args.history),
                "--progress",
                "off",
            ],
            timeout=args.train_timeout,
        )
        train_seconds = time.perf_counter() - started

        if train_summary is None:
            record = RunRecord(
                instance=stem,
                instance_file=str(instance),
                family=family,
                encoder=encoder,
                seed=seed,
                device=args.device,
                status=train_error.split(":")[0],
                train_seconds=round(train_seconds, 1),
                error=train_error,
            )
            write_run(args.log, record)
            failures += 1
            print(f"    train failed: {train_error}", flush=True)
            continue

        solve_summary, solve_error = _run_json(
            [
                sys.executable,
                "-m",
                "horarium.cli.solve",
                "--instance",
                str(instance),
                "--formulation",
                args.formulation,
                "--checkpoint",
                str(checkpoint),
                "--seed",
                str(seed),
                "--restarts",
                str(args.restarts),
                "--device",
                args.device,
                "--out",
                str(args.checkpoints / f"{stem}-{args.formulation}-{encoder}-seed{seed}.sol"),
            ],
            timeout=args.solve_timeout,
        )
        solve = solve_summary or {}
        validated = solve.get("validated")
        record = RunRecord(
            instance=stem,
            instance_file=str(instance),
            family=family,
            encoder=encoder,
            seed=seed,
            device=args.device,
            status="ok",
            train_steps=int(train_summary.get("steps", 0) or 0),
            train_episodes=int(train_summary.get("episodes", 0) or 0),
            train_feasibility_rate=float(train_summary.get("feasibility_rate", 0.0) or 0.0),
            train_recent_feasibility_rate=float(
                train_summary.get("recent_feasibility_rate", 0.0) or 0.0
            ),
            train_best_cost=train_summary.get("best_cost"),
            train_seconds=round(float(train_summary.get("seconds", train_seconds) or 0.0), 1),
            train_steps_per_episode=round(
                int(train_summary.get("steps", 0) or 0)
                / max(1, int(train_summary.get("episodes", 0) or 0)),
                2,
            ),
            solve_status=str(solve.get("status", solve_error or "unknown")),
            solve_feasibility_rate=float(solve.get("feasibility_rate", 0.0) or 0.0),
            solve_total_cost=solve.get("total_cost"),
            solve_components={str(k): int(v) for k, v in (solve.get("components") or {}).items()},
            solve_validated=validated if isinstance(validated, bool) else None,
            error=solve_error,
        )
        write_run(args.log, record)
        cost = record.solve_total_cost
        print(
            f"    ok :: train feas {record.train_feasibility_rate:.2f} :: "
            f"solve {'cost ' + format(cost, 'g') if cost is not None else record.solve_status}",
            flush=True,
        )

    print(f"done: {total - failures}/{total} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
