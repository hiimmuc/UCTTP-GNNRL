"""`python -m horarium.cli.train`: run PPO on one instance with a chosen encoder and formulation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from horarium.agents.checkpoint import save_agent
from horarium.agents.ppo import PPOConfig, TrainingReport, resolve_device, train
from horarium.data.ectt import read_ectt
from horarium.envs.construct import ConstructEnv
from horarium.models.encoders import ENCODERS, Encoder, EncoderSpec
from horarium.problem.formulations import FORMULATIONS

__all__ = ["main"]


def _progress_reporter(total_updates: int, started: float) -> Callable[[TrainingReport], None]:
    """A one-line progress bar for the training loop, redrawn on stderr after every update.

    It writes to stderr so stdout stays a single JSON object, which is what `horarium.cli.sweep`
    parses. `recent` is feasibility over the last `RECENT_EPISODES` episodes -- the number that
    actually moves while an agent learns -- and `best` is the cheapest feasible timetable so far.

    Args:
        total_updates: How many PPO updates the run will do, for the bar's denominator.
        started: `time.perf_counter()` at the start of the run, for the elapsed clock.

    Returns:
        A callback suitable for `train(..., on_update=...)`.
    """

    def report_progress(report: TrainingReport) -> None:
        """Redraw the bar for one finished update.

        Args:
            report: The training report as of this update.
        """
        done = report.updates
        filled = round(30 * done / total_updates) if total_updates else 30
        best = "-" if report.best_stats is None else f"{report.best_cost:g}"
        elapsed = time.perf_counter() - started
        line = (
            f"\r  {'█' * filled}{'░' * (30 - filled)} {done}/{total_updates}  "
            f"{report.steps} steps  {report.episodes} eps  "
            f"feas {report.recent_feasibility_rate:.2f}  best {best}  {elapsed:.0f}s"
        )
        sys.stderr.write(line.ljust(100))
        sys.stderr.flush()
        if done == total_updates:
            sys.stderr.write("\n")

    return report_progress


def main(argv: list[str] | None = None) -> int:
    """Train an agent on one instance, checkpoint it, and print a JSON summary of the run.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--formulation", choices=sorted(FORMULATIONS), default="UD2")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="episodes are finite and the terminal penalty is the point, so do not discount",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="the tensors here are small, so extra CPU threads cost more than they save",
    )
    parser.add_argument("--checkpoint", type=Path, help="where to write the trained policy")
    parser.add_argument("--out", type=Path, help="append the run summary to this JSONL file")
    parser.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help="live progress bar on stderr; 'auto' shows it only when stderr is a terminal",
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    resolve_device(args.device)  # fail here, not several minutes into training

    instance = read_ectt(args.instance)
    env = ConstructEnv(instance, FORMULATIONS[args.formulation])
    spec = EncoderSpec(name=args.encoder, embedding_dim=args.embedding_dim)

    def build_encoder() -> Encoder:
        """Built inside `train`, after the seed is set, so runs are reproducible.

        Returns:
            A freshly constructed encoder.
        """
        return spec.build(
            node_features=env.node_feature_dim,
            global_features=env.global_feature_dim,
            edge_features=env.edge_feature_dim,
        )

    config = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        seed=args.seed,
        device=args.device,
    )

    show_progress = args.progress == "on" or (args.progress == "auto" and sys.stderr.isatty())
    started = time.perf_counter()
    on_update = (
        _progress_reporter(max(1, args.total_steps // args.rollout_steps), started)
        if show_progress
        else None
    )
    report = train(env, build_encoder, config, on_update=on_update)
    elapsed = time.perf_counter() - started

    checkpoint = args.checkpoint or Path("experiments/checkpoints") / (
        f"{args.instance.stem}-{args.formulation}-{args.encoder}-seed{args.seed}.pt"
    )
    if report.agent is not None:
        save_agent(checkpoint, report.agent, env, spec, args.seed)

    summary = {
        "instance": instance.name,
        "instance_file": str(args.instance),
        "formulation": args.formulation,
        "encoder": args.encoder,
        "seed": args.seed,
        "gamma": args.gamma,
        "entropy_coef": args.entropy_coef,
        "device": args.device,
        "steps": report.steps,
        "updates": report.updates,
        "episodes": report.episodes,
        "feasibility_rate": round(report.feasibility_rate, 4),
        "recent_feasibility_rate": round(report.recent_feasibility_rate, 4),
        "best_cost": report.best_cost if report.best_stats else None,
        "checkpoint": str(checkpoint),
        "seconds": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as sink:
            sink.write(json.dumps({**summary, "history": report.history}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
