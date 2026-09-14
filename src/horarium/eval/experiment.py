"""The encoder sweep: one training run per (instance, encoder, seed), and how to compare them.

`RunRecord` is one row of `experiments/runs.jsonl`. `aggregate` rolls the rows up per encoder,
`comparison_markdown` renders the tables, and `comparison_plots` draws the figures. Nothing here
trains anything; `horarium.cli.sweep` produces the rows and `horarium.cli.report` consumes them.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EncoderSummary",
    "RunRecord",
    "aggregate",
    "comparison_markdown",
    "comparison_plots",
    "learning_curve_plots",
    "load_histories",
    "load_runs",
    "write_run",
]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One training run and the greedy solve that followed it."""

    instance: str
    instance_file: str
    family: str
    encoder: str
    seed: int
    device: str
    status: str
    """`"ok"` if training finished, otherwise a short error tag."""
    train_steps: int = 0
    train_episodes: int = 0
    train_feasibility_rate: float = 0.0
    train_recent_feasibility_rate: float = 0.0
    train_best_cost: float | None = None
    train_seconds: float = 0.0
    train_steps_per_episode: float = 0.0
    """Steps each episode took, which is what makes an equal-step budget unequal.

    An episode places one lecture per step, so its length is the instance's lecture count. A
    fixed step budget therefore buys 45 episodes on UUMCAS_A131 and 628 on comp01 -- recorded
    here so the comparison can be read honestly rather than assumed fair.
    """
    solve_status: str = "skipped"
    solve_feasibility_rate: float = 0.0
    solve_total_cost: float | None = None
    solve_components: dict[str, int] = field(default_factory=dict)
    """Weighted soft cost per component.

    RoomCapacity and RoomStability are decided by stage 2, the rest by the policy's period
    assignment, so reporting the split is what separates a claim about the learned part from a
    claim about the room solver.
    """
    solve_validated: bool | None = None
    solve_seconds: float = 0.0
    error: str = ""


def load_runs(path: Path) -> list[RunRecord]:
    """Read every run record from a JSONL file.

    Args:
        path: Path to the JSONL log; a missing file reads as no runs.

    Returns:
        One `RunRecord` per line, unknown keys ignored so the schema can grow.
    """
    if not path.exists():
        return []
    fields = set(RunRecord.__dataclass_fields__)
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            raw: dict[str, Any] = json.loads(line)
            records.append(RunRecord(**{k: v for k, v in raw.items() if k in fields}))
    return records


def write_run(path: Path, record: RunRecord) -> None:
    """Append one run record to a JSONL file, creating it and its parent if needed.

    Args:
        path: Path to the JSONL log.
        record: The record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as sink:
        sink.write(json.dumps(asdict(record)) + "\n")


def load_histories(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    """Read the per-update training curves `horarium.cli.train --out` writes.

    Each line is one training run: its summary fields plus a `"history"` list holding one
    record per PPO update.

    Args:
        path: Path to the history JSONL; a missing file reads as no histories.

    Returns:
        The history of each run, keyed by `(encoder, instance_file)`.
    """
    if not path.exists():
        return {}
    curves: dict[tuple[str, str], list[dict[str, float]]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            raw: dict[str, Any] = json.loads(line)
            history = raw.get("history") or []
            if history:
                curves[str(raw.get("encoder", "?")), str(raw.get("instance_file", "?"))] = history
    return curves


def _mean_curve(
    curves: list[list[dict[str, float]]], field_name: str
) -> tuple[list[float], list[float]]:
    """Average one field across runs of differing length, truncated to the shortest.

    Args:
        curves: One history per run.
        field_name: Which per-update field to average.

    Returns:
        Step counts and the mean of `field_name` at each of them.
    """
    if not curves:
        return [], []
    length = min(len(c) for c in curves)
    steps = [curves[0][i]["steps"] for i in range(length)]
    means = [statistics.fmean(c[i][field_name] for c in curves) for i in range(length)]
    return steps, means


@dataclass(frozen=True, slots=True)
class EncoderSummary:
    """What one encoder achieved across every instance it was run on."""

    encoder: str
    runs: int
    errors: int
    train_feasible_instances: int
    """Instances where training produced at least one feasible episode."""
    solve_feasible_instances: int
    """Instances where the greedy solve produced a validated feasible timetable."""
    mean_train_feasibility_rate: float
    mean_solve_cost: float | None
    """Mean greedy-solve cost over instances both this encoder and `flat` solved feasibly."""
    median_train_seconds: float
    wins: int
    """Instances where this encoder's solve cost was the best (or tied) among all encoders."""


def _family(instance_file: str) -> str:
    """Family folder an instance lives in, e.g. `ITC-2007`.

    Args:
        instance_file: Path to the `.ectt` file.

    Returns:
        The parent directory name.
    """
    return Path(instance_file).parent.name


def aggregate(records: list[RunRecord]) -> list[EncoderSummary]:
    """Roll run records up into one summary per encoder.

    The mean solve cost is taken only over instances every encoder solved feasibly, so the
    numbers compare like with like rather than rewarding an encoder for failing on hard
    instances. `wins` counts best-or-tied greedy-solve cost per instance.

    Args:
        records: Every run record, typically from `load_runs`.

    Returns:
        One `EncoderSummary` per encoder, ordered by mean solve cost then encoder name.
    """
    encoders = sorted({r.encoder for r in records})
    instances = sorted({r.instance_file for r in records})

    solved_cost: dict[tuple[str, str], float] = {
        (r.encoder, r.instance_file): cost
        for r in records
        if (cost := r.solve_total_cost) is not None and r.solve_validated is not False
    }
    common = [inst for inst in instances if all((enc, inst) in solved_cost for enc in encoders)]
    best_per_instance = {
        inst: min(solved_cost[(enc, inst)] for enc in encoders if (enc, inst) in solved_cost)
        for inst in instances
        if any((enc, inst) in solved_cost for enc in encoders)
    }

    summaries = []
    for enc in encoders:
        rows = [r for r in records if r.encoder == enc]
        costs_on_common = [solved_cost[(enc, inst)] for inst in common]
        summaries.append(
            EncoderSummary(
                encoder=enc,
                runs=len(rows),
                errors=sum(r.status != "ok" for r in rows),
                train_feasible_instances=sum((r.train_best_cost is not None) for r in rows),
                solve_feasible_instances=sum(
                    r.solve_total_cost is not None and r.solve_validated is not False for r in rows
                ),
                mean_train_feasibility_rate=(
                    statistics.fmean(r.train_feasibility_rate for r in rows) if rows else 0.0
                ),
                mean_solve_cost=(statistics.fmean(costs_on_common) if costs_on_common else None),
                median_train_seconds=(
                    statistics.median(r.train_seconds for r in rows) if rows else 0.0
                ),
                wins=sum(
                    1
                    for inst in instances
                    if (enc, inst) in solved_cost
                    and inst in best_per_instance
                    and solved_cost[(enc, inst)] <= best_per_instance[inst] + 1e-9
                ),
            )
        )
    summaries.sort(key=lambda s: (s.mean_solve_cost is None, s.mean_solve_cost or 0.0, s.encoder))
    return summaries


def comparison_markdown(records: list[RunRecord]) -> str:
    """Render the encoder comparison: the headline table, a per-family view, and per-instance costs.

    Args:
        records: Every run record, typically from `load_runs`.

    Returns:
        The complete Markdown document, ending in a newline.
    """
    summaries = aggregate(records)
    encoders = [s.encoder for s in summaries]
    instances = sorted({r.instance_file for r in records}, key=lambda p: (_family(p), p))
    n_instances = len(instances)

    lines = [
        "# Encoder comparison",
        "",
        "Generated by `python -m horarium.cli.report` from `experiments/runs.jsonl`. Do not edit.",
        "",
        f"{len(records)} runs over {n_instances} instances and {len(encoders)} encoders. "
        "`solve cost` is the greedy rollout cost, averaged only over instances every encoder "
        "solved feasibly, so the column compares like with like.",
        "",
        "## Encoders",
        "",
        "| encoder | train-feasible | solve-feasible | mean solve cost | wins | "
        "mean train feas. rate | median train s | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        cost = "-" if s.mean_solve_cost is None else f"{s.mean_solve_cost:.1f}"
        lines.append(
            f"| `{s.encoder}` | {s.train_feasible_instances}/{n_instances} | "
            f"{s.solve_feasible_instances}/{n_instances} | {cost} | {s.wins} | "
            f"{s.mean_train_feasibility_rate:.3f} | {s.median_train_seconds:.0f} | {s.errors} |"
        )

    families = sorted({_family(p) for p in instances})
    lines += [
        "",
        "## Solve-feasible instances per family",
        "",
        "| family | n | " + " | ".join(f"`{e}`" for e in encoders) + " |",
        "|---|---:|" + "---:|" * len(encoders),
    ]
    for family in families:
        members = [p for p in instances if _family(p) == family]
        cells = []
        for enc in encoders:
            feasible = sum(
                1
                for r in records
                if r.encoder == enc
                and r.instance_file in members
                and r.solve_total_cost is not None
                and r.solve_validated is not False
            )
            cells.append(f"{feasible}/{len(members)}")
        lines.append(f"| {family} | {len(members)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Greedy solve cost per instance",
        "",
        "`-` means no validated feasible timetable. Best per row is in **bold**.",
        "",
        "| instance | " + " | ".join(f"`{e}`" for e in encoders) + " |",
        "|---|" + "---:|" * len(encoders),
    ]
    by_key = {(r.encoder, r.instance_file): r for r in records}
    for inst in instances:
        costs = {
            enc: r.solve_total_cost
            for enc in encoders
            if (r := by_key.get((enc, inst))) is not None
            and r.solve_total_cost is not None
            and r.solve_validated is not False
        }
        best = min(costs.values()) if costs else None
        cells = []
        for enc in encoders:
            if enc not in costs:
                cells.append("-")
            elif best is not None and costs[enc] <= best + 1e-9:
                cells.append(f"**{costs[enc]:g}**")
            else:
                cells.append(f"{costs[enc]:g}")
        label = f"{_family(inst)}/{Path(inst).stem}"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join([*lines, ""])


def comparison_plots(records: list[RunRecord], out_dir: Path) -> list[Path]:
    """Draw the comparison figures: solve-feasibility and mean solve cost by encoder.

    Args:
        records: Every run record, typically from `load_runs`.
        out_dir: Directory to write the PNGs into; created if missing.

    Returns:
        Paths of the figures written.
    """
    import matplotlib  # noqa: PLC0415 - optional `report` extra, imported only when plotting

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    summaries = aggregate(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoders = [s.encoder for s in summaries]
    written = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(encoders, [s.solve_feasible_instances for s in summaries], color="#4c72b0")
    ax.set_ylabel("instances solved feasibly")
    ax.set_title("Greedy-solve feasibility by encoder")
    fig.tight_layout()
    feasibility_path = out_dir / "encoder_feasibility.png"
    fig.savefig(feasibility_path, dpi=120)
    plt.close(fig)
    written.append(feasibility_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    costs = [s.mean_solve_cost for s in summaries]
    labelled = [(e, c) for e, c in zip(encoders, costs, strict=True) if c is not None]
    if labelled:
        ax.bar([e for e, _ in labelled], [c for _, c in labelled], color="#c44e52")
        ax.set_ylabel("mean greedy-solve cost (common instances)")
        ax.set_title("Solution quality by encoder (lower is better)")
        fig.tight_layout()
        cost_path = out_dir / "encoder_solve_cost.png"
        fig.savefig(cost_path, dpi=120)
        written.append(cost_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(encoders, [s.mean_train_feasibility_rate for s in summaries], color="#55a868")
    ax.set_ylabel("mean training feasibility rate")
    ax.set_title("Training feasibility rate by encoder")
    fig.tight_layout()
    train_path = out_dir / "encoder_train_feasibility.png"
    fig.savefig(train_path, dpi=120)
    plt.close(fig)
    written.append(train_path)

    return written


def learning_curve_plots(
    histories: dict[tuple[str, str], list[dict[str, float]]], out_dir: Path
) -> list[Path]:
    """Draw the learning curves: how each encoder's feasibility rises as training proceeds.

    Runs are averaged per encoder over the instances they share, truncated to the shortest run,
    so the lines compare like with like. Feasibility is used rather than cost because it is
    already on a 0-1 scale and therefore commensurable across instances of very different size.

    Args:
        histories: Per-run training curves, from `load_histories`.
        out_dir: Directory to write the PNGs into; created if missing.

    Returns:
        Paths of the figures written, empty if there is nothing to plot.
    """
    if not histories:
        return []
    import matplotlib  # noqa: PLC0415 - optional `report` extra, imported only when plotting

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    encoders = sorted({encoder for encoder, _ in histories})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for encoder in encoders:
        runs = [curve for (enc, _), curve in histories.items() if enc == encoder]
        steps, means = _mean_curve(runs, "recent_feasibility_rate")
        if steps:
            ax.plot(steps, means, label=f"{encoder} (n={len(runs)})", linewidth=1.8)
            plotted = True
    if not plotted:
        plt.close(fig)
        return []

    ax.set_xlabel("environment steps")
    ax.set_ylabel("recent feasibility rate")
    ax.set_title("Learning curves by encoder (mean over instances)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]
