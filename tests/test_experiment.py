"""Aggregation and rendering of the encoder sweep log."""

from __future__ import annotations

from pathlib import Path

from horarium.eval.experiment import (
    RunRecord,
    aggregate,
    comparison_markdown,
    load_runs,
    write_run,
)


def _record(encoder: str, instance: str, cost: float | None, **kw: object) -> RunRecord:
    return RunRecord(
        instance=Path(instance).stem,
        instance_file=instance,
        family=Path(instance).parent.name,
        encoder=encoder,
        seed=0,
        device="cpu",
        status="ok",
        train_feasibility_rate=kw.get("train_feasibility_rate", 0.5),  # type: ignore[arg-type]
        train_best_cost=None if cost is None else cost + 1,
        solve_total_cost=cost,
        solve_validated=None if cost is None else True,
    )


def test_load_runs_round_trips_and_ignores_unknown_keys(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    write_run(log, _record("gcn", "data/raw/ITC-2007/comp01.ectt", 42.0))
    # A line carrying a key the dataclass does not know must still load.
    with log.open("a") as sink:
        sink.write(
            '{"instance": "x", "instance_file": "f/x.ectt", "family": "f", "encoder": "gat", '
            '"seed": 0, "device": "cpu", "status": "ok", "unknown_future_field": 9}\n'
        )
    records = load_runs(log)
    assert {r.encoder for r in records} == {"gcn", "gat"}


def test_aggregate_compares_only_on_commonly_solved_instances() -> None:
    records = [
        _record("flat", "data/raw/ITC-2007/comp01.ectt", 100.0),
        _record("flat", "data/raw/ITC-2007/comp02.ectt", 200.0),
        _record("gcn", "data/raw/ITC-2007/comp01.ectt", 90.0),
        _record("gcn", "data/raw/ITC-2007/comp02.ectt", None),  # gcn failed comp02
    ]
    summaries = {s.encoder: s for s in aggregate(records)}
    # Only comp01 is solved by both, so the means are over comp01 alone.
    assert summaries["flat"].mean_solve_cost == 100.0
    assert summaries["gcn"].mean_solve_cost == 90.0
    assert summaries["gcn"].wins == 1
    assert summaries["flat"].wins == 1  # flat alone solved comp02
    assert summaries["gcn"].solve_feasible_instances == 1
    # Sorted best-first by mean solve cost.
    assert next(s.encoder for s in aggregate(records)) == "gcn"


def test_comparison_markdown_marks_the_best_per_instance() -> None:
    records = [
        _record("flat", "data/raw/ITC-2007/comp01.ectt", 100.0),
        _record("gcn", "data/raw/ITC-2007/comp01.ectt", 90.0),
    ]
    text = comparison_markdown(records)
    assert "# Encoder comparison" in text
    assert "**90**" in text
    assert "| `gcn` |" in text
