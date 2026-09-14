"""`python -m horarium.cli.report`: turn the sweep log into the comparison doc and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from horarium.eval.experiment import (
    comparison_markdown,
    comparison_plots,
    learning_curve_plots,
    load_histories,
    load_runs,
)

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Write the comparison Markdown and, unless `--no-plots`, the PNG figures.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: 0 on success, 1 if the run log is empty or missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("experiments/runs.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("experiments/history.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/encoder_comparison.md"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    records = load_runs(args.log)
    if not records:
        print(f"no runs in {args.log}; run `python -m horarium.cli.sweep` first", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(comparison_markdown(records))
    print(f"wrote {args.out} ({len(records)} runs)")

    if not args.no_plots:
        for path in comparison_plots(records, args.figures):
            print(f"wrote {path}")
        curves = learning_curve_plots(load_histories(args.history), args.figures)
        for path in curves:
            print(f"wrote {path}")
        if not curves:
            print(f"no training curves in {args.history}; skipped the learning-curve figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
