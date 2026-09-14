"""`python -m horarium.cli.watch`: a live progress bar for a running `horarium.cli.sweep`.

Reads `experiments/runs.jsonl` (one row per finished pair) and the sweep's stdout log to show
a refreshing bar, throughput, ETA, and a rolling tail of results. It only ever reads, so it is
safe to start, stop, and restart while the sweep runs.

    python -m horarium.cli.watch
    python -m horarium.cli.watch --sweep-log experiments/sweep.log --once
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import time
from pathlib import Path
from typing import Any

__all__ = ["main"]

_HEADER = re.compile(r"(\d+) pairs to run")
_CURRENT = re.compile(r"\[(\d+)/(\d+)\]\s+(.*)")
_SWEEP_LOG_FALLBACKS = (Path("experiments/sweep.log"), Path("experiments/sweep-full.log"))


def _read_log(path: Path) -> list[dict[str, Any]]:
    """Every JSON row currently in the sweep log, ignoring a half-written trailing line.

    Args:
        path: Path to `runs.jsonl`.

    Returns:
        One dict per parseable line.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    return rows


def _sweep_state(path: Path, fallback_total: int) -> tuple[int, str]:
    """The total pair count and the pair the sweep last announced, scraped from its stdout.

    Args:
        path: Path to the sweep's captured stdout.
        fallback_total: Pair count to assume until the sweep prints its header.

    Returns:
        The total pair count and a description of the pair currently running.
    """
    if not path.exists():
        path = next((p for p in _SWEEP_LOG_FALLBACKS if p.exists()), path)
    total, current = fallback_total, "(waiting for first pair)"
    if path.exists():
        for line in path.read_text().splitlines():
            if match := _HEADER.search(line):
                total = int(match.group(1))
            elif match := _CURRENT.search(line):
                current = f"[{match.group(1)}/{match.group(2)}] {match.group(3)}"
    return total, current


def _bar(done: int, total: int, width: int = 40) -> str:
    filled = 0 if total == 0 else round(width * done / total)
    return "█" * filled + "░" * (width - filled)


def _fmt_eta(seconds: float) -> str:
    if not seconds > 0:  # also catches NaN
        return "?"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _row(record: dict[str, Any], keys_widths: list[tuple[str, int]], tail: str) -> str:
    cells = [f"{record.get(key, '')!s:{width}}" for key, width in keys_widths]
    return "    " + " ".join(cells) + f" {tail}"


def _frame(rows: list[dict[str, Any]], total: int, current: str, rate: float, eta: float) -> str:
    """The full screen for one refresh, as one string starting with a clear-screen escape.

    Args:
        rows: Every run record logged so far.
        total: Total pairs the sweep will run.
        current: Description of the pair currently running.
        rate: Pairs completed per second since the watcher started.
        eta: Estimated seconds until the sweep finishes.

    Returns:
        The frame text.
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    feasible = [r for r in ok if r.get("solve_total_cost") is not None]
    errors = [r for r in rows if r.get("status") != "ok"]
    done = len(rows)
    pct = 100.0 * done / total if total else 0.0

    lines = [
        "\033[2J\033[H",
        "  UCTTP encoder sweep",
        "",
        f"  {_bar(done, total)}  {done}/{total}  ({pct:5.1f}%)",
        "",
        f"  ok {len(ok)}   feasible {len(feasible)}   errored {len(errors)}"
        f"   rate {rate * 3600:.1f}/h   ETA {_fmt_eta(eta)}",
        f"  now: {current}",
        "",
        "  recent:",
    ]
    shape = [("family", 12), ("instance", 10), ("encoder", 6), ("status", 9)]
    for record in rows[-8:]:
        cost = record.get("solve_total_cost")
        tail = f"cost {cost:g}" if cost is not None else str(record.get("solve_status", ""))
        lines.append(_row(record, shape, tail))
    if errors:
        lines += ["", "  errors:"]
        for record in errors[-5:]:
            reason = str(record.get("error", ""))[:70]
            lines.append(_row(record, [("instance", 10), ("encoder", 6)], reason))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Render the progress frame on a loop until the sweep finishes or the user quits.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("experiments/runs.jsonl"))
    parser.add_argument("--sweep-log", type=Path, default=Path("experiments/sweep.log"))
    parser.add_argument("--total", type=int, default=366, help="fallback pair count")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    args = parser.parse_args(argv)

    start = time.time()
    start_done: int | None = None
    try:
        while True:
            rows = _read_log(args.log)
            total, current = _sweep_state(args.sweep_log, args.total)
            if start_done is None:
                start_done = len(rows)

            elapsed = time.time() - start
            processed = len(rows) - start_done
            rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.0
            eta = (total - len(rows)) / rate if rate > 0 else float("nan")

            print(_frame(rows, total, current, rate, eta), flush=True)
            if args.once or len(rows) >= total:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
