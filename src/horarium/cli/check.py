"""`python -m horarium.cli.check`: the quality gates a commit must pass.

Runs Ruff (lint + format check), mypy in strict mode, and the test suite, in that order,
reporting each and exiting non-zero if any fails. `--fix` applies Ruff's formatter and safe
autofixes instead of only checking, then still runs mypy and the tests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

__all__ = ["main"]

_SRC = ("src", "tests")


def _run(label: str, argv: list[str]) -> bool:
    """Run one gate as a subprocess, echoing a header and returning whether it passed.

    Args:
        label: Human-readable name printed before the command runs.
        argv: The command to run.

    Returns:
        `True` if the command exited zero.
    """
    print(f"\n=== {label} ===", flush=True)
    return subprocess.run(argv, check=False).returncode == 0  # noqa: S603 - fixed dev tooling


def main(argv: list[str] | None = None) -> int:
    """Run the gates and return a non-zero exit code if any of them fails.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: 0 only if every gate passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="apply Ruff formatting and safe fixes before checking"
    )
    parser.add_argument("--no-tests", action="store_true", help="skip pytest (lint and types only)")
    args = parser.parse_args(argv)

    py = sys.executable
    passed = True

    if args.fix:
        passed &= _run("ruff format", [py, "-m", "ruff", "format", *_SRC])
        passed &= _run("ruff check --fix", [py, "-m", "ruff", "check", "--fix", *_SRC])
    else:
        passed &= _run("ruff check", [py, "-m", "ruff", "check", *_SRC])
        passed &= _run("ruff format --check", [py, "-m", "ruff", "format", "--check", *_SRC])

    passed &= _run("mypy --strict src", [py, "-m", "mypy", "--strict", "src"])

    if not args.no_tests:
        passed &= _run("pytest", [py, "-m", "pytest"])

    print("\nall gates passed" if passed else "\nsome gates failed", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
