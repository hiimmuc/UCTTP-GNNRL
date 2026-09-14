"""Bridge to the compiled reference validator: run it, parse its output, compare to `cost`."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from horarium.problem.cost import CostBreakdown
from horarium.problem.formulations import Hard, Soft

__all__ = [
    "ValidatorError",
    "ValidatorReport",
    "compare",
    "default_validator_path",
    "run_validator",
]

_HARD = re.compile(r"^Violations of (\w+) \(hard\) : (\d+)$")
_SOFT = re.compile(r"^Cost of (\w+) \(soft\) : (\d+)$")


class ValidatorError(RuntimeError):
    """The validator binary is missing, failed, or produced output this module cannot parse."""


@dataclass(frozen=True, slots=True)
class ValidatorReport:
    """Per-constraint numbers the validator printed for one instance, solution and formulation."""

    formulation: str
    hard: dict[Hard, int]
    soft: dict[Soft, int]
    warnings: int

    @property
    def total(self) -> int:
        """Total weighted soft cost, as the validator sums it."""
        return sum(self.soft.values())

    @property
    def violations(self) -> int:
        """Total hard violations."""
        return sum(self.hard.values())


def default_validator_path() -> Path:
    """Path `scripts/setup.sh` compiles the binary to.

    Returns:
        The expected path of the compiled validator binary.
    """
    return Path(__file__).resolve().parents[3] / "external" / "validator" / "validator"


def run_validator(
    formulation: str,
    instance_path: str | Path,
    solution_path: str | Path,
    *,
    binary: Path | None = None,
) -> ValidatorReport:
    """Run the compiled validator and parse every constraint line it prints.

    Args:
        formulation: Which formulation to validate against, e.g. `"UD2"`.
        instance_path: Path to the .ectt instance.
        solution_path: Path to the .sol solution.
        binary: Path to the validator executable. Defaults to `default_validator_path()`.

    Returns:
        The per-constraint numbers the validator printed.

    Raises:
        ValidatorError: The binary is missing, exits non-zero, or prints no cost lines.
    """
    executable = binary or default_validator_path()
    if not executable.exists():
        msg = f"validator not built at {executable}; run scripts/setup.sh"
        raise ValidatorError(msg)
    completed = subprocess.run(  # noqa: S603
        [str(executable), formulation, str(instance_path), str(solution_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        msg = f"validator exited {completed.returncode}: {completed.stderr.strip()}"
        raise ValidatorError(msg)

    hard: dict[Hard, int] = {}
    soft: dict[Soft, int] = {}
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if match := _HARD.match(stripped):
            hard[Hard(match.group(1))] = int(match.group(2))
        elif match := _SOFT.match(stripped):
            soft[Soft(match.group(1))] = int(match.group(2))
    if not soft:
        msg = f"validator printed no cost lines:\n{completed.stdout}\n{completed.stderr}"
        raise ValidatorError(msg)
    return ValidatorReport(
        formulation=formulation,
        hard=hard,
        soft=soft,
        warnings=completed.stderr.count("WARNING:"),
    )


def compare(ours: CostBreakdown, theirs: ValidatorReport) -> list[str]:
    """Describe every component where our breakdown and the validator's disagree.

    Args:
        ours: Our own cost breakdown for a solution.
        theirs: The reference validator's report for the same solution.

    Returns:
        One message per disagreeing component; empty means the two agree exactly.
    """
    problems = []
    if ours.formulation != theirs.formulation:
        problems.append(f"formulation {ours.formulation} vs {theirs.formulation}")
    for key in sorted(set(ours.hard) | set(theirs.hard), key=str):
        mine, yours = ours.hard.get(key), theirs.hard.get(key)
        if mine != yours:
            problems.append(f"hard {key.value}: ours={mine} validator={yours}")
    for soft_key in sorted(set(ours.soft) | set(theirs.soft), key=str):
        mine, yours = ours.soft.get(soft_key), theirs.soft.get(soft_key)
        if mine != yours:
            problems.append(f"soft {soft_key.value}: ours={mine} validator={yours}")
    if ours.total != theirs.total:
        problems.append(f"total: ours={ours.total} validator={theirs.total}")
    return problems
