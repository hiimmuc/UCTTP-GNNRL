import ast
from pathlib import Path

import pytest

PROBLEM = Path(__file__).resolve().parents[1] / "src" / "horarium" / "problem"
BANNED_ROOTS = {
    "torch",
    "torch_geometric",
    "numpy",
    "scipy",
    "gymnasium",
    "ortools",
    "horarium.data",
    "horarium.solvers",
}


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("source", sorted(PROBLEM.glob("*.py")), ids=lambda p: p.name)
def test_problem_package_stays_pure(source: Path) -> None:
    """problem/ is plain semantics over plain data: no torch, no arrays, no IO modules."""
    offending = {
        name
        for name in _imported_modules(source)
        if any(name == root or name.startswith(f"{root}.") for root in BANNED_ROOTS)
    }
    assert not offending, f"{source.name} imports {sorted(offending)}"
