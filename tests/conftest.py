from pathlib import Path

import pytest

from horarium.data.ectt import read_ectt
from horarium.problem.instance import Instance

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"

INSTANCE_PATHS = sorted(RAW.rglob("*.ectt"))
_needs_data = pytest.mark.skipif(
    not INSTANCE_PATHS, reason="run `python -m horarium.cli.prepare` to fetch instances"
)


def _identify(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


#: A spread of families and sizes: the per-test corpus. The exhaustive all-61 checks live in
#: `python -m horarium.cli.prepare` (parse + validate) and `python -m horarium.cli.verify_cost`
#: (cost vs the reference validator), not in pytest.
SAMPLE = [
    p
    for p in INSTANCE_PATHS
    if p.stem in {"toy", "test1", "test4", "comp01", "comp05", "Udine1", "EA01", "DDS2"}
]


@pytest.fixture(params=SAMPLE, ids=[_identify(p) for p in SAMPLE])
def sample_path(request: pytest.FixtureRequest) -> Path:
    path: Path = request.param
    return path


@pytest.fixture
def sample(sample_path: Path) -> Instance:
    return read_ectt(sample_path)


@pytest.fixture(scope="session")
def toy() -> Instance:
    return read_ectt(RAW / "Test" / "toy.ectt")


@pytest.fixture(scope="session")
def comp01() -> Instance:
    return read_ectt(RAW / "ITC-2007" / "comp01.ectt")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if {"sample", "sample_path", "toy", "comp01"} & set(item.fixturenames):
            item.add_marker(_needs_data)
