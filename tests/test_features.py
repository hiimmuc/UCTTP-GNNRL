import numpy as np
import pytest

from horarium.graph.build import CURRICULUM, TEACHER, build_conflict_graph
from horarium.graph.features import encode_instance, feature_dictionary_markdown
from horarium.problem.instance import Instance


def test_graph_edges_match_the_instance_conflict_sets(sample: Instance) -> None:
    graph = build_conflict_graph(sample)
    for course in range(sample.n_courses):
        assert set(graph.neighbors(course)) == sample.conflicts[course]


def test_every_edge_records_why_it_exists(sample: Instance) -> None:
    for u, v, data in build_conflict_graph(sample).edges(data=True):
        assert data["sources"] <= {CURRICULUM, TEACHER}
        assert data["sources"]
        if TEACHER in data["sources"]:
            assert sample.courses[u].teacher == sample.courses[v].teacher
        if CURRICULUM in data["sources"]:
            assert data["shared_curricula"] >= 1
            assert set(sample.curricula_of_course[u]) & set(sample.curricula_of_course[v])


def test_dropping_teacher_edges_leaves_only_curriculum_edges(sample: Instance) -> None:
    without = build_conflict_graph(sample, include_teacher=False)
    assert all(data["sources"] == {CURRICULUM} for *_, data in without.edges(data=True))


def test_features_have_no_nan_or_inf(sample: Instance) -> None:
    """The Phase 4 gate: every sample encodes cleanly, including the Erlangen ones."""
    encoded = encode_instance(sample)
    for block in (encoded.nodes, encoded.edges, encoded.globals):
        assert np.isfinite(block.raw).all(), block.names
        assert np.isfinite(block.values).all(), block.names
    assert encoded.n_nodes == sample.n_courses
    assert encoded.edge_index.shape == (2, encoded.n_edges)


def test_edge_index_is_symmetric(sample: Instance) -> None:
    encoded = encode_instance(sample)
    forward = {(int(u), int(v)) for u, v in encoded.edge_index.T}
    assert all((v, u) in forward for u, v in forward)


def test_zscore_columns_are_standardised(comp01: Instance) -> None:
    encoded = encode_instance(comp01)
    from horarium.graph.features import NODE_FEATURES  # noqa: PLC0415

    for column, spec in enumerate(NODE_FEATURES):
        values = encoded.nodes.values[:, column]
        if spec.normalization == "zscore" and encoded.nodes.raw[:, column].std() > 0:
            assert values.mean() == pytest.approx(0, abs=1e-4)
            assert values.std() == pytest.approx(1, abs=1e-3)


def test_feature_dictionary_lists_every_declared_feature() -> None:
    from horarium.graph.features import (  # noqa: PLC0415
        EDGE_FEATURES,
        GLOBAL_FEATURES,
        NODE_FEATURES,
    )

    document = feature_dictionary_markdown()
    for spec in (*NODE_FEATURES, *EDGE_FEATURES, *GLOBAL_FEATURES):
        assert f"`{spec.name}`" in document
