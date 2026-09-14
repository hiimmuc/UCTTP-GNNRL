"""Conflict graph over courses, with each edge recording why the two courses conflict."""

from __future__ import annotations

import networkx as nx

from horarium.problem.instance import Instance

__all__ = ["CURRICULUM", "TEACHER", "build_conflict_graph"]

#: Edge `sources` flags. An edge carries one or both.
CURRICULUM = "curriculum"
TEACHER = "teacher"


def build_conflict_graph(instance: Instance, *, include_teacher: bool = True) -> nx.Graph[int]:
    """Conflict graph over courses: an edge means the two courses cannot share a period.

    Edges come from shared curricula and, if include_teacher, from a shared teacher. Edge
    attribute 'sources' is a frozenset of CURRICULUM and/or TEACHER; 'shared_curricula' counts
    the curricula the pair has in common. Nodes are course indices and carry 'course_id'.

    Args:
        instance: The instance to build the conflict graph for.
        include_teacher: Whether a shared teacher also creates a conflict edge.

    Returns:
        An undirected graph over course indices.
    """
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(
        (index, {"course_id": course.id}) for index, course in enumerate(instance.courses)
    )

    for members in instance.courses_of_curriculum:
        for position, a in enumerate(members):
            for b in members[position + 1 :]:
                if a != b:
                    _link(graph, a, b, CURRICULUM)

    if include_teacher:
        for taught in instance.courses_of_teacher.values():
            for position, a in enumerate(taught):
                for b in taught[position + 1 :]:
                    _link(graph, a, b, TEACHER)

    for _, _, data in graph.edges(data=True):
        data["sources"] = frozenset(data["sources"])
    return graph


def _link(graph: nx.Graph[int], a: int, b: int, source: str) -> None:
    """Add or extend the conflict edge between two courses, recording why it exists.

    Args:
        graph: Graph to add or extend the edge in, mutated in place.
        a: One endpoint's course index.
        b: The other endpoint's course index.
        source: `CURRICULUM` or `TEACHER`, the reason this edge exists.
    """
    edge = graph.edges.get((a, b))
    if edge is None:
        graph.add_edge(a, b, sources={source}, shared_curricula=int(source == CURRICULUM))
        return
    edge["sources"].add(source)
    if source == CURRICULUM:
        edge["shared_curricula"] += 1
