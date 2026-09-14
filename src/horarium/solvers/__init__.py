"""Algorithms that build or improve timetables, as opposed to `problem/`, which defines them.

`problem/` is pure semantics over plain data and is purity-tested to stay that way. Anything
that *searches* -- the stage-2 room oracle, the construction heuristics, the annealer -- lives
here instead, and may use numpy and scipy.
"""

from __future__ import annotations

from horarium.solvers.anneal import AnnealConfig, AnnealResult, anneal
from horarium.solvers.construction import Construction, Order, dsatur, random_order
from horarium.solvers.rooms import (
    RoomAssignmentError,
    assign_rooms,
    cheapest_room,
    reassign_rooms,
    usable_rooms,
)

__all__ = [
    "AnnealConfig",
    "AnnealResult",
    "Construction",
    "Order",
    "RoomAssignmentError",
    "anneal",
    "assign_rooms",
    "cheapest_room",
    "dsatur",
    "random_order",
    "reassign_rooms",
    "usable_rooms",
]
