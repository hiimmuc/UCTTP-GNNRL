"""The five CB-CTT formulations UD1-UD5 as data: which components are active, and at what weight.

Every weight and every hard/soft assignment here is transcribed from the reference validator
(`external/validator/validator.cc`, `Validator::Validator` and `Validator::PrintCosts`), which
is the authority. RoomCapacity is added unweighted in all five, so its weight is 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["FORMULATIONS", "UD1", "UD2", "UD3", "UD4", "UD5", "Formulation", "Hard", "Soft"]


class Hard(StrEnum):
    """Hard constraints, labelled exactly as the validator prints them for line-by-line diffing."""

    LECTURES = "Lectures"
    CONFLICTS = "Conflicts"
    AVAILABILITY = "Availability"
    ROOM_OCCUPATION = "RoomOccupation"
    ROOM_CONSTRAINTS = "RoomConstraints"


class Soft(StrEnum):
    """Soft cost components. The label is exactly what the validator prints."""

    ROOM_CAPACITY = "RoomCapacity"
    MIN_WORKING_DAYS = "MinWorkingDays"
    ISOLATED_LECTURES = "IsolatedLectures"
    CURRICULUM_COMPACTNESS = "CurriculumCompactness"
    ROOM_STABILITY = "RoomStability"
    DOUBLE_LECTURES = "DoubleLectures"
    ROOM_CONSTRAINTS = "RoomConstraints"
    STUDENT_LOAD = "StudentLoad"
    TRAVEL_DISTANCE = "TravelDistance"


#: The four hard constraints every formulation enforces. UD4 adds Hard.ROOM_CONSTRAINTS.
ALWAYS_HARD = (Hard.LECTURES, Hard.CONFLICTS, Hard.AVAILABILITY, Hard.ROOM_OCCUPATION)


@dataclass(frozen=True, slots=True)
class Formulation:
    """One formulation: the soft components it charges for, and whether room constraints bind.

    UD4 is the only formulation where a course sitting in a forbidden room is a hard violation
    rather than a soft cost, which is why this is a field and not just another weight.
    """

    name: str
    weights: Mapping[Soft, int]
    room_constraints_are_hard: bool = False

    @property
    def hard(self) -> tuple[Hard, ...]:
        """Hard constraints active in this formulation, in the validator's print order."""
        if self.room_constraints_are_hard:
            return (*ALWAYS_HARD, Hard.ROOM_CONSTRAINTS)
        return ALWAYS_HARD


UD1 = Formulation(
    name="UD1",
    weights={Soft.ROOM_CAPACITY: 1, Soft.MIN_WORKING_DAYS: 5, Soft.ISOLATED_LECTURES: 1},
)
UD2 = Formulation(
    name="UD2",
    weights={
        Soft.ROOM_CAPACITY: 1,
        Soft.MIN_WORKING_DAYS: 5,
        Soft.ISOLATED_LECTURES: 2,
        Soft.ROOM_STABILITY: 1,
    },
)
UD3 = Formulation(
    name="UD3",
    weights={
        Soft.ROOM_CAPACITY: 1,
        Soft.CURRICULUM_COMPACTNESS: 4,
        Soft.ROOM_CONSTRAINTS: 3,
        Soft.STUDENT_LOAD: 2,
    },
)
UD4 = Formulation(
    name="UD4",
    weights={
        Soft.ROOM_CAPACITY: 1,
        Soft.MIN_WORKING_DAYS: 1,
        Soft.CURRICULUM_COMPACTNESS: 1,
        Soft.DOUBLE_LECTURES: 1,
        Soft.STUDENT_LOAD: 1,
    },
    room_constraints_are_hard=True,
)
UD5 = Formulation(
    name="UD5",
    weights={
        Soft.ROOM_CAPACITY: 1,
        Soft.MIN_WORKING_DAYS: 5,
        Soft.CURRICULUM_COMPACTNESS: 2,
        Soft.ISOLATED_LECTURES: 1,
        Soft.TRAVEL_DISTANCE: 2,
        Soft.STUDENT_LOAD: 2,
    },
)

FORMULATIONS: Mapping[str, Formulation] = {f.name: f for f in (UD1, UD2, UD3, UD4, UD5)}
