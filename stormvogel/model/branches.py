"""Branches are distributions over states."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stormvogel.model.value import Value
from stormvogel.model.distribution import Distribution

if TYPE_CHECKING:
    from stormvogel.model.state import State


@dataclass(order=True, eq=True)
class Branches[ValueType: Value]:
    """Represents branches, which is a distribution over states.
    Args:
        branch: The distribution over successors.
    """

    branch: "Distribution[ValueType, State[ValueType]]"

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.branch = args[0]
        elif len(args) == 2:
            self.branch = [(args[0], args[1])]
        else:
            raise TypeError(
                "expects either (list of (value,state) tuples) or (value, state)"
            )

    @property
    def successors(self) -> set["State[ValueType]"]:
        """Returns the set of successor states."""
        return set(s for _, s in self.branch)

    def __str__(self):
        parts = []
        for prob, state in self.branch:
            parts.append(f"{prob} -> {state}")
        return ", ".join(parts)

    def __add__(self, other):
        if not isinstance(other, Branches):
            raise TypeError("Can only add Branches to Branches")
        return Branches(self.branch + other.branch)

    def __iter__(self):
        return iter(self.branch)

    def sort_states(self):
        """Sorts the branch list by the state's position in model.states."""
        self.branch.sort(key=lambda t: t[1].model.states.index(t[1]))
