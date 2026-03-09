"""Branches are distributions over states."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stormvogel.model.value import Value
from stormvogel.model.distribution import Distribution

if TYPE_CHECKING:
    from stormvogel.model.state import State


@dataclass(eq=False)
class Branches[ValueType: Value]:
    """Represent branches, which is a distribution over states.

    :param branches: The distribution over successors.
    """

    branches: "Distribution[ValueType, State[ValueType]]"

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.branches = Distribution(args[0])
        elif len(args) == 1 and isinstance(args[0], Distribution):
            self.branches = args[0]
        elif len(args) == 2:
            self.branches = Distribution([(args[0], args[1])])
        else:
            raise TypeError(
                "expects either (list of (value,state) tuples) or (value, state)"
            )

    @property
    def successors(self) -> set["State[ValueType]"]:
        """Return the set of successor states."""
        return set(s for _, s in self.branches)

    def __str__(self):
        parts = []
        for prob, state in self.branches:
            parts.append(f"{prob} -> {state}")
        return ", ".join(parts)

    def __add__(self, other):
        if not isinstance(other, Branches):
            raise TypeError("Can only add Branches to Branches")
        return Branches(self.branches + other.branches)

    def __iter__(self):
        return iter(self.branches)

    def sort_states(self):
        """Sort the distribution by the state's position in model.states."""
        self.branches.sort()
