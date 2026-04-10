"""A model action."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """Represent an action, e.g., in MDPs.

    The action object is independent of its corresponding branch;
    their relation is managed by :class:`Choices`.
    Two actions with the same label are considered equal.

    :param label: The label of this action. Corresponds to a Storm label.
    """

    label: str | None

    def __post_init__(self):
        if self.label == "":
            object.__setattr__(self, "label", None)

    def __lt__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        if self.label is None and other.label is not None:
            return True
        return str(self.label) < str(other.label)

    def __repr__(self):
        return f"Action({self.label!r})"

    __str__ = __repr__


# The empty action. Used for DTMCs and empty action transitions in mdps.
EmptyAction = Action(None)
