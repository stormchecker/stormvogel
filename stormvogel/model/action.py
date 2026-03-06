"""A model action."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """Represents an action, e.g., in MDPs.
        Note that this action object is completely independent of its corresponding branch.
        Their relation is managed by Choices.
        Two actions with the same labels are considered equal.

    Args:
        label: The label of this action. Corresponds to Storm label.
    """

    label: str | None

    def __post_init__(self):
        if self.label == "":
            object.__setattr__(self, "label", None)

    def __lt__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return str(self.label) < str(other.label)

    def __str__(self):
        return f"Action with label {self.label}"


# The empty action. Used for DTMCs and empty action transitions in mdps.
EmptyAction = Action(None)
