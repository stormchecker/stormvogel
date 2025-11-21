from typing import Self, cast
from stormvogel.model.value import Value
from typing import TYPE_CHECKING

from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.branches import Branches
from stormvogel.model.value import Number

if TYPE_CHECKING:
    from stormvogel.model.state import State
    from stormvogel.model.model import Model

class Choices[ValueType: Value]:
    """Represents a choice, which map actions to branches.
        Note that an EmptyAction may be used if we want a non-action choice.
        Note that a single Choices might correspond to multiple 'arrows'.

    Args:
        choice: The choice dictionary. For each available action, we have a branch containing the transitions.
    """

    choices: dict[Action, Branches[ValueType]]

    def __init__(self, choices: dict[Action, Branches[ValueType]]):
        # Input validation, see RuntimeError.
        if len(choices) > 1 and EmptyAction in choices:
            raise RuntimeError(
                "It is impossible to create a choice that contains more than one action, and an emtpy action"
            )
        self.choices = choices
  
    @property
    def actions(self) -> list[Action]:
        """Returns the actions for the choices"""
        return list(self.choices.keys())

    def __str__(self):
        parts = []
        for action, branch in self:
            if action == EmptyAction:
                parts.append(f"{branch}")
            else:
                parts.append(f"{action} => {branch}")
        return "; ".join(parts + [])

    def has_empty_action(self) -> bool:
        # Note that we don't have to deal with the corner case where there are both empty and non-empty choices. This is dealt with at __init__.
        return self.choices.keys() == {EmptyAction}

    def __eq__(self, other):
        if not isinstance(other, Choices):
            return False

        if len(self.choices) != len(other.choices):
            return False

        for action, other_action in zip(
            sorted(self.choices.keys()), sorted(other.choices.keys())
        ):
            if not (
                action == other_action and self.choices[action] == other.choices[action]
            ):
                return False

        return True


    def is_stochastic(self, epsilon: Number) -> bool:
        """Returns whether the probabilities in the branches sum to 1"""
        return all([abs(self.choices[a].branches.is_probabilistic(precision=epsilon) - 1) <= epsilon for a in self.choices])

    def has_zero_transition(self) -> bool:
        """Returns whether any of the branches contains a zero-probability transition."""
        for _, branch in self:
            for transition in branch:
                if isinstance(transition[0], Number) and transition[0] == 0:
                    return True
        return False
    
    def add(self, other: Self):
        """Adds two Choices together, provided they have no overlapping actions."""
        for action, branch in other:
            if action in self.choices:
                raise RuntimeError(
                    "Cannot add two Choices that have overlapping actions."
                )
            else:
                self.choices[action] = branch
    
    def __add__(self, other: Self) -> "Choices[ValueType]":
        new_choices = Choices(self.choices.copy())
        new_choices.add(other)
        return new_choices

    def __getitem__(self, item):
        return self.choices[item]

    def __iter__(self):
        return iter(self.choices.items())

    def __len__(self) -> int:
        return len(self.choices)

ChoicesShorthand = (
    list[tuple[Value, "State[Value]"]]
    | list[tuple[Action, "State[Value]"]]
    | dict[Action, list[tuple[Value, "State[Value]"]]]
)

def choices_from_shorthand[ValueType: Value](shorthand: ChoicesShorthand) -> Choices[ValueType]:
    """Get a Choice object from a ChoicesShorthand. Use for all choices in DTMCs and for empty actions in MDPs.

    There are two possible ways to define a ChoicesShorthand.
    - using only the probability and the target state (implies default action when in an MDP).
    - using only the action and the target state (implies probability=1)."""

    if isinstance(shorthand, dict):
        # We convert the shorthand so that we have states instead of ids
        converted_shorthand = dict()
        for action, branch in shorthand.items():
            converted_shorthand[action] = []
            for value, state in branch:
                converted_shorthand[action].append((value, state))
        shorthand = converted_shorthand

        transition_content = dict()
        for action, branch in shorthand.items():
            assert isinstance(action, Action)
            transition_content[action] = Branches(branch)
        return Choices(transition_content)
    else:
        # We convert the shorthand so that we have states instead of ids
        converted_shorthand = []
        for value_or_action, state in shorthand:
            converted_shorthand.append((value_or_action, state))
        shorthand = converted_shorthand

        # Check the type of the first element
        first_element = shorthand[0][0]
        if isinstance(first_element, Action):
            transition_content = dict()
            for action, state in shorthand:
                assert isinstance(action, Action)
                transition_content[action] = Branches(1, state)
            return Choices(transition_content)
        elif isinstance(first_element, Value):
            return Choices(
                {EmptyAction: Branches(cast(list[tuple[Value, "State"]], shorthand))}
            )
        raise RuntimeError(
            f"Type of {first_element} not supported in choice {shorthand}"
        )