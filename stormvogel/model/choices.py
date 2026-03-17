from typing import Self, cast
from typing import TYPE_CHECKING

from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.value import Value, is_zero
from stormvogel.model.distribution import Distribution

if TYPE_CHECKING:
    from stormvogel.model.state import State


class Choices[ValueType: Value]:
    """Represents a choice, which map actions to branches.
        Note that an EmptyAction may be used if we want a non-action choice.
        Note that a single Choices might correspond to multiple 'arrows'.

    Args:
        choice: The choice dictionary. For each available action, we have a branch containing the transitions.
    """

    _choices: dict[Action, Distribution[ValueType, State[ValueType]]]

    def __init__(
        self, choices: dict[Action, Distribution[ValueType, State[ValueType]]]
    ):
        if len(choices) > 1 and EmptyAction in choices:
            raise RuntimeError(
                "It is impossible to create a choice that contains more than one action, and an empty action"
            )
        self._choices = choices

    @property
    def actions(self) -> list[Action]:
        """Returns the actions for the choices"""
        return list(self._choices.keys())

    def __str__(self):
        parts = []
        for action, branch in self:
            if action == EmptyAction:
                parts.append(f"{branch}")
            else:
                parts.append(f"{action} => {branch}")
        return "; ".join(parts)

    def has_empty_action(self) -> bool:
        return EmptyAction in self._choices.keys()

    def is_stochastic(self, epsilon=1e-6) -> bool:
        """Returns whether the probabilities in the branches sum to 1"""
        for a in self._choices:
            if not self._choices[a].is_stochastic(epsilon):
                return False
        return True

    def has_zero_transition(self) -> bool:
        """Returns whether any of the branches contains a zero-probability transition."""
        for _, branch in self:
            for transition in branch:
                if is_zero(transition[0]):
                    return True
        return False

    def add(self, other: Self):
        """Adds two Choices together, provided they have no overlapping actions."""
        # Check EmptyAction invariant before adding
        if self.has_empty_action() and not other.has_empty_action():
            raise RuntimeError(
                "You cannot add a choice with an non-empty action to a choice which has an empty action. Use set_choice instead."
            )
        if (
            not self.has_empty_action()
            and len(self._choices) > 0
            and other.has_empty_action()
        ):
            raise RuntimeError(
                "You cannot add a choice with an empty action to a choice which has no empty action. Use set_choice instead."
            )
        for action, branch in other:
            if action in self._choices:
                if action == EmptyAction:
                    # Merge branches for EmptyAction
                    self._choices[action] = self._choices[action] + branch
                else:
                    raise RuntimeError(
                        "Cannot add two Choices that have overlapping actions."
                    )
            else:
                self._choices[action] = branch

    def __add__(self, other: Self) -> "Choices[ValueType]":
        new_choices = Choices(self._choices)
        new_choices.add(other)
        return new_choices

    def __iter__(self):
        return iter(self._choices.items())

    def __len__(self) -> int:
        return len(self._choices)

    def __setitem__(self, key, value):
        # These sanity checks are intended to preserve the DTMC / MDP semantics.
        if key is None:
            raise ValueError("Action cannot be None.")
        if key != EmptyAction and self.has_empty_action():
            raise RuntimeError(
                "You cannot set a non-empty action on a choice that has an empty action."
            )
        if (
            key == EmptyAction
            and not self.has_empty_action()
            and len(self._choices) > 0
        ):
            raise RuntimeError(
                "You cannot set an empty action on a choice that has non-empty actions."
            )
        self._choices[key] = value

    def __getitem__(self, item):
        return self._choices[item]

    def __delitem__(self, key):
        del self._choices[key]


ChoicesShorthand = (
    list[tuple[Value, "State[Value]"]]
    | list[tuple[Action, "State[Value]"]]
    | dict[Action, list[tuple[Value, "State[Value]"]]]
)


def choices_from_shorthand(
    shorthand: ChoicesShorthand,
) -> Choices[Value]:
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
            transition_content[action] = Distribution(branch)
        return Choices(transition_content)
    else:
        if not shorthand:
            raise ValueError(
                "Cannot create Choices from an empty list shorthand. "
                "Provide at least one (value, state) or (action, state) tuple."
            )

        # Check the type of the first element
        first_element = shorthand[0][0]
        if isinstance(first_element, Action):
            transition_content = dict()
            for action, state in shorthand:
                assert isinstance(action, Action)
                transition_content[action] = Distribution([(1, state)])
            return Choices(transition_content)
        elif isinstance(first_element, Value):
            return Choices(
                {
                    EmptyAction: Distribution(
                        cast(list[tuple[Value, "State"]], shorthand)
                    )
                }
            )
        raise RuntimeError(
            f"Type of {first_element} not supported in choice {shorthand}"
        )
