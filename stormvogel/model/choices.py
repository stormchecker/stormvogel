from typing import Self, cast
from stormvogel.model.value import Value
from typing import TYPE_CHECKING

from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.branches import Branches
from stormvogel.model.value import Number, Interval
from stormvogel import parametric
from fractions import Fraction

if TYPE_CHECKING:
    from stormvogel.model.state import State


class Choices[ValueType: Value]:
    """Represent a choice, which maps actions to branches.

    An :class:`EmptyAction` may be used for a non-action choice.
    A single ``Choices`` instance might correspond to multiple 'arrows'.

    :param choices: The choice dictionary. For each available action, a branch
        containing the transitions.
    """

    choices: dict[Action, Branches[ValueType]]

    def __init__(self, choices: dict[Action, Branches[ValueType]]):
        if len(choices) > 1 and EmptyAction in choices:
            raise RuntimeError(
                "It is impossible to create a choice that contains more than one action, and an empty action"
            )
        self.choices = choices

    @property
    def actions(self) -> list[Action]:
        """Return the actions for the choices."""
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

    def is_stochastic(self, epsilon: Number) -> bool:
        """Check whether the probabilities in the branches sum to 1.

        :param epsilon: Tolerance for floating-point comparison.
        :returns: ``True`` if all branches are stochastic within *epsilon*.
        """
        for a in self.choices:
            total = sum(
                v for v, _ in self.choices[a].branches if isinstance(v, (int, float))
            )
            if abs(total - 1) > epsilon:
                return False
        return True

    def has_zero_transition(self) -> bool:
        """Check whether any of the branches contains a zero-probability transition.

        :returns: ``True`` if a zero-probability transition exists.
        """
        for _, branch in self:
            for transition in branch:
                if isinstance(transition[0], Number) and transition[0] == 0:
                    return True
        return False

    def add(self, other: Self):
        """Add another :class:`Choices` to this one in-place.

        :param other: The choices to merge in.
        :raises RuntimeError: If the two choices have incompatible or overlapping actions.
        """
        # Check EmptyAction invariant before adding
        if self.has_empty_action() and not other.has_empty_action():
            raise RuntimeError(
                "You cannot add a choice with an non-empty action to a choice which has an empty action. Use set_choice instead."
            )
        if (
            not self.has_empty_action()
            and len(self.choices) > 0
            and other.has_empty_action()
        ):
            raise RuntimeError(
                "You cannot add a choice with an empty action to a choice which has no empty action. Use set_choice instead."
            )
        for action, branch in other:
            if action in self.choices:
                if action == EmptyAction:
                    # Merge branches for EmptyAction
                    self.choices[action] = self.choices[action] + branch
                else:
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


def choices_from_shorthand[ValueType: Value](
    shorthand: ChoicesShorthand,
) -> Choices[ValueType]:
    """Create a :class:`Choices` object from a :data:`ChoicesShorthand`.

    Two shorthand modes are supported:

    - A list of ``(probability, target_state)`` tuples (implies the default
      action when in an MDP).
    - A list of ``(action, target_state)`` tuples (implies probability 1).
    - A dict mapping actions to lists of ``(probability, target_state)`` tuples.

    :param shorthand: The shorthand representation to convert.
    :returns: A new :class:`Choices` instance.
    """

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
        elif isinstance(
            first_element, (int, float, Fraction, parametric.Parametric, Interval)
        ):
            return Choices(
                {EmptyAction: Branches(cast(list[tuple[Value, "State"]], shorthand))}
            )
        raise RuntimeError(
            f"Type of {first_element} not supported in choice {shorthand}"
        )
