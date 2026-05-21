from typing import Self, cast
from typing import TYPE_CHECKING

from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.value import Value, is_zero
from stormvogel.model.distribution import Distribution

if TYPE_CHECKING:
    from stormvogel.model.state import State


ChoicesShorthand = (
    list[tuple[Value, "State[Value]"]]
    | list[tuple[Action, "State[Value]"]]
    | dict[Action, list[tuple[Value, "State[Value]"]]]
)


class Choices[ValueType: Value]:
    """Represent a choice, which maps actions to branches.

    An :class:`EmptyAction` may be used for a non-action choice.
    A single ``Choices`` instance might correspond to multiple 'arrows'.

    :param choices: The choice dictionary. For each available action, a branch
        containing the transitions.
    """

    _choices: dict[Action, Distribution[ValueType, "State[ValueType]"]]

    def __init__(
        self, choices: dict[Action, Distribution[ValueType, "State[ValueType]"]]
    ):
        if len(choices) > 1 and EmptyAction in choices:
            raise RuntimeError(
                "It is impossible to create a choice that contains more than one action, and an empty action"
            )
        self._choices = choices

    @property
    def actions(self) -> list[Action]:
        """Return the actions for the choices."""
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
        """Check whether the probabilities in the branches sum to 1.

        :param epsilon: Tolerance for floating-point comparison.
        :returns: ``True`` if all branches are stochastic within *epsilon*.
        """
        for a in self._choices:
            if not self._choices[a].is_stochastic(epsilon):
                return False
        return True

    def has_zero_transition(self) -> bool:
        """Check whether any of the branches contains a zero-probability transition.

        :returns: ``True`` if a zero-probability transition exists.
        """
        for _, branch in self:
            for transition in branch:
                if is_zero(transition[0]):
                    return True
        return False

    def add(self, other_choices: Self | ChoicesShorthand):
        """Add another :class:`Choices` to this one in-place.

        :param other: The choices to merge in.
        :raises RuntimeError: If the two choices have incompatible or overlapping actions.
        """

        if not isinstance(other_choices, Choices):
            other = cast("Choices[ValueType]", choices_from_shorthand(other_choices))
        else:
            other = other_choices

        if other.has_zero_transition():
            raise RuntimeError("All transition probabilities should be nonzero.")

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
                raise RuntimeError(
                    f"Cannot add choices with overlapping actions. Action {action} is in both choices."
                )
            self._choices[action] = branch

    def __add__(self, other: Self | ChoicesShorthand) -> "Choices[ValueType]":
        new_choices = Choices(self._choices)
        new_choices.add(other)
        return new_choices

    def __iter__(self):
        return iter(self._choices.items())

    def __len__(self) -> int:
        return len(self._choices)

    def __setitem__(self, key, value):
        # These sanity checks are intended to preserve the invariant that a choice cannot have both an empty action and a non-empty action.
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


def choices_from_shorthand(
    shorthand: ChoicesShorthand,
) -> Choices[Value]:
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
