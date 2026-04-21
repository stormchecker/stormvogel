"""A model state."""

from dataclasses import dataclass, field
from typing import Any, Iterable, TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from stormvogel.model.model import Model

from stormvogel.model.choices import Choices, ChoicesShorthand
from stormvogel.model.distribution import Distribution
from stormvogel.model.observation import Observation
from stormvogel.model.value import Value
from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.variable import Variable


@dataclass(order=False, eq=False)
class State[ValueType: Value]:
    """Represent a state in a Model.

    :param model: The model this state belongs to.
    :param state_id: The unique identifier of this state.
    """

    model: "Model[ValueType]"
    state_id: UUID = field(default_factory=uuid4)

    @property
    def labels(self) -> Iterable[str]:
        """Return an iterator over the state's labels."""
        return (
            label
            for (label, states) in self.model.state_labels.items()
            if self in states
        )

    def set_friendly_name(self, friendly_name: str | None) -> None:
        self.model.friendly_names[self] = friendly_name

    @property
    def friendly_name(self) -> str | None:
        """Returns the friendly name of this state."""
        return self.model.friendly_names.get(self, None)

    def set_labels(self, labels: set[str]):
        """Set the labels of this state, adding and removing as needed.

        :param labels: The complete set of labels this state should have.
        """
        for label in labels:
            if label not in self.model.state_labels:
                import warnings

                warnings.warn(
                    f"Label {label} is not in the model's state labels. Adding it to the model."
                )
                self.model.add_label(label)
        for label in self.model.state_labels:
            if label in labels and self not in self.model.state_labels[label]:
                self.model.state_labels[label].add(self)
            elif label not in labels and self in self.model.state_labels[label]:
                self.model.state_labels[label].remove(self)

    def has_label(self, label: str):
        """Check whether this state has the given label.

        :param label: The label to check for.
        :returns: ``True`` if the label is present.
        """
        if label not in self.model.state_labels:
            return False
        return self in self.model.state_labels[label]

    def add_label(self, label: str):
        """Add a new label to this state.

        :param label: The label to add.
        """
        if label not in self.model.state_labels:
            self.model.add_label(label)
        self.model.state_labels[label].add(self)

    @property
    def observation(self) -> Observation | Distribution[ValueType, Observation] | None:
        """Return the observation associated with this state.

        :raises RuntimeError: If the model does not support observations.
        """
        if (
            self.model.supports_observations()
            and self.model.state_observations is not None
        ):
            return self.model.state_observations[self]
        else:
            raise RuntimeError(
                "The model this state belongs to does not have observations"
            )

    @observation.setter
    def observation(
        self, observation: Observation | Distribution[ValueType, Observation]
    ):
        if (
            self.model.supports_observations()
            and self.model.state_observations is not None
        ):
            self.model.state_observations[self] = observation
        else:
            raise RuntimeError(
                "The model this state belongs to does not have observations"
            )

    @property
    def choices(self) -> "Choices[ValueType]":
        """Return the choices for this state.

        :raises RuntimeError: If no choices exist for this state.
        """
        if self in self.model.transitions:
            return self.model.transitions[self]
        else:
            raise RuntimeError("The model this state belongs to does not have choices")

    def set_choices(self, choices: Choices | ChoicesShorthand):
        """Set the choices for this state.

        :param choices: The choices to set.
        """
        self.model.set_choices(self, choices)

    def add_choices(self, choices: Choices | ChoicesShorthand):
        """Add choices to this state.

        :param choices: The choices to add.
        """
        self.model.add_choices(self, choices)

    def has_choices(self) -> bool:
        """Check whether this state has choices."""
        if self not in self.model.transitions:
            return False
        return len(self.choices) != 0

    @property
    def nr_choices(self) -> int:
        """The number of choices in this state."""
        if self.has_choices():
            n = len(self.choices)
            return n if n > 0 else 1
        else:
            return 1

    @property
    def valuations(self) -> dict[Variable, Any]:
        """The valuations of this state."""
        return self.model.state_valuations[self]

    @valuations.setter
    def valuations(self, value: dict[Variable, Any]):
        self.model.state_valuations[self] = value

    def add_valuation(self, variable: Variable, value: Any):
        """Add a valuation to this state.

        :param variable: The variable name.
        :param value: The value for the variable.
        """
        self.valuations[variable] = value

    def get_valuation(self, variable: Variable) -> Any:
        """Return the valuation for the given variable.

        :param variable: The variable name.
        :returns: The value associated with the variable.
        """
        return self.valuations[variable]

    def available_actions(self) -> list["Action"]:
        """Return the list of available actions in this state."""
        if self.model.supports_actions():
            return self.choices.actions
        return [EmptyAction]

    def get_branches(
        self, action: Action | None = None
    ) -> Distribution[ValueType, "State[ValueType]"]:
        """Get the branches of this state for a specific action.

        :param action: The action to get branches for.
        :returns: The branches.
        :raises RuntimeError: If the requested action does not exist for this state.
        """
        if self not in self.model.transitions:
            raise RuntimeError("This state does not have any choices in the model.")

        if self.model.supports_actions():
            if action is None:
                if EmptyAction in self.choices.actions:
                    return self.choices[EmptyAction]
                raise RuntimeError("You need to provide a specific action")
            if action in self.choices.actions:
                return self.choices[action]
            raise RuntimeError(f"Action {action} not found in choices for this state")

        if action is not None and action != EmptyAction:
            raise RuntimeError("This state does not support non-empty actions")
        return self.choices[EmptyAction]

    def get_outgoing_transitions(
        self, action: Action | None = None
    ) -> Distribution[ValueType, "State[ValueType]"] | None:
        """Get the outgoing transitions of this state for a specific action.

        :param action: The action to get transitions for.
        :returns: The distribution over successor states, or ``None`` if not found.
        :raises RuntimeError: If the model supports actions but none is provided.
        """
        if self.model.supports_actions():
            if action is None:
                raise RuntimeError("You need to provide a specific action")
            if self in self.model.transitions and action in self.choices.actions:
                return self.choices[action]
            return None

        if self in self.model.transitions and EmptyAction in self.choices.actions:
            return self.choices[EmptyAction]
        return None

    def is_absorbing(self) -> bool:
        """Check whether this state is absorbing (no nonzero transitions to other states)."""

        # if the state has no choice it is trivially true
        if self not in self.model.transitions:
            return True
        choice = self.choices
        # for all actions we check if the state has outgoing transitions to a different state with value != 0
        for _, branch in choice:
            for transition in branch:
                if transition[1] != self:
                    return False
        return True

    def is_initial(self):
        """Check whether this state is the initial state."""
        return self.has_label("init")

    def __repr__(self):
        name = self.friendly_name
        labels = list(self.labels)
        obs = None
        if (
            self.model.supports_observations()
            and self.model.state_observations is not None
        ):
            obs = self.model.state_observations.get(self)
        base = (
            f"State({name!r}, labels={labels!r})"
            if name
            else f"State(id={self.state_id}, labels={labels!r})"
        )
        if obs is not None:
            return base[:-1] + f", observation={obs!r})"
        return base

    __str__ = __repr__
