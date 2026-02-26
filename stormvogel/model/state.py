"""A model state."""

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Iterable, Self, TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from stormvogel.model.model import Model

from stormvogel.model.choices import Choices, ChoicesShorthand
from stormvogel.model.distribution import Distribution
from stormvogel.model.observation import Observation
from stormvogel.model.value import Value
from stormvogel.model.action import Action, EmptyAction


@dataclass(order=False, eq=False)
class State[ValueType: Value]:
    """Represents a state in a Model.

    Args:
        id: The id of this state.
        model: The model this state belongs to.
    """

    model: "Model[ValueType]"
    state_id: UUID = UUID(int=0)

    def __init__(self, model: "Model[ValueType]"):
        self.model = model
        self.state_id = uuid4()

    def __setattr__(self, name, value):
        if name == "state_id" and self.state_id != UUID(int=0):
            raise FrozenInstanceError("Cannot modify values of State")

        if name == "model" and self.state_id != UUID(int=0):
            raise FrozenInstanceError("Cannot modify values of State")

        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ["state_id", "model"]:
            raise FrozenInstanceError("Cannot delete properties of State")

        super().__delattr__(name)

    @property
    def labels(self) -> Iterable[str]:
        """Returns an iterator over the state's labels."""
        return (
            label
            for (label, states) in self.model.state_labels.items()
            if self in states
        )

    def set_labels(self, labels: set[str]):
        for label in self.model.state_labels:
            if label in labels and self not in self.model.state_labels[label]:
                self.model.state_labels[label].add(self)
            elif label not in labels and self in self.model.state_labels[label]:
                self.model.state_labels[label].remove(self)

    def has_label(self, label: str):
        """Returns whether this state has this label."""
        if label not in self.model.state_labels:
            return False
        return self in self.model.state_labels[label]

    def add_label(self, label: str):
        """Adds a new label to the state."""
        if label not in self.model.state_labels:
            self.model.add_label(label)
        self.model.state_labels[label].add(self)

    @property
    def observation(self) -> Observation | Distribution[ValueType, Observation] | None:
        """The observation associated with this state."""
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
    def choices(self) -> Choices:
        if self in self.model.choices:
            return self.model.choices[self]
        else:
            raise RuntimeError("The model this state belongs to does not have choices")

    def set_choices(self, choices: Choices | ChoicesShorthand):
        self.model.set_choices(self, choices)

    def add_choices(self, choices: Choices | ChoicesShorthand):
        """Add choices to this state."""
        self.model.add_choices(self, choices)

    def has_choices(self) -> bool:
        """Returns whether this state has choices."""
        if self not in self.model.choices:
            return False
        return len(self.choices.choices) != 0

    @property
    def nr_choices(self) -> int:
        """The number of choices in this state."""
        try:
            choices = self.choices
            n = len(choices)
            return n if n > 0 else 1
        except RuntimeError:
            return 1

    @property
    def valuations(self) -> dict[str, Any]:
        """The valuations of this state."""
        return self.model.state_valuations[self]

    @valuations.setter
    def valuations(self, value: dict[str, Any]):
        self.model.state_valuations[self] = value

    def add_valuation(self, variable: str, value: Any):
        """Adds a valuation to the state."""
        self.valuations[variable] = value

    def get_valuation(self, variable: str) -> Any:
        return self.valuations[variable]

    def available_actions(self) -> list["Action"]:
        """returns the list of all available actions in this state"""
        if self.model.supports_actions():
            return self.choices.actions
        return [EmptyAction]

    def get_outgoing_transitions(
        self, action: Action | None = None
    ) -> list[tuple[Value, Self]] | None:
        """gets the outgoing transitions of this state (after a specific action)"""

        choice = self.choices
        assert choice is not None

        # if the model supports actions we need to provide an action
        if action and self.model.supports_actions():
            if self in self.model.choices:
                return choice.choices[action].branch
        elif not action and self.model.supports_actions():
            raise RuntimeError("You need to provide a specific action")
        else:
            if self in self.model.choices:
                return choice.choices[EmptyAction].branch

    def is_absorbing(self) -> bool:
        """returns whether the state has a nonzero transition going to another state or not"""

        # if the state has no choice it is trivially true
        choice = self.choices
        if choice is None:
            return True

        # for all actions we check if the state has outgoing transitions to a different state with value != 0
        for _, branch in choice:
            for transition in branch:
                if transition[1] != self:
                    return False
        return True

    def is_initial(self):
        """Returns whether this state is initial."""
        return self.has_label("init")

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.state_id == other.state_id

    def __hash__(self):
        return hash(self.state_id)

    def __str__(self):
        res = (
            f"id: {self.state_id}, labels: {self.labels}, valuations: {self.valuations}"
        )
        if self.model.supports_observations() and self.observation is not None:
            res += f", observation: {str(self.observation)}"
        return res
