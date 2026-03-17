"""Reward model."""

from dataclasses import dataclass
from stormvogel.model.value import Value
from stormvogel.model.state import State

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model


@dataclass(eq=False)
class RewardModel[ValueType: Value]:
    """Represents a state-exit reward model.
    Args:
        name: Name of the reward model.
        model: The model this rewardmodel belongs to.
        rewards: The rewards, the keys states.
    """

    name: str
    model: "Model"
    rewards: dict[State, ValueType]

    def __init__(self, name: str, model: "Model", rewards: dict[State, ValueType]):
        self.name = name
        self.rewards = rewards
        self.model = model

    def set_from_rewards_vector(
        self, vector: list[ValueType], state_action: bool = False
    ) -> None:
        """Set the rewards of this model according to a (stormpy) rewards vector.

        Args:
            vector: The reward values.
            state_action: If True, the vector has one entry per (state, action)
                pair; only the first entry for each state is used.
        """
        if state_action:
            # Give a warning if this flag is true
            import warnings

            warnings.warn(
                "Warning: Only using first entry of state-action reward vector."
            )
        combined_id = 0
        self.rewards = dict()
        for s in self.model:
            if combined_id < len(vector):
                self.rewards[s] = vector[combined_id]
            if (
                state_action
                and self.model.supports_actions()
                and s in self.model.transitions
            ):
                combined_id += len(list(s.available_actions()))
            else:
                combined_id += 1

    def get_state_reward(self, state: State) -> ValueType | None:
        """Gets the reward at said state. Return None if no reward is present."""
        if state not in self.rewards:
            return None
        return self.rewards[state]

    def set_state_reward(self, state: State, value: ValueType):
        """Sets the reward at said state."""
        self.rewards[state] = value

    def set_unset_rewards(self, value: ValueType):
        """Fills up rewards that were not set yet with the specified value."""
        for s in self.model:
            if s not in self.rewards:
                self.rewards[s] = value

    def __iter__(self):
        return iter(self.rewards.items())

    def get_reward_vector(self) -> list[float]:
        """Returns a list of all rewards ordered appropriately."""
        vector = []
        for state in self.model:
            val = self.get_state_reward(state)
            if any(not isinstance(val, (int, float)) for val in self.rewards.values()):
                raise RuntimeError(
                    "Cannot get reward vector if not all rewards are numeric."
                )
            val_float = (
                float(val) if val is not None and isinstance(val, (int, float)) else 0.0
            )
            if self.model.supports_actions() and state in self.model.transitions:
                for _ in state.available_actions():
                    vector.append(val_float)
            else:
                vector.append(val_float)
        return vector
