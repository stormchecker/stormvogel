"""Reward model."""

from dataclasses import dataclass
from stormvogel.model.value import Value
from stormvogel.model.state import State
from stormvogel.model.action import Action

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model


@dataclass(eq=True, order=True)
class RewardModel[ValueType: Value]:
    """Represents a state-exit reward model.
    Args:
        name: Name of the reward model.
        model: The model this rewardmodel belongs to.
        rewards: The rewards, the keys state action pairs.
    """

    name: str
    model: "Model"
    rewards: dict[State, ValueType]
    """Rewards dict. Hashed by state id and Action.
    The function update_rewards can be called to update rewards. After this, rewards will correspond to intermediate_rewards.
    Note that in models without actions, EmptyAction will be used here."""

    def __init__(self, name: str, model: "Model", rewards: dict[State, ValueType]):
        self.name = name
        self.rewards = rewards
        self.model = model

    def set_from_rewards_vector(self, vector: list[ValueType]) -> None:
        """Set the rewards of this model according to a (stormpy) state rewards vector."""
        combined_id = 0
        self.rewards = dict()
        for s in self.model:
            if combined_id < len(vector):
                self.rewards[s] = vector[combined_id]
            combined_id += 1

    def get_state_reward(self, state: State) -> ValueType | None:
        """Gets the reward at said state. Return None if no reward is present."""
        if state not in self.rewards:
            return None
        return self.rewards[state]

    def set_state_reward(self, state: State, value: ValueType):
        """Sets the reward at said state."""
        self.rewards[state] = value

    def set_state_action_reward(self, state: State, action: Action, value: ValueType):
        """Backward-compatibility shim. Sets the state reward."""
        self.rewards[state] = value

    def get_state_action_reward(self, state: State, action: Action) -> ValueType | None:
        """Backward-compatibility shim. Gets the state reward."""
        return self.get_state_reward(state)

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
            val_float = float(val) if val is not None else 0.0
            if self.model.supports_actions() and state in self.model.choices:
                for _ in state.available_actions():
                    vector.append(val_float)
            else:
                vector.append(val_float)
        return vector