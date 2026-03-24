"""Reward model."""

from dataclasses import dataclass
from stormvogel.model.value import Value
from stormvogel.model.state import State

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model


@dataclass(eq=False)
class RewardModel[ValueType: Value]:
    """Represent a state-exit reward model.

    :param name: Name of the reward model.
    :param model: The model this reward model belongs to.
    :param rewards: Mapping from states to their reward values.
    """

    name: str
    model: "Model"
    rewards: dict[State, ValueType]

    def set_from_rewards_vector(
        self, vector: list[ValueType], state_action: bool = False
    ) -> None:
        """Set the rewards of this model according to a (stormpy) rewards vector.

        :param vector: The reward values.
        :param state_action: If ``True``, the vector has one entry per (state, action)
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
            if combined_id >= len(vector):
                raise ValueError(
                    f"Reward vector too short: expected entry at index {combined_id} "
                    f"for state {s}, but vector has length {len(vector)}."
                )
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
        """Get the reward at the given state.

        :param state: The state to look up.
        :returns: The reward value, or ``None`` if no reward is present.
        """
        if state not in self.rewards:
            return None
        return self.rewards[state]

    def set_state_reward(self, state: State, value: ValueType):
        """Set the reward at the given state.

        :param state: The state to set the reward for.
        :param value: The reward value to assign.
        """
        self.rewards[state] = value

    def set_unset_rewards(self, value: ValueType):
        """Fill up rewards that were not set yet with the specified value.

        :param value: The default reward value to assign to unset states.
        """
        for s in self.model:
            if s not in self.rewards:
                self.rewards[s] = value

    def __iter__(self):
        return iter(self.rewards.items())

    def get_reward_vector(self) -> list[float]:
        """Return a list of all rewards ordered appropriately.

        :returns: A flat list of reward values as floats.
        """
        if any(not isinstance(val, (int, float)) for val in self.rewards.values()):
            raise RuntimeError(
                "Cannot get reward vector if not all rewards are numeric."
            )
        numeric_rewards: dict[State, float] = {
            state: float(val)
            for state, val in self.rewards.items()
            if isinstance(val, (int, float))
        }
        vector = []
        for state in self.model:
            val_float = numeric_rewards.get(state, 0.0)
            if self.model.supports_actions() and state in self.model.transitions:
                for _ in state.available_actions():
                    vector.append(val_float)
            else:
                vector.append(val_float)
        return vector
