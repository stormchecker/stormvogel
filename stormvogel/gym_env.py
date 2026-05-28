"""Gymnasium-compliant environment wrapping a stormvogel MDP, DTMC, POMDP, or HMM."""

from typing import Any, Literal

import gymnasium as gym

import stormvogel.model
import stormvogel.simulator
from stormvogel.model.observation import Observation
from stormvogel.model.variable import (
    Variable,
    BoolDomain,
    IntDomain,
    VariableDomain,
)


class ActionUnavailableError(Exception):
    """Raised when the chosen action is not available in the current state."""


def _domain_size(domain: VariableDomain) -> int:
    extra = 1 if domain.allow_none else 0
    if isinstance(domain, IntDomain):
        return domain.hi - domain.lo + 1 + extra
    if isinstance(domain, BoolDomain):
        return 2 + extra
    return len(domain.values) + extra  # CategoricalDomain


def _encode_value(domain: VariableDomain, value: Any) -> int:
    """Encode a single domain value as a non-negative integer for gymnasium."""
    if value is None:
        return _domain_size(domain) - 1
    if isinstance(domain, IntDomain):
        return int(value) - domain.lo
    if isinstance(domain, BoolDomain):
        return int(bool(value))
    return domain.values.index(value)  # CategoricalDomain


class ModelEnv(gym.Env):
    """Gymnasium environment wrapping a stormvogel MDP, DTMC, POMDP, or HMM.

    Two observation modes are supported, selected by ``obs_type``:

    - ``"index"`` (default):
      - MDP/DTMC: ``Discrete(n_states)`` — integer index of the current state.
      - POMDP/HMM: ``Discrete(n_observations)`` — integer index of the
        observation associated with the current state.
    - ``"valuations"``:
      - MDP/DTMC: ``Dict`` space built from state variables that carry a
        declared domain (``Variable.domain is not None``).
      - POMDP/HMM: ``Dict`` space built from observation variables that carry
        a declared domain (``Variable.domain is not None``).
      Each variable becomes a ``Discrete`` component; values are encoded as
      non-negative integers. ``IntDomain(lo, hi)``: value ``v`` → ``v - lo``;
      ``BoolDomain``: ``False→0``, ``True→1``; ``CategoricalDomain``: index
      into ``domain.values``. ``None`` maps to the last index when
      ``allow_none`` is set. Missing variable values are raised as
      ``ValueError`` at step time.

    For **MDP/POMDP** models the action space is ``Discrete(n_actions)`` where
    actions are indexed in the order returned by ``model.actions``.

    For **DTMC/HMM** models the action space is ``Discrete(1)``; the single
    legal action index is 0 (mapped internally to ``EmptyAction``).

    Stochastic observations (``Distribution`` over observations) are not
    supported and raise ``NotImplementedError`` at step time.

    :param model: A stormvogel model of type MDP, DTMC, POMDP, or HMM.
    :param reward_model_name: Name of the reward model to use for the step
        reward. If ``None`` and exactly one reward model exists it is used
        automatically. If ``None`` and multiple reward models exist a
        ``ValueError`` is raised. No reward models → reward is always 0.
    :param obs_type: ``"index"`` for a flat observation space, ``"valuations"``
        for a ``Dict`` space built from variable domains.
    :raises ValueError: If ``obs_type="valuations"`` but no domain-bearing
        variables are found.
    """

    def __init__(
        self,
        model: stormvogel.model.Model,
        reward_model_name: str | None = None,
        obs_type: Literal["index", "valuations"] = "index",
    ):
        supported = {
            stormvogel.model.ModelType.MDP,
            stormvogel.model.ModelType.DTMC,
            stormvogel.model.ModelType.POMDP,
            stormvogel.model.ModelType.HMM,
        }
        if model.model_type not in supported:
            raise ValueError(
                f"ModelEnv requires a model of type MDP, DTMC, POMDP, or HMM, "
                f"got {model.model_type}"
            )

        self.model = model
        self._obs_type = obs_type
        self._is_obs_model = model.supports_observations()

        self._index_to_state: list[stormvogel.model.State] = list(model.states)
        self._state_to_index: dict[stormvogel.model.State, int] = {
            s: i for i, s in enumerate(self._index_to_state)
        }

        if self._is_obs_model:
            self._index_to_observation: list[stormvogel.model.Observation] = list(
                model.observations
            )
            self._observation_to_index: dict[stormvogel.model.Observation, int] = {
                o: i for i, o in enumerate(self._index_to_observation)
            }

        if model.supports_actions():
            self._index_to_action: list[stormvogel.model.Action] = list(model.actions)
            self._action_to_index: dict[stormvogel.model.Action, int] = {
                a: i for i, a in enumerate(self._index_to_action)
            }
            self.action_space = gym.spaces.Discrete(len(self._index_to_action))
        else:
            self._index_to_action = []
            self._action_to_index = {}
            self.action_space = gym.spaces.Discrete(1)

        if len(model.rewards) == 0:
            self._reward_model: stormvogel.model.RewardModel | None = None
        elif reward_model_name is None:
            if len(model.rewards) > 1:
                names = [r.name for r in model.rewards]
                raise ValueError(
                    f"Model has multiple reward models {names}; "
                    "specify reward_model_name."
                )
            self._reward_model = model.rewards[0]
        else:
            matches = [r for r in model.rewards if r.name == reward_model_name]
            if not matches:
                raise ValueError(f"No reward model named {reward_model_name!r}.")
            self._reward_model = matches[0]

        if obs_type == "valuations":
            self._obs_vars: list[Variable] = self._collect_obs_vars()
            if not self._obs_vars:
                raise ValueError(
                    "obs_type='valuations' requires at least one variable with a "
                    "declared domain across the model's observations."
                    if self._is_obs_model
                    else "obs_type='valuations' requires at least one variable with a "
                    "declared domain across the model's states."
                )
            self.observation_space = gym.spaces.Dict(
                {
                    var.label: gym.spaces.Discrete(_domain_size(var.domain))  # type: ignore[arg-type]
                    for var in self._obs_vars
                }
            )
        else:
            self._obs_vars = []
            if self._is_obs_model:
                self.observation_space = gym.spaces.Discrete(
                    len(self._index_to_observation)
                )
            else:
                self.observation_space = gym.spaces.Discrete(len(self._index_to_state))

        self._current_state: stormvogel.model.State = model.initial_state

    def _collect_obs_vars(self) -> list[Variable]:
        """Return domain-bearing variables in sorted label order."""
        seen: set[str] = set()
        result: list[Variable] = []
        if self._is_obs_model:
            sources = list(self.model.observation_valuations.values())
        else:
            sources = [s.valuations for s in self.model.states]
        for valuation in sources:
            for var in valuation:
                if (
                    isinstance(var, Variable)
                    and var.domain is not None
                    and var.label not in seen
                ):
                    seen.add(var.label)
                    result.append(var)
        result.sort(key=lambda v: v.label)
        return result

    def _encode_obs(self, state: stormvogel.model.State) -> int | dict[str, int]:
        if self._is_obs_model:
            observation = state.observation
            if not isinstance(observation, Observation):
                raise NotImplementedError(
                    "ModelEnv does not support stochastic (Distribution) observations."
                )
            if self._obs_type == "index":
                return self._observation_to_index[observation]
            vals = observation.valuations
            obs: dict[str, int] = {}
            for var in self._obs_vars:
                if var not in vals:
                    raise ValueError(
                        f"Observation {observation!r} has no value for variable {var!r}."
                    )
                obs[var.label] = _encode_value(var.domain, vals[var])  # type: ignore[arg-type]
            return obs
        if self._obs_type == "index":
            return self._state_to_index[state]
        vals = state.valuations
        obs = {}
        for var in self._obs_vars:
            if var not in vals:
                raise ValueError(f"State {state!r} has no value for variable {var!r}.")
            obs[var.label] = _encode_value(var.domain, vals[var])  # type: ignore[arg-type]
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int | dict[str, int], dict[str, Any]]:
        """Reset the environment to the model's initial state.

        :returns: ``(observation, info)`` where ``info["state"]`` is the initial
            stormvogel ``State``.
        """
        super().reset(seed=seed)
        self._current_state = self.model.initial_state
        return self._encode_obs(self._current_state), {"state": self._current_state}

    def step(
        self, action: int
    ) -> tuple[int | dict[str, int], float, bool, bool, dict[str, Any]]:
        """Take one step in the environment.

        :param action: Integer action index.  For MDPs this selects from
            ``_index_to_action``; for DTMCs only ``0`` is valid.
        :returns: ``(observation, reward, terminated, truncated, info)`` where
            ``info["state"]`` is the next stormvogel ``State`` and
            ``info["labels"]`` are its labels.
        :raises ActionUnavailableError: If the chosen action is not available in
            the current state, or if ``action != 0`` for a DTMC.
        """
        if self.model.supports_actions():
            sv_action = self._index_to_action[action]
            available = self._current_state.available_actions()
            if sv_action not in available:
                raise ActionUnavailableError(
                    f"Action {sv_action!r} is not available in state "
                    f"{self._current_state.state_id}. "
                    f"Available: {available}"
                )
        else:
            if action != 0:
                raise ActionUnavailableError(
                    f"DTMC only supports action index 0, got {action}."
                )
            sv_action = stormvogel.model.EmptyAction

        if self._reward_model is not None:
            r = self._reward_model.get_state_reward(self._current_state)
            reward = float(r) if r is not None else 0.0
        else:
            reward = 0.0

        next_state, _, labels = stormvogel.simulator.step(
            self._current_state, sv_action
        )

        self._current_state = next_state
        terminated = next_state.is_absorbing()

        return (
            self._encode_obs(next_state),
            reward,
            terminated,
            False,
            {"state": next_state, "labels": labels},
        )
