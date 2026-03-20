"""Defines the stormvogel Model."""

from enum import Enum
from typing import Iterable, Any, Iterator
from uuid import UUID

from copy import deepcopy

from deprecated import deprecated

from stormvogel.model.choices import Choices, ChoicesShorthand, choices_from_shorthand
from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.distribution import Distribution
from stormvogel.model.observation import Observation
from stormvogel.model.value import Value, Interval, Number
from stormvogel.model.state import State
from stormvogel.parametric import Parametric
from stormvogel.model.reward_model import RewardModel
from stormvogel.model.variable import Variable


class ModelType(Enum):
    """The type of the model."""

    # implemented
    DTMC = 1
    MDP = 2
    CTMC = 3
    POMDP = 4
    MA = 5
    HMM = 6


class Model[ValueType: Value]:
    """Represent a model.

    :param model_type: The model type.
    :param states: The states of the model.
    :param transitions: The transitions of this model. The keys are State objects.
    :param state_valuations: The state valuations of this model, mapping states to variable-value pairs.
    :param friendly_names: Optional mapping from states to friendly names for easier debugging and visualization.
    :param rewards: The reward models of this model.
    :param observation_aliases: The mapping from observations to their aliases (empty for non-observation models).
    :param observation_valuations: The mapping from observations to variable-value pairs (empty for non-observation models).
    :param state_observations: The mapping from states to their observations (empty for non-observation models).
    :param markovian_states: The set of states that are markovian (only for Markov automata).
    """

    # All models:

    model_type: ModelType

    states: list[State[ValueType]]

    transitions: dict[State[ValueType], Choices]

    state_valuations: dict[State[ValueType], dict[Variable, Any]]

    state_labels: dict[str, set[State[ValueType]]]

    friendly_names: dict[State[ValueType], str | None]

    rewards: list[RewardModel]

    # Partially observable models:

    observation_aliases: dict[Observation, str]

    observation_valuations: dict[Observation, dict[Variable, Any]]

    state_observations: dict[
        State[ValueType], Observation | Distribution[ValueType, Observation]
    ]

    # Markov automata:

    markovian_states: set[State[ValueType]]

    def __init__(self, model_type: ModelType, create_initial_state: bool = True):
        self.model_type = model_type
        self.states = list()
        self.transitions = dict()
        self.state_valuations = dict()
        self.state_labels = dict()
        self.rewards = []
        self.friendly_names = dict()
        self.observation_aliases = dict()
        self.observation_valuations = dict()
        self.state_observations = dict()
        self.markovian_states = set()
        self._is_parametric: bool | None = None
        self._is_interval: bool | None = None
        self._state_index_cache: dict[State, int] | None = None

        # Add the initial state if specified to do so
        if create_initial_state:
            if self.supports_observations():
                obs = self.new_observation("init")
                self.new_state(["init"], observation=obs)
            else:
                self.new_state(["init"])

    @property
    def actions(self) -> Iterable[Action]:
        """Extract the actions from a model that supports actions.

        :raises RuntimeError: If the model does not support actions.
        """
        if not self.supports_actions():
            raise RuntimeError("This model does not support actions.")

        seen: set[Action] = set()
        for choice in self.transitions.values():
            for action in choice.actions:
                if action not in seen:
                    seen.add(action)
                    yield action

    @property
    def observations(self) -> Iterable[Observation]:
        """Extract the observations from a model that supports observations.

        :raises RuntimeError: If the model does not support observations.
        """
        if not self.supports_observations():
            raise RuntimeError("This model does not support observations.")

        seen: set[Observation] = set()
        for obs in self.state_observations.values():
            if isinstance(obs, Observation) and obs not in seen:
                seen.add(obs)
                yield obs
            elif isinstance(obs, Distribution):
                for _, o in obs:
                    if o not in seen:
                        seen.add(o)
                        yield o

    def summary(self):
        """Give a short summary of the model."""
        choices_bit = (
            f"{sum(len(choices) for choices in self.transitions.values())} choices, "
            if self.supports_actions()
            else ""
        )
        return (
            f"{self.model_type} model with {len(self.states)} states, "
            + choices_bit
            + f"and {len(self.state_labels)} distinct labels."
        )

    def supports_actions(self) -> bool:
        """Return whether this model supports actions."""
        return self.model_type in (ModelType.MDP, ModelType.POMDP, ModelType.MA)

    def supports_rates(self) -> bool:
        """Return whether this model supports rates."""
        return self.model_type in (ModelType.CTMC, ModelType.MA)

    def supports_observations(self) -> bool:
        """Return whether this model supports observations."""
        return self.model_type in (ModelType.POMDP, ModelType.HMM)

    def is_interval_model(self) -> bool:
        """Return whether this model is an interval model, i.e., contains interval values."""
        if self._is_interval is None:
            for choice in self.transitions.values():
                for _, branch in choice:
                    for value, _ in branch:
                        if issubclass(type(value), Interval):
                            self._is_interval = True
                            return self._is_interval
            self._is_interval = False
        return self._is_interval

    @property
    def parameters(self) -> set[str]:
        """Return the set of parameters of this model."""
        parameters = set()
        for _, choice in self.transitions.items():
            for _, branch in choice:
                for transition in branch:
                    if isinstance(transition[0], Parametric):
                        parameters = parameters.union(transition[0].get_variables())
        return parameters

    def is_parametric(self) -> bool:
        """Return whether this model contains parametric transition values."""
        if self._is_parametric is None:
            for _, choice in self.transitions.items():
                for _, branch in choice:
                    for value, _ in branch:
                        if issubclass(type(value), Parametric):
                            self._is_parametric = True
                            return self._is_parametric
            self._is_parametric = False
        return self._is_parametric

    def is_stochastic(self, epsilon=1e-6) -> bool | None:
        """Check whether the model is stochastic.

        For discrete models, check that all sums of outgoing transition
        probabilities for all states equal 1, with at most *epsilon* rounding
        error. For continuous models, check that each outgoing rate is
        positive.

        :param epsilon: Maximum allowed rounding error for probability sums.
        :returns: ``True`` if the model is stochastic, ``False`` otherwise, or
            ``True`` trivially for parametric / interval models.
        """

        # if the model is parametric or an interval model, it should be trivially true, as the probabilities do
        # not even sum to a constant.
        if self.is_parametric() or self.is_interval_model():
            return True
        if not self.supports_rates():
            return all(
                [self.transitions[state].is_stochastic(epsilon) for state in self]
            )
        else:
            for state in self:
                for action in state.available_actions():
                    transitions = state.get_outgoing_transitions(action)
                    assert transitions is not None
                    for transition in transitions:
                        if isinstance(transition[0], Number) and transition[0] <= 0:
                            return False
        return True

    def normalize(self):
        """Normalize the model.

        For states where outgoing transition probabilities do not sum to 1,
        divide each probability by the sum. For rate-based models, only add
        self-loops.

        :raises RuntimeError: If the model is parametric or an interval model.
        """
        if self.is_parametric() or self.is_interval_model():
            raise RuntimeError(
                "normalize method undefined for parametric or interval models"
            )

        if not self.supports_rates():
            self.add_self_loops()
            for state in self:
                for action in state.available_actions():
                    # we first calculate the sum
                    sum_prob = 0
                    transitions = state.get_outgoing_transitions(action)
                    assert transitions is not None
                    for t in transitions:
                        if isinstance(t[0], Number):
                            sum_prob += t[0]

                    # then we divide each value by the sum
                    new_distr = Distribution()
                    for t in transitions:
                        if isinstance(t[0], Number):
                            new_distr[t[1]] = t[0] / sum_prob
                    self.transitions[state][action] = new_distr
        else:
            # for ctmcs and mas we currently only add self loops
            self.add_self_loops()

    def get_sub_model(self, states: Iterable[State], normalize: bool = True) -> "Model":
        """Return a submodel containing only the given states.

        :param states: The states to keep in the submodel.
        :param normalize: Whether to normalize the submodel after construction.
        :returns: A new model containing only the specified states.
        """
        keep_ids = {s.state_id for s in states}
        sub_model = deepcopy(self)
        remove = [state for state in sub_model if state.state_id not in keep_ids]
        for state in remove:
            sub_model.remove_state(state, normalize=False, suppress_warning=True)

        if normalize:
            sub_model.normalize()
        return sub_model

    def get_instantiated_model(self, values: dict[str, Number]) -> "Model":
        """Evaluate all parametric transitions with the given values and return the instantiated model.

        :param values: Mapping from parameter names to their numeric values.
        :returns: A new model with all parametric transitions evaluated.
        """
        evaluated_model = deepcopy(self)
        for state, transition in evaluated_model.transitions.items():
            for action, branch in transition:
                new_distr = Distribution()
                for val, target in branch:
                    if isinstance(val, Parametric):
                        new_distr[target] = val.evaluate(values)
                    else:
                        new_distr[target] = val
                evaluated_model.transitions[state][action] = new_distr
        return evaluated_model

    def add_self_loops(self):
        """Add self-loops to all states that do not have an outgoing transition."""
        if self.supports_rates():
            return
        for state in self:
            if not state.has_choices():  # state has no outgoing transitions
                self.set_choices(state, [(float(1), state)])

    def add_valuation_at_remaining_states(
        self, variables: list[Variable] | None = None, value: Any = 0
    ):
        """Set a dummy value for variables in all states where they are unassigned.

        :param variables: List of variable names to set. If ``None``, all
            variables in the model are used.
        :param value: The value to assign to unassigned variables.
        """

        # we either set it at all variables or just at a given subset of variables
        if variables is not None:
            v = variables
        else:
            v = self.variables

        # we set the values
        for state in self:
            for var in v:
                if var not in state.valuations.keys():
                    state.valuations[var] = value

    def unassigned_variables(self) -> Iterator[tuple[State, Variable]]:
        """Yield tuples of state-variable pairs that are unassigned."""
        variables = self.variables
        for state in self:
            for variable in variables:
                if variable not in state.valuations:
                    yield (state, variable)

    def iterate_transitions(self) -> Iterator[tuple[ValueType, State]]:
        """Iterate through all transitions in all choices of the model."""
        for choice in self.transitions.values():
            for _action, branch in choice:
                for transition in branch:
                    yield transition

    def has_zero_transition(self) -> bool:
        """Check whether the model has transitions with probability zero."""
        for choice in self.transitions.values():
            if choice.has_zero_transition():
                return True
        return False

    def add_markovian_state(self, markovian_state: State):
        """Add a state to the markovian states.

        :param markovian_state: The state to mark as markovian.
        :raises RuntimeError: If the model is not a Markov automaton.
        """
        if self.model_type == ModelType.MA and self.markovian_states is not None:
            self.markovian_states.add(markovian_state)
        else:
            raise RuntimeError("This model is not a MA")

    def set_choices(self, s: State, choices: Choices | ChoicesShorthand) -> None:
        """Set the choices for a state.

        :param s: The state to set choices for.
        :param choices: The choices to assign.
        :raises RuntimeError: If any transition probability is zero.
        """
        self._is_interval = None
        self._is_parametric = None
        if not isinstance(choices, Choices):
            choices = choices_from_shorthand(choices)

        if choices.has_zero_transition():
            raise RuntimeError("All transition probabilities should be nonzero.")

        self.transitions[s] = choices

    def add_choices(self, s: State, choices: Choices | ChoicesShorthand) -> None:
        """Add new choices from a state to the model.

        If no choice currently exists, the result is the same as
        :meth:`set_choices`.

        :param s: The state to add choices to.
        :param choices: The choices to add.
        :raises RuntimeError: If any transition probability is zero.
        """
        self._is_interval = None
        self._is_parametric = None
        if s not in self.transitions:
            self.transitions[s] = Choices(dict())
        self.transitions[s].add(choices)

    def get_successor_states(self, state: State) -> set[State]:
        """Return the set of successor states of the given state.

        :param state: The state whose successors to retrieve.
        :returns: The set of successor states.
        """
        result = set()
        for _, branch in self.transitions[state]:
            for _, target in branch:
                result.add(target)
        return result

    def get_distribution(
        self, state: State
    ) -> Distribution[ValueType, State[ValueType]]:
        """Get the distribution at the given state.

        Only intended for distribution with EmptyAction; raises an error otherwise.

        :param state: The state to retrieve branches for.
        :returns: The branches for the empty action at this state.
        :raises RuntimeError: If the state has non-empty choices.
        """
        choices = self.transitions[state]
        if not choices.has_empty_action():
            raise RuntimeError("Called get_distribution on a non-empty choice.")
        return choices[EmptyAction]

    def remove_state(
        self,
        state: State,
        normalize: bool = True,
        suppress_warning: bool = False,
    ):
        """Remove a state from the model.

        Removes all incoming and outgoing transitions, labels, reward entries,
        and markovian-state membership for the given state.

        :param state: The state to remove.
        :param normalize: Whether to normalize the model after removal.
        :param suppress_warning: If ``True``, suppress the warning about
            existing references to states by id.
        :raises RuntimeError: If the state is not part of this model.
        """
        self._is_interval = None
        self._is_parametric = None
        self._state_index_cache = None
        if not suppress_warning:
            import warnings

            warnings.warn(
                "Warning: Using this can cause problems in your code if there are existing references to states by id."
            )
        if state not in self.states:
            raise RuntimeError("This state is not part of this model.")

        # remove all incoming transitions to this state
        states_to_remove_from_transitions = []
        for source_state, transition in self.transitions.items():
            actions_to_remove = []
            for action, branch in transition:
                # rebuild the distribution without the removed state
                new_dist = Distribution({s: v for v, s in branch if s != state})
                transition[action] = new_dist
                if len(new_dist) == 0:
                    actions_to_remove.append(action)

            for action in actions_to_remove:
                del transition[action]

            if len(transition) == 0 and source_state != state:
                states_to_remove_from_transitions.append(source_state)

        for s in states_to_remove_from_transitions:
            del self.transitions[s]

        # remove the state's outgoing choices
        if state in self.transitions:
            del self.transitions[state]

        # remove the state itself from the list
        self.states.remove(state)
        self.state_valuations.pop(state, None)
        self.friendly_names.pop(state, None)
        self.state_observations.pop(state, None)
        self.markovian_states.discard(state)

        # remove from labels
        for states in self.state_labels.values():
            if state in states:
                states.remove(state)

        # remove from reward models
        for reward_model in self.rewards:
            if state in reward_model.rewards:
                del reward_model.rewards[state]

        if normalize:
            self.normalize()

    def get_state_index(self, state: State) -> int:
        """Return the index of the given state in the model, with O(1) amortized lookup.

        :param state: The state to look up.
        :returns: The index of the state, or -1 if not found.
        """
        # Check if the cache is valid for this state
        if self._state_index_cache is not None:
            idx = self._state_index_cache.get(state)
            # We verify the cache is still correct by checking the actual list
            if idx is not None and idx < len(self.states) and self.states[idx] is state:
                return idx

        # If not, we rebuild the whole cache
        self._state_index_cache = {s: i for i, s in enumerate(self.states)}
        return self._state_index_cache.get(state, -1)

    def new_state(
        self,
        labels: list[str] | str | None = None,
        valuations: dict[Variable, Any] | None = None,
        observation: Observation | Distribution[ValueType, Observation] | None = None,
    ) -> State:
        """Create a new state and return it.

        :param labels: Labels to assign to the new state.
        :param valuations: Variable-value pairs to assign as valuations.
        :param observation: Observation to assign (required for models that
            support observations).
        :returns: The newly created state.
        :raises RuntimeError: If the model supports observations but none is
            provided, or if an observation is provided but not supported.
        """
        if self.supports_observations() and observation is None:
            raise RuntimeError(
                "Tried to create a state in a model that supports observations without providing an observation."
            )
        if observation is not None and not self.supports_observations():
            raise RuntimeError(
                "Tried to set an observation on a model that does not support observations."
            )

        self._is_interval = None
        self._is_parametric = None
        self._state_index_cache = None
        state = State(self)

        self.states.append(state)

        self.state_valuations[state] = dict()

        if labels is not None and isinstance(labels, list):
            for label in labels:
                state.add_label(label)
        elif labels is not None and isinstance(labels, str):
            state.add_label(labels)

        if valuations is not None:
            for var, val in valuations.items():
                state.add_valuation(var, val)

        self.transitions[state] = Choices(dict())

        if observation is not None:
            state.observation = observation

        return state

    def new_observation(
        self, alias: str, valuations: dict[Variable, Any] | None = None
    ) -> Observation:
        """Create a new observation and return it.

        :param alias: The alias for the new observation.
        :param valuations: Variable-value pairs to assign as valuations.
        :returns: The newly created observation.
        :raises RuntimeError: If the model does not support observations, or if
            an observation with the given alias already exists.
        """
        if not self.supports_observations():
            raise RuntimeError(
                "Tried to create an observation in a model that does not support observations."
            )
        if alias in self.observation_aliases.values():
            raise RuntimeError(
                f"An observation with alias {alias} already exists in this model."
            )
        obs = Observation(self)
        self.observation_aliases[obs] = alias
        self.observation_valuations[obs] = (
            valuations if valuations is not None else dict()
        )
        return obs

    def get_observation(self, alias: str) -> Observation:
        """Get an existing observation with the given alias.

        :param alias: The alias of the observation.
        :returns: The matching observation.
        :raises RuntimeError: If the model does not support observations, or if
            no observation with the given alias is found.
        """
        if not self.supports_observations():
            raise RuntimeError(
                "Called get_observation on a model that does not support observations"
            )
        for obs, obs_alias in self.observation_aliases.items():
            if obs_alias == alias:
                return obs
        raise KeyError(f"Observation with alias {alias} not found.")

    def observation(self, alias: str) -> Observation:
        """Makes a new observation if the given alias does not exist and returns it, otherwise returns the existing observation with the given alias."""
        if not self.supports_observations():
            raise RuntimeError(
                "Called observation on a model that does not support observations"
            )
        for obs, obs_alias in self.observation_aliases.items():
            if obs_alias == alias:
                return obs
        return self.new_observation(alias)

    def new_action(self, label: str) -> Action:
        """Create a new action with the given label.

        :param label: The label for the new action.
        :returns: The newly created action.
        """
        return Action(label)

    def action(self, label: str) -> Action:
        """Alias of new_action."""
        return self.new_action(label)

    def get_states_with_label(self, label: str) -> set[State]:
        """Get all states with a given label.

        :param label: The label to search for.
        :returns: The set of states that carry the label.
        """
        return self.state_labels[label]

    def get_state_by_id(self, state_id: UUID) -> State:
        """Get a state by its UUID.

        :param state_id: The UUID of the state.
        :returns: The matching state.
        :raises RuntimeError: If no state with the given id is found.
        """
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise RuntimeError(f"State with id {state_id} not found.")

    @property
    def initial_state(self) -> State:
        """Get the initial state (the state with label ``"init"``).

        :raises RuntimeError: If the model does not have exactly one initial state.
        """

        if "init" not in self.state_labels or len(self.state_labels["init"]) != 1:
            raise RuntimeError(
                "Model does not have exactly one initial state with label 'init'."
            )
        return next(iter(self.state_labels["init"]))

    def add_label(self, label: str):
        """Add a label to the model.

        :param label: The label to add.
        """
        if label in self.state_labels:
            raise RuntimeError(f"Label {label} already exists in the model.")
        self.state_labels[label] = set()

    @property
    def variables(self) -> set[Variable]:
        """Get the set of all variables present in this model (corresponding to valuations)."""
        variables: set[Variable] = set()
        for state in self.states:
            variables = variables | set(state.valuations.keys())
            variables = variables | set(
                self.state_observations.get(state, {}).valuations.keys()
            )
        return variables

    def get_default_rewards(self) -> RewardModel:
        """Get the default reward model.

        :returns: The first reward model.
        :raises RuntimeError: If there are no reward models.
        """
        if len(self.rewards) == 0:
            raise RuntimeError("This model has no reward models.")
        return self.rewards[0]

    def get_rewards(self, name: str) -> RewardModel:
        """Get the reward model with the specified name.

        :param name: The name of the reward model.
        :returns: The matching reward model.
        :raises RuntimeError: If no reward model with the given name exists.
        """
        for model in self.rewards:
            if model.name == name:
                return model
        raise RuntimeError(f"Reward model {name} not present in model.")

    def new_reward_model(self, name: str) -> RewardModel:
        """Create a reward model with the specified name, add it, and return it.

        :param name: The name for the new reward model.
        :returns: The newly created reward model.
        :raises RuntimeError: If a reward model with the given name already exists.
        """
        for model in self.rewards:
            if model.name == name:
                raise RuntimeError(f"Reward model {name} already present in model.")
        reward_model = RewardModel(name, self, {})
        self.rewards.append(reward_model)
        return reward_model

    @deprecated(version="0.10.0", reason="use model_type instead.")
    def get_type(self) -> ModelType:
        """Get the type of this model."""
        return self.model_type

    @property
    def nr_states(self) -> int:
        """Return the number of states in this model.

        Note that not all states need to be reachable.
        """
        return len(self.states)

    @property
    def nr_choices(self) -> int:
        """Return the number of choices in the model (summed over all states)."""
        return sum(state.nr_choices for state in self.states)

    def to_dot(self) -> str:
        """Generate a dot representation of this model."""
        dot = "digraph model {\n"
        for state in self:
            dot += f'{state.state_id} [ label = "{state.state_id}: {", ".join(state.labels)}" ];\n'
        for state_id, transition in self.transitions.items():
            for action, branch in transition:
                if action != EmptyAction:
                    dot += f'{state_id}{action.label} [ label = "", shape=point ];\n'
        for state_id, transition in self.transitions.items():
            for action, branch in transition:
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch:
                        dot += (
                            f'{state_id} -> {target.state_id} [ label = "{prob}" ];\n'
                        )
                else:
                    # Draw actions, then probabilities
                    dot += f'{state_id} -> {state_id}{action.label} [ label = "{action.label}" ];\n'
                    for prob, target in branch:
                        dot += f'{state_id}{action.label} -> {target.state_id} [ label = "{prob}" ];\n'
        dot += "}"
        return dot

    def __str__(self) -> str:
        return self.summary()

    @property
    def sorted_states(self):
        return sorted(self.states, key=lambda state: state.friendly_name or "")

    def __getitem__(self, state_index: int):
        return self.states[state_index]

    def __iter__(self):
        return iter(self.states)

    def make_observations_deterministic(self):
        """Make observations deterministic by splitting states with multiple observations.

        In case of POMDPs or HMMs, split each state that has a distribution
        over observations into multiple states with single observations.

        :raises RuntimeError: If the model does not support observations.
        """
        if not self.supports_observations():
            raise RuntimeError(
                "This method only works for models that support observations."
            )

        for state in list(self.states):
            observation = state.observation
            if isinstance(observation, Distribution):
                # from stormvogel.model.value import ValueType
                new_states_distribution = []

                # Create new states for each observation possible in this state
                for prob, obs in observation:
                    new_state = self.new_state(
                        labels=list(state.labels),
                        valuations=state.valuations,
                        observation=obs,
                    )
                    self.transitions[new_state] = self.transitions[state]
                    new_states_distribution.append((prob, new_state))

                # Replace transitions to the original state with transitions to the new states
                for other_state in self.states:
                    for action in other_state.available_actions():
                        transitions = other_state.get_outgoing_transitions(action)
                        if transitions is not None:
                            new_transitions = []
                            for transition_prob, target in transitions:
                                if target == state:
                                    for new_prob, new_state in new_states_distribution:
                                        new_transitions.append(
                                            (transition_prob * new_prob, new_state)
                                        )
                                else:
                                    new_transitions.append((transition_prob, target))
                            self.transitions[other_state][action] = Distribution(
                                new_transitions
                            )

                # Remove the original state and choices
                del self.transitions[state]
                self.states.remove(state)
                # Remove labels for this state
                for label in state.labels:
                    self.state_labels[label].remove(state)


def new_dtmc(create_initial_state: bool = True) -> Model:
    """Create a DTMC.

    :param create_initial_state: Whether to create an initial state.
    :returns: A new DTMC model.
    """
    return Model(ModelType.DTMC, create_initial_state)


def new_mdp(create_initial_state: bool = True) -> Model:
    """Create an MDP.

    :param create_initial_state: Whether to create an initial state.
    :returns: A new MDP model.
    """
    return Model(ModelType.MDP, create_initial_state)


def new_ctmc(create_initial_state: bool = True) -> Model:
    """Create a CTMC.

    :param create_initial_state: Whether to create an initial state.
    :returns: A new CTMC model.
    """
    return Model(ModelType.CTMC, create_initial_state)


def new_pomdp(create_initial_state: bool = True) -> Model:
    """Create a POMDP.

    :param create_initial_state: Whether to create an initial state.
    :returns: A new POMDP model.
    """
    return Model(ModelType.POMDP, create_initial_state)


def new_hmm(create_initial_state: bool = True) -> Model:
    """Create an HMM.

    :param create_initial_state: Whether to create an initial state.
    :returns: A new HMM model.
    """
    return Model(ModelType.HMM, create_initial_state)


def new_ma(create_initial_state: bool = True) -> Model:
    """Create a MA.

    :param create_initial_state: Whether to create an initial state.
    :returns: A new Markov automaton model.
    """
    return Model(ModelType.MA, create_initial_state)


def new_model(modeltype: ModelType, create_initial_state: bool = True) -> Model:
    """Create a model of the given type.

    :param modeltype: The type of model to create.
    :param create_initial_state: Whether to create an initial state.
    :returns: A new model of the specified type.
    """
    return Model(modeltype, create_initial_state)
