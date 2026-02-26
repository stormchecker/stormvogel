from enum import Enum
from typing import Iterable, Any
from uuid import UUID

from deprecated import deprecated

from copy import deepcopy

from stormvogel.model.choices import Choices, ChoicesShorthand, choices_from_shorthand
from stormvogel.model.branches import Branches
from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.distribution import Distribution
from stormvogel.model.observation import Observation
from stormvogel.model.value import Value, Interval, Number
from stormvogel.model.state import State
from stormvogel.parametric import Parametric
from stormvogel.model.reward_model import RewardModel


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
    """Represents a model.

    Args:
        type: The model type.
        states: The states of the model. The keys are the state's ids.
        choices: The choices of this model. The keys are the state ids.
        actions: The actions of the model, if this is a model that supports actions.
        rewards: The rewardsmodels of this model.
        markovian_states: list of markovian states in the case of a ma.
    """

    model_type: ModelType

    states: list[State[ValueType]]

    choices: dict[State[ValueType], Choices]
    observations: list[Observation] | None

    state_valuations: dict[State[ValueType], dict[str, Any]]
    state_observations: (
        dict[State[ValueType], Observation | Distribution[ValueType, Observation]]
        | None
    )
    state_names: dict[State[ValueType], str | None]

    state_labels: dict[str, set[State[ValueType]]]

    rewards: list[RewardModel]

    markovian_states: set[State[ValueType]] | None

    def __init__(self, model_type: ModelType, create_initial_state: bool = True):
        self.model_type = model_type
        self.states = list()
        self.choices = dict()
        self.state_valuations = dict()
        self.state_labels = dict()
        self.rewards = []
        self.parametric: bool | None = None
        self.interval: bool | None = None
        self.state_names = dict()

        # Initialize observations if those are supported by the model type (pomdps)
        if self.model_type == ModelType.POMDP:
            self.state_observations = dict()
            self.observations: list[Observation] | None = []
        else:
            self.state_observations = None
            self.observations = None
        self.markovian_states: set[State] | None
        # Initialize markovian states if applicable (in the case of MA's)
        if self.model_type == ModelType.MA:
            self.markovian_states = set()
        else:
            self.markovian_states = None

        # Add the initial state if specified to do so
        if create_initial_state:
            if self.supports_observations():
                obs = self.new_observation("init")
                self.new_state(["init"], observation=obs)
            else:
                self.new_state(["init"])

    def _get_value_type(self) -> type:
        """Returns the ValueType of this model."""
        return self.__orig_class__.__args__[0]

    @property
    def actions(self) -> Iterable[Action]:
        """Extracts the actions from a model that supports actions."""
        if not self.supports_actions():
            raise RuntimeError("This model does not support actions.")

        seen: set[Action] = set()
        for choice in self.choices.values():
            for action in choice.choices.keys():
                if action not in seen:
                    seen.add(action)
                    yield action

    def summary(self):
        """Give a short summary of the model."""
        choices_bit = (
            f"{sum(len(choices.choices) for choices in self.choices)} choices, "
            if self.supports_actions() is not None
            else ""
        )
        return (
            f"{self.model_type} model with {len(self.states)} states, "
            + choices_bit
            + f"and {len(self.get_labels())} distinct labels."
        )

    def supports_actions(self) -> bool:
        """Returns whether this model supports actions."""
        return self.model_type in (ModelType.MDP, ModelType.POMDP, ModelType.MA)

    def supports_rates(self) -> bool:
        """Returns whether this model supports rates."""
        return self.model_type in (ModelType.CTMC, ModelType.MA)

    def supports_observations(self) -> bool:
        """Returns whether this model supports observations."""
        return self.model_type == ModelType.POMDP

    def is_interval_model(self) -> bool:
        """Returns whether this model is an interval model, i.e., containts interval values)"""
        if self.interval is None:
            for _, choice in self.choices.items():
                for _, branch in choice:
                    for value, _ in branch:
                        if issubclass(type(value), Interval):
                            self.interval = True
                            return self.interval
            self.interval = False
        return self.interval

    @property
    def parameters(self) -> set[str]:
        """Returns the set of parameters of this model"""
        parameters = set()
        for _, choice in self.choices.items():
            for _, branch in choice:
                for transition in branch:
                    if isinstance(transition[0], Parametric):
                        parameters = parameters.union(transition[0].get_variables())
        return parameters


    def is_parametric(self) -> bool:
        """Returns whether this model contains parametric transition values"""
        if self.parametric is None:
            for _, choice in self.choices.items():
                for _, branch in choice:
                    for value, _ in branch:
                        if issubclass(type(value), Parametric):
                            self.parametric = True
                            return self.parametric
            self.parametric = False
        return self.parametric

    def is_stochastic(self, epsilon=1e-6) -> bool | None:
        """For discrete models: Checks if all sums of outgoing transition probabilities for all states equal 1, with at most epsilon rounding error.
        For continuous models: Checks if all sums of outgoing rates sum to 0
        """

        # if the model is parametric or an interval model, it should be trivially true, as the probabilities do
        # not even sum to a constant.
        if self.is_parametric() or self.is_interval_model():
            return True
        if not self.supports_rates():
            return all([self.choices[state].is_stochastic(epsilon) for state in self])
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
        """Normalizes a model (for states where outgoing transition probabilities don't sum to 1, we divide each probability by the sum)"""
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
                    new_transitions = []
                    for t in transitions:
                        if isinstance(t[0], Number):
                            normalized_transition = (
                                t[0] / sum_prob,
                                t[1],
                            )
                            new_transitions.append(normalized_transition)
                    self.choices[state].choices[action].branch = new_transitions
        else:
            # for ctmcs and mas we currently only add self loops
            self.add_self_loops()

    def get_sub_model(self, states: list[State], normalize: bool = True) -> "Model":
        """Returns a submodel of the model based on a collection of states.
        The states in the collection are the states that stay in the model."""
        sub_model = deepcopy(self)
        remove = []
        for state in sub_model:
            if state not in states:
                remove.append(state)
        for state in remove:
            sub_model.remove_state(state, normalize=False)

        if normalize:
            sub_model.normalize()
        return sub_model

    def parameter_valuation(self, values: dict[str, Number]) -> "Model":
        """evaluates all parametric transitions with the given values and returns the induced model"""
        evaluated_model = deepcopy(self)
        for state, transition in evaluated_model.choices.items():
            for action, branch in transition:
                new_branch = []
                for tup in branch:
                    if isinstance(tup[0], Parametric):
                        tup = (tup[0].evaluate(values), tup[1])
                    new_branch.append(tup)
                evaluated_model.choices[state][action].branch = new_branch
        return evaluated_model

    def add_self_loops(self):
        """adds self loops to all states that do not have an outgoing transition"""
        for state in self:
            if not state.has_choices():  # state has no outgoing transitions
                self.set_choices(
                    state, [(float(0) if self.supports_rates() else float(1), state)]
                )

    def add_valuation_at_remaining_states(
        self, variables: list[str] | None = None, value: Any = 0
    ):
        """Sets (dummy) value to variables in all states where they don't have a value yet."""

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

    def unassigned_variables(self) -> list[tuple[State, str]]:
        """Return a list of tuples of state variable pairs that are unassigned."""
        # we check all variables in all states
        unassigned = []
        for state in self:
            for variable in self.variables:
                if variable not in state.valuations.keys():
                    unassigned.append((state, variable))
        return unassigned

    def all_states_outgoing_transition(self) -> bool:
        """Checks if all states have a choice."""
        for state in self.states:
            if not state.has_choices():
                return False
        return True

    def iterate_transitions(self) -> list[tuple[ValueType, State]]:
        """Iterates through all transitions in all choices of the model."""
        transitions = []
        for choice in self.choices.values():
            for _action, branch in choice:
                for transition in branch:
                    transitions.append(transition)
        return transitions

    def all_non_init_states_incoming_transition(self) -> bool:
        """Checks if all states except the initial state have an incoming transition."""
        remaining_states = set(self.states)
        for transition in self.iterate_transitions():
            remaining_states.discard(transition[1])
        for s in remaining_states:
            if not s.is_initial():
                return False
        return True

    def has_zero_transition(self) -> bool:
        """checks if the model has transitions with probability zero"""
        for _, choice in self.choices.items():
            if choice.has_zero_transition():
                return True
        return False

    def add_markovian_state(self, markovian_state: State):
        """Adds a state to the markovian states (in case of markov automata)."""
        if self.model_type == ModelType.MA and self.markovian_states is not None:
            self.markovian_states.add(markovian_state)
        else:
            raise RuntimeError("This model is not a MA")

    def set_choices(self, s: State, choices: Choices | ChoicesShorthand) -> None:
        """Set the choice for a state."""
        if not isinstance(choices, Choices):
            choices = choices_from_shorthand(choices)

        if choices.has_zero_transition() and not self.supports_rates():
            raise RuntimeError("All transition probabilities should be nonzero.")

        self.choices[s] = choices

    def add_choices(self, s: State, choices: Choices | ChoicesShorthand) -> None:
        """Add new choices from a state to the model. If no choice currently exists, the result will be the same as set_choice."""
        if s not in self.choices:
            self.choices[s] = Choices(dict())

        if not isinstance(choices, Choices):
            choices = choices_from_shorthand(choices)

        if choices.has_zero_transition() and not self.supports_rates():
            raise RuntimeError("All transition probabilities should be nonzero.")

        self.choices[s].add(choices)

    def get_successor_states(self, state: State) -> set[State]:
        """Returns the set of successors of state_or_id."""
        result = set()
        for _, branches in self.choices[state]:
            result |= branches.successors
        return result


    def get_branches(self, state: State) -> Branches:
        """Get the branch at state s. Only intended for emtpy choices, otherwise a RuntimeError is thrown."""
        choice = self.choices[state].choices
        if EmptyAction not in choice:
            raise RuntimeError("Called get_branch on a non-empty choice.")
        return choice[EmptyAction]

    def get_action_with_label(self, label: str | None) -> Action | None:
        """Get the action with provided label"""
        assert self.actions is not None
        for action in self.actions:
            if action.label == label:
                return action
        return None

    def action(self, label: str | None = None) -> Action:
        """Creates a new action and returns it."""
        if not self.supports_actions():
            raise RuntimeError(
                "Called new_action on a model that does not support actions"
            )
        assert self.actions is not None
        action = Action(label)
        return action

    def new_action(self, label: str | None = None) -> Action:
        """Creates a new action and returns it."""
        return self.action(label)

    def remove_state(
        self, state: State, normalize: bool = True, reassign_ids: bool = False, suppress_warning: bool = False
    ):
        """Properly removes a state, it can optionally normalize the model and reassign ids automatically."""
        if not suppress_warning:
            import warnings
            warnings.warn("Warning: Using this can cause problems in your code if there are existing references to states by id.")
        if state not in self.states:
            raise RuntimeError("This state is not part of this model.")

        # remove all incoming transitions to this state
        states_to_remove_from_choices = []
        for source_state, transition in self.choices.items():
            actions_to_remove = []
            for action, branch in transition.choices.items():
                # filter out the tuple referencing the state
                new_branch = [
                    (prob, target) for prob, target in branch.branch if target != state
                ]
                branch.branch = new_branch
                # if branch is empty, we must remove the action
                if len(branch.branch) == 0:
                    actions_to_remove.append(action)
            
            for action in actions_to_remove:
                del transition.choices[action]
            
            # if state has no choices left, remove it from choices dict
            if len(transition.choices) == 0 and source_state != state:
                states_to_remove_from_choices.append(source_state)

        for s in states_to_remove_from_choices:
            del self.choices[s]

        # remove the state's outgoing choices
        if state in self.choices:
            del self.choices[state]

        # remove the state itself from the list
        self.states.remove(state)

        # remove from markovian states
        if self.markovian_states is not None and state in self.markovian_states:
            self.markovian_states.remove(state)

        # remove from labels
        for label, states in self.state_labels.items():
            if state in states:
                states.remove(state)

        # remove from reward models
        for reward_model in self.rewards:
            if state in reward_model.rewards:
                del reward_model.rewards[state]

        if normalize:
            self.normalize()
        if reassign_ids:
            self.reassign_ids()

    def reassign_ids(self):
        """Reassigns the ids of states, choices and rates to be in order again.
        Mainly useful to keep consistent with storm."""
        pass # no longer needed, using UUIDs
    def get_action(self, name: str) -> Action:
        """Gets an existing action."""
        if not self.supports_actions():
            raise RuntimeError(
                "Called get_action on a model that does not support actions"
            )
        for action in self.actions:
            if action.label == name:
                return action

        raise RuntimeError(f"Action with name {name} not found.")

    def new_state(
        self,
        labels: list[str] | str | None = None,
        valuations: dict[str, Any] | None = None,
        observation: Observation | Distribution[ValueType, Observation] | None = None,
        name: str | None = None,
    ) -> State:
        """Creates a new state and returns it."""
        state = State(self)

        self.states.append(state)

        self.state_valuations[state] = dict()
        self.state_names[state] = name
        self.state_valuations[state] = dict()

        if labels is not None and isinstance(labels, list):
            for l in labels:
                state.add_label(l)
        elif labels is not None and isinstance(labels, str):
            state.add_label(labels)

        if valuations is not None:
            for var, val in valuations.items():
                state.add_valuation(var, val)

        self.choices[state] = Choices(dict())

        if self.supports_observations() and observation is None:
            raise RuntimeError(
                "Tried to create a state in a model that supports observations without providing an observation."
            )
        if observation is not None:
            if not self.supports_observations():
                raise RuntimeError(
                    "Tried to set an observation on a model that does not support observations."
                )
            state.observation = observation

        return state

    def get_states_with_label(self, label: str) -> set[State]:
        """Get all states with a given label."""
        return self.state_labels[label]

    def get_state_by_id(self, state_id: UUID) -> State:
        """Get a state by its UUID id."""
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise RuntimeError(f"State with id {state_id} not found.")

    def get_state_by_name(self, state_name) -> State | None:
        """Get a state by its name."""
        for state, name in self.state_names.items():
            if name == state_name:
                return state
        return None

    def get_state_by_stormpy_id(self, stormpy_id: int) -> State:
        """Get a state by its stormpy id (index in the states list)."""
        if stormpy_id < 0 or stormpy_id >= len(self.states):
            raise RuntimeError(f"State with stormpy id {stormpy_id} not found.")
        return self.states[stormpy_id]

    @property
    def initial_state(self) -> State:
        """Gets the initial state (contains label "init", or has id 0)."""

        if len(self.state_labels["init"]) != 1:
            raise RuntimeError(
                "Model does not have exactly one initial state with label 'init'."
            )

        return next(iter(self.state_labels["init"]))
    def add_label(self, label: str):
        """Adds a label to the model."""
        self.state_labels[label] = set()

    @property
    def variables(self) -> set[str]:
        """Gets the set of all variables present in this model (corresponding to valuations)."""
        variables: set[str] = set()
        for state in self.states:
            variables = variables | set(state.valuations.keys())
        return variables


    def get_default_rewards(self) -> RewardModel:
        """Gets the default reward model, throws a RuntimeError if there is none."""
        if len(self.rewards) == 0:
            raise RuntimeError("This model has no reward models.")
        return self.rewards[0]

    def get_rewards(self, name: str) -> RewardModel:
        """Gets the reward model with the specified name. Throws a RuntimeError if said model does not exist."""
        for model in self.rewards:
            if model.name == name:
                return model
        raise RuntimeError(f"Reward model {name} not present in model.")

    def new_reward_model(self, name: str) -> RewardModel:
        """Creates a reward model with the specified name, adds it and returns it."""
        for model in self.rewards:
            if model.name == name:
                raise RuntimeError(f"Reward model {name} already present in model.")
        reward_model = RewardModel(name, self, {})
        self.rewards.append(reward_model)
        return reward_model

    @deprecated(version="0.10.0", reason="use type instead.")
    def get_type(self) -> ModelType:
        """Gets the type of this model"""
        return self.model_type

    @property
    def nr_states(self) -> int:
        """
        Returns the number of states in this model.
        Note that not all states need to be reachable.
        """
        return len(self.states)

    @property
    def nr_choices(self) -> int:
        """
        Returns the number of choices in the model (summed over all states).
        """
        return sum(state.nr_choices for state in self.states)

    def new_observation(
        self,
        alias: str,
    ) -> Observation:
        """Creates a new observation with the given alias and returns it."""
        if not self.supports_observations():
            raise RuntimeError(
                "Called new_observation on a model that does not support observations"
            )
        assert self.observations is not None

        for observation in self.observations:
            if observation.alias == alias:
                raise RuntimeError(
                    f"An observation with alias {alias} already exists, namely {observation}."
                )

        observation = Observation(alias)
        self.observations.append(observation)
        return observation

    def get_observation(self, alias: str) -> Observation:
        """Gets an existing observation with the given alias."""
        if not self.supports_observations():
            raise RuntimeError(
                "Called get_observation on a model that does not support observations"
            )
        assert self.observations is not None
        for observation in self.observations:
            if observation.alias == alias:
                return observation

        raise RuntimeError(f"Observation with alias {alias} not found.")

    def observation(
        self,
        alias: str,
    ) -> Observation:
        """New observation or get observation if it exists."""
        if not self.supports_observations():
            raise RuntimeError(
                "Called method observation on a model that does not support observations"
            )
        assert self.observations is not None

        try:
            return self.get_observation(alias)
        except RuntimeError:
            return self.new_observation(alias)

    def to_dot(self) -> str:
        """Generates a dot representation of this model."""
        dot = "digraph model {\n"
        for state in self:
            dot += f'{state.state_id} [ label = "{state.state_id}: {", ".join(state.labels)}" ];\n'
        for state_id, transition in self.choices.items():
            for action, branch in transition:
                if action != EmptyAction:
                    dot += f'{state_id} [ label = "", shape=point ];\n'
        for state_id, transition in self.choices.items():
            for action, branch in transition:
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch:
                        dot += f'{state_id} -> {target.id} [ label = "{prob}" ];\n'
                else:
                    # Draw actions, then probabilities
                    dot += f'{state_id} -> {state_id} [ label = "{action.label}" ];\n'
                    for prob, target in branch:
                        dot += f'{state_id} -> {target.id} [ label = "{prob}" ];\n'
        dot += "}"
        return dot

    def __str__(self) -> str:
        res = [f"{self.model_type}"]
        res += ["", "States:"] + [f"{state}" for state in self]
        res += ["", "Choices:"] + [
            f"{id}: {transition}" for (id, transition) in self.choices.items()
        ]

        if (
            self.supports_actions()
            and self.supports_rates()
            and self.markovian_states is not None
        ):
            markovian_states = [state.state_id for state in self.markovian_states]
            res += ["", "Markovian states:"] + [f"{markovian_states}"]

        return "\n".join(res)

    def __getitem__(self, state_index: int):
        return self.states[state_index]

    def __iter__(self):
        return iter(self.states)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        if self.model_type != other.model_type:
            return False
        if len(self.states) != len(other.states):
            return False
        # Build identity-based index maps to avoid calling State.__eq__
        # (which only compares state_id and cannot match cross-model states).
        self_idx = {id(s): i for i, s in enumerate(self.states)}
        other_idx = {id(s): i for i, s in enumerate(other.states)}

        def _branches_eq(b1: Branches, b2: Branches) -> bool:
            if len(b1.branch) != len(b2.branch):
                return False
            for (v1, st1), (v2, st2) in zip(sorted(b1.branch, key=lambda t: self_idx[id(t[1])]), sorted(b2.branch, key=lambda t: other_idx[id(t[1])])):
                if v1 != v2:
                    return False
                if self_idx[id(st1)] != other_idx[id(st2)]:
                    return False
            return True

        def _choices_eq(c1: Choices, c2: Choices) -> bool:
            if len(c1.choices) != len(c2.choices):
                return False
            for a1, a2 in zip(sorted(c1.choices.keys()), sorted(c2.choices.keys())):
                if a1 != a2:
                    return False
                if not _branches_eq(c1.choices[a1], c2.choices[a2]):
                    return False
            return True

        for s1, s2 in zip(self.states, other.states):
            if s1 not in self.choices or s2 not in other.choices:
                if (s1 in self.choices) != (s2 in other.choices):
                    return False
                continue
            if not _choices_eq(self.choices[s1], other.choices[s2]):
                return False
        self_labels = {k: v for k, v in self.state_labels.items() if len(v) > 0}
        other_labels = {k: v for k, v in other.state_labels.items() if len(v) > 0}
        if set(self_labels.keys()) != set(other_labels.keys()):
            return False
        for label in self_labels:
            self_indices = {self.states.index(s) for s in self_labels[label]}
            other_indices = {other.states.index(s) for s in other_labels[label]}
            if self_indices != other_indices:
                return False
        return True


def new_dtmc(create_initial_state: bool = True) -> Model:
    """Creates a DTMC."""
    return Model(ModelType.DTMC, create_initial_state)


def new_mdp(create_initial_state: bool = True) -> Model:
    """Creates an MDP."""
    return Model(ModelType.MDP, create_initial_state)


def new_ctmc(create_initial_state: bool = True) -> Model:
    """Creates a CTMC."""
    return Model(ModelType.CTMC, create_initial_state)


def new_pomdp(create_initial_state: bool = True) -> Model:
    """Creates a POMDP."""
    return Model(ModelType.POMDP, create_initial_state)


def new_hmm(create_initial_state: bool = True) -> Model:
    """Creates a HMM."""
    return Model(ModelType.HMM, create_initial_state)


def new_ma(create_initial_state: bool = True) -> Model:
    """Creates a MA."""
    return Model(ModelType.MA, create_initial_state)


def new_model(modeltype: ModelType, create_initial_state: bool = True) -> Model:
    """More general model creation function"""
    return Model(modeltype, create_initial_state)
