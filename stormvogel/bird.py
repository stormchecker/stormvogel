import warnings

import stormvogel.model
from dataclasses import dataclass
from typing import cast, Any, Callable, Sequence
import inspect
from collections import deque
from collections.abc import Iterable
from stormvogel.model import Variable


@dataclass
class State:
    """Represent a bird state as a dynamic attribute container."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"state({self.__dict__})"

    def __hash__(self):
        return hash(str(self.__dict__))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__dict__ == other.__dict__
        return False


type Action = str


def valid_input[ValueType: stormvogel.model.Value](
    delta: Callable[[Any, Action], Any] | Callable[[Any], Any],
    init: Any,
    rewards: Callable[[Any], dict[str, ValueType]] | None = None,
    labels: Callable[[Any], Sequence[str] | str | None] | None = None,
    available_actions: Callable[[Any], list[Action]] | None = None,
    observations: Callable[[Any], int | list[tuple[ValueType, int]]] | None = None,
    rates: Callable[[Any], float] | None = None,
    valuations: Callable[[Any], dict[Variable, float | int | bool]] | None = None,
    observation_valuations: (
        Callable[[int], dict[Variable, float | int | bool]] | None
    ) = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
):
    """Validate the input for the bird model builder.

    Check that all user-supplied callbacks have the correct number of
    parameters and that required callbacks are provided for the chosen
    model type.

    :param delta: Transition function. Takes ``(state, action)`` for models
        that support actions, or ``(state,)`` otherwise.
    :param init: Initial state passed to the callbacks.
    :param rewards: Optional callback mapping a state to a dict of reward
        model name to reward value.
    :param labels: Optional callback mapping a state to a list of label
        strings, a single label string, or ``None``.
    :param available_actions: Optional callback returning the list of
        available action strings for a state. Required for MDP, POMDP,
        and MA model types.
    :param observations: Optional callback returning an observation id or a
        distribution over observations for a state. Required for POMDP and
        HMM model types.
    :param rates: Optional callback returning the exit rate for a state.
    :param valuations: Optional callback returning a dict of variable name
        to value for a state.
    :param observation_valuations: Optional callback returning a dict of
        variable name to value for a given observation id.
    :param modeltype: The type of model to build.
    :raises ValueError: If a required callback is missing or any callback
        has an incorrect number of parameters.
    """

    supports_actions = modeltype in (
        stormvogel.model.ModelType.MDP,
        stormvogel.model.ModelType.POMDP,
        stormvogel.model.ModelType.MA,
    )

    # we first check if we have an available_actions function in case our model supports actions
    if supports_actions and available_actions is None:
        raise ValueError(
            "You have to provide an available actions function for models that support actions"
        )

    # and we check if we have an observations function in case our model is a POMDP
    if (
        modeltype in (stormvogel.model.ModelType.POMDP, stormvogel.model.ModelType.HMM)
        and observations is None
    ):
        raise ValueError(
            "You have to provide an observations function for POMDPs or HMMs"
        )

    # we check if the provided functions have the right number of parameters
    if supports_actions:
        assert available_actions is not None
        sig = inspect.signature(available_actions)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The available_actions function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    sig = inspect.signature(delta)
    num_params = len(sig.parameters)
    if supports_actions:
        if num_params != 2:
            raise ValueError(
                f"Your delta function must take exactly two arguments (state and action), but it takes {num_params} arguments"
            )
    else:
        if num_params != 1:
            raise ValueError(
                f"Your delta function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if rewards is not None:
        sig = inspect.signature(rewards)
        num_params = len(sig.parameters)
        if num_params != 1:
            if num_params == 2:
                warnings.warn(
                    "State-action rewards are not supported in this version of stormvogel. Will assign None to the second parameter of your reward function."
                )
            else:
                raise ValueError(
                    f"The rewards function must take exactly one argument (state), but it takes {num_params} arguments"
                )

    if labels is not None:
        sig = inspect.signature(labels)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The labels function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if observations is not None:
        sig = inspect.signature(observations)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The observations function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if valuations is not None:
        sig = inspect.signature(valuations)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The valuations function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if observation_valuations is not None:
        sig = inspect.signature(observation_valuations)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The observation_valuations function must take exactly one argument (observation id), but it takes {num_params} arguments"
            )


def build_bird[ValueType: stormvogel.model.Value](
    delta: (
        Callable[[Any, Action], Sequence[tuple[ValueType, Any]] | Sequence[Any] | None]
        | Callable[[Any], Sequence[tuple[ValueType, Any]] | Sequence[Any] | None]
    ),
    init: Any,
    rewards: Callable[[Any], dict[str, ValueType]] | None = None,
    labels: Callable[[Any], Sequence[str] | str | None] | None = None,
    friendly_names: Callable[[Any], str] | None = None,
    available_actions: Callable[[Any], list[Action]] | None = None,
    observations: Callable[[Any], int | list[tuple[ValueType, int]]] | None = None,
    rates: Callable[[Any], float] | None = None,
    valuations: Callable[[Any], dict[Variable, float | int | bool]] | None = None,
    observation_valuations: (
        Callable[[int], dict[Variable, float | int | bool]] | None
    ) = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
    max_size: int = 10000,
) -> stormvogel.model.Model[ValueType]:
    """Build a stormvogel model from user-supplied callbacks.

    Explore the state space starting from *init* by repeatedly calling
    *delta* (and *available_actions* for action-based models) until no new
    states are discovered, analogous to a PRISM module.

    :param delta: Transition function. For action-based models it takes
        ``(state, action)`` and returns a list of ``(probability, state)``
        tuples (or a single successor, or ``None`` for a self-loop).
        For models without actions it takes ``(state,)``.
    :param init: Initial state of the model.
    :param rewards: Optional callback mapping a state to a dict of reward
        model name to reward value.
    :param labels: Optional callback mapping a state to a list of label
        strings, a single label string, or ``None``.
    :param available_actions: Optional callback returning the list of
        available action strings for a state. Required for MDP, POMDP,
        and MA model types.
    :param observations: Optional callback returning an observation id or a
        distribution over observations for a state. Required for POMDP and
        HMM model types.
    :param rates: Optional callback returning the exit rate for a state.
    :param valuations: Optional callback returning a dict of variable name
        to value for a state.
    :param observation_valuations: Optional callback returning a dict of
        variable name to value for a given observation id.
    :param modeltype: The type of model to build.
    :param max_size: Maximum number of states before aborting.
    :returns: The constructed stormvogel model.
    :raises ValueError: If callbacks return invalid results during
        exploration.
    :raises RuntimeError: If the state space exceeds *max_size*.
    """

    def add_new_choices(tuples, state):
        """Add newly discovered choices and states to the model.

        :param tuples: Sequence of ``(probability, state)`` tuples, bare
            states, or ``None`` (interpreted as a self-loop).
        :param state: The source state in the bird state space.
        :returns: List of ``(value, stormvogel_state)`` branch entries.
        :raises ValueError: If a transition tuple has an unexpected length.
        """
        branch = []
        if tuples is not None:
            for tup in tuples:
                # in case only a state is provided, we assume the probability is 1
                if not isinstance(tup, tuple):
                    s = tup
                    val = 1
                elif len(tup) == 2:
                    s = tup[1]
                    val = tup[0]
                else:
                    raise ValueError(
                        f"Invalid transition tuple {tup}. Expected (probability, state) or (state)."
                    )

                if s not in state_lookup:
                    obs_kwarg = {}
                    if model.supports_observations() and observations is not None:
                        given_obs = observations(s)
                        if isinstance(given_obs, int):
                            obs_kwarg["observation"] = model.observation(str(given_obs))
                        elif isinstance(given_obs, list):
                            obs_kwarg["observation"] = stormvogel.model.Distribution(
                                [
                                    (prob, model.observation(str(o)))
                                    for prob, o in given_obs
                                ]
                            )
                    new_state = model.new_state(**obs_kwarg)
                    state_lookup[s] = new_state
                    branch.append((val, new_state))
                    states_to_be_visited.append(s)
                else:
                    branch.append((val, state_lookup[s]))
        else:
            # if we have no return value, we add a self loop
            branch.append((1, state_lookup[state]))
        return branch

    valid_input(
        delta,
        init,
        rewards,
        labels,
        available_actions,
        observations,
        rates,
        valuations,
        observation_valuations,
        modeltype,
    )

    # we create the model with the given type and initial state
    model = stormvogel.model.new_model(modeltype=modeltype, create_initial_state=False)
    obs_kwarg = {}
    if model.supports_observations() and observations is not None:
        given_obs = observations(init)
        if isinstance(given_obs, int):
            obs_kwarg["observation"] = model.observation(str(given_obs))
        elif isinstance(given_obs, list):
            obs_kwarg["observation"] = stormvogel.model.Distribution(
                [(prob, model.observation(str(o))) for prob, o in given_obs]
            )
    init_state = model.new_state(labels=["init"], **obs_kwarg)

    # we continue calling delta and adding new states until no states are
    # left to be visited
    states_to_be_visited = deque([init])
    # the state type needs to be hashable
    state_lookup = {init: init_state}
    while len(states_to_be_visited) > 0:
        state = states_to_be_visited.popleft()
        choice = {}

        if model.supports_actions():
            # we loop over all available actions and call the delta function for each actions
            assert available_actions is not None
            actionslist = available_actions(state)

            if actionslist is None:
                raise ValueError(
                    f"On input {state}, the available actions function does not have a return value"
                )

            if not isinstance(actionslist, list):
                raise ValueError(
                    f"On input {state}, the available actions function does not return a list. Make sure to change it to [{actionslist}]"
                )

            for action in actionslist:
                # Actions must be strings
                if not isinstance(action, str):
                    raise ValueError(
                        f"On input {state}, the available actions function returns an action that is not a string: {action}"
                    )

                stormvogel_action = (
                    stormvogel.model.EmptyAction
                    if action == ""
                    else model.action(action)
                )

                delta = cast(Callable[[Any, str], Any], delta)
                tuples = delta(state, action)

                if not isinstance(tuples, list) and tuples is not None:
                    # If the delta does not return a list, we assume it's a single transition with probability 1
                    tuples = [(1, tuples)]

                branch = add_new_choices(tuples, state)

                if branch != []:
                    choice[stormvogel_action] = stormvogel.model.Distribution(branch)
        else:
            delta = cast(Callable[[Any], Any], delta)
            tuples = delta(state)

            if not isinstance(tuples, list) and tuples is not None:
                raise ValueError(
                    f"On input {state}, the delta function does not return a list. Make sure to change the format to [(<value>,<state>),...]"
                )

            branch = add_new_choices(tuples, state)

            if branch != []:
                choice[stormvogel.model.EmptyAction] = branch

        s = state_lookup[state]
        assert s is not None
        model.add_choices(
            s,
            choice,
        )

        # if at some point we discovered more than max_size states, we complain
        # if len(list(state_lookup)) % 1000 == 0:
        #     print(f"Building model is slow, discovered {len(list(state_lookup))} states...", file=sys.stderr)
        if len(state_lookup) > max_size:
            raise RuntimeError(
                f"The model you want to create has a very large amount of states (at least {max_size}). If you wish to proceed, set max_size to some larger number."
            )

    # we add the rewards
    if rewards is not None:
        # TODO: Support state-action rewards in the future.
        sig = inspect.signature(rewards)
        num_params = len(sig.parameters)

        if num_params == 2:

            def rewards_1(s: Any) -> dict[str, ValueType]:
                return rewards(s, None)  # type: ignore
        else:
            rewards_1 = cast(Callable[[Any], dict[str, ValueType]], rewards)  # type: ignore

        for name in rewards_1(init).keys():
            model.new_reward_model(name)

        initial_state_rewards = rewards_1(init)
        for state, s in state_lookup.items():
            rewarddict = rewards_1(state)

            if rewarddict is None:
                raise ValueError(
                    f"On input {state}, the rewards function does not have a return value"
                )
            if not isinstance(rewarddict, dict):
                raise ValueError(
                    f"On input {state}, the rewards function does not return a dictionary. Make sure to change it to the format {{<rewardmodel name>:<reward>,...}}"
                )
            if rewarddict.keys() != initial_state_rewards.keys():
                raise ValueError(
                    "Make sure that the rewards function returns a dictionary with the same keys on each return"
                )

            s = state_lookup[state]
            assert s is not None
            for index, reward in enumerate(rewarddict.items()):
                model.rewards[index].set_state_reward(s, reward[1])
    # we add the observations
    if observations is not None:
        for state, s in state_lookup.items():
            # we check for the observations when it does not return an integer
            given_obs = observations(state)
            if given_obs is None:
                raise ValueError(
                    f"On input {state}, the observations function does not have a return value"
                )

            if isinstance(given_obs, int):
                obs = model.observation(str(given_obs))
                s.observation = obs
            elif isinstance(given_obs, list):
                obs_distribution = stormvogel.model.Distribution(
                    [(prob, model.observation(str(o))) for prob, o in given_obs]
                )
                s.observation = obs_distribution
            else:
                raise ValueError(
                    f"On input {state}, the observations function does not return an integer or a distribution"
                )

        if observation_valuations is not None and model.observations is not None:
            # TODO this seems fragile
            observation_valuation_keys = observation_valuations(
                int(next(iter(model.observations)).alias)
            ).keys()
            for obs in model.observations:
                valuation_dict = observation_valuations(int(obs.alias))
                if valuation_dict is None:
                    raise ValueError(
                        f"On input observation id {obs.alias}, the observation_valuations function does not have a return value"
                    )

                if not isinstance(valuation_dict, dict):
                    raise ValueError(
                        f"On input observation id {obs.alias}, the observation_valuations function does not return a dictionary. Make sure to change the format to [<variable>: <value>,...]"
                    )

                if valuation_dict.keys() != observation_valuation_keys:
                    raise RuntimeError(
                        "Make sure that you have a value for each variable in each observation valuation"
                    )

                for val in valuation_dict.values():
                    if not (
                        isinstance(val, int)
                        or isinstance(val, bool)
                        or isinstance(val, float)
                    ):
                        raise ValueError(
                            f"On input observation id {obs.alias}, the dictionary that the observation_valuations function returns contains a value {val} which is not of type int, float or bool"
                        )

                model.observation_valuations[obs] = valuation_dict

    # we add the exit rates
    if rates is not None:
        for state, s in state_lookup.items():
            r = rates(state)
            if not isinstance(r, stormvogel.model.Value):
                raise ValueError(
                    f"On input {state}, the rates function does not return a number"
                )
            if model.model_type.name == "CTMC":
                if s in model.transitions:
                    for a in model.transitions[s].actions:
                        model.transitions[s][a] = stormvogel.model.Distribution(
                            [(v * r, t) for v, t in model.transitions[s][a]]
                        )
            else:
                pass
    # we add the valuations
    if valuations is not None:
        initial_state_valuations = valuations(init)
        for state, s in state_lookup.items():
            valuation_dict = valuations(state)
            if valuation_dict is None:
                raise ValueError(
                    f"On input {state}, the valuations function does not have a return value"
                )

            if not isinstance(valuation_dict, dict):
                raise ValueError(
                    f"On input {state}, the valuations function does not return a dictionary. Make sure to change the format to [<variable>: <value>,...]"
                )

            if valuation_dict.keys() != initial_state_valuations.keys():
                raise RuntimeError(
                    "Make sure that you have a value for each variable in each state"
                )

            for val in valuation_dict.values():
                if not (
                    isinstance(val, int)
                    or isinstance(val, bool)
                    or isinstance(val, float)
                ):
                    raise ValueError(
                        f"On input {state}, the dictionary that the valuations function returns contains a value {val} which is not of type int, float or bool"
                    )

            s.valuations = valuation_dict

    # we add the labels
    if labels is not None:
        for state, s in state_lookup.items():
            labellist = labels(state)

            assert s is not None

            # if the labels function has no return value we assume this state simply has no labels
            if labellist is None:
                continue

            if isinstance(labellist, str):
                if labellist not in s.labels:
                    s.add_label(labellist)
            elif isinstance(labellist, Iterable):
                for label in labellist:
                    if not isinstance(label, str):
                        raise ValueError(
                            f"On input {state}, the labels function does not return a string or an iterable over strings"
                        )
                    if label not in s.labels:
                        s.add_label(label)
            else:
                raise ValueError(
                    f"On input {state}, the labels function does not return a string or an iterable over strings"
                )

    # friendly names
    if friendly_names is not None:
        for state, s in state_lookup.items():
            name = friendly_names(state)
            s.set_friendly_name(name)

    return model
