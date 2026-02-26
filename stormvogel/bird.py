import stormvogel.model
from dataclasses import dataclass
from typing import cast, Any, Callable, Sequence
import inspect
from collections import deque


@dataclass
class State:
    """bird state object. Can contain any number of any type of arguments"""

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
    rewards: (
        Callable[[Any, Action], dict[str, ValueType]]
        | Callable[[Any], dict[str, ValueType]]
        | None
    ) = None,
    labels: Callable[[Any], list[str] | str | None] | None = None,
    available_actions: Callable[[Any], list[Action]] | None = None,
    observations: Callable[[Any], int | list[tuple[ValueType, int]]] | None = None,
    rates: Callable[[Any], float] | None = None,
    valuations: Callable[[Any], dict[str, float | int | bool]] | None = None,
    observation_valuations: Callable[[int], dict[str, float | int | bool]]
    | None = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
):
    """
    function that checks if the input for the bird model builder is valid
    it will give a runtime error if it isn't.
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
        if supports_actions:
            if num_params != 2:
                raise ValueError(
                    f"The rewards function must take exactly two arguments (state, action), but it takes {num_params} arguments"
                )
        else:
            if num_params != 1:
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
    delta: Callable[
        [Any, Action], Sequence[tuple[ValueType, Any]] | Sequence[Any] | None
    ]
    | Callable[[Any], Sequence[tuple[ValueType, Any]] | Sequence[Any] | None],
    init: Any,
    rewards: (
        Callable[[Any, Action], dict[str, ValueType]]
        | Callable[[Any], dict[str, ValueType]]
        | None
    ) = None,
    labels: Callable[[Any], list[str] | str | None] | None = None,
    available_actions: Callable[[Any], list[Action]] | None = None,
    observations: Callable[[Any], int | list[tuple[ValueType, int]]] | None = None,
    rates: Callable[[Any], float] | None = None,
    valuations: Callable[[Any], dict[str, float | int | bool]] | None = None,
    observation_valuations: Callable[[int], dict[str, float | int | bool]]
    | None = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
    max_size: int = 10000,
) -> stormvogel.model.Model[ValueType]:
    """
    function that converts a delta function, an available_actions function an initial state and a model type
    to a stormvogel model

    this works analogous to a prism file, where the delta is the module in this case.

    (this function uses the bird classes state and action instead of the ones from stormvogel.model)
    """

    def add_new_choices(tuples, state):
        """
        helper function to add all the newly found choices and states to the model
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
                            obs_kwarg["observation"] = [
                                (prob, model.observation(str(o)))
                                for prob, o in given_obs
                            ]
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
            obs_kwarg["observation"] = [
                (prob, model.observation(str(o))) for prob, o in given_obs
            ]
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

                # Convert empty strings to None for the empty action
                action_label = None if action == "" else action
                stormvogel_action = model.action(action_label)

                delta = cast(Callable[[Any, str], Any], delta)
                tuples = delta(state, action)

                if not isinstance(tuples, list) and tuples is not None:
                    # If the delta does not return a list, we assume it's a single transition with probability 1
                    tuples = [(1, tuples)]

                branch = add_new_choices(tuples, state)

                if branch != []:
                    choice[stormvogel_action] = stormvogel.model.Branches(branch)
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
        if model.supports_actions():
            assert available_actions is not None
            rewards_2 = cast(Callable[[Any, Action], dict[str, ValueType]], rewards)
            # discover reward model names using first state and its first action
            sample_action = available_actions(init)[0]
            for name in rewards_2(init, sample_action).keys():
                model.new_reward_model(name)

            initial_state_rewards = rewards_2(init, sample_action)
            for state, s in state_lookup.items():
                for action_str in available_actions(state):
                    rewarddict = rewards_2(state, action_str)

                    if rewarddict is None:
                        raise ValueError(
                            f"On input {state}, {action_str}, the rewards function does not have a return value"
                        )
                    if not isinstance(rewarddict, dict):
                        raise ValueError(
                            f"On input {state}, {action_str}, the rewards function does not return a dictionary. Make sure to change it to the format {{<rewardmodel name>:<reward>,...}}"
                        )
                    if rewarddict.keys() != initial_state_rewards.keys():
                        raise ValueError(
                            "Make sure that the rewards function returns a dictionary with the same keys on each return"
                        )

                    s = state_lookup[state]
                    assert s is not None
                    action_label = None if action_str == "" else action_str
                    stormvogel_action = model.action(action_label)
                    for index, reward in enumerate(rewarddict.items()):
                        model.rewards[index].set_state_reward(s, reward[1])
        else:
            rewards_1 = cast(Callable[[Any], dict[str, ValueType]], rewards)
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
                obs_distribution = [
                    (prob, model.observation(str(o))) for prob, o in given_obs
                ]
                s.observation = obs_distribution
            else:
                raise ValueError(
                    f"On input {state}, the observations function does not return an integer or a distribution"
                )

        if observation_valuations is not None and model.observations is not None:
            observation_valuation_keys = observation_valuations(
                int(model.observations[0].alias)
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

                obs.valuations = valuation_dict

    # we add the exit rates
    if rates is not None:
        for state, s in state_lookup.items():
            r = rates(state)
            if not isinstance(r, stormvogel.model.Value):
                raise ValueError(
                    f"On input {state}, the rates function does not return a number"
                )
            if model.model_type.name == "CTMC":
                if s in model.choices:
                    for a in model.choices[s].choices:
                        new_branch = [
                            (v * r, t) for v, t in model.choices[s].choices[a].branch
                        ]
                        model.choices[s].choices[a].branch = new_branch
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

            # we check for the labels when the function does not return a list object, or when
            # the list does not consist of strings
            if not isinstance(labellist, list):
                if not isinstance(labellist, str):
                    raise ValueError(
                        f"On input {state}, the labels function does not return a string or a list of strings"
                    )
                # if we don't get a list, we assume there is just one label
                if labellist not in s.labels:
                    s.add_label(labellist)
            else:
                for label in labellist:
                    if not isinstance(label, str):
                        raise ValueError(
                            f"On input {state}, the labels function does not return a string or a list of strings"
                        )
                    if label not in s.labels:
                        s.add_label(label)

    return model
