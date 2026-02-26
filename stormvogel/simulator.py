import stormvogel.result
import stormvogel.model
from typing import Callable
import random


class Path:
    """
    Path object that represents a path created by a simulator on a certain model.

    Args:
        path: The path itself is a dictionary where we either store for each step a state or a state action pair,
        depending on if we are working with a dtmc or an mdp.
        model: model that the path traverses through
    """

    path: (
        list[tuple[stormvogel.model.Action, stormvogel.model.State]]
        | list[stormvogel.model.State]
    )
    model: stormvogel.model.Model

    def __init__(
        self,
        path: (
            list[tuple[stormvogel.model.Action, stormvogel.model.State]]
            | list[stormvogel.model.State]
        ),
        model: stormvogel.model.Model,
    ):
        if model.model_type != stormvogel.model.ModelType.MA:
            self.path = path
            self.model = model
        else:
            # TODO make the simulators work for markov automata
            raise NotImplementedError

    def get_state_in_step(self, step: int) -> stormvogel.model.State | None:
        """returns the state discovered in the given step in the path"""
        if not self.model.supports_actions():
            state = self.path[step]
            assert isinstance(state, stormvogel.model.State)
            return state
        if self.model.supports_actions():
            t = self.path[step]
            assert isinstance(t, tuple)
            action, state = t
            assert isinstance(action, stormvogel.model.Action) and isinstance(
                state, stormvogel.model.State
            )
            return state

    def get_action_in_step(self, step: int) -> stormvogel.model.Action | None:
        """returns the action discovered in the given step in the path"""
        if self.model.supports_actions():
            t = self.path[step]
            assert (
                isinstance(t, tuple)
                and isinstance(t[0], stormvogel.model.Action)
                and isinstance(t[1], stormvogel.model.State)
            )
            action = t[0]
            return action

    def get_step(
        self, step: int
    ) -> (
        tuple[stormvogel.model.Action, stormvogel.model.State] | stormvogel.model.State
    ):
        """returns the state or state action pair discovered in the given step"""
        return self.path[step]

    def to_state_action_sequence(
        self,
    ) -> list[stormvogel.model.Action | stormvogel.model.State]:
        """Convert a Path to a list containing actions and states."""
        res: list[stormvogel.model.Action | stormvogel.model.State] = [
            self.model.initial_state
        ]
        for v in self.path:
            if isinstance(v, tuple):
                res += list(v)
            else:
                res.append(v)
        return res

    def __str__(self) -> str:
        path = "initial state"
        if self.model.supports_actions():
            for t in self.path:
                assert (
                    isinstance(t, tuple)
                    and isinstance(t[0], stormvogel.model.Action)
                    and isinstance(t[1], stormvogel.model.State)
                )
                path += f" --(action: {t[0].label})--> state: {t[1].state_id}"
        else:
            for state in self.path:
                path += f" --> state: {state}"
        return path

    def __eq__(self, other):
        if not isinstance(other, Path):
            return False

        if len(self.path) != len(other.path):
            return False

        if self.model.supports_actions():
            for tuple, other_tuple in zip(self.path, other.path):
                assert not (
                    isinstance(tuple, stormvogel.model.State)
                    or isinstance(other_tuple, stormvogel.model.State)
                )
                if not (tuple[0] == other_tuple[0] or tuple[1] == other_tuple[1]):
                    return False

            return self.model == other.model
        else:
            return self.path == other.path and self.model == other.model

    def __len__(self):
        return len(self.path)


def get_action_at_state(
    state: stormvogel.model.State,
    scheduler: (
        stormvogel.result.Scheduler
        | Callable[[stormvogel.model.State], stormvogel.model.Action]
    ),
) -> stormvogel.model.Action:
    """Helper function to obtain the chosen action in a state by a scheduler."""
    assert scheduler is not None
    if isinstance(scheduler, stormvogel.result.Scheduler):
        action = scheduler.get_action_at_state(state)
    elif callable(scheduler):
        action = scheduler(state)
    else:
        raise TypeError("Must be of type Scheduler or a function")

    return action


def step(
    state: stormvogel.model.State,
    action: stormvogel.model.Action | None = None,
    seed: int | None = None,
) -> tuple[stormvogel.model.State, list[stormvogel.model.Number], list[str]]:
    """given a state, action and seed we simulate a step and return information on the state we discover"""

    # we go to the next state according to the probability distribution of the transition
    transitions = state.get_outgoing_transitions(action)
    assert transitions is not None  # what if there are no transitions?

    # we build the probability distribution
    probability_distribution = []
    for t in transitions:
        # this action is not supported for parametric and interval models
        assert isinstance(t[0], stormvogel.model.Number)
        probability_distribution.append(float(t[0]))

    # we select the next state (according to the seed)
    states = [t[1] for t in transitions]
    if seed is not None:
        rng = random.Random(seed)
        next_state = rng.choices(states, k=1, weights=probability_distribution)[0]
    else:
        next_state = random.choices(states, k=1, weights=probability_distribution)[0]

    # we also add the rewards
    rewards = []
    if not next_state.model.supports_actions():
        for rewardmodel in next_state.model.rewards:
            rewards.append(rewardmodel.get_state_reward(next_state))
    else:
        for rewardmodel in next_state.model.rewards:
            assert action is not None
            rewards.append(rewardmodel.get_state_reward(state))
    return next_state, rewards, next_state.labels


def simulate_path(
    model: stormvogel.model.Model,
    steps: int = 1,
    scheduler: (
        stormvogel.result.Scheduler
        | Callable[[stormvogel.model.State], stormvogel.model.Action]
        | None
    ) = None,
    seed: int | None = None,
) -> Path:
    """
    Simulates the model and returns the path created by the process.
    Args:
        model: The stormvogel model that the simulator should run on.
        steps: The number of steps the simulator walks through the model.
        scheduler: A stormvogel scheduler to determine what actions should be taken. Random if not provided.
                    (instead of a stormvogel scheduler, a function from states to actions can also be provided.)
        seed: The seed for the function that determines for each state what the next state will be. Random seed if not provided.

    Returns a path object.
    """

    # we need to set the seed for choosing actions in case no scheduler is provided
    random.seed(seed)

    # we start adding states or state action pairs to the path
    state = model.states[0]
    path = []
    if not model.supports_actions():
        for i in range(steps):
            # for each step we add a state to the path
            if not state.is_absorbing():
                next_state, _, _ = step(
                    state,
                    seed=seed + i if seed is not None else None,
                )
                state = next_state
                path.append(state)
            else:
                break
    else:
        for i in range(steps):
            # we first choose an action (randomly or according to scheduler)
            action = (
                get_action_at_state(state, scheduler)
                if scheduler
                else random.choice(state.available_actions())
            )

            # we append the next state action pair
            if not state.is_absorbing():
                next_state, _, _ = step(
                    state,
                    action,
                    seed=seed + i if seed is not None else None,
                )
                state = next_state
                path.append((action, state))
            else:
                break

    return Path(path, model)


def simulate(
    model: stormvogel.model.Model,
    steps: int = 1,
    runs: int = 1,
    scheduler: (
        stormvogel.result.Scheduler
        | Callable[[stormvogel.model.State], stormvogel.model.Action]
        | None
    ) = None,
    seed: int | None = None,
) -> stormvogel.model.Model | None:
    """
    Simulates the model.
    Args:
        model: The stormvogel model that the simulator should run on
        steps: The number of steps the simulator walks through the model
        runs: The number of times the model gets simulated.
        scheduler: A stormvogel scheduler to determine what actions should be taken. Random if not provided.
                    (instead of a stormvogel scheduler, a function from states to actions can also be provided.)
        seed: The seed for the function that determines for each state what the next state will be. Random seed if not provided.

    Returns the partial model discovered by all the runs of the simulator together
    """

    # we need to set the seed for choosing actions in case no scheduler is provided
    random.seed(seed)

    # we keep track of all discovered states over all runs and add them to the partial model
    # we also add the discovered rewards and actions to the partial model if present
    partial_model = stormvogel.model.new_model(
        model.model_type, create_initial_state=False
    )

    # we create the initial state ourselves because we want to force the name to be 0
    init = partial_model.new_state(name="0", labels="init")
    init.valuations = model.initial_state.valuations

    # we add each (empty) rewardmodel to the partial model
    if model.rewards:
        for index, reward in enumerate(model.rewards):
            reward_model = partial_model.new_reward_model(model.rewards[index].name)

            # we already set the rewards for the initial state/stateaction
            if model.supports_actions():
                try:
                    r = model.rewards[index].get_state_reward(model.initial_state)
                    assert r is not None
                    reward_model.set_state_reward(
                        partial_model.initial_state,
                        r,
                    )
                except RuntimeError:
                    pass
            else:
                r = model.rewards[index].get_state_reward(model.initial_state)
                assert r is not None
                reward_model.set_state_reward(
                    partial_model.initial_state,
                    r,
                )

    # we keep track of all discovered states over all runs
    discovered_states: set[stormvogel.model.State] = {model.states[0]}
    discovered_transitions: set[tuple] = set()
    # map from original model states to partial model states
    state_map: dict[stormvogel.model.State, stormvogel.model.State] = {
        model.states[0]: init
    }

    # we distinguish between models with and without actions
    if not partial_model.supports_actions():
        discovered_states_before_transitions: set[stormvogel.model.State] = set()
        # now we start stepping through the model for the given number of runs
        for i in range(runs):
            # we start at state 0 and we begin taking steps
            last_state = model.states[0]
            for j in range(steps):
                # we make a step
                next_state, reward, labels = step(
                    last_state,
                    seed=seed + i + j if seed is not None else None,
                )

                # we add to the partial model what we discovered (if new)
                if next_state not in discovered_states:
                    discovered_states.add(next_state)
                    new_state = partial_model.new_state(
                        list(labels),
                        valuations=next_state.valuations,
                    )
                    state_map[next_state] = new_state

                    # we add the rewards
                    for index, rewardmodel in enumerate(partial_model.rewards):
                        rewardmodel.set_state_reward(new_state, reward[index])
                else:
                    new_state = state_map[next_state]

                # we also add the transitions that we travelled through, so we need to keep track of the last state
                # and of the discovered transitions so that we don't add duplicates
                if (last_state, next_state) not in discovered_transitions:
                    discovered_transitions.add((last_state, next_state))
                    assert new_state is not None

                    # calculate the transition probability
                    choice = last_state.choices
                    probability = 0
                    for t in choice.choices[stormvogel.model.EmptyAction].branch:
                        if t[1] == next_state:
                            assert isinstance(t[0], (float, int))
                            probability += float(t[0])

                    s = state_map[last_state]
                    if last_state in discovered_states_before_transitions:
                        branch = partial_model.choices[s].choices[
                            stormvogel.model.EmptyAction
                        ]
                        branch.branch.append((probability, new_state))
                    else:
                        discovered_states_before_transitions.add(last_state)
                        s.add_choices([(probability, new_state)])

                last_state = next_state
    else:
        # we additionally keep track of actions
        discovered_actions: set[tuple] = set()
        # now we start stepping through the model for the given number of runs
        for i in range(runs):
            # we start at state 0 and we begin taking steps
            last_state = model.states[0]
            for j in range(steps):
                # we first choose an action
                action = (
                    get_action_at_state(last_state, scheduler)
                    if scheduler
                    else random.choice(last_state.available_actions())
                )

                # we add the action to the partial model (if new)
                assert partial_model.actions is not None
                if action not in partial_model.actions:
                    partial_model.new_action(action.label)

                # we get the new discovery
                next_state, reward, labels = step(
                    last_state,
                    action,
                    seed=seed + i + j if seed is not None else None,
                )

                # we add the state to the model
                if next_state not in discovered_states:
                    discovered_states.add(next_state)
                    new_state = partial_model.new_state(
                        list(labels),
                        valuations=next_state.valuations,
                    )
                    state_map[next_state] = new_state
                else:
                    new_state = state_map[next_state]

                # we also add the transitions that we travelled through, so we need to keep track of the last state
                # and of the discovered transitions so that we don't add duplicates
                if (last_state, next_state, action) not in discovered_transitions:
                    transitions = last_state.get_outgoing_transitions(action)
                    discovered_transitions.add((last_state, next_state, action))

                    # calculate the transition probability
                    probability = 0
                    assert transitions is not None
                    for t in transitions:
                        if t[1] == next_state:
                            assert isinstance(t[0], (float, int))
                            probability += float(t[0])

                    assert new_state is not None
                    s = state_map[last_state]
                    if (last_state, action) in discovered_actions:
                        branch = partial_model.choices[s].choices[action]
                        branch.branch.append((probability, new_state))
                    else:
                        discovered_actions.add((last_state, action))
                        s.add_choices({action: [(probability, new_state)]})
                        # set the rewards now that the action is available
                        for index, rewardmodel in enumerate(partial_model.rewards):
                            r = model.rewards[index].get_state_reward(last_state)
                            if r is not None:
                                rewardmodel.set_state_reward(s, r)

                last_state = next_state

    return partial_model
