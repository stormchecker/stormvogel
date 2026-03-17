import stormvogel.result
import stormvogel.model
from typing import Callable
import random


class Path:
    """Represent a path created by a simulator on a certain model.

    :param path: The path itself is a list where we either store for each step a state or a
        state-action pair, depending on whether we are working with a DTMC or an MDP.
    :param model: Model that the path traverses through.
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
        """Return the state discovered in the given step in the path.

        :param step: The step index to look up.
        :returns: The state at the given step, or ``None``.
        """
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
        """Return the action discovered in the given step in the path.

        :param step: The step index to look up.
        :returns: The action at the given step, or ``None``.
        """
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
        """Return the state or state-action pair discovered in the given step.

        :param step: The step index to look up.
        :returns: The state or state-action pair at the given step.
        """
        return self.path[step]

    def to_state_action_sequence(
        self,
    ) -> list[stormvogel.model.Action | stormvogel.model.State]:
        """Convert a Path to a list containing actions and states.

        :returns: A flat list of actions and states from this path.
        """
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

    def __len__(self):
        return len(self.path)


def get_action_at_state(
    state: stormvogel.model.State,
    scheduler: (
        stormvogel.result.Scheduler
        | Callable[[stormvogel.model.State], stormvogel.model.Action]
    ),
) -> stormvogel.model.Action:
    """Obtain the chosen action in a state by a scheduler.

    :param state: The state to get the action for.
    :param scheduler: A scheduler or callable that maps states to actions.
    :returns: The action chosen by the scheduler at the given state.
    :raises TypeError: If scheduler is not a Scheduler or callable.
    """
    assert scheduler is not None
    if isinstance(scheduler, stormvogel.result.Scheduler):
        action = scheduler.get_action_at_state(state)
    elif callable(scheduler):
        action = scheduler(state)
    else:
        raise TypeError("Must be of type Scheduler or a function")

    return action


def _copy_rewards(
    model: stormvogel.model.Model,
    partial_model: stormvogel.model.Model,
    original_state: stormvogel.model.State,
    partial_state: stormvogel.model.State,
) -> None:
    """Copy rewards from the original model to the partial model for a state.

    Missing rewards default to 0.

    :param model: The original model containing rewards.
    :param partial_model: The partial model to copy rewards into.
    :param original_state: The state in the original model.
    :param partial_state: The corresponding state in the partial model.
    """
    for index, rewardmodel in enumerate(partial_model.rewards):
        r = model.rewards[index].get_state_reward(original_state)
        rewardmodel.set_state_reward(partial_state, r if r is not None else 0)


def _transition_probability(
    from_state: stormvogel.model.State,
    to_state: stormvogel.model.State,
    action: stormvogel.model.Action | None = None,
) -> float:
    """Calculate the total transition probability from one state to another.

    :param from_state: The source state.
    :param to_state: The target state.
    :param action: The action taken, or ``None`` for models without actions.
    :returns: The total transition probability.
    """
    transitions = from_state.get_outgoing_transitions(action)
    assert transitions is not None
    probability = 0.0
    for prob, target in transitions:
        if target == to_state:
            assert isinstance(prob, (float, int))
            probability += float(prob)
    return probability


def step(
    state: stormvogel.model.State,
    action: stormvogel.model.Action | None = None,
    seed: int | None = None,
) -> tuple[stormvogel.model.State, list[stormvogel.model.Number], list[str]]:
    """Simulate one step from the given state.

    Rewards are always the state-exit rewards of the current state.
    Missing rewards default to 0.

    :param state: The current state to step from.
    :param action: The action to take, or ``None`` for models without actions.
    :param seed: Seed for the random state selection. Random if not provided.
    :returns: A tuple of (next_state, state-exit rewards, next_state labels).
    """

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

    # State-exit rewards: always from the current state, regardless of model type.
    rewards: list[stormvogel.model.Number] = []
    for rewardmodel in state.model.rewards:
        r = rewardmodel.get_state_reward(state)
        rewards.append(r if r is not None else 0)

    return next_state, rewards, list(next_state.labels)


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
    """Simulate the model and return the path created by the process.

    :param model: The stormvogel model that the simulator should run on.
    :param steps: The number of steps the simulator walks through the model.
    :param scheduler: A stormvogel scheduler to determine what actions should be taken.
        Random if not provided. A callable from states to actions can also be provided.
    :param seed: The seed for the random state selection. Random if not provided.
    :returns: A path object representing the simulated trajectory.
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
) -> stormvogel.model.Model:
    """Simulate the model over multiple runs.

    :param model: The stormvogel model that the simulator should run on.
    :param steps: The number of steps the simulator walks through the model.
    :param runs: The number of times the model gets simulated.
    :param scheduler: A stormvogel scheduler to determine what actions should be taken.
        Random if not provided. A callable from states to actions can also be provided.
    :param seed: The seed for the random state selection. Random if not provided.
    :returns: The partial model discovered by all the runs of the simulator together.
    """

    # we need to set the seed for choosing actions in case no scheduler is provided
    random.seed(seed)

    partial_model = stormvogel.model.new_model(
        model.model_type, create_initial_state=False
    )

    init = partial_model.new_state(labels="init")
    init.valuations = model.initial_state.valuations

    for reward in model.rewards:
        partial_model.new_reward_model(reward.name)

    # Track discovered states and map original -> partial
    discovered_states: set[stormvogel.model.State] = {model.states[0]}
    discovered_transitions: set[tuple] = set()
    state_map: dict[stormvogel.model.State, stormvogel.model.State] = {
        model.states[0]: init
    }

    _copy_rewards(model, partial_model, model.initial_state, init)

    def discover_state(
        original_state: stormvogel.model.State,
    ) -> stormvogel.model.State:
        """Register original_state if new; return its partial model counterpart."""
        if original_state not in discovered_states:
            discovered_states.add(original_state)
            new = partial_model.new_state(
                list(original_state.labels),
                valuations=original_state.valuations,
            )
            state_map[original_state] = new
            _copy_rewards(model, partial_model, original_state, new)
        return state_map[original_state]

    if not partial_model.supports_actions():
        discovered_states_before_transitions: set[stormvogel.model.State] = set()
        for i in range(runs):
            last_state = model.states[0]
            for j in range(steps):
                next_state, _, _ = step(
                    last_state,
                    seed=seed + i + j if seed is not None else None,
                )

                new_state = discover_state(next_state)

                if (last_state, next_state) not in discovered_transitions:
                    discovered_transitions.add((last_state, next_state))
                    probability = _transition_probability(last_state, next_state)

                    s = state_map[last_state]
                    if last_state in discovered_states_before_transitions:
                        branch = partial_model.choices[s].choices[
                            stormvogel.model.EmptyAction
                        ]
                        branch.branches.distribution.append((probability, new_state))
                    else:
                        discovered_states_before_transitions.add(last_state)
                        s.add_choices([(probability, new_state)])

                last_state = next_state
    else:
        discovered_actions: set[tuple] = set()
        for i in range(runs):
            last_state = model.states[0]
            for j in range(steps):
                action = (
                    get_action_at_state(last_state, scheduler)
                    if scheduler
                    else random.choice(last_state.available_actions())
                )

                assert partial_model.actions is not None
                if action not in partial_model.actions:
                    partial_model.new_action(action.label)

                next_state, _, _ = step(
                    last_state,
                    action,
                    seed=seed + i + j if seed is not None else None,
                )

                new_state = discover_state(next_state)

                if (last_state, next_state, action) not in discovered_transitions:
                    discovered_transitions.add((last_state, next_state, action))
                    probability = _transition_probability(
                        last_state, next_state, action
                    )

                    s = state_map[last_state]
                    if (last_state, action) in discovered_actions:
                        branch = partial_model.choices[s].choices[action]
                        branch.branches.distribution.append((probability, new_state))
                    else:
                        discovered_actions.add((last_state, action))
                        s.add_choices({action: [(probability, new_state)]})

                last_state = next_state

    return partial_model
