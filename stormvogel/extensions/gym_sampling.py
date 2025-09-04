from typing import Callable, Any
import gymnasium as gym
from collections import defaultdict

from typing import Tuple
from stormvogel import bird
import stormvogel.model


def sample_gym(
    env: gym.Env,
    no_samples: int = 10,
    sample_length: int = 20,
    gymnasium_scheduler: Callable[[Any], int] | None = None,
    convert_obs: Callable[[Any], Any] = lambda x: x,
) -> Tuple:
    """Sample the gym environment. In reality, gym environments are POMDPs, and gymnasium only allows us to access the observation.
    States that are different in gym, but have the same observation and termination will be considered the same state in the result.

    Args:
        env (gym.Env): Gymnasium env.
        no_samples (int): Total number of samples (starting at an initial state).
            To resolve multiple initial states, a new, single initial state is added if necessary.
        sample_length (int): The maximum length of a single sample.
        gymnasium_scheduler (Callable[[any], int] | None): A function from states to action numbers.
        convert_obs (Callable[[any], any]): Converts the observations to a hashable type. You can also apply rounding here.

    Returns:
        A 4-tuple consiting of four defaultdicts and one integer.
        * initial_states (defaultdict[state, int]): The initial state in gym may be non-deterministic. This maps the initial states to the amount of times they were observed as the initial state.
        * states (defaultdict[state, int]): Maps states to the amount of times they were observed.
        * transition_counts (defaultdict[(state,action), defaultdict[state, int]]): Counts how many times the transition between this state-action pair and state was observed.
        * transition_samples (defaultdict[(state,action), int]): Counts how many times this state-action pair was observed.
        * reward_sums (defaultdict[(state,action), int]): The sum of the rewards for this state-action pair.
        * no_actions (int): The number of different actions observed.
    """
    initial_states = defaultdict(lambda: 0)
    states = defaultdict(lambda: 0)
    transition_counts = defaultdict(lambda: defaultdict(lambda: 0))
    transition_samples = defaultdict(lambda: 0)
    reward_sums = defaultdict(lambda: 0.0)

    for s_no in range(no_samples):
        prev_state = None
        obs, _ = env.reset()
        state = (convert_obs(obs), False)
        initial_states[state] += 1
        states[state] += 1
        for _ in range(sample_length):
            action = (
                env.action_space.sample()
                if gymnasium_scheduler is None
                else gymnasium_scheduler(state)
            )
            prev_state = state
            obs, reward, terminated, truncated, info = env.step(action)
            state = (convert_obs(obs), terminated)
            states[state] += 1
            transition_counts[(prev_state, action)][state] += 1
            transition_samples[(prev_state, action)] += 1
            reward_sums[(prev_state, action)] += float(reward)
            if terminated:
                break
    return (
        initial_states,
        states,
        transition_counts,
        transition_samples,
        reward_sums,
        env.action_space.n,
    )


def sample_to_stormvogel(
    initial_states: defaultdict[Any, int],
    transition_counts: defaultdict[Tuple[Any, Any], defaultdict[Any, int]],
    transition_samples: defaultdict[Tuple[Any, Any], int],
    reward_sums: defaultdict[Tuple[Any, Any], int],
    no_actions: int,
    no_samples: int,
    max_size: int = 10000,
) -> stormvogel.model.Model:
    """Create a Stormvogel mdp from a sampling (see sample_gym to obtain a sample from gym).
    Probablities are frequentist estimates. Their accuracy depends on how often each "state" is visited.

    Args:
        initial_states (defaultdict[state, int]): The initial state in gym may be non-deterministic.
            This maps the initial states to the amount of times they were observed as the initial state.
        transition_counts (defaultdict[(state,action), defaultdict[state, int]]):
            Counts how many times the transition between this state-action pair and state was observed.
        transition_samples (defaultdict[(state,action), int]): Counts how many times this state-action pair was observed.
        reward_sums (defaultdict[(state,action), int]): The sum of the rewards for this state-action pair.
        no_actions (int): The number of different actions observed.
        no_samples (int): The number of samples that were used to obtain this sampling.
        max_size (int): The maximum number of states in the resulting model. Defaults to 10000.
    """
    NEW_INITIAL_STATE = "GYM_SAMPLE_INIT"
    ALL_ACTIONS = [[str(x)] for x in range(no_actions)]
    INV_MAP = {a[0]: no for no, a in enumerate(ALL_ACTIONS)}

    if len(initial_states) == 1:
        (init,) = initial_states
    else:
        init = NEW_INITIAL_STATE

    def available_actions(s):
        if s is NEW_INITIAL_STATE:
            return [[]]
        elif s[1]:
            return [[]]
        return [a for a in ALL_ACTIONS if transition_counts[(s, INV_MAP[a[0]])]]

    def delta(s, a):
        if s is NEW_INITIAL_STATE:
            return [(count / no_samples, s_) for s_, count in initial_states.items()]
        elif s[1]:
            return [(1, s)]
        return [
            (count / transition_samples[(s, INV_MAP[a[0]])], s_)
            for s_, count in transition_counts[(s, INV_MAP[a[0]])].items()
        ]

    def rewards(s, a) -> dict[str, stormvogel.model.Value]:
        if s is NEW_INITIAL_STATE or s[1]:
            return {"R": 0}
        return {
            "R": reward_sums[s, INV_MAP[a[0]]] / transition_samples[(s, INV_MAP[a[0]])]
        }

    def labels(s):
        if s is NEW_INITIAL_STATE:
            return []
        done = ["done"] if s[1] else []
        return [str(s[0])] + done

    return bird.build_bird(
        delta=delta,
        init=init,
        available_actions=available_actions,
        labels=labels,
        rewards=rewards,  # type: ignore
        modeltype=stormvogel.model.ModelType.MDP,
        max_size=max_size,
    )


def sample_gym_to_stormvogel(
    env: gym.Env,
    no_samples: int = 10,
    sample_length: int = 20,
    gymnasium_scheduler: Callable[[Any], int] | None = None,
    convert_obs: Callable[[Any], Any] = lambda x: x,
    max_size: int = 10000,
):
    """Sample the gym environment and convert it to a Stormvogel MDP.
    In reality, gym environments are POMDPs, and gymnasium only allows us to access the observation.
    The result is an MDP where states with the same observations (and termination) are lumped together.
    Probablities are frequentist estimates. Their accuracy depends on how often each "state" is visited.

    Args:
        env (gym.Env): Gymnasium env.
        no_samples (int): Total number of samples (starting at an initial state).
            To resolve multiple initial states, a new, single initial state is added if necessary.
        sample_length (int): The maximum length of a single sample.
        gymnasium_scheduler (Callable[[any], int] | None): A function from states to action numbers.
        convert_obs (Callable[[any], any]): Converts the observations to a hashable type. You can also apply rounding here.
        max_size (int): The maximum number of states in the resulting model. Defaults to 10000.
    """
    (
        initial_states,
        _,
        transition_counts,
        transition_samples,
        reward_sums,
        no_actions,
    ) = sample_gym(env, no_samples, sample_length, gymnasium_scheduler, convert_obs)
    return sample_to_stormvogel(
        initial_states,
        transition_counts,
        transition_samples,
        reward_sums,
        no_actions,
        no_samples,
        max_size,
    )
