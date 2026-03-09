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
    """Sample the gym environment.

    Gym environments are POMDPs; gymnasium only exposes observations.
    States that differ in gym but share the same observation and termination
    are considered the same state in the result.

    :param env: Gymnasium environment.
    :param no_samples: Total number of samples (starting at an initial state).
        To resolve multiple initial states, a new single initial state is added
        if necessary.
    :param sample_length: Maximum length of a single sample.
    :param gymnasium_scheduler: Function mapping states to action numbers.
    :param convert_obs: Convert observations to a hashable type. Rounding can
        also be applied here.
    :returns: A 6-tuple ``(initial_states, states, transition_counts,
        transition_samples, reward_sums, no_actions)``.
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
    """Create a stormvogel MDP from a sampling.

    Use :func:`sample_gym` to obtain a sample from gym.
    Probabilities are frequentist estimates whose accuracy depends on how
    often each state is visited.

    :param initial_states: Mapping from initial states to observation counts.
    :param transition_counts: Counts how many times each ``(state, action)``
        to state transition was observed.
    :param transition_samples: Counts how many times each ``(state, action)``
        pair was observed.
    :param reward_sums: Sum of rewards for each ``(state, action)`` pair.
    :param no_actions: Number of different actions observed.
    :param no_samples: Number of samples used to obtain this sampling.
    :param max_size: Maximum number of states in the resulting model.
    :returns: Stormvogel MDP model.
    """
    NEW_INITIAL_STATE = "GYM_SAMPLE_INIT"
    ALL_ACTIONS = [str(x) for x in range(no_actions)]
    INV_MAP = {a: no for no, a in enumerate(ALL_ACTIONS)}

    if len(initial_states) == 1:
        init_obs, init_done = list(initial_states.keys())[0]
        init = (init_obs, init_done, None)
    else:
        init = NEW_INITIAL_STATE

    def available_actions(s):
        if s is NEW_INITIAL_STATE:
            return [""]
        # s is now (obs, done, proxy_action)
        if s[1] or s[2] is not None:
            return [""]
        return [a for a in ALL_ACTIONS if transition_counts[((s[0], s[1]), INV_MAP[a])]]

    def delta(s, a):
        if s is NEW_INITIAL_STATE:
            return [
                (count / no_samples, (s_[0], s_[1], None))
                for s_, count in initial_states.items()
            ]
        elif s[1]:
            return [(1, s)]
        elif s[2] is not None:
            # Proxy state: transition to the next state
            proxy_action = s[2]
            return [
                (
                    count / transition_samples[((s[0], s[1]), proxy_action)],
                    (s_[0], s_[1], None),
                )
                for s_, count in transition_counts[((s[0], s[1]), proxy_action)].items()
            ]
        else:
            # Normal state: transition to proxy state with probability 1
            return [(1.0, (s[0], s[1], INV_MAP[a]))]

    def rewards(s) -> dict[str, stormvogel.model.Value]:
        if s is NEW_INITIAL_STATE or s[1] or s[2] is None:
            return {"R": 0.0}

        proxy_action = s[2]
        if transition_samples[((s[0], s[1]), proxy_action)] == 0:
            return {"R": 0.0}

        return {
            "R": reward_sums[((s[0], s[1]), proxy_action)]
            / transition_samples[((s[0], s[1]), proxy_action)]
        }

    def labels(s):
        if s is NEW_INITIAL_STATE or s[2] is not None:
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
    """Sample the gym environment and convert it to a stormvogel MDP.

    Gym environments are POMDPs; gymnasium only exposes observations.
    The result is an MDP where states with the same observations (and
    termination) are lumped together. Probabilities are frequentist estimates
    whose accuracy depends on how often each state is visited.

    :param env: Gymnasium environment.
    :param no_samples: Total number of samples (starting at an initial state).
        To resolve multiple initial states, a new single initial state is added
        if necessary.
    :param sample_length: Maximum length of a single sample.
    :param gymnasium_scheduler: Function mapping states to action numbers.
    :param convert_obs: Convert observations to a hashable type. Rounding can
        also be applied here.
    :param max_size: Maximum number of states in the resulting model.
    :returns: Stormvogel MDP model.
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
