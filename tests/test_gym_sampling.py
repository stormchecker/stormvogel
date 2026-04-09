"""Tests for stormvogel/extensions/gym_sampling.py."""

from collections import defaultdict

import pytest

import stormvogel.model as model
from stormvogel.extensions.gym_sampling import (
    sample_gym,
    sample_to_stormvogel,
    sample_gym_to_stormvogel,
)

gymnasium = pytest.importorskip("gymnasium")


# ── Helpers ──────────────────────────────────────────────────────────


def _frozen_lake():
    return gymnasium.make("FrozenLake-v1", is_slippery=False)


# ── sample_gym ───────────────────────────────────────────────────────


def test_sample_gym_initial_states_nonempty():
    env = _frozen_lake()
    initial_states, states, tc, ts, rs, no_actions = sample_gym(
        env, no_samples=5, sample_length=10
    )
    assert len(initial_states) >= 1
    assert no_actions == env.action_space.n
    env.close()


def test_sample_gym_with_scheduler():
    """A fixed scheduler (always action 0) must produce only action-0 transitions."""
    env = _frozen_lake()
    initial_states, states, tc, ts, rs, no_actions = sample_gym(
        env, no_samples=3, sample_length=5, gymnasium_scheduler=lambda s: 0
    )
    # Every (state, action) key recorded must have action == 0
    for _state, action in ts:
        assert action == 0, f"Expected action 0, got {action}"
    env.close()


def test_sample_gym_with_convert_obs():
    """convert_obs=str must produce states whose observation component is a str."""
    env = _frozen_lake()
    initial_states, states, tc, ts, rs, no_actions = sample_gym(
        env, no_samples=2, sample_length=3, convert_obs=str
    )
    # States are (obs, terminated); obs must be a str after conversion
    for obs, _terminated in states:
        assert isinstance(obs, str), f"Expected str observation, got {type(obs)}"
    env.close()


# ── sample_to_stormvogel ─────────────────────────────────────────────


def _minimal_sample_data():
    """Build hand-crafted sample data for a 2-state, 2-action MDP."""
    obs_a = (0, False)
    obs_b = (1, True)
    initial_states = defaultdict(lambda: 0)
    initial_states[obs_a] = 5

    transition_counts = defaultdict(lambda: defaultdict(lambda: 0))
    transition_counts[(obs_a, 0)][obs_b] = 5

    transition_samples = defaultdict(lambda: 0)
    transition_samples[(obs_a, 0)] = 5

    reward_sums = defaultdict(lambda: 0.0)
    reward_sums[(obs_a, 0)] = 10.0

    return initial_states, transition_counts, transition_samples, reward_sums


def test_sample_to_stormvogel_creates_mdp():
    init_s, tc, ts, rs = _minimal_sample_data()
    m = sample_to_stormvogel(init_s, tc, ts, rs, no_actions=2, no_samples=5)
    assert m is not None
    assert m.model_type == model.ModelType.MDP


def test_sample_to_stormvogel_multiple_initial_states():
    """When there are multiple initial states a proxy init state is created,
    so the model has more states than the two observed base states."""
    obs_a = (0, False)
    obs_b = (1, False)
    initial_states = defaultdict(lambda: 0)
    initial_states[obs_a] = 3
    initial_states[obs_b] = 2

    transition_counts = defaultdict(lambda: defaultdict(lambda: 0))
    transition_counts[(obs_a, 0)][(obs_a, False)] = 3
    transition_counts[(obs_b, 0)][(obs_b, False)] = 2

    transition_samples = defaultdict(lambda: 0)
    transition_samples[(obs_a, 0)] = 3
    transition_samples[(obs_b, 0)] = 2

    reward_sums = defaultdict(lambda: 0.0)

    m = sample_to_stormvogel(
        initial_states,
        transition_counts,
        transition_samples,
        reward_sums,
        no_actions=1,
        no_samples=5,
    )
    assert m.model_type == model.ModelType.MDP
    # proxy init + obs_a + obs_b (+ self-loops on obs_a/obs_b) → at least 3 states
    assert m.nr_states >= 3


def test_sample_to_stormvogel_reward_differs_by_action():
    """When rewards differ by action, proxy states are created,
    so the model has more states than without proxies."""
    obs_a = (0, False)
    obs_b = (1, True)
    initial_states = defaultdict(lambda: 0)
    initial_states[obs_a] = 4

    transition_counts = defaultdict(lambda: defaultdict(lambda: 0))
    transition_counts[(obs_a, 0)][obs_b] = 2
    transition_counts[(obs_a, 1)][obs_b] = 2

    transition_samples = defaultdict(lambda: 0)
    transition_samples[(obs_a, 0)] = 2
    transition_samples[(obs_a, 1)] = 2

    reward_sums = defaultdict(lambda: 0.0)
    reward_sums[(obs_a, 0)] = 4.0  # avg = 2.0
    reward_sums[(obs_a, 1)] = 10.0  # avg = 5.0  (differs from action 0)

    m = sample_to_stormvogel(
        initial_states,
        transition_counts,
        transition_samples,
        reward_sums,
        no_actions=2,
        no_samples=4,
    )
    assert m.model_type == model.ModelType.MDP
    # obs_a + two proxy states (one per action) + obs_b → at least 4 states
    assert m.nr_states >= 4


# ── sample_gym_to_stormvogel ─────────────────────────────────────────


def test_sample_gym_to_stormvogel_produces_mdp():
    env = _frozen_lake()
    m = sample_gym_to_stormvogel(env, no_samples=5, sample_length=10)
    assert m is not None
    assert m.model_type == model.ModelType.MDP
    env.close()
