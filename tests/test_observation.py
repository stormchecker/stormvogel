"""Tests for stormvogel.model.observation."""

import pytest
import stormvogel.model
from stormvogel.model.variable import Variable


def test_observation_alias():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("my_obs")
    assert obs.alias == "my_obs"


def test_observation_str():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("my_obs")
    s = str(obs)
    assert "my_obs" in s
    assert "Observation(" in s


def test_observation_valuations_with_values():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("obs", valuations={Variable("x"): 5})
    assert obs.valuations[Variable("x")] == 5


def test_observation_display_with_valuations():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("obs_with_vals", valuations={Variable("x"): 1})
    result = obs.display()
    assert "obs_with_vals" in result


def test_observation_display_without_valuations_returns_only_alias():
    """display() returns just the alias when no valuations were provided."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("clean_obs")  # no valuations argument
    result = obs.display()
    assert result == "clean_obs"


def test_observation_alias_missing_raises_runtimeerror():
    """alias raises RuntimeError (not RecursionError) when missing from the model."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("obs")
    del pomdp.observation_aliases[obs]
    with pytest.raises(RuntimeError, match="does not have an alias"):
        _ = obs.alias


def test_observation_state_has_observation():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("obs")
    state = pomdp.new_state(observation=obs)
    assert state.observation is obs


# ---------------------------------------------------------------------------
# compute_states_per_observation
# ---------------------------------------------------------------------------


def _make_pomdp():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs_a = pomdp.new_observation("a")
    obs_b = pomdp.new_observation("b")
    s0 = pomdp.new_state(friendly_name="s0", observation=obs_a)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_b)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_b)
    pomdp.set_choices(s0, [(1, s1)])
    pomdp.set_choices(s1, [(1, s2)])
    pomdp.set_choices(s2, [(1, s2)])
    return pomdp, obs_a, obs_b, s0, s1, s2


def test_compute_states_per_observation_by_object():
    pomdp, obs_a, obs_b, s0, s1, s2 = _make_pomdp()
    assert pomdp.compute_states_per_observation()[obs_a] == {s0}


def test_compute_states_per_observation_by_alias():
    pomdp, _, _, s0, s1, s2 = _make_pomdp()
    assert pomdp.compute_states_per_observation()[pomdp.get_observation("a")] == {s0}


def test_compute_states_per_observation_multiple_states():
    pomdp, _, obs_b, s0, s1, s2 = _make_pomdp()
    assert pomdp.compute_states_per_observation()[obs_b] == {s1, s2}


def test_compute_states_per_observation_multiple_states_by_alias():
    pomdp, _, _, s0, s1, s2 = _make_pomdp()
    assert pomdp.compute_states_per_observation()[pomdp.get_observation("b")] == {
        s1,
        s2,
    }


def test_compute_states_per_observation_unknown_alias_raises():
    pomdp, _, _, _, _, _ = _make_pomdp()
    with pytest.raises(KeyError):
        pomdp.get_observation("no_such_obs")


def test_compute_states_per_observation_non_pomdp_raises():
    mdp = stormvogel.model.new_mdp()
    with pytest.raises(RuntimeError, match="does not support observations"):
        mdp.compute_states_per_observation()


def test_compute_states_per_observation_four_state_example():
    from stormvogel.examples.four_state_reachability import (
        create_4state_reachability_pomdp,
    )

    model = create_4state_reachability_pomdp()
    states = model.compute_states_per_observation()[model.get_observation("z")]
    names = {model.friendly_names[s] for s in states}
    assert names == {"s1", "s2"}
