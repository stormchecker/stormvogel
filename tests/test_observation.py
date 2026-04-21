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
