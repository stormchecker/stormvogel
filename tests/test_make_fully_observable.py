"""Tests for Model.make_fully_observable."""

import pytest

import stormvogel.model as sv_model
from stormvogel.model.model import ModelType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pomdp():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_a = pomdp.new_observation("a")
    obs_b = pomdp.new_observation("b")
    s0 = pomdp.new_state(["init"], friendly_name="s0", observation=obs_a)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_b)
    pomdp.set_choices(s0, [(1, s1)])
    pomdp.set_choices(s1, [(1, s1)])
    return pomdp


def _make_hmm():
    hmm = sv_model.Model(ModelType.HMM, create_initial_state=False)
    obs = hmm.new_observation("x")
    s0 = hmm.new_state(["init"], friendly_name="s0", observation=obs)
    hmm.set_choices(s0, [(1, s0)])
    return hmm


# ---------------------------------------------------------------------------
# POMDP → MDP
# ---------------------------------------------------------------------------


def test_pomdp_becomes_mdp():
    assert _make_pomdp().make_fully_observable().model_type == ModelType.MDP


def test_pomdp_returns_self():
    pomdp = _make_pomdp()
    assert pomdp.make_fully_observable() is pomdp


def test_pomdp_observations_cleared():
    mdp = _make_pomdp().make_fully_observable()
    assert len(mdp.observation_aliases) == 0
    assert len(mdp.observation_valuations) == 0
    assert len(mdp.state_observations) == 0


def test_pomdp_states_preserved():
    pomdp = _make_pomdp()
    n = pomdp.nr_states
    assert pomdp.make_fully_observable().nr_states == n


def test_pomdp_labels_preserved():
    mdp = _make_pomdp().make_fully_observable()
    assert "init" in mdp.state_labels


def test_pomdp_transitions_preserved():
    pomdp = _make_pomdp()
    n_transitions = sum(
        len(list(branch))
        for choices in pomdp.transitions.values()
        for _, branch in choices
    )
    mdp = pomdp.make_fully_observable()
    n_after = sum(
        len(list(branch))
        for choices in mdp.transitions.values()
        for _, branch in choices
    )
    assert n_after == n_transitions


# ---------------------------------------------------------------------------
# HMM → DTMC
# ---------------------------------------------------------------------------


def test_hmm_becomes_dtmc():
    assert _make_hmm().make_fully_observable().model_type == ModelType.DTMC


def test_hmm_returns_self():
    hmm = _make_hmm()
    assert hmm.make_fully_observable() is hmm


def test_hmm_observations_cleared():
    dtmc = _make_hmm().make_fully_observable()
    assert len(dtmc.observation_aliases) == 0
    assert len(dtmc.state_observations) == 0


# ---------------------------------------------------------------------------
# Cheese maze POMDP (non-trivial example)
# ---------------------------------------------------------------------------


def test_cheese_maze_becomes_mdp():
    from stormvogel.examples.cheese_maze import create_cheese_maze

    pomdp = create_cheese_maze()
    mdp = pomdp.make_fully_observable()
    assert mdp.model_type == ModelType.MDP
    assert len(mdp.state_observations) == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_raises_for_dtmc():
    with pytest.raises(ValueError, match="POMDP or HMM"):
        sv_model.new_dtmc().make_fully_observable()


def test_raises_for_mdp():
    with pytest.raises(ValueError, match="POMDP or HMM"):
        sv_model.new_mdp().make_fully_observable()
