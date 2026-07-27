"""Tests for stormvogel.transformations.pomdp_to_hmm."""

from fractions import Fraction

import pytest

import stormvogel.model as sv_model
from stormvogel.model.model import ModelType
from stormvogel.transformations.pomdp_to_hmm import pomdp_to_hmm


def _pomdp():
    """Two-action POMDP: from s0, ``a`` goes to s1, ``b`` splits s1/s2."""
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    quiet = pomdp.new_observation("quiet")
    loud = pomdp.new_observation("loud")
    s0 = pomdp.new_state(["init"], friendly_name="s0", observation=quiet)
    s1 = pomdp.new_state(["goal"], friendly_name="s1", observation=loud)
    s2 = pomdp.new_state(friendly_name="s2", observation=quiet)
    act_a = pomdp.new_action("a")
    act_b = pomdp.new_action("b")
    pomdp.set_choices(
        s0,
        {
            act_a: [(Fraction(1), s1)],
            act_b: [(Fraction(1, 2), s1), (Fraction(1, 2), s2)],
        },
    )
    pomdp.set_choices(s1, [(1, s1)])
    pomdp.set_choices(s2, [(1, s2)])
    return pomdp, s0, s1, s2


def _successors(model, state):
    """Return {target friendly name: probability} for a single-choice state."""
    _action, branch = next(iter(model.transitions[state]))
    return {model.friendly_names.get(t): p for p, t in branch}


def test_result_is_hmm():
    pomdp, *_ = _pomdp()
    assert pomdp_to_hmm(pomdp).model_type == ModelType.HMM


def test_actions_are_averaged():
    """s0 has two actions, so each contributes half its distribution."""
    pomdp, s0, _s1, _s2 = _pomdp()
    hmm = pomdp_to_hmm(pomdp)
    new_s0 = next(s for s in hmm.states if hmm.friendly_names.get(s) == "s0")
    # a: s1 with 1.  b: s1 with 1/2, s2 with 1/2.  Averaged: s1 3/4, s2 1/4.
    assert _successors(hmm, new_s0) == {"s1": Fraction(3, 4), "s2": Fraction(1, 4)}


def test_every_state_has_one_choice():
    pomdp, *_ = _pomdp()
    hmm = pomdp_to_hmm(pomdp)
    for state in hmm.states:
        assert len(list(hmm.transitions[state])) == 1


def test_distributions_still_sum_to_one():
    pomdp, *_ = _pomdp()
    hmm = pomdp_to_hmm(pomdp)
    for state in hmm.states:
        assert sum(_successors(hmm, state).values()) == Fraction(1)


def test_labels_and_observations_are_preserved():
    pomdp, *_ = _pomdp()
    hmm = pomdp_to_hmm(pomdp)
    assert hmm.nr_states == pomdp.nr_states
    assert set(hmm.state_labels) >= {"init", "goal"}
    aliases = {
        hmm.state_observations[s].alias
        for s in hmm.states  # type: ignore[union-attr]
    }
    assert aliases == {"quiet", "loud"}


def test_friendly_names_are_preserved():
    pomdp, *_ = _pomdp()
    hmm = pomdp_to_hmm(pomdp)
    assert {hmm.friendly_names.get(s) for s in hmm.states} == {"s0", "s1", "s2"}


def test_rewards_are_preserved():
    pomdp, s0, s1, _s2 = _pomdp()
    reward_model = pomdp.new_reward_model("steps")
    reward_model.rewards[s0] = 2
    reward_model.rewards[s1] = 5

    hmm = pomdp_to_hmm(pomdp)
    assert len(hmm.rewards) == 1
    by_name = {hmm.friendly_names.get(s): v for s, v in hmm.rewards[0].rewards.items()}
    assert by_name["s0"] == 2
    assert by_name["s1"] == 5


def test_source_model_is_untouched():
    pomdp, s0, *_ = _pomdp()
    before = len(list(pomdp.transitions[s0]))
    pomdp_to_hmm(pomdp)
    assert len(list(pomdp.transitions[s0])) == before
    assert pomdp.model_type == ModelType.POMDP


def test_raises_for_non_pomdp():
    with pytest.raises(ValueError, match="POMDP"):
        pomdp_to_hmm(sv_model.new_dtmc())


def test_raises_for_stochastic_observation():
    from stormvogel.model.distribution import Distribution

    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_a = pomdp.new_observation("a")
    obs_b = pomdp.new_observation("b")
    s = pomdp.new_state(["init"], observation=Distribution({obs_a: 0.5, obs_b: 0.5}))
    pomdp.set_choices(s, [(1, s)])
    with pytest.raises(ValueError, match="stochastic"):
        pomdp_to_hmm(pomdp)


def test_belief_search_runs_on_the_result():
    """The whole point: the HMM can be searched without an action choice."""
    from stormvogel.teaching.belief_search import dijkstra_belief_search

    pomdp, *_ = _pomdp()
    hmm = pomdp_to_hmm(pomdp)
    initial = next(iter(hmm.state_labels["init"]))
    result = dijkstra_belief_search(hmm, {initial: Fraction(1)}, "goal", Fraction(1))
    assert result.found
    # One step, observation "loud", probability 3/4 -- no action to choose.
    assert result.probability == Fraction(3, 4)
    assert result.actions == [""]
