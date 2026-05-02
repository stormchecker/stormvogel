"""Tests for stormvogel.transformations.belief_mdp."""

import warnings
from fractions import Fraction

import pytest

import stormvogel.model as sv_model
from stormvogel.model.model import ModelType
from stormvogel.transformations.belief_mdp import Belief, FrontierBelief, belief_mdp


# ---------------------------------------------------------------------------
# Shared POMDP fixture
# ---------------------------------------------------------------------------
#
# Two-state POMDP:
#
#   s0 (obs "same") --a--> s0 (prob 1/2), s1 (prob 1/2)
#                   --b--> s1 (prob 1)
#   s1 (obs "same") --a--> s0 (prob 1/2), s1 (prob 1/2)
#                   --b--> s0 (prob 1)
#
# Both states share observation "same", so the belief is the only
# distinguishing information.  A point belief {s0:1} or {s1:1} can be used
# as the initial belief.


def _make_pomdp():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("same")
    s0 = pomdp.new_state(["init", "s0"], observation=obs)
    s1 = pomdp.new_state(["s1"], observation=obs)
    act_a = pomdp.new_action("a")
    act_b = pomdp.new_action("b")
    pomdp.set_choices(
        s0, {act_a: [(Fraction(1, 2), s0), (Fraction(1, 2), s1)], act_b: [(1, s1)]}
    )
    pomdp.set_choices(
        s1, {act_a: [(Fraction(1, 2), s0), (Fraction(1, 2), s1)], act_b: [(1, s0)]}
    )
    return pomdp, s0, s1


# ---------------------------------------------------------------------------
# Belief class unit tests
# ---------------------------------------------------------------------------


def test_belief_hash_stable():
    pomdp, s0, s1 = _make_pomdp()
    b1 = Belief({s0: Fraction(1, 3), s1: Fraction(2, 3)})
    b2 = Belief({s0: Fraction(1, 3), s1: Fraction(2, 3)})
    assert hash(b1) == hash(b2)
    assert b1 == b2


def test_belief_drops_zeros():
    pomdp, s0, s1 = _make_pomdp()
    b = Belief({s0: Fraction(1), s1: Fraction(0)})
    assert s1 not in b.dist


def test_frontier_differs_from_normal():
    pomdp, s0, s1 = _make_pomdp()
    b = Belief({s0: Fraction(1)})
    f = FrontierBelief({s0: Fraction(1)})
    assert b != f
    assert hash(b) != hash(f)


def test_frontier_equality():
    pomdp, s0, s1 = _make_pomdp()
    f1 = FrontierBelief({s0: Fraction(1)})
    f2 = FrontierBelief({s0: Fraction(1)})
    assert f1 == f2
    assert hash(f1) == hash(f2)


# ---------------------------------------------------------------------------
# belief_mdp structural tests
# ---------------------------------------------------------------------------


def test_result_is_mdp():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2))
    assert mdp.model_type == ModelType.MDP


def test_has_init_label():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2))
    assert "init" in mdp.state_labels
    assert len(mdp.state_labels["init"]) == 1


def test_target_and_sink_labels_present():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=1
    )
    assert "target" in mdp.state_labels
    assert "sink" in mdp.state_labels


def test_frontier_label_present_when_budget_exceeded():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=1
    )
    assert "frontier" in mdp.state_labels


def test_no_frontier_when_budget_large():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=1000
    )
    assert "frontier" not in mdp.state_labels


def test_more_budget_gives_more_belief_states():
    pomdp, s0, _ = _make_pomdp()
    small = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=2
    )
    large = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=10
    )

    def _belief_count(mdp):
        terminals = mdp.state_labels.get("target", set()) | mdp.state_labels.get(
            "sink", set()
        )
        return mdp.nr_states - len(terminals)

    assert _belief_count(large) >= _belief_count(small)


# ---------------------------------------------------------------------------
# Cutoff semantics
# ---------------------------------------------------------------------------


def test_cutoff_one_all_frontier_go_to_target():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(pomdp, {s0: Fraction(1)}, cutoff_value=1, max_states=1)
    for frontier_state in mdp.state_labels.get("frontier", set()):
        succs = set()
        for _, branch in mdp.transitions[frontier_state]:
            for _, tgt in branch:
                succs |= set(tgt.labels)
        assert "target" in succs
        assert "sink" not in succs


def test_cutoff_zero_all_frontier_go_to_sink():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(pomdp, {s0: Fraction(1)}, cutoff_value=0, max_states=1)
    for frontier_state in mdp.state_labels.get("frontier", set()):
        succs = set()
        for _, branch in mdp.transitions[frontier_state]:
            for _, tgt in branch:
                succs |= set(tgt.labels)
        assert "sink" in succs
        assert "target" not in succs


def test_cutoff_half_frontier_transitions_sum_to_one():
    pomdp, s0, _ = _make_pomdp()
    mdp = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=1
    )
    for frontier_state in mdp.state_labels.get("frontier", set()):
        for _, branch in mdp.transitions[frontier_state]:
            total = sum(p for p, _ in branch)
            assert total == 1


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def test_rewards_propagated():
    pomdp, s0, s1 = _make_pomdp()
    rm = pomdp.new_reward_model("steps")
    rm.rewards[s0] = 2
    rm.rewards[s1] = 4

    mdp = belief_mdp(
        pomdp, {s0: Fraction(1, 2), s1: Fraction(1, 2)}, cutoff_value=Fraction(1, 2)
    )
    assert len(mdp.rewards) == 1
    # Initial belief {s0: 1/2, s1: 1/2} → expected reward = 1/2*2 + 1/2*4 = 3
    init_state = next(iter(mdp.state_labels["init"]))
    assert mdp.rewards[0].rewards[init_state] == 3


def test_frontier_reward_is_zero():
    pomdp, s0, s1 = _make_pomdp()
    rm = pomdp.new_reward_model("r")
    rm.rewards[s0] = 1
    rm.rewards[s1] = 1

    mdp = belief_mdp(
        pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(1, 2), max_states=1
    )
    for frontier_state in mdp.state_labels.get("frontier", set()):
        assert mdp.rewards[0].rewards.get(frontier_state, 0) == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_raises_for_non_pomdp():
    dtmc = sv_model.new_dtmc()
    with pytest.raises(ValueError, match="POMDP"):
        belief_mdp(dtmc, {}, cutoff_value=Fraction(1, 2))


def test_raises_for_bad_cutoff():
    pomdp, s0, _ = _make_pomdp()
    with pytest.raises(ValueError, match="cutoff_value"):
        belief_mdp(pomdp, {s0: Fraction(1)}, cutoff_value=Fraction(3, 2))


def test_raises_for_unnormalised_belief():
    pomdp, s0, s1 = _make_pomdp()
    with pytest.raises(ValueError, match="sum to 1"):
        belief_mdp(
            pomdp, {s0: Fraction(1, 3), s1: Fraction(1, 3)}, cutoff_value=Fraction(1, 2)
        )


def test_raises_for_stochastic_observation():
    from stormvogel.model.distribution import Distribution

    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_a = pomdp.new_observation("a")
    obs_b = pomdp.new_observation("b")
    s = pomdp.new_state(["init"], observation=Distribution({obs_a: 0.5, obs_b: 0.5}))
    pomdp.set_choices(s, [(1, s)])
    with pytest.raises(ValueError, match="stochastic"):
        belief_mdp(pomdp, {s: Fraction(1)}, cutoff_value=Fraction(1, 2))


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------


def test_label_propagated_when_all_support_agree():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("o")
    s0 = pomdp.new_state(["init", "goal"], observation=obs)
    s1 = pomdp.new_state(["goal"], observation=obs)
    act = pomdp.new_action("a")
    pomdp.set_choices(s0, {act: [(Fraction(1, 2), s0), (Fraction(1, 2), s1)]})
    pomdp.set_choices(s1, {act: [(1, s1)]})

    mdp = belief_mdp(
        pomdp, {s0: Fraction(1, 2), s1: Fraction(1, 2)}, cutoff_value=Fraction(1, 2)
    )
    init_state = next(iter(mdp.state_labels["init"]))
    assert "goal" in list(init_state.labels)


def test_label_not_propagated_when_support_disagrees():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("o")
    s0 = pomdp.new_state(["init", "goal"], observation=obs)
    s1 = pomdp.new_state([], observation=obs)  # no "goal" label
    act = pomdp.new_action("a")
    pomdp.set_choices(s0, {act: [(Fraction(1, 2), s0), (Fraction(1, 2), s1)]})
    pomdp.set_choices(s1, {act: [(1, s1)]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mdp = belief_mdp(
            pomdp, {s0: Fraction(1, 2), s1: Fraction(1, 2)}, cutoff_value=Fraction(1, 2)
        )
    assert any("goal" in str(w.message) for w in caught)
    init_state = next(iter(mdp.state_labels["init"]))
    assert "goal" not in list(init_state.labels)
