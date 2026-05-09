"""Tests for stormvogel.teaching.pomdp belief tracking."""

import pytest
from fractions import Fraction

from stormvogel.examples.four_state_reachability import create_4state_reachability_pomdp
from stormvogel.teaching.pomdp import belief_trace, belief_update, initial_belief


@pytest.fixture(scope="module")
def model():
    return create_4state_reachability_pomdp()


def _name(model, s):
    return model.friendly_names.get(s)


# ---------------------------------------------------------------------------
# initial_belief
# ---------------------------------------------------------------------------


def test_initial_belief_sums_to_one(model):
    b = initial_belief(model, "z")
    assert sum(b.values()) == Fraction(1)


def test_initial_belief_support_is_z_states(model):
    b = initial_belief(model, "z")
    names = {_name(model, s) for s in b}
    assert names == {"s1", "s2"}


def test_initial_belief_uniform(model):
    b = initial_belief(model, "z")
    assert all(v == Fraction(1, 2) for v in b.values())


def test_initial_belief_unknown_obs_raises(model):
    with pytest.raises((ValueError, KeyError)):
        initial_belief(model, "no_such_obs")


def test_initial_belief_unreachable_obs_raises(model):
    # z_sink is unreachable from the initial EmptyAction transition
    with pytest.raises(ValueError):
        initial_belief(model, "z_sink")


# ---------------------------------------------------------------------------
# belief_update
# ---------------------------------------------------------------------------


def test_belief_update_after_b_z(model):
    """After action b from uniform, Pr(s1) = 11/20."""
    b0 = initial_belief(model, "z")
    b1 = belief_update(model, b0, "b", "z")
    s1 = next(s for s in b1 if _name(model, s) == "s1")
    s2 = next(s for s in b1 if _name(model, s) == "s2")
    assert b1[s1] == Fraction(11, 20)
    assert b1[s2] == Fraction(9, 20)


def test_belief_update_sums_to_one(model):
    b0 = initial_belief(model, "z")
    b1 = belief_update(model, b0, "b", "z")
    assert sum(b1.values()) == Fraction(1)


def test_belief_update_target_absorbing(model):
    """After action a from uniform, observing z_target gives point mass on target."""
    b0 = initial_belief(model, "z")
    b1 = belief_update(model, b0, "a", "z_target")
    assert len(b1) == 1
    assert _name(model, next(iter(b1))) == "target"
    assert next(iter(b1.values())) == Fraction(1)


def test_belief_update_unreachable_obs_raises(model):
    b0 = initial_belief(model, "z")
    with pytest.raises(ValueError, match="unreachable"):
        belief_update(model, b0, "a", "z")


# ---------------------------------------------------------------------------
# belief_trace
# ---------------------------------------------------------------------------


def test_belief_trace_length(model):
    b0 = initial_belief(model, "z")
    trace = [("b", "z"), ("b", "z"), ("a", "z_target")]
    beliefs = belief_trace(model, b0, trace)
    assert len(beliefs) == 4  # b0 + one per step


def test_belief_trace_first_is_initial(model):
    b0 = initial_belief(model, "z")
    beliefs = belief_trace(model, b0, [("b", "z")])
    assert beliefs[0] == b0


def test_belief_trace_empty_returns_singleton(model):
    b0 = initial_belief(model, "z")
    beliefs = belief_trace(model, b0, [])
    assert beliefs == [b0]


def test_belief_trace_two_steps(model):
    """Two b-steps from uniform: Pr(s1) after step 2."""
    b0 = initial_belief(model, "z")
    beliefs = belief_trace(model, b0, [("b", "z"), ("b", "z")])
    b2 = beliefs[2]
    # After two b-steps: apply the transition matrix twice to [1/2, 1/2].
    # Step 1: s1 = 11/20, s2 = 9/20
    # Step 2: s1 = 0.8*(11/20) + 0.3*(9/20) = 88/200 + 27/200 = 115/200 = 23/40
    s1 = next(s for s in b2 if _name(model, s) == "s1")
    assert b2[s1] == Fraction(23, 40)
