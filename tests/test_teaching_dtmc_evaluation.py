"""Tests for stormvogel.teaching.dtmc_evaluation."""

import sympy as sp

import stormvogel.model
from stormvogel.teaching.dtmc_evaluation import (
    compute_one_states,
    compute_zero_states,
    equations_reachability,
    solve_reachability,
)


def _simple_dtmc():
    """3-state DTMC: init -1/3-> A, init -2/3-> B; A and B absorbing."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(labels=["init"])
    s_a = dtmc.new_state(labels=["A"])
    s_b = dtmc.new_state(labels=["B"])
    dtmc.set_choices(s0, [(sp.Rational(1, 3), s_a), (sp.Rational(2, 3), s_b)])
    dtmc.set_choices(s_a, [(1, s_a)])
    dtmc.set_choices(s_b, [(1, s_b)])
    return dtmc, s0, s_a, s_b


def test_equations_reachability_structure():
    dtmc, s0, s_a, s_b = _simple_dtmc()
    eqs = equations_reachability(dtmc, one_states=[s_a])
    assert len(eqs) == 3
    assert all(isinstance(eq, sp.Expr) for eq in eqs)


def test_solve_reachability_simple():
    dtmc, s0, s_a, s_b = _simple_dtmc()
    # s_b cannot reach s_a, so it is auto-detected as a zero state.
    values = solve_reachability(dtmc, one_states=[s_a])
    assert values[s0] == sp.Rational(1, 3)
    assert values[s_a] == sp.Integer(1)
    assert values[s_b] == sp.Integer(0)


def test_solve_reachability_explicit_zero():
    """Explicit zero_states override is still supported."""
    dtmc, s0, s_a, s_b = _simple_dtmc()
    values = solve_reachability(dtmc, one_states=[s_a], zero_states=[s_b])
    assert values[s0] == sp.Rational(1, 3)


def test_solve_reachability_chain():
    """Linear chain s0 -> s1 -> s2 (target), all prob 1."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(labels=["init"])
    s1 = dtmc.new_state(labels=["mid"])
    s2 = dtmc.new_state(labels=["target"])
    dtmc.set_choices(s0, [(1, s1)])
    dtmc.set_choices(s1, [(1, s2)])
    dtmc.set_choices(s2, [(1, s2)])

    values = solve_reachability(dtmc, one_states=[s2])
    assert values[s0] == sp.Integer(1)
    assert values[s1] == sp.Integer(1)
    assert values[s2] == sp.Integer(1)


def test_compute_zero_states():
    dtmc, s0, s_a, s_b = _simple_dtmc()
    zeros = compute_zero_states(dtmc, one_states=[s_a])
    assert zeros == {s_b}


def test_compute_one_states_simple():
    dtmc, s0, s_a, s_b = _simple_dtmc()
    # s0 reaches s_a with prob 1/3 < 1; only s_a itself is a one_state.
    ones = compute_one_states(dtmc, target_states=[s_a])
    assert ones == {s_a}


def test_compute_one_states_chain():
    """In a deterministic chain s0->s1->s2, all states have P=1."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(labels=["init"])
    s1 = dtmc.new_state(labels=["mid"])
    s2 = dtmc.new_state(labels=["target"])
    dtmc.set_choices(s0, [(1, s1)])
    dtmc.set_choices(s1, [(1, s2)])
    dtmc.set_choices(s2, [(1, s2)])
    ones = compute_one_states(dtmc, target_states=[s2])
    assert ones == {s0, s1, s2}


def test_compute_one_states_target_not_absorbing():
    """Target state with an edge to a zero state should still be in Syes."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(labels=["init"])
    target = dtmc.new_state(labels=["target"])
    sink = dtmc.new_state(labels=["sink"])
    dtmc.set_choices(s0, [(1, target)])
    # target has an edge to sink, but target is absorbing in the Prob1 sense.
    dtmc.set_choices(target, [(sp.Rational(1, 2), target), (sp.Rational(1, 2), sink)])
    dtmc.set_choices(sink, [(1, sink)])
    ones = compute_one_states(dtmc, target_states=[target])
    # s0 reaches target with prob 1; target is in Syes by definition.
    assert target in ones
    assert s0 in ones
    assert sink not in ones


def test_solve_reachability_accepts_foreign_states():
    """one_states can come from a different model (matched by UUID)."""
    dtmc, s0, s_a, s_b = _simple_dtmc()
    dtmc2 = dtmc.copy()
    orig_s_a = s_a  # state from original model

    values = solve_reachability(dtmc2, one_states=[orig_s_a])
    copied_s0 = dtmc2.get_state_by_id(s0.state_id)
    assert values[copied_s0] == sp.Rational(1, 3)
