"""Tests for stormvogel.teaching.lp.

Sections:
- Pure LP tests (no model needed)
- LP built from a stormvogel MDP
- Cross-validation against stormpy (skipped if stormpy not installed)
"""

import fractions

import pytest
import sympy as sp
import stormvogel
import stormvogel.model
from stormvogel.teaching.lp import (
    LP,
    LPSolution,
    lp_dual_maxreachprob,
    lp_dual_prob,
    lp_maxreachprob,
    lp_minreachprob,
    lp_prob,
    solve_lp,
)
from stormvogel.teaching.multiobjective import goal_unfolding, lp_dual_multireachprob


# ---------------------------------------------------------------------------
# Pure LP tests (no model needed)
# ---------------------------------------------------------------------------


def test_solve_returns_lpsolution():
    x = sp.Symbol("x")
    lp = LP("maximize", x, [sp.Le(x, sp.Integer(1)), sp.Ge(x, sp.Integer(0))])
    sol = solve_lp(lp)
    assert isinstance(sol, LPSolution)


def test_simple_maximize():
    x = sp.Symbol("x")
    lp = LP("maximize", x, [sp.Le(x, sp.Integer(1)), sp.Ge(x, sp.Integer(0))])
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(1)
    assert sol.values[x] == fractions.Fraction(1)


def test_simple_minimize():
    x = sp.Symbol("x")
    lp = LP(
        "minimize",
        x,
        [sp.Ge(x, sp.Rational(1, 3)), sp.Ge(x, sp.Integer(0))],
    )
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(1, 3)


def test_infeasible_returns_none():
    x = sp.Symbol("x")
    lp = LP("maximize", x, [sp.Ge(x, sp.Integer(2)), sp.Le(x, sp.Integer(1))])
    assert solve_lp(lp) is None


def test_equality_constraint():
    x, y = sp.symbols("x y")
    lp = LP(
        "maximize",
        x + y,
        [sp.Eq(x + y, sp.Integer(3)), sp.Ge(x, sp.Integer(0)), sp.Ge(y, sp.Integer(0))],
    )
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(3)


def test_two_variable_maximize():
    x, y = sp.symbols("x y")
    lp = LP(
        "maximize",
        x + y,
        [
            sp.Le(x + y, sp.Integer(1)),
            sp.Ge(x, sp.Integer(0)),
            sp.Ge(y, sp.Integer(0)),
        ],
    )
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(1)


def test_exact_rational_coefficients():
    """Coefficients 3/7 produce an exact rational optimal."""
    x = sp.Symbol("x")
    lp = LP(
        "maximize",
        x,
        [sp.Le(x, sp.Rational(3, 7)), sp.Ge(x, sp.Integer(0))],
    )
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(3, 7)


# ---------------------------------------------------------------------------
# LP built from a stormvogel MDP
# ---------------------------------------------------------------------------


def _lp_chain_mdp():
    """s0 --a--> {0.7: s1(T), 0.3: s2(absorbing)}."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T"], friendly_name="s1")
    s2 = mdp.new_state(friendly_name="s2")
    a = mdp.action("a")
    s0.set_choices({a: [(0.7, s1), (0.3, s2)]})
    return mdp, s0, s1, s2


def _lp_choice_mdp():
    """s0 --a--> s1(T1), s0 --b--> s2(T2)."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T1"], friendly_name="s1")
    s2 = mdp.new_state(labels=["T2"], friendly_name="s2")
    a, b = mdp.action("a"), mdp.action("b")
    s0.set_choices({a: [(1.0, s1)], b: [(1.0, s2)]})
    return mdp, s0, s1, s2


def test_lp_prob_chain_max_reachability():
    """lp_prob on chain MDP: max P(reach T) = 0.7."""
    mdp, _, s1, s2 = _lp_chain_mdp()
    lp = lp_prob(mdp, min=False, zero_states=[s2], one_states=[s1])
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(7, 10)


def test_lp_dual_prob_chain():
    """Dual LP on chain MDP: max P(reach T) = 0.7 (by LP duality)."""
    mdp, _, s1, s2 = _lp_chain_mdp()
    lp = lp_dual_prob(mdp, zero_states=[s2], one_states=[s1])
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(7, 10)


def test_lp_dual_multireachprob_choice_mdp_weight_a():
    """Dual multi-reachability: weight [1, 0] maximises P(reach T1) = 1."""
    mdp, *_ = _lp_choice_mdp()
    unfolded, bits_map = goal_unfolding(mdp, ["T1", "T2"], return_state_bits=True)
    lp = lp_dual_multireachprob(unfolded, bits_map, ["T1", "T2"], weights=[1.0, 0.0])
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(1)


def test_lp_dual_multireachprob_choice_mdp_weight_b():
    """Dual multi-reachability: weight [0, 1] maximises P(reach T2) = 1."""
    mdp, *_ = _lp_choice_mdp()
    unfolded, bits_map = goal_unfolding(mdp, ["T1", "T2"], return_state_bits=True)
    lp = lp_dual_multireachprob(unfolded, bits_map, ["T1", "T2"], weights=[0.0, 1.0])
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(1)


def test_lp_dual_multireachprob_equal_weights():
    """Equal weights on choice MDP: optimal is 0.5 (split evenly)."""
    mdp, *_ = _lp_choice_mdp()
    unfolded, bits_map = goal_unfolding(mdp, ["T1", "T2"], return_state_bits=True)
    lp = lp_dual_multireachprob(unfolded, bits_map, ["T1", "T2"], weights=[0.5, 0.5])
    sol = solve_lp(lp)
    assert sol is not None
    assert sol.objective == fractions.Fraction(1, 2)


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------


def test_lp_dual_multireachprob_threshold_infeasible():
    """Threshold [0.6, 0.6] is infeasible on the choice MDP.

    Both thresholds require P(T1) + P(T2) >= 1.2, but the flow constraint
    forces P(T1) + P(T2) = 1.
    """
    mdp, *_ = _lp_choice_mdp()
    unfolded, bits_map = goal_unfolding(mdp, ["T1", "T2"], return_state_bits=True)
    lp = lp_dual_multireachprob(
        unfolded,
        bits_map,
        ["T1", "T2"],
        weights=[0.5, 0.5],
        threshold=[0.6, 0.6],
    )
    assert solve_lp(lp) is None


def test_lp_dual_multireachprob_threshold_nonbinding():
    """A threshold strictly below the unconstrained optimum does not change the objective.

    On the chain MDP with one target, Pmax = 0.7. A threshold of 0.5 is
    achievable and leaves the optimal value unchanged.
    """
    mdp, _, s1, s2 = _lp_chain_mdp()
    lp_no_thr = lp_dual_prob(mdp, zero_states=[s2], one_states=[s1])
    unfolded, bits_map = goal_unfolding(mdp, ["T"], return_state_bits=True)
    lp_thr = lp_dual_multireachprob(
        unfolded,
        bits_map,
        ["T"],
        weights=[1.0],
        threshold=[0.5],
    )
    sol_no_thr = solve_lp(lp_no_thr)
    sol_thr = solve_lp(lp_thr)
    assert sol_no_thr is not None
    assert sol_thr is not None
    assert sol_thr.objective == sol_no_thr.objective


# ---------------------------------------------------------------------------
# Cross-validation against stormpy (skipped if stormpy not installed)
# ---------------------------------------------------------------------------


def _pmax(mdp: stormvogel.model.Model, label: str) -> float:
    result = stormvogel.model_checking(mdp, f'Pmax=? [F "{label}"]', scheduler=False)
    assert result is not None
    val = result.at(mdp.initial_state)
    return float(val)  # type: ignore[arg-type]


def _pmin(mdp: stormvogel.model.Model, label: str) -> float:
    result = stormvogel.model_checking(mdp, f'Pmin=? [F "{label}"]', scheduler=False)
    assert result is not None
    val = result.at(mdp.initial_state)
    return float(val)  # type: ignore[arg-type]


def _x_init(sol, mdp: stormvogel.model.Model) -> fractions.Fraction:
    """Return the LP variable value for the initial state."""
    sym = sp.Symbol(f"x_{mdp.initial_state.friendly_name}")
    return sol.values[sym]


def _no_choice_mdp():
    """Single action: s0 --a--> {0.7: T, 0.3: sink}. Pmax = Pmin = 0.7."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T"], friendly_name="s1")
    s2 = mdp.new_state(friendly_name="s2")
    a = mdp.action("a")
    s0.set_choices({a: [(0.7, s1), (0.3, s2)]})
    mdp.add_self_loops()
    return mdp


def _choice_mdp():
    """s0 --a--> T, s0 --b--> sink. Pmax = 1, Pmin = 0."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T"], friendly_name="s1")
    s2 = mdp.new_state(friendly_name="s2")
    a, b = mdp.action("a"), mdp.action("b")
    s0.set_choices({a: [(1.0, s1)], b: [(1.0, s2)]})
    mdp.add_self_loops()
    return mdp


def _prob_choice_mdp():
    """s0 --a--> {0.8: T, 0.2: sink}, s0 --b--> {0.4: T, 0.6: sink}. Pmax = 0.8, Pmin = 0.4."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T"], friendly_name="s1")
    s2 = mdp.new_state(friendly_name="s2")
    a, b = mdp.action("a"), mdp.action("b")
    s0.set_choices({a: [(0.8, s1), (0.2, s2)], b: [(0.4, s1), (0.6, s2)]})
    mdp.add_self_loops()
    return mdp


def _two_step_mdp():
    """s0 --a--> s1, s0 --b--> s2; Pmax = 0.9 (via s1), Pmin = 0.5 (via s2)."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(friendly_name="s1")
    s2 = mdp.new_state(friendly_name="s2")
    s3 = mdp.new_state(labels=["T"], friendly_name="s3")
    s4 = mdp.new_state(friendly_name="s4")
    a, b = mdp.action("a"), mdp.action("b")
    s0.set_choices({a: [(1.0, s1)], b: [(1.0, s2)]})
    s1.set_choices({a: [(0.9, s3), (0.1, s4)]})
    s2.set_choices({a: [(0.5, s3), (0.5, s4)]})
    mdp.add_self_loops()
    return mdp


@pytest.mark.parametrize(
    "mdp_factory",
    [_no_choice_mdp, _choice_mdp, _prob_choice_mdp, _two_step_mdp],
)
def test_dual_lp_pmax(mdp_factory):
    pytest.importorskip("stormpy")
    mdp = mdp_factory()
    lp = lp_dual_maxreachprob(mdp, "T")
    sol = solve_lp(lp)
    assert sol is not None
    assert float(sol.objective) == pytest.approx(_pmax(mdp, "T"), abs=1e-6)


@pytest.mark.parametrize(
    "mdp_factory",
    [_no_choice_mdp, _choice_mdp, _prob_choice_mdp, _two_step_mdp],
)
def test_primal_lp_pmax(mdp_factory):
    pytest.importorskip("stormpy")
    mdp = mdp_factory()
    lp = lp_maxreachprob(mdp, "T")
    sol = solve_lp(lp)
    assert sol is not None
    assert float(_x_init(sol, mdp)) == pytest.approx(_pmax(mdp, "T"), abs=1e-6)


@pytest.mark.parametrize(
    "mdp_factory",
    [_no_choice_mdp, _choice_mdp, _prob_choice_mdp, _two_step_mdp],
)
def test_primal_lp_pmin(mdp_factory):
    pytest.importorskip("stormpy")
    mdp = mdp_factory()
    lp = lp_minreachprob(mdp, "T")
    sol = solve_lp(lp)
    assert sol is not None
    assert float(_x_init(sol, mdp)) == pytest.approx(_pmin(mdp, "T"), abs=1e-6)
