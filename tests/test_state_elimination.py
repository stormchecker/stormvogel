"""Tests for state elimination on the Knuth–Yao pMC.

For each (target-set, x-value) pair, checks that solve_reachability evaluated
at the given x matches stormpy model checking of the instantiated DTMC at the
same valuation.
"""

from fractions import Fraction

import pytest
import sympy as sp

import stormvogel.model as sv_model
import stormvogel.stormpy_utils.model_checking as sv_mc
from stormvogel.examples.knuth_yao_pmc import create_knuth_yao_pmc
from stormvogel.teaching.parametric import (
    eliminate_selfloop,
    eliminate_state,
    eliminate_transition,
    solve_reachability,
)

stormpy = pytest.importorskip("stormpy")

_X = sp.Symbol("x")

PARAM_VALUES = [Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)]

# Target sets: single faces and multi-face unions, expressed as label lists.
TARGET_SETS = [
    (["rolled1"], '"rolled1"'),
    (["rolled6"], '"rolled6"'),
    (["rolled1", "rolled2", "rolled3"], '"rolled1" | "rolled2" | "rolled3"'),
    (["rolled2", "rolled4", "rolled6"], '"rolled2" | "rolled4" | "rolled6"'),
]


@pytest.fixture(scope="module")
def pmc():
    return create_knuth_yao_pmc()


@pytest.fixture(scope="module", params=PARAM_VALUES, ids=[str(v) for v in PARAM_VALUES])
def concrete(request, pmc):
    return request.param, pmc.get_instantiated_model({"x": request.param})


@pytest.mark.parametrize("labels,formula_labels", TARGET_SETS)
def test_state_elimination_vs_stormpy(pmc, concrete, labels, formula_labels):
    param_val, concrete_model = concrete

    # solution function via state elimination on the pMC
    target = [s for label in labels for s in pmc.get_states_with_label(label)]
    solution_fn = solve_reachability(pmc, target)
    sv_val = float(solution_fn.subs(_X, sp.Rational(*param_val.as_integer_ratio())))

    # ground truth via stormpy on the instantiated DTMC
    result = sv_mc.model_checking(concrete_model, f"P=? [F ({formula_labels})]")
    assert result is not None
    stormpy_val = float(result.values[concrete_model.initial_state])

    assert pytest.approx(sv_val, rel=1e-6) == stormpy_val


# ---------------------------------------------------------------------------
# Parametric model checking via stormvogel vs state elimination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels,formula_labels", TARGET_SETS)
def test_stormpy_parametric_vs_state_elimination(pmc, labels, formula_labels):
    """stormpy parametric model checking returns a sympy expression that is
    symbolically equal to the state-elimination solution function."""
    prop = f"P=? [F ({formula_labels})]"
    result = sv_mc.model_checking(pmc, prop, scheduler=False)
    assert result is not None

    stormpy_fn = result.values[pmc.initial_state]

    target = [s for label in labels for s in pmc.get_states_with_label(label)]
    elimination_fn = solve_reachability(pmc, target)

    assert sp.cancel(stormpy_fn - elimination_fn) == 0


# ---------------------------------------------------------------------------
# Model-type guard
# ---------------------------------------------------------------------------


def _make_dtmc_pair():
    """Two-state DTMC: s0 -p-> s1, s0 -(1-p)-> s0 (selfloop); s1 absorbing."""
    dtmc = sv_model.new_dtmc(create_initial_state=False)
    p = dtmc.declare_parameter("p")
    s0 = dtmc.new_state(["init"])
    s1 = dtmc.new_state(["target"])
    dtmc.set_choices(s0, [(p, s1), (1 - p, s0)])
    dtmc.set_choices(s1, [(1, s1)])
    return dtmc, s0, s1


def test_eliminate_selfloop_raises_for_mdp():
    mdp = sv_model.new_mdp()
    with pytest.raises(ValueError, match="DTMC"):
        eliminate_selfloop(mdp, mdp.initial_state)


def test_eliminate_transition_raises_for_mdp():
    mdp = sv_model.new_mdp()
    with pytest.raises(ValueError, match="DTMC"):
        eliminate_transition(mdp, mdp.initial_state, mdp.initial_state)


def test_eliminate_state_raises_for_mdp():
    mdp = sv_model.new_mdp()
    with pytest.raises(ValueError, match="DTMC"):
        eliminate_state(mdp, mdp.initial_state)


# ---------------------------------------------------------------------------
# remove=True flag
# ---------------------------------------------------------------------------


def test_eliminate_state_remove_deletes_state():
    dtmc, s0, s1 = _make_dtmc_pair()
    eliminate_selfloop(dtmc, s0)
    eliminate_state(dtmc, s0, remove=True)
    assert s0 not in dtmc.states


def test_eliminate_state_remove_false_keeps_state():
    dtmc, s0, s1 = _make_dtmc_pair()
    eliminate_selfloop(dtmc, s0)
    eliminate_state(dtmc, s0, remove=False)
    assert s0 in dtmc.states


def test_eliminate_state_remove_leaves_no_predecessors():
    dtmc, s0, s1 = _make_dtmc_pair()
    eliminate_selfloop(dtmc, s0)
    # After removal, s0 is no longer a predecessor of s1 (s1 keeps its self-loop).
    eliminate_state(dtmc, s0, remove=True)
    assert s0 not in dtmc.predecessors(s1)
