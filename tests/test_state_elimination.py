"""Tests for state elimination on the Knuth–Yao pMC.

For each (target-set, x-value) pair, checks that solve_reachability evaluated
at the given x matches stormpy model checking of the instantiated DTMC at the
same valuation.
"""

from fractions import Fraction

import pytest
import sympy as sp

import stormvogel.stormpy_utils.model_checking as sv_mc
from stormvogel.examples.knuth_yao_pmc import create_knuth_yao_pmc
from stormvogel.parametric.state_elimination import solve_reachability

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
