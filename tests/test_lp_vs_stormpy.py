"""Cross-validate LP solutions against stormpy model-checking results.

All tests require stormpy and are skipped otherwise.
Stormpy is treated as ground truth; LP solutions must match to within 1e-6.
"""

import fractions

import pytest
import sympy as sp
import stormvogel
import stormvogel.model
from stormvogel.teaching.lp import (
    lp_dual_maxreachprob,
    lp_maxreachprob,
    lp_minreachprob,
    solve_lp,
)

pytestmark = pytest.mark.skipif(
    pytest.importorskip("stormpy", reason="stormpy not installed") is None,
    reason="stormpy not installed",
)


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# MDPs
# ---------------------------------------------------------------------------


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
    """s0 --a--> s1, s0 --b--> s2; s1 --a--> {0.9: T, 0.1: sink}; s2 --a--> {0.5: T, 0.5: sink}.
    Pmax = 0.9 (via s1), Pmin = 0.5 (via s2).
    """
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


# ---------------------------------------------------------------------------
# Dual LP vs stormpy Pmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mdp_factory",
    [
        _no_choice_mdp,
        _choice_mdp,
        _prob_choice_mdp,
        _two_step_mdp,
    ],
)
def test_dual_lp_pmax(mdp_factory):
    pytest.importorskip("stormpy")
    mdp = mdp_factory()
    lp = lp_dual_maxreachprob(mdp, "T")
    sol = solve_lp(lp)
    assert sol is not None
    assert float(sol.objective) == pytest.approx(_pmax(mdp, "T"), abs=1e-6)


# ---------------------------------------------------------------------------
# Primal LP vs stormpy Pmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mdp_factory",
    [
        _no_choice_mdp,
        _choice_mdp,
        _prob_choice_mdp,
        _two_step_mdp,
    ],
)
def test_primal_lp_pmax(mdp_factory):
    pytest.importorskip("stormpy")
    mdp = mdp_factory()
    lp = lp_maxreachprob(mdp, "T")
    sol = solve_lp(lp)
    assert sol is not None
    assert float(_x_init(sol, mdp)) == pytest.approx(_pmax(mdp, "T"), abs=1e-6)


# ---------------------------------------------------------------------------
# Primal LP vs stormpy Pmin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mdp_factory",
    [
        _no_choice_mdp,
        _choice_mdp,
        _prob_choice_mdp,
        _two_step_mdp,
    ],
)
def test_primal_lp_pmin(mdp_factory):
    pytest.importorskip("stormpy")
    mdp = mdp_factory()
    lp = lp_minreachprob(mdp, "T")
    sol = solve_lp(lp)
    assert sol is not None
    assert float(_x_init(sol, mdp)) == pytest.approx(_pmin(mdp, "T"), abs=1e-6)
