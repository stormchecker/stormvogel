"""Tests for stormvogel.teaching.parametric.

Covers:
- State elimination (solve_reachability, eliminate_*)
- Parametric analysis (AnalyseParametric, rectangular_region_to_stormpy)
- Parameter space partitioning

All tests require stormpy and are skipped if not installed.
"""

from fractions import Fraction

import pytest
import sympy as sp

import stormvogel.model as sv_model
import stormvogel.stormpy_utils.model_checking as sv_mc
from stormvogel.examples.knuth_yao_pmc import (
    create_knuth_yao_pmc,
    create_knuth_yao_pmc_twocoins,
)
from stormvogel.parametric.region import AnnotatedRegion, RectangularRegion
from stormvogel.stormpy_utils.parametric_analysis import (
    AnalyseParametric,
    rectangular_region_to_stormpy,
)
from stormvogel.teaching.parametric import (
    eliminate_selfloop,
    eliminate_state,
    eliminate_transition,
    parameter_space_partitioning,
    solve_reachability,
)

stormpy = pytest.importorskip("stormpy")

_PROP = 'P=? [F "rolled1"]'
_THRESHOLD = 1 / 6
_FAIR = {"x": 0.5}
_REGION_FAIR = RectangularRegion({"x": (0.4, 0.6)})
_FAIR_XY = {"x": 0.5, "y": 0.5}
_REGION_FAIR_XY = RectangularRegion({"x": (0.4, 0.6), "y": (0.4, 0.6)})

_X = sp.Symbol("x")
PARAM_VALUES = [Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)]
TARGET_SETS = [
    (["rolled1"], '"rolled1"'),
    (["rolled6"], '"rolled6"'),
    (["rolled1", "rolled2", "rolled3"], '"rolled1" | "rolled2" | "rolled3"'),
    (["rolled2", "rolled4", "rolled6"], '"rolled2" | "rolled4" | "rolled6"'),
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pmc():
    return create_knuth_yao_pmc()


@pytest.fixture(scope="module")
def pmc_xy():
    return create_knuth_yao_pmc_twocoins()


@pytest.fixture(scope="module", params=PARAM_VALUES, ids=[str(v) for v in PARAM_VALUES])
def concrete(request, pmc):
    return request.param, pmc.get_instantiated_model({"x": request.param})


@pytest.fixture(scope="module")
def analyser(pmc):
    return AnalyseParametric(pmc, _PROP)


@pytest.fixture(scope="module")
def analyser_xy(pmc_xy):
    return AnalyseParametric(pmc_xy, _PROP)


@pytest.fixture(scope="module")
def analyser_xy_rolled2():
    return AnalyseParametric(create_knuth_yao_pmc_twocoins(), 'P=? [F "rolled2"]')


# ---------------------------------------------------------------------------
# State elimination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels,formula_labels", TARGET_SETS)
def test_state_elimination_vs_stormpy(pmc, concrete, labels, formula_labels):
    param_val, concrete_model = concrete
    target = [s for label in labels for s in pmc.get_states_with_label(label)]
    solution_fn = solve_reachability(pmc, target)
    sv_val = float(solution_fn.subs(_X, sp.Rational(*param_val.as_integer_ratio())))
    result = sv_mc.model_checking(concrete_model, f"P=? [F ({formula_labels})]")
    assert result is not None
    stormpy_val = float(result.values[concrete_model.initial_state])
    assert pytest.approx(sv_val, rel=1e-6) == stormpy_val


@pytest.mark.parametrize("labels,formula_labels", TARGET_SETS)
def test_stormpy_parametric_vs_state_elimination(pmc, labels, formula_labels):
    """stormpy parametric model checking returns a sympy expression symbolically
    equal to the state-elimination solution function."""
    prop = f"P=? [F ({formula_labels})]"
    result = sv_mc.model_checking(pmc, prop, scheduler=False)
    assert result is not None
    stormpy_fn = result.values[pmc.initial_state]
    target = [s for label in labels for s in pmc.get_states_with_label(label)]
    elimination_fn = solve_reachability(pmc, target)
    assert sp.cancel(stormpy_fn - elimination_fn) == 0


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


def _make_dtmc_pair():
    dtmc = sv_model.new_dtmc(create_initial_state=False)
    p = dtmc.declare_parameter("p")
    s0 = dtmc.new_state(["init"])
    s1 = dtmc.new_state(["target"])
    dtmc.set_choices(s0, [(p, s1), (1 - p, s0)])
    dtmc.set_choices(s1, [(1, s1)])
    return dtmc, s0, s1


def test_eliminate_state_remove_deletes_state():
    dtmc, s0, _ = _make_dtmc_pair()
    eliminate_selfloop(dtmc, s0)
    eliminate_state(dtmc, s0, remove=True)
    assert s0 not in dtmc.states


def test_eliminate_state_remove_false_keeps_state():
    dtmc, s0, _ = _make_dtmc_pair()
    eliminate_selfloop(dtmc, s0)
    eliminate_state(dtmc, s0, remove=False)
    assert s0 in dtmc.states


def test_eliminate_state_remove_leaves_no_predecessors():
    dtmc, s0, s1 = _make_dtmc_pair()
    eliminate_selfloop(dtmc, s0)
    eliminate_state(dtmc, s0, remove=True)
    assert s0 not in dtmc.compute_predecessors()[s1]


# ---------------------------------------------------------------------------
# AnalyseParametric — single-parameter (x)
# ---------------------------------------------------------------------------


def test_evaluate_fair_coin(analyser):
    """Fair coin gives P(rolled1) = 1/6."""
    result = analyser.evaluate_at_point(_FAIR)
    assert abs(result - 1 / 6) < 1e-6


def test_evaluate_symmetric_outcomes(analyser):
    """All six faces have equal probability under a fair coin."""
    for face in range(1, 7):
        prop = f'P=? [F "rolled{face}"]'
        a = AnalyseParametric(analyser.model, prop)
        assert abs(a.evaluate_at_point(_FAIR) - 1 / 6) < 1e-6


def test_evaluate_extreme_bias(analyser):
    """Result is in [0, 1] for extreme parameter values."""
    for x in (0.01, 0.99):
        result = analyser.evaluate_at_point({"x": x})
        assert 0.0 <= result <= 1.0


def test_region_bound_max_above_fair(analyser):
    """Upper bound over [0.4, 0.6] is >= P(rolled1) at the fair coin."""
    upper = analyser.get_region_bound(_REGION_FAIR, maximize=True)
    assert upper >= 1 / 6 - 1e-6


def test_region_bound_min_below_fair(analyser):
    """Lower bound over [0.4, 0.6] is <= P(rolled1) at the fair coin."""
    lower = analyser.get_region_bound(_REGION_FAIR, maximize=False)
    assert lower <= 1 / 6 + 1e-6


def test_region_bound_max_geq_min(analyser):
    upper = analyser.get_region_bound(_REGION_FAIR, maximize=True)
    lower = analyser.get_region_bound(_REGION_FAIR, maximize=False)
    assert upper >= lower - 1e-9


def test_region_bound_in_unit_interval(analyser):
    for maximize in (True, False):
        b = analyser.get_region_bound(_REGION_FAIR, maximize=maximize)
        assert 0.0 <= b <= 1.0


def test_rectangular_region_to_stormpy_type(analyser):
    sp_region = rectangular_region_to_stormpy(analyser.name_to_var, _REGION_FAIR)
    assert isinstance(sp_region, stormpy.pars.ParameterRegion)


def test_rectangular_region_to_stormpy_unknown_param(analyser):
    bad_region = RectangularRegion({"z": (0.0, 1.0)})
    with pytest.raises(KeyError):
        rectangular_region_to_stormpy(analyser.name_to_var, bad_region)


def test_annotate_region_returns_annotated_region(analyser):
    ar = analyser.annotate_region(_REGION_FAIR)
    assert isinstance(ar, AnnotatedRegion)
    assert ar.region is _REGION_FAIR


def test_annotate_region_bounds_ordered(analyser):
    ar = analyser.annotate_region(_REGION_FAIR)
    lo_min, hi_min = ar.min_value
    lo_max, hi_max = ar.max_value
    assert lo_min <= hi_min
    assert lo_max <= hi_max
    assert hi_min <= lo_max


def test_annotate_region_verified_bounds_are_tight(analyser):
    """Pmin <= sample_min and sample_max <= Pmax."""
    ar = analyser.annotate_region(_REGION_FAIR, sample=True)
    ar_no_sample = analyser.annotate_region(_REGION_FAIR, sample=False)
    pmin = ar_no_sample.min_value[0]
    pmax = ar_no_sample.max_value[1]
    assert ar.min_value[0] == pytest.approx(pmin, rel=1e-9)
    assert ar.max_value[1] == pytest.approx(pmax, rel=1e-9)
    assert ar.min_value[1] >= pmin - 1e-9
    assert ar.max_value[0] <= pmax + 1e-9


def test_annotate_region_no_sample_uses_pmin_pmax(analyser):
    """Without sampling, min_value == max_value == (Pmin, Pmax)."""
    ar = analyser.annotate_region(_REGION_FAIR, sample=False)
    assert ar.min_value == ar.max_value
    assert ar.min_value[0] <= ar.min_value[1]


def test_annotate_region_center_value_within_verified_bounds(analyser):
    """The center sample (x=0.5, P~1/6) lies within [Pmin, Pmax]."""
    ar = analyser.annotate_region(_REGION_FAIR, sample=True)
    pmin = ar.min_value[0]
    pmax = ar.max_value[1]
    center_val = analyser.evaluate_at_point({"x": 0.5})
    assert pmin <= center_val + 1e-9
    assert center_val <= pmax + 1e-9


def test_raises_for_non_parametric_model():
    dtmc = sv_model.new_dtmc()
    with pytest.raises((ValueError, ImportError)):
        AnalyseParametric(dtmc, _PROP)


def test_get_region_bound_raises_for_non_graph_preserving(analyser):
    region = RectangularRegion({"x": (0.0, 0.6)})
    with pytest.raises(ValueError, match="graph-preserving"):
        analyser.get_region_bound(region)


# ---------------------------------------------------------------------------
# AnalyseParametric — two-parameter (x, y)
# ---------------------------------------------------------------------------


def test_twocoins_evaluate_fair(analyser_xy):
    """At x=y=0.5 (both fair) P(rolled1) = 1/6."""
    result = analyser_xy.evaluate_at_point(_FAIR_XY)
    assert abs(result - 1 / 6) < 1e-6


def test_twocoins_evaluate_all_faces(pmc_xy):
    """All six faces have equal probability at x=y=0.5."""
    for face in range(1, 7):
        a = AnalyseParametric(pmc_xy, f'P=? [F "rolled{face}"]')
        assert abs(a.evaluate_at_point(_FAIR_XY) - 1 / 6) < 1e-6


def test_twocoins_evaluate_in_unit_interval(analyser_xy):
    for x, y in ((0.1, 0.9), (0.9, 0.1), (0.3, 0.7)):
        result = analyser_xy.evaluate_at_point({"x": x, "y": y})
        assert 0.0 <= result <= 1.0


def test_twocoins_region_bound_max_above_fair(analyser_xy):
    upper = analyser_xy.get_region_bound(_REGION_FAIR_XY, maximize=True)
    assert upper >= 1 / 6 - 1e-6


def test_twocoins_region_bound_min_below_fair(analyser_xy):
    lower = analyser_xy.get_region_bound(_REGION_FAIR_XY, maximize=False)
    assert lower <= 1 / 6 + 1e-6


def test_twocoins_region_bound_max_geq_min(analyser_xy):
    upper = analyser_xy.get_region_bound(_REGION_FAIR_XY, maximize=True)
    lower = analyser_xy.get_region_bound(_REGION_FAIR_XY, maximize=False)
    assert upper >= lower - 1e-9


def test_twocoins_annotate_region_valid(analyser_xy):
    ar = analyser_xy.annotate_region(_REGION_FAIR_XY)
    assert isinstance(ar, AnnotatedRegion)
    lo_min, hi_min = ar.min_value
    lo_max, hi_max = ar.max_value
    assert lo_min <= hi_min
    assert lo_max <= hi_max


def test_twocoins_annotate_region_center_within_bounds(analyser_xy):
    ar = analyser_xy.annotate_region(_REGION_FAIR_XY, sample=True)
    center_val = analyser_xy.evaluate_at_point({"x": 0.5, "y": 0.5})
    assert ar.min_value[0] <= center_val + 1e-9
    assert center_val <= ar.max_value[1] + 1e-9


def test_twocoins_stormpy_parametric_vs_state_elimination(pmc_xy):
    """stormpy solution function agrees with state elimination for the 2-param model."""
    result = sv_mc.model_checking(pmc_xy, _PROP, scheduler=False)
    assert result is not None
    stormpy_fn = result.values[pmc_xy.initial_state]
    target = pmc_xy.get_states_with_label("rolled1")
    elimination_fn = solve_reachability(pmc_xy, target)
    assert sp.cancel(stormpy_fn - elimination_fn) == 0


# ---------------------------------------------------------------------------
# AnalyseParametric — two-coin / rolled2 regression
# ---------------------------------------------------------------------------

_REGION_ROLLED2 = RectangularRegion({"x": (0.4, 0.6), "y": (0.4, 0.6)})


def test_rolled2_annotate_region_does_not_raise(analyser_xy_rolled2):
    ar = analyser_xy_rolled2.annotate_region(_REGION_ROLLED2, sample=True)
    assert ar is not None


def test_rolled2_bounds_ordered(analyser_xy_rolled2):
    ar = analyser_xy_rolled2.annotate_region(_REGION_ROLLED2, sample=True)
    lo_min, hi_min = ar.min_value
    lo_max, hi_max = ar.max_value
    assert lo_min <= hi_min
    assert lo_max <= hi_max
    assert lo_min <= hi_max


def test_rolled2_get_region_bound_returns_fraction(analyser_xy_rolled2):
    upper = analyser_xy_rolled2.get_region_bound(_REGION_ROLLED2, maximize=True)
    assert isinstance(upper, Fraction)


def test_rolled2_pmin_leq_pmax(analyser_xy_rolled2):
    pmin = analyser_xy_rolled2.get_region_bound(_REGION_ROLLED2, maximize=False)
    pmax = analyser_xy_rolled2.get_region_bound(_REGION_ROLLED2, maximize=True)
    assert pmin <= pmax


# ---------------------------------------------------------------------------
# Parameter space partitioning — single-parameter (x)
# ---------------------------------------------------------------------------


def test_psp_returns_list(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    assert isinstance(result, list)


def test_psp_all_items_are_annotated_regions(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    assert all(isinstance(r, AnnotatedRegion) for r in result)


def test_psp_decided_regions_have_correct_label(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=10)
    for ar in result:
        assert ar.classify(_THRESHOLD) in ("safe", "unsafe", "unknown", "neither")


def test_psp_safe_regions_classify_safe(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=10)
    for ar in result:
        if ar.classify(_THRESHOLD) == "safe":
            assert ar.min_value[0] >= _THRESHOLD


def test_psp_unsafe_regions_classify_unsafe(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=10)
    for ar in result:
        if ar.classify(_THRESHOLD) == "unsafe":
            assert ar.max_value[1] < _THRESHOLD


def test_psp_more_splits_produce_more_regions(pmc):
    few = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    many = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=20)
    assert len(many) >= len(few)


def test_psp_default_initial_region(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    assert len(result) > 0


def test_psp_explicit_initial_region(pmc):
    region = RectangularRegion({"x": (Fraction(1, 10), Fraction(9, 10))})
    result = parameter_space_partitioning(
        pmc, _PROP, _THRESHOLD, initial_region=region, max_iterations=4
    )
    assert len(result) > 0


def test_psp_raises_for_non_graph_preserving_region(pmc):
    bad = RectangularRegion({"x": (Fraction(0), Fraction(9, 10))})
    with pytest.raises(ValueError, match="graph-preserving"):
        parameter_space_partitioning(pmc, _PROP, _THRESHOLD, initial_region=bad)


def test_psp_zero_iterations_returns_annotated_initial_region(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=0)
    assert len(result) == 1
    assert isinstance(result[0], AnnotatedRegion)


def test_psp_chain_pmc():
    """parameter_space_partitioning succeeds on a 3-step chain with a bounded formula."""
    pmc = sv_model.new_dtmc(create_initial_state=False)
    x = pmc.declare_parameter("x")
    s0 = pmc.new_state(["init"], friendly_name="s0")
    s1 = pmc.new_state(friendly_name="s1")
    s2 = pmc.new_state(friendly_name="s2")
    s3 = pmc.new_state(["T"], friendly_name="s3")
    s4 = pmc.new_state(["sink"], friendly_name="s4")
    pmc.set_choices(s0, [(x, s1), (1 - x, s4)])
    pmc.set_choices(s1, [(1 - x, s2), (x, s4)])
    pmc.set_choices(s2, [(x, s3), (1 - x, s4)])
    pmc.set_choices(s3, [(1, s3)])
    pmc.set_choices(s4, [(1, s4)])

    region = RectangularRegion({"x": (Fraction(1, 10), Fraction(9, 10))})
    result = parameter_space_partitioning(
        pmc, 'P=? [F "T"]', 0.2, initial_region=region, max_iterations=5
    )
    assert isinstance(result, list)
    assert all(isinstance(r, AnnotatedRegion) for r in result)


# ---------------------------------------------------------------------------
# Parameter space partitioning — two-parameter (x, y)
# ---------------------------------------------------------------------------


def test_psp_twocoins_returns_list(pmc_xy):
    result = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    assert isinstance(result, list)


def test_psp_twocoins_all_items_are_annotated_regions(pmc_xy):
    result = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    assert all(isinstance(r, AnnotatedRegion) for r in result)


def test_psp_twocoins_default_region_covers_both_params(pmc_xy):
    result = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    assert len(result) > 0
    for ar in result:
        assert "x" in ar.region.bounds
        assert "y" in ar.region.bounds


def test_psp_twocoins_explicit_2d_region(pmc_xy):
    region = RectangularRegion(
        {
            "x": (Fraction(1, 10), Fraction(9, 10)),
            "y": (Fraction(1, 10), Fraction(9, 10)),
        }
    )
    result = parameter_space_partitioning(
        pmc_xy, _PROP, _THRESHOLD, initial_region=region, max_iterations=4
    )
    assert len(result) > 0


def test_psp_twocoins_more_splits_produce_more_regions(pmc_xy):
    few = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    many = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=20)
    assert len(many) >= len(few)


def test_psp_twocoins_raises_for_non_graph_preserving_region(pmc_xy):
    bad = RectangularRegion(
        {"x": (Fraction(0), Fraction(9, 10)), "y": (Fraction(1, 10), Fraction(9, 10))}
    )
    with pytest.raises(ValueError, match="graph-preserving"):
        parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, initial_region=bad)
