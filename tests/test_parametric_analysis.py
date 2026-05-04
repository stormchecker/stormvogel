"""Tests for AnalyseParametric and rectangular_region_to_stormpy."""

import pytest
import sympy as sp

from stormvogel.examples.knuth_yao_pmc import (
    create_knuth_yao_pmc,
    create_knuth_yao_pmc_twocoins,
)
from stormvogel.parametric.region import RectangularRegion
from stormvogel.teaching.parametric import solve_reachability
from stormvogel.stormpy_utils.parametric_analysis import (
    AnalyseParametric,
    rectangular_region_to_stormpy,
)

stormpy = pytest.importorskip("stormpy")

_PROP = 'P=? [F "rolled1"]'
_FAIR = {"x": 0.5}
_REGION_FAIR = RectangularRegion({"x": (0.4, 0.6)})


@pytest.fixture(scope="module")
def analyser():
    pmc = create_knuth_yao_pmc()
    return AnalyseParametric(pmc, _PROP)


@pytest.fixture(scope="module")
def analyser_region():
    pmc = create_knuth_yao_pmc()
    return AnalyseParametric(pmc, _PROP)


# ---------------------------------------------------------------------------
# evaluate_at_point
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


# ---------------------------------------------------------------------------
# get_region_bound
# ---------------------------------------------------------------------------


def test_region_bound_max_above_fair(analyser_region):
    """Upper bound over [0.4, 0.6] is ≥ P(rolled1) at the fair coin."""
    upper = analyser_region.get_region_bound(_REGION_FAIR, maximize=True)
    assert upper >= 1 / 6 - 1e-6


def test_region_bound_min_below_fair(analyser_region):
    """Lower bound over [0.4, 0.6] is ≤ P(rolled1) at the fair coin."""
    lower = analyser_region.get_region_bound(_REGION_FAIR, maximize=False)
    assert lower <= 1 / 6 + 1e-6


def test_region_bound_max_geq_min(analyser_region):
    """Upper bound is always ≥ lower bound."""
    upper = analyser_region.get_region_bound(_REGION_FAIR, maximize=True)
    lower = analyser_region.get_region_bound(_REGION_FAIR, maximize=False)
    assert upper >= lower - 1e-9


def test_region_bound_in_unit_interval(analyser_region):
    """Bounds are probabilities and lie in [0, 1]."""
    for maximize in (True, False):
        b = analyser_region.get_region_bound(_REGION_FAIR, maximize=maximize)
        assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# rectangular_region_to_stormpy
# ---------------------------------------------------------------------------


def test_rectangular_region_to_stormpy_type(analyser_region):
    sp_region = rectangular_region_to_stormpy(analyser_region.name_to_var, _REGION_FAIR)
    assert isinstance(sp_region, stormpy.pars.ParameterRegion)


def test_rectangular_region_to_stormpy_unknown_param(analyser_region):
    bad_region = RectangularRegion({"z": (0.0, 1.0)})
    with pytest.raises(KeyError):
        rectangular_region_to_stormpy(analyser_region.name_to_var, bad_region)


# ---------------------------------------------------------------------------
# annotate_region
# ---------------------------------------------------------------------------


def test_annotate_region_returns_annotated_region(analyser_region):
    from stormvogel.parametric.region import AnnotatedRegion

    ar = analyser_region.annotate_region(_REGION_FAIR)
    assert isinstance(ar, AnnotatedRegion)
    assert ar.region is _REGION_FAIR


def test_annotate_region_bounds_ordered(analyser_region):
    ar = analyser_region.annotate_region(_REGION_FAIR)
    lo_min, hi_min = ar.min_value
    lo_max, hi_max = ar.max_value
    assert lo_min <= hi_min
    assert lo_max <= hi_max
    assert hi_min <= lo_max


def test_annotate_region_verified_bounds_are_tight(analyser_region):
    """Pmin ≤ sample_min and sample_max ≤ Pmax."""
    ar = analyser_region.annotate_region(_REGION_FAIR, sample=True)
    ar_no_sample = analyser_region.annotate_region(_REGION_FAIR, sample=False)
    pmin = ar_no_sample.min_value[0]
    pmax = ar_no_sample.max_value[1]
    assert ar.min_value[0] == pytest.approx(pmin, rel=1e-9)
    assert ar.max_value[1] == pytest.approx(pmax, rel=1e-9)
    assert ar.min_value[1] >= pmin - 1e-9
    assert ar.max_value[0] <= pmax + 1e-9


def test_annotate_region_no_sample_uses_pmin_pmax(analyser_region):
    """Without sampling, min_value == max_value == (Pmin, Pmax)."""
    ar = analyser_region.annotate_region(_REGION_FAIR, sample=False)
    assert ar.min_value == ar.max_value
    assert ar.min_value[0] <= ar.min_value[1]


def test_annotate_region_center_value_within_verified_bounds(analyser_region):
    """The center sample (x=0.5, P≈1/6) lies within [Pmin, Pmax]."""
    ar = analyser_region.annotate_region(_REGION_FAIR, sample=True)
    pmin = ar.min_value[0]
    pmax = ar.max_value[1]
    center_val = analyser_region.evaluate_at_point({"x": 0.5})
    assert pmin <= center_val + 1e-9
    assert center_val <= pmax + 1e-9


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------


def test_raises_for_non_parametric_model():
    import stormvogel.model as sv_model

    dtmc = sv_model.new_dtmc()
    with pytest.raises((ValueError, ImportError)):
        AnalyseParametric(dtmc, _PROP)


def test_get_region_bound_raises_for_non_graph_preserving(analyser_region):
    # x in [0, 0.6]: at x=0 the transition x reaches 0 → not graph-preserving
    region = RectangularRegion({"x": (0.0, 0.6)})
    with pytest.raises(ValueError, match="graph-preserving"):
        analyser_region.get_region_bound(region)


# ===========================================================================
# Two-coin variant (parameters x and y)
# ===========================================================================

_FAIR_XY = {"x": 0.5, "y": 0.5}
_REGION_FAIR_XY = RectangularRegion({"x": (0.4, 0.6), "y": (0.4, 0.6)})


@pytest.fixture(scope="module")
def analyser_xy():
    pmc = create_knuth_yao_pmc_twocoins()
    return AnalyseParametric(pmc, _PROP)


@pytest.fixture(scope="module")
def analyser_xy_region():
    pmc = create_knuth_yao_pmc_twocoins()
    return AnalyseParametric(pmc, _PROP)


# ---------------------------------------------------------------------------
# evaluate_at_point
# ---------------------------------------------------------------------------


def test_twocoins_evaluate_fair(analyser_xy):
    """At x=y=0.5 (both fair) P(rolled1) = 1/6."""
    result = analyser_xy.evaluate_at_point(_FAIR_XY)
    assert abs(result - 1 / 6) < 1e-6


def test_twocoins_evaluate_all_faces(analyser_xy):
    """All six faces have equal probability at x=y=0.5."""
    pmc = create_knuth_yao_pmc_twocoins()
    for face in range(1, 7):
        a = AnalyseParametric(pmc, f'P=? [F "rolled{face}"]')
        assert abs(a.evaluate_at_point(_FAIR_XY) - 1 / 6) < 1e-6


def test_twocoins_evaluate_in_unit_interval(analyser_xy):
    for x, y in ((0.1, 0.9), (0.9, 0.1), (0.3, 0.7)):
        result = analyser_xy.evaluate_at_point({"x": x, "y": y})
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# get_region_bound
# ---------------------------------------------------------------------------


def test_twocoins_region_bound_max_above_fair(analyser_xy_region):
    upper = analyser_xy_region.get_region_bound(_REGION_FAIR_XY, maximize=True)
    assert upper >= 1 / 6 - 1e-6


def test_twocoins_region_bound_min_below_fair(analyser_xy_region):
    lower = analyser_xy_region.get_region_bound(_REGION_FAIR_XY, maximize=False)
    assert lower <= 1 / 6 + 1e-6


def test_twocoins_region_bound_max_geq_min(analyser_xy_region):
    upper = analyser_xy_region.get_region_bound(_REGION_FAIR_XY, maximize=True)
    lower = analyser_xy_region.get_region_bound(_REGION_FAIR_XY, maximize=False)
    assert upper >= lower - 1e-9


# ---------------------------------------------------------------------------
# annotate_region
# ---------------------------------------------------------------------------


def test_twocoins_annotate_region_valid(analyser_xy_region):
    from stormvogel.parametric.region import AnnotatedRegion

    ar = analyser_xy_region.annotate_region(_REGION_FAIR_XY)
    assert isinstance(ar, AnnotatedRegion)
    lo_min, hi_min = ar.min_value
    lo_max, hi_max = ar.max_value
    assert lo_min <= hi_min
    assert lo_max <= hi_max


def test_twocoins_annotate_region_center_within_bounds(analyser_xy_region):
    ar = analyser_xy_region.annotate_region(_REGION_FAIR_XY, sample=True)
    center_val = analyser_xy_region.evaluate_at_point({"x": 0.5, "y": 0.5})
    assert ar.min_value[0] <= center_val + 1e-9
    assert center_val <= ar.max_value[1] + 1e-9


# ===========================================================================
# Two-coin / rolled2 — regression for float-vs-Fraction clamping bug
# ===========================================================================

_PROP_ROLLED2 = 'P=? [F "rolled2"]'
_REGION_ROLLED2 = RectangularRegion({"x": (0.4, 0.6), "y": (0.4, 0.6)})


@pytest.fixture(scope="module")
def analyser_xy_rolled2():
    pmc = create_knuth_yao_pmc_twocoins()
    return AnalyseParametric(pmc, _PROP_ROLLED2)


def test_rolled2_annotate_region_does_not_raise(analyser_xy_rolled2):
    # Previously raised ValueError from AnnotatedRegion when a float sample
    # exceeded the exact Fraction region bound due to float arithmetic.
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
    from fractions import Fraction

    upper = analyser_xy_rolled2.get_region_bound(_REGION_ROLLED2, maximize=True)
    assert isinstance(upper, Fraction)


def test_rolled2_pmin_leq_pmax(analyser_xy_rolled2):
    pmin = analyser_xy_rolled2.get_region_bound(_REGION_ROLLED2, maximize=False)
    pmax = analyser_xy_rolled2.get_region_bound(_REGION_ROLLED2, maximize=True)
    assert pmin <= pmax


# ---------------------------------------------------------------------------
# Parametric model checking vs state elimination (two parameters)
# ---------------------------------------------------------------------------


def test_twocoins_stormpy_parametric_vs_state_elimination():
    """stormpy solution function agrees with state elimination for the 2-param model."""
    import stormvogel.stormpy_utils.model_checking as sv_mc

    pmc = create_knuth_yao_pmc_twocoins()
    result = sv_mc.model_checking(pmc, _PROP, scheduler=False)
    assert result is not None
    stormpy_fn = result.values[pmc.initial_state]

    target = pmc.get_states_with_label("rolled1")
    elimination_fn = solve_reachability(pmc, target)

    assert sp.cancel(stormpy_fn - elimination_fn) == 0


# ---------------------------------------------------------------------------
# parameter_space_partitioning
# ---------------------------------------------------------------------------


def _make_chain_pmc():
    """3-step chain pMC with target T and absorbing sink."""
    import stormvogel.model as sv_model

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
    return pmc


def test_parameter_space_partitioning_chain_pmc():
    """parameter_space_partitioning succeeds on a 3-step chain with a bounded formula."""
    from fractions import Fraction

    from stormvogel.parametric.region import AnnotatedRegion
    from stormvogel.teaching.parametric import parameter_space_partitioning

    region = RectangularRegion({"x": (Fraction(1, 10), Fraction(9, 10))})
    result = parameter_space_partitioning(
        _make_chain_pmc(),
        'P=? [F "T"]',
        0.2,
        initial_region=region,
        max_iterations=5,
    )
    assert isinstance(result, list)
    assert all(isinstance(r, AnnotatedRegion) for r in result)
