"""Tests for RectangularRegion and to_interval_mdp."""

from fractions import Fraction

import pytest
import sympy as sp

import stormvogel.model as sv_model
from stormvogel.model.value import Interval
from stormvogel.parametric import RectangularRegion, to_interval_mdp


# ---------------------------------------------------------------------------
# RectangularRegion
# ---------------------------------------------------------------------------


def test_region_contains_interior():
    r = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert r.contains({"x": Fraction(1, 2)})


def test_region_contains_boundary():
    r = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert r.contains({"x": Fraction(1, 4)})
    assert r.contains({"x": Fraction(3, 4)})


def test_region_does_not_contain_outside():
    r = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert not r.contains({"x": Fraction(0)})
    assert not r.contains({"x": Fraction(1)})


def test_region_contains_multi_parameter():
    r = RectangularRegion(
        {
            "x": (Fraction(0), Fraction(1, 2)),
            "y": (Fraction(1, 4), Fraction(3, 4)),
        }
    )
    assert r.contains({"x": Fraction(1, 4), "y": Fraction(1, 2)})
    assert not r.contains({"x": Fraction(3, 4), "y": Fraction(1, 2)})


def test_region_vertices_single_parameter():
    r = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    vs = r.vertices()
    assert len(vs) == 2
    assert {"x": Fraction(1, 4)} in vs
    assert {"x": Fraction(3, 4)} in vs


def test_region_vertices_two_parameters():
    r = RectangularRegion(
        {
            "x": (Fraction(0), Fraction(1)),
            "y": (Fraction(0), Fraction(1)),
        }
    )
    assert len(r.vertices()) == 4


def test_region_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower bound"):
        RectangularRegion({"x": (Fraction(3, 4), Fraction(1, 4))})


# ---------------------------------------------------------------------------
# to_interval_mdp — simple pMDP (x and 1-x)
# ---------------------------------------------------------------------------


def _simple_pmc():
    """DTMC: init --x--> A, init --(1-x)--> B; A and B absorbing."""
    pmc = sv_model.new_dtmc(create_initial_state=False)
    x = pmc.declare_parameter("x")
    s0 = pmc.new_state(labels=["init"])
    s_a = pmc.new_state(labels=["A"])
    s_b = pmc.new_state(labels=["B"])
    pmc.set_choices(s0, [(x, s_a), (1 - x, s_b)])
    pmc.set_choices(s_a, [(1, s_a)])
    pmc.set_choices(s_b, [(1, s_b)])
    return pmc, s0, s_a, s_b


def test_to_interval_mdp_preserves_state_uuids():
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    imdp = to_interval_mdp(pmc, region)
    assert imdp.get_state_by_id(s0.state_id) is not None
    assert imdp.get_state_by_id(s_a.state_id) is not None


def test_to_interval_mdp_is_interval_model():
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    imdp = to_interval_mdp(pmc, region)
    assert imdp.is_interval_model()


def test_to_interval_mdp_no_parameters():
    pmc, s0, _, _ = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    imdp = to_interval_mdp(pmc, region)
    assert not imdp.is_parametric()
    assert imdp.parameters == ()


def test_to_interval_mdp_simple_intervals():
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    imdp = to_interval_mdp(pmc, region)

    i_s0 = imdp.get_state_by_id(s0.state_id)
    i_sa = imdp.get_state_by_id(s_a.state_id)
    i_sb = imdp.get_state_by_id(s_b.state_id)

    # Collect intervals out of s0
    intervals: dict = {}
    for _, branch in imdp.transitions[i_s0]:
        for val, target in branch:
            intervals[target.state_id] = val

    # x -> [1/4, 3/4]
    assert intervals[i_sa.state_id] == Interval(Fraction(1, 4), Fraction(3, 4))
    # 1-x -> [1/4, 3/4]
    assert intervals[i_sb.state_id] == Interval(Fraction(1, 4), Fraction(3, 4))


def test_to_interval_mdp_point_interval_for_concrete():
    """Concrete probability 1 in absorbing states becomes [1, 1]."""
    pmc, _, s_a, _ = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    imdp = to_interval_mdp(pmc, region)

    i_sa = imdp.get_state_by_id(s_a.state_id)
    for _, branch in imdp.transitions[i_sa]:
        for val, _ in branch:
            assert isinstance(val, Interval)
            assert val.lower == val.upper


# ---------------------------------------------------------------------------
# to_interval_mdp — affine multi-parameter expression
# ---------------------------------------------------------------------------


def test_to_interval_mdp_affine_multiparameter():
    """p = 0.2*x + 0.3*y; x in [0, 1], y in [0, 1] -> [0, 0.5]."""
    pmc = sv_model.new_dtmc(create_initial_state=False)
    x = pmc.declare_parameter("x")
    y = pmc.declare_parameter("y")
    s0 = pmc.new_state(labels=["init"])
    s_a = pmc.new_state(labels=["A"])
    s_b = pmc.new_state(labels=["B"])
    p = sp.Rational(1, 5) * x + sp.Rational(3, 10) * y
    pmc.set_choices(s0, [(p, s_a), (1 - p, s_b)])
    pmc.set_choices(s_a, [(1, s_a)])
    pmc.set_choices(s_b, [(1, s_b)])

    region = RectangularRegion(
        {
            "x": (Fraction(0), Fraction(1)),
            "y": (Fraction(0), Fraction(1)),
        }
    )
    imdp = to_interval_mdp(pmc, region)

    i_s0 = imdp.get_state_by_id(s0.state_id)
    i_sa = imdp.get_state_by_id(s_a.state_id)

    intervals: dict = {}
    for _, branch in imdp.transitions[i_s0]:
        for val, target in branch:
            intervals[target.state_id] = val

    iv = intervals[i_sa.state_id]
    assert iv.lower == Fraction(0)
    assert iv.upper == Fraction(1, 2)


def test_to_interval_mdp_negative_coefficient():
    """p = 1 - 0.5*x; x in [0.2, 0.8] -> [0.6, 0.9]."""
    pmc = sv_model.new_dtmc(create_initial_state=False)
    x = pmc.declare_parameter("x")
    s0 = pmc.new_state(labels=["init"])
    s_a = pmc.new_state(labels=["A"])
    s_b = pmc.new_state(labels=["B"])
    p = 1 - sp.Rational(1, 2) * x
    pmc.set_choices(s0, [(p, s_a), (sp.Rational(1, 2) * x, s_b)])
    pmc.set_choices(s_a, [(1, s_a)])
    pmc.set_choices(s_b, [(1, s_b)])

    region = RectangularRegion({"x": (Fraction(1, 5), Fraction(4, 5))})
    imdp = to_interval_mdp(pmc, region)

    i_s0 = imdp.get_state_by_id(s0.state_id)
    i_sa = imdp.get_state_by_id(s_a.state_id)

    intervals: dict = {}
    for _, branch in imdp.transitions[i_s0]:
        for val, target in branch:
            intervals[target.state_id] = val

    iv = intervals[i_sa.state_id]
    # 1 - 0.5*x: min at x=0.8 -> 0.6, max at x=0.2 -> 0.9
    assert iv.lower == Fraction(3, 5)
    assert iv.upper == Fraction(9, 10)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_to_interval_mdp_raises_for_nonlinear():
    pmc = sv_model.new_dtmc(create_initial_state=False)
    x = pmc.declare_parameter("x")
    s0 = pmc.new_state(labels=["init"])
    s_a = pmc.new_state(labels=["A"])
    s_b = pmc.new_state(labels=["B"])
    pmc.set_choices(s0, [(x * x, s_a), (1 - x * x, s_b)])
    pmc.set_choices(s_a, [(1, s_a)])
    pmc.set_choices(s_b, [(1, s_b)])

    region = RectangularRegion({"x": (Fraction(0), Fraction(1))})
    with pytest.raises(ValueError, match="affine"):
        to_interval_mdp(pmc, region)


def test_to_interval_mdp_raises_for_missing_parameter():
    pmc = sv_model.new_dtmc(create_initial_state=False)
    x = pmc.declare_parameter("x")
    s0 = pmc.new_state(labels=["init"])
    s_a = pmc.new_state(labels=["A"])
    s_b = pmc.new_state(labels=["B"])
    pmc.set_choices(s0, [(x, s_a), (1 - x, s_b)])
    pmc.set_choices(s_a, [(1, s_a)])
    pmc.set_choices(s_b, [(1, s_b)])

    region = RectangularRegion({"y": (Fraction(0), Fraction(1))})  # x missing
    with pytest.raises(ValueError, match="'x'"):
        to_interval_mdp(pmc, region)
