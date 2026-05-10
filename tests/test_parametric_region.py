"""Tests for RectangularRegion and to_interval_mdp."""

from fractions import Fraction

import matplotlib
import pytest
import stormvogel
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import stormvogel.model as sv_model
from stormvogel.model.value import Interval
from stormvogel.parametric import RectangularRegion, to_interval_mdp
from stormvogel.parametric.region import (
    AnnotatedRegion,
    plot_regions,
)


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


# ---------------------------------------------------------------------------
# RectangularRegion.split
# ---------------------------------------------------------------------------


def test_split_named_param_bounds():
    r = RectangularRegion({"x": (Fraction(0), Fraction(1))})
    lo, hi = r.split("x")
    assert lo.bounds["x"] == (Fraction(0), Fraction(1, 2))
    assert hi.bounds["x"] == (Fraction(1, 2), Fraction(1))


def test_split_fraction_midpoint_exact():
    r = RectangularRegion({"x": (Fraction(1, 3), Fraction(2, 3))})
    lo, hi = r.split("x")
    assert lo.bounds["x"][1] == Fraction(1, 2)
    assert hi.bounds["x"][0] == Fraction(1, 2)


def test_split_covers_original():
    r = RectangularRegion({"x": (Fraction(0), Fraction(1))})
    lo, hi = r.split("x")
    assert lo.contains({"x": Fraction(1, 4)})
    assert hi.contains({"x": Fraction(3, 4)})
    assert not lo.contains({"x": Fraction(3, 4)})
    assert not hi.contains({"x": Fraction(1, 4)})


def test_split_auto_picks_largest_dimension():
    r = RectangularRegion(
        {
            "x": (Fraction(0), Fraction(1)),
            "y": (Fraction(0), Fraction(1, 5)),
        }
    )
    lo, hi = r.split()
    # x has range 1, y has range 1/5 — should split x
    assert lo.bounds["x"] == (Fraction(0), Fraction(1, 2))
    assert lo.bounds["y"] == r.bounds["y"]


def test_split_2d_named_leaves_other_intact():
    r = RectangularRegion(
        {
            "x": (Fraction(0), Fraction(1)),
            "y": (Fraction(0), Fraction(1)),
        }
    )
    lo, hi = r.split("y")
    assert lo.bounds["x"] == r.bounds["x"]
    assert hi.bounds["x"] == r.bounds["x"]
    assert lo.bounds["y"] == (Fraction(0), Fraction(1, 2))
    assert hi.bounds["y"] == (Fraction(1, 2), Fraction(1))


def test_split_unknown_param_raises():
    r = RectangularRegion({"x": (Fraction(0), Fraction(1))})
    with pytest.raises(ValueError, match="'z'"):
        r.split("z")


# ---------------------------------------------------------------------------
# plot_regions
# ---------------------------------------------------------------------------


def test_plot_regions_returns_axes():
    r = RectangularRegion({"p": (0.0, 0.5), "q": (0.0, 1.0)})
    try:
        ax = plot_regions([(r, "safe")])
        import matplotlib.axes

        assert isinstance(ax, matplotlib.axes.Axes)
    finally:
        plt.close("all")


def test_plot_regions_axis_labels():
    r = RectangularRegion({"p": (0.0, 1.0), "q": (0.0, 1.0)})
    try:
        ax = plot_regions([(r, "unknown")])
        assert ax.get_xlabel() == "p"
        assert ax.get_ylabel() == "q"
    finally:
        plt.close("all")


def test_plot_regions_param_order():
    r = RectangularRegion({"p": (0.0, 1.0), "q": (0.0, 1.0)})
    try:
        ax = plot_regions([(r, "safe")], param_order=("q", "p"))
        assert ax.get_xlabel() == "q"
        assert ax.get_ylabel() == "p"
    finally:
        plt.close("all")


def test_plot_regions_raises_for_non_2d():
    r = RectangularRegion({"p": (0.0, 1.0), "q": (0.0, 1.0), "r": (0.0, 1.0)})
    with pytest.raises(ValueError, match="2 parameters"):
        plot_regions([(r, "safe")])


# ---------------------------------------------------------------------------
# AnnotatedRegion
# ---------------------------------------------------------------------------


def _ar(
    lo_min: float,
    hi_min: float,
    lo_max: float,
    hi_max: float,
    p_lo: float = 0.0,
    p_hi: float = 1.0,
    q_lo: float = 0.0,
    q_hi: float = 1.0,
) -> AnnotatedRegion:
    return AnnotatedRegion(
        region=RectangularRegion({"p": (p_lo, p_hi), "q": (q_lo, q_hi)}),
        min_value=(lo_min, hi_min),
        max_value=(lo_max, hi_max),
    )


def test_annotated_region_classify_safe():
    ar = _ar(0.7, 0.8, 0.9, 1.0)
    assert ar.classify(0.6) == "safe"


def test_annotated_region_classify_unsafe():
    ar = _ar(0.1, 0.2, 0.3, 0.4)
    assert ar.classify(0.5) == "unsafe"


def test_annotated_region_classify_neither():
    # hi_min=0.4 < 0.5 and lo_max=0.7 >= 0.5 → certifiably straddles threshold
    ar = _ar(0.3, 0.4, 0.7, 0.8)
    assert ar.classify(0.5) == "neither"


def test_annotated_region_classify_unknown_hi_min_above_threshold():
    # hi_min=0.6 >= 0.5 → min might be above threshold, can't certify "neither"
    ar = _ar(0.3, 0.6, 0.7, 0.8)
    assert ar.classify(0.5) == "unknown"


def test_annotated_region_classify_unknown_lo_max_below_threshold():
    # lo_max=0.4 < 0.5 → max might be below threshold, can't certify "neither"
    ar = _ar(0.1, 0.3, 0.4, 0.8)
    assert ar.classify(0.5) == "unknown"


def test_annotated_region_classify_boundary_safe():
    ar = _ar(0.5, 0.6, 0.8, 0.9)
    assert ar.classify(0.5) == "safe"


def test_annotated_region_classify_boundary_unsafe():
    ar = _ar(0.1, 0.2, 0.3, 0.49)
    assert ar.classify(0.5) == "unsafe"


def test_annotated_region_rejects_inverted_min():
    with pytest.raises(ValueError, match="min_value interval is inverted"):
        AnnotatedRegion(
            region=RectangularRegion({"p": (0.0, 1.0), "q": (0.0, 1.0)}),
            min_value=(0.8, 0.3),
            max_value=(0.5, 0.9),
        )


def test_annotated_region_rejects_inverted_max():
    with pytest.raises(ValueError, match="max_value interval is inverted"):
        AnnotatedRegion(
            region=RectangularRegion({"p": (0.0, 1.0), "q": (0.0, 1.0)}),
            min_value=(0.1, 0.2),
            max_value=(0.9, 0.4),
        )


def test_annotated_region_rejects_min_above_max():
    with pytest.raises(ValueError, match="max_value upper bound"):
        AnnotatedRegion(
            region=RectangularRegion({"p": (0.0, 1.0), "q": (0.0, 1.0)}),
            min_value=(0.9, 0.95),
            max_value=(0.1, 0.8),
        )


# ---------------------------------------------------------------------------
# plot_regions with annotated regions
# ---------------------------------------------------------------------------


def test_plot_annotated_regions_returns_axes():
    ar = _ar(0.7, 0.8, 0.9, 1.0)
    try:
        import matplotlib.axes

        ax = plot_regions([ar], threshold=0.6)
        assert isinstance(ax, matplotlib.axes.Axes)
    finally:
        plt.close("all")


def test_plot_annotated_regions_mixed():
    safe = _ar(0.7, 0.8, 0.9, 1.0, p_lo=0.0, p_hi=0.5)
    unsafe = _ar(0.1, 0.2, 0.3, 0.4, p_lo=0.5, p_hi=1.0)
    try:
        ax = plot_regions([safe, unsafe], threshold=0.6)
        assert ax is not None
    finally:
        plt.close("all")


def test_plot_annotated_regions_param_order():
    ar = _ar(0.3, 0.4, 0.7, 0.8)
    try:
        ax = plot_regions([ar], threshold=0.5, param_order=("q", "p"))
        assert ax.get_xlabel() == "q"
        assert ax.get_ylabel() == "p"
    finally:
        plt.close("all")


def test_plot_solution_function():
    pytest.importorskip("stormpy")
    pkydie = stormvogel.examples.create_knuth_yao_pmc_twocoins()
    threshold = 1 / 6
    prop = 'P=? [F "rolled3"]'
    mc_result = stormvogel.model_checking(pkydie, prop)
    assert mc_result is not None
    solfunc = mc_result.at_init()
    plot_regions(
        [],
        threshold=threshold,
        solution_fn=solfunc,
        shade_safe=True,
        param_order=("x", "y"),
        x_lim=(0.0001, 0.99999),
        y_lim=(0.0001, 0.99999),
    )


# ---------------------------------------------------------------------------
# is_well_defined / is_graph_preserving
# ---------------------------------------------------------------------------


def test_is_well_defined_true_for_interior_region():
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert region.is_well_defined(pmc)


def test_is_well_defined_true_at_boundary():
    # x and 1-x both stay in [0, 1] when x in [0, 1]
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(0), Fraction(1))})
    assert region.is_well_defined(pmc)


def test_is_well_defined_false_when_prob_goes_negative():
    # x in [-1/10, 1/2]: x can be negative
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(-1, 10), Fraction(1, 2))})
    assert not region.is_well_defined(pmc)


def test_is_well_defined_false_when_prob_exceeds_one():
    # transition 2*x with x in [2/5, 3/5]: 2*x reaches 6/5 > 1
    pmc = sv_model.new_dtmc()
    x = pmc.declare_parameter("x")
    s0 = pmc.initial_state
    sa = pmc.new_state()
    sb = pmc.new_state()
    pmc.set_choices(s0, [(2 * x, sa), (1 - 2 * x, sb)])
    pmc.set_choices(sa, [(1, sa)])
    pmc.set_choices(sb, [(1, sb)])
    region = RectangularRegion({"x": (Fraction(2, 5), Fraction(3, 5))})
    assert not region.is_well_defined(pmc)


def test_is_well_defined_false_for_negative_reward():
    # reward -x with x in [1/4, 3/4]: reward in [-3/4, -1/4]
    pmc, s0, s_a, s_b = _simple_pmc()
    x = sp.Symbol("x")
    rm = pmc.new_reward_model("R")
    rm.set_state_reward(s0, -x)
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert not region.is_well_defined(pmc)


def test_is_well_defined_true_for_nonneg_reward():
    # reward x with x in [1/4, 3/4]: reward in [1/4, 3/4] >= 0
    pmc, s0, s_a, s_b = _simple_pmc()
    x = sp.Symbol("x")
    rm = pmc.new_reward_model("R")
    rm.set_state_reward(s0, x)
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert region.is_well_defined(pmc)


def test_is_well_defined_false_for_transition_1_minus_x_outside_range():
    # 1-x with x in [1/2, 3/2]: min of 1-x = 1-3/2 = -1/2 < 0
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 2), Fraction(3, 2))})
    assert not region.is_well_defined(pmc)


def test_is_well_defined_true_for_non_parametric():
    dtmc = sv_model.new_dtmc()
    s = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(1.0, s)])
    region = RectangularRegion({})
    assert region.is_well_defined(dtmc)


def test_is_graph_preserving_true_for_interior_region():
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 4), Fraction(3, 4))})
    assert region.is_graph_preserving(pmc)


def test_is_graph_preserving_false_when_x_reaches_zero():
    # x in [0, 1]: min of x = 0
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(0), Fraction(1))})
    assert not region.is_graph_preserving(pmc)


def test_is_graph_preserving_false_when_complement_reaches_zero():
    # x in [1/2, 1]: min of 1-x = 0
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 2), Fraction(1))})
    assert not region.is_graph_preserving(pmc)


def test_is_graph_preserving_false_for_transition_1_minus_x_outside_range():
    # 1-x with x in [1/2, 3/2]: min of 1-x = -1/2 <= 0
    pmc, s0, s_a, s_b = _simple_pmc()
    region = RectangularRegion({"x": (Fraction(1, 2), Fraction(3, 2))})
    assert not region.is_graph_preserving(pmc)


def test_is_graph_preserving_true_for_non_parametric():
    dtmc = sv_model.new_dtmc()
    s = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(1.0, s)])
    region = RectangularRegion({})
    assert region.is_graph_preserving(dtmc)
