"""Tests for stormvogel.teaching.pareto, including explore_pareto."""

import numpy as np
import pytest
import stormvogel.model
from stormvogel.result import ParetoResult, plot_pareto_result
from stormvogel.teaching.pareto import (
    ParetoQuery,
    _feasible_polygon,
    explore_pareto,
    plot_pareto,
)


def _two_queries():
    return [
        ParetoQuery(w=(1.0, 0.0), p=(0.8, 0.5)),
        ParetoQuery(w=(0.0, 1.0), p=(0.4, 0.9)),
    ]


def _diagonal_queries():
    return [
        ParetoQuery(w=(1.0, 0.0), p=(0.8, 0.3)),
        ParetoQuery(w=(0.0, 1.0), p=(0.3, 0.8)),
        ParetoQuery(w=(1.0, 1.0), p=(0.6, 0.6)),
    ]


def test_feasible_polygon_contains_achievable_points():
    """Every achievable point must lie inside (or on) the feasible region."""
    queries = _diagonal_queries()
    verts = _feasible_polygon(queries, -2.0, -2.0)
    assert verts is not None

    from matplotlib.path import Path

    path = Path(np.vstack([verts, verts[:1]]))  # closed path

    for q in queries:
        assert path.contains_point(q.p, radius=1e-6), f"{q.p} not in feasible region"


def test_feasible_polygon_excludes_dominated_points():
    """A point strictly dominating all p_i must lie outside the feasible region."""
    queries = _diagonal_queries()
    verts = _feasible_polygon(queries, -2.0, -2.0)
    assert verts is not None

    from matplotlib.path import Path

    path = Path(np.vstack([verts, verts[:1]]))

    dominated = (1.0, 1.0)  # better than all p_i in every weighted direction
    assert not path.contains_point(dominated, radius=-1e-6)


def test_feasible_polygon_vertex_count():
    queries = _two_queries()
    verts = _feasible_polygon(queries, -2.0, -2.0)
    assert verts is not None
    assert len(verts) >= 3


def test_plot_pareto_runs():
    import matplotlib

    matplotlib.use("Agg")
    ax = plot_pareto(_diagonal_queries())
    assert ax is not None


def test_plot_pareto_axis_limits_sensible():
    import matplotlib

    matplotlib.use("Agg")
    queries = _two_queries()
    ax = plot_pareto(queries)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    for q in queries:
        assert x_lo < q.p[0] < x_hi
        assert y_lo < q.p[1] < y_hi


# ---------------------------------------------------------------------------
# plot_pareto_result
# ---------------------------------------------------------------------------


def _simple_pareto_result():
    return ParetoResult(
        lower_points=[[0.0, 0.0], [0.7, 0.0], [0.5, 0.5], [0.0, 0.9]],
        upper_points=[[0.0, 0.0], [0.8, 0.0], [0.6, 0.6], [0.0, 1.0]],
        property_labels=["Pmax=? [F a]", "Pmax=? [F b]"],
    )


def test_plot_pareto_result_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    ax = plot_pareto_result(_simple_pareto_result())
    assert ax is not None


def test_plot_pareto_result_axis_limits_cover_points():
    import matplotlib

    matplotlib.use("Agg")
    result = _simple_pareto_result()
    ax = plot_pareto_result(result)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    for p in result.lower_points + result.upper_points:
        assert x_lo <= p[0] <= x_hi
        assert y_lo <= p[1] <= y_hi


def test_plot_pareto_result_uses_property_labels():
    import matplotlib

    matplotlib.use("Agg")
    result = _simple_pareto_result()
    ax = plot_pareto_result(result)
    labels = result.property_labels
    assert labels is not None
    assert ax.get_xlabel() == labels[0]
    assert ax.get_ylabel() == labels[1]


def test_plot_pareto_result_label_override():
    import matplotlib

    matplotlib.use("Agg")
    ax = plot_pareto_result(_simple_pareto_result(), labels=("X", "Y"))
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"


def test_plot_pareto_result_raises_on_wrong_dimension():
    result = ParetoResult(
        lower_points=[[0.1, 0.2, 0.3]],
        upper_points=[[0.2, 0.3, 0.4]],
    )
    with pytest.raises(ValueError, match="2-objective"):
        plot_pareto_result(result)


def test_plot_pareto_result_raises_on_empty():
    with pytest.raises(ValueError, match="no points"):
        plot_pareto_result(ParetoResult(lower_points=[], upper_points=[]))


def test_pareto_result_plot_method_delegates():
    """ParetoResult.plot() should call plot_pareto_result and return an Axes."""
    import matplotlib
    import matplotlib.axes

    matplotlib.use("Agg")
    ax = _simple_pareto_result().plot()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_model_checking_multi_returns_pareto_result():
    """End-to-end: model_checking with a multi() property returns a ParetoResult."""
    pytest.importorskip("stormpy")
    import stormvogel.model
    import stormvogel

    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    init = mdp.new_state(labels=["init"], friendly_name="init")
    s1 = mdp.new_state(labels=["T1", "done"], friendly_name="s1")
    s2 = mdp.new_state(labels=["T2", "done"], friendly_name="s2")
    a, b = mdp.action("a"), mdp.action("b")
    init.set_choices({a: [(1.0, s1)], b: [(1.0, s2)]})
    s1.set_choices({a: [(1.0, s1)]})
    s2.set_choices({b: [(1.0, s2)]})

    result = stormvogel.model_checking(mdp, 'multi(Pmax=? [F "T1"], Pmax=? [F "T2"])')
    assert isinstance(result, ParetoResult)
    assert result.property_labels is not None
    assert len(result.property_labels) == 2
    assert result.lower_points is not None
    assert result.upper_points is not None


# ---------------------------------------------------------------------------
# explore_pareto (requires stormpy)
# ---------------------------------------------------------------------------


def _ep_choice_mdp():
    """Two deterministic actions; s0 -a-> s1(T1,done), s0 -b-> s2(T2,done)."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T1", "done"], friendly_name="s1")
    s2 = mdp.new_state(labels=["T2", "done"], friendly_name="s2")
    a, b = mdp.action("a"), mdp.action("b")
    s0.set_choices({a: [(1.0, s1)], b: [(1.0, s2)]})
    s1.set_choices({a: [(1.0, s1)]})
    s2.set_choices({b: [(1.0, s2)]})
    return mdp


def _ep_two_target_mdp():
    """Single action; prob 0.6 to T1, 0.4 to T2, then done."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T1"], friendly_name="s1")
    s2 = mdp.new_state(labels=["T2"], friendly_name="s2")
    s3 = mdp.new_state(labels=["done"], friendly_name="s3")
    a = mdp.action("a")
    s0.set_choices({a: [(0.6, s1), (0.4, s2)]})
    s1.set_choices({a: [(1.0, s3)]})
    s2.set_choices({a: [(1.0, s3)]})
    s3.set_choices({a: [(1.0, s3)]})
    return mdp


def test_explore_pareto_returns_correct_number_of_queries():
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    queries = explore_pareto(_ep_choice_mdp(), ["T1", "T2"], weights)
    assert len(queries) == len(weights)


def test_explore_pareto_queries_are_pareto_query_instances():
    pytest.importorskip("stormpy")
    queries = explore_pareto(_ep_choice_mdp(), ["T1", "T2"], [(1.0, 0.0), (0.0, 1.0)])
    assert all(isinstance(q, ParetoQuery) for q in queries)


def test_explore_pareto_weight_vectors_preserved():
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.0, 1.0)]
    queries = explore_pareto(_ep_choice_mdp(), ["T1", "T2"], weights)
    assert queries[0].w == (1.0, 0.0)
    assert queries[1].w == (0.0, 1.0)


def test_explore_pareto_achievable_points_in_unit_square():
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    queries = explore_pareto(_ep_choice_mdp(), ["T1", "T2"], weights)
    for q in queries:
        assert 0.0 <= q.p[0] <= 1.0
        assert 0.0 <= q.p[1] <= 1.0


def test_explore_pareto_extreme_weights_give_extreme_points():
    """Weight (1,0) maximises P(T1); weight (0,1) maximises P(T2)."""
    pytest.importorskip("stormpy")
    queries = explore_pareto(_ep_choice_mdp(), ["T1", "T2"], [(1.0, 0.0), (0.0, 1.0)])
    q_t1, q_t2 = queries
    assert q_t1.p == pytest.approx((1.0, 0.0), abs=1e-6)
    assert q_t2.p == pytest.approx((0.0, 1.0), abs=1e-6)


def test_explore_pareto_no_choice_mdp_same_point_every_weight():
    """When the MDP has no real choice, all weights give the same achievable point."""
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    queries = explore_pareto(_ep_two_target_mdp(), ["T1", "T2"], weights)
    for q in queries:
        assert q.p == pytest.approx((0.6, 0.4), abs=1e-6)


def test_explore_pareto_wrong_target_count_raises():
    with pytest.raises(ValueError, match="exactly 2"):
        explore_pareto(_ep_choice_mdp(), ["T1", "T2", "T3"], [(1.0, 0.0)])
