"""Tests for stormvogel.teaching.pareto."""

import numpy as np
import pytest
from stormvogel.result import ParetoResult, plot_pareto_result
from stormvogel.teaching.pareto import (
    ParetoQuery,
    _feasible_polygon,
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
