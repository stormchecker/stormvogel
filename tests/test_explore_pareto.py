"""Tests for explore_pareto (requires stormpy)."""

import matplotlib
import pytest
import stormvogel.model
from stormvogel.teaching.pareto import ParetoQuery, explore_pareto

matplotlib.use("Agg")


def _choice_mdp():
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


def _two_target_mdp():
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


def test_returns_correct_number_of_queries():
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    queries = explore_pareto(_choice_mdp(), ["T1", "T2"], weights)
    assert len(queries) == len(weights)


def test_queries_are_pareto_query_instances():
    pytest.importorskip("stormpy")
    queries = explore_pareto(_choice_mdp(), ["T1", "T2"], [(1.0, 0.0), (0.0, 1.0)])
    assert all(isinstance(q, ParetoQuery) for q in queries)


def test_weight_vectors_preserved():
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.0, 1.0)]
    queries = explore_pareto(_choice_mdp(), ["T1", "T2"], weights)
    assert queries[0].w == (1.0, 0.0)
    assert queries[1].w == (0.0, 1.0)


def test_achievable_points_in_unit_square():
    """All probability values must be in [0, 1]."""
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    queries = explore_pareto(_choice_mdp(), ["T1", "T2"], weights)
    for q in queries:
        assert 0.0 <= q.p[0] <= 1.0
        assert 0.0 <= q.p[1] <= 1.0


def test_extreme_weights_give_extreme_points():
    """Weight (1,0) should maximise P(T1); weight (0,1) should maximise P(T2)."""
    pytest.importorskip("stormpy")
    queries = explore_pareto(_choice_mdp(), ["T1", "T2"], [(1.0, 0.0), (0.0, 1.0)])
    q_t1, q_t2 = queries
    assert q_t1.p == pytest.approx((1.0, 0.0), abs=1e-6)
    assert q_t2.p == pytest.approx((0.0, 1.0), abs=1e-6)


def test_no_choice_mdp_same_point_every_weight():
    """When the MDP has no real choice, all weights give the same achievable point."""
    pytest.importorskip("stormpy")
    weights = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    queries = explore_pareto(_two_target_mdp(), ["T1", "T2"], weights)
    for q in queries:
        assert q.p == pytest.approx((0.6, 0.4), abs=1e-6)


def test_wrong_target_count_raises():
    with pytest.raises(ValueError, match="exactly 2"):
        explore_pareto(_choice_mdp(), ["T1", "T2", "T3"], [(1.0, 0.0)])
