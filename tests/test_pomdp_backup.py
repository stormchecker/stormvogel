"""Tests for stormvogel.teaching.pomdp_backup.

Uses the 4-state reachability POMDP (belief space is 1-D: p = Pr(s1)).

Closed-form values used for assertions:
  V_1(p) = 0.5·p + 0.2          (action a always wins at horizon 1)
  V_2(p) = max(0.5·p + 0.2,  0.25·p + 0.35)   (crossover at p = 0.6)
"""

import pytest
from fractions import Fraction

from stormvogel.examples.four_state_reachability import create_4state_reachability_pomdp
from stormvogel.teaching.pomdp_backup import (
    AlphaVI,
    AlphaVector,
    BeliefBackupOperator,
    ExactBeliefBackupOperator,
    dot,
    initial_alpha,
    make_operator_pomdp_maxreachprob,
    make_operator_pomdp_maxreachprob_exact,
    value_function,
    mdp_bound_alpha,
    qmdp_alphas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return create_4state_reachability_pomdp()


def _state(model, name):
    return next(s for s in model.states if model.friendly_names.get(s) == name)


@pytest.fixture(scope="module")
def s1(model):
    return _state(model, "s1")


@pytest.fixture(scope="module")
def s2(model):
    return _state(model, "s2")


@pytest.fixture(scope="module")
def target(model):
    return _state(model, "target")


@pytest.fixture(scope="module")
def sink(model):
    return _state(model, "sink")


def _belief(s1, s2, p: Fraction):
    return {s1: p, s2: 1 - p}


# ---------------------------------------------------------------------------
# initial_alpha
# ---------------------------------------------------------------------------


def test_initial_alpha_target_is_one(model, target):
    assert initial_alpha(model, "target").values[target] == Fraction(1)


def test_initial_alpha_non_target_is_zero(model, s1, s2, sink):
    alpha = initial_alpha(model, "target")
    for s in (s1, s2, sink):
        assert alpha.values[s] == Fraction(0)


def test_initial_alpha_is_leaf(model):
    alpha = initial_alpha(model, "target")
    assert alpha.action is None
    assert alpha.successors == {}


# ---------------------------------------------------------------------------
# dot / value_function
# ---------------------------------------------------------------------------


def test_dot_non_target_belief_is_zero(model, s1, s2):
    alpha = initial_alpha(model, "target")
    assert dot(alpha, _belief(s1, s2, Fraction(1, 2))) == Fraction(0)


def test_dot_point_mass_target(model, target):
    alpha = initial_alpha(model, "target")
    assert dot(alpha, {target: Fraction(1)}) == Fraction(1)


def test_value_function_single_alpha_zero(model, s1, s2):
    alphas = [initial_alpha(model, "target")]
    assert value_function(alphas, _belief(s1, s2, Fraction(1, 2))) == Fraction(0)


# ---------------------------------------------------------------------------
# make_operator_pomdp_maxreachprob — errors
# ---------------------------------------------------------------------------


def test_raises_for_non_pomdp():
    import stormvogel.model as sv_model

    with pytest.raises(ValueError, match="POMDP"):
        make_operator_pomdp_maxreachprob(sv_model.new_mdp(), "target", beliefs=[])


# ---------------------------------------------------------------------------
# One-step backup  (V_1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alpha_1(model, s1, s2):
    """Alpha vector produced by one backup step at belief p = 1/2."""
    op = make_operator_pomdp_maxreachprob(
        model, "target", beliefs=[_belief(s1, s2, Fraction(1, 2))]
    )
    return op.apply([initial_alpha(model, "target")])[0]


def test_one_step_action_is_a(alpha_1):
    # At horizon 1, action b yields 0 (no progress without a commit); action a wins.
    assert alpha_1.action.label == "a"


def test_one_step_s1_value(alpha_1, s1):
    assert alpha_1.values[s1] == Fraction(7, 10)


def test_one_step_s2_value(alpha_1, s2):
    assert alpha_1.values[s2] == Fraction(2, 10)


def test_one_step_target_value(alpha_1, target):
    assert alpha_1.values[target] == Fraction(1)


def test_one_step_successor_keys_match_observations(model, alpha_1):
    obs_aliases = set(model.observation_aliases.values())
    assert set(alpha_1.successors.keys()) == obs_aliases


def test_one_step_successors_are_alpha_vectors(alpha_1):
    for v in alpha_1.successors.values():
        assert isinstance(v, AlphaVector)


# ---------------------------------------------------------------------------
# Two-step backup  (V_2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vi_2(model, s1, s2):
    """AlphaVI after 2 steps, beliefs at p = 1/2 and p = 7/10."""
    beliefs = [
        _belief(s1, s2, Fraction(1, 2)),
        _belief(s1, s2, Fraction(7, 10)),
    ]
    op = make_operator_pomdp_maxreachprob(model, "target", beliefs=beliefs)
    vi = AlphaVI(op, [initial_alpha(model, "target")])
    vi.step()
    vi.step()
    return vi


def test_two_step_value_at_half(vi_2, s1, s2):
    """V_2(1/2) = 0.25·(1/2) + 0.35 = 19/40."""
    b = _belief(s1, s2, Fraction(1, 2))
    assert value_function(vi_2.current_alphas, b) == Fraction(19, 40)


def test_two_step_value_at_seven_tenths(vi_2, s1, s2):
    """V_2(7/10) = 0.5·(7/10) + 0.2 = 11/20."""
    b = _belief(s1, s2, Fraction(7, 10))
    assert value_function(vi_2.current_alphas, b) == Fraction(11, 20)


def test_two_step_action_b_below_crossover(vi_2, s1, s2):
    """p = 1/2 < 0.6 → action b (belief evolution) is optimal."""
    b = _belief(s1, s2, Fraction(1, 2))
    best = max(vi_2.current_alphas, key=lambda α: dot(α, b))
    assert best.action.label == "b"


def test_two_step_action_a_above_crossover(vi_2, s1, s2):
    """p = 7/10 > 0.6 → action a (commit) is optimal."""
    b = _belief(s1, s2, Fraction(7, 10))
    best = max(vi_2.current_alphas, key=lambda α: dot(α, b))
    assert best.action.label == "a"


def test_two_step_equal_at_crossover(vi_2, s1, s2):
    """At p = 3/5 both actions tie; V_2(3/5) = 1/2."""
    b = _belief(s1, s2, Fraction(3, 5))
    assert value_function(vi_2.current_alphas, b) == Fraction(1, 2)


# ---------------------------------------------------------------------------
# AlphaVI step-by-step
# ---------------------------------------------------------------------------


def test_zero_steps_returns_initial(model, s1, s2, target):
    op = make_operator_pomdp_maxreachprob(
        model, "target", beliefs=[_belief(s1, s2, Fraction(1, 2))]
    )
    vi = AlphaVI(op, [initial_alpha(model, "target")])
    assert vi.current_alphas[0].values[target] == Fraction(1)
    assert vi.current_alphas[0].action is None


def test_step_returns_list(model, s1, s2):
    op = make_operator_pomdp_maxreachprob(
        model, "target", beliefs=[_belief(s1, s2, Fraction(1, 2))]
    )
    vi = AlphaVI(op, [initial_alpha(model, "target")])
    result = vi.step()
    assert isinstance(result, list)
    assert all(isinstance(a, AlphaVector) for a in result)


def test_operator_is_belief_backup_operator(model, s1, s2):
    op = make_operator_pomdp_maxreachprob(
        model, "target", beliefs=[_belief(s1, s2, Fraction(1, 2))]
    )
    assert isinstance(op, BeliefBackupOperator)


# ---------------------------------------------------------------------------
# Exact (non-point-based) operator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exact_op(model):
    return make_operator_pomdp_maxreachprob_exact(model, "target")


def test_exact_operator_type(exact_op):
    assert isinstance(exact_op, ExactBeliefBackupOperator)


def test_exact_raises_for_non_pomdp():
    import stormvogel.model as sv_model

    with pytest.raises(ValueError, match="POMDP"):
        make_operator_pomdp_maxreachprob_exact(sv_model.new_mdp(), "target")


def test_exact_one_step_action_is_a(model, exact_op):
    k1 = AlphaVI(exact_op, [initial_alpha(model, "target")]).step()
    assert len(k1) == 1
    assert k1[0].action is not None
    assert k1[0].action.label == "a"


def test_exact_one_step_values(model, exact_op, s1, s2):
    k1 = AlphaVI(exact_op, [initial_alpha(model, "target")]).step()
    assert k1[0].values[s1] == Fraction(7, 10)
    assert k1[0].values[s2] == Fraction(2, 10)


@pytest.fixture(scope="module")
def exact_vi_2(model, exact_op):
    vi = AlphaVI(exact_op, [initial_alpha(model, "target")])
    vi.step()
    vi.step()
    return vi


def test_exact_two_step_count(exact_vi_2):
    """Exact backup yields one alpha per action at k=2."""
    assert len(exact_vi_2.current_alphas) == 2


def test_exact_two_step_actions(exact_vi_2):
    labels = {a.action.label for a in exact_vi_2.current_alphas}
    assert labels == {"a", "b"}


def test_exact_two_step_value_at_half(exact_vi_2, s1, s2):
    b = _belief(s1, s2, Fraction(1, 2))
    assert value_function(exact_vi_2.current_alphas, b) == Fraction(19, 40)


def test_exact_two_step_value_at_seven_tenths(exact_vi_2, s1, s2):
    b = _belief(s1, s2, Fraction(7, 10))
    assert value_function(exact_vi_2.current_alphas, b) == Fraction(11, 20)


def test_exact_two_step_equal_at_crossover(exact_vi_2, s1, s2):
    b = _belief(s1, s2, Fraction(3, 5))
    assert value_function(exact_vi_2.current_alphas, b) == Fraction(1, 2)


def test_exact_three_step_count(model, exact_op):
    vi = AlphaVI(exact_op, [initial_alpha(model, "target")])
    for _ in range(3):
        vi.step()
    assert len(vi.current_alphas) == 3


# ---------------------------------------------------------------------------
# MDP bound and QMDP  (require stormpy)
# ---------------------------------------------------------------------------

_stormpy = pytest.importorskip("stormpy", reason="stormpy required")


@pytest.fixture(scope="module")
def mdp_alpha(model):
    return mdp_bound_alpha(model, "target")


@pytest.fixture(scope="module")
def qmdp(model):
    return qmdp_alphas(model, "target")


def test_mdp_bound_alpha_target_is_one(mdp_alpha, target):
    assert mdp_alpha.values[target] == Fraction(1)


def test_mdp_bound_alpha_sink_is_zero(mdp_alpha, sink):
    assert mdp_alpha.values[sink] == Fraction(0)


def test_mdp_bound_alpha_s1_equals_s2(mdp_alpha, s1, s2):
    """With full observability both states reach V=7/10 via action b."""
    assert mdp_alpha.values[s1] == mdp_alpha.values[s2]
    assert mdp_alpha.values[s1] == Fraction(7, 10)


def test_mdp_bound_value_at_half(mdp_alpha, s1, s2):
    b = _belief(s1, s2, Fraction(1, 2))
    assert dot(mdp_alpha, b) == Fraction(7, 10)


def test_qmdp_one_alpha_per_action(qmdp):
    assert len(qmdp) == 2
    assert {a.action.label for a in qmdp} == {"a", "b"}


def test_qmdp_value_at_half(qmdp, s1, s2):
    b = _belief(s1, s2, Fraction(1, 2))
    assert value_function(qmdp, b) == Fraction(7, 10)


def test_qmdp_upper_bounds_exact_vi(model, qmdp, s1, s2):
    """V_QMDP(b) >= V*(b) for a range of beliefs."""
    op = make_operator_pomdp_maxreachprob_exact(model, "target")
    vi = AlphaVI(op, [initial_alpha(model, "target")])
    for _ in range(5):
        vi.step()
    for p_num in range(1, 10):
        p = Fraction(p_num, 10)
        b = _belief(s1, s2, p)
        assert value_function(qmdp, b) >= value_function(vi.current_alphas, b)


def test_qmdp_leq_mdp_bound(mdp_alpha, qmdp, s1, s2):
    """V_QMDP(b) <= V_MDP(b) for all test beliefs."""
    for p_num in range(1, 10):
        p = Fraction(p_num, 10)
        b = _belief(s1, s2, p)
        assert value_function(qmdp, b) <= dot(mdp_alpha, b)


def test_qmdp_raises_for_non_pomdp():
    import stormvogel.model as sv_model

    with pytest.raises(ValueError, match="POMDP"):
        qmdp_alphas(sv_model.new_mdp(), "target")
