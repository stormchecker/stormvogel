"""Tests for compute_weighted_reachability_policy and evaluate_policy_reachability (requires stormpy)."""

import pytest
import stormvogel.model
from stormvogel.teaching.multiobjective import (
    compute_weighted_reachability_policy,
    evaluate_policy_reachability,
)


def _chain_mdp():
    """Single-action chain: s0 -a,0.7-> s1(T1) -a,1-> s2(done) / s0 -a,0.3-> s2(done).

    Unique optimal value at initial state: 0.7 * w.
    """
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T1"], friendly_name="s1")
    s2 = mdp.new_state(labels=["done"], friendly_name="s2")
    a = mdp.action("a")
    s0.set_choices({a: [(0.7, s1), (0.3, s2)]})
    s1.set_choices({a: [(1.0, s2)]})
    s2.set_choices({a: [(1.0, s2)]})
    return mdp


def _choice_mdp():
    """Two deterministic actions from s0, each leading directly to a terminal state.

    s0 -a,1-> s1(T1,done)   weight 5
    s0 -b,1-> s2(T2,done)   weight 2
    Optimal: choose 'a', value = 5.
    """
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"], friendly_name="s0")
    s1 = mdp.new_state(labels=["T1", "done"], friendly_name="s1")
    s2 = mdp.new_state(labels=["T2", "done"], friendly_name="s2")
    a = mdp.action("a")
    b = mdp.action("b")
    s0.set_choices({a: [(1.0, s1)], b: [(1.0, s2)]})
    s1.set_choices({a: [(1.0, s1)]})
    s2.set_choices({b: [(1.0, s2)]})
    return mdp


def _two_target_mdp():
    """Two independent targets; prob 0.6 to T1, prob 0.4 to T2, then done.

    s0 -a,0.6-> s1(T1) -a,1-> s3(done)
       -a,0.4-> s2(T2) -a,1-> s3(done)
    Expected weighted reward: 0.6 * w1 + 0.4 * w2.
    """
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_target_value():
    """Expected reward at initial state equals weight * P(reach T1)."""
    pytest.importorskip("stormpy")
    mdp = _chain_mdp()
    result = compute_weighted_reachability_policy(mdp, ["T1"], [5.0])
    val = result.at_init()
    assert val == pytest.approx(3.5, abs=1e-6)


def test_single_target_scheduler_present():
    """A scheduler is returned."""
    pytest.importorskip("stormpy")
    result = compute_weighted_reachability_policy(_chain_mdp(), ["T1"], [5.0])
    assert result.scheduler is not None


def test_policy_picks_higher_weight_action():
    """Policy chooses the action that maximises weighted reward."""
    pytest.importorskip("stormpy")
    mdp = _choice_mdp()
    result = compute_weighted_reachability_policy(mdp, ["T1", "T2"], [5.0, 2.0])
    init = result.model.initial_state
    assert result.at(init) == pytest.approx(5.0, abs=1e-6)
    assert result.scheduler is not None
    chosen = result.scheduler.get_action_at_state(init)
    assert chosen.label == "a"


def test_policy_picks_lower_action_when_weight_reversed():
    """When weights are reversed, policy should choose 'b'."""
    pytest.importorskip("stormpy")
    mdp = _choice_mdp()
    result = compute_weighted_reachability_policy(mdp, ["T1", "T2"], [2.0, 5.0])
    init = result.model.initial_state
    assert result.at(init) == pytest.approx(5.0, abs=1e-6)
    assert result.scheduler is not None
    chosen = result.scheduler.get_action_at_state(init)
    assert chosen.label == "b"


def test_two_targets_value():
    """Expected reward matches 0.6 * w1 + 0.4 * w2 for independent targets."""
    pytest.importorskip("stormpy")
    w1, w2 = 3.0, 4.0
    mdp = _two_target_mdp()
    result = compute_weighted_reachability_policy(mdp, ["T1", "T2"], [w1, w2])
    expected = 0.6 * w1 + 0.4 * w2
    val = result.at_init()
    assert val == pytest.approx(expected, abs=1e-6)


def test_zero_weight_target_ignored():
    """A target with weight 0 contributes nothing to the objective."""
    pytest.importorskip("stormpy")
    mdp = _choice_mdp()
    result = compute_weighted_reachability_policy(mdp, ["T1", "T2"], [3.0, 0.0])
    init = result.model.initial_state
    assert result.at(init) == pytest.approx(3.0, abs=1e-6)
    assert result.scheduler is not None
    assert result.scheduler.get_action_at_state(init).label == "a"


# ---------------------------------------------------------------------------
# evaluate_policy_reachability
# ---------------------------------------------------------------------------


def test_evaluate_single_target_prob():
    """Induced DTMC reachability matches the known P(reach T1) = 0.7."""
    pytest.importorskip("stormpy")
    result = compute_weighted_reachability_policy(_chain_mdp(), ["T1"], [1.0])
    probs = evaluate_policy_reachability(result, ["T1"])
    assert probs == [pytest.approx(0.7, abs=1e-6)]


def test_evaluate_returns_vector_length():
    """Output length equals the number of target labels queried."""
    pytest.importorskip("stormpy")
    result = compute_weighted_reachability_policy(
        _two_target_mdp(), ["T1", "T2"], [1.0, 1.0]
    )
    probs = evaluate_policy_reachability(result, ["T1", "T2"])
    assert len(probs) == 2


def test_evaluate_two_targets_probs():
    """Independent targets: P(reach T1)=0.6, P(reach T2)=0.4."""
    pytest.importorskip("stormpy")
    result = compute_weighted_reachability_policy(
        _two_target_mdp(), ["T1", "T2"], [1.0, 1.0]
    )
    probs = evaluate_policy_reachability(result, ["T1", "T2"])
    assert probs[0] == pytest.approx(0.6, abs=1e-6)
    assert probs[1] == pytest.approx(0.4, abs=1e-6)


def test_evaluate_policy_that_avoids_target():
    """Policy optimising for T1 only never reaches T2 → P(reach T2) = 0."""
    pytest.importorskip("stormpy")
    result = compute_weighted_reachability_policy(
        _choice_mdp(), ["T1", "T2"], [1.0, 0.0]
    )
    probs = evaluate_policy_reachability(result, ["T1", "T2"])
    assert probs[0] == pytest.approx(1.0, abs=1e-6)
    assert probs[1] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_probs_consistent_with_weighted_value():
    """w · probs should equal the model-checked expected reward."""
    pytest.importorskip("stormpy")
    w1, w2 = 3.0, 4.0
    result = compute_weighted_reachability_policy(
        _two_target_mdp(), ["T1", "T2"], [w1, w2]
    )
    probs = evaluate_policy_reachability(result, ["T1", "T2"])
    weighted_sum = w1 * probs[0] + w2 * probs[1]
    raw = result.at_init()
    assert isinstance(raw, (int, float))
    expected_reward = float(raw)
    assert weighted_sum == pytest.approx(expected_reward, abs=1e-5)
