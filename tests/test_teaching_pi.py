"""Tests for stormvogel.teaching.policy_iteration."""

import sympy as sp

import stormvogel.examples as examples
from stormvogel.teaching.policy_iteration import (
    PI,
    initial_scheduler,
    policy_improvement,
)


def _parker_setup():
    mdp = examples.create_parker_mdp()
    s0 = mdp.get_states_with_label("s0").pop()
    s1 = mdp.get_states_with_label("s1").pop()
    s2 = mdp.get_states_with_label("s2").pop()
    s3 = mdp.get_states_with_label("s3").pop()
    return mdp, s0, s1, s2, s3


def test_pi_maximize_converges():
    mdp, s0, s1, s2, s3 = _parker_setup()
    sched0 = initial_scheduler(mdp, one_states=[s2], minimize=False)
    pi = PI(mdp, sched0, one_states=[s2], minimize=False)

    for _ in range(20):
        if pi.has_converged():
            break
        pi.step()

    assert pi.has_converged()
    values = pi.current_values
    assert values is not None
    # All states can reach s2 with probability 1 under the max scheduler.
    assert values[s0] == sp.Integer(1)
    assert values[s1] == sp.Integer(1)
    assert values[s2] == sp.Integer(1)
    assert values[s3] == sp.Integer(1)


def test_pi_minimize_converges():
    mdp, s0, s1, s2, s3 = _parker_setup()
    # Pessimistic initialisation: prefer actions that stay away from s2.
    sched0 = initial_scheduler(mdp, one_states=[s2], minimize=True)
    pi = PI(mdp, sched0, one_states=[s2], minimize=True)

    for _ in range(20):
        if pi.has_converged():
            break
        pi.step()

    assert pi.has_converged()
    values = pi.current_values
    assert values is not None
    assert values[s2] == sp.Integer(1)
    assert values[s3] == sp.Integer(0)
    assert values[s0] == sp.Rational(2, 3)
    assert values[s1] == sp.Rational(14, 15)


def test_policy_improvement_selects_best_action():
    mdp, s0, s1, s2, s3 = _parker_setup()
    sched0 = initial_scheduler(mdp, one_states=[s2], minimize=False)
    values = {s: sp.Integer(0) for s in mdp.states}
    values[s2] = sp.Integer(1)

    scheduler = policy_improvement(
        mdp, values, one_states=[s2], minimize=False, current_scheduler=sched0
    )
    # From s3, action a goes directly to s2 (value 1); action b self-loops (value 0).
    action_s3 = scheduler.get_action_at_state(s3)
    assert action_s3.label == "a"
