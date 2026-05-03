"""Tests for stormvogel.teaching.policy_iteration."""

import sympy as sp

import stormvogel.model as model
import stormvogel.examples as examples
from stormvogel.teaching.policy_iteration import (
    PI,
    initial_scheduler,
    policy_improvement,
)


def _simple_mdp():
    """Three-state MDP where every state is reachable under any scheduler.

    s0: a -> s1 (prob 1),  b -> s1 (0.5) + target (0.5)
    s1: a -> target (1),   b -> s0 (0.5) + target (0.5)
    target: self-loop (absorbing)

    Max reachability from s0: pick b from s0 (0.5) then always a from s1 => 1.
    Under scheduler (s0:a, s1:a): s0 -> s1 -> target, so s0=1, s1=1.
    Min reachability: pick a from s0 (goes to s1), then b from s1 loops back;
    that traps the system but s1->target still has 0.5 via b, so min s1=2/3,
    min s0 under a = min s1 = 2/3, but under b = 0.5 + 0.5*1... let's just
    verify convergence and correct sign of values.
    """
    mdp = model.new_mdp()
    s_target = mdp.new_state(friendly_name="target", labels=["target"])
    s0 = mdp.initial_state
    s0.set_friendly_name("s0")
    s1 = mdp.new_state(friendly_name="s1")

    act_a = mdp.action("a")
    act_b = mdp.action("b")

    s_target.set_choices({act_a: [(1.0, s_target)]})
    s0.set_choices(
        {
            act_a: [(1.0, s1)],
            act_b: [(0.5, s1), (0.5, s_target)],
        }
    )
    s1.set_choices(
        {
            act_a: [(1.0, s_target)],
            act_b: [(0.5, s0), (0.5, s_target)],
        }
    )
    return mdp, s0, s1, s_target


def test_pi_maximize_converges():
    mdp, s0, s1, s_target = _simple_mdp()
    pi = PI(mdp, "target", minimize=False)

    for _ in range(20):
        if pi.has_converged():
            break
        pi.step()

    assert pi.has_converged()
    values = pi.current_values
    assert values is not None
    # Every state can reach target with probability 1.
    assert values[s_target] == sp.Integer(1)
    assert values[s0] == sp.Integer(1)
    assert values[s1] == sp.Integer(1)


def test_pi_minimize_converges():
    """Min reachability: s0 -a-> s1, s1 -b-> loop or target.

    Under minimising scheduler:
      s1: b gives 0.5*s0 + 0.5*1  =>  s1 = 0.5*s0 + 0.5
          a gives 1  =>  min picks b when 0.5*s0+0.5 < 1, i.e. s0 < 1.
      s0: a gives s1, b gives 0.5*s1 + 0.5  =>  min picks whichever is smaller.

    Solving s0=s1 (symmetric fixed point under b-from-s0, b-from-s1):
      s0 = 0.5*s1 + 0.5,  s1 = 0.5*s0 + 0.5
      => s0 = 0.5*(0.5*s0 + 0.5) + 0.5 = 0.25*s0 + 0.75
      => 0.75*s0 = 0.75  => s0 = 1.
    Both b actions give value 1, so min is also 1 (every policy reaches target
    with prob 1 from this MDP).  PI must converge and all values == 1.
    """
    mdp, s0, s1, s_target = _simple_mdp()
    pi = PI(mdp, "target", minimize=True)

    for _ in range(20):
        if pi.has_converged():
            break
        pi.step()

    assert pi.has_converged()
    values = pi.current_values
    assert values is not None
    assert values[s_target] == sp.Integer(1)
    assert values[s0] == sp.Integer(1)
    assert values[s1] == sp.Integer(1)


def _parker_setup():
    mdp = examples.create_parker_mdp()
    s0 = next(iter(mdp.get_states_with_label("s0")))
    s1 = next(iter(mdp.get_states_with_label("s1")))
    s2 = next(iter(mdp.get_states_with_label("target")))
    s3 = next(iter(mdp.get_states_with_label("s3")))
    return mdp, s0, s1, s2, s3


def test_pi_parker_maximize():
    mdp, s0, s1, s2, s3 = _parker_setup()
    pi = PI(mdp, "target", minimize=False)

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


def test_pi_parker_minimize():
    mdp, s0, s1, s2, s3 = _parker_setup()
    pi = PI(mdp, "target", minimize=True)

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
    sched0 = initial_scheduler(mdp, "target", minimize=False)
    values = {s: sp.Integer(0) for s in mdp.states}
    values[s2] = sp.Integer(1)

    scheduler = policy_improvement(
        mdp, values, one_states=[s2], minimize=False, current_scheduler=sched0
    )
    # From s3, action a goes directly to s2 (value 1); action b self-loops (value 0).
    action_s3 = scheduler.get_action_at_state(s3)
    assert action_s3.label == "a"
