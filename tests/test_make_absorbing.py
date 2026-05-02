"""Tests for stormvogel.transformations.make_absorbing."""

import pytest

import stormvogel.model as sv_model
from stormvogel.model.action import EmptyAction
from stormvogel.transformations.make_absorbing import make_absorbing


def _make_dtmc():
    """init -0.5-> a, -0.5-> b; a -> b (prob 1); b -> b (absorbing)."""
    dtmc = sv_model.new_dtmc(create_initial_state=False)
    init = dtmc.new_state(["init"])
    a = dtmc.new_state(["a"])
    b = dtmc.new_state(["b"])
    dtmc.set_choices(init, [(0.5, a), (0.5, b)])
    dtmc.set_choices(a, [(1, b)])
    dtmc.set_choices(b, [(1, b)])
    return dtmc, init, a, b


def _successors(model, state):
    """Return the set of successor states from *state*."""
    result = set()
    for _, branch in model.transitions[state]:
        for _, s in branch:
            result.add(s)
    return result


# ---------------------------------------------------------------------------
# Single state
# ---------------------------------------------------------------------------


def test_single_state_becomes_self_loop():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, a)
    assert _successors(dtmc, a) == {a}


def test_single_state_uses_empty_action():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, a)
    assert dtmc.transitions[a].has_empty_action()


def test_single_state_self_loop_prob_one():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, a)
    branch = dtmc.transitions[a][EmptyAction]
    assert abs(float(branch[a]) - 1.0) < 1e-12


def test_other_states_unchanged():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, a)
    assert _successors(dtmc, init) == {a, b}


# ---------------------------------------------------------------------------
# Set of states
# ---------------------------------------------------------------------------


def test_set_of_states():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, {a, b})
    assert _successors(dtmc, a) == {a}
    assert _successors(dtmc, b) == {b}


def test_empty_set_is_noop():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, set())
    assert _successors(dtmc, a) == {b}


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def test_label_string():
    dtmc, init, a, b = _make_dtmc()
    make_absorbing(dtmc, "a")
    assert _successors(dtmc, a) == {a}


def test_label_targets_all_matching_states():
    dtmc = sv_model.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(["target"])
    s1 = dtmc.new_state(["target"])
    s2 = dtmc.new_state(["other"])
    dtmc.set_choices(s0, [(1, s2)])
    dtmc.set_choices(s1, [(1, s2)])
    dtmc.set_choices(s2, [(1, s2)])

    make_absorbing(dtmc, "target")
    assert _successors(dtmc, s0) == {s0}
    assert _successors(dtmc, s1) == {s1}
    assert _successors(dtmc, s2) == {s2}  # unchanged


def test_unknown_label_raises():
    dtmc, _, _, _ = _make_dtmc()
    with pytest.raises(KeyError):
        make_absorbing(dtmc, "nonexistent")


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------


def test_mdp_state_becomes_self_loop():
    mdp = sv_model.new_mdp(create_initial_state=False)
    s = mdp.new_state(["init"])
    t = mdp.new_state(["target"])
    a = mdp.new_action("go")
    mdp.set_choices(s, {a: [(1, t)]})
    mdp.set_choices(t, [(1, t)])

    make_absorbing(mdp, s)
    assert _successors(mdp, s) == {s}
