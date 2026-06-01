"""Tests for stormvogel.model.fingerprint."""

from fractions import Fraction

import stormvogel.model as m


def _simple_mdp() -> m.Model:
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"], friendly_name="s0")
    s1 = mdp.new_state(["target"], friendly_name="s1")
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(Fraction(1), s1)]})
    mdp.set_choices(s1, {a: [(Fraction(1), s1)]})
    return mdp


def test_identical_models_same_hash():
    assert _simple_mdp().semantic_hash() == _simple_mdp().semantic_hash()


def test_hash_is_hex_string():
    h = _simple_mdp().semantic_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_different_labels_different_hash():
    mdp1 = _simple_mdp()
    mdp2 = m.new_mdp(create_initial_state=False)
    s0 = mdp2.new_state(["init"], friendly_name="s0")
    s1 = mdp2.new_state(["other"], friendly_name="s1")  # different label
    a = mdp2.new_action("go")
    mdp2.set_choices(s0, {a: [(Fraction(1), s1)]})
    mdp2.set_choices(s1, {a: [(Fraction(1), s1)]})
    assert mdp1.semantic_hash() != mdp2.semantic_hash()


def test_different_probabilities_different_hash():
    mdp1 = m.new_mdp(create_initial_state=False)
    s0 = mdp1.new_state(["init"])
    s1 = mdp1.new_state(["a"])
    s2 = mdp1.new_state(["b"])
    a = mdp1.new_action("go")
    mdp1.set_choices(s0, {a: [(Fraction(1, 2), s1), (Fraction(1, 2), s2)]})
    mdp1.set_choices(s1, {a: [(Fraction(1), s1)]})
    mdp1.set_choices(s2, {a: [(Fraction(1), s2)]})

    mdp2 = m.new_mdp(create_initial_state=False)
    t0 = mdp2.new_state(["init"])
    t1 = mdp2.new_state(["a"])
    t2 = mdp2.new_state(["b"])
    b = mdp2.new_action("go")
    mdp2.set_choices(t0, {b: [(Fraction(3, 4), t1), (Fraction(1, 4), t2)]})
    mdp2.set_choices(t1, {b: [(Fraction(1), t1)]})
    mdp2.set_choices(t2, {b: [(Fraction(1), t2)]})

    assert mdp1.semantic_hash() != mdp2.semantic_hash()


def test_different_action_label_different_hash():
    mdp1 = _simple_mdp()
    mdp2 = m.new_mdp(create_initial_state=False)
    s0 = mdp2.new_state(["init"], friendly_name="s0")
    s1 = mdp2.new_state(["target"], friendly_name="s1")
    a = mdp2.new_action("stay")  # different action label
    mdp2.set_choices(s0, {a: [(Fraction(1), s1)]})
    mdp2.set_choices(s1, {a: [(Fraction(1), s1)]})
    assert mdp1.semantic_hash() != mdp2.semantic_hash()


def test_different_model_type_different_hash():
    dtmc = m.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(["init"], friendly_name="s0")
    s1 = dtmc.new_state(["target"], friendly_name="s1")
    dtmc.set_choices(s0, [(Fraction(1), s1)])
    dtmc.set_choices(s1, [(Fraction(1), s1)])

    mdp = _simple_mdp()
    assert dtmc.semantic_hash() != mdp.semantic_hash()


def test_reward_model_affects_hash():
    mdp1 = _simple_mdp()
    mdp2 = _simple_mdp()
    rm = mdp2.new_reward_model("cost")
    rm.set_state_reward(mdp2.states[0], Fraction(5))
    assert mdp1.semantic_hash() != mdp2.semantic_hash()


def test_state_order_matters():
    """Swapping the order of states changes the hash (indices differ)."""
    mdp1 = m.new_mdp(create_initial_state=False)
    s0 = mdp1.new_state(["init", "a"])
    s1 = mdp1.new_state(["b"])
    a = mdp1.new_action("go")
    mdp1.set_choices(s0, {a: [(Fraction(1), s1)]})
    mdp1.set_choices(s1, {a: [(Fraction(1), s0)]})

    mdp2 = m.new_mdp(create_initial_state=False)
    t0 = mdp2.new_state(["b"])  # reversed order
    t1 = mdp2.new_state(["init", "a"])
    b = mdp2.new_action("go")
    mdp2.set_choices(t1, {b: [(Fraction(1), t0)]})
    mdp2.set_choices(t0, {b: [(Fraction(1), t1)]})

    assert mdp1.semantic_hash() != mdp2.semantic_hash()


def test_pomdp_observations_affect_hash():
    pomdp1 = m.new_pomdp(create_initial_state=False)
    obs_a = pomdp1.new_observation("A")
    obs_b = pomdp1.new_observation("B")
    s0 = pomdp1.new_state(["init"], observation=obs_a)
    s1 = pomdp1.new_state(["t"], observation=obs_b)
    a = pomdp1.new_action("go")
    pomdp1.set_choices(s0, {a: [(Fraction(1), s1)]})
    pomdp1.set_choices(s1, {a: [(Fraction(1), s1)]})

    pomdp2 = m.new_pomdp(create_initial_state=False)
    obs_c = pomdp2.new_observation("C")  # different alias
    obs_d = pomdp2.new_observation("D")
    t0 = pomdp2.new_state(["init"], observation=obs_c)
    t1 = pomdp2.new_state(["t"], observation=obs_d)
    b = pomdp2.new_action("go")
    pomdp2.set_choices(t0, {b: [(Fraction(1), t1)]})
    pomdp2.set_choices(t1, {b: [(Fraction(1), t1)]})

    assert pomdp1.semantic_hash() != pomdp2.semantic_hash()


def test_pomdp_same_observations_same_hash():
    def _make():
        pomdp = m.new_pomdp(create_initial_state=False)
        obs_a = pomdp.new_observation("A")
        obs_b = pomdp.new_observation("B")
        s0 = pomdp.new_state(["init"], observation=obs_a)
        s1 = pomdp.new_state(["t"], observation=obs_b)
        a = pomdp.new_action("go")
        pomdp.set_choices(s0, {a: [(Fraction(1), s1)]})
        pomdp.set_choices(s1, {a: [(Fraction(1), s1)]})
        return pomdp

    assert _make().semantic_hash() == _make().semantic_hash()
