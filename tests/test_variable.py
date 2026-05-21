import pytest
import stormvogel.model
from stormvogel.model.variable import (
    BoolDomain,
    CategoricalDomain,
    IntDomain,
    Predicate,
    Variable,
)


# --- IntDomain ---


def test_int_domain_contains_valid():
    d = IntDomain(0, 5)
    assert d.contains(0)
    assert d.contains(3)
    assert d.contains(5)


def test_int_domain_contains_out_of_range():
    d = IntDomain(0, 5)
    assert not d.contains(-1)
    assert not d.contains(6)


def test_int_domain_rejects_bool():
    d = IntDomain(0, 1)
    assert not d.contains(True)
    assert not d.contains(False)


def test_int_domain_none_disallowed():
    d = IntDomain(0, 5)
    assert not d.contains(None)


def test_int_domain_none_allowed():
    d = IntDomain(0, 5, allow_none=True)
    assert d.contains(None)
    assert d.contains(3)


# --- BoolDomain ---


def test_bool_domain_contains_valid():
    d = BoolDomain()
    assert d.contains(True)
    assert d.contains(False)


def test_bool_domain_rejects_int():
    d = BoolDomain()
    assert not d.contains(0)
    assert not d.contains(1)


def test_bool_domain_none_disallowed():
    d = BoolDomain()
    assert not d.contains(None)


def test_bool_domain_none_allowed():
    d = BoolDomain(allow_none=True)
    assert d.contains(None)
    assert d.contains(True)


# --- CategoricalDomain ---


def test_categorical_domain_contains_valid():
    d = CategoricalDomain(("N", "S", "E", "W"))
    assert d.contains("N")
    assert d.contains("W")


def test_categorical_domain_rejects_unknown():
    d = CategoricalDomain(("N", "S", "E", "W"))
    assert not d.contains("X")
    assert not d.contains(0)


def test_categorical_domain_none_disallowed():
    d = CategoricalDomain(("open", "closed"))
    assert not d.contains(None)


def test_categorical_domain_none_allowed():
    d = CategoricalDomain(("open", "closed"), allow_none=True)
    assert d.contains(None)
    assert d.contains("open")


# --- Variable equality is label-only ---


def test_variable_equality_ignores_domain():
    assert Variable("x") == Variable("x", IntDomain(0, 4))
    assert Variable("x", BoolDomain()) == Variable("x")


def test_variable_hash_ignores_domain():
    d = {Variable("x"): 1}
    assert d[Variable("x", IntDomain(0, 4))] == 1


# --- Enforcement in state.add_valuation ---


def test_add_valuation_valid():
    mdp = stormvogel.model.new_mdp()
    x = Variable("x", IntDomain(0, 3))
    mdp.initial_state.add_valuation(x, 2)
    assert mdp.initial_state.get_valuation(x) == 2


def test_add_valuation_out_of_range_raises():
    mdp = stormvogel.model.new_mdp()
    x = Variable("x", IntDomain(0, 3))
    with pytest.raises(ValueError, match="not in domain"):
        mdp.initial_state.add_valuation(x, 5)


def test_add_valuation_none_disallowed_raises():
    mdp = stormvogel.model.new_mdp()
    door = Variable("door", BoolDomain())
    with pytest.raises(ValueError, match="not in domain"):
        mdp.initial_state.add_valuation(door, None)


def test_add_valuation_none_allowed():
    mdp = stormvogel.model.new_mdp()
    door = Variable("door", BoolDomain(allow_none=True))
    mdp.initial_state.add_valuation(door, None)
    assert mdp.initial_state.get_valuation(door) is None


def test_add_valuation_no_domain_always_valid():
    mdp = stormvogel.model.new_mdp()
    x = Variable("x")
    mdp.initial_state.add_valuation(x, "anything")
    assert mdp.initial_state.get_valuation(x) == "anything"


# --- Enforcement in state.valuations setter ---


def test_valuations_setter_valid():
    mdp = stormvogel.model.new_mdp()
    x = Variable("x", IntDomain(0, 4))
    y = Variable("y", BoolDomain())
    mdp.initial_state.valuations = {x: 2, y: True}
    assert mdp.initial_state.get_valuation(x) == 2


def test_valuations_setter_invalid_raises():
    mdp = stormvogel.model.new_mdp()
    x = Variable("x", IntDomain(0, 4))
    with pytest.raises(ValueError, match="not in domain"):
        mdp.initial_state.valuations = {x: 99}


def test_categorical_valuation_enforcement():
    mdp = stormvogel.model.new_mdp()
    direction = Variable("dir", CategoricalDomain(("N", "S", "E", "W")))
    mdp.initial_state.add_valuation(direction, "N")
    with pytest.raises(ValueError, match="not in domain"):
        mdp.initial_state.add_valuation(direction, "X")


# --- Predicate ---


def test_predicate_label_only_equality():
    p1 = Predicate("cangonorth", BoolDomain())
    p2 = Predicate("cangonorth", IntDomain(0, 1))
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_predicate_different_labels_not_equal():
    p1 = Predicate("cangonorth", BoolDomain())
    p2 = Predicate("cangosouth", BoolDomain())
    assert p1 != p2


def test_predicate_expr_not_compared():
    p1 = Predicate("p", BoolDomain(), expr=lambda vals: True)
    p2 = Predicate("p", BoolDomain(), expr=lambda vals: False)
    assert p1 == p2


def test_predicate_expr_callable():
    x = Variable("x", IntDomain(0, 10))
    threshold = Predicate("big", BoolDomain(), expr=lambda vals: vals[x] > 5)
    assert threshold.expr is not None
    assert threshold.expr({x: 7}) is True
    assert threshold.expr({x: 3}) is False


def test_predicate_expr_none_allowed():
    p = Predicate("imported", IntDomain(0, 7))
    assert p.expr is None


def test_predicate_repr():
    p = Predicate("cangonorth", BoolDomain())
    assert "cangonorth" in repr(p)


# --- Predicate domain enforcement in new_observation ---


def test_new_observation_predicate_valid():
    pomdp = stormvogel.model.new_pomdp()
    cangonorth = Predicate("cangonorth", BoolDomain())
    obs = pomdp.new_observation("o0", {cangonorth: True})
    assert pomdp.observation_valuations[obs][cangonorth] is True


def test_new_observation_predicate_out_of_domain_raises():
    pomdp = stormvogel.model.new_pomdp()
    fuelmeter = Predicate("fuelmeter", IntDomain(0, 3))
    with pytest.raises(ValueError, match="not in domain"):
        pomdp.new_observation("o0", {fuelmeter: 99})


def test_new_observation_variable_domain_enforced():
    pomdp = stormvogel.model.new_pomdp()
    x = Variable("x", IntDomain(0, 5))
    with pytest.raises(ValueError, match="not in domain"):
        pomdp.new_observation("o0", {x: 10})


def test_new_observation_mixed_keys():
    pomdp = stormvogel.model.new_pomdp()
    x = Variable("x", IntDomain(0, 5))
    p = Predicate("big", BoolDomain())
    obs = pomdp.new_observation("o0", {x: 3, p: False})
    assert pomdp.observation_valuations[obs][x] == 3
    assert pomdp.observation_valuations[obs][p] is False
