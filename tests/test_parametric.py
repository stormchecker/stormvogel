from fractions import Fraction

import sympy as sp
import pytest

import stormvogel.model
import stormvogel.parametric
import stormvogel.stormpy_utils.mapping as mapping
from model_testing import assert_models_equal


stormpy = pytest.importorskip("stormpy")


def test_pmc_conversion():
    """Round-trip a pMC: build it with sympy transitions, push to stormpy, pull back."""
    pmc = stormvogel.model.new_dtmc()

    init = pmc.initial_state

    # Declare the parameters up front so the model's insertion order is stable.
    x = pmc.declare_parameter("x")
    y = pmc.declare_parameter("y")
    z = pmc.declare_parameter("z")

    p1 = 4 * x * y * z
    p2 = x**2 - x**2 * y**2

    pmc.new_state(labels=["A"])
    pmc.new_state(labels=["B"])

    init.set_choices(
        [
            (p1, next(iter(pmc.get_states_with_label("A")))),
            (p2, next(iter(pmc.get_states_with_label("B")))),
        ]
    )

    pmc.add_self_loops()

    stormpy_pmc = mapping.stormvogel_to_stormpy(pmc)
    new_pmc = mapping.stormpy_to_stormvogel(stormpy_pmc)

    assert_models_equal(pmc, new_pmc)


def test_pmdp_conversion():
    """Round-trip a pMDP with two actions."""
    pmdp = stormvogel.model.new_mdp()

    init = pmdp.initial_state
    x = pmdp.declare_parameter("x")

    p1 = x
    p2 = 1 - x

    goal = pmdp.new_state(labels=["goal"])
    sink = pmdp.new_state(labels=["sink"])

    action_a = pmdp.new_action("a")
    action_b = pmdp.new_action("b")
    branch0 = stormvogel.model.Distribution([(p1, goal), (p2, sink)])
    branch1 = stormvogel.model.Distribution([(p1, sink), (p2, goal)])

    pmdp.add_choices(
        init, stormvogel.model.Choices({action_a: branch0, action_b: branch1})
    )
    pmdp.add_self_loops()

    stormpy_pmdp = mapping.stormvogel_to_stormpy(pmdp)
    new_pmdp = mapping.stormpy_to_stormvogel(stormpy_pmdp)

    assert_models_equal(pmdp, new_pmdp)


def test_pmc_conversion_from_stormpy():
    import stormvogel.examples.stormpy_examples.stormpy_pmc

    stormpy_pmc = (
        stormvogel.examples.stormpy_examples.stormpy_pmc.example_parametric_models_01()
    )
    stormvogel_pmc = stormvogel.mapping.stormpy_to_stormvogel(stormpy_pmc)
    assert stormvogel_pmc is not None
    new_stormpy_pmc = stormvogel.mapping.stormvogel_to_stormpy(stormvogel_pmc)
    new_stormvogel_pmc = stormvogel.mapping.stormpy_to_stormvogel(new_stormpy_pmc)
    assert_models_equal(stormvogel_pmc, new_stormvogel_pmc)


def test_pmc_valuations():
    """Instantiate a pMC with a rational function transition and check the result."""
    pmc = stormvogel.model.new_dtmc()

    init = pmc.initial_state
    x = pmc.declare_parameter("x")
    y = pmc.declare_parameter("y")
    z = pmc.declare_parameter("z")
    w = pmc.declare_parameter("w")

    p1 = 4 * x * z * w**2
    # A rational function: (x^2 - x^2 y^2) / (2 z^2)
    r1 = (x**2 - x**2 * y**2) / (2 * z**2)

    pmc.new_state(labels=["A"])
    pmc.new_state(labels=["B"])

    init.set_choices(
        [
            (p1, next(iter(pmc.get_states_with_label("A")))),
            (r1, next(iter(pmc.get_states_with_label("B")))),
        ]
    )
    pmc.add_self_loops()

    induced_pmc = pmc.get_instantiated_model({"x": 1, "y": 2, "w": 1, "z": 5})

    new_induced_pmc = stormvogel.model.new_dtmc()
    init = new_induced_pmc.initial_state

    new_induced_pmc.new_state(labels=["A"])
    new_induced_pmc.new_state(labels=["B"])

    # p1 at x=1, z=5, w=1 -> 4*1*5*1 = 20
    # r1 at x=1, y=2, z=5 -> (1 - 4) / 50 = -3/50
    init.set_choices(
        [
            (20, next(iter(new_induced_pmc.get_states_with_label("A")))),
            (Fraction(-3, 50), next(iter(new_induced_pmc.get_states_with_label("B")))),
        ]
    )
    new_induced_pmc.add_self_loops()

    assert_models_equal(induced_pmc, new_induced_pmc)


def test_declare_parameter_is_idempotent():
    pmc = stormvogel.model.new_dtmc()
    x1 = pmc.declare_parameter("x")
    x2 = pmc.declare_parameter("x")
    assert x1 is x2
    assert pmc.parameters == ("x",)


def test_undeclared_parameter_raises():
    """Using a symbol that was not declared raises ValueError."""
    pmc = stormvogel.model.new_dtmc()
    x = sp.Symbol("x")
    s = pmc.new_state(labels=["A"])
    t = pmc.new_state(labels=["B"])
    with pytest.raises(ValueError, match="undeclared parameter"):
        pmc.initial_state.set_choices([(x, s), (1 - x, t)])


def test_unused_and_prune_parameters():
    pmc = stormvogel.model.new_dtmc()
    x = pmc.declare_parameter("x")
    pmc.declare_parameter("stale")
    s = pmc.new_state(labels=["A"])
    t = pmc.new_state(labels=["B"])
    pmc.initial_state.set_choices([(x, s), (1 - x, t)])
    assert pmc.unused_parameters() == {"stale"}
    removed = pmc.prune_parameters()
    assert removed == {"stale"}
    assert pmc.parameters == ("x",)


def test_is_parametric_excludes_plain_numbers():
    assert not stormvogel.parametric.is_parametric(1)
    assert not stormvogel.parametric.is_parametric(0.5)
    # sympy natives are recognised by the backend.
    assert stormvogel.parametric.is_parametric(sp.Symbol("x"))
    assert stormvogel.parametric.is_parametric(sp.Symbol("x") + 1)


def test_partial_instantiation_preserves_remaining_parameters():
    """Partially substituting leaves the rest of the parameters in place and
    keeps the model parametric — both in the registry and in is_parametric()."""
    pmc = stormvogel.model.new_dtmc()
    x = pmc.declare_parameter("x")
    y = pmc.declare_parameter("y")
    a = pmc.new_state(labels=["A"])
    b = pmc.new_state(labels=["B"])
    # Two transitions: the first depends only on x, the second mixes x and y.
    pmc.initial_state.set_choices([(x, a), (1 - x + y, b)])
    pmc.add_self_loops()

    # Substitute x; y should remain.
    partial = pmc.get_instantiated_model({"x": Fraction(1, 4)})

    assert partial.parameters == ("y",)
    assert partial.is_parametric()

    # Collect the remaining free symbol names across all transitions out of
    # the initial state — must be exactly {"y"}.
    remaining_names: set[str] = set()
    for _, branch in partial.transitions[partial.initial_state]:
        for v, _ in branch:
            if stormvogel.parametric.is_parametric(v):
                remaining_names |= stormvogel.parametric.free_symbol_names(v)
    assert remaining_names == {"y"}

    # A follow-up full substitution on the remaining parameter should flip
    # is_parametric() back to False and empty the registry.
    full = partial.get_instantiated_model({"y": Fraction(1, 4)})
    assert not full.is_parametric()
    assert full.parameters == ()


def test_instantiation_ignores_unknown_keys():
    """Substituting a parameter that doesn't exist on the model is a no-op."""
    pmc = stormvogel.model.new_dtmc()
    x = pmc.declare_parameter("x")
    s = pmc.new_state(labels=["A"])
    pmc.initial_state.set_choices([(x, s), (1 - x, s)])

    # Pass a bogus key alongside a real one.
    partial = pmc.get_instantiated_model({"x": Fraction(1, 2), "bogus": 99})
    assert partial.parameters == ()
    assert not partial.is_parametric()
