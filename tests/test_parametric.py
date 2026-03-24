import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.parametric
import stormvogel.model
import pytest
from model_testing import assert_models_equal


try:
    import stormpy
except ImportError:
    stormpy = None


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_pmc_conversion():
    # Create a new model
    pmc = stormvogel.model.new_dtmc()

    init = pmc.initial_state

    # From the initial state, we have two choices that either bring us to state A or state B
    p1 = stormvogel.parametric.Polynomial(["x", "y", "z"])
    p1.add_term((1, 1, 1), 4)

    # the other transition is a rational function with two polynomials
    p2 = stormvogel.parametric.Polynomial(["x", "y"])
    # p3 = stormvogel.parametric.Polynomial(["z"])
    p2.add_term((2, 0), 1)
    p2.add_term((2, 2), -1)

    # p3.add_term((2,), 2)
    # r1 = stormvogel.parametric.RationalFunction(p2,p3)

    pmc.new_state(labels=["A"])
    pmc.new_state(labels=["B"])

    init.set_choices(
        [
            (p1, next(iter(pmc.get_states_with_label("A")))),
            (p2, next(iter(pmc.get_states_with_label("B")))),
        ]
    )

    # we add self loops to all states with no outgoing choices
    pmc.add_self_loops()

    # we test the mapping
    stormpy_pmc = mapping.stormvogel_to_stormpy(pmc)
    new_pmc = mapping.stormpy_to_stormvogel(stormpy_pmc)

    assert_models_equal(pmc, new_pmc)


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_pmdp_conversion():
    # Create a new model
    pmdp = stormvogel.model.new_mdp()

    init = pmdp.initial_state

    # From the initial state, we have two actions with choices that either bring us to a goal state or sink state

    p1 = stormvogel.parametric.Polynomial(["x"])
    p2 = stormvogel.parametric.Polynomial(["x"])
    p1.add_term((1,), 1)
    p2.add_term((0,), 1)
    p2.add_term((1,), -1)

    goal = pmdp.new_state(labels=["goal"])
    sink = pmdp.new_state(labels=["sink"])

    action_a = pmdp.new_action("a")
    action_b = pmdp.new_action("b")
    branch0 = stormvogel.model.Distribution(
        [
            (p1, goal),
            (p2, sink),
        ]
    )
    branch1 = stormvogel.model.Distribution(
        [
            (p1, sink),
            (p2, goal),
        ]
    )

    pmdp.add_choices(
        init, stormvogel.model.Choices({action_a: branch0, action_b: branch1})
    )

    # we add self loops to all states with no outgoing choices
    pmdp.add_self_loops()

    # we test the mapping
    stormpy_pmdp = mapping.stormvogel_to_stormpy(pmdp)

    new_pmdp = mapping.stormpy_to_stormvogel(stormpy_pmdp)

    assert_models_equal(pmdp, new_pmdp)


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
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


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_pmc_valuations():
    # we build a simple pmc
    pmc = stormvogel.model.new_dtmc()

    init = pmc.initial_state

    # From the initial state, we have two choices that either bring us to state A or state B
    p1 = stormvogel.parametric.Polynomial(["x", "z", "w"])
    p1.add_term((1, 1, 2), 4)

    # the other transition is a rational function with two polynomials
    p2 = stormvogel.parametric.Polynomial(["x", "y"])
    p3 = stormvogel.parametric.Polynomial(["z"])
    p2.add_term((2, 0), 1)
    p2.add_term((2, 2), -1)
    p3.add_term((2,), 2)
    r1 = stormvogel.parametric.RationalFunction(p2, p3)

    pmc.new_state(labels=["A"])
    pmc.new_state(labels=["B"])

    init.set_choices(
        [
            (p1, next(iter(pmc.get_states_with_label("A")))),
            (r1, next(iter(pmc.get_states_with_label("B")))),
        ]
    )

    # we add self loops to all states with no outgoing choices
    pmc.add_self_loops()

    induced_pmc = pmc.get_instantiated_model({"x": 1, "y": 2, "w": 1, "z": 5})

    # we build what the induced pmc is supposed to look like
    new_induced_pmc = stormvogel.model.new_dtmc()

    init = new_induced_pmc.initial_state

    new_induced_pmc.new_state(labels=["A"])
    new_induced_pmc.new_state(labels=["B"])

    init.set_choices(
        [
            (20, next(iter(new_induced_pmc.get_states_with_label("A")))),
            (-0.06, next(iter(new_induced_pmc.get_states_with_label("B")))),
        ]
    )

    # we add self loops to all states with no outgoing choices
    new_induced_pmc.add_self_loops()

    assert_models_equal(induced_pmc, new_induced_pmc)
