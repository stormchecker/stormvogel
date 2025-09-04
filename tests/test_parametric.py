import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.parametric
import stormvogel.model
import pytest


@pytest.mark.tags("stormpy")
def test_pmc_conversion():
    # Create a new model with the name "simple pmc"
    pmc = stormvogel.model.new_dtmc()

    init = pmc.get_initial_state()

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

    # TODO make it work for proper rational functions

    pmc.new_state(labels=["A"])
    pmc.new_state(labels=["B"])

    init.set_choice(
        [
            (p1, pmc.get_states_with_label("A")[0]),
            (p2, pmc.get_states_with_label("B")[0]),
        ]
    )

    # we add self loops to all states with no outgoing choices
    pmc.add_self_loops()

    # we test the mapping
    stormpy_pmc = mapping.stormvogel_to_stormpy(pmc)
    new_pmc = mapping.stormpy_to_stormvogel(stormpy_pmc)

    assert pmc == new_pmc


@pytest.mark.tags("stormpy")
def test_pmdp_conversion():
    # Create a new model with the name "simple pmdp"
    pmdp = stormvogel.model.new_mdp()

    init = pmdp.get_initial_state()

    # From the initial state, we have two actions with choices that either bring us to a goal state or sink state

    p1 = stormvogel.parametric.Polynomial(["x"])
    p2 = stormvogel.parametric.Polynomial(["x"])
    p1.add_term((1,), 1)
    p2.add_term((0,), 1)
    p2.add_term((1,), -1)

    goal = pmdp.new_state(labels=["goal"])
    sink = pmdp.new_state(labels=["sink"])

    action_a = pmdp.new_action(frozenset({"a"}))
    action_b = pmdp.new_action(frozenset({"b"}))
    branch0 = stormvogel.model.Branch(
        [
            (p1, goal),
            (p2, sink),
        ]
    )
    branch1 = stormvogel.model.Branch(
        [
            (p1, sink),
            (p2, goal),
        ]
    )

    pmdp.add_choice(
        init, stormvogel.model.Choice({action_a: branch0, action_b: branch1})
    )

    # we add self loops to all states with no outgoing choices
    pmdp.add_self_loops()

    # we test the mapping
    stormpy_pmdp = mapping.stormvogel_to_stormpy(pmdp)

    new_pmdp = mapping.stormpy_to_stormvogel(stormpy_pmdp)

    assert pmdp == new_pmdp


def test_pmc_valuations():
    # we build a simple pmc
    pmc = stormvogel.model.new_dtmc()

    init = pmc.get_initial_state()

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

    init.set_choice(
        [
            (p1, pmc.get_states_with_label("A")[0]),
            (r1, pmc.get_states_with_label("B")[0]),
        ]
    )

    # we add self loops to all states with no outgoing choices
    pmc.add_self_loops()

    induced_pmc = pmc.parameter_valuation({"x": 1, "y": 2, "w": 1, "z": 5})

    # we build what the induced pmc is supposed to look like
    new_induced_pmc = stormvogel.model.new_dtmc()

    init = new_induced_pmc.get_initial_state()

    new_induced_pmc.new_state(labels=["A"])
    new_induced_pmc.new_state(labels=["B"])

    init.set_choice(
        [
            (20, new_induced_pmc.get_states_with_label("A")[0]),
            (-0.06, new_induced_pmc.get_states_with_label("B")[0]),
        ]
    )

    # we add self loops to all states with no outgoing choices
    new_induced_pmc.add_self_loops()

    assert induced_pmc == new_induced_pmc
