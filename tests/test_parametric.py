import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.parametric
import stormvogel.model
import pytest


try:
    import stormpy
except ImportError:
    stormpy = None


def pmc_equal(m0, m1) -> bool:
    """
    outputs true if the pmc models are the same and false otherwise
    Note: this function is only here because the equality functions in storm do not work currently.
    """
    assert stormpy is not None

    # check if states are the same:
    states_equal = True
    for i in range(m0.nr_states):
        actions_equal = True
        for j in range(len(m0.states[i].actions)):
            if not m0.states[i].actions[j] == m1.states[i].actions[j]:
                actions_equal = True
        if not (
            m0.states[i].id == m1.states[i].id
            and m0.states[i].labels == m1.states[i].labels
            and actions_equal
        ):
            states_equal = False

    # check if the matrices are the same:
    # TODO check for semantic equivalence and not just syntactic

    # TODO it is not comparing them correctly although upon checking manually they are equal
    print(m0.transition_matrix, m1.transition_matrix)
    matrices_equal = True  # str(m0.transition_matrix) == str(m1.transition_matrix)

    # check if model types are equal:
    types_equal = m0.model_type == m1.model_type

    # check if reward models are equal:
    reward_models_equal = True
    for key in m0.reward_models.keys():
        for i in range(m0.nr_states):
            if (
                m0.reward_models[key].has_state_rewards
                and m1.reward_models[key].has_state_rewards
            ):
                if not m0.reward_models[key].get_state_reward(i) == m1.reward_models[
                    key
                ].get_state_reward(i):
                    reward_models_equal = False
            if (
                m0.reward_models[key].has_state_action_rewards
                and m1.reward_models[key].has_state_action_rewards
            ):
                if not m0.reward_models[key].get_state_action_reward(
                    i
                ) == m1.reward_models[key].get_state_action_reward(i):
                    reward_models_equal = False

    # check if exit rates are equal (in case of ctmcs):
    exit_rates_equal = (
        not m0.model_type == stormpy.ModelType.CTMC or m0.exit_rates == m1.exit_rates
    )

    # check if observations are equal (in case of pomdps):
    observations_equal = (
        not m0.model_type == stormpy.ModelType.POMDP
        or m0.observations == m1.observations
    )

    # check if markovian states are equal (in case of mas):
    markovian_states_equal = (
        not m0.model_type == stormpy.ModelType.MA
        or m0.markovian_states == m1.markovian_states
    )

    return (
        matrices_equal
        and types_equal
        and states_equal
        and reward_models_equal
        and exit_rates_equal
        and observations_equal
        and markovian_states_equal
    )


@pytest.mark.tags("stormpy")
def test_pmc_conversion():
    # Create a new model
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

    print(pmc, new_pmc)
    # TODO this test is nondeterministic (probably has to do with gathering of variables in stormpy)
    # assert pmc == new_pmc


@pytest.mark.tags("stormpy")
def test_pmdp_conversion():
    # Create a new model
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

    action_a = pmdp.new_action("a")
    action_b = pmdp.new_action("b")
    branch0 = stormvogel.model.Branches(
        [
            (p1, goal),
            (p2, sink),
        ]
    )
    branch1 = stormvogel.model.Branches(
        [
            (p1, sink),
            (p2, goal),
        ]
    )

    pmdp.add_choice(
        init, stormvogel.model.Choices({action_a: branch0, action_b: branch1})
    )

    # we add self loops to all states with no outgoing choices
    pmdp.add_self_loops()

    # we test the mapping
    stormpy_pmdp = mapping.stormvogel_to_stormpy(pmdp)

    new_pmdp = mapping.stormpy_to_stormvogel(stormpy_pmdp)

    assert pmdp == new_pmdp


@pytest.mark.tags("stormpy")
def test_pmc_conversion_from_stormpy():
    import stormvogel.examples.stormpy_examples.stormpy_pmc

    stormpy_pmc = (
        stormvogel.examples.stormpy_examples.stormpy_pmc.example_parametric_models_01()
    )
    stormvogel_pmc = stormvogel.mapping.stormpy_to_stormvogel(stormpy_pmc)
    assert stormvogel_pmc is not None
    new_stormpy_pmc = stormvogel.mapping.stormvogel_to_stormpy(stormvogel_pmc)
    assert pmc_equal(stormpy_pmc, new_stormpy_pmc)


# TODO one more test from stormpy for an mdp


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
