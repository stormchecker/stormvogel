import stormvogel.model
import stormvogel.examples.monty_hall
import stormvogel.examples.die
import stormvogel.examples.nuclear_fusion_ctmc
import pytest
from typing import cast
from model_testing import assert_models_equal, assert_choices_equal
from stormvogel.model.variable import Variable


def test_available_actions():
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    action = [
        stormvogel.model.Action(label="open0"),
        stormvogel.model.Action(label="open1"),
        stormvogel.model.Action(label="open2"),
    ]
    assert list(mdp.states[1].available_actions()) == action

    # we also test it for a state with no available actions
    mdp = stormvogel.model.new_mdp()
    assert list(mdp.initial_state.available_actions()) == []


def test_get_outgoing_transitions():
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    choices = mdp.initial_state.get_outgoing_transitions(stormvogel.model.EmptyAction)

    probabilities, states = zip(*choices)  # type: ignore

    assert pytest.approx(list(probabilities)) == [1 / 3, 1 / 3, 1 / 3]
    assert list(states) == [
        mdp.states[1],
        mdp.states[2],
        mdp.states[3],
    ]


def test_is_absorbing():
    # one example of a ctmc state that is absorbing and one that isn't
    ctmc = stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    state0 = ctmc.states[4]
    state1 = ctmc.states[3]
    assert state0.is_absorbing()
    assert not state1.is_absorbing()

    # one example of a dtmc state that is absorbing and one that isn't
    dtmc = stormvogel.examples.die.create_die_dtmc()
    state0 = dtmc.initial_state
    state1 = dtmc.states[1]
    assert state1.is_absorbing()
    assert not state0.is_absorbing()

    # A state with literally no choices in the model
    empty_dtmc = stormvogel.model.new_dtmc()
    empty_state = empty_dtmc.new_state()
    assert empty_state.is_absorbing()


def test_choices_from_shorthand():
    # First we test it for a model without actions
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    transition_shorthand = [(1 / 2, state)]
    branch = stormvogel.model.Branches(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            transition_shorthand,
        )
    )
    action = stormvogel.model.EmptyAction
    transition = stormvogel.model.Choices({action: branch})

    assert_choices_equal(
        stormvogel.model.choices_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Value, stormvogel.model.State]],
                transition_shorthand,
            )
        ),
        transition,
    )

    # Then we test it for a model with actions
    mdp = stormvogel.model.new_mdp()
    state = mdp.new_state()
    action = mdp.new_action("action")
    transition_shorthand = [(action, state)]
    branch = stormvogel.model.Branches(
        cast(list[tuple[stormvogel.model.Value, stormvogel.model.State]], [(1, state)])
    )
    transition = stormvogel.model.Choices({action: branch})

    assert_choices_equal(
        stormvogel.model.choices_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Action, stormvogel.model.State]],
                transition_shorthand,
            )
        ),
        transition,
    )


def test_state_str():
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    state.set_labels({"a", "b"})
    s1 = str(state)
    s2 = str(state)
    assert s1 == s2
    assert s1.startswith(f"id: {state.state_id}, labels: ")
    assert "['" in s1 or '["' in s1 or "[]" in s1
    assert ", valuations: " in s1
    # we test it for nontrivial action transitions


def test_choices_from_shorthand_dict_state():
    mdp = stormvogel.model.new_mdp()
    state1 = mdp.new_state()
    state2 = mdp.new_state()
    action0 = mdp.new_action("0")
    action1 = mdp.new_action("1")
    transition_shorthand = {
        action0: [(1 / 2, state1), (1 / 2, state2)],
        action1: [(1 / 2, state1), (1 / 2, state2)],
    }
    branch = stormvogel.model.Branches(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 2, state1), (1 / 2, state2)],
        )
    )
    transition = stormvogel.model.Choices({action0: branch, action1: branch})

    assert_choices_equal(
        stormvogel.model.choices_from_shorthand(
            cast(
                dict[
                    stormvogel.model.Action,
                    list[tuple[stormvogel.model.Value, stormvogel.model.State]],
                ],
                transition_shorthand,
            )
        ),
        transition,
    )


def test_is_stochastic():
    # we check for an instance where it is not stochastic
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    dtmc.set_choices(
        dtmc.initial_state,
        [(1 / 2, state)],
    )

    assert not dtmc.is_stochastic()

    # we check for an instance where it is stochastic
    dtmc.set_choices(
        dtmc.initial_state,
        [(1 / 2, state), (1 / 2, state)],
    )

    dtmc.add_self_loops()

    assert dtmc.is_stochastic()

    # we check it for a continuous time model
    ctmc = stormvogel.model.new_ctmc()
    ctmc.set_choices(ctmc.initial_state, [(1, ctmc.new_state())])

    ctmc.add_self_loops()

    assert not ctmc.is_stochastic()


def test_normalize():
    # we make a dtmc that has outgoing choices with sum of probabilities != 0 and we normalize it
    dtmc0 = stormvogel.model.new_dtmc()
    state = dtmc0.new_state()
    dtmc0.set_choices(
        dtmc0.initial_state,
        [(1 / 4, state), (1 / 2, state)],
    )
    dtmc0.add_self_loops()
    dtmc0.normalize()

    # we make the same dtmc but with the already normalized probabilities
    dtmc1 = stormvogel.model.new_dtmc()
    state = dtmc1.new_state()
    dtmc1.set_choices(
        dtmc1.initial_state,
        [(1 / 3, state), (2 / 3, state)],
    )
    dtmc1.add_self_loops()

    # TODO test for mdps as well

    assert_models_equal(dtmc0, dtmc1)


def test_remove_state():
    # we make a normal ctmc and remove a state
    ctmc = stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    ctmc.remove_state(ctmc.states[3], suppress_warning=True)

    # we make a ctmc with the state already missing
    new_ctmc = stormvogel.model.new_ctmc()
    new_ctmc.states[0].set_choices([(3, new_ctmc.new_state("helium"))])
    new_ctmc.states[1].set_choices([(2, new_ctmc.new_state("carbon"))])

    new_ctmc.new_state("Supernova")

    new_ctmc.add_self_loops()

    assert_models_equal(ctmc, new_ctmc)

    # we also test if it works for a model that has nontrivial actions:
    mdp = stormvogel.model.new_mdp()
    state1 = mdp.new_state()
    state2 = mdp.new_state()
    action0 = mdp.new_action("0")
    action1 = mdp.new_action("1")
    branch0 = stormvogel.model.Branches(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 2, state1), (1 / 2, state2)],
        )
    )
    branch1 = stormvogel.model.Branches(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 4, state1), (3 / 4, state2)],
        )
    )
    transition = stormvogel.model.Choices({action0: branch0, action1: branch1})
    mdp.set_choices(mdp.initial_state, transition)

    # we remove a state
    mdp.remove_state(mdp.states[0], suppress_warning=True)

    # we make the mdp with the state already missing
    new_mdp = stormvogel.model.new_mdp(create_initial_state=False)
    new_mdp.new_state()
    new_mdp.new_state()
    new_mdp.add_self_loops()
    new_mdp.new_action("0")
    new_mdp.new_action(
        "1"
    )  # TODO are the models the same? the choices don't look the same

    assert_models_equal(mdp, new_mdp)

    # this should fail:
    new_dtmc = stormvogel.examples.die.create_die_dtmc()
    state0 = new_dtmc.states[0]
    new_dtmc.remove_state(new_dtmc.initial_state, suppress_warning=True)
    state1 = new_dtmc.states[0]

    assert state0 != state1


def test_remove_state_ids():
    dtmc = stormvogel.examples.die.create_die_dtmc()
    dtmc.remove_state(dtmc.initial_state, suppress_warning=True)

    other_dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    for i in range(6):
        other_dtmc.new_state(
            labels=[f"rolled{i + 1}"], valuations={Variable("rolled"): i + 1}
        )
    other_dtmc.add_self_loops()

    assert_models_equal(dtmc, other_dtmc)


def test_summary_counts_model_choices():
    mdp = stormvogel.model.new_mdp()
    state1 = mdp.new_state()
    state2 = mdp.new_state()
    action_a = mdp.new_action("a")
    action_b = mdp.new_action("b")
    mdp.set_choices(
        mdp.initial_state,
        {action_a: [(1, state1)], action_b: [(1, state2)]},
    )

    summary = mdp.summary()

    assert "2 choices" in summary


def test_new_state_validation_is_transactional():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    with pytest.raises(
        RuntimeError,
        match="without providing an observation",
    ):
        pomdp.new_state()

    assert len(pomdp.states) == 0
    assert len(pomdp.transitions) == 0
    assert len(pomdp.state_valuations) == 0

    dtmc = stormvogel.model.new_dtmc()
    obs_model = stormvogel.model.new_pomdp(create_initial_state=False)
    observation = obs_model.new_observation("obs")
    states_before = len(dtmc.states)
    transitions_before = len(dtmc.transitions)
    valuations_before = len(dtmc.state_valuations)

    with pytest.raises(
        RuntimeError,
        match="does not support observations",
    ):
        dtmc.new_state(observation=observation)

    assert len(dtmc.states) == states_before
    assert len(dtmc.transitions) == transitions_before
    assert len(dtmc.state_valuations) == valuations_before


def test_remove_state_cleans_state_bookkeeping():
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    observation = pomdp.new_observation("obs")
    state = pomdp.new_state(
        labels=["tmp"],
        valuations={Variable("value"): 1},
        observation=observation,
    )
    state.set_friendly_name("tmp-state")
    reward_model = pomdp.new_reward_model("reward")
    reward_model.rewards[state] = 1

    pomdp.remove_state(state, normalize=False, suppress_warning=True)

    assert state not in pomdp.states
    assert state not in pomdp.state_valuations
    assert state not in pomdp.friendly_names
    assert state not in pomdp.state_observations
    assert state not in pomdp.state_labels["tmp"]
    assert state not in reward_model.rewards


def test_remove_state_updates_markovian_states():
    ma = stormvogel.model.new_ma(create_initial_state=False)
    state = ma.new_state()
    ma.add_markovian_state(state)

    ma.remove_state(state, normalize=False, suppress_warning=True)

    assert state not in ma.markovian_states


def test_add_choices():
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    # A non-action model should throw an exception.
    # with pytest.raises(RuntimeError) as excinfo:
    #    dtmc.add_choices(
    #        dtmc.initial_state,
    #        [(0.5, state)],
    #    )
    # assert (
    #    str(excinfo.value)
    #    == "Models without actions do not support add_choice. Use set_choice instead."
    # )

    # Empty transition case, act exactly like set_choice.
    mdp = stormvogel.model.new_mdp()
    state = mdp.new_state()
    mdp.add_choices(
        mdp.initial_state,
        [(0.5, state)],
    )
    mdp2 = stormvogel.model.new_mdp()
    state2 = mdp2.new_state()
    mdp2.set_choices(
        mdp2.initial_state,
        [(0.5, state2)],
    )
    assert_models_equal(mdp, mdp2)

    # Fail to add a real action to an empty action.
    mdp3 = stormvogel.model.new_mdp()
    state3 = mdp2.new_state()
    mdp3.set_choices(
        mdp3.initial_state,
        [(0.5, state3)],
    )
    action3 = mdp3.new_action("action")
    with pytest.raises(RuntimeError) as excinfo:
        mdp3.add_choices(mdp3.initial_state, [(action3, state3)])
    assert (
        str(excinfo.value)
        == "You cannot add a choice with an non-empty action to a choice which has an empty action. Use set_choice instead."
    )
    # And the other way round.
    mdp3 = stormvogel.model.new_mdp()
    state3 = mdp2.new_state()
    action3 = mdp3.new_action("action")
    mdp3.set_choices(
        mdp3.initial_state,
        [(action3, state3)],
    )

    with pytest.raises(RuntimeError) as excinfo:
        mdp3.add_choices(mdp3.initial_state, [(0.5, state3)])
    assert (
        str(excinfo.value)
        == "You cannot add a choice with an empty action to a choice which has no empty action. Use set_choice instead."
    )

    # Empty action case, add the branches together.
    mdp5 = stormvogel.model.new_mdp()
    state5 = mdp5.new_state()
    mdp5.set_choices(mdp5.initial_state, [(0.4, state5)])
    mdp5.add_choices(mdp5.initial_state, [(0.6, state5)])
    assert mdp5.get_branches(mdp5.initial_state).branches.distribution == [
        (1, state5),
    ]

    # Non-empty action case, add the actions to the list.
    mdp6 = stormvogel.model.new_mdp()
    state6 = mdp6.new_state()
    action6a = mdp6.new_action("a")
    action6b = mdp6.new_action("b")
    mdp6.set_choices(mdp6.initial_state, [(action6a, state6)])
    mdp6.add_choices(mdp6.initial_state, [(action6b, state6)])
    # print(mdp6.get_choice(mdp6.initial_state).choice)
    # print([(action6a, state6), (action6b, state6)])
    assert len(mdp6.choices[mdp6.initial_state].choices) == 2


def test_get_sub_model():
    # we create the die dtmc and take a submodel
    dtmc = stormvogel.examples.die.create_die_dtmc()
    states = [dtmc.states[0], dtmc.states[1], dtmc.states[2]]
    sub_model = dtmc.get_sub_model(states)

    # we build what the submodel should look like
    new_dtmc = stormvogel.model.new_dtmc()
    init = new_dtmc.initial_state
    init.valuations = {"rolled": 0}
    init.set_choices(
        [
            (1 / 6, new_dtmc.new_state(f"rolled{i + 1}", {Variable("rolled"): i + 1}))
            for i in range(2)
        ]
    )
    new_dtmc.normalize()
    assert_models_equal(sub_model, new_dtmc)


def test_get_state_reward():
    # we create an mdp:
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    # we add a reward model:
    rewardmodel = mdp.new_reward_model("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))

    state = mdp.states[2]

    assert rewardmodel.get_state_reward(state) == 2


def test_valuation_methods():
    # first we test the get_variables function
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()
    assert mdp.variables == {"car_pos", "chosen_pos", "reveal_pos"}

    # we test the unassigned_variables function + the add_valuation_at_remaining_states function on the die model
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    init.set_choices(
        [
            (
                1 / 6,
                dtmc.new_state(
                    labels=f"rolled{i + 1}", valuations={Variable("rolled"): i + 1}
                ),
            )
            for i in range(6)
        ]
    )
    dtmc.add_self_loops()

    assert list(dtmc.unassigned_variables()) == [(init, "rolled")]

    dtmc.add_valuation_at_remaining_states()

    assert not list(dtmc.unassigned_variables())


def test_choices_from_shorthand_empty_list():
    """An empty list shorthand must raise ValueError, not IndexError."""
    import pytest

    with pytest.raises(ValueError, match="empty list shorthand"):
        stormvogel.model.choices_from_shorthand([])


def test_choices_from_shorthand_empty_dict():
    """An empty dict shorthand is valid and produces an empty Choices."""
    choices = stormvogel.model.choices_from_shorthand({})
    assert len(choices) == 0
