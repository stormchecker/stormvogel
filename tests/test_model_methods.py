import stormvogel.model
import stormvogel.examples.monty_hall
import stormvogel.examples.die
import stormvogel.examples.nuclear_fusion_ctmc
import pytest
from typing import cast


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

    assert (
        stormvogel.model.choices_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Value, stormvogel.model.State]],
                transition_shorthand,
            )
        )
        == transition
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

    assert (
        stormvogel.model.choices_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Action, stormvogel.model.State]],
                transition_shorthand,
            )
        )
        == transition
    )

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

    assert (
        stormvogel.model.choices_from_shorthand(
            cast(
                dict[
                    stormvogel.model.Action,
                    list[tuple[stormvogel.model.Value, stormvogel.model.State]],
                ],
                transition_shorthand,
            )
        )
        == transition
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

    assert dtmc0 == dtmc1


def test_remove_state():
    # we make a normal ctmc and remove a state
    ctmc = stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    ctmc.remove_state(ctmc.states[3], reassign_ids=True)

    # we make a ctmc with the state already missing
    new_ctmc = stormvogel.model.new_ctmc()
    new_ctmc.states[0].set_choices([(3, new_ctmc.new_state("helium"))])
    new_ctmc.states[1].set_choices([(2, new_ctmc.new_state("carbon"))])

    new_ctmc.new_state("Supernova")

    new_ctmc.add_self_loops()

    assert ctmc == new_ctmc

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
    mdp.remove_state(mdp.states[0], reassign_ids=True)

    # we make the mdp with the state already missing
    new_mdp = stormvogel.model.new_mdp(create_initial_state=False)
    new_mdp.new_state()
    new_mdp.new_state()
    new_mdp.add_self_loops()
    new_mdp.new_action("0")
    new_mdp.new_action(
        "1"
    )  # TODO are the models the same? the choices don't look the same

    assert mdp == new_mdp

    # this should fail:
    new_dtmc = stormvogel.examples.die.create_die_dtmc()
    state0 = new_dtmc.states[0]
    new_dtmc.remove_state(new_dtmc.initial_state, reassign_ids=True)
    state1 = new_dtmc.states[0]

    assert state0 != state1


def test_reassign_ids_removed_states():
    # we test if reassigning ids works after states are removed

    # we first make the die dtmc, remove one state and reassign ids
    dtmc = stormvogel.examples.die.create_die_dtmc()
    dtmc.remove_state(dtmc.initial_state)
    dtmc.reassign_ids()

    # we make the dtmc with the state already removed and ids already reassigned
    other_dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    for i in range(6):
        other_dtmc.new_state(labels=[f"rolled{i + 1}"], valuations={"rolled": i + 1})
    other_dtmc.add_self_loops()

    assert dtmc == other_dtmc


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
    assert mdp == mdp2

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
    assert mdp5.get_branches(mdp5.initial_state).branch == [
        (0.4, state5),
        (0.6, state5),
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
            (1 / 6, new_dtmc.new_state(f"rolled{i + 1}", {"rolled": i + 1}))
            for i in range(2)
        ]
    )
    new_dtmc.normalize()
    assert sub_model == new_dtmc


def test_get_state_reward():
    # we create an mdp:
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    # we add a reward model:
    rewardmodel = mdp.new_reward_model("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))

    state = mdp.states[2]

    assert rewardmodel.get_state_reward(state) == 2


# TODO re-introduce this test once names are removed from actions.
# def test_set_state_action_reward():
#     # we create an mdp:
#     mdp = stormvogel.model.new_mdp()
#     action = stormvogel.model.Action(frozenset({"0"}))
#     mdp.add_choices(mdp.initial_state, [(action, mdp.initial_state)])

#     # we make a reward model using the set_state_action_reward method:
#     rewardmodel = mdp.new_reward_model("rewardmodel")
#     rewardmodel.set_state_action_reward(mdp.initial_state, action, 5)

#     # we make a reward model manually:
#     other_rewardmodel = stormvogel.model.RewardModel("rewardmodel" {(0, stormvogel.model.EmptyAction): 5})

#     print(rewardmodel.rewards)
#     print()
#     print(other_rewardmodel.rewards)
#     quit()

#     assert rewardmodel == other_rewardmodel

#     # we create an mdp:
#     mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

#     # we add a reward model with only one reward
#     rewardmodel = mdp.new_reward_model("rewardmodel")
#     state = mdp.states[2]
#     action = state.available_actions()[1]
#     rewardmodel.set_state_action_reward(state, action, 3)

#     # we make a reward model manually:
#     other_rewardmodel = stormvogel.model.RewardModel("rewardmodel" {(5, EmptyAction): 3})

#     assert rewardmodel == other_rewardmodel


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
                dtmc.new_state(labels=f"rolled{i + 1}", valuations={"rolled": i + 1}),
            )
            for i in range(6)
        ]
    )
    dtmc.add_self_loops()

    assert list(dtmc.unassigned_variables()) == [(init, "rolled")]

    dtmc.add_valuation_at_remaining_states()

    assert not list(dtmc.unassigned_variables())
