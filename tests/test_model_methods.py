import stormvogel.model
import stormvogel.examples.monty_hall
import stormvogel.examples.die
import stormvogel.examples.nuclear_fusion_ctmc
import pytest
from typing import cast


def test_available_actions():
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    action = [
        stormvogel.model.Action(labels=frozenset({"open0"})),
        stormvogel.model.Action(labels=frozenset({"open1"})),
        stormvogel.model.Action(labels=frozenset({"open2"})),
    ]
    assert mdp.get_state_by_id(1).available_actions() == action

    # we also test it for a state with no available actions
    mdp = stormvogel.model.new_mdp()
    assert mdp.get_initial_state().available_actions()


def test_get_outgoing_choice():
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    choices = mdp.get_initial_state().get_outgoing_choice(stormvogel.model.EmptyAction)

    probabilities, states = zip(*choices)  # type: ignore

    assert pytest.approx(list(probabilities)) == [1 / 3, 1 / 3, 1 / 3]
    assert list(states) == [
        mdp.get_state_by_id(1),
        mdp.get_state_by_id(2),
        mdp.get_state_by_id(3),
    ]


def test_is_absorbing():
    # one example of a ctmc state that is absorbing and one that isn't
    ctmc = stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    state0 = ctmc.get_state_by_id(4)
    state1 = ctmc.get_state_by_id(3)
    assert state0.is_absorbing()
    assert not state1.is_absorbing()

    # one example of a dtmc state that is absorbing and one that isn't
    dtmc = stormvogel.examples.die.create_die_dtmc()
    state0 = dtmc.get_initial_state()
    state1 = dtmc.get_state_by_id(1)
    assert state1.is_absorbing()
    assert not state0.is_absorbing()


def test_choice_from_shorthand():
    # First we test it for a model without actions
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    transition_shorthand = [(1 / 2, state)]
    branch = stormvogel.model.Branch(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            transition_shorthand,
        )
    )
    action = stormvogel.model.EmptyAction
    transition = stormvogel.model.Choice({action: branch})

    assert (
        stormvogel.model.choice_from_shorthand(
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
    action = mdp.new_action(frozenset({"action"}))
    transition_shorthand = [(action, state)]
    branch = stormvogel.model.Branch(
        cast(list[tuple[stormvogel.model.Value, stormvogel.model.State]], [(1, state)])
    )
    transition = stormvogel.model.Choice({action: branch})

    assert (
        stormvogel.model.choice_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Action, stormvogel.model.State]],
                transition_shorthand,
            )
        )
        == transition
    )


def test_is_stochastic():
    # we check for an instance where it is not stochastic
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    dtmc.set_choice(
        dtmc.get_initial_state(),
        [(1 / 2, state)],
    )

    assert not dtmc.is_stochastic()

    # we check for an instance where it is stochastic
    dtmc.set_choice(
        dtmc.get_initial_state(),
        [(1 / 2, state), (1 / 2, state)],
    )

    dtmc.add_self_loops()

    assert dtmc.is_stochastic()

    # we check it for a continuous time model
    ctmc = stormvogel.model.new_ctmc()
    ctmc.set_choice(ctmc.get_initial_state(), [(1, ctmc.new_state())])

    ctmc.add_self_loops()

    assert not ctmc.is_stochastic()


def test_normalize():
    # we make a dtmc that has outgoing choices with sum of probabilities != 0 and we normalize it
    dtmc0 = stormvogel.model.new_dtmc()
    state = dtmc0.new_state()
    dtmc0.set_choice(
        dtmc0.get_initial_state(),
        [(1 / 4, state), (1 / 2, state)],
    )
    dtmc0.add_self_loops()
    dtmc0.normalize()

    # we make the same dtmc but with the already normalized probabilities
    dtmc1 = stormvogel.model.new_dtmc()
    state = dtmc1.new_state()
    dtmc1.set_choice(
        dtmc1.get_initial_state(),
        [(1 / 3, state), (2 / 3, state)],
    )
    dtmc1.add_self_loops()

    # TODO test for mdps as well

    assert dtmc0 == dtmc1


def test_remove_state():
    # we make a normal ctmc and remove a state
    ctmc = stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    ctmc.remove_state(ctmc.get_state_by_id(3), reassign_ids=True)

    # we make a ctmc with the state already missing
    new_ctmc = stormvogel.model.new_ctmc()
    new_ctmc.get_state_by_id(0).set_choice([(3, new_ctmc.new_state("helium"))])
    new_ctmc.get_state_by_id(1).set_choice([(2, new_ctmc.new_state("carbon"))])

    new_ctmc.new_state("Supernova")

    rates = [3, 2, 7, 0]
    for i in range(4):
        new_ctmc.set_rate(new_ctmc.get_state_by_id(i), rates[i])
    new_ctmc.add_self_loops()

    assert ctmc == new_ctmc

    # we also test if it works for a model that has nontrivial actions:
    mdp = stormvogel.model.new_mdp()
    state1 = mdp.new_state()
    state2 = mdp.new_state()
    action0 = mdp.new_action("0")
    action1 = mdp.new_action("1")
    branch0 = stormvogel.model.Branch(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 2, state1), (1 / 2, state2)],
        )
    )
    branch1 = stormvogel.model.Branch(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 4, state1), (3 / 4, state2)],
        )
    )
    transition = stormvogel.model.Choice({action0: branch0, action1: branch1})
    mdp.set_choice(mdp.get_initial_state(), transition)

    # we remove a state
    mdp.remove_state(mdp.get_state_by_id(0), reassign_ids=True)

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
    state0 = new_dtmc.get_state_by_id(0)
    new_dtmc.remove_state(new_dtmc.get_initial_state(), reassign_ids=True)
    state1 = new_dtmc.get_state_by_id(0)

    assert state0 != state1

    # This should complain that names are the same:
    try:
        new_dtmc.new_state()
        assert False
    except RuntimeError:
        pass

    # But no longer if we do this:
    try:
        new_dtmc.new_state(name="new_name")
    except RuntimeError:
        assert False


def test_reassign_ids_removed_states():
    # we test if reassigning ids works after states are removed

    # we first make the die dtmc, remove one state and reassign ids
    dtmc = stormvogel.examples.die.create_die_dtmc()
    dtmc.remove_state(dtmc.get_initial_state())
    dtmc.reassign_ids()

    # we make the dtmc with the state already removed and ids already reassigned
    other_dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    for i in range(6):
        other_dtmc.new_state(labels=[f"rolled{i+1}"], valuations={"rolled": i + 1})
    other_dtmc.add_self_loops()

    assert dtmc == other_dtmc


def test_remove_choices_between_states():
    # we make a model and remove choices between two states
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    dtmc.set_choice(
        dtmc.get_initial_state(),
        [(1, state)],
    )
    dtmc.set_choice(state, [(1, dtmc.get_initial_state())])
    dtmc.remove_choices_between_states(state, dtmc.get_initial_state())

    # we create a model with the choices between the two states already missing
    new_dtmc = stormvogel.model.new_dtmc()
    state = new_dtmc.new_state()
    new_dtmc.set_choice(
        dtmc.get_initial_state(),
        [(1, state)],
    )
    new_dtmc.add_self_loops()

    assert dtmc == new_dtmc


def test_add_choice():
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    # A non-action model should throw an exception.
    # with pytest.raises(RuntimeError) as excinfo:
    #    dtmc.add_choice(
    #        dtmc.get_initial_state(),
    #        [(0.5, state)],
    #    )
    # assert (
    #    str(excinfo.value)
    #    == "Models without actions do not support add_choice. Use set_choice instead."
    # )

    # Empty transition case, act exactly like set_choice.
    mdp = stormvogel.model.new_mdp()
    state = mdp.new_state()
    mdp.add_choice(
        mdp.get_initial_state(),
        [(0.5, state)],
    )
    mdp2 = stormvogel.model.new_mdp()
    state2 = mdp2.new_state()
    mdp2.set_choice(
        mdp2.get_initial_state(),
        [(0.5, state2)],
    )
    assert mdp == mdp2

    # Fail to add a real action to an empty action.
    mdp3 = stormvogel.model.new_mdp()
    state3 = mdp2.new_state()
    mdp3.set_choice(
        mdp3.get_initial_state(),
        [(0.5, state3)],
    )
    action3 = mdp3.new_action("action")
    with pytest.raises(RuntimeError) as excinfo:
        mdp3.add_choice(mdp3.get_initial_state(), [(action3, state3)])
    assert (
        str(excinfo.value)
        == "You cannot add a transition with an non-empty action to a transition which has an empty action. Use set_choice instead."
    )
    # And the other way round.
    mdp3 = stormvogel.model.new_mdp()
    state3 = mdp2.new_state()
    action3 = mdp3.new_action("action")
    mdp3.set_choice(
        mdp3.get_initial_state(),
        [(action3, state3)],
    )

    with pytest.raises(RuntimeError) as excinfo:
        mdp3.add_choice(mdp3.get_initial_state(), [(0.5, state3)])
    assert (
        str(excinfo.value)
        == "You cannot add a transition with an empty action to a transition which has no empty action. Use set_choice instead."
    )

    # Empty action case, add the branches together.
    mdp5 = stormvogel.model.new_mdp()
    state5 = mdp5.new_state()
    mdp5.set_choice(mdp5.get_initial_state(), [((0.4), state5)])
    mdp5.add_choice(mdp5.get_initial_state(), [(0.6, state5)])
    assert mdp5.get_branch(mdp5.get_initial_state()).branch == [
        ((0.4), state5),
        (0.6, state5),
    ]

    # Non-empty action case, add the actions to the list.
    mdp6 = stormvogel.model.new_mdp()
    state6 = mdp6.new_state()
    action6a = mdp6.new_action("a")
    action6b = mdp6.new_action("b")
    mdp6.set_choice(mdp6.get_initial_state(), [(action6a, state6)])
    mdp6.add_choice(mdp6.get_initial_state(), [(action6b, state6)])
    # print(mdp6.get_choice(mdp6.get_initial_state()).transition)
    # print([(action6a, state6), (action6b, state6)])
    assert len(mdp6.get_choice(mdp6.get_initial_state()).transition) == 2


def test_get_sub_model():
    # we create the die dtmc and take a submodel
    dtmc = stormvogel.examples.die.create_die_dtmc()
    states = [dtmc.get_state_by_id(0), dtmc.get_state_by_id(1), dtmc.get_state_by_id(2)]
    sub_model = dtmc.get_sub_model(states)

    # we build what the submodel should look like
    new_dtmc = stormvogel.model.new_dtmc()
    init = new_dtmc.get_initial_state()
    init.valuations = {"rolled": 0}
    init.set_choice(
        [
            (1 / 6, new_dtmc.new_state(f"rolled{i+1}", {"rolled": i + 1}))
            for i in range(2)
        ]
    )
    new_dtmc.normalize()
    assert sub_model == new_dtmc


def test_get_state_action_id():
    # we create an mdp:
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()
    state = mdp.get_state_by_id(2)
    action = state.available_actions()[1]

    assert mdp.get_state_action_id(state, action) == 5


def test_get_state_action_reward():
    # we create an mdp:
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    # we add a reward model:
    rewardmodel = mdp.new_reward_model("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))

    state = mdp.get_state_by_id(2)
    action = state.available_actions()[1]

    assert rewardmodel.get_state_action_reward(state, action) == 5


# TODO re-introduce this test once names are removed from actions.
# def test_set_state_action_reward():
#     # we create an mdp:
#     mdp = stormvogel.model.new_mdp()
#     action = stormvogel.model.Action(frozenset({"0"}))
#     mdp.add_choice(mdp.get_initial_state(), [(action, mdp.get_initial_state())])

#     # we make a reward model using the set_state_action_reward method:
#     rewardmodel = mdp.new_reward_model("rewardmodel")
#     rewardmodel.set_state_action_reward(mdp.get_initial_state(), action, 5)

#     # we make a reward model manually:
#     other_rewardmodel = stormvogel.model.RewardModel("rewardmodel", mdp, {(0, stormvogel.model.EmptyAction): 5})

#     print(rewardmodel.rewards)
#     print()
#     print(other_rewardmodel.rewards)
#     quit()

#     assert rewardmodel == other_rewardmodel

#     # we create an mdp:
#     mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

#     # we add a reward model with only one reward
#     rewardmodel = mdp.new_reward_model("rewardmodel")
#     state = mdp.get_state_by_id(2)
#     action = state.available_actions()[1]
#     rewardmodel.set_state_action_reward(state, action, 3)

#     # we make a reward model manually:
#     other_rewardmodel = stormvogel.model.RewardModel("rewardmodel", mdp, {(5, EmptyAction): 3})

#     assert rewardmodel == other_rewardmodel


def test_valuation_methods():
    # first we test the get_variables function
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()
    assert mdp.get_variables() == {"car_pos", "chosen_pos", "reveal_pos"}

    # we test the unassigned_variables function + the set_valuation_at_remaining_states function on the die model
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.get_initial_state()
    init.set_choice(
        [
            (1 / 6, dtmc.new_state(labels=f"rolled{i+1}", valuations={"rolled": i + 1}))
            for i in range(6)
        ]
    )
    dtmc.add_self_loops()

    assert dtmc.has_unassigned_variables()

    dtmc.set_valuation_at_remaining_states()  # TODO more elaborate test, especially when unassigned_variables() returns more information

    assert not dtmc.has_unassigned_variables()
