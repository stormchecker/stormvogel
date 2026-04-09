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
    branch = stormvogel.model.Distribution(
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
    branch = stormvogel.model.Distribution(
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
    dtmc.add_label("a")
    dtmc.add_label("b")
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
    branch = stormvogel.model.Distribution(
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
    branch0 = stormvogel.model.Distribution(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 2, state1), (1 / 2, state2)],
        )
    )
    branch1 = stormvogel.model.Distribution(
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

    # Overlapping empty action should fail consistently.
    mdp5 = stormvogel.model.new_mdp()
    state5 = mdp5.new_state()
    mdp5.set_choices(mdp5.initial_state, [(0.4, state5)])
    with pytest.raises(RuntimeError) as excinfo:
        mdp5.add_choices(mdp5.initial_state, [(0.6, state5)])
    assert (
        str(excinfo.value)
        == "Cannot add choices with overlapping actions. Action EmptyAction is in both choices."
    )

    # Non-empty action case, add the actions to the list.
    mdp6 = stormvogel.model.new_mdp()
    state6 = mdp6.new_state()
    action6a = mdp6.new_action("a")
    action6b = mdp6.new_action("b")
    mdp6.set_choices(mdp6.initial_state, [(action6a, state6)])
    mdp6.add_choices(mdp6.initial_state, [(action6b, state6)])
    # print(mdp6.get_choice(mdp6.initial_state).choice)
    # print([(action6a, state6), (action6b, state6)])
    assert len(mdp6.transitions[mdp6.initial_state]) == 2


def test_get_sub_model():
    # we create the die dtmc and take a submodel
    dtmc = stormvogel.examples.die.create_die_dtmc()
    states = [dtmc.states[0], dtmc.states[1], dtmc.states[2]]
    sub_model = dtmc.get_sub_model(states)

    # we build what the submodel should look like
    new_dtmc = stormvogel.model.new_dtmc()
    init = new_dtmc.initial_state
    init.valuations = {Variable("rolled"): 0}
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
    assert mdp.variables == {
        Variable("car_pos"),
        Variable("chosen_pos"),
        Variable("reveal_pos"),
    }

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

    assert list(dtmc.unassigned_variables()) == [(init, Variable("rolled"))]

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


# ── Choices edge cases ────────────────────────────────────────────────────────


def test_choices_constructor_rejects_empty_and_nonempty_action():
    """Choices with >1 action including EmptyAction raises RuntimeError."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution([(1.0, s)])
    a = stormvogel.model.Action("go")
    with pytest.raises(RuntimeError):
        stormvogel.model.Choices({stormvogel.model.EmptyAction: d, a: d})


def test_choices_str():
    """Choices.__str__ includes the action label."""
    mdp = stormvogel.model.new_mdp()
    s2 = mdp.new_state()
    a = mdp.new_action("move")
    d2 = stormvogel.model.Distribution([(1.0, s2)])
    c2 = stormvogel.model.Choices({a: d2})
    assert "move" in str(c2)


def test_choices_has_zero_transition_returns_true():
    """has_zero_transition returns True when a zero-prob transition exists."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution({s: 0})
    c = stormvogel.model.Choices({stormvogel.model.EmptyAction: d})
    assert c.has_zero_transition()


def test_choices_add_raises_on_zero_transition():
    """Choices.add raises RuntimeError when added choices contain zero prob."""
    dtmc = stormvogel.model.new_dtmc()
    dtmc.new_state()
    mdp = stormvogel.model.new_mdp()
    s2 = mdp.new_state()
    a1 = mdp.new_action("a1")
    a2 = mdp.new_action("a2")
    d_ok = stormvogel.model.Distribution([(1.0, s2)])
    d_zero = stormvogel.model.Distribution({s2: 0.0})
    c1 = stormvogel.model.Choices({a1: d_ok})
    c_bad = stormvogel.model.Choices({a2: d_zero})
    with pytest.raises(RuntimeError, match="nonzero"):
        c1.add(c_bad)


def test_choices_add_operator():
    """Choices.__add__ returns a new merged Choices."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a1 = mdp.new_action("a")
    a2 = mdp.new_action("b")
    d = stormvogel.model.Distribution([(1.0, s)])
    c1 = stormvogel.model.Choices({a1: d})
    c2 = stormvogel.model.Choices({a2: d})
    c3 = c1 + c2
    assert len(c3) == 2


def test_choices_setitem_none_key():
    """Choices.__setitem__ raises ValueError when key is None."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution([(1.0, s)])
    c = stormvogel.model.Choices({stormvogel.model.EmptyAction: d})
    with pytest.raises(ValueError, match="None"):
        c[None] = d  # type: ignore


def test_choices_setitem_nonempty_on_empty_action():
    """Choices.__setitem__ raises RuntimeError when setting non-empty action on empty-action Choices."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution([(1.0, s)])
    c = stormvogel.model.Choices({stormvogel.model.EmptyAction: d})
    a = stormvogel.model.Action("go")
    with pytest.raises(RuntimeError):
        c[a] = d


def test_choices_setitem_empty_on_nonempty_action():
    """Choices.__setitem__ raises RuntimeError when setting empty action on non-empty Choices."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution([(1.0, s)])
    a = stormvogel.model.Action("go")
    c = stormvogel.model.Choices({a: d})
    with pytest.raises(RuntimeError):
        c[stormvogel.model.EmptyAction] = d


def test_choices_from_shorthand_unsupported_type():
    """choices_from_shorthand raises RuntimeError for unsupported element types."""
    with pytest.raises(RuntimeError):
        # "hello" is a str — neither an Action nor a numeric Value — triggers line 203
        stormvogel.model.choices_from_shorthand([("hello", "state")])  # type: ignore


# ── Model property and method edge cases ──────────────────────────────────────


def test_model_actions_raises_for_dtmc():
    """Model.actions raises RuntimeError for models without actions."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="does not support actions"):
        _ = list(dtmc.actions)


def test_model_observations_raises_for_dtmc():
    """Model.observations raises RuntimeError for non-observation models."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="does not support observations"):
        _ = list(dtmc.observations)


def test_model_observations_distribution():
    """Model.observations yields observations from Distribution-typed state_observations."""
    from stormvogel.model.distribution import Distribution as Dist

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs1 = pomdp.new_observation("a")
    obs2 = pomdp.new_observation("b")
    pomdp.new_state(observation=Dist([(0.5, obs1), (0.5, obs2)]))
    obs_list = list(pomdp.observations)
    assert obs1 in obs_list or obs2 in obs_list


def test_model_initial_state_raises_when_missing():
    """Model.initial_state raises RuntimeError when no 'init' state exists."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    with pytest.raises(RuntimeError, match="initial state"):
        _ = dtmc.initial_state


def test_model_add_label_duplicate_raises():
    """Model.add_label raises RuntimeError for duplicate labels."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="already exists"):
        dtmc.add_label("init")  # "init" is already added


def test_model_get_state_by_id_not_found():
    """Model.get_state_by_id raises RuntimeError when no state has the given id."""
    import uuid

    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="not found"):
        dtmc.get_state_by_id(uuid.uuid4())


def test_model_get_state_by_id_found():
    """Model.get_state_by_id returns the correct state."""
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.initial_state
    assert dtmc.get_state_by_id(state.state_id) is state


def test_model_new_observation_raises_for_dtmc():
    """Model.new_observation raises RuntimeError for non-POMDP models."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="does not support observations"):
        dtmc.new_observation("obs")


def test_model_new_observation_raises_duplicate_alias():
    """Model.new_observation raises RuntimeError for duplicate aliases."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    pomdp.new_observation("obs")
    with pytest.raises(RuntimeError, match="already exists"):
        pomdp.new_observation("obs")


def test_model_get_observation_raises_for_dtmc():
    """Model.get_observation raises RuntimeError for non-POMDP models."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="does not support observations"):
        dtmc.get_observation("x")


def test_model_get_observation_raises_not_found():
    """Model.get_observation raises KeyError when alias not found."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    with pytest.raises(KeyError):
        pomdp.get_observation("nonexistent")


def test_model_get_observation_found():
    """Model.get_observation returns the matching observation."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("my_obs")
    assert pomdp.get_observation("my_obs") is obs


def test_model_observation_method_raises_for_dtmc():
    """Model.observation raises RuntimeError for non-observation models."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="does not support observations"):
        dtmc.observation("x")


def test_model_observation_method_creates_new():
    """Model.observation creates a new observation if the alias doesn't exist."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.observation("fresh")
    assert obs.alias == "fresh"


def test_model_observation_method_returns_existing():
    """Model.observation returns the existing observation if the alias exists."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs1 = pomdp.new_observation("obs")
    obs2 = pomdp.observation("obs")
    assert obs1 is obs2


def test_model_make_observations_deterministic_raises_for_dtmc():
    """Model.make_observations_deterministic raises RuntimeError for non-POMDP."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="support observations"):
        dtmc.make_observations_deterministic()


def test_model_add_markovian_state_raises_for_non_ma():
    """Model.add_markovian_state raises RuntimeError for non-MA models."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="not a MA"):
        dtmc.add_markovian_state(dtmc.initial_state)


def test_model_set_choices_raises_on_zero_transition():
    """Model.set_choices raises RuntimeError when a zero-probability transition exists."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution({s: 0})
    c = stormvogel.model.Choices({stormvogel.model.EmptyAction: d})
    with pytest.raises(RuntimeError, match="nonzero"):
        dtmc.set_choices(dtmc.initial_state, c)


def test_model_get_distribution_raises_nonempty_choice():
    """Model.get_distribution raises RuntimeError for non-empty choices."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    with pytest.raises(RuntimeError, match="non-empty"):
        mdp.get_distribution(mdp.initial_state)


def test_model_get_distribution_returns_for_empty_action():
    """Model.get_distribution returns the distribution for EmptyAction."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(1.0, s)])
    dist = dtmc.get_distribution(dtmc.initial_state)
    assert dist is not None


def test_model_has_zero_transition_returns_true():
    """Model.has_zero_transition returns True when a zero-prob transition exists."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    d = stormvogel.model.Distribution({s: 0})
    c = stormvogel.model.Choices({stormvogel.model.EmptyAction: d})
    dtmc.transitions[dtmc.initial_state] = c  # bypass set_choices check
    assert dtmc.has_zero_transition()


def test_model_new_reward_model_duplicate_raises():
    """Model.new_reward_model raises RuntimeError for duplicate name."""
    dtmc = stormvogel.model.new_dtmc()
    dtmc.new_reward_model("R")
    with pytest.raises(RuntimeError, match="already present"):
        dtmc.new_reward_model("R")


def test_model_get_default_rewards_raises_when_empty():
    """Model.get_default_rewards raises RuntimeError when no reward models exist."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="no reward models"):
        dtmc.get_default_rewards()


def test_model_get_default_rewards_returns_first():
    """Model.get_default_rewards returns the first reward model."""
    dtmc = stormvogel.model.new_dtmc()
    rm = dtmc.new_reward_model("R")
    assert dtmc.get_default_rewards() is rm


def test_model_get_rewards_raises_not_found():
    """Model.get_rewards raises RuntimeError when reward model not found."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="not present"):
        dtmc.get_rewards("nonexistent")


def test_model_normalize_raises_for_parametric():
    """Model.normalize raises RuntimeError for parametric models."""
    from stormvogel.parametric import Polynomial

    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    p = Polynomial(["x"])  # x^1 — a non-zero polynomial
    p.add_term((1,), 1.0)
    d = stormvogel.model.Distribution({s: p})
    c = stormvogel.model.Choices({stormvogel.model.EmptyAction: d})
    dtmc.transitions[dtmc.initial_state] = c  # bypass set_choices zero-check
    with pytest.raises(RuntimeError, match="normalize"):
        dtmc.normalize()


def test_model_add_valuation_at_remaining_states_with_variables():
    """add_valuation_at_remaining_states with explicit variables list."""
    from stormvogel.model.variable import Variable

    dtmc = stormvogel.model.new_dtmc()
    x = Variable("x")
    dtmc.add_valuation_at_remaining_states(variables=[x], value=99)
    for state in dtmc:
        assert state.valuations.get(x) == 99


def test_model_to_dot_with_actions():
    """Model.to_dot generates correct DOT structure for an MDP with one action."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state("s1")
    a = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    dot = mdp.to_dot()

    assert dot.startswith("digraph model {")
    assert dot.strip().endswith("}")
    # Both state nodes must carry their labels
    assert ': init"' in dot
    assert ': s1"' in dot
    # The action node must be rendered as a point
    assert "shape=point" in dot
    # Edge from state to action carries the action name; edge from action to target carries prob
    assert 'label = "go"' in dot
    assert 'label = "1.0"' in dot
    # Exactly two directed edges: init→action_node and action_node→s1
    assert dot.count("->") == 2


def test_model_to_dot_dtmc():
    """Model.to_dot generates correct DOT structure for a 2-state DTMC."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state("s1")
    dtmc.set_choices(dtmc.initial_state, [(1.0, s)])
    dot = dtmc.to_dot()

    assert dot.startswith("digraph model {")
    assert dot.strip().endswith("}")
    # Both state nodes must carry their labels
    assert ': init"' in dot
    assert ': s1"' in dot
    # The single edge must carry the probability as its label
    assert 'label = "1.0"' in dot
    # Exactly one directed edge (no action nodes for DTMCs)
    assert dot.count("->") == 1


def test_model_str():
    """Model.__str__ returns summary string."""
    dtmc = stormvogel.model.new_dtmc()
    s = str(dtmc)
    assert "DTMC" in s


def test_model_getitem():
    """Model.__getitem__ returns state by index."""
    dtmc = stormvogel.model.new_dtmc()
    assert dtmc[0] is dtmc.initial_state


def test_model_deprecated_get_type():
    """Model.get_type (deprecated) returns the model type."""
    import warnings

    dtmc = stormvogel.model.new_dtmc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dtmc.get_type() == stormvogel.model.ModelType.DTMC


def test_model_deprecated_get_initial_state():
    """Model.get_initial_state (deprecated) returns the initial state."""
    import warnings

    dtmc = stormvogel.model.new_dtmc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dtmc.get_initial_state() is dtmc.initial_state


def test_new_hmm():
    """new_hmm creates an HMM model."""
    hmm = stormvogel.model.new_hmm()
    assert hmm.model_type == stormvogel.model.ModelType.HMM


def test_model_remove_state_warns():
    """Model.remove_state emits a warning when suppress_warning=False."""
    import warnings

    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dtmc.remove_state(s, normalize=False)
        assert len(w) >= 1


def test_model_remove_state_raises_not_in_model():
    """Model.remove_state raises RuntimeError when state not in model."""
    dtmc = stormvogel.model.new_dtmc()
    other_dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError):
        dtmc.remove_state(other_dtmc.initial_state, suppress_warning=True)


def test_model_is_stochastic_ctmc_negative_rate():
    """is_stochastic returns False for CTMC with non-positive rate."""
    ctmc = stormvogel.model.new_ctmc()
    s = ctmc.new_state()
    ctmc.set_choices(ctmc.initial_state, [(-1, s)])
    assert not ctmc.is_stochastic()


def test_model_is_stochastic_interval():
    """is_stochastic returns True for interval models."""
    from stormvogel.model.value import Interval

    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(Interval(0.4, 0.6), s)])
    assert dtmc.is_stochastic()


# ── State edge cases ──────────────────────────────────────────────────────────


def test_state_set_labels_new_label_warns():
    """State.set_labels warns and adds a label that's not in the model."""
    import warnings

    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.initial_state
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        state.set_labels({"totally_new_label"})
        assert any("Adding it to the model" in str(x.message) for x in w)
    assert state.has_label("totally_new_label")


def test_state_set_labels_removes_label():
    """State.set_labels removes a label the state previously had."""
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.initial_state
    assert state.has_label("init")
    state.set_labels(set())  # remove all labels
    assert not state.has_label("init")


def test_state_observation_raises_for_dtmc():
    """State.observation raises RuntimeError for non-POMDP model."""
    dtmc = stormvogel.model.new_dtmc()
    with pytest.raises(RuntimeError, match="does not have observations"):
        _ = dtmc.initial_state.observation


def test_state_observation_setter_raises_for_dtmc():
    """State.observation setter raises RuntimeError for non-POMDP model."""
    dtmc = stormvogel.model.new_dtmc()
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("obs")
    with pytest.raises(RuntimeError, match="does not have observations"):
        dtmc.initial_state.observation = obs


def test_state_choices_raises_when_no_choices():
    """State.choices raises RuntimeError when state has no choices."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s = dtmc.new_state()
    del dtmc.transitions[s]
    with pytest.raises(RuntimeError, match="does not have choices"):
        _ = s.choices


def test_state_nr_choices_returns_one_for_empty():
    """State.nr_choices returns 1 for state with empty Choices."""
    dtmc = stormvogel.model.new_dtmc()
    # Initial state has an empty Choices by default (no outgoing transitions set)
    assert dtmc.initial_state.nr_choices == 1


def test_state_get_valuation():
    """State.get_valuation returns the correct value."""
    from stormvogel.model.variable import Variable

    dtmc = stormvogel.model.new_dtmc()
    x = Variable("x")
    dtmc.initial_state.add_valuation(x, 42)
    assert dtmc.initial_state.get_valuation(x) == 42


def test_state_get_branches_no_choices_raises():
    """State.get_branches raises RuntimeError when state has no choices."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s = dtmc.new_state()
    del dtmc.transitions[s]
    with pytest.raises(RuntimeError, match="does not have any choices"):
        s.get_branches()


def test_state_get_branches_mdp_no_action_raises():
    """State.get_branches raises RuntimeError for MDP when action is None and EmptyAction not present."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    with pytest.raises(RuntimeError, match="specific action"):
        mdp.initial_state.get_branches(action=None)


def test_state_get_branches_mdp_wrong_action_raises():
    """State.get_branches raises RuntimeError for unknown action."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a = mdp.new_action("go")
    other_action = stormvogel.model.Action("other")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    with pytest.raises(RuntimeError, match="not found"):
        mdp.initial_state.get_branches(action=other_action)


def test_state_get_branches_dtmc_nonempty_action_raises():
    """State.get_branches raises RuntimeError for DTMC with non-empty action."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(1.0, s)])
    a = stormvogel.model.Action("go")
    with pytest.raises(RuntimeError, match="non-empty actions"):
        dtmc.initial_state.get_branches(action=a)


def test_state_get_outgoing_transitions_mdp_no_action_raises():
    """State.get_outgoing_transitions raises RuntimeError for MDP without action."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    with pytest.raises(RuntimeError, match="specific action"):
        mdp.initial_state.get_outgoing_transitions(action=None)


def test_state_get_outgoing_transitions_returns_none_no_transition():
    """State.get_outgoing_transitions returns None when no transitions exist."""
    dtmc = stormvogel.model.new_dtmc()
    # No choices set — transitions dict has empty Choices
    result = dtmc.initial_state.get_outgoing_transitions()
    assert result is None


def test_state_is_absorbing_true_all_self_loops():
    """State.is_absorbing returns True when all transitions go to self."""
    dtmc = stormvogel.model.new_dtmc()
    dtmc.set_choices(dtmc.initial_state, [(1.0, dtmc.initial_state)])
    assert dtmc.initial_state.is_absorbing()


def test_state_str_with_observation():
    """State.__str__ includes observation for POMDP states."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("my_obs")
    state = pomdp.new_state(observation=obs)
    s = str(state)
    assert "observation" in s
    assert "my_obs" in s


# ── Reward model edge cases ───────────────────────────────────────────────────


def test_reward_model_set_from_rewards_vector_too_short():
    """set_from_rewards_vector raises ValueError when vector is too short."""
    dtmc = stormvogel.model.new_dtmc()
    dtmc.new_state()  # 2 states total
    rm = dtmc.new_reward_model("R")
    with pytest.raises(ValueError, match="too short"):
        rm.set_from_rewards_vector([1])  # only 1 entry for 2 states


def test_reward_model_iter():
    """RewardModel.__iter__ iterates over (state, value) pairs."""
    dtmc = stormvogel.model.new_dtmc()
    rm = dtmc.new_reward_model("R")
    rm.set_state_reward(dtmc.initial_state, 5)
    items = list(rm)
    assert len(items) == 1
    state, value = items[0]
    assert value == 5


def test_reward_model_get_reward_vector_raises_non_numeric():
    """get_reward_vector raises RuntimeError when rewards are non-numeric."""
    dtmc = stormvogel.model.new_dtmc()
    rm = dtmc.new_reward_model("R")
    rm.rewards[dtmc.initial_state] = "not_a_number"  # type: ignore
    with pytest.raises(RuntimeError, match="not all rewards are numeric"):
        rm.get_reward_vector()


# ── value.py edge case ────────────────────────────────────────────────────────


def test_value_to_string_other_type():
    """value_to_string handles fallthrough case for unknown types."""
    from stormvogel.model.value import value_to_string

    result = value_to_string("hello_world")  # type: ignore
    assert result == "hello_world"


# ── variable.py edge case ────────────────────────────────────────────────────


def test_variable_lt_non_variable_returns_notimplemented():
    """Variable.__lt__ returns NotImplemented for non-Variable."""
    from stormvogel.model.variable import Variable

    v = Variable("x")
    result = v.__lt__("not_a_variable")
    assert result is NotImplemented


# ── observation.py edge case ─────────────────────────────────────────────────


def test_model_add_choices_creates_empty_choices_for_untracked_state():
    """Model.add_choices creates an empty Choices for a state not yet in transitions."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    del dtmc.transitions[s]  # manually remove to trigger line 599 path
    dtmc.add_choices(s, [(1.0, dtmc.initial_state)])
    assert s in dtmc.transitions


def test_model_observations_plain_observation():
    """Model.observations yields plain Observation instances."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("plain")
    _state = pomdp.new_state(observation=obs)
    obs_list = list(pomdp.observations)
    assert obs in obs_list


def test_model_observations_includes_distribution_observations():
    """Model.observations yields observations inside Distribution-typed state observations."""
    from stormvogel.model.distribution import Distribution as Dist

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs1 = pomdp.new_observation("oa")
    obs2 = pomdp.new_observation("ob")
    # Assign a Distribution as the state observation
    pomdp.new_state(observation=Dist([(0.5, obs1), (0.5, obs2)]))
    obs_list = list(pomdp.observations)
    assert obs1 in obs_list
    assert obs2 in obs_list


def test_reward_model_set_from_rewards_vector_state_action():
    """set_from_rewards_vector with state_action=True advances combined_id by nr_actions per state."""
    mdp = stormvogel.model.new_mdp()
    s1 = mdp.new_state()
    a1 = mdp.new_action("a")
    a2 = mdp.new_action("b")
    mdp.set_choices(mdp.initial_state, {a1: [(1.0, s1)], a2: [(1.0, s1)]})
    mdp.add_self_loops()
    rm = mdp.new_reward_model("R")
    # init has 2 actions, s1 has 2 actions → need 4 entries, all same per state
    rm.set_from_rewards_vector([10, 10, 20, 20], state_action=True)
    assert rm.get_state_reward(mdp.initial_state) == 10
    assert rm.get_state_reward(s1) == 20


def test_reward_model_set_from_rewards_vector_state_action_too_short_for_actions():
    """set_from_rewards_vector raises ValueError when vector is too short for multi-action state."""
    mdp = stormvogel.model.new_mdp()
    s1 = mdp.new_state()
    a1 = mdp.new_action("a")
    a2 = mdp.new_action("b")
    mdp.set_choices(mdp.initial_state, {a1: [(1.0, s1)], a2: [(1.0, s1)]})
    mdp.add_self_loops()
    rm = mdp.new_reward_model("R")
    # init has 2 actions but we only provide 1 entry for it
    with pytest.raises(ValueError, match="too short"):
        rm.set_from_rewards_vector([10], state_action=True)


def test_reward_model_set_from_rewards_vector_state_action_different_values():
    """set_from_rewards_vector raises ValueError when per-action rewards differ for the same state."""
    mdp = stormvogel.model.new_mdp()
    s1 = mdp.new_state()
    a1 = mdp.new_action("a")
    a2 = mdp.new_action("b")
    mdp.set_choices(mdp.initial_state, {a1: [(1.0, s1)], a2: [(1.0, s1)]})
    mdp.add_self_loops()
    rm = mdp.new_reward_model("R")
    # init has 2 actions; provide 10 and 99 → they differ → should raise
    with pytest.raises(ValueError, match="different values"):
        rm.set_from_rewards_vector([10, 99, 20, 20], state_action=True)


# ── State remaining paths ──────────────────────────────────────────────────────


def test_state_get_branches_mdp_with_empty_action():
    """State.get_branches returns EmptyAction distribution for MDP state using EmptyAction."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    # Set choices using a probability list → creates EmptyAction
    mdp.set_choices(mdp.initial_state, [(1.0, s)])
    branches = mdp.initial_state.get_branches(action=None)
    assert branches is not None


def test_state_get_branches_mdp_valid_action():
    """State.get_branches returns the distribution for a valid action in an MDP."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    branches = mdp.initial_state.get_branches(action=a)
    assert branches is not None


def test_state_get_branches_dtmc_returns_empty_action_distribution():
    """State.get_branches for a DTMC returns the EmptyAction distribution."""
    dtmc = stormvogel.model.new_dtmc()
    s = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(1.0, s)])
    branches = dtmc.initial_state.get_branches()
    assert branches is not None


def test_state_get_outgoing_transitions_mdp_unknown_action_returns_none():
    """State.get_outgoing_transitions returns None for an unknown action in MDP."""
    mdp = stormvogel.model.new_mdp()
    s = mdp.new_state()
    a = mdp.new_action("go")
    unknown = stormvogel.model.Action("unknown")
    mdp.set_choices(mdp.initial_state, {a: [(1.0, s)]})
    result = mdp.initial_state.get_outgoing_transitions(action=unknown)
    assert result is None


def test_state_is_absorbing_true_when_not_in_transitions():
    """State.is_absorbing returns True immediately when state not in transitions."""
    dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
    s = dtmc.new_state()
    del dtmc.transitions[s]
    assert s.is_absorbing()


def test_observation_valuations_missing_raises_runtimeerror():
    """Observation.valuations raises RuntimeError when missing from model."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("obs")
    del pomdp.observation_valuations[obs]
    with pytest.raises(RuntimeError, match="does not have valuations"):
        _ = obs.valuations
