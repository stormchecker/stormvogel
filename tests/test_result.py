import stormvogel.stormpy_utils.convert_results as convert_results
import pytest
import stormvogel.examples.monty_hall
import stormvogel.model
import stormvogel.result
from typing import cast
from model_testing import assert_models_equal


def test_convert_model_checker_results_dtmc():
    stormpy = pytest.importorskip("stormpy")
    import stormpy.examples.files

    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0])

    stormvogel_model = stormvogel.stormpy_utils.mapping.stormpy_to_stormvogel(model)
    assert stormvogel_model is not None
    stormvogel_result = convert_results.convert_model_checking_result(
        stormvogel_model, result
    )
    assert stormvogel_result is not None
    assert pytest.approx(list(stormvogel_result.values.values())) == [
        0.16666666666666669,
        0.33333333333333337,
        0.0,
        0.16666666666666674,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_convert_model_checker_results_dtmc_qualitative():
    stormpy = pytest.importorskip("stormpy")
    import stormpy.examples.files

    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P>=0.5 [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0])

    stormvogel_model = stormvogel.stormpy_utils.mapping.stormpy_to_stormvogel(model)
    assert stormvogel_model is not None
    stormvogel_result = convert_results.convert_model_checking_result(
        stormvogel_model, result
    )
    assert stormvogel_result is not None

    assert pytest.approx(list(stormvogel_result.values.values())) == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_convert_model_checker_results_mdp():
    stormpy = pytest.importorskip("stormpy")
    import stormpy.examples.files

    path = stormpy.examples.files.prism_mdp_coin_2_2

    prism_program = stormpy.parse_prism_program(path)
    formula_str = 'Pmin=? [F "finished" & "all_coins_equal_1"]'
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)

    result = stormpy.model_checking(model, properties[0], extract_scheduler=True)

    stormvogel_model = stormvogel.stormpy_utils.mapping.stormpy_to_stormvogel(model)

    assert stormvogel_model is not None
    stormvogel_result = convert_results.convert_model_checking_result(
        stormvogel_model, result
    )
    assert stormvogel_result is not None

    assert pytest.approx(list(stormvogel_result.values.values())) == [
        0.3828112753064229,
        0.265623936195459,
        0.4999986144173868,
        0.265623936195459,
        0.4999986144173868,
        0.265623936195459,
        0.15624925797353115,
        0.3749986144173868,
        0.5039048397998056,
        0.3749986144173868,
        0.6249986144173868,
        0.265623936195459,
        0.5039048397998056,
        0.265623936195459,
        0.15624925797353115,
        0.3749986144173868,
        0.15624925797353115,
        0.3828110651822245,
        0.5039048397998056,
        0.3828110651822245,
        0.6249986144173868,
        0.3749986144173868,
        0.6249986144173868,
        0.15624925797353115,
        0.3749986144173868,
        0.15624925797353115,
        0.3749986144173868,
        0.15624925797353115,
        0.3828110651822245,
        0.3828110651822245,
        0.6249986144173868,
        0.3828110651822245,
        0.6249986144173868,
        0.6269517249967417,
        0.15624925797353115,
        0.06249966380128266,
        0.24999885214577963,
        0.3828110651822245,
        0.24999885214577963,
        0.49999837668899394,
        0.15624925797353115,
        0.3828110651822245,
        0.26562375367545515,
        0.49999837668899394,
        0.6269517249967417,
        0.49999837668899394,
        0.7499988521457797,
        0.6269517249967417,
        0.15624925797353115,
        0.06249966380128266,
        0.24999885214577963,
        0.06249966380128266,
        0.26562375367545515,
        0.26562375367545515,
        0.49999837668899394,
        0.24999885214577963,
        0.49999837668899394,
        0.5039045978477037,
        0.6269517249967417,
        0.5039045978477037,
        0.7499988521457797,
        0.7499988521457797,
        0.06249966380128266,
        0.24999885214577963,
        0.06249966380128266,
        0.24999885214577963,
        0.06249966380128266,
        0.26562375367545515,
        0.5039045978477037,
        0.5039045978477037,
        0.7499988521457797,
        0.5039045978477037,
        0.7499988521457797,
        0.7509754061983613,
        0.06249966380128266,
        0.0,
        0.12499932760256532,
        0.12499932760256532,
        0.37499837668899394,
        0.06249966380128266,
        0.38281081900641334,
        0.6249983766889939,
        0.7509754061983613,
        0.6249983766889939,
        0.8749993276025654,
        0.7509754061983613,
        0.06249966380128266,
        0.0,
        0.12499932760256532,
        0.0,
        0.12499932760256532,
        0.6269514847941573,
        0.7509754061983613,
        0.6269514847941573,
        0.8749993276025654,
        0.8749993276025654,
        0.0,
        0.12499932760256532,
        0.0,
        0.12499932760256532,
        0.0,
        0.6269514847941573,
        0.8749993276025654,
        0.6269514847941573,
        0.8749993276025654,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.24999865520513065,
        0.0,
        0.0,
        0.0,
        0.5039043143831838,
        0.7499986552051308,
        0.8754876039041822,
        0.7499986552051308,
        1.0,
        0.8754876039041822,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.7509752078083645,
        0.937499790931729,
        0.7509752078083645,
        1.0,
        1.0,
        0.937499790931729,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.8749995818634579,
        0.8749995818634579,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.8749995818634579,
        0.8749995818634579,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.8749995818634579,
        0.8749995818634579,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.7499991637269159,
        1.0,
        0.7499991637269159,
        1.0,
        0.0,
        0.0,
        0.7499991637269159,
        0.7499991637269159,
        0.0,
        0.0,
        0.7499991637269159,
        0.7499991637269159,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6249988173312626,
        0.8749995101225692,
        0.6249988173312626,
        0.8749995101225692,
        0.0,
        0.0,
        0.6249988173312626,
        0.6249988173312626,
        0.0,
        0.0,
        0.6249988173312626,
        0.6249988173312626,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4999986144173868,
        0.7499990202451383,
        0.4999986144173868,
        0.7499990202451383,
        0.0,
        0.0,
        0.4999986144173868,
        0.4999986144173868,
        0.0,
        0.0,
        0.4999986144173868,
        0.4999986144173868,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3749986144173868,
        0.6249986144173868,
        0.3749986144173868,
        0.6249986144173868,
        0.0,
        0.0,
        0.3749986144173868,
        0.3749986144173868,
        0.0,
        0.0,
        0.3749986144173868,
        0.3749986144173868,
        0.0,
        0.0,
        0.0,
        0.0,
        0.24999885214577963,
        0.49999837668899394,
        0.24999885214577963,
        0.49999837668899394,
        0.0,
        0.0,
        0.24999885214577963,
        0.24999885214577963,
        0.0,
        0.0,
        0.24999885214577963,
        0.24999885214577963,
        0.0,
        0.0,
        0.0,
        0.0,
        0.12499932760256532,
        0.37499837668899394,
        0.12499932760256532,
        0.37499837668899394,
        0.0,
        0.0,
        0.12499932760256532,
        0.12499932760256532,
        0.0,
        0.0,
        0.12499932760256532,
        0.12499932760256532,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.24999865520513065,
        0.0,
        0.24999865520513065,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    assert stormvogel_result.scheduler is not None
    assert [
        0,
        2,
        5,
        6,
        8,
        10,
        12,
        14,
        16,
        19,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        43,
        45,
        46,
        49,
        50,
        52,
        54,
        56,
        58,
        61,
        62,
        64,
        66,
        68,
        70,
        72,
        75,
        77,
        78,
        80,
        82,
        84,
        86,
        88,
        91,
        92,
        94,
        96,
        98,
        100,
        102,
        104,
        107,
        108,
        111,
        113,
        114,
        116,
        118,
        120,
        123,
        124,
        127,
        128,
        130,
        132,
        134,
        136,
        139,
        141,
        142,
        144,
        146,
        148,
        150,
        152,
        155,
        156,
        158,
        160,
        162,
        164,
        167,
        168,
        170,
        172,
        174,
        176,
        178,
        181,
        183,
        184,
        186,
        188,
        191,
        192,
        195,
        196,
        198,
        200,
        203,
        205,
        206,
        208,
        210,
        212,
        214,
        216,
        219,
        220,
        222,
        224,
        225,
        226,
        228,
        231,
        233,
        234,
        236,
        238,
        239,
        240,
        241,
        243,
        245,
        248,
        249,
        250,
        251,
        253,
        255,
        256,
        258,
        260,
        261,
        262,
        263,
        264,
        266,
        267,
        268,
        269,
        270,
        271,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
        298,
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
        327,
        328,
        329,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        352,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        360,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        382,
        383,
        384,
        385,
        386,
        387,
        388,
        389,
        390,
        391,
        392,
        393,
        394,
        395,
        396,
        397,
        398,
        399,
    ] == [
        int(action.label)
        for action in stormvogel_result.scheduler.taken_actions.values()
        if action.label is not None
    ]


def test_convert_model_checker_results_mdp_qualitative():
    stormpy = pytest.importorskip("stormpy")
    import stormpy.examples.files

    path = stormpy.examples.files.prism_mdp_coin_2_2

    prism_program = stormpy.parse_prism_program(path)
    formula_str = 'P>=0.5 [F "finished" & "all_coins_equal_1"]'
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0], extract_scheduler=True)

    stormvogel_model = stormvogel.stormpy_utils.mapping.stormpy_to_stormvogel(model)
    assert stormvogel_model is not None
    stormvogel_result = convert_results.convert_model_checking_result(
        stormvogel_model, result
    )
    assert stormvogel_result is not None

    assert pytest.approx(list(stormvogel_result.values.values())) == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_induced_dtmc():
    # we create a simple mdp
    mdp = stormvogel.model.new_mdp()
    state1 = mdp.new_state()
    state2 = mdp.new_state()
    action0 = mdp.new_action("0")
    action1 = mdp.new_action("1")
    branch0 = stormvogel.model.Distribution([(1 / 2, state1), (1 / 2, state2)])
    branch1 = stormvogel.model.Distribution([(1 / 4, state1), (3 / 4, state2)])
    transition = stormvogel.model.Choices({action0: branch0, action1: branch1})
    mdp.set_choices(mdp.initial_state, transition)
    mdp.add_self_loops()

    # we set rewards (because we must also check if they are carried over)
    rewardmodel = mdp.new_reward_model("r1")
    for i, state in enumerate(mdp.states):
        rewardmodel.set_state_reward(state, i)

    # we create the induced dtmc
    chosen_actions = dict()
    for state in mdp:
        chosen_actions[state] = list(mdp.transitions[state])[0][0]
    scheduler = stormvogel.result.Scheduler(mdp, chosen_actions)

    dtmc = scheduler.generate_induced_dtmc()

    # we create what the induced dtmc should look like
    other_dtmc = stormvogel.model.new_dtmc()
    state1 = other_dtmc.new_state()
    state2 = other_dtmc.new_state()
    branch0 = stormvogel.model.Distribution(
        cast(
            list[tuple[stormvogel.model.Value, stormvogel.model.State]],
            [(1 / 2, state1), (1 / 2, state2)],
        )
    )
    transition = stormvogel.model.Choices({stormvogel.model.EmptyAction: branch0})
    other_dtmc.set_choices(other_dtmc.initial_state, transition)
    other_dtmc.add_self_loops()

    # and the rewards of the induced dtmc
    rewardmodel = other_dtmc.new_reward_model("r1")
    rewardmodel.set_state_reward(other_dtmc.states[0], 0)
    rewardmodel.set_state_reward(other_dtmc.states[1], 1)
    rewardmodel.set_state_reward(other_dtmc.states[2], 2)

    assert dtmc is not None
    assert_models_equal(dtmc, other_dtmc)


def test_random_scheduler():
    lion = stormvogel.examples.create_lion_mdp()
    sched = stormvogel.result.random_scheduler(lion)
    for state in lion:
        sched.get_action_at_state(state)


def test_random_scheduler_raises_on_no_actions():
    """random_scheduler must raise ValueError when a state has no available actions."""
    mdp = stormvogel.model.new_mdp()
    # initial_state exists but has no choices set -> no available actions
    with pytest.raises(ValueError, match="no available actions"):
        stormvogel.result.random_scheduler(mdp)


def test_random_scheduler_covers_all_states():
    """random_scheduler must map every state, not just those with actions."""
    mdp = stormvogel.model.new_mdp()
    s1 = mdp.new_state()
    action = mdp.new_action("go")
    mdp.set_choices(
        mdp.initial_state,
        {action: [(1.0, s1)]},
    )
    mdp.add_self_loops()

    sched = stormvogel.result.random_scheduler(mdp)
    for state in mdp:
        # Must not raise KeyError
        sched.get_action_at_state(state)


def test_result_eq_different_state_count():
    """Result.__eq__ must return False, not raise IndexError, when models differ in state count."""
    # Model A: 3 states
    model_a = stormvogel.model.new_dtmc()
    sa1 = model_a.new_state()
    sa2 = model_a.new_state()
    model_a.set_choices(model_a.initial_state, [(0.5, sa1), (0.5, sa2)])
    model_a.add_self_loops()

    # Model B: 2 states (fewer than A)
    model_b = stormvogel.model.new_dtmc()
    sb1 = model_b.new_state()
    model_b.set_choices(model_b.initial_state, [(1.0, sb1)])
    model_b.add_self_loops()

    # Same number of values to bypass len(values) guard; different state counts
    values_a: dict[stormvogel.model.State, stormvogel.model.Value] = {
        model_a.states[0]: 1.0,
        model_a.states[1]: 2.0,
    }
    values_b: dict[stormvogel.model.State, stormvogel.model.Value] = {
        model_b.states[0]: 1.0,
        model_b.states[1]: 2.0,
    }

    result_a = stormvogel.result.Result(model_a, values_a)
    result_b = stormvogel.result.Result(model_b, values_b)

    # Before the fix this would raise IndexError; now it returns False
    assert result_a != result_b


def test_result_eq_same_models():
    """Two identical Results must still compare equal."""
    model_a = stormvogel.model.new_dtmc()
    s1 = model_a.new_state()
    model_a.set_choices(model_a.initial_state, [(1.0, s1)])
    model_a.add_self_loops()

    model_b = stormvogel.model.new_dtmc()
    s1b = model_b.new_state()
    model_b.set_choices(model_b.initial_state, [(1.0, s1b)])
    model_b.add_self_loops()

    values_a: dict[stormvogel.model.State, stormvogel.model.Value] = {
        state: float(i) for i, state in enumerate(model_a.states)
    }
    values_b: dict[stormvogel.model.State, stormvogel.model.Value] = {
        state: float(i) for i, state in enumerate(model_b.states)
    }

    result_a = stormvogel.result.Result(model_a, values_a)
    result_b = stormvogel.result.Result(model_b, values_b)

    assert result_a == result_b


def _make_simple_result():
    """Helper: DTMC with 3 states and a Result where init=0.1, s1=0.5, s2=0.9."""
    m = stormvogel.model.new_dtmc()
    s1 = m.new_state("s1")
    s2 = m.new_state("s2")
    m.set_choices(m.initial_state, [(0.5, s1), (0.5, s2)])
    m.add_self_loops()
    values: dict[stormvogel.model.State, stormvogel.model.Value] = {
        m.initial_state: 0.1,
        s1: 0.5,
        s2: 0.9,
    }
    return stormvogel.result.Result(m, values), m


def test_result_filter():
    result, _ = _make_simple_result()
    high = result.filter(lambda v: v > 0.4)  # type: ignore[operator]
    assert len(high) == 2  # s1=0.5 and s2=0.9


def test_result_filter_true():
    m = stormvogel.model.new_dtmc()
    s1 = m.new_state("s1")
    m.set_choices(m.initial_state, [(1.0, s1)])
    m.add_self_loops()
    values: dict[stormvogel.model.State, stormvogel.model.Value] = {
        m.initial_state: False,
        s1: True,
    }
    result = stormvogel.result.Result(m, values)
    true_states = result.filter_true()
    assert len(true_states) == 1
    assert s1 in true_states


def test_result_maximum_result():
    result, _ = _make_simple_result()
    assert result.maximum_result() == pytest.approx(0.9)


def test_result_maximum_result_raises_for_interval():
    from stormvogel.model.value import Interval

    m = stormvogel.model.new_dtmc()
    s1 = m.new_state("s1")
    m.set_choices(m.initial_state, [(1.0, s1)])
    m.add_self_loops()
    values: dict[stormvogel.model.State, stormvogel.model.Value] = {
        m.initial_state: Interval(0, 1),
        s1: Interval(0, 1),
    }
    result = stormvogel.result.Result(m, values)
    with pytest.raises(RuntimeError, match="interval"):
        result.maximum_result()


def test_result_str():
    result, _ = _make_simple_result()
    s = str(result)
    # The string must contain the section headers produced by Result.__str__
    assert "values" in s
    assert "scheduler" in s
    # And the actual result values for each state must appear in the output
    assert "0.1" in s
    assert "0.5" in s
    assert "0.9" in s


def test_result_iter():
    result, m = _make_simple_result()
    items = list(result)
    assert len(items) == 3
    states = [s for s, _ in items]
    assert m.initial_state in states


def test_scheduler_get_action_raises_for_unknown_state():
    mdp = stormvogel.model.new_mdp()
    s1 = mdp.new_state()
    act = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {act: [(1.0, s1)]})
    mdp.add_self_loops()

    other_mdp = stormvogel.model.new_mdp()
    other_state = other_mdp.new_state()

    scheduler = stormvogel.result.Scheduler(mdp, {mdp.initial_state: act, s1: act})
    with pytest.raises(RuntimeError, match="not a part of the model"):
        scheduler.get_action_at_state(other_state)


def test_scheduler_str():
    mdp = stormvogel.model.new_mdp()
    s1 = mdp.new_state()
    act = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {act: [(1.0, s1)]})
    mdp.add_self_loops()
    sched = stormvogel.result.Scheduler(mdp, {mdp.initial_state: act, s1: act})
    s = str(sched)
    # The prefix produced by Scheduler.__str__ must be present
    assert "taken actions" in s
    # The actual action label must appear (Action.__str__ returns its label)
    assert "go" in s


def test_generate_induced_dtmc_returns_none_for_non_mdp():
    dtmc = stormvogel.model.new_dtmc()
    s1 = dtmc.new_state()
    dtmc.set_choices(dtmc.initial_state, [(1.0, s1)])
    dtmc.add_self_loops()
    taken_actions = {
        dtmc.initial_state: stormvogel.model.EmptyAction,
        s1: stormvogel.model.EmptyAction,
    }
    sched = stormvogel.result.Scheduler(dtmc, taken_actions)
    assert sched.generate_induced_dtmc() is None
