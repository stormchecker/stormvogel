"""Roundtrip tests for the stormvogel ↔ UMBI translation layer."""

import tempfile
import os

import pytest

pytest.importorskip("umbi")

import stormvogel.examples as ex
import stormvogel.model as sv
import stormvogel.umbi as svu
from stormvogel.model.variable import Variable, Predicate, IntDomain, BoolDomain


def _roundtrip(model):
    """Translate model → SimpleAts → model and return the result."""
    ats = svu.translate_to_umbi(model)
    ats.validate()
    return svu.translate_to_stormvogel(ats)


def _file_roundtrip(model):
    """Translate model → .umb file → model via write_to_umb / read_from_umb."""
    with tempfile.NamedTemporaryFile(suffix=".umb", delete=False) as f:
        path = f.name
    try:
        svu.write_to_umb(model, path)
        return svu.read_from_umb(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Basic model-type roundtrips
# ---------------------------------------------------------------------------


def test_dtmc_roundtrip():
    m = ex.create_die_dtmc()
    m2 = _roundtrip(m)
    assert m2.nr_states == m.nr_states
    assert m2.nr_choices == m.nr_choices


def test_ctmc_roundtrip():
    m = ex.create_nuclear_fusion_ctmc()
    m2 = _roundtrip(m)
    assert m2.nr_states == m.nr_states
    assert m2.nr_choices == m.nr_choices


def test_mdp_roundtrip():
    m = ex.create_monty_hall_mdp()
    m2 = _roundtrip(m)
    assert m2.nr_states == m.nr_states
    assert m2.nr_choices == m.nr_choices


def test_pomdp_roundtrip():
    m = ex.create_cheese_maze()
    m2 = _roundtrip(m)
    assert m2.nr_states == m.nr_states
    assert m2.nr_choices == m.nr_choices
    # Observations must be preserved in count
    obs_before = len(set(m.state_observations.values()))
    obs_after = len(set(m2.state_observations.values()))
    assert obs_after == obs_before


# ---------------------------------------------------------------------------
# File I/O roundtrips
# ---------------------------------------------------------------------------


def test_dtmc_file_roundtrip():
    m = ex.create_die_dtmc()
    m2 = _file_roundtrip(m)
    assert m2.nr_states == m.nr_states


def test_ctmc_file_roundtrip():
    m = ex.create_nuclear_fusion_ctmc()
    m2 = _file_roundtrip(m)
    assert m2.nr_states == m.nr_states


# ---------------------------------------------------------------------------
# Labels and APs
# ---------------------------------------------------------------------------


def test_labels_preserved():
    m = ex.create_die_dtmc()
    m2 = _roundtrip(m)
    for label in m.state_labels:
        assert label in m2.state_labels
        assert len(m2.state_labels[label]) == len(m.state_labels[label])


# ---------------------------------------------------------------------------
# State rewards
# ---------------------------------------------------------------------------


def test_state_rewards_roundtrip():
    m = sv.new_dtmc()
    init = m.initial_state
    s1 = m.new_state("a")
    s2 = m.new_state("b")
    init.set_choices([(0.5, s1), (0.5, s2)])
    s1.set_choices([(1.0, s1)])
    s2.set_choices([(1.0, s2)])

    rm = m.new_reward_model("steps")
    rm.set_state_reward(init, 0.0)
    rm.set_state_reward(s1, 1.0)
    rm.set_state_reward(s2, 2.5)

    m2 = _roundtrip(m)
    assert len(m2.rewards) == 1
    rm2 = m2.rewards[0]
    assert rm2.name == "steps"
    rewards2 = [rm2.get_state_reward(s) or 0.0 for s in m2.states]
    assert rewards2 == [0.0, 1.0, 2.5]


def test_multiple_reward_models():
    m = sv.new_dtmc()
    init = m.initial_state
    s1 = m.new_state()
    init.set_choices([(1.0, s1)])
    s1.set_choices([(1.0, s1)])

    m.new_reward_model("cost").set_state_reward(init, 3.0)
    m.new_reward_model("time").set_state_reward(s1, 7.0)

    m2 = _roundtrip(m)
    assert {r.name for r in m2.rewards} == {"cost", "time"}


# ---------------------------------------------------------------------------
# State valuations
# ---------------------------------------------------------------------------


def test_state_valuations_roundtrip():
    m = sv.new_dtmc()
    init = m.initial_state
    s1 = m.new_state()
    init.set_choices([(1.0, s1)])
    s1.set_choices([(1.0, s1)])

    x = Variable("x", IntDomain(0, 10))
    m.state_valuations[init][x] = 0
    m.state_valuations[s1][x] = 5

    m2 = _roundtrip(m)
    # Variable names must be preserved
    all_var_names = {var.label for vals in m2.state_valuations.values() for var in vals}
    assert "x" in all_var_names
    # Values must match by state index
    vals_by_index = [
        next((v for k, v in m2.state_valuations[s].items() if k.label == "x"), None)
        for s in m2.states
    ]
    assert vals_by_index == [0, 5]


def test_state_valuations_multiple_vars():
    m = sv.new_dtmc()
    init = m.initial_state
    s1 = m.new_state()
    init.set_choices([(1.0, s1)])
    s1.set_choices([(1.0, s1)])

    x = Variable("x", IntDomain(0, 10))
    flag = Variable("flag", BoolDomain())
    m.state_valuations[init][x] = 3
    m.state_valuations[init][flag] = True
    m.state_valuations[s1][x] = 7
    m.state_valuations[s1][flag] = False

    m2 = _roundtrip(m)
    vars_by_name = {
        var.label: var for vals in m2.state_valuations.values() for var in vals
    }
    assert {"x", "flag"} <= vars_by_name.keys()
    assert isinstance(vars_by_name["x"].domain, IntDomain)
    assert vars_by_name["x"].domain.lo == 3 and vars_by_name["x"].domain.hi == 7
    assert isinstance(vars_by_name["flag"].domain, BoolDomain)


# ---------------------------------------------------------------------------
# CTMC: exit rates preserved up to normalisation
# ---------------------------------------------------------------------------


def test_observation_valuations_roundtrip():
    m = sv.new_pomdp()
    val_var = Variable("value", IntDomain(0, 10))
    obs_a = m.new_observation("a", {val_var: 3})
    obs_b = m.new_observation("b", {val_var: 7})

    init = m.initial_state
    init.observation = obs_a
    s1 = m.new_state(observation=obs_b)
    init.set_choices([(1.0, s1)])
    s1.set_choices([(1.0, s1)])

    m2 = _roundtrip(m)

    # Values must be preserved; aliases map to "0" / "1" after roundtrip
    values_by_name = {
        next(iter(vals.values())) for vals in m2.observation_valuations.values() if vals
    }
    assert values_by_name == {3, 7}


def test_predicate_observation_valuations_roundtrip():
    """Predicate-keyed observation valuations are exported as UMBI variables."""
    m = sv.new_pomdp()
    pred = Predicate("is_ripe", BoolDomain())
    obs_a = m.new_observation("a", {pred: True})
    obs_b = m.new_observation("b", {pred: False})

    init = m.initial_state
    init.observation = obs_a
    s1 = m.new_state(observation=obs_b)
    init.set_choices([(1.0, s1)])
    s1.set_choices([(1.0, s1)])

    m2 = _roundtrip(m)

    all_var_names = {
        var.label for vals in m2.observation_valuations.values() for var in vals
    }
    assert "is_ripe" in all_var_names
    values = {
        next(iter(vals.values())) for vals in m2.observation_valuations.values() if vals
    }
    assert values == {True, False}


def test_unsupported_rewards_raises():
    """Choice- and branch-level rewards raise by default."""
    import umbi.ats.examples as umbi_ex

    ats = umbi_ex.grid("i..\n...\n..g")
    with pytest.raises(ValueError, match="ignore_unsupported_rewards"):
        svu.translate_to_stormvogel(ats)


def test_unsupported_rewards_ignored_with_flag():
    """Setting ignore_unsupported_rewards=True skips unsupported rewards."""
    import umbi.ats.examples as umbi_ex

    ats = umbi_ex.grid("i..\n...\n..g")
    m = svu.translate_to_stormvogel(ats, ignore_unsupported_rewards=True)
    assert m.nr_states == 9
    assert m.rewards == []


def test_transition_rewards_raise():
    """Transition rewards in stormvogel raise by default."""
    from stormvogel.model.action import EmptyAction

    m = sv.new_dtmc()
    init = m.initial_state
    s1 = m.new_state()
    init.set_choices([(1.0, s1)])
    s1.set_choices([(1.0, s1)])
    m.new_reward_model("tr").set_transition_reward(init, EmptyAction, s1, 1.0)

    with pytest.raises(ValueError, match="ignore_unsupported_rewards"):
        svu.translate_to_umbi(m)

    ats = svu.translate_to_umbi(m, ignore_unsupported_rewards=True)
    assert not ats.has_reward_annotations


def test_parametric_model_raises():
    """Parametric models raise a clear error."""
    m = sv.new_dtmc()
    p = m.declare_parameter("p")
    init = m.initial_state
    s1 = m.new_state()
    init.set_choices([(p, s1)])
    s1.set_choices([(1, s1)])

    with pytest.raises(ValueError, match="Parametric"):
        svu.translate_to_umbi(m)


def _make_ats_with_choice_annotations():
    """Minimal MDP ATS with choice-level variable valuations."""
    import umbi.ats
    from umbi.ats.simple_ats import SimpleAts

    ats = SimpleAts()
    ats.time = umbi.ats.TimeType.DISCRETE
    ats.num_players = 1
    ats.new_states(2)
    ats.initial_states = [0]
    ats.num_choice_actions = 1
    ats.new_choice_action_to_name()
    ats.choice_action_to_name[0] = "a"
    c0 = ats.new_state_choice(0)
    ats.choice_to_choice_action[c0] = 0
    ats.new_choice_branch(c0, 1, 1)
    c1 = ats.new_state_choice(1)
    ats.choice_to_choice_action[c1] = 0
    ats.new_choice_branch(c1, 1, 1)
    vv = ats.add_variable_valuations()
    cv = vv.add_choice_valuations()
    var = cv.new_variable("cost")
    cv.set_entity_valuation(c0, {var: 5})
    cv.set_entity_valuation(c1, {var: 3})
    ats.validate()
    return ats


def _make_ats_with_branch_annotations():
    """Minimal DTMC ATS with branch-level variable valuations."""
    import umbi.ats
    from umbi.ats.simple_ats import SimpleAts

    ats = SimpleAts()
    ats.time = umbi.ats.TimeType.DISCRETE
    ats.num_players = 0
    ats.new_states(2)
    ats.initial_states = [0]
    c0 = ats.new_state_choice(0)
    b0 = ats.new_choice_branch(c0, 1, 1)
    c1 = ats.new_state_choice(1)
    b1 = ats.new_choice_branch(c1, 1, 1)
    vv = ats.add_variable_valuations()
    bv = vv.add_branch_valuations()
    var = bv.new_variable("weight")
    bv.set_entity_valuation(b0, {var: 1})
    bv.set_entity_valuation(b1, {var: 2})
    ats.validate()
    return ats


def test_choice_annotations_raise():
    ats = _make_ats_with_choice_annotations()
    with pytest.raises(ValueError, match="ignore_choice_annotations"):
        svu.translate_to_stormvogel(ats)


def test_choice_annotations_ignored():
    ats = _make_ats_with_choice_annotations()
    m = svu.translate_to_stormvogel(ats, ignore_choice_annotations=True)
    assert m.nr_states == 2


def test_branch_annotations_raise():
    ats = _make_ats_with_branch_annotations()
    with pytest.raises(ValueError, match="ignore_branch_annotations"):
        svu.translate_to_stormvogel(ats)


def test_branch_annotations_ignored():
    ats = _make_ats_with_branch_annotations()
    m = svu.translate_to_stormvogel(ats, ignore_branch_annotations=True)
    assert m.nr_states == 2


def test_ctmc_rates_roundtrip():
    """
    In stormvogel CTMCs, transition values are rates.
    After a roundtrip the rates must match the originals.
    """
    m = ex.create_nuclear_fusion_ctmc()
    m2 = _roundtrip(m)

    def collect_rates(model):
        rates = {}
        for i, state in enumerate(model.states):
            if state.has_choices():
                for _, branch in state.choices:
                    for rate, _ in branch:
                        rates[i] = float(rate)
        return rates

    r1 = collect_rates(m)
    r2 = collect_rates(m2)
    for i in r1:
        assert (
            abs(r1[i] - r2[i]) < 1e-9
        ), f"Rate mismatch at state {i}: {r1[i]} vs {r2[i]}"
