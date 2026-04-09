import pytest
import tempfile
import os
import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.stormpy_utils.model_checking
import stormvogel.model
from model_testing import assert_models_equal

try:
    import stormpy
except ImportError:
    stormpy = None

MA_DRN = """
// Markov Automaton test model
@type: Markov Automaton
@parameters

@reward_models

@nr_states
19
@nr_choices
19
@model
state 0 !1 init tau
	action {a1,a2}
		2 : 0.5
		3 : 0.1
		4 : 0.1
		5 : 0.1
		6 : 0.1
		8 : 0.025
		9 : 0.025
		12 : 0.0125
		13 : 0.0125
		15 : 0.008333333333
		16 : 0.008333333333
		17 : 0.008333333333
state 1 !1 goal deadlock
	action __NOLABEL__
		1 : 1
state 2 !1200 s2
	action __NOLABEL__
		3 : 0.2
		4 : 0.2
		5 : 0.2
		6 : 0.2
		8 : 0.05
		9 : 0.05
		12 : 0.025
		13 : 0.025
		15 : 0.01666666667
		16 : 0.01666666667
		17 : 0.01666666667
state 3 !0.03247623493 s3
	action __NOLABEL__
		8 : 0.25
		9 : 0.25
		12 : 0.125
		13 : 0.125
		15 : 0.08333333333
		16 : 0.08333333333
		17 : 0.08333333333
state 4 !21.17647059 s4
	action __NOLABEL__
		8 : 0.25
		9 : 0.25
		12 : 0.125
		13 : 0.125
		15 : 0.08333333333
		16 : 0.08333333333
		17 : 0.08333333333
state 5 !0.007870067643 s5
	action __NOLABEL__
		8 : 0.25
		9 : 0.25
		12 : 0.125
		13 : 0.125
		15 : 0.08333333333
		16 : 0.08333333333
		17 : 0.08333333333
state 6 !0.03605026124 s6
	action __NOLABEL__
		7 : 0.5
		11 : 0.5
state 7 !0.03356307526 s7
	action __NOLABEL__
		10 : 0.5
		11 : 0.5
state 8 !0.003753228997 s8
	action __NOLABEL__
		13 : 0.5
		15 : 0.1666666667
		16 : 0.1666666667
		17 : 0.1666666667
state 9 !0.02195537721 s9
	action __NOLABEL__
		12 : 0.5
		15 : 0.1666666667
		16 : 0.1666666667
		17 : 0.1666666667
state 10 !0.02061436088 s10
	action __NOLABEL__
		11 : 1
state 11 !0.03356307526 s11
	action __NOLABEL__
		14 : 1
state 12 !0.003753228997 s12
	action __NOLABEL__
		15 : 0.3333333333
		16 : 0.3333333333
		17 : 0.3333333333
state 13 !0.01315352245 s13
	action __NOLABEL__
		15 : 0.3333333333
		16 : 0.3333333333
		17 : 0.3333333333
state 14 !0.0196102706 target
	action __NOLABEL__
		1 : 1
state 15 !0.007358261377 s15
	action __NOLABEL__
		18 : 1
state 16 !0.02139861071 s16
	action __NOLABEL__
		18 : 1
state 17 !0.01778820342 s17
	action __NOLABEL__
		18 : 1
state 18 !0 s18
	action __NOLABEL__
		6 : 1
"""


@pytest.fixture
def stormpy_ma():
    assert stormpy is not None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".drn", delete=False) as f:
        f.write(MA_DRN)
        tmp_path = f.name
    try:
        return stormpy.build_model_from_drn(tmp_path)
    finally:
        os.unlink(tmp_path)


@pytest.fixture
def stormvogel_ma(stormpy_ma):
    return mapping.stormpy_to_stormvogel(stormpy_ma)


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_model_type(stormvogel_ma):
    """The converted model should have type MA."""
    assert stormvogel_ma.model_type == stormvogel.model.ModelType.MA


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_state_count(stormvogel_ma):
    """Converted MA should have 19 states."""
    assert len(stormvogel_ma.states) == 19


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_state_labels(stormvogel_ma):
    """Key state labels should be preserved after conversion."""
    all_labels = set()
    for state in stormvogel_ma.states:
        all_labels.update(state.labels)
    assert "init" in all_labels
    assert "goal" in all_labels
    assert "target" in all_labels
    assert "s18" in all_labels


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_markovian_states(stormvogel_ma, stormpy_ma):
    """Markovian states should be preserved through the conversion."""
    assert stormvogel_ma.markovian_states is not None

    expected_markovian_indices = set(stormpy_ma.markovian_states)
    actual_markovian_indices = {
        i
        for i, s in enumerate(stormvogel_ma.states)
        if s in stormvogel_ma.markovian_states
    }
    assert actual_markovian_indices == expected_markovian_indices


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_roundtrip_stormvogel(stormvogel_ma):
    """stormvogel -> stormpy -> stormvogel should give a structurally equal model."""
    stormpy_ma2 = mapping.stormvogel_to_stormpy(stormvogel_ma)
    stormvogel_ma2 = mapping.stormpy_to_stormvogel(stormpy_ma2)
    assert_models_equal(stormvogel_ma, stormvogel_ma2)


# Expected result vectors computed from storm via stormpy.
# Pmax=? [F "target"]: all states reach target with probability 1,
# except goal (deadlock absorbing state) which has probability 0.
_EXPECTED_REACH = {
    0: 1.0,  # init / tau
    1: 0.0,  # goal (deadlock)
    2: 1.0,  # s2
    3: 1.0,  # s3
    4: 1.0,  # s4
    5: 1.0,  # s5
    6: 1.0,  # s6
    7: 1.0,  # s7
    8: 1.0,  # s8
    9: 1.0,  # s9
    10: 1.0,  # s10
    11: 1.0,  # s11
    12: 1.0,  # s12
    13: 1.0,  # s13
    14: 1.0,  # target
    15: 1.0,  # s15
    16: 1.0,  # s16
    17: 1.0,  # s17
    18: 1.0,  # s18
}

# Tmin=? [F "goal"]: minimum expected time to reach goal.
_EXPECTED_TIME = {
    0: 7.45,  # init / tau
    1: 0.0,  # goal (already there)
    2: 6.95,  # s2
    3: 6.75,  # s3
    4: 6.75,  # s4
    5: 6.75,  # s5
    6: 3.75,  # s6
    7: 3.5,  # s7
    8: 6.25,  # s8
    9: 6.25,  # s9
    10: 3.0,  # s10
    11: 2.0,  # s11
    12: 5.75,  # s12
    13: 5.75,  # s13
    14: 1.0,  # target
    15: 4.75,  # s15
    16: 4.75,  # s16
    17: 4.75,  # s17
    18: 3.75,  # s18
}


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_model_checking_reachability(stormvogel_ma):
    """Pmax=? [F target]: all states reach target with prob 1, except the deadlock (goal)."""
    prop = 'Pmax=? [F "target"]'
    result = stormvogel.stormpy_utils.model_checking.model_checking(
        stormvogel_ma, prop, scheduler=True
    )
    assert result is not None

    for i, state in enumerate(stormvogel_ma.states):
        value = float(result.get_result_of_state(state))
        assert (
            abs(value - _EXPECTED_REACH[i]) < 1e-6
        ), f"state {i} {sorted(state.labels)}: expected {_EXPECTED_REACH[i]}, got {value}"


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_model_checking_expected_time(stormvogel_ma):
    """Tmin=? [F goal]: minimum expected time matches storm's result."""
    prop = 'Tmin=? [F "goal"]'
    result = stormvogel.stormpy_utils.model_checking.model_checking(
        stormvogel_ma, prop, scheduler=False
    )
    assert result is not None

    for i, state in enumerate(stormvogel_ma.states):
        value = float(result.get_result_of_state(state))
        assert (
            abs(value - _EXPECTED_TIME[i]) < 1e-4
        ), f"state {i} {sorted(state.labels)}: expected {_EXPECTED_TIME[i]}, got {value}"


@pytest.mark.skipif(stormpy is None, reason="stormpy is not available")
def test_ma_double_roundtrip_model_checking(stormvogel_ma):
    """Model checking result should be identical after a double roundtrip."""
    stormpy_ma2 = mapping.stormvogel_to_stormpy(stormvogel_ma)
    stormvogel_ma2 = mapping.stormpy_to_stormvogel(stormpy_ma2)

    prop = 'Tmin=? [F "goal"]'
    result1 = stormvogel.stormpy_utils.model_checking.model_checking(
        stormvogel_ma, prop, scheduler=False
    )
    result2 = stormvogel.stormpy_utils.model_checking.model_checking(
        stormvogel_ma2, prop, scheduler=False
    )
    assert result1 is not None
    assert result2 is not None

    v1 = float(result1.get_result_of_state(stormvogel_ma.initial_state))
    v2 = float(result2.get_result_of_state(stormvogel_ma2.initial_state))
    assert abs(v1 - v2) < 1e-6
