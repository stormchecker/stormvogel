"""Model checking tests for interval MCs and interval MDPs via stormpy.

Models are kept small, fully labelled, and use only intervals that do not
contain zero so that all transitions are active under every realization.

stormpy requires an explicit ``UncertaintyResolutionMode`` on the CheckTask:
  - MINIMIZE for Pmin queries
  - MAXIMIZE for Pmax queries
"""

from fractions import Fraction

import pytest

import stormvogel.model as sv_model
import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.stormpy_utils.model_checking as sv_mc
from stormvogel.model.value import Interval

stormpy = pytest.importorskip("stormpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(sp_model, formula: str, mode) -> float:
    """Run interval model checking and return the result at the initial state."""
    props = stormpy.parse_properties(formula)
    task = stormpy.CheckTask(props[0].raw_formula, only_initial_states=False)
    task.set_uncertainty_resolution_mode(mode)
    if sp_model.model_type == stormpy.ModelType.DTMC:
        result = stormpy.check_interval_dtmc(sp_model, task, stormpy.Environment())
    else:
        result = stormpy.check_interval_mdp(sp_model, task, stormpy.Environment())
    return result.at(sp_model.initial_states[0])


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _make_imc_simple():
    """IMC: init --[1/4,3/4]--> A, init --[1/4,3/4]--> B; A and B self-loop."""
    m = sv_model.new_dtmc(create_initial_state=False)
    s0 = m.new_state(labels=["init"])
    sa = m.new_state(labels=["A"])
    sb = m.new_state(labels=["B"])
    m.set_choices(
        s0,
        [
            (Interval(Fraction(1, 4), Fraction(3, 4)), sa),
            (Interval(Fraction(1, 4), Fraction(3, 4)), sb),
        ],
    )
    m.set_choices(sa, [(1, sa)])
    m.set_choices(sb, [(1, sb)])
    return m


def _make_imc_chain():
    """IMC chain: init --[1/4,3/4]--> mid --[1/2,3/4]--> goal (rest to sink).

    Pmin[F goal] = 1/4 * 1/2 = 1/8
    Pmax[F goal] = 3/4 * 3/4 = 9/16
    """
    m = sv_model.new_dtmc(create_initial_state=False)
    s0 = m.new_state(labels=["init"])
    s1 = m.new_state(labels=["mid"])
    goal = m.new_state(labels=["goal"])
    sink = m.new_state(labels=["sink"])
    m.set_choices(
        s0,
        [
            (Interval(Fraction(1, 4), Fraction(3, 4)), s1),
            (Interval(Fraction(1, 4), Fraction(3, 4)), sink),
        ],
    )
    m.set_choices(
        s1,
        [
            (Interval(Fraction(1, 2), Fraction(3, 4)), goal),
            (Interval(Fraction(1, 4), Fraction(1, 2)), sink),
        ],
    )
    m.set_choices(goal, [(1, goal)])
    m.set_choices(sink, [(1, sink)])
    return m


def _make_imdp():
    """IMDP: two actions with complementary interval weights.

    act_lo: A <- [1/4,1/2], B <- [1/2,3/4]
    act_hi: A <- [1/2,3/4], B <- [1/4,1/2]

    Pmin[F A] = 1/4  (act_lo, interval lower)
    Pmax[F A] = 3/4  (act_hi, interval upper)
    """
    m = sv_model.new_mdp(create_initial_state=False)
    s0 = m.new_state(labels=["init"])
    sa = m.new_state(labels=["A"])
    sb = m.new_state(labels=["B"])
    act_lo = m.new_action("lo")
    act_hi = m.new_action("hi")
    m.add_choices(
        s0,
        sv_model.Choices(
            {
                act_lo: sv_model.Distribution(
                    [
                        (Interval(Fraction(1, 4), Fraction(1, 2)), sa),
                        (Interval(Fraction(1, 2), Fraction(3, 4)), sb),
                    ]
                ),
                act_hi: sv_model.Distribution(
                    [
                        (Interval(Fraction(1, 2), Fraction(3, 4)), sa),
                        (Interval(Fraction(1, 4), Fraction(1, 2)), sb),
                    ]
                ),
            }
        ),
    )
    m.set_choices(sa, [(1, sa)])
    m.set_choices(sb, [(1, sb)])
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

MINIMIZE = stormpy.UncertaintyResolutionMode.MINIMIZE
MAXIMIZE = stormpy.UncertaintyResolutionMode.MAXIMIZE


def test_imc_simple_pmin():
    sp_m = mapping.stormvogel_to_stormpy(_make_imc_simple())
    assert pytest.approx(_check(sp_m, 'Pmin=? [F "A"]', MINIMIZE)) == 0.25


def test_imc_simple_pmax():
    sp_m = mapping.stormvogel_to_stormpy(_make_imc_simple())
    assert pytest.approx(_check(sp_m, 'Pmax=? [F "A"]', MAXIMIZE)) == 0.75


def test_imc_chain_pmin():
    """Two-hop chain: Pmin = 1/4 * 1/2 = 1/8."""
    sp_m = mapping.stormvogel_to_stormpy(_make_imc_chain())
    assert pytest.approx(_check(sp_m, 'Pmin=? [F "goal"]', MINIMIZE)) == 0.125


def test_imc_chain_pmax():
    """Two-hop chain: Pmax = 3/4 * 3/4 = 9/16."""
    sp_m = mapping.stormvogel_to_stormpy(_make_imc_chain())
    assert pytest.approx(_check(sp_m, 'Pmax=? [F "goal"]', MAXIMIZE)) == pytest.approx(
        9 / 16
    )


def test_imdp_pmin():
    sp_m = mapping.stormvogel_to_stormpy(_make_imdp())
    assert pytest.approx(_check(sp_m, 'Pmin=? [F "A"]', MINIMIZE)) == 0.25


def test_imdp_pmax():
    sp_m = mapping.stormvogel_to_stormpy(_make_imdp())
    assert pytest.approx(_check(sp_m, 'Pmax=? [F "A"]', MAXIMIZE)) == 0.75


def test_interval_dtmc_via_model_checking_raises():
    """model_checking() on an interval DTMC should raise ValueError."""
    m = _make_imc_simple()
    with pytest.raises(ValueError, match="check_interval_dtmc"):
        sv_mc.model_checking(m, 'Pmin=? [F "A"]')
