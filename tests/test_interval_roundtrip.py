"""Roundtrip tests for interval models: stormvogel → stormpy → stormvogel.

Each helper builds a small, fully-labelled model so that states are
unambiguously identifiable before and after the roundtrip.  We then call
assert_models_equal to check that labels, model type, and interval-valued
transition weights are all preserved.
"""

from fractions import Fraction

import pytest

import stormvogel.model as sv_model
import stormvogel.stormpy_utils.mapping as mapping
from stormvogel.model.value import Interval
from model_testing import assert_models_equal

stormpy = pytest.importorskip("stormpy")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _make_imc():
    """IMC: init --[1/4,3/4]--> A, init --[1/4,3/4]--> B; A and B self-loop.

    States are labelled "init", "A", "B" so they survive the roundtrip
    without relying on ordering.
    """
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


def _make_imdp():
    """IMDP: init has two actions with complementary interval weights to A/B.

    Action "lo": A gets [1/4,1/2], B gets [1/2,3/4].
    Action "hi": A gets [1/2,3/4], B gets [1/4,1/2].
    A and B are absorbing.
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
# Roundtrip tests
# ---------------------------------------------------------------------------


def test_imc_roundtrip():
    m = _make_imc()
    assert m.is_interval_model()

    stormpy_m = mapping.stormvogel_to_stormpy(m)
    m2 = mapping.stormpy_to_stormvogel(stormpy_m)

    assert m2.is_interval_model()
    assert_models_equal(m, m2)


def test_imdp_roundtrip():
    m = _make_imdp()
    assert m.is_interval_model()

    stormpy_m = mapping.stormvogel_to_stormpy(m)
    m2 = mapping.stormpy_to_stormvogel(stormpy_m)

    assert m2.is_interval_model()
    assert_models_equal(m, m2)


def test_imc_labels_survive_roundtrip():
    """Labels on every state are preserved exactly through the roundtrip."""
    m = _make_imc()
    stormpy_m = mapping.stormvogel_to_stormpy(m)
    m2 = mapping.stormpy_to_stormvogel(stormpy_m)

    for label in ("init", "A", "B"):
        original = m.get_states_with_label(label)
        recovered = m2.get_states_with_label(label)
        assert (
            len(original) == 1
        ), f"expected one state with label {label!r} in original"
        assert (
            len(recovered) == 1
        ), f"expected one state with label {label!r} after roundtrip"


def test_imdp_labels_survive_roundtrip():
    """Labels on every state are preserved exactly through the roundtrip."""
    m = _make_imdp()
    stormpy_m = mapping.stormvogel_to_stormpy(m)
    m2 = mapping.stormpy_to_stormvogel(stormpy_m)

    for label in ("init", "A", "B"):
        original = m.get_states_with_label(label)
        recovered = m2.get_states_with_label(label)
        assert (
            len(original) == 1
        ), f"expected one state with label {label!r} in original"
        assert (
            len(recovered) == 1
        ), f"expected one state with label {label!r} after roundtrip"


def test_imc_intervals_survive_roundtrip():
    """Interval bounds on transitions are preserved through the roundtrip."""
    m = _make_imc()
    stormpy_m = mapping.stormvogel_to_stormpy(m)
    m2 = mapping.stormpy_to_stormvogel(stormpy_m)

    init2 = next(iter(m2.get_states_with_label("init")))
    sa2 = next(iter(m2.get_states_with_label("A")))
    sb2 = next(iter(m2.get_states_with_label("B")))

    _, branch = next(iter(m2.transitions[init2]))
    vals = {t.state_id: v for v, t in branch}

    assert vals[sa2.state_id] == Interval(Fraction(1, 4), Fraction(3, 4))
    assert vals[sb2.state_id] == Interval(Fraction(1, 4), Fraction(3, 4))
