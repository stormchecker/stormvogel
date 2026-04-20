import math
from fractions import Fraction

import pytest
import sympy as sp
from hypothesis import given, strategies as st

from stormvogel.model import value_to_string, Interval
from stormvogel.model.value import is_zero
from stormvogel import parametric  # noqa: F401 (ensures default backend registers)


def test_int():
    assert value_to_string(46, False, 4, 1000) == "46.0"
    assert value_to_string(46, True, 4, 1000) == "46"


def test_float():
    assert value_to_string(46.0, False, 4, 1000) == "46.0"
    assert value_to_string(46.0, True, 4, 1000) == "46"
    assert value_to_string(0.66666666, False, 4, 1000) == "0.6667"


def test_fraction():
    assert value_to_string(Fraction(30, 40), True, 4, 1000) == "3/4"
    assert value_to_string(Fraction(30, 40), False, 4, 1000) == "0.75"
    assert value_to_string(Fraction(1, 3), False, 4, 1000) == "0.3333"
    assert value_to_string(Fraction(1, 1500), True, 4, 1000) == "1/1000"


def test_parametric_polynomial():
    x, y, z = sp.symbols("x y z")
    pol = 4 * x * y**2 * z**3 + 3 * x
    # sympy's standard pretty-printer is used; the exact rendering depends on
    # sympy's term ordering, which is deterministic but not necessarily what a
    # human would pick by hand.
    assert value_to_string(pol) == sp.sstr(pol)


def test_parametric_rational():
    x, y, z = sp.symbols("x y z")
    rat = (4 * x * y**2 * z**3 + 3 * x) / (2 * z)
    assert value_to_string(rat) == sp.sstr(rat)


def test_interval():
    itvl = Interval(Fraction(1, 3), Fraction(5, 8))
    assert value_to_string(itvl) == "[1/3,5/8]"


def test_infinity():
    assert value_to_string(math.inf) == "inf"
    assert value_to_string(float("inf")) == "inf"


# --- is_zero ---


def test_is_zero_int():
    assert is_zero(0)
    assert not is_zero(1)


def test_is_zero_float():
    assert is_zero(0.0)
    assert not is_zero(0.1)


def test_is_zero_fraction():
    assert is_zero(Fraction(0))
    assert not is_zero(Fraction(1, 3))


def test_is_zero_interval():
    assert is_zero(Interval(0, 0))
    assert not is_zero(Interval(0, 1))
    assert not is_zero(Interval(1, 2))


def test_is_zero_parametric():
    x = sp.Symbol("x")
    # Symbolic cancellation: x - x == 0.
    assert is_zero(x - x)
    assert is_zero(sp.Integer(0))
    assert not is_zero(x)
    assert not is_zero(x + 1)


def test_is_zero_unsupported_type():
    with pytest.raises(TypeError):
        is_zero("string")  # type: ignore


# --- Interval ordering ---


def test_interval_lt():
    a = Interval(0, 1)
    b = Interval(1, 2)
    assert a < b
    assert not b < a


def test_interval_lt_type_error():
    with pytest.raises(TypeError):
        _ = Interval(0, 1) < 5  # type: ignore


def test_interval_str():
    assert str(Interval(1, 3)) == "[1,3]"


# --- Hypothesis-based property tests ---


@given(st.integers())
def test_value_to_string_int_no_crash(n):
    result = value_to_string(n)
    assert isinstance(result, str)


@given(st.floats(allow_nan=False))
def test_value_to_string_float_no_crash(f):
    result = value_to_string(f)
    assert isinstance(result, str)
