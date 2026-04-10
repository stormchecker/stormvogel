import math
from fractions import Fraction

import pytest
from hypothesis import given, strategies as st

from stormvogel.model import value_to_string, Interval
from stormvogel.model.value import is_zero
from stormvogel import parametric


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
    pol = parametric.Polynomial(["x", "y", "z"])
    pol.add_term((1, 2, 3), 4)
    pol.add_term((1, 0, 0), 3)
    assert value_to_string(pol) == "4.0*xy^2z^3 + 3.0*x"


def test_parametric_rational():
    pol1 = parametric.Polynomial(["x", "y", "z"])
    pol1.add_term((1, 2, 3), 4)
    pol1.add_term((1, 0, 0), 3)
    pol2 = parametric.Polynomial(["z"])
    pol2.add_term((1,), 2)
    rat = parametric.RationalFunction(pol1, pol2)
    assert value_to_string(rat) == "(4.0*xy^2z^3 + 3.0*x)/(2.0*z)"


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
    pol = parametric.Polynomial(["x"])
    assert is_zero(pol)  # empty polynomial is zero
    pol.add_term((1,), 1)
    assert not is_zero(pol)


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
