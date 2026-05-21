from dataclasses import dataclass
from fractions import Fraction

from stormvogel import parametric
from stormvogel.parametric import Parametric

import math

Number = int | float | Fraction


@dataclass
class Interval:
    """Represent an interval value for interval models.

    :param bottom: The bottom (left) element of the interval.
    :param top: The top (right) element of the interval.
    """

    lower: Number
    upper: Number

    def __lt__(self, other):
        if not isinstance(other, Interval):
            raise TypeError("Can only compare Interval to Interval")
        return (self.lower, self.upper) < (other.lower, other.upper)

    def __str__(self):
        return f"[{self.lower},{self.upper}]"


Value = Number | Parametric | Interval


def is_zero(value: Value) -> bool:
    """Returns whether a value is zero."""
    if isinstance(value, (int, float, Fraction)):
        return value == 0
    if isinstance(value, Interval):
        return value.lower == 0 and value.upper == 0
    if parametric.is_parametric(value):
        return parametric.is_zero(value)
    raise TypeError(f"Unsupported type for is_zero: {type(value).__name__}")


def value_to_string(
    n: Value, use_fractions: bool = True, round_digits: int = 4, denom_limit: int = 1000
) -> str:
    """Convert a :class:`Value` to a string.

    :param n: The value to convert.
    :param use_fractions: If ``True``, represent numbers as fractions.
    :param round_digits: Number of decimal places when not using fractions.
    :param denom_limit: Maximum denominator when limiting fractions.
    :returns: String representation of the value.
    """
    if isinstance(n, (int, float)):
        if math.isinf(float(n)):
            return "inf"
        if use_fractions:
            return str(Fraction(n).limit_denominator(denom_limit))
        else:
            return str(round(float(n), round_digits))
    if isinstance(
        n, Fraction
    ):  # In the case of Fraction, a denominator of zero would have caused an error before.
        if use_fractions:
            return str(n.limit_denominator(denom_limit))
        else:
            return str(round(float(n), round_digits))
    if isinstance(n, Interval):
        return (
            f"[{value_to_string(n.lower, use_fractions, round_digits, denom_limit)},"
            f"{value_to_string(n.upper, use_fractions, round_digits, denom_limit)}]"
        )
    if parametric.is_parametric(n):
        return parametric.to_str(n)
    return str(n)
