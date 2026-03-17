from dataclasses import dataclass
from fractions import Fraction

from stormvogel import parametric

import math

Number = int | float | Fraction


@dataclass
class Interval:
    """represents an interval value for interval models

    Args:
        bottom: the bottom (left) element of the interval
        top: the top (right) element of the interval
    """

    lower: Number
    upper: Number

    def __init__(self, bottom: Number, top: Number):
        self.lower = bottom
        self.upper = top

    def __getitem__(self, idx):
        if idx == 0:
            return self.lower
        elif idx == 1:
            return self.upper
        else:
            raise IndexError(
                "Intervals only have two elements (the bottom and top element)"
            )

    def __lt__(self, other):
        if not isinstance(other, Interval):
            raise TypeError("Can only compare Interval to Interval")
        if (self.lower, self.upper) < (other.lower, other.upper):
            return True
        return False

    def __str__(self):
        return f"[{self.lower},{self.upper}]"


Value = Number | parametric.Parametric | Interval


def is_zero(value: Value) -> bool:
    """Returns whether a value is zero."""
    if isinstance(value, (int, float, Fraction)):
        return value == 0
    elif isinstance(value, Interval):
        return value.lower == 0 and value.upper == 0
    elif isinstance(value, parametric.Parametric):
        return value.is_zero()
    else:
        raise TypeError("Unsupported type for is_zero")


def value_to_string(
    n: Value, use_fractions: bool = True, round_digits: int = 4, denom_limit: int = 1000
) -> str:
    """Convert a Value to a string."""
    if isinstance(n, (int, float)):
        if math.isinf(float(n)):
            return "inf"
        if use_fractions:
            return str(Fraction(n).limit_denominator(denom_limit))
        else:
            return str(round(float(n), round_digits))
    elif isinstance(
        n, Fraction
    ):  # In the case of Fraction, a denominator of zero would have caused an error before.
        if use_fractions:
            return str(n.limit_denominator(denom_limit))
        else:
            return str(round(float(n), round_digits))
    elif isinstance(n, parametric.Parametric):
        return str(n)
    elif isinstance(n, Interval):
        return f"[{value_to_string(n.lower, use_fractions, round_digits, denom_limit)},{value_to_string(n.upper, use_fractions, round_digits, denom_limit)}]"
    else:
        return str(n)
