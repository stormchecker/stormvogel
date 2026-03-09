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

    bottom: Number
    top: Number

    def __init__(self, bottom: Number, top: Number):
        self.bottom = bottom
        self.top = top

    def __getitem__(self, idx):
        if idx == 0:
            return self.bottom
        elif idx == 1:
            return self.top
        else:
            raise IndexError(
                "Intervals only have two elements (the bottom and top element)"
            )

    def __lt__(self, other):
        if (self.bottom, self.top) < (other.bottom, other.top):
            return True
        return False

    def __str__(self):
        return f"[{self.bottom},{self.top}]"


Value = Number | parametric.Parametric | Interval


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
        return f"[{value_to_string(n.bottom, use_fractions, round_digits, denom_limit)},{value_to_string(n.top, use_fractions, round_digits, denom_limit)}]"
    else:
        return str(n)
