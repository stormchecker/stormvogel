"""A model variable."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntDomain:
    """A bounded integer domain [lo, hi].

    :param lo: Lower bound (inclusive).
    :param hi: Upper bound (inclusive).
    :param allow_none: Whether ``None`` is a valid value.
    """

    lo: int
    hi: int
    allow_none: bool = False

    def contains(self, value: Any) -> bool:
        if value is None:
            return self.allow_none
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and self.lo <= value <= self.hi
        )

    def __repr__(self):
        suffix = ", allow_none=True" if self.allow_none else ""
        return f"IntDomain({self.lo}, {self.hi}{suffix})"


@dataclass
class BoolDomain:
    """A boolean domain {False, True}.

    :param allow_none: Whether ``None`` is a valid value.
    """

    allow_none: bool = False

    def contains(self, value: Any) -> bool:
        if value is None:
            return self.allow_none
        return isinstance(value, bool)

    def __repr__(self):
        suffix = ", allow_none=True" if self.allow_none else ""
        return f"BoolDomain({suffix.lstrip(', ')})"


@dataclass
class CategoricalDomain:
    """A finite categorical domain with an explicit ordered set of values.

    :param values: All valid values, as a tuple to preserve order.
    :param allow_none: Whether ``None`` is a valid value.
    """

    values: tuple[Any, ...]
    allow_none: bool = False

    def contains(self, value: Any) -> bool:
        if value is None:
            return self.allow_none
        return value in self.values

    def __repr__(self):
        suffix = ", allow_none=True" if self.allow_none else ""
        return f"CategoricalDomain({self.values!r}{suffix})"


VariableDomain = IntDomain | BoolDomain | CategoricalDomain


@dataclass(frozen=True)
class Variable:
    label: str
    domain: VariableDomain | None = field(default=None, compare=False, hash=False)

    def __lt__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return str(self.label) < str(other.label)

    def check_valuation(self, value: Any) -> None:
        """Raise ``ValueError`` if ``value`` is outside this variable's declared domain."""
        if self.domain is not None and not self.domain.contains(value):
            raise ValueError(
                f"Value {value!r} is not in domain {self.domain!r} "
                f"for variable {self!r}."
            )

    def __repr__(self):
        if self.domain is not None:
            return f"Variable({self.label!r}, {self.domain!r})"
        return f"Variable({self.label!r})"

    __str__ = __repr__


@dataclass(frozen=True)
class Predicate:
    """A named observable predicate with a required domain and an optional defining expression.

    :param label: Human-readable name for the predicate.
    :param domain: The domain of values this predicate can take.
    :param expr: Optional callable ``f(valuations) -> value`` that computes the
        predicate value from a state's ``{Variable: value}`` dict.  ``None``
        when the predicate was imported from an external tool and the expression
        is not available.
    """

    label: str
    domain: VariableDomain = field(compare=False, hash=False)
    expr: Callable[[dict[Variable, Any]], Any] | None = field(
        default=None, compare=False, hash=False
    )

    def check_valuation(self, value: Any) -> None:
        """Raise ``ValueError`` if ``value`` is outside this predicate's domain."""
        if not self.domain.contains(value):
            raise ValueError(
                f"Value {value!r} is not in domain {self.domain!r} "
                f"for predicate {self!r}."
            )

    def __repr__(self):
        return f"Predicate({self.label!r}, {self.domain!r})"

    __str__ = __repr__


VariableKey = Variable | Predicate
