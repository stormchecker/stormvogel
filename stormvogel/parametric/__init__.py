"""Parametric values for parametric Markov models.

A parametric value is a polynomial or rational function over a set of named
parameters. Stormvogel represents such values as *backend-native* objects
(by default, :class:`sympy.Expr`), and exposes a small set of dispatching
helpers in this package so that the rest of the codebase never has to know
which backend produced a given value::

    import sympy as sp
    from stormvogel import model
    from stormvogel import parametric

    pmdp = model.new_mdp()
    x = pmdp.declare_parameter("x")
    # x is now a sp.Symbol that pmdp owns — reuse it everywhere.

    parametric.free_symbol_names(1 - x)         # {"x"}
    parametric.evaluate(1 - x, {"x": 0.25})     # 0.75
    parametric.is_zero(x - x)                   # True

The helpers below (:func:`is_zero`, :func:`free_symbol_names`,
:func:`degree`, :func:`evaluate`, :func:`numerator_denominator`,
:func:`to_str`) are :func:`functools.singledispatch` generics. The
:mod:`stormvogel.parametric.sympy_backend` module registers overloads for
:class:`sympy.Expr` and is imported by default. A future
``pycarl_backend`` would register overloads for its own types with the
same public API — no changes outside this package.
"""

from __future__ import annotations

from fractions import Fraction
from functools import singledispatch
from typing import TypeAlias, Union

import sympy as sp

from stormvogel.parametric._backend import (
    ParametricBackend,
    backend_for,
    get_default,
    register,
    registered_types,
    set_default,
)

Number: TypeAlias = Union[int, float, Fraction]
# The authoritative leaf type for parametric values. Widened at runtime as
# additional backends are registered (see `registered_types()`), but here the
# declared alias matches the default (sympy) backend so static checkers have
# something concrete to chew on.
Parametric: TypeAlias = sp.Expr


# ---------------------------------------------------------------------------
# Public helpers — each dispatches on the value's concrete type.
# Backends call `.register(type)` on these in their own module.
# ---------------------------------------------------------------------------


@singledispatch
def is_zero(value) -> bool:
    """Return ``True`` iff ``value`` is structurally / symbolically zero."""
    raise TypeError(
        f"is_zero: no parametric backend registered for type {type(value).__name__}"
    )


@singledispatch
def free_symbol_names(value) -> set[str]:
    """Return the set of parameter names occurring in ``value``."""
    raise TypeError(
        f"free_symbol_names: no parametric backend registered for type {type(value).__name__}"
    )


@singledispatch
def degree(value) -> int:
    """Return the total degree of ``value`` (0 for constants)."""
    raise TypeError(
        f"degree: no parametric backend registered for type {type(value).__name__}"
    )


@singledispatch
def evaluate(value, values: dict[str, Number]) -> "Number | Parametric":
    """Substitute the given parameter values.

    String keys are resolved by parameter *name*. The result is returned as
    a native Python :class:`Number` when all parameters are substituted; a
    symbolic expression otherwise.
    """
    raise TypeError(
        f"evaluate: no parametric backend registered for type {type(value).__name__}"
    )


@singledispatch
def numerator_denominator(value) -> "tuple[Parametric, Parametric]":
    """Return ``(numerator, denominator)`` of ``value`` as a rational function."""
    raise TypeError(
        f"numerator_denominator: no parametric backend registered for type {type(value).__name__}"
    )


@singledispatch
def to_str(value) -> str:
    """Return a stable human-readable string form of ``value``."""
    raise TypeError(
        f"to_str: no parametric backend registered for type {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Construction & type-check helpers — go through the *default* backend.
# Import the backend module lazily to keep the package importable if a user
# swaps defaults before first use.
# ---------------------------------------------------------------------------


def is_parametric(value) -> bool:
    """Return ``True`` iff ``value``'s concrete type is owned by any
    registered parametric backend.

    This is what downstream stormvogel code uses (in ``Model.is_parametric``,
    ``Distribution.is_stochastic``, the simulator, …) — never an
    :func:`isinstance` on a concrete backend class.

    Pure Python numbers (:class:`int`, :class:`float`, :class:`Fraction`,
    :class:`bool`) are **not** parametric, even though e.g. ``sp.Integer(1)``
    technically is a :class:`sp.Expr`; we keep the split by excluding types
    that are also valid as a :class:`Number`.
    """
    if isinstance(value, (int, float, Fraction)):
        return False
    return isinstance(value, registered_types())


def symbol(name: str, **kwargs) -> Parametric:
    """Construct a parameter symbol using the currently-selected default
    backend.

    Extra keyword arguments are forwarded to the backend's factory (for the
    sympy backend, these are :func:`sympy.Symbol` assumptions such as
    ``positive=True``).
    """
    return get_default().symbol(name, **kwargs)


def constant(n: Number) -> Parametric:
    """Lift a Python :class:`Number` into a parametric value via the default
    backend. Rarely needed — stormvogel internals accept :class:`Number` as a
    first-class ``Value`` — but useful when a parametric context demands a
    backend-native zero / one."""
    return get_default().constant(n)


__all__ = [
    # Types / aliases
    "Number",
    "Parametric",
    "ParametricBackend",
    # Generic helpers
    "is_zero",
    "free_symbol_names",
    "degree",
    "evaluate",
    "numerator_denominator",
    "to_str",
    # Type checks and construction
    "is_parametric",
    "symbol",
    "constant",
    # Backend management
    "backend_for",
    "get_default",
    "register",
    "registered_types",
    "set_default",
]


# Import the default backend last so it can register itself against the
# generics defined above. Keep this import at module bottom to avoid
# circular-import pitfalls.
from stormvogel.parametric import sympy_backend as _sympy_backend  # noqa: E402, F401
