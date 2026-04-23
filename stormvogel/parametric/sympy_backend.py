"""Sympy implementation of the parametric backend.

Registers itself on import. Works whether or not pycarl / stormpy is
available: the pycarl bridge methods import ``stormpy.pycarl`` lazily and
only when called.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import sympy as sp

from stormvogel.parametric import (
    degree,
    evaluate,
    free_symbol_names,
    is_zero,
    numerator_denominator,
    to_str,
)
from stormvogel.parametric._backend import Number, register


# ---------------------------------------------------------------------------
# Dispatch overloads for sp.Expr.
#
# Note: sp.Integer, sp.Rational, and sp.Float are all sp.Expr subclasses, so
# these overloads also cover literal sympy constants. The `is_parametric`
# helper in the package __init__ excludes plain Python numbers from the
# "is parametric" set; sympy-native constants go through the overloads
# below and compute sensible values (e.g. `free_symbol_names(sp.Integer(1))
# == set()`).
# ---------------------------------------------------------------------------


@is_zero.register(sp.Expr)
def _is_zero_sympy(value: sp.Expr) -> bool:
    # sympy's .is_zero may return None for non-simplified expressions; fall
    # back to structural simplification in that case.
    result = value.is_zero
    if result is not None:
        return bool(result)
    return bool(sp.simplify(value) == 0)


@free_symbol_names.register(sp.Expr)
def _free_symbol_names_sympy(value: sp.Expr) -> set[str]:
    return {s.name for s in value.free_symbols}


@degree.register(sp.Expr)
def _degree_sympy(value: sp.Expr) -> int:
    if not value.free_symbols:
        return 0
    # Put the rational function on a common denominator and take the numerator's
    # total degree — consistent with the old behaviour for polynomials, and
    # predictable for rational functions.
    num, _ = sp.fraction(sp.together(value))
    symbols = sorted(num.free_symbols, key=lambda s: s.name)
    if not symbols:
        return 0
    return int(sp.Poly(num, *symbols).total_degree())


@evaluate.register(sp.Expr)
def _evaluate_sympy(value: sp.Expr, values: dict[str, Number]):
    subs: dict[sp.Symbol, Number] = {}
    # Map string keys to whatever Symbol objects the expression already uses,
    # so that assumptions on declared symbols are preserved.
    by_name = {s.name: s for s in value.free_symbols}
    for k, v in values.items():
        if isinstance(k, sp.Symbol):
            subs[k] = v
        else:
            sym = by_name.get(str(k))
            if sym is None:
                # Key not referenced by this expression — harmless, just skip.
                continue
            subs[sym] = v
    substituted = value.subs(subs)
    # Coerce to a native Python number when possible.
    if substituted.is_Integer:
        return int(substituted)
    if substituted.is_Rational:
        return Fraction(int(substituted.p), int(substituted.q))
    if substituted.is_Float:
        return float(substituted)
    if not substituted.free_symbols:
        return float(substituted)
    return substituted


@numerator_denominator.register(sp.Expr)
def _numden_sympy(value: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    return sp.fraction(sp.together(value))


@to_str.register(sp.Expr)
def _to_str_sympy(value: sp.Expr) -> str:
    return sp.sstr(value)


# ---------------------------------------------------------------------------
# The backend object.
# ---------------------------------------------------------------------------


class SympyBackend:
    """Sympy-backed :class:`ParametricBackend`."""

    name = "sympy"
    expr_types: tuple[type, ...] = (sp.Expr,)

    def symbol(self, name: str, **kwargs: Any) -> sp.Symbol:
        return sp.Symbol(name, **kwargs)

    def constant(self, n: Number) -> sp.Expr:
        if isinstance(n, Fraction):
            return sp.Rational(n.numerator, n.denominator)
        if isinstance(n, int):
            return sp.Integer(n)
        return sp.Float(n)

    # -- pycarl bridge --------------------------------------------------

    def to_pycarl(self, value: Any, var_map: dict[str, Any]) -> Any:
        """Convert a :class:`sp.Expr` into a pycarl factorized rational
        function, reusing the pycarl :class:`Variable`\\s in ``var_map``.

        ``var_map`` must be an *ordered* mapping (insertion order matters):
        we take its values as the polynomial-variable ordering, so that
        exponent tuples produced by :func:`sympy.Poly.terms` line up with
        the pycarl variable map.
        """
        import stormpy  # noqa: F401  (re-export check)
        from stormpy import pycarl

        expr = sp.sympify(value)
        num_expr, den_expr = sp.fraction(sp.together(expr))

        num_poly = self._poly_to_pycarl(num_expr, var_map)
        factorized_num = pycarl.cln.FactorizedPolynomial(
            num_poly, pycarl.cln.factorization_cache
        )
        if den_expr == 1:
            return pycarl.cln.FactorizedRationalFunction(factorized_num)

        den_poly = self._poly_to_pycarl(den_expr, var_map)
        factorized_den = pycarl.cln.FactorizedPolynomial(
            den_poly, pycarl.cln.factorization_cache
        )
        return pycarl.cln.FactorizedRationalFunction(factorized_num, factorized_den)

    @staticmethod
    def _poly_to_pycarl(expr: sp.Expr, var_map: dict[str, Any]) -> Any:
        """Build a pycarl polynomial from a sympy polynomial expression,
        using the pycarl :class:`Variable`\\s in ``var_map`` in insertion
        order."""
        from stormpy import pycarl

        # Polynomial variables in insertion order, matching var_map.
        sym_order = [sp.Symbol(name) for name in var_map]

        # A constant polynomial — no need to call sp.Poly.
        expr_symbols = {s.name for s in expr.free_symbols}
        if not expr_symbols:
            rational = pycarl.cln.Rational(sp.Rational(expr))
            return pycarl.cln.Polynomial(rational)

        # Restrict the variable list to the ones sp.Poly will actually see.
        # sp.Poly is happy to accept extra symbols, but narrowing keeps
        # exponent tuples small.
        active = [s for s in sym_order if s.name in expr_symbols]
        poly = sp.Poly(expr, *active)
        pycarl_vars = [var_map[s.name] for s in active]

        terms = []
        for exponents, coeff in poly.terms():
            # coeff is a sp.Rational / sp.Integer / sp.Float in general.
            if isinstance(coeff, sp.Float):
                rat = sp.Rational(coeff).limit_denominator(10**12)
            else:
                rat = sp.Rational(coeff)
            pycarl_coeff = pycarl.cln.Rational(
                pycarl.cln.Integer(int(rat.p)), pycarl.cln.Integer(int(rat.q))
            )
            term = pycarl.cln.Term(pycarl_coeff)
            for var, exp in zip(pycarl_vars, exponents):
                for _ in range(int(exp)):
                    term = term * var
            terms.append(term)
        return pycarl.cln.Polynomial(terms)

    def from_pycarl(self, pycarl_value: Any) -> sp.Expr:
        """Convert a pycarl ``FactorizedRationalFunction`` (or bare
        ``RationalFunction`` / ``Polynomial``) into a :class:`sp.Expr`.

        When the denominator is constant 1 and the numerator is constant,
        the plain Python number is returned instead so downstream code can
        treat it as a :class:`Number` first-class.
        """
        # Everything pycarl prints is valid sympy once we swap '^' for '**'.
        # We build a controlled local_dict so that a parameter called `E`
        # or `I` doesn't collide with sympy built-in constants.
        numerator = pycarl_value.numerator
        denominator = pycarl_value.denominator

        def _poly_str(pycarl_poly) -> str:
            return str(pycarl_poly).replace("^", "**")

        def _poly_symbols(pycarl_poly) -> list[str]:
            # pycarl exposes `gather_variables()` returning a set-like of
            # Variable objects whose str() starts with the variable name.
            names: list[str] = []
            for v in pycarl_poly.gather_variables():
                s = str(v)
                # pycarl Variable str form is typically '<Variable x ...>';
                # extract the name robustly.
                if s.startswith("<Variable "):
                    s = s[len("<Variable ") :]
                    s = s.split(" ", 1)[0].rstrip(">")
                names.append(s)
            return names

        if denominator.is_constant() and float(denominator.constant_part()) == 1:
            if numerator.is_constant():
                # Plain number — hand back a Python float so the caller can
                # treat it as non-parametric.
                return float(numerator.constant_part())
            names = _poly_symbols(numerator)
            locals_ = {n: sp.Symbol(n) for n in names}
            return sp.sympify(_poly_str(numerator), locals=locals_)

        names = set(_poly_symbols(numerator)) | set(_poly_symbols(denominator))
        locals_ = {n: sp.Symbol(n) for n in names}
        num = sp.sympify(_poly_str(numerator), locals=locals_)
        den = sp.sympify(_poly_str(denominator), locals=locals_)
        return num / den


register(SympyBackend())
