"""Rectangular parameter regions and the induced interval MDP transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import sympy as sp

from stormvogel.parametric import degree, is_parametric
from stormvogel.parametric._backend import Number

if TYPE_CHECKING:
    import stormvogel.model as model


@dataclass
class RectangularRegion:
    """A rectangular parameter region: a Cartesian product of closed intervals.

    :param bounds: Mapping from parameter name to ``(lower, upper)`` bound.
        Bounds can be any :class:`~stormvogel.parametric.Number`.
    """

    bounds: dict[str, tuple[Number, Number]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (lo, hi) in self.bounds.items():
            if lo > hi:
                raise ValueError(
                    f"Parameter '{name}': lower bound {lo} > upper bound {hi}."
                )

    def contains(self, valuation: dict[str, Number]) -> bool:
        """Return True iff *valuation* lies inside every interval."""
        for name, (lo, hi) in self.bounds.items():
            v = valuation.get(name)
            if v is None or v < lo or v > hi:
                return False
        return True

    def vertices(self) -> list[dict[str, Number]]:
        """Return all 2^n corner points of the region."""
        corners: list[dict[str, Number]] = [{}]
        for name, (lo, hi) in self.bounds.items():
            corners = [{**c, name: b} for c in corners for b in (lo, hi)]
        return corners

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}: [{lo}, {hi}]" for n, (lo, hi) in self.bounds.items())
        return f"RectangularRegion({{{parts}}})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_number(x: sp.Expr) -> Number:
    x = sp.nsimplify(x)
    if x.is_Integer:
        return int(x)
    if x.is_Rational:
        return Fraction(int(x.p), int(x.q))
    return float(x)


def _affine_image(expr: sp.Expr, region: RectangularRegion) -> tuple[Number, Number]:
    """Return ``(min, max)`` of an affine sympy expression over *region*.

    For each parameter *x* with coefficient *c*, the minimum uses the lower
    bound when *c ≥ 0* and the upper bound when *c < 0* (and vice-versa for
    the maximum).

    :raises ValueError: If *expr* has total degree > 1, or a free symbol is
        absent from *region*.
    """
    d = degree(expr)
    if d > 1:
        raise ValueError(
            f"Only affine transition probabilities are supported; "
            f"got expression of degree {d}: {expr}"
        )

    coeffs = expr.as_coefficients_dict()
    lo: sp.Expr = sp.Integer(0)
    hi: sp.Expr = sp.Integer(0)

    for term, coeff in coeffs.items():
        if term.is_number:
            lo = lo + coeff
            hi = hi + coeff
        else:
            name = str(term)
            if name not in region.bounds:
                raise ValueError(
                    f"Parameter '{name}' appears in transition but is not in region."
                )
            lb, ub = region.bounds[name]
            l_s = sp.nsimplify(lb)
            u_s = sp.nsimplify(ub)
            if coeff >= 0:
                lo = lo + coeff * l_s
                hi = hi + coeff * u_s
            else:
                lo = lo + coeff * u_s
                hi = hi + coeff * l_s

    return _to_number(lo), _to_number(hi)


# ---------------------------------------------------------------------------
# Public transformation
# ---------------------------------------------------------------------------


def to_interval_mdp(pmdp: "model.Model", region: RectangularRegion) -> "model.Model":
    """Return the interval MDP induced by *pmdp* and a rectangular *region*.

    Each parametric transition probability (which must be affine in the
    parameters) is replaced by the interval ``[min, max]`` of its image over
    *region*.  Non-parametric probabilities become point intervals ``[p, p]``.

    Uses :meth:`~stormvogel.model.Model.copy` to preserve state UUIDs, so the
    returned model can be cross-referenced with the original via
    ``get_state_by_id``.

    :param pmdp: A parametric stormvogel model.
    :param region: A rectangular region covering all parameters in *pmdp*.
    :returns: A new stormvogel model with
        :class:`~stormvogel.model.value.Interval` transition probabilities.
    :raises ValueError: If any transition probability is not affine, or if a
        parameter is absent from *region*.
    """
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.value import Interval

    imdp = pmdp.copy()
    imdp._parameters.clear()
    imdp._is_parametric = None  # type: ignore[attr-defined]

    for state in list(imdp.transitions):
        for action, branch in list(imdp.transitions[state]):
            new_distr: Distribution = Distribution()
            for val, target in branch:
                if is_parametric(val):
                    lo, hi = _affine_image(val, region)  # type: ignore[arg-type]
                else:
                    lo = hi = val  # type: ignore[assignment]
                new_distr[target] = Interval(lower=lo, upper=hi)
            imdp.transitions[state][action] = new_distr

    return imdp
