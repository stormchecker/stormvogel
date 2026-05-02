"""Thin stormpy wrappers for parametric model analysis.

Provides a class that fixes a stormvogel model and a property formula once,
then reuses the stormpy model and checkers across repeated queries.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING

import stormvogel.model
import stormvogel.stormpy_utils.mapping as mapping
from stormvogel.parametric._backend import Number
from stormvogel.parametric.region import AnnotatedRegion, RectangularRegion

try:
    import stormpy
    import stormpy.pars
except ImportError:
    stormpy = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass


def rectangular_region_to_stormpy(
    name_to_var: dict[str, object],
    region: RectangularRegion,
) -> "stormpy.pars.ParameterRegion":
    """Convert a :class:`~stormvogel.parametric.region.RectangularRegion` to a stormpy ``ParameterRegion``.

    :param name_to_var: Mapping from parameter name to the corresponding
        ``pycarl.Variable`` object (as returned by
        :meth:`AnalyseParametric.name_to_var`).
    :param region: The rectangular region to convert.
    :returns: A stormpy ``ParameterRegion`` with the same bounds.
    :raises KeyError: If a parameter name in *region* is absent from *name_to_var*.
    """
    assert stormpy is not None
    bounds: dict = {}
    for name, (lo, hi) in region.bounds.items():
        var = name_to_var[name]
        bounds[var] = (
            stormpy.pycarl.cln.Rational(float(lo)),
            stormpy.pycarl.cln.Rational(float(hi)),
        )
    return stormpy.pars.ParameterRegion(bounds)


class AnalyseParametric:
    """Reusable analysis context for a fixed parametric model and property.

    The stormvogel→stormpy conversion, formula parsing, instantiation checker,
    and region checker are all created once in the constructor and reused across
    every query call.

    :param model: A parametric stormvogel model.
    :param prop: A Storm property string, e.g. ``'P=? [F "target"]'``.
    :param env: A stormpy ``Environment`` to use; a fresh default environment
        is created when ``None``.
    :raises ImportError: If stormpy is not installed.
    :raises ValueError: If *model* is not parametric.
    """

    def __init__(
        self,
        model: stormvogel.model.Model,
        prop: str,
        env: "stormpy.Environment | None" = None,
    ) -> None:
        if stormpy is None:
            raise ImportError("stormpy is required for AnalyseParametric.")
        if not model.is_parametric():
            raise ValueError("AnalyseParametric requires a parametric model.")

        self.model = model
        self._sp_model = mapping.stormvogel_to_stormpy(model)
        self.env: "stormpy.Environment" = (
            env if env is not None else stormpy.Environment()
        )

        props = stormpy.parse_properties(prop)
        self._formula = props[0].raw_formula

        # Build name → pycarl Variable mapping from the converted stormpy model.
        # collect_all_parameters() returns the Variable objects that were placed
        # in the pycarl pool during stormvogel_to_stormpy; matching by .name
        # gives a stable, name-keyed dict.
        self._name_to_var: dict[str, object] = {
            var.name: var for var in self._sp_model.collect_all_parameters()
        }

        _checker_cls = {
            stormvogel.model.ModelType.DTMC: stormpy.pars.PDtmcInstantiationChecker,
            stormvogel.model.ModelType.MDP: stormpy.pars.PMdpInstantiationChecker,
            stormvogel.model.ModelType.CTMC: stormpy.pars.PCtmcInstantiationChecker,
        }.get(model.model_type)
        if _checker_cls is None:
            raise ValueError(
                f"AnalyseParametric does not support model type {model.model_type}."
            )
        if model.model_type != stormvogel.model.ModelType.DTMC:
            warnings.warn(
                f"AnalyseParametric has only been tested on DTMCs; "
                f"{model.model_type} support is experimental and assumptions "
                f"may be violated."
            )
        self._instantiation_checker = _checker_cls(self._sp_model)
        self._instantiation_checker.specify_formula(
            stormpy.ParametricCheckTask(self._formula, only_initial_states=True)
        )

        self._region_checker = stormpy.pars.create_region_checker(
            self.env, self._sp_model, self._formula, allow_model_simplification=True
        )

    @property
    def name_to_var(self) -> dict[str, object]:
        """Mapping from parameter name to pycarl ``Variable``.

        Pass this to :func:`rectangular_region_to_stormpy` when you need to
        convert regions outside of this class.
        """
        return self._name_to_var

    def evaluate_at_point(self, valuation: Mapping[str, Number]) -> float:
        """Evaluate the property at a single parameter instantiation.

        :param valuation: Mapping from parameter name to a concrete value.
        :returns: The property value at the initial state as a ``float``.
        :raises KeyError: If a name in *valuation* is not a parameter of the model.

        TODO: once the instantiation checker also exposes exact rationals,
        use exact arithmetic here.
        See https://github.com/stormchecker/stormpy/discussions/389
        """
        assert stormpy is not None
        instantiation: dict = {
            self._name_to_var[name]: stormpy.pycarl.cln.Rational(float(val))
            for name, val in valuation.items()
        }
        result = self._instantiation_checker.check(self.env, instantiation)
        return float(result.at(0))

    def get_region_bound(
        self,
        region: RectangularRegion,
        maximize: bool = True,
        assume_graph_preserving: bool = False,
    ) -> Number:
        """Return the optimistic bound on the property value over *region*.

        Calls ``RegionChecker.get_bound``; the result is an upper bound when
        *maximize* is ``True`` and a lower bound when ``False``.

        The bound is returned as an exact :class:`~fractions.Fraction` derived
        from the rational number produced by stormpy's region checker.

        :param region: A rectangular parameter region.  Must be graph-preserving
            for *model*.
        :param maximize: ``True`` to get an upper bound (Pmax), ``False`` for
            a lower bound (Pmin).
        :param assume_graph_preserving: Skip the graph-preserving check.  Set
            this only when the caller has already verified the region.
        :returns: The bound as a :class:`~fractions.Fraction`.
        :raises ValueError: If *region* is not graph-preserving for the model.
        """
        if not assume_graph_preserving and not region.is_graph_preserving(self.model):
            raise ValueError(
                "get_region_bound requires a graph-preserving region: at least one "
                "transition probability reaches 0 somewhere within the region."
            )
        sp_region = rectangular_region_to_stormpy(self._name_to_var, region)
        bound = self._region_checker.get_bound(self.env, sp_region, maximize)
        return Fraction(str(bound))

    def annotate_region(
        self,
        region: RectangularRegion,
        sample: bool = True,
        assume_graph_preserving: bool = False,
    ) -> AnnotatedRegion:
        """Annotate *region* with interval bounds on its minimum and maximum property value.

        Always calls :meth:`get_region_bound` twice to obtain a verified lower
        bound on the minimum (Pmin) and a verified upper bound on the maximum
        (Pmax) via the region checker.

        When *sample* is ``True``, the property is also evaluated at every
        vertex of *region* and at its center point.  The minimum and maximum of
        those samples tighten the inner bounds:

        - ``min_value = (Pmin, min(samples))``
        - ``max_value = (max(samples), Pmax)``

        When *sample* is ``False``, only the verified bounds are used and the
        inner bounds default to ``(Pmin, Pmax)``:

        - ``min_value = (Pmin, Pmax)``
        - ``max_value = (Pmin, Pmax)``

        :param region: The rectangular parameter region to annotate.
        :param sample: Whether to evaluate at vertices and the center.
        :param assume_graph_preserving: Skip the graph-preserving check inside
            :meth:`get_region_bound`.  Set this only when the caller has already
            verified that *region* is graph-preserving for the model.
        :returns: An :class:`~stormvogel.parametric.region.AnnotatedRegion`.
        """
        pmin = self.get_region_bound(
            region, maximize=False, assume_graph_preserving=assume_graph_preserving
        )
        pmax = self.get_region_bound(
            region, maximize=True, assume_graph_preserving=assume_graph_preserving
        )

        if sample:
            center = {name: (lo + hi) / 2 for name, (lo, hi) in region.bounds.items()}
            sample_points = region.vertices() + [center]
            values = [self.evaluate_at_point(pt) for pt in sample_points]
            lo_min, hi_max = pmin, pmax
            # Clamp against the verified bounds: the instantiation checker uses
            # floating-point arithmetic and can produce values that numerically
            # violate the exact rational bounds from the region checker.
            # TODO: once the instantiation checker also exposes exact rationals, use them here.
            hi_min = max(pmin, min(values))
            lo_max = min(pmax, max(values))
        else:
            lo_min, hi_min = pmin, pmax
            lo_max, hi_max = pmin, pmax

        return AnnotatedRegion(
            region=region,
            min_value=(lo_min, hi_min),
            max_value=(lo_max, hi_max),
        )
