"""Rectangular parameter regions and the induced interval MDP transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

from deprecated import deprecated  # type: ignore[import]

import sympy as sp

from stormvogel.parametric import degree, is_parametric
from stormvogel.parametric._backend import Number

if TYPE_CHECKING:
    import stormvogel.model as model
    from matplotlib.axes import Axes


@dataclass
class AnnotatedRegion:
    """A rectangular region with interval bounds on its minimum and maximum property value.

    :param region: The underlying rectangular parameter region.
    :param min_value: ``(lo, hi)`` interval for the minimum property value over *region*.
    :param max_value: ``(lo, hi)`` interval for the maximum property value over *region*.
    """

    region: "RectangularRegion"
    min_value: tuple[Number, Number]
    max_value: tuple[Number, Number]

    def __post_init__(self) -> None:
        lo_min, hi_min = self.min_value
        lo_max, hi_max = self.max_value
        if lo_min > hi_min:
            raise ValueError(f"min_value interval is inverted: [{lo_min}, {hi_min}].")
        if lo_max > hi_max:
            raise ValueError(f"max_value interval is inverted: [{lo_max}, {hi_max}].")
        if lo_min > hi_max:
            raise ValueError(
                f"min_value lower bound {lo_min} > max_value upper bound {hi_max}."
            )

    def classify(self, threshold: Number) -> str:
        """Return ``"safe"``, ``"unsafe"``, ``"neither"``, or ``"unknown"`` relative to *threshold*.

        - ``"safe"``: ``min_value[0] >= threshold`` — every instantiation is ≥ threshold.
        - ``"unsafe"``: ``max_value[1] < threshold`` — every instantiation is < threshold.
        - ``"neither"``: ``max_value[0] >= threshold`` and ``min_value[1] < threshold`` —
          the region certifiably straddles the threshold (some instantiations above, some below).
        - ``"unknown"``: otherwise — not enough information to classify.
        """
        if self.min_value[0] >= threshold:
            return "safe"
        if self.max_value[1] < threshold:
            return "unsafe"
        if self.max_value[0] >= threshold and self.min_value[1] < threshold:
            return "neither"
        return "unknown"


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

    def split(
        self, param: str | None = None
    ) -> tuple[RectangularRegion, RectangularRegion]:
        """Split the region into two equal halves at the midpoint of *param*.

        If *param* is ``None`` the dimension with the largest range is chosen.

        :param param: Parameter name to split along.
        :returns: Pair ``(lower_half, upper_half)`` whose union is this region.
        :raises ValueError: If *param* is not present in this region.
        """
        if not self.bounds:
            raise ValueError("Cannot split an empty region.")
        if param is None:
            param = max(
                self.bounds, key=lambda n: self.bounds[n][1] - self.bounds[n][0]
            )
        elif param not in self.bounds:
            raise ValueError(f"Parameter '{param}' is not in this region.")
        lo, hi = self.bounds[param]
        mid = (lo + hi) / 2
        lower = RectangularRegion({**self.bounds, param: (lo, mid)})
        upper = RectangularRegion({**self.bounds, param: (mid, hi)})
        return lower, upper

    def is_well_defined(self, model: "model.Model") -> bool:
        """Return True iff the region is well-defined for *model*.

        A region is well-defined if every transition probability lies in
        [0, 1], every reward is non-negative for all instantiations within
        the region, and the probabilities of every choice sum to exactly 1
        as a formal identity.  Requires an affine parametric model.

        For non-parametric models, the check basically asks whether the model is stochastic.
        """
        for val in _unique_transition_values(model):
            if is_parametric(val):
                lo, hi = _affine_image(val, self)
            else:
                lo = hi = val  # type: ignore[assignment]
            if lo < 0 or hi > 1:
                return False
        for val in _unique_reward_values(model):
            if is_parametric(val):
                lo, _ = _affine_image(val, self)
            else:
                lo = val  # type: ignore[assignment]
            if lo < 0:
                return False
        for s in model.states:
            for _, branch in s.choices:
                total: sp.Expr = sum(  # type: ignore[assignment]
                    (val if isinstance(val, sp.Expr) else sp.nsimplify(val))
                    for val, _ in branch
                )
                if not sp.simplify(total - sp.Integer(1)).is_zero:
                    return False
        return True

    def is_graph_preserving(self, model: "model.Model") -> bool:
        """Return True iff the region is graph-preserving for *model*.

        A well-defined region is graph-preserving if no transition probability evaluates to
        0 for any instantiation within the region. Regions that are not-welldefined are not graph-preserving.
        Requires an affine parametric model.
        """
        if not self.is_well_defined(model):
            return False
        for val in _unique_transition_values(model):
            if is_parametric(val):
                lo, _ = _affine_image(val, self)
            else:
                lo = val  # type: ignore[assignment]
            if lo <= 0:
                return False
        return True

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}: [{lo}, {hi}]" for n, (lo, hi) in self.bounds.items())
        return f"RectangularRegion({{{parts}}})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unique_transition_values(model: "model.Model") -> set:
    seen: set = set()
    for _, choices in model.transitions.items():
        for _, branch in choices:
            for val, _ in branch:
                seen.add(val)
    return seen


def _unique_reward_values(model: "model.Model") -> set:
    seen: set = set()
    for rm in model.rewards:
        for val in rm.rewards.values():
            seen.add(val)
        for val in rm.transition_rewards.values():
            seen.add(val)
    return seen


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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_DEFAULT_REGION_COLORS: dict[str, str] = {
    "safe": "#4aa02c",
    "unsafe": "#c11b17",
    "neither": "#f5a623",
    "unknown": "#aaaaaa",
}
_EXTRA_COLORS = ["#5b9bd5", "#ed7d31", "#a9d18e", "#ffc000"]


def _draw_solution_isoline(
    ax: "Axes",
    solution_fn,
    threshold: Number,
    x_param: str,
    y_param: str,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    resolution: int,
    shade_safe: bool = False,
) -> list:
    """Draw the isoline ``solution_fn == threshold`` on *ax*.

    *solution_fn* may be a sympy expression with free symbols matching
    *x_param* and *y_param*, or any callable ``(x_val, y_val) -> float``
    that accepts the x and y values as positional arguments.

    When *shade_safe* is ``True``, the region where ``solution_fn >= threshold``
    is filled with a semi-transparent green overlay.

    Returns a list of legend proxy artists that the caller should pass to
    ``ax.legend()``.
    """
    import numpy as np
    import sympy as sp
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    if isinstance(solution_fn, sp.Expr):
        fn = sp.lambdify([sp.Symbol(x_param), sp.Symbol(y_param)], solution_fn, "numpy")
    else:
        fn = solution_fn

    xs = np.linspace(x_lim[0], x_lim[1], resolution)
    ys = np.linspace(y_lim[0], y_lim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y)

    t = float(threshold)
    ax.contour(
        X, Y, Z, levels=[t], colors=["black"], linestyles=["--"], linewidths=[1.5]
    )
    handles: list = [
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"f = {threshold}",
        )
    ]

    if shade_safe:
        z_max = float(np.nanmax(Z))
        if z_max > t:
            ax.contourf(X, Y, Z, levels=[t, z_max], colors=["green"], alpha=0.25)
        handles.append(
            mpatches.Patch(facecolor="green", alpha=0.25, label=f"f ≥ {threshold}")
        )

    return handles


def plot_regions(
    regions: "list[tuple[RectangularRegion, str] | AnnotatedRegion]",
    ax: "Axes | None" = None,
    figsize: tuple[float, float] | None = None,
    colors: dict[str, str] | None = None,
    param_order: tuple[str, str] | None = None,
    solution_fn=None,
    threshold: "Number | None" = None,
    resolution: int = 200,
    shade_safe: bool = False,
    x_lim: "tuple[float, float] | None" = None,
    y_lim: "tuple[float, float] | None" = None,
) -> "Axes":
    """Plot a 2D partition of rectangular regions as colored patches.

    Each ``(region, label)`` pair is rendered as a filled rectangle, similar
    to the region plots produced by `prophesy`.  The default color mapping is
    ``"safe"`` → green, ``"unsafe"`` → red, ``"unknown"`` → gray; override or
    extend via *colors*.

    An optional solution-function isoline can be overlaid by supplying both
    *solution_fn* and *threshold*: the curve where ``solution_fn == threshold``
    is drawn as a dashed black line.  Set *shade_safe* to ``True`` to also fill
    the region where ``solution_fn >= threshold`` with a semi-transparent green
    overlay.

    *regions* may be empty; in that case *param_order* is required and the axis
    limits default to ``(0, 1)`` unless *x_lim* / *y_lim* are given explicitly.

    :param regions: List of ``(region, label)`` pairs or :class:`AnnotatedRegion`
        objects to plot. When :class:`AnnotatedRegion` objects are given,
        *threshold* is required for classification.
    :param ax: Target axes; creates a new figure if ``None``.
    :param figsize: Figure size in inches as ``(width, height)``. Ignored when
        *ax* is provided.
    :param colors: Mapping from label to matplotlib color string, merged with
        the defaults.
    :param param_order: ``(x_param, y_param)`` to control axis assignment.
        Defaults to the insertion order of the first region's ``bounds``.
        Required when *regions* is empty.
    :param solution_fn: A sympy expression or callable ``(x, y) -> float``
        representing the solution function.  Requires *threshold* when given.
    :param threshold: The threshold value to draw the isoline at.  Required
        when *solution_fn* is provided.
    :param resolution: Number of grid points per axis for isoline evaluation.
    :param shade_safe: When ``True``, fill the region where
        ``solution_fn >= threshold`` with a semi-transparent green overlay.
        Requires *solution_fn* and *threshold*.
    :param x_lim: Override the x-axis range as ``(lo, hi)``. Defaults to the
        range spanned by *regions*, or ``(0, 1)`` when *regions* is empty.
    :param y_lim: Override the y-axis range as ``(lo, hi)``. Same default logic
        as *x_lim*.
    :raises ValueError: If any region does not have exactly 2 parameters, a
        *param_order* name is absent from the region bounds, *param_order* is
        omitted when *regions* is empty, or *solution_fn* is given without
        *threshold*.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if solution_fn is not None and threshold is None:
        raise ValueError("threshold is required when solution_fn is provided.")

    # Normalise: convert AnnotatedRegion entries to (region, label) pairs.
    labeled: list[tuple[RectangularRegion, str]]
    if regions and isinstance(regions[0], AnnotatedRegion):
        if threshold is None:
            raise ValueError(
                "threshold is required when regions contains AnnotatedRegion objects."
            )
        labeled = [
            (ar.region, ar.classify(threshold))  # type: ignore[union-attr]
            for ar in regions
        ]
    else:
        labeled = regions  # type: ignore[assignment]

    if not labeled:
        if param_order is None:
            raise ValueError("param_order is required when regions is empty.")
        x_param, y_param = param_order
    else:
        first = labeled[0][0]
        if len(first.bounds) != 2:
            raise ValueError(
                f"plot_regions requires exactly 2 parameters; "
                f"got {len(first.bounds)}: {list(first.bounds)}."
            )
        if param_order is not None:
            x_param, y_param = param_order
            for name in (x_param, y_param):
                if name not in first.bounds:
                    raise ValueError(
                        f"param_order name '{name}' not found in region bounds."
                    )
        else:
            keys = list(first.bounds)
            x_param, y_param = keys[0], keys[1]

    color_map: dict[str, str] = {**_DEFAULT_REGION_COLORS, **(colors or {})}
    extra_idx = 0
    seen: dict[str, str] = {}  # label → color, for legend deduplication

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    assert ax is not None

    for region, label in labeled:
        if label not in color_map:
            color_map[label] = _EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)]
            extra_idx += 1
        color = color_map[label]

        x_lo, x_hi = float(region.bounds[x_param][0]), float(region.bounds[x_param][1])
        y_lo, y_hi = float(region.bounds[y_param][0]), float(region.bounds[y_param][1])

        ax.add_patch(
            mpatches.Rectangle(
                (x_lo, y_lo),
                x_hi - x_lo,
                y_hi - y_lo,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.6,
                label=label if label not in seen else "_nolegend_",
            )
        )
        seen[label] = color

    if x_lim is None:
        all_x = [float(r.bounds[x_param][i]) for r, _ in labeled for i in (0, 1)]
        x_lim = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    if y_lim is None:
        all_y = [float(r.bounds[y_param][i]) for r, _ in labeled for i in (0, 1)]
        y_lim = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    extra_handles: list = []
    if solution_fn is not None:
        assert threshold is not None  # enforced by the guard above
        extra_handles = _draw_solution_isoline(
            ax,
            solution_fn,
            threshold,
            x_param,
            y_param,
            x_lim,
            y_lim,
            resolution,
            shade_safe=shade_safe,
        )

    ax.legend(
        handles=[*ax.get_legend_handles_labels()[0], *extra_handles], loc="upper right"
    )
    return ax


@deprecated(
    version="0.12.0", reason="use plot_regions() with AnnotatedRegion objects directly."
)
def plot_annotated_regions(
    regions: list[AnnotatedRegion],
    threshold: Number,
    ax: "Axes | None" = None,
    figsize: tuple[float, float] | None = None,
    colors: dict[str, str] | None = None,
    param_order: tuple[str, str] | None = None,
    solution_fn=None,
    resolution: int = 200,
    shade_safe: bool = False,
    x_lim: "tuple[float, float] | None" = None,
    y_lim: "tuple[float, float] | None" = None,
) -> "Axes":
    """.. deprecated:: Use :func:`plot_regions` with :class:`AnnotatedRegion` objects instead."""
    return plot_regions(
        regions,  # type: ignore[arg-type]
        ax=ax,
        figsize=figsize,
        colors=colors,
        param_order=param_order,
        solution_fn=solution_fn,
        threshold=threshold,
        resolution=resolution,
        shade_safe=shade_safe,
        x_lim=x_lim,
        y_lim=y_lim,
    )


def plot_annotated_regions_1d(
    regions: list[AnnotatedRegion],
    threshold: Number,
    ax: "Axes | None" = None,
    figsize: tuple[float, float] | None = None,
    colors: dict[str, str] | None = None,
    solution_fn=None,
    resolution: int = 500,
) -> "Axes":
    """Plot 1-parameter annotated regions as value bands on a 2D axis.

    The x-axis is the single parameter; the y-axis is the property value.
    Each annotated region ``[x_lo, x_hi]`` is drawn as two nested shaded
    rectangles:

    - **outer band** ``[lo_min, hi_max]`` (alpha 0.25): the full uncertainty
      range — the property value is guaranteed to lie here for all
      instantiations in the region.
    - **inner band** ``[hi_min, lo_max]`` (alpha 0.55): the tighter verified
      range, drawn only when ``hi_min ≤ lo_max``.

    Both bands are colored by the region's classification relative to
    *threshold* (green/red/orange/gray for safe/unsafe/neither/unknown).
    A horizontal dashed line marks *threshold* on the value axis.

    If *solution_fn* is provided — a sympy expression in the single parameter
    symbol, or a callable ``(x_val) -> float`` — its graph is overlaid as a
    solid black curve.

    :param regions: Annotated regions with exactly 1 parameter each.
    :param threshold: Classification threshold, also drawn as a horizontal line.
    :param ax: Target axes; creates a new figure if ``None``.
    :param figsize: Figure size in inches as ``(width, height)``. Ignored when
        *ax* is provided.
    :param colors: Mapping from classification label to matplotlib color string,
        merged with the defaults.
    :param solution_fn: A sympy expression or callable ``(x) -> float``.
    :param resolution: Number of x-points for evaluating *solution_fn*.
    :raises ValueError: If *regions* is empty or any region has ≠ 1 parameter.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    if not regions:
        raise ValueError("No regions to plot.")

    first_bounds = regions[0].region.bounds
    if len(first_bounds) != 1:
        raise ValueError(
            f"plot_annotated_regions_1d requires exactly 1 parameter; "
            f"got {len(first_bounds)}: {list(first_bounds)}."
        )
    (param,) = first_bounds.keys()

    color_map: dict[str, str] = {**_DEFAULT_REGION_COLORS, **(colors or {})}
    seen: set[str] = set()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    assert ax is not None

    all_x: list[float] = []
    all_y: list[float] = []

    for ar in regions:
        if len(ar.region.bounds) != 1:
            raise ValueError(
                f"All regions must have exactly 1 parameter; "
                f"got {list(ar.region.bounds)}."
            )
        x_lo = float(ar.region.bounds[param][0])
        x_hi = float(ar.region.bounds[param][1])
        lo_min, hi_min = float(ar.min_value[0]), float(ar.min_value[1])
        lo_max, hi_max = float(ar.max_value[0]), float(ar.max_value[1])

        label = ar.classify(threshold)
        color = color_map.get(label, "#aaaaaa")

        # Outer band: full uncertainty range [lo_min, hi_max]
        ax.add_patch(
            mpatches.Rectangle(
                (x_lo, lo_min),
                x_hi - x_lo,
                hi_max - lo_min,
                facecolor=color,
                edgecolor="none",
                alpha=0.25,
                label=label if label not in seen else "_nolegend_",
            )
        )
        seen.add(label)

        # Inner band: tighter verified range [hi_min, lo_max]
        if hi_min <= lo_max:
            ax.add_patch(
                mpatches.Rectangle(
                    (x_lo, hi_min),
                    x_hi - x_lo,
                    lo_max - hi_min,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.55,
                    label="_nolegend_",
                )
            )

        # Gray dotted vertical lines at region boundaries
        for xv in (x_lo, x_hi):
            ax.axvline(xv, color="gray", linestyle=":", linewidth=0.8, zorder=1)

        # Thick colored line on the x-axis highlighting the region span
        ax.plot(
            [x_lo, x_hi],
            [0, 0],
            color=color,
            linewidth=4,
            solid_capstyle="butt",
            zorder=2,
            clip_on=False,
        )

        all_x += [x_lo, x_hi]
        all_y += [lo_min, hi_max]

    x_lim = (min(all_x), max(all_x))

    # Threshold line
    ax.axhline(
        float(threshold),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold = {threshold}",
    )
    all_y.append(float(threshold))

    # Solution function curve
    if solution_fn is not None:
        import sympy as sp

        if isinstance(solution_fn, sp.Expr):
            sym = sp.Symbol(param)
            fn = sp.lambdify(sym, solution_fn, "numpy")
        else:
            fn = solution_fn
        xs = np.linspace(x_lim[0], x_lim[1], resolution)
        ys = fn(xs)
        ax.plot(xs, ys, color="black", linewidth=1.5, label="solution")
        all_y += list(ys)

    # Add legend entries for each seen classification label
    handles, existing_labels = ax.get_legend_handles_labels()
    # Build proxy patches for classification labels not yet in the legend
    for lbl in seen:
        if lbl not in existing_labels:
            handles.append(
                mpatches.Patch(facecolor=color_map.get(lbl, "#aaaaaa"), label=lbl)
            )

    y_pad = (max(all_y) - min(all_y)) * 0.05 or 0.05
    ax.set_xlim(*x_lim)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
    ax.set_xlabel(param)
    ax.set_ylabel("value")
    ax.legend(loc="upper right")
    return ax
