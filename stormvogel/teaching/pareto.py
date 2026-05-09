"""Pareto front geometry visualization for 2-objective MDP model checking.

Each query (w, p) defines:
  - supporting hyperplane   w·x = w·p   (black dashed)
  - infeasible halfspace    w·x ≥ w·p   one red region per query
The green region is the downward closure of the convex hull of all achievable points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import stormvogel.model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from scipy.spatial import ConvexHull, QhullError


EPS: float = 1e-9
_BIG: float = 1e6  # patch coordinates far outside any visible area; matplotlib clips


@dataclass
class ParetoQuery:
    """A single direction-vector query result.

    :param w: Weight/direction vector; both components must be non-negative.
        When ``None``, the point is plotted but no infeasible region or
        supporting hyperplane is drawn for it.
    :param p: Achievable point returned by the model checker for direction w.
    """

    w: tuple[float, float] | None
    p: tuple[float, float]


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _feasible_polygon(
    queries: list[ParetoQuery],
    x_lo: float,
    y_lo: float,
) -> np.ndarray | None:
    """Return ordered (N, 2) vertices of the downward closure of conv({p_i}).

    Adds three corner points — origin (x_lo, y_lo), (max_x, y_lo), (x_lo, max_y) —
    and takes the convex hull of all points. This is the standard construction for the
    downward-closed feasible region of a 2D Pareto front (maximization setting).
    """
    pts = np.array([q.p for q in queries], dtype=float)
    origin = np.array([[x_lo, y_lo]])
    x_cut = np.array([[pts[:, 0].max(), y_lo]])
    y_cut = np.array([[x_lo, pts[:, 1].max()]])
    all_pts = np.concatenate([pts, origin, x_cut, y_cut], axis=0)

    try:
        hull = ConvexHull(all_pts)
    except QhullError:
        return None
    verts = all_pts[hull.vertices]

    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
    return verts[np.argsort(angles)]


def _infeasible_patch(q: ParetoQuery) -> mpatches.Polygon:
    """Large parallelogram covering {x | w · x ≥ w · p}; matplotlib clips to axes."""
    w = np.array(q.w, dtype=float)
    p = np.array(q.p, dtype=float)
    w_norm = w / (np.linalg.norm(w) + EPS)
    d_norm = np.array([-w[1], w[0]], dtype=float)
    d_norm /= np.linalg.norm(d_norm) + EPS

    verts = np.array(
        [
            p - _BIG * d_norm,
            p + _BIG * d_norm,
            p + _BIG * d_norm + _BIG * w_norm,
            p - _BIG * d_norm + _BIG * w_norm,
        ]
    )
    return mpatches.Polygon(
        verts,
        closed=True,
        facecolor="red",
        edgecolor="none",
        alpha=0.25,
        label="_nolegend_",
    )


def _add_hyperplane(q: ParetoQuery, ax: Axes, label: str | None = None) -> None:
    """Draw the supporting hyperplane w · x = w · p as a dashed line on ax."""
    assert q.w is not None
    w1, w2 = q.w
    kw: dict = dict(
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=label if label else "_nolegend_",
    )
    if abs(w2) < EPS:
        ax.axvline(x=q.p[0], **kw)
    elif abs(w1) < EPS:
        ax.axhline(y=q.p[1], **kw)
    else:
        ax.axline(q.p, slope=-w1 / w2, **kw)


# ---------------------------------------------------------------------------
# Shared axis helpers
# ---------------------------------------------------------------------------


def _prepare_ax(
    points: list,
    bbox_pad: float,
    ax: Axes | None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Axes, float, float]:
    """Create axes if needed; return (ax, x_hi, y_hi) with padding applied."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    out = cast(Axes, ax)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_hi = (max(xs) or 1.0) * (1 + bbox_pad)
    y_hi = (max(ys) or 1.0) * (1 + bbox_pad)
    return out, x_hi, y_hi


def _finalize_ax(
    ax: Axes,
    x_hi: float,
    y_hi: float,
    xlabel: str,
    ylabel: str,
    legend: str | bool = "inside",
) -> None:
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(0.0, y_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend == "inside" or legend is True:
        ax.legend(loc="upper right")
    elif legend == "outside":
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_pareto(
    queries: list[ParetoQuery],
    ax: Axes | None = None,
    bbox_pad: float = 0.2,
    labels: tuple[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    legend: str | bool = "inside",
) -> Axes:
    """Visualize Pareto front geometry for a set of direction-vector queries.

    Renders (bottom to top):
    - Red semitransparent halfplanes: infeasible regions, one per query
    - Green filled polygon: feasible region (downward closure of convex hull)
    - Black dashed lines: supporting hyperplanes
    - Black dots: achievable points

    :param queries: Direction-vector query results.
    :param ax: Target axes; creates a new figure if None.
    :param bbox_pad: Fractional padding added around achievable points for axis limits.
    :param labels: Pair of target label strings used for the axis labels.
        Defaults to ``("T_1", "T_2")``.
    :param figsize: Figure size in inches as ``(width, height)``. Ignored when
        *ax* is provided by the caller.
    :param legend: Legend placement. ``"inside"`` (default) places it inside
        the axes; ``"outside"`` places it to the right; ``False`` suppresses it.
    :returns: The populated axes.
    """
    l1, l2 = labels if labels is not None else ("T_1", "T_2")
    resolved_ax, x_hi, y_hi = _prepare_ax([q.p for q in queries], bbox_pad, ax, figsize)

    _infeasible_label_added = False
    _hyperplane_label_added = False
    for q in queries:
        if q.w is None:
            continue
        patch = _infeasible_patch(q)
        if not _infeasible_label_added:
            patch.set_label("Infeasible region")
            _infeasible_label_added = True
        resolved_ax.add_patch(patch)

    verts = _feasible_polygon(queries, 0.0, 0.0)
    if verts is not None:
        resolved_ax.add_patch(
            mpatches.Polygon(
                verts,
                closed=True,
                facecolor="green",
                edgecolor="darkgreen",
                linewidth=0.5,
                alpha=0.4,
                label="Feasible region",
            )
        )
    else:
        # Degenerate (collinear) case: draw a line along the feasible boundary.
        pts = np.array([q.p for q in queries], dtype=float)
        all_pts = np.concatenate([pts, [[0.0, 0.0]]], axis=0)
        axis = int(np.argmax(all_pts.max(axis=0) - all_pts.min(axis=0)))
        line_pts = all_pts[np.argsort(all_pts[:, axis])]
        resolved_ax.plot(
            line_pts[:, 0],
            line_pts[:, 1],
            color="green",
            linewidth=1.5,
            alpha=0.7,
            label="Feasible region",
        )

    for q in queries:
        if q.w is None:
            continue
        _add_hyperplane(
            q,
            resolved_ax,
            label="Supporting hyperplane" if not _hyperplane_label_added else None,
        )
        _hyperplane_label_added = True

    resolved_ax.scatter(
        [q.p[0] for q in queries],
        [q.p[1] for q in queries],
        color="black",
        zorder=5,
        s=40,
        label="Achievable points",
    )
    _finalize_ax(
        resolved_ax,
        x_hi,
        y_hi,
        rf"$P(\diamond\, {l1})$",
        rf"$P(\diamond\, {l2})$",
        legend,
    )
    return resolved_ax


def explore_pareto(
    mdp: stormvogel.model.Model,
    target_labels: list[str],
    weight_vectors: list[tuple[float, float]],
    ax: Axes | None = None,
    bbox_pad: float = 0.2,
    figsize: tuple[float, float] | None = None,
    legend: str | bool = "inside",
) -> list[ParetoQuery]:
    """Query an MDP with a set of weight vectors and visualize the Pareto geometry.

    For each weight vector *w*:

    1. Compute the policy that maximises ``w · (P(reach T_1), P(reach T_2))``
       via :func:`~stormvogel.teaching.multiobjective.compute_weighted_reachability_policy`.
    2. Induce the DTMC from that policy and evaluate individual reachability
       probabilities via
       :func:`~stormvogel.teaching.multiobjective.evaluate_policy_reachability`.
    3. Record the result as a :class:`ParetoQuery` with ``p = (P(T_1), P(T_2))``.

    All collected queries are then passed to :func:`plot_pareto`.

    :param mdp: The input MDP.
    :param target_labels: Exactly two target labels (2-objective setting).
    :param weight_vectors: Sequence of 2D weight vectors to query.
    :param ax: Target axes; a new figure is created if *None*.
    :param bbox_pad: Fractional padding around achievable points for axis limits.
    :param figsize: Figure size in inches as ``(width, height)``. Ignored when
        *ax* is provided by the caller.
    :param legend: Legend placement; forwarded to :func:`plot_pareto`.
    :returns: The collected queries. The plot is rendered as a side effect; use
        :func:`plot_pareto` directly if you need access to the axes.
    :raises ValueError: If *target_labels* does not have exactly two entries.
    :raises ImportError: If stormpy is not installed.

    #TODO this method currently fails to handle MECs.
    """

    if len(target_labels) != 2:
        raise ValueError(
            f"explore_pareto requires exactly 2 target labels; got {len(target_labels)}."
        )

    from stormvogel.teaching.multiobjective import (
        compute_weighted_reachability_policy,
        evaluate_policy_reachability,
    )

    queries: list[ParetoQuery] = []
    for w in weight_vectors:
        result = compute_weighted_reachability_policy(mdp, target_labels, list(w))
        probs = evaluate_policy_reachability(result, target_labels)
        queries.append(ParetoQuery(w=w, p=(probs[0], probs[1])))

    plot_pareto(
        queries,
        ax=ax,
        bbox_pad=bbox_pad,
        labels=(target_labels[0], target_labels[1]),
        figsize=figsize,
        legend=legend,
    )
    return queries
