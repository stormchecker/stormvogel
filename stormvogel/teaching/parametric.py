"""Parameter space partitioning for parametric Markov models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import sympy as sp


import stormvogel.model as model
from stormvogel.model.distribution import Distribution
from stormvogel.parametric._backend import Number
from stormvogel.parametric.region import AnnotatedRegion, RectangularRegion
from stormvogel.teaching.dtmc_evaluation import (
    _check_dtmc,
    _to_expr,
    compute_zero_states,
)


def parameter_space_partitioning(
    model: "model.Model",
    prop: str,
    threshold: Number,
    initial_region: RectangularRegion | None = None,
    max_iterations: int = 100,
) -> list[AnnotatedRegion]:
    """Partition the parameter space into annotated regions.

    Starting from *initial_region* (defaulting to ``[0.01, 0.99]^n``), the
    algorithm repeatedly annotates the front region in the queue and classifies
    it against *threshold* via :meth:`~AnnotatedRegion.classify`:

    - **safe / unsafe**: added to the result; not split further.
    - **unknown / neither**: split along the widest dimension and both halves
      re-queued, as long as the split count is below *max_iterations*.  Once
      the budget is exhausted the queue is drained — each remaining region is
      annotated and collected without further splitting.

    :param model: An affine parametric stormvogel model.
    :param prop: A Storm property string, e.g. ``'P=? [F "target"]'``.
    :param threshold: Classification threshold passed to
        :meth:`~AnnotatedRegion.classify`.
    :param initial_region: Starting region.  Defaults to ``[0.01, 0.99]^n``
        where *n* is the number of declared parameters.
    :param max_iterations: Maximum number of splits before the queue is drained
        without further splitting.
    :returns: Flat list of :class:`~stormvogel.parametric.region.AnnotatedRegion`.
    :raises ValueError: If the initial region is not graph-preserving.
    :raises ImportError: If stormpy is not installed.
    """
    from stormvogel.stormpy_utils.parametric_analysis import AnalyseParametric

    if initial_region is None:
        param_names = list(model.parameters)
        initial_region = RectangularRegion(
            {name: (Fraction(1, 100), Fraction(99, 100)) for name in param_names}
        )

    if not initial_region.is_graph_preserving(model):
        raise ValueError(
            "The initial region is not graph-preserving: at least one transition "
            "probability reaches 0 somewhere within the region."
        )

    analyser = AnalyseParametric(model, prop)

    results: list[AnnotatedRegion] = []
    queue: deque[RectangularRegion] = deque([initial_region])
    splits = 0

    while queue:
        region = queue.popleft()
        annotated = analyser.annotate_region(region, assume_graph_preserving=True)
        label = annotated.classify(threshold)
        if label in ("safe", "unsafe"):
            results.append(annotated)
        elif splits < max_iterations:
            splits += 1
            lo, hi = region.split()
            queue.append(lo)
            queue.append(hi)
        else:
            results.append(annotated)

    return results


def _state_vars_readable(pmc: model.Model) -> dict[model.State, sp.Symbol]:
    """Create teaching-friendly sympy symbols ``p_{name}`` for each state."""
    used: set[str] = set()
    result: dict[model.State, sp.Symbol] = {}
    for i, s in enumerate(pmc.sorted_states):
        if s.friendly_name:
            base = "p_" + s.friendly_name.replace(" ", "_").replace("-", "_")
        else:
            base = f"p_{i}"
        name = base
        suffix = 1
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result[s] = sp.Symbol(name)
    return result


@dataclass
class FeasibilityProblem:
    """Structured encoding of the parametric reachability feasibility problem.

    Mirrors the existential encoding::

        ∃ x̄  ∃ p_{s_0}…p_{s_n}
          ∧  lo_x ≤ x ≤ hi_x          for each parameter  (parameter_bounds)
          ∧  p_s = 0                   for s ∈ S_zero       (zero_equations)
          ∧  p_s = 1                   for s ∈ T            (target_equations)
          ∧  p_s = Σ δ(s,s') p_{s'}   for interior s       (interior_equations)
          ∧  p_{s_init} ≥ λ                                 (threshold)

    Constraints are stored as sympy relational objects (``sp.Eq``, ``sp.Ge``).

    :param parameter_bounds: Mapping from parameter symbol to ``(lower, upper)``.
    :param zero_equations: ``sp.Eq(p_s, 0)`` for each s ∈ S_zero.
    :param target_equations: ``sp.Eq(p_s, 1)`` for each s ∈ T.
    :param interior_equations: ``sp.Eq(p_s, Σ δ·p_{s'})`` for interior s.
    :param threshold: ``sp.Ge(p_{s_init}, λ)``.
    :param state_variables: Mapping from state to its sympy symbol.
    """

    parameter_bounds: dict[sp.Symbol, tuple[sp.Expr, sp.Expr]]
    zero_equations: list[sp.Basic]
    target_equations: list[sp.Basic]
    interior_equations: list[sp.Basic]
    threshold: sp.Basic
    state_variables: dict[model.State, sp.Symbol]

    def _repr_latex_(self) -> str:
        rows = []
        params = ", ".join(sp.latex(s) for s in self.parameter_bounds)
        pvars = ", ".join(sp.latex(s) for s in self.state_variables.values())
        rows.append(rf"\exists & {params},\; {pvars} \\")
        for sym, (lo, hi) in self.parameter_bounds.items():
            rows.append(
                rf"\land & {sp.latex(lo)} \leq {sp.latex(sym)} \leq {sp.latex(hi)} \\"
            )
        for c in (
            *self.zero_equations,
            *self.target_equations,
            *self.interior_equations,
        ):
            rows.append(rf"\land & {sp.latex(c)} \\")
        rows.append(rf"\land & {sp.latex(self.threshold)}")
        body = "\n".join(rows)
        return f"$$\\begin{{array}}{{ll}}\n{body}\n\\end{{array}}$$"


def feasibility_problem(
    pmc: model.Model,
    target_label: str,
    threshold: "sp.Expr | float",
    region: "RectangularRegion",
    zero_states: "Iterable[model.State] | None" = None,
) -> FeasibilityProblem:
    """Build the feasibility encoding for parametric reachability in a pMC.

    Constructs a :class:`FeasibilityProblem` whose constraints are equivalent
    to: *does there exist a parameter valuation inside* ``region`` *such that
    the reachability probability of states labelled* ``target_label`` *at the
    initial state is at least* ``threshold``?

    :param pmc: A stormvogel DTMC (plain or parametric).
    :param target_label: Label identifying target states (reachability = 1).
    :param threshold: Lower bound λ on the initial-state reachability.
    :param region: Rectangular parameter region providing bounds per parameter.
    :param zero_states: States with known reachability 0, or ``None`` to
        auto-detect from the graph structure.
    :returns: A :class:`FeasibilityProblem` with all constraint groups filled.
    :raises ValueError: If ``pmc`` is not a DTMC.
    :raises KeyError: If ``target_label`` is not present in ``pmc``.
    """
    _check_dtmc(pmc)

    param_bounds: dict[sp.Symbol, tuple[sp.Expr, sp.Expr]] = {
        sp.Symbol(name): (_to_expr(lo), _to_expr(hi))
        for name, (lo, hi) in region.bounds.items()
    }

    target_set = pmc.get_states_with_label(target_label)
    if zero_states is not None:
        zero_set = set(zero_states)
    else:
        zero_set = compute_zero_states(pmc, target_set)

    p = _state_vars_readable(pmc)

    zero_eqs: list[sp.Basic] = []
    target_eqs: list[sp.Basic] = []
    interior_eqs: list[sp.Basic] = []

    for s in pmc.sorted_states:
        if s in target_set:
            target_eqs.append(sp.Eq(p[s], sp.Integer(1)))
        elif s in zero_set:
            zero_eqs.append(sp.Eq(p[s], sp.Integer(0)))
        else:
            _, branch = next(iter(s.choices))
            rhs: sp.Expr = sum(  # type: ignore[assignment]
                _to_expr(prob) * p[s_next] for prob, s_next in branch
            )
            interior_eqs.append(sp.Eq(p[s], rhs))

    thresh = sp.Ge(p[pmc.initial_state], _to_expr(threshold))

    return FeasibilityProblem(
        parameter_bounds=param_bounds,
        zero_equations=zero_eqs,
        target_equations=target_eqs,
        interior_equations=interior_eqs,
        threshold=thresh,
        state_variables=p,
    )


# ---------------------------------------------------------------------------
# State elimination (Junges 2020 §8.1.2, Algorithms 2 and 3)
#
# Primitives operate in-place on a model copy so that intermediate steps can
# be inspected in a notebook:
#
#     copy = pmc.copy()
#     eliminate_selfloop(copy, s3)
#     eliminate_state(copy, s3)   # inspect, then continue
#
# The high-level solve_reachability creates the copy internally.
# ---------------------------------------------------------------------------


def _check_markov_chain(pmc: model.Model) -> None:
    if pmc.model_type != model.ModelType.DTMC:
        raise ValueError(
            f"State elimination requires a DTMC (Markov chain); got {pmc.model_type}."
        )


def _t_reachable_nontargets(
    pmc: model.Model,
    target_states: list[model.State],
    order: list[model.State] | None,
) -> list[model.State]:
    """Return non-target states from which some target is reachable, in elimination order."""
    target_set = set(target_states)
    reachable: set[model.State] = set(target_set)
    queue = list(target_set)
    i = 0
    while i < len(queue):
        for p in pmc.predecessors(queue[i]):
            if p not in reachable:
                reachable.add(p)
                queue.append(p)
        i += 1
    candidates = [
        s for s in reachable if s not in target_set and s != pmc.initial_state
    ]
    if order is not None:
        cand_set = set(candidates)
        return [s for s in order if s in cand_set]
    return sorted(candidates, key=lambda s: s.friendly_name or str(s.state_id))


def eliminate_selfloop(pmc: model.Model, s: model.State) -> None:
    """Rescale *s*'s outgoing distribution by ``1/(1 − P(s,s))`` and zero the loop.

    Precondition: ``P(s, s) ≠ 1``.  No-op when ``P(s, s) = 0``.
    """
    _check_markov_chain(pmc)
    if len(pmc.transitions[s]) == 0:
        raise ValueError(f"State {s!r} has no outgoing transitions.")
    action, branch = next(iter(pmc.transitions[s]))
    loop = sp.cancel(_to_expr(branch[s])) if s in branch else sp.Integer(0)
    if loop.is_zero:
        return
    denom = sp.Integer(1) - loop
    new_distr: Distribution = Distribution()
    for val, t in branch:
        if t != s:
            new_distr[t] = sp.cancel(_to_expr(val) / denom)
    pmc.transitions[s][action] = new_distr


def eliminate_transition(pmc: model.Model, s_in: model.State, s: model.State) -> None:
    """Add shortcuts from *s_in* to every successor of *s*; zero ``P(s_in, s)``.

    Precondition: ``P(s, s) = 0`` — call :func:`eliminate_selfloop` first.
    """
    _check_markov_chain(pmc)
    if len(pmc.transitions[s_in]) == 0:
        raise ValueError(f"State {s_in!r} has no outgoing transitions.")
    if len(pmc.transitions[s]) == 0:
        raise ValueError(f"State {s!r} has no outgoing transitions.")
    action_in, branch_in = next(iter(pmc.transitions[s_in]))
    _, branch_s = next(iter(pmc.transitions[s]))

    w = sp.cancel(_to_expr(branch_in[s])) if s in branch_in else sp.Integer(0)
    if w.is_zero:
        return

    merged: dict[model.State, sp.Expr] = {
        t: _to_expr(val) for val, t in branch_in if t != s
    }
    for val, t in branch_s:
        current = merged.get(t, sp.Integer(0))
        merged[t] = sp.cancel(current + w * _to_expr(val))

    pmc.transitions[s_in][action_in] = Distribution(
        {t: v for t, v in merged.items() if not v.is_zero}
    )


def eliminate_state(pmc: model.Model, s: model.State, remove: bool = False) -> None:
    """Eliminate all incoming edges to *s* (must be loop-free).

    Calls :func:`eliminate_transition` for every predecessor of *s*.

    :param remove: If ``True``, remove *s* from the model after elimination.
    """
    _check_markov_chain(pmc)
    for s_in in list(pmc.predecessors(s)):
        eliminate_transition(pmc, s_in, s)
    if remove:
        pmc.remove_state(s, normalize=False, suppress_warning=True)


def solve_reachability(
    pmc: model.Model,
    target_states: Iterable[model.State],
    order: list[model.State] | None = None,
) -> sp.Expr:
    """Return the reachability solution function at the initial state via state elimination.

    Implements Algorithm 2: eliminates all non-target T-reachable states,
    collapses the initial state's self-loop, then sums its outgoing edges to T.
    Creates a copy of *pmc* internally; the original is not modified.

    :param pmc: A parametric DTMC.
    :param target_states: Target states (matched by UUID, may come from the
        original model before copying).
    :param order: Explicit elimination order.  Defaults to lexicographic by
        friendly name.  Affects expression size but not correctness.
    :returns: A :class:`sympy.Expr` rational function in the parameters.
    """
    target_ids = frozenset(s.state_id for s in target_states)
    copy = pmc.copy()
    target_copy = [copy.get_state_by_id(uid) for uid in target_ids]

    for s in _t_reachable_nontargets(copy, target_copy, order):
        eliminate_selfloop(copy, s)
        eliminate_state(copy, s)

    init = copy.initial_state
    eliminate_selfloop(copy, init)

    _, branch = next(iter(copy.transitions[init]))
    result: sp.Expr = sp.Integer(0)
    for val, t in branch:
        if t.state_id in target_ids:
            result = sp.cancel(result + _to_expr(val))
    return result
