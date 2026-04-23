"""State elimination for parametric Markov chains.

Implements the primitives and outer loops from Junges (2020) §8.1.2,
Algorithms 2 and 3.  State elimination is the parametric analogue of the
NFA-to-regex construction: non-target states are bypassed one by one until
the solution function can be read off the initial state's outgoing edges.

The primitives (:func:`eliminate_selfloop`, :func:`eliminate_transition`,
:func:`eliminate_state`) operate on a **copy of the model** so that
intermediate steps can be inspected in a notebook::

    copy = pmc.copy()
    eliminate_selfloop(copy, s3)
    eliminate_state(copy, s3)   # inspect copy, then continue

The high-level functions (:func:`solve_reachability`,
:func:`solve_reachability_all`) create the copy internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import sympy as sp

from stormvogel.model.distribution import Distribution

if TYPE_CHECKING:
    import stormvogel.model as model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_expr(val) -> sp.Expr:
    """Convert any stormvogel Value to a sympy Expr."""
    if isinstance(val, sp.Expr):
        return val
    return sp.nsimplify(val)


def _t_reachable_nontargets(
    pmc: "model.Model",
    target_states: list["model.State"],
    order: "list[model.State] | None",
) -> list["model.State"]:
    """Return non-target states from which some target is reachable, in elimination order."""
    target_set = set(target_states)
    # Backward BFS from targets.
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


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------


def eliminate_selfloop(pmc: "model.Model", s: "model.State") -> None:
    """Rescale *s*'s outgoing distribution by ``1/(1 − P(s,s))`` and zero the loop.

    Precondition: ``P(s, s) ≠ 1``.  No-op when ``P(s, s) = 0``.
    """
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


def eliminate_transition(
    pmc: "model.Model", s_in: "model.State", s: "model.State"
) -> None:
    """Add shortcuts from *s_in* to every successor of *s*; zero ``P(s_in, s)``.

    Precondition: ``P(s, s) = 0`` — call :func:`eliminate_selfloop` first.
    """
    action_in, branch_in = next(iter(pmc.transitions[s_in]))
    _, branch_s = next(iter(pmc.transitions[s]))

    w = sp.cancel(_to_expr(branch_in[s])) if s in branch_in else sp.Integer(0)
    if w.is_zero:
        return

    # Merge s_in's existing outgoing edges (minus s) with the shortcut edges
    # through s, computing all values before constructing the Distribution.
    merged: dict[model.State, sp.Expr] = {
        t: _to_expr(val) for val, t in branch_in if t != s
    }
    for val, t in branch_s:
        current = merged.get(t, sp.Integer(0))
        merged[t] = sp.cancel(current + w * _to_expr(val))

    pmc.transitions[s_in][action_in] = Distribution(
        {t: v for t, v in merged.items() if not v.is_zero}
    )


def eliminate_state(pmc: "model.Model", s: "model.State") -> None:
    """Eliminate all incoming edges to *s* (must be loop-free).

    Calls :func:`eliminate_transition` for every predecessor of *s*.
    Afterwards, *s* has no predecessors in the model.
    """
    for s_in in pmc.predecessors(s):
        eliminate_transition(pmc, s_in, s)


# ---------------------------------------------------------------------------
# Outer loops
# ---------------------------------------------------------------------------


def solve_reachability(
    pmc: "model.Model",
    target_states: Iterable["model.State"],
    order: "list[model.State] | None" = None,
) -> sp.Expr:
    """Return the reachability solution function at the initial state.

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
