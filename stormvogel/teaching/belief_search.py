"""Teaching module: most-probable-path search over the belief space of a POMDP.

Dijkstra and A* over the belief MDP.  Both search for the *most probable*
observation trace leading from an initial belief to a belief that is **good**:
one that puts at least ``threshold`` probability mass on a set of target
states.

Because a path's probability is the *product* of the observation
probabilities along it, the searches run Dijkstra's algorithm in its
multiplicative ("most probable path") form: the priority queue is a max-heap
on path probability instead of a min-heap on path cost.  This is equivalent to
minimising :math:`\\sum -\\log P` but stays exact, since every probability is
a :class:`~fractions.Fraction`.

Every belief the search settles is remembered, so afterwards you can inspect
the whole explored part of the belief space, and reconstruct the most probable
path to *any* of those beliefs, not just to the goal.

Typical usage::

    from stormvogel.teaching.belief_search import dijkstra_belief_search

    result = dijkstra_belief_search(
        pomdp, initial_belief, targets="cheese", threshold=Fraction(1, 2)
    )
    print(result)                 # includes the ending observation
    result.table()                # full belief trace, in a notebook
    len(result.visited)           # every belief seen on the way

**Which one to use.** Prefer :func:`dijkstra_belief_search` unless you know
your heuristic bites.  A* only pays off when it prunes a substantial part of
the belief space, and on POMDPs where some action reliably produces one
observation — mazes, grid worlds, anything with walls to bump into — the
default heuristic is exactly ``1`` everywhere and A* expands precisely the
same beliefs as Dijkstra while paying for the bound.  Measured on the cheese
maze, A* is 10-90% *slower*; on a model with genuinely uncertain observations
it cuts expansions roughly fourfold and the search itself runs 1.2-1.9 times
faster.  Note also that building the transition index is ``O(|S| · |A|)`` and
dominates the total runtime whenever the search only has to touch a few
beliefs of a large POMDP.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count
from typing import TYPE_CHECKING, Callable

from stormvogel.teaching.belief import (
    FLOAT_IMPROVEMENT_TOLERANCE,
    Belief,
    BeliefTransitions,
    BeliefValue,
)

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.observation import Observation
    from stormvogel.model.state import State


__all__ = [
    "BeliefSearchResult",
    "BeliefSearchStep",
    "astar_belief_search",
    "dijkstra_belief_search",
    "make_lookahead_heuristic",
    "make_value_bound_heuristic",
    "target_probability",
]


# ---------------------------------------------------------------------------
# Goal predicate
# ---------------------------------------------------------------------------


def target_probability(
    belief: Belief, targets: "Iterable[State] | frozenset[State]"
) -> "BeliefValue":
    """Return the probability that *belief* assigns to the target states.

    :param belief: A belief over POMDP states.
    :param targets: The set of target (goal) states.
    :returns: :math:`\\sum_{s \\in T} b(s)`, exact unless the belief is
        tracked in ``float``.
    """
    target_set = targets if isinstance(targets, frozenset) else frozenset(targets)
    return sum((p for s, p in belief.items() if s in target_set), Fraction(0))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeliefSearchStep:
    """A single ``action → observation`` transition in the belief MDP.

    :param source: The belief the step starts from.
    :param action: Label of the action taken (empty string for the empty
        action).
    :param observation: The observation received, which is what the agent
        actually gets to see.  ``None`` if the successor states carry no
        observation.
    :param probability: :math:`P(o \\mid b, a)`, the probability of receiving
        this observation after taking this action in *source*.
    :param belief: The belief that results from the Bayesian update.
    """

    source: Belief
    action: str
    observation: "Observation | None"
    probability: "BeliefValue"
    belief: Belief


@dataclass(repr=False)
class BeliefSearchResult:
    """Outcome of a most-probable-path search over the belief space.

    :param steps: The most probable path found, as a list of
        :class:`BeliefSearchStep`.  Empty if the initial belief was already
        good, or if no good belief was found.
    :param probability: Probability of that path (the product of the step
        probabilities); ``1`` for the empty path.
    :param found: Whether a good belief was reached.
    :param initial_belief: The belief the search started from.
    :param targets: The target states defining the goal predicate.
    :param threshold: A belief is good when its target probability is at
        least this value.
    :param visited: Every settled belief, mapped to the probability of the
        most probable path reaching it.  Insertion-ordered by settling time,
        so ``list(visited)`` is the order in which the search explored the
        belief space.
    :param frontier: Beliefs that were discovered but never settled, mapped
        to the best path probability known for them.
    :param parents: For every discovered belief, the step that reaches it on
        its most probable known path.  Used by :meth:`path_to`.
    :param expansions: Number of belief expansions performed.
    :param truncated: Whether the search stopped because it hit
        ``max_expansions`` rather than because it ran out of beliefs.
    :param algorithm: Name of the algorithm used, for reporting.
    :param elapsed: Wall-clock seconds spent searching, excluding
        :attr:`setup_elapsed`.
    :param setup_elapsed: Wall-clock seconds spent building the transition
        index before the search started.  This is ``O(|S| · |A|)`` in the
        POMDP, so on a large model with a short search it can dominate
        :attr:`elapsed` — keep the two apart when quoting throughput.
    """

    steps: "list[BeliefSearchStep]"
    probability: "BeliefValue"
    found: bool
    initial_belief: Belief
    targets: "frozenset[State]"
    threshold: Fraction
    visited: "dict[Belief, BeliefValue]" = field(default_factory=dict)
    frontier: "dict[Belief, BeliefValue]" = field(default_factory=dict)
    parents: "dict[Belief, BeliefSearchStep]" = field(default_factory=dict)
    expansions: int = 0
    truncated: bool = False
    algorithm: str = "Dijkstra"
    elapsed: float = 0.0
    setup_elapsed: float = 0.0

    # --- Path accessors ------------------------------------------------------

    @property
    def beliefs(self) -> "list[Belief]":
        """The beliefs along the path, starting with the initial belief."""
        return [self.initial_belief] + [step.belief for step in self.steps]

    @property
    def actions(self) -> "list[str]":
        """The action labels along the path."""
        return [step.action for step in self.steps]

    @property
    def observations(self) -> "list[Observation | None]":
        """The observations received along the path."""
        return [step.observation for step in self.steps]

    @property
    def trace(self) -> "list[tuple[str, str]]":
        """The path as ``(action, observation alias)`` pairs.

        Has the shape accepted by
        :func:`~stormvogel.teaching.belief.belief_trace`.
        """
        return [(step.action, _obs_name(step.observation)) for step in self.steps]

    @property
    def final_belief(self) -> Belief:
        """The last belief on the path (the initial belief if it is empty)."""
        return self.steps[-1].belief if self.steps else self.initial_belief

    @property
    def final_observation(self) -> "Observation | None":
        """The observation that ends the path — the one that made the belief good.

        ``None`` when the path is empty (the initial belief was already good)
        or when the target states carry no observation.
        """
        return self.steps[-1].observation if self.steps else None

    @property
    def final_target_probability(self) -> "BeliefValue":
        """Probability that the final belief assigns to the target states."""
        return target_probability(self.final_belief, self.targets)

    # --- Search effort -------------------------------------------------------

    @property
    def explored(self) -> int:
        """Number of distinct beliefs the search settled, i.e. ``len(visited)``."""
        return len(self.visited)

    @property
    def discovered(self) -> int:
        """Number of distinct beliefs reached at all, settled or still open."""
        return len(self.visited) + len(self.frontier)

    @property
    def total_elapsed(self) -> float:
        """Wall-clock seconds for the index build plus the search."""
        return self.setup_elapsed + self.elapsed

    @property
    def beliefs_per_second(self) -> float:
        """Beliefs settled per second of :attr:`elapsed` search time.

        Excludes the index build, which is a fixed cost paid before the search
        begins; use :attr:`total_elapsed` for end-to-end throughput.  Returns
        ``0.0`` if no time was measured.
        """
        return self.explored / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def expansions_per_second(self) -> float:
        """Belief expansions per second of :attr:`elapsed` search time.

        Differs from :attr:`beliefs_per_second` only when a belief is reopened
        because a better path to it was found.
        """
        return self.expansions / self.elapsed if self.elapsed > 0 else 0.0

    def path_to(self, belief: Belief) -> "list[BeliefSearchStep]":
        """Reconstruct the most probable known path to *belief*.

        Works for any belief the search discovered, not just the goal.

        :param belief: A belief in :attr:`visited` or :attr:`frontier`.
        :returns: The steps leading from the initial belief to *belief*.
        :raises KeyError: If *belief* was never discovered by the search.
        """
        if belief == self.initial_belief:
            return []
        if belief not in self.parents:
            raise KeyError(f"Belief was not discovered by the search: {belief!r}")
        steps: "list[BeliefSearchStep]" = []
        current = belief
        while current != self.initial_belief:
            step = self.parents[current]
            steps.append(step)
            current = step.source
        steps.reverse()
        return steps

    # --- Reporting -----------------------------------------------------------

    def __str__(self) -> str:
        lines: "list[str]" = []
        if not self.found:
            reason = (
                f"the expansion limit of {self.expansions} was reached"
                if self.truncated
                else "the reachable belief space was exhausted"
            )
            lines.append(
                f"{self.algorithm}: no belief with target probability "
                f">= {_fmt(self.threshold)} found; {reason}."
            )
        elif not self.steps:
            lines.append(
                f"{self.algorithm}: the initial belief is already good "
                f"(target probability {_fmt(self.final_target_probability)} "
                f">= {_fmt(self.threshold)})."
            )
        else:
            lines.append(
                f"{self.algorithm}: found a good belief in {len(self.steps)} step(s) "
                f"with path probability {_fmt(self.probability)}."
            )
            lines.append(
                f"  target probability of the final belief: "
                f"{_fmt(self.final_target_probability)} (threshold {_fmt(self.threshold)})"
            )
            lines.append(f"  actions:      {', '.join(self.actions)}")
            lines.append(
                f"  observations: "
                f"{', '.join(_obs_name(o) for o in self.observations)}"
            )
            lines.append(f"  ending observation: {_obs_name(self.final_observation)}")
        lines.append(
            f"  {self.explored} belief(s) explored, {self.expansions} expansion(s), "
            f"{len(self.frontier)} left on the frontier ({self.discovered} discovered)"
        )
        if self.elapsed > 0:
            lines.append(
                f"  {self.elapsed * 1000:.1f} ms searching "
                f"({self.beliefs_per_second:,.0f} beliefs/s) "
                f"+ {self.setup_elapsed * 1000:.1f} ms building the index"
            )
        return "\n".join(lines)

    __repr__ = __str__

    def table(self) -> None:
        """Render the path as an HTML table in a Jupyter notebook.

        Columns are Step / Action / Observation / Probability / Belief, where
        the probability is the observation probability of that single step.
        Row 0 shows the initial belief.
        """
        try:
            from IPython.display import HTML, display
        except ImportError as e:
            raise ImportError(
                "BeliefSearchResult.table requires IPython (pip install ipython)."
            ) from e

        display(HTML(self._html()))

    def _repr_html_(self) -> str:
        return self._html()

    def _html(self) -> str:
        header = (
            "<tr><th>Step</th><th>Action</th><th>Observation</th>"
            "<th>P(o | b, a)</th><th>Belief</th></tr>"
        )
        rows = [
            "<tr><td>0</td><td>—</td><td>—</td><td>—</td>"
            f"<td style='white-space:nowrap'>{self.initial_belief._repr_latex_()}</td></tr>"
        ]
        for i, step in enumerate(self.steps, start=1):
            rows.append(
                f"<tr><td>{i}</td><td>{step.action or '—'}</td>"
                f"<td>{_obs_name(step.observation)}</td>"
                f"<td>{_fmt(step.probability)}</td>"
                f"<td style='white-space:nowrap'>{step.belief._repr_latex_()}</td></tr>"
            )
        summary = str(self).replace("\n", "<br>")
        return f"<table>{header}{''.join(rows)}</table><p>{summary}</p>"


# ---------------------------------------------------------------------------
# Transition index
# ---------------------------------------------------------------------------


class _MemoTransitions(BeliefTransitions):
    """A :class:`BeliefTransitions` that remembers every expansion it computes.

    A* asks the heuristic to expand each belief as it is *generated*, and the
    search then expands the same belief again when it is *settled*.  Sharing
    one memo between the two makes the second expansion free, which is what
    keeps the heuristic from costing more than it saves.
    """

    def __init__(self, pomdp: "Model", exact: bool = True) -> None:
        super().__init__(pomdp, exact)
        self._actions: "dict[Belief, list[str]]" = {}
        self._successors: dict[
            "tuple[Belief, str]",
            "list[tuple[BeliefValue, Observation | None, Belief]]",
        ] = {}

    def actions(self, belief: Belief) -> "list[str]":
        cached = self._actions.get(belief)
        if cached is None:
            cached = super().actions(belief)
            self._actions[belief] = cached
        return cached

    def successors(
        self, belief: Belief, action_label: str
    ) -> "list[tuple[BeliefValue, Observation | None, Belief]]":
        key = (belief, action_label)
        cached = self._successors.get(key)
        if cached is None:
            cached = list(super().successors(belief, action_label))
            self._successors[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def make_lookahead_heuristic(
    pomdp: "Model",
    targets: "Iterable[State] | str",
    threshold: "Fraction | float | int",
    depth: int = 1,
    exact: bool = True,
) -> "Callable[[Belief], BeliefValue]":
    """Build the default admissible heuristic for :func:`astar_belief_search`.

    The heuristic returns an *upper bound* on the probability of the most
    probable path from ``b`` to a good belief, obtained by looking *depth*
    steps ahead and optimistically assuming that everything after that is
    free::

        h_k(b) = 1                                        if b is good
        h_0(b) = 1                                        otherwise
        h_k(b) = max_{a, o} P(o | b, a) · h_{k-1}(b')     otherwise

    Each level is an upper bound because a belief that is not yet good needs
    at least one more step, and that step contributes at most
    :math:`\\max_{a,o} P(o \\mid b, a)`.  Deeper lookahead is never weaker
    (``h_k <= h_{k-1}``) and never overestimates, so any *depth* is
    admissible — and in fact consistent, so A* never has to reopen a belief.

    ``depth=0`` gives ``h ≡ 1``, which turns A* back into Dijkstra.  On a
    POMDP whose actions each produce a single observation with probability 1,
    every bound degenerates to ``1`` for the same reason: the heuristic only
    pays off when observations are genuinely uncertain.  Deeper lookahead
    prunes more of the belief space but costs an extra expansion layer per
    level, so it trades search time for successor computations.

    :param pomdp: The POMDP being searched.
    :param targets: Target states, or the label identifying them.
    :param threshold: Minimum target probability for a belief to be good.
    :param depth: Number of steps to look ahead.  Must be non-negative.
    :param exact: Track beliefs in exact arithmetic; ``False`` uses ``float``.
    :returns: A memoised heuristic function.
    """
    index = _MemoTransitions(pomdp, exact)
    return _lookahead_heuristic(
        index, _resolve_targets(pomdp, targets), _check_threshold(threshold), depth
    )


def _lookahead_heuristic(
    index: BeliefTransitions,
    targets: "frozenset[State]",
    threshold: Fraction,
    depth: int,
) -> "Callable[[Belief], BeliefValue]":
    """Memoised implementation of :func:`make_lookahead_heuristic`."""
    if depth < 0:
        raise ValueError(f"depth must be non-negative; got {depth}.")
    cache: "dict[tuple[Belief, int], BeliefValue]" = {}

    def bound(belief: Belief, remaining: int) -> "BeliefValue":
        if target_probability(belief, targets) >= threshold:
            return Fraction(1)
        if remaining == 0:
            return Fraction(1)
        cached = cache.get((belief, remaining))
        if cached is not None:
            return cached
        best = Fraction(0)
        for action in index.actions(belief):
            for obs_prob, _obs, successor in index.successors(belief, action):
                # Deeper bounds are at most 1, so this branch cannot win.
                if obs_prob <= best:
                    continue
                value = obs_prob * bound(successor, remaining - 1)
                if value > best:
                    best = value
        cache[(belief, remaining)] = best
        return best

    return lambda belief: bound(belief, depth)


def make_value_bound_heuristic(
    pomdp: "Model",
    targets: "Iterable[State] | str",
    threshold: "Fraction | float | int",
    values: "Mapping[State, Fraction | float | int] | None" = None,
) -> "Callable[[Belief], BeliefValue]":
    """Build an admissible heuristic from a model-checked state value function.

    Takes the fully observable value :math:`V_\\text{MDP}(s) = P_{\\max}(s
    \\models F\\, T)`, lifts it to the belief by expectation, and rescales it
    by the threshold::

        h(b) = 1                                          if b is good
        h(b) = min(1, (Σ_s b(s) · V_MDP(s)) / threshold)  otherwise

    **The division by the threshold is what makes this admissible.**  The
    search maximises the probability of an *observation trace*, while
    :math:`V_\\text{MDP}` bounds the probability of *being in a target state*,
    and the two only line up at ``threshold = 1``.  If a trace of probability
    ``p`` ends in a good belief, then playing that trace reaches a target
    state with probability at least ``p · threshold``, so

    .. math::

        p \\cdot \\text{threshold}
            \\leq V_\\text{POMDP}(b)
            \\leq \\textstyle\\sum_s b(s)\\, V_\\text{MDP}(s),

    using the standard QMDP argument that an MDP oracle does at least as well
    as any observation-based policy.  Dropping the division underestimates
    ``h`` as soon as ``threshold < 1`` and A* can then return a suboptimal
    path.  The bound is also consistent, since :math:`V_\\text{MDP}` satisfies
    the Bellman inequality :math:`V(s) \\geq \\sum_{s'} P(s'\\mid s,a) V(s')`.

    The bound is only informative when :math:`V_\\text{MDP}` is genuinely
    below the threshold somewhere.  On a model where the target is reachable
    with probability 1 from every state — a maze with no dead ends, say —
    :math:`V_\\text{MDP} \\equiv 1` and the heuristic degenerates to ``1``.
    It pays off when the underlying MDP has unavoidable failure probability,
    and it is complementary to :func:`make_lookahead_heuristic`, which instead
    needs the *observations* to be uncertain.

    :param pomdp: The POMDP being searched.
    :param targets: Target states, or the label identifying them.
    :param threshold: Minimum target probability for a belief to be good.
    :param values: The state value function, typically from
        :func:`~stormvogel.teaching.pomdp_backup.mdp_bound_alpha` or from
        model checking ``Pmax=? [F "label"]`` on the fully observable model.
        Defaults to computing it with
        :func:`~stormvogel.teaching.pomdp_backup.mdp_bound_alpha`, which needs
        *targets* to be a label and requires stormpy.
    :returns: A memoised heuristic function.
    :raises ValueError: If *values* is omitted and *targets* is not a label.
    """
    target_states = _resolve_targets(pomdp, targets)
    checked_threshold = _check_threshold(threshold)

    if values is None:
        if not isinstance(targets, str):
            raise ValueError(
                "Computing the MDP value function needs a target label; pass "
                "targets as a label, or supply `values` yourself."
            )
        from stormvogel.teaching.pomdp_backup import mdp_bound_alpha

        value_map: "dict[State, Fraction]" = dict(
            mdp_bound_alpha(pomdp, targets).values
        )
    else:
        value_map = {s: Fraction(v) for s, v in values.items()}

    return _value_bound_heuristic(value_map, target_states, checked_threshold)


def _value_bound_heuristic(
    values: "dict[State, Fraction]",
    targets: "frozenset[State]",
    threshold: Fraction,
) -> "Callable[[Belief], BeliefValue]":
    """Memoised implementation of :func:`make_value_bound_heuristic`."""
    cache: "dict[Belief, BeliefValue]" = {}
    one = Fraction(1)

    def heuristic(belief: Belief) -> "BeliefValue":
        cached = cache.get(belief)
        if cached is not None:
            return cached
        if threshold == 0 or target_probability(belief, targets) >= threshold:
            bound = one
        else:
            expected = sum(
                (values.get(s, Fraction(0)) * p for s, p in belief.items()),
                Fraction(0),
            )
            bound = min(one, expected / threshold)
        cache[belief] = bound
        return bound

    return heuristic


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def dijkstra_belief_search(
    pomdp: "Model",
    initial_belief: "Belief | Mapping[State, Fraction | float | int]",
    targets: "Iterable[State] | str",
    threshold: "Fraction | float | int" = Fraction(1),
    *,
    exact: bool = True,
    max_expansions: int = 10_000,
) -> BeliefSearchResult:
    """Find the most probable path to a good belief with Dijkstra's algorithm.

    Explores the belief MDP from *initial_belief*, always expanding the
    reachable belief whose most probable path is most probable overall, and
    stops at the first belief that assigns at least *threshold* probability
    to *targets*.  Because path probabilities only ever decrease along a path,
    the first good belief that is settled is reached by the globally most
    probable path.

    Every settled belief is kept in :attr:`BeliefSearchResult.visited`, in
    exploration order.

    :param pomdp: A POMDP with deterministic (non-stochastic) state
        observations.
    :param initial_belief: The belief to start from, as a :class:`Belief` or
        as a mapping from states to probabilities summing to 1.
    :param targets: The target states, or the label identifying them.
    :param threshold: A belief is good when its target probability is at
        least this value.  Defaults to 1, i.e. certainty.
    :param exact: Track beliefs in exact :class:`~fractions.Fraction`
        arithmetic.  Set to ``False`` for ``float``, which is faster on long
        traces but makes belief identity approximate; see
        :data:`~stormvogel.teaching.belief.FLOAT_KEY_PRECISION`.
    :param max_expansions: Safety limit on the number of belief expansions.
        The belief space can be infinite, so the search needs a budget.
    :returns: The :class:`BeliefSearchResult`.
    :raises ValueError: If *pomdp* is not a POMDP, if a state has a stochastic
        observation, if *initial_belief* does not sum to 1, or if *threshold*
        is outside [0, 1].
    """
    setup_started = time.perf_counter()
    index = _MemoTransitions(pomdp, exact)
    initial = _as_belief(initial_belief, exact)
    target_states = _resolve_targets(pomdp, targets)
    checked_threshold = _check_threshold(threshold)
    setup_elapsed = time.perf_counter() - setup_started

    return _search(
        index=index,
        initial_belief=initial,
        targets=target_states,
        threshold=checked_threshold,
        heuristic=lambda _belief: Fraction(1),
        algorithm="Dijkstra",
        max_expansions=max_expansions,
        setup_elapsed=setup_elapsed,
    )


def astar_belief_search(
    pomdp: "Model",
    initial_belief: "Belief | Mapping[State, Fraction | float | int]",
    targets: "Iterable[State] | str",
    threshold: "Fraction | float | int" = Fraction(1),
    *,
    heuristic: "Callable[[Belief], BeliefValue] | None" = None,
    depth: int = 1,
    exact: bool = True,
    max_expansions: int = 10_000,
) -> BeliefSearchResult:
    """Find the most probable path to a good belief with A*.

    Identical to :func:`dijkstra_belief_search` except that beliefs are
    prioritised by ``g(b) * h(b)`` instead of by ``g(b)`` alone, where ``g(b)``
    is the probability of the best path found to *b* and ``h(b)`` is an
    optimistic estimate of the probability still available from *b* onwards.

    The heuristic must be **admissible**: ``h(b)`` may never be smaller than
    the true probability of the most probable path from *b* to a good belief,
    or the search may return a suboptimal path.  ``h ≡ 1`` is trivially
    admissible and reduces A* to Dijkstra.  Beliefs are reopened when a better
    path to them is found, so a merely admissible (non-consistent) heuristic
    is still handled correctly.

    :param pomdp: A POMDP with deterministic (non-stochastic) state
        observations.
    :param initial_belief: The belief to start from, as a :class:`Belief` or
        as a mapping from states to probabilities summing to 1.
    :param targets: The target states, or the label identifying them.
    :param threshold: A belief is good when its target probability is at
        least this value.  Defaults to 1, i.e. certainty.
    :param heuristic: Upper bound on the remaining path probability.  Defaults
        to :func:`make_lookahead_heuristic` with the given *depth*.
    :param depth: Lookahead depth of the default heuristic.  Ignored when
        *heuristic* is given.
    :param exact: Track beliefs in exact :class:`~fractions.Fraction`
        arithmetic; ``False`` uses ``float``.
    :param max_expansions: Safety limit on the number of belief expansions.
    :returns: The :class:`BeliefSearchResult`.
    :raises ValueError: If *pomdp* is not a POMDP, if a state has a stochastic
        observation, if *initial_belief* does not sum to 1, if *threshold*
        is outside [0, 1], or if *depth* is negative.
    """
    setup_started = time.perf_counter()
    index = _MemoTransitions(pomdp, exact)
    initial = _as_belief(initial_belief, exact)
    target_states = _resolve_targets(pomdp, targets)
    checked_threshold = _check_threshold(threshold)
    if heuristic is None:
        heuristic = _lookahead_heuristic(index, target_states, checked_threshold, depth)
    setup_elapsed = time.perf_counter() - setup_started

    return _search(
        index=index,
        initial_belief=initial,
        targets=target_states,
        threshold=checked_threshold,
        heuristic=heuristic,
        algorithm="A*",
        max_expansions=max_expansions,
        setup_elapsed=setup_elapsed,
    )


def _search(
    index: BeliefTransitions,
    initial_belief: Belief,
    targets: "frozenset[State]",
    threshold: Fraction,
    heuristic: "Callable[[Belief], BeliefValue]",
    algorithm: str,
    max_expansions: int,
    setup_elapsed: float = 0.0,
) -> BeliefSearchResult:
    """Shared max-product Dijkstra/A* driver.

    ``best[b]`` is the probability of the most probable path to *b* found so
    far, which plays the role of the ``g`` value in an additive A*.  The queue
    is a min-heap on ``-(best[b] * heuristic(b))``, i.e. a max-heap on the
    optimistic path probability through *b*.
    """
    started = time.perf_counter()
    # A better path must beat the known one by this much to count.  Exact
    # arithmetic is exact, so any gain counts; float noise is not, so a bare
    # ">" would let the same belief improve on itself indefinitely.
    # Integer zero, not 0.0: multiplying a Fraction by a float would silently
    # drop the search out of exact arithmetic.
    tolerance = 0 if index.exact else FLOAT_IMPROVEMENT_TOLERANCE
    # Most probable path known to each discovered belief.
    best: "dict[Belief, BeliefValue]" = {initial_belief: Fraction(1)}
    # The step reaching each belief on that path.
    parents: "dict[Belief, BeliefSearchStep]" = {}
    # Beliefs already expanded, and the probability they were expanded with.
    visited: "dict[Belief, BeliefValue]" = {}

    tiebreak = count()
    queue: "list[tuple[BeliefValue, int, Belief]]" = [
        (-heuristic(initial_belief), next(tiebreak), initial_belief)
    ]

    expansions = 0
    truncated = False

    while queue:
        _priority, _tie, belief = heapq.heappop(queue)
        probability = best[belief]
        # Skip stale queue entries and re-expansions that cannot improve.
        settled = visited.get(belief)
        if settled is not None and probability <= settled * (1 + tolerance):
            continue
        visited[belief] = probability

        if target_probability(belief, targets) >= threshold:
            return _result(
                steps=_reconstruct(parents, initial_belief, belief),
                probability=probability,
                found=True,
                initial_belief=initial_belief,
                targets=targets,
                threshold=threshold,
                visited=visited,
                best=best,
                parents=parents,
                expansions=expansions,
                truncated=False,
                algorithm=algorithm,
                elapsed=time.perf_counter() - started,
                setup_elapsed=setup_elapsed,
            )

        if expansions >= max_expansions:
            truncated = True
            break
        expansions += 1

        for action in index.actions(belief):
            for obs_prob, obs, successor in index.successors(belief, action):
                candidate = probability * obs_prob
                known = best.get(successor)
                if known is not None and candidate <= known * (1 + tolerance):
                    continue
                best[successor] = candidate
                parents[successor] = BeliefSearchStep(
                    source=belief,
                    action=action,
                    observation=obs,
                    probability=obs_prob,
                    belief=successor,
                )
                heapq.heappush(
                    queue,
                    (-(candidate * heuristic(successor)), next(tiebreak), successor),
                )

    return _result(
        steps=[],
        probability=Fraction(0),
        found=False,
        initial_belief=initial_belief,
        targets=targets,
        threshold=threshold,
        visited=visited,
        best=best,
        parents=parents,
        expansions=expansions,
        truncated=truncated,
        algorithm=algorithm,
        elapsed=time.perf_counter() - started,
        setup_elapsed=setup_elapsed,
    )


def _result(
    *,
    steps: "list[BeliefSearchStep]",
    probability: "BeliefValue",
    found: bool,
    initial_belief: Belief,
    targets: "frozenset[State]",
    threshold: Fraction,
    visited: "dict[Belief, BeliefValue]",
    best: "dict[Belief, BeliefValue]",
    parents: "dict[Belief, BeliefSearchStep]",
    expansions: int,
    truncated: bool,
    algorithm: str,
    elapsed: float,
    setup_elapsed: float,
) -> BeliefSearchResult:
    """Assemble a :class:`BeliefSearchResult`, splitting off the open frontier."""
    return BeliefSearchResult(
        steps=steps,
        probability=probability,
        found=found,
        initial_belief=initial_belief,
        targets=targets,
        threshold=threshold,
        visited=visited,
        frontier={b: p for b, p in best.items() if b not in visited},
        parents=parents,
        expansions=expansions,
        truncated=truncated,
        algorithm=algorithm,
        elapsed=elapsed,
        setup_elapsed=setup_elapsed,
    )


def _reconstruct(
    parents: "dict[Belief, BeliefSearchStep]",
    initial_belief: Belief,
    belief: Belief,
) -> "list[BeliefSearchStep]":
    """Walk the parent pointers back from *belief* to *initial_belief*."""
    steps: "list[BeliefSearchStep]" = []
    current = belief
    while current != initial_belief:
        step = parents[current]
        steps.append(step)
        current = step.source
    steps.reverse()
    return steps


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def _as_belief(
    initial_belief: "Belief | Mapping[State, Fraction | float | int]",
    exact: bool = True,
) -> Belief:
    """Coerce *initial_belief* to a :class:`Belief` and check that it sums to 1."""
    convert = Fraction if exact else float
    belief = (
        initial_belief
        if isinstance(initial_belief, Belief)
        else Belief(
            {s: convert(p) for s, p in initial_belief.items() if p != 0}  # type: ignore[misc]
        )
    )
    total = sum(belief.values(), Fraction(0))
    if abs(total - 1) > Fraction(1, 10**9):
        raise ValueError(f"initial_belief must sum to 1; got {total}.")
    return belief


def _resolve_targets(
    pomdp: "Model", targets: "Iterable[State] | str"
) -> "frozenset[State]":
    """Resolve *targets* to a set of states, accepting a label as a shorthand."""
    if isinstance(targets, str):
        if targets not in pomdp.state_labels:
            raise ValueError(f"The model has no states labelled {targets!r}.")
        return frozenset(pomdp.get_states_with_label(targets))
    return frozenset(targets)


def _check_threshold(threshold: "Fraction | float | int") -> Fraction:
    """Coerce *threshold* to a :class:`~fractions.Fraction` in [0, 1]."""
    value = Fraction(threshold)
    if not (Fraction(0) <= value <= Fraction(1)):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}.")
    return value


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _obs_name(observation: "Observation | None") -> str:
    """Best short name for *observation*, tolerating missing aliases."""
    if observation is None:
        return "—"
    try:
        return observation.alias
    except RuntimeError:
        return str(observation.observation_id)


def _fmt(probability: "BeliefValue") -> str:
    """Format a probability as an exact fraction plus a decimal approximation."""
    if not isinstance(probability, Fraction):
        return f"{float(probability):.6g}"
    if probability.denominator == 1:
        return str(probability.numerator)
    return f"{probability} ≈ {float(probability):.4g}"
