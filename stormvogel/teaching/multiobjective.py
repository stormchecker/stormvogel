"""Goal unfolding and weighted multi-target reachability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import sympy as sp

import stormvogel.bird
import stormvogel.model
from stormvogel.transformations.eliminate_transition_rewards import (
    eliminate_transition_rewards,
)

if TYPE_CHECKING:
    import stormvogel.result
    from stormvogel.teaching.lp import LP


@overload
def goal_unfolding(
    mdp: stormvogel.model.Model,
    goal_labels: list[str],
    return_state_bits: Literal[True],
) -> "tuple[stormvogel.model.Model, dict]": ...


@overload
def goal_unfolding(
    mdp: stormvogel.model.Model,
    goal_labels: list[str],
    return_state_bits: Literal[False] = ...,
) -> "stormvogel.model.Model": ...


def goal_unfolding(
    mdp: stormvogel.model.Model,
    goal_labels: list[str],
    return_state_bits: bool = False,
) -> "stormvogel.model.Model | tuple[stormvogel.model.Model, dict]":
    """Build the goal-unfolding product of an MDP.

    The product state (s, b) tracks which goals have been visited on the path
    to reach it. b[i] = True iff a state labelled goal_labels[i] has been
    visited (including s itself). Bits never flip back to False.

    Transition semantics: (s, b) --α--> (t, delta(b, L(t)))
    where delta(b, L(t))[i] = b[i] OR (goal_labels[i] in L(t)).

    Product state labels: L(s).

    :param return_state_bits: When True, return a pair (model, bits_map) where
        bits_map maps each unfolded State to its bit tuple.
    """
    m = len(goal_labels)
    _bit_vars = [stormvogel.model.Variable(f"_bit_{i}") for i in range(m)]

    _init_labels = frozenset(mdp.initial_state.labels)
    _init_bits = tuple(goal_labels[i] in _init_labels for i in range(m))
    _init = stormvogel.bird.BirdState(_mdp_state=mdp.initial_state, _bits=_init_bits)

    def _next_bits(bits: tuple, state_labels: frozenset[str]) -> tuple:
        return tuple(bits[i] or (goal_labels[i] in state_labels) for i in range(m))

    def _delta(sq, a: str):
        next_distr = sq._mdp_state.get_outgoing_transitions(mdp.action(a))
        if next_distr is None:
            raise RuntimeError(
                f"No transitions from {sq._mdp_state} under action '{a}'"
            )
        return [
            (
                prob,
                stormvogel.bird.BirdState(
                    _mdp_state=t, _bits=_next_bits(sq._bits, frozenset(t.labels))
                ),
            )
            for prob, t in next_distr
        ]

    def _labels(sq) -> list[str]:
        labels = [lbl for lbl in sq._mdp_state.labels if lbl != "init"]
        if sq == _init:
            labels = ["init"] + labels
        return labels

    def _friendly_name(sq) -> str:
        name = sq._mdp_state.friendly_name or str(sq._mdp_state.state_id)
        bits_str = "".join("1" if b else "0" for b in sq._bits)
        return f"({name},{bits_str})"

    def _available_actions(sq) -> list[str]:
        return [
            a.label for a in sq._mdp_state.available_actions() if a.label is not None
        ]

    def _valuations(sq) -> dict:
        return {_bit_vars[i]: sq._bits[i] for i in range(m)}

    result = stormvogel.bird.build_bird(
        delta=_delta,
        init=_init,
        labels=_labels,
        friendly_names=_friendly_name,
        available_actions=_available_actions,
        valuations=_valuations if return_state_bits else None,
    )

    result.add_self_loops()
    if return_state_bits:
        bits_map = {
            s: tuple(s.valuations.get(_bit_vars[i], False) for i in range(m))
            for s in result.states
        }
        return result, bits_map
    return result


def weighted_multi_target_reachability(
    mdp: stormvogel.model.Model,
    target_labels: list[str],
    weights: list[float],
) -> stormvogel.model.Model:
    """Build a model for weighted multi-target reachability via goal unfolding.

    Each target T_i is identified by label target_labels[i] with weight weights[i].
    The returned model has a single state reward model "weighted_reach" whose
    expected total reward equals sum_i weights[i] * P(reach T_i).

    Uses goal_unfolding to track first visits and transition rewards (eliminated
    to state rewards via auxiliary entry states) to assign weights.

    :param mdp: The MDP.
    :param target_labels: Labels identifying each target set T_i.
    :param weights: Non-negative weight w_i for each T_i.
    :returns: Transformed MDP with state reward model "weighted_reach".
    """
    m = len(target_labels)
    result = goal_unfolding(mdp, target_labels, return_state_bits=True)
    assert isinstance(result, tuple)
    unfolded, state_bits = result

    rw = unfolded.new_reward_model("weighted_reach")
    for s, choice in unfolded.transitions.items():
        bits = state_bits[s]
        for a, branch in choice:
            for _, s_next in branch:
                r = sum(
                    weights[i]
                    for i in range(m)
                    if target_labels[i] in s_next.labels and not bits[i]
                )
                if r:
                    rw.set_transition_reward(s, a, s_next, r)

    return eliminate_transition_rewards(unfolded)


def compute_weighted_reachability_policy(
    mdp: stormvogel.model.Model,
    target_labels: list[str],
    weights: list[float],
) -> stormvogel.result.Result:
    """Compute an optimal policy for weighted multi-target reachability.

    Reduces the problem to single-objective total reward maximization via
    :func:`weighted_multi_target_reachability`, then calls stormpy to solve
    ``R{"weighted_reach"}max=? [C]`` and extract the scheduler.

    :param mdp: The input MDP.
    :param target_labels: Labels identifying each target set T_i.
    :param weights: Non-negative weight w_i for each T_i.
    :returns: Model-checking result on the transformed MDP, including the
        optimal scheduler in ``result.scheduler``.
    :raises ImportError: If stormpy is not installed.
    :raises RuntimeError: If model checking returns no result.
    """
    try:
        from stormvogel.stormpy_utils.model_checking import model_checking
    except ImportError as exc:
        raise ImportError(
            "stormpy is required for policy computation; install stormvogel[storm]."
        ) from exc

    transformed = weighted_multi_target_reachability(mdp, target_labels, weights)
    transformed.add_self_loops()
    prop = 'R{"weighted_reach"}max=? [C]'
    result = model_checking(transformed, prop, scheduler=True)
    if result is None:
        raise RuntimeError("Model checking returned no result for the transformed MDP.")
    return result


def evaluate_policy_reachability(
    result: stormvogel.result.Result,
    target_labels: list[str],
) -> list[float]:
    """Evaluate a policy's reachability probability for each individual target.

    Induces a DTMC from the scheduler in *result* and runs one reachability
    query ``P=? [F "label"]`` per entry in *target_labels*.

    :param result: Output of :func:`compute_weighted_reachability_policy`;
        ``result.scheduler`` must be non-None.
    :param target_labels: Labels of the individual target sets T_i.
    :returns: List of length ``len(target_labels)`` where entry *i* is
        ``P_policy(reach T_i)`` at the initial state.
    :raises ImportError: If stormpy is not installed.
    :raises RuntimeError: If the induced DTMC cannot be constructed or model
        checking fails for any label.
    """
    try:
        from stormvogel.stormpy_utils.model_checking import model_checking
    except ImportError as exc:
        raise ImportError(
            "stormpy is required for policy evaluation; install stormvogel[storm]."
        ) from exc

    if result.scheduler is None:
        raise RuntimeError("result.scheduler is None; no policy to evaluate.")

    dtmc = result.scheduler.generate_induced_dtmc()
    if dtmc is None:
        raise RuntimeError("Could not generate induced DTMC from scheduler.")

    probs: list[float] = []
    for label in target_labels:
        mc = model_checking(dtmc, f'P=? [F "{label}"]', scheduler=False)
        if mc is None:
            raise RuntimeError(
                f"Model checking returned no result for label '{label}'."
            )
        val = mc.at(dtmc.initial_state)
        if not isinstance(val, (int, float)):
            raise RuntimeError(
                f"Model checking for label '{label}' returned a non-numeric value: {val!r}"
            )
        probs.append(float(val))
    return probs


def _compute_stuck_states(
    unfolded: stormvogel.model.Model,
    bits_map: dict,
    m: int,
) -> frozenset:
    """Return states from which no state with strictly more bits set is reachable.

    A state is *stuck* if the bit vector can never improve from it — either
    because all bits are already True, or because the state lies in a part of
    the model from which no new goal label can ever be reached.
    """
    # Build reverse graph for backward BFS.
    backward: dict = {s: set() for s in unfolded.states}
    for s in unfolded.states:
        for _action, branch in s.choices:
            for _prob, t in branch:
                backward[t].add(s)

    # Seed: states with a direct successor that has strictly more bits set.
    non_stuck: set = set()
    queue: list = []
    for s in unfolded.states:
        for _action, branch in s.choices:
            for _prob, t in branch:
                if any(bits_map[t][i] and not bits_map[s][i] for i in range(m)):
                    if s not in non_stuck:
                        non_stuck.add(s)
                        queue.append(s)
                    break

    # Propagate backwards: any predecessor of a non-stuck state is also non-stuck.
    head = 0
    while head < len(queue):
        s = queue[head]
        head += 1
        for pred in backward[s]:
            if pred not in non_stuck:
                non_stuck.add(pred)
                queue.append(pred)

    return frozenset(s for s in unfolded.states if s not in non_stuck)


def lp_dual_multireachprob(
    unfolded: stormvogel.model.Model,
    bits_map: dict,
    goal_labels: list[str],
    weights: list[float] | None = None,
    threshold: list[float] | None = None,
) -> "LP":
    """Occupancy-measure LP for multiobjective reachability on a goal unfolding.

    *unfolded* and *bits_map* must be the pair returned by
    ``goal_unfolding(mdp, goal_labels, return_state_bits=True)``.  Passing any
    other model raises :class:`ValueError`.

    Variables ``y_{s,a}`` represent expected visit counts for each free
    state-action pair.  A state is *free* if progress (a new goal bit flipping)
    is still reachable from it; all other states are treated as absorbing and
    receive no flow variable.

    The reach expression for goal *i* counts only transitions
    ``(s, a, t)`` where ``bits_map[s][i]`` is False and
    ``goal_labels[i] in t.labels``, ensuring each first-visit is counted
    exactly once.

    :param unfolded: Goal-unfolded MDP (output of :func:`goal_unfolding`).
    :param bits_map: Mapping from each state in *unfolded* to its bit tuple
        (output of :func:`goal_unfolding` with ``return_state_bits=True``).
    :param goal_labels: Labels identifying each goal, in the same order used
        when calling :func:`goal_unfolding`.
    :param weights: Objective weights λ_k; objective is zero if None.
    :param threshold: Optional lower-bound vector of length K. Adds
        ``P(reach goal_i) ≥ threshold[i]`` constraints.
    :raises ValueError: If *bits_map* is empty or its tuple length does not
        match *goal_labels*.
    """
    from stormvogel.teaching.lp import LP

    if not bits_map:
        raise ValueError(
            "bits_map is empty; pass the output of "
            "goal_unfolding(..., return_state_bits=True)"
        )
    m = len(goal_labels)
    sample = next(iter(bits_map.values()))
    if len(sample) != m:
        raise ValueError(
            f"bits_map tuple length {len(sample)} does not match "
            f"len(goal_labels)={m}; ensure unfolded and goal_labels "
            f"come from the same goal_unfolding call"
        )

    stuck = _compute_stuck_states(unfolded, bits_map, m)
    free_states = [s for s in unfolded.sorted_states if s not in stuck]
    free_set = frozenset(free_states)

    if weights is None:
        weights = [0.0] * m

    init = unfolded.initial_state

    y: dict = {
        (s, action): sp.Symbol(f"y_{s.friendly_name}_{action.label}")
        for s in free_states
        for action, _ in s.choices
    }

    def _reach_expr(i: int) -> sp.Expr:
        label = goal_labels[i]
        terms: list[sp.Expr] = []
        for s in free_states:
            if bits_map[s][i]:
                continue
            for action, branch in s.choices:
                reach = sum(prob for prob, t in branch if label in t.labels)
                if reach:
                    terms.append(sp.Float(reach) * y[s, action])
        return sp.Add(*terms) if terms else sp.Integer(0)

    obj_terms: list[sp.Expr] = [
        sp.Float(lam) * _reach_expr(i) for i, lam in enumerate(weights) if lam != 0
    ]
    objective = sp.Add(*obj_terms) if obj_terms else sp.Integer(0)

    inflow: dict = {s: sp.Integer(0) for s in free_states}
    for s_prime in free_states:
        for action, branch in s_prime.choices:
            for prob, t in branch:
                if t in free_set:
                    inflow[t] = inflow[t] + sp.Float(prob) * y[s_prime, action]

    flow_constraints: list[sp.Basic] = []
    for s in free_states:
        outflow = sp.Add(*[y[s, action] for action, _ in s.choices])
        rhs = sp.Integer(1) if s is init else sp.Integer(0)
        flow_constraints.append(sp.Eq(outflow - inflow[s], rhs))

    nonneg: list[sp.Basic] = [sp.Ge(v, sp.Integer(0)) for v in y.values()]

    threshold_constraints: list[sp.Basic] = []
    if threshold is not None:
        for i, thr in enumerate(threshold):
            threshold_constraints.append(sp.Ge(_reach_expr(i), sp.Float(thr)))

    return LP(
        sense="maximize",
        objective=objective,
        constraints=flow_constraints + nonneg + threshold_constraints,
    )
