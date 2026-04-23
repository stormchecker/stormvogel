"""Exact reachability evaluation for DTMCs via sympy linear system solving.

Intended for teaching: results are exact rationals, and the intermediate
linear system can be displayed in a notebook.
"""

from typing import Iterable, cast

import sympy as sp

import stormvogel.model as model


def _check_dtmc(dtmc: model.Model) -> None:
    if dtmc.model_type != model.ModelType.DTMC:
        raise ValueError(f"Expected a DTMC, got {dtmc.model_type}.")


def _state_variables(dtmc: model.Model) -> dict[model.State, sp.Symbol]:
    return {s: sp.Symbol(f"x_{s.state_id.hex}") for s in dtmc.states}


def _predecessors(dtmc: model.Model) -> dict[model.State, set[model.State]]:
    pred: dict[model.State, set[model.State]] = {s: set() for s in dtmc.states}
    for s in dtmc.states:
        _, branch = next(iter(s.choices))
        for _prob, t in branch:
            pred[t].add(s)
    return pred


def compute_zero_states(
    dtmc: model.Model,
    one_states: Iterable[model.State],
) -> set[model.State]:
    """Return states from which no *one_state* is reachable in *dtmc*.

    Backward BFS from *one_states*; any state not reached has reachability
    probability 0.  The *one_states* may belong to a different model — they
    are matched against *dtmc* by ``state_id``.

    :param dtmc: A stormvogel DTMC.
    :param one_states: States with reachability probability 1.
    :returns: Set of states in *dtmc* with reachability probability 0.
    """
    _check_dtmc(dtmc)
    one_ids = {s.state_id for s in one_states}
    pred = _predecessors(dtmc)

    can_reach: set[model.State] = set()
    queue = [s for s in dtmc.states if s.state_id in one_ids]
    can_reach.update(queue)
    i = 0
    while i < len(queue):
        for p in pred[queue[i]]:
            if p not in can_reach:
                can_reach.add(p)
                queue.append(p)
        i += 1

    return {s for s in dtmc.states if s not in can_reach}


def compute_one_states(
    dtmc: model.Model,
    target_states: Iterable[model.State],
) -> set[model.State]:
    """Return states that reach *target_states* with probability 1 in *dtmc*.

    Uses the Prob1 algorithm:

    1. Compute ``Sno`` via :func:`compute_zero_states`.
    2. Backward BFS from ``Sno``, treating *target_states* as absorbing
       (paths do not continue through them).  The result is the set of states
       that can reach ``Sno`` with positive probability — i.e. states with
       reachability strictly less than 1.
    3. ``Syes`` = all states \\ states that can reach ``Sno``.

    :param dtmc: A stormvogel DTMC.
    :param target_states: States to reach.
    :returns: Set of states with reachability probability exactly 1.
    """
    _check_dtmc(dtmc)
    target_set: set[model.State] = set(target_states)
    sno = compute_zero_states(dtmc, target_set)

    pred = _predecessors(dtmc)

    # Backward BFS from Sno; do not cross target states.
    can_reach_sno: set[model.State] = set(sno)
    queue = list(sno)
    i = 0
    while i < len(queue):
        for p in pred[queue[i]]:
            if p not in can_reach_sno and p not in target_set:
                can_reach_sno.add(p)
                queue.append(p)
        i += 1

    return {s for s in dtmc.states if s not in can_reach_sno}


def equations_reachability(
    dtmc: model.Model,
    one_states: Iterable[model.State],
    zero_states: Iterable[model.State] | None = None,
) -> list[sp.Expr]:
    """Return the sympy residuals for the reachability linear system in a DTMC.

    Each residual equals zero when the system is satisfied.  States in
    *one_states* contribute ``x_s - 1``; states in *zero_states* contribute
    ``x_s``; all other states contribute ``x_s - Σ P(s,s') x_{s'}``.

    :param dtmc: A stormvogel DTMC.
    :param one_states: States with reachability probability 1.
    :param zero_states: States with reachability probability 0, or ``None``
        to auto-detect from the graph structure.
    :returns: List of sympy expressions (residuals), one per state in
        ``dtmc.sorted_states``.
    """
    _check_dtmc(dtmc)
    one_ids = {s.state_id for s in one_states}
    if zero_states is not None:
        zero_ids = {s.state_id for s in zero_states}
    else:
        # Auto-detect: match one_states to dtmc by UUID, then BFS.
        dtmc_one = [s for s in dtmc.states if s.state_id in one_ids]
        zero_ids = {s.state_id for s in compute_zero_states(dtmc, dtmc_one)}

    x = _state_variables(dtmc)

    residuals: list[sp.Expr] = []
    for s in dtmc.sorted_states:
        if s.state_id in one_ids:
            residuals.append(x[s] - sp.Integer(1))
        elif s.state_id in zero_ids:
            residuals.append(x[s])
        else:
            _, branch = next(iter(s.choices))
            rhs: sp.Expr = sum(  # type: ignore[assignment]
                sp.nsimplify(prob) * x[s_next] for prob, s_next in branch
            )
            residuals.append(x[s] - rhs)
    return residuals


def equations_expected_reward(
    dtmc: model.Model,
    reward_model: model.RewardModel,
    terminal_states: Iterable[model.State],
    discount: sp.Expr = sp.Integer(1),
) -> list[sp.Expr]:
    """Return the sympy residuals for the expected-reward linear system in a DTMC.

    Each residual equals zero when the system is satisfied.  Terminal states
    contribute ``x_s``; non-terminal states contribute
    ``x_s - r_s - γ Σ P(s,s') x_{s'}``.

    For *undiscounted* problems (*discount* = 1) the system is ill-defined when
    a non-terminal state cannot reach any terminal state, because the expected
    reward would be infinite.  A :exc:`ValueError` is raised in that case
    rather than producing a wrong answer.

    For *discounted* problems (*discount* < 1) the system always has a unique
    solution and no reachability check is needed.

    :param dtmc: A stormvogel DTMC.
    :param reward_model: Per-state rewards; missing states are treated as 0.
    :param terminal_states: States fixed to value 0 (reward collection stops).
    :param discount: Discount factor γ ∈ (0, 1].  Must be a sympy expression
        or a value that :func:`sympy.nsimplify` can convert.
    :raises ValueError: If *discount* > 1, or if *discount* = 1 and any
        non-terminal state cannot reach a terminal state.
    """
    _check_dtmc(dtmc)
    discount_sym = sp.nsimplify(discount)
    if discount_sym > sp.Integer(1):
        raise ValueError(f"Discount factor must be ≤ 1, got {discount}.")

    terminal_ids = {s.state_id for s in terminal_states}

    if discount_sym == sp.Integer(1):
        dtmc_terminals = [s for s in dtmc.states if s.state_id in terminal_ids]
        unreachable = {
            s
            for s in compute_zero_states(dtmc, dtmc_terminals)
            if s.state_id not in terminal_ids
        }
        if unreachable:
            names = sorted(s.friendly_name or str(s.state_id) for s in unreachable)
            raise ValueError(
                f"Undiscounted expected reward is undefined for states that cannot "
                f"reach any terminal state: {names}"
            )

    x = _state_variables(dtmc)
    residuals: list[sp.Expr] = []
    for s in dtmc.sorted_states:
        if s.state_id in terminal_ids:
            residuals.append(x[s])
        else:
            _, branch = next(iter(s.choices))
            reward = sp.nsimplify(reward_model.get_state_reward(s) or 0)
            successor_sum: sp.Expr = sum(  # type: ignore[assignment]
                sp.nsimplify(prob) * x[s_next] for prob, s_next in branch
            )
            residuals.append(x[s] - reward - discount_sym * successor_sum)
    return residuals


def solve_expected_reward(
    dtmc: model.Model,
    reward_model: model.RewardModel,
    terminal_states: Iterable[model.State],
    discount: sp.Expr = sp.Integer(1),
) -> dict[model.State, sp.Expr]:
    """Solve the expected reward system exactly and return per-state values.

    :param dtmc: A stormvogel DTMC.
    :param reward_model: Per-state rewards; missing states are treated as 0.
    :param terminal_states: States fixed to value 0.
    :param discount: Discount factor γ ∈ (0, 1].
    :returns: Mapping from state to exact sympy expected reward.
    :raises ValueError: See :func:`equations_expected_reward`.
    """
    _check_dtmc(dtmc)
    x = _state_variables(dtmc)
    symbols = [x[s] for s in dtmc.sorted_states]

    residuals = equations_expected_reward(dtmc, reward_model, terminal_states, discount)
    solution = sp.linsolve(residuals, symbols)
    if not solution or solution == sp.EmptySet:
        raise ValueError("Expected reward system has no unique solution.")
    values_tuple = cast(tuple[sp.Expr, ...], next(iter(solution)))
    return {s: v for s, v in zip(dtmc.sorted_states, values_tuple)}


def solve_reachability(
    dtmc: model.Model,
    one_states: Iterable[model.State],
    zero_states: Iterable[model.State] | None = None,
) -> dict[model.State, sp.Expr]:
    """Solve the reachability linear system exactly and return per-state values.

    :param dtmc: A stormvogel DTMC.
    :param one_states: States with reachability probability 1.
    :param zero_states: States with reachability probability 0, or ``None``
        to auto-detect via :func:`compute_zero_states`.
    :returns: Mapping from state to exact rational reachability probability.
    :raises ValueError: If the system has no unique solution.
    """
    _check_dtmc(dtmc)
    x = _state_variables(dtmc)
    symbols = [x[s] for s in dtmc.sorted_states]

    residuals = equations_reachability(dtmc, one_states, zero_states)
    solution = sp.linsolve(residuals, symbols)
    if not solution or solution == sp.EmptySet:
        raise ValueError("Reachability system has no unique solution.")
    values_tuple = cast(tuple[sp.Expr, ...], next(iter(solution)))
    return {s: v for s, v in zip(dtmc.sorted_states, values_tuple)}
