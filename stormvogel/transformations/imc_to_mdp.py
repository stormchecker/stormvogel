"""Transformation from interval Markov chains to MDPs."""

from __future__ import annotations

import itertools
import warnings
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import stormvogel.model as _model


def _vertices(
    lows: list[Fraction],
    highs: list[Fraction],
) -> list[tuple[Fraction, ...]]:
    """Enumerate vertices of the interval distribution polytope.

    A vertex has at most one index k strictly between its bounds; the remaining
    n-1 indices are fixed to either their lower or upper bound. For each choice
    of k and each 2^(n-1) assignment of the remaining indices, compute
    p_k = 1 − Σ(fixed) and accept if lows[k] ≤ p_k ≤ highs[k].

    :param lows: Lower bounds for each successor probability.
    :param highs: Upper bounds for each successor probability.
    :returns: Deduplicated list of feasible vertex tuples.
    """
    n = len(lows)
    seen: set[tuple[Fraction, ...]] = set()
    result: list[tuple[Fraction, ...]] = []

    for k in range(n):
        others = [j for j in range(n) if j != k]
        for bits in itertools.product((0, 1), repeat=n - 1):
            fixed = [lows[j] if b == 0 else highs[j] for j, b in zip(others, bits)]
            p_k = Fraction(1) - sum(fixed)
            if lows[k] <= p_k <= highs[k]:
                probs: list[Fraction] = list(fixed)
                probs.insert(k, p_k)
                t = tuple(probs)
                if t not in seen:
                    seen.add(t)
                    result.append(t)

    return result


def imc_to_mdp(model: "_model.Model") -> "_model.Model":
    """Convert an interval Markov chain to an MDP.

    Each state's interval distribution defines a polytope of feasible
    distributions. The vertices of that polytope become the actions of the
    corresponding MDP state. Labels, state valuations, friendly names, and
    state rewards are preserved. Transition rewards are not transferred.

    :param model: An interval MC (DTMC with Interval-valued transitions).
    :returns: A new MDP with one action per vertex per state.
    :raises ValueError: If *model* is not an interval model.
    """
    from stormvogel.model.action import Action
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.model import new_mdp
    from stormvogel.model.state import State

    if not model.is_interval_model():
        raise ValueError("imc_to_mdp requires an interval model.")

    mdp = new_mdp(create_initial_state=False)

    state_map: dict[State, State] = {}
    for old_state in model.states:
        new_state = mdp.new_state(
            labels=list(old_state.labels),
            valuations=dict(model.state_valuations[old_state]),
            friendly_name=model.friendly_names.get(old_state),
        )
        state_map[old_state] = new_state

    for old_state, choices in model.transitions.items():
        new_state = state_map[old_state]
        for _, branch in choices:
            entries = list(branch)  # [(Interval, target_state), ...]
            n = len(entries)

            if n > 8:
                warnings.warn(
                    f"State {old_state!r} has fan-out {n} > 8; "
                    "vertex enumeration may be slow.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            lows = [Fraction(e[0].lower) for e in entries]
            highs = [Fraction(e[0].upper) for e in entries]
            old_targets = [e[1] for e in entries]

            for i, probs in enumerate(_vertices(lows, highs)):
                action = Action(f"v{i}")
                distr: dict[State, Fraction] = {
                    state_map[old_targets[j]]: probs[j]
                    for j in range(n)
                    if probs[j] != 0
                }
                mdp.transitions[new_state][action] = Distribution(distr)

    for rm in model.rewards:
        new_rm = mdp.new_reward_model(rm.name)
        for old_s, value in rm.rewards.items():
            new_rm.rewards[state_map[old_s]] = value

    return mdp
