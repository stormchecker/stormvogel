"""Teaching module: POMDP belief computations.

Provides exact Bayesian belief tracking for POMDPs with deterministic state
observations.  All arithmetic uses :class:`~fractions.Fraction`.

Typical usage::

    from stormvogel.teaching.pomdp import initial_belief, belief_trace

    b0 = initial_belief(pomdp, "z")
    beliefs = belief_trace(pomdp, b0, [("b", "z"), ("a", "z_target")])
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State

#: A belief is an exact rational probability distribution over states.
#: Absent states have implicit probability zero.
Belief = dict["State", Fraction]


def initial_belief(pomdp: "Model", obs_alias: str) -> Belief:
    """Derive the initial belief from the POMDP's initial-state distribution.

    The initial state is assumed to have a single EmptyAction transition that
    encodes the prior distribution over states.  The resulting distribution is
    filtered to states whose observation matches *obs_alias* and then
    normalised.

    :param pomdp: The POMDP model.
    :param obs_alias: Observation alias that filters which successor states
        belong to the initial belief support.
    :returns: Normalised belief over states with observation *obs_alias*.
    :raises ValueError: If no states with *obs_alias* are reachable from the
        initial state, or if the initial state has no EmptyAction transition.
    """
    from stormvogel.model.action import EmptyAction

    init = pomdp.initial_state
    obs_states = pomdp.compute_states_per_observation()[
        pomdp.get_observation(obs_alias)
    ]

    unnorm: dict["State", Fraction] = {}
    for action, branch in pomdp.transitions[init]:
        if action is not EmptyAction:
            continue
        for prob, tgt in branch:
            if tgt in obs_states:
                unnorm[tgt] = unnorm.get(tgt, Fraction(0)) + Fraction(prob)

    total = sum(unnorm.values(), Fraction(0))
    if total == 0:
        raise ValueError(
            f"No states with observation '{obs_alias}' are reachable from the "
            f"initial state under the EmptyAction transition."
        )
    return {s: v / total for s, v in unnorm.items()}


def belief_update(
    pomdp: "Model",
    belief: Belief,
    action_label: str,
    obs_alias: str,
) -> Belief:
    """Compute the updated belief after taking an action and receiving an observation.

    Applies the standard Bayesian filter for POMDPs with deterministic
    observations::

        b'(s') ∝  Σ_s  P(s' | s, a) · b(s)   if obs(s') = o
                  0                             otherwise

    :param pomdp: The POMDP model.
    :param belief: Current belief distribution over states.
    :param action_label: Label of the action taken.
    :param obs_alias: Alias of the observation received after the action.
    :returns: Updated, normalised belief.
    :raises ValueError: If the observation is unreachable from the current
        belief under the given action.
    """
    from stormvogel.model.action import EmptyAction

    obs_states = pomdp.compute_states_per_observation()[
        pomdp.get_observation(obs_alias)
    ]

    unnorm: dict["State", Fraction] = {}
    for state, choices in pomdp.transitions.items():
        b_s = belief.get(state, Fraction(0))
        if b_s == 0:
            continue
        for action, branch in choices:
            lbl = action.label if action is not EmptyAction else None
            if lbl != action_label:
                continue
            for prob, tgt in branch:
                if tgt in obs_states:
                    unnorm[tgt] = unnorm.get(tgt, Fraction(0)) + Fraction(prob) * b_s

    total = sum(unnorm.values(), Fraction(0))
    if total == 0:
        raise ValueError(
            f"Belief update failed: observation '{obs_alias}' is unreachable "
            f"from the current belief under action '{action_label}'."
        )
    return {s: v / total for s, v in unnorm.items()}


def belief_trace(
    pomdp: "Model",
    b0: Belief,
    trace: list[tuple[str, str]],
) -> list[Belief]:
    """Compute the sequence of beliefs induced by an observation trace.

    :param pomdp: The POMDP model.
    :param b0: Initial belief distribution.
    :param trace: Sequence of ``(action_label, obs_alias)`` pairs.
    :returns: List of beliefs of length ``len(trace) + 1``: the initial belief
        followed by one updated belief per step.
    """
    beliefs: list[Belief] = [b0]
    current = b0
    for action_label, obs_alias in trace:
        current = belief_update(pomdp, current, action_label, obs_alias)
        beliefs.append(current)
    return beliefs
