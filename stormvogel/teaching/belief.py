"""Teaching module: POMDP belief type and belief computations.

Provides the canonical :class:`Belief` type and exact Bayesian belief
tracking for POMDPs with deterministic state observations.  All arithmetic
uses :class:`~fractions.Fraction`.

Typical usage::

    from stormvogel.teaching.belief import Belief, initial_belief, belief_trace

    b0 = initial_belief(pomdp, "z")
    beliefs = belief_trace(pomdp, b0, [("b", "z"), ("a", "z_target")])
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State


class Belief(Mapping["State", Fraction]):
    """Exact probability distribution over POMDP states.

    Implements the :class:`~collections.abc.Mapping` interface over
    ``State → Fraction``, so ``b[s]``, ``b.get(s, 0)``, ``b.items()``,
    etc. all work directly.  Zero-probability states are silently dropped.

    :param dist: Mapping from POMDP states to their belief probabilities.
    """

    def __init__(self, dist: "dict[State, Fraction]") -> None:
        self.dist: dict["State", Fraction] = {s: p for s, p in dist.items() if p > 0}
        self._key: tuple[tuple[UUID, Fraction], ...] = tuple(
            sorted(((s.state_id, p) for s, p in self.dist.items()), key=lambda x: x[0])
        )

    # --- Mapping interface ---------------------------------------------------

    def __getitem__(self, key: "State") -> Fraction:
        return self.dist[key]

    def __iter__(self) -> Iterator["State"]:
        return iter(self.dist)

    def __len__(self) -> int:
        return len(self.dist)

    # --- Identity ------------------------------------------------------------

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Belief):
            return self._key == other._key
        return NotImplemented

    def __repr__(self) -> str:
        return f"Belief({self.dist!r})"

    @classmethod
    def normalize(cls, unnorm: "dict[State, Fraction]") -> "Belief":
        """Normalize *unnorm* to a probability distribution and return a Belief.

        :param unnorm: Unnormalized weights (non-negative, at least one > 0).
        :raises ValueError: If all weights are zero.
        """
        total = sum(unnorm.values(), Fraction(0))
        if total == 0:
            raise ValueError("Cannot normalize a zero-weight distribution.")
        return cls({s: v / total for s, v in unnorm.items()})


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

    unnorm: defaultdict["State", Fraction] = defaultdict(Fraction)
    for action, branch in pomdp.transitions[init]:
        if action is not EmptyAction:
            continue
        for prob, tgt in branch:
            if tgt in obs_states:
                unnorm[tgt] += Fraction(prob)

    if not unnorm:
        raise ValueError(
            f"No states with observation '{obs_alias}' are reachable from the "
            f"initial state under the EmptyAction transition."
        )
    return Belief.normalize(unnorm)


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

    unnorm: defaultdict["State", Fraction] = defaultdict(Fraction)
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
                    unnorm[tgt] += Fraction(prob) * b_s

    if not unnorm:
        raise ValueError(
            f"Belief update failed: observation '{obs_alias}' is unreachable "
            f"from the current belief under action '{action_label}'."
        )
    return Belief.normalize(unnorm)


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
