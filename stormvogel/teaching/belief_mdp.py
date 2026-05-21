"""Bounded belief-MDP exploration for POMDPs."""

from __future__ import annotations

import warnings
from fractions import Fraction
from collections.abc import Mapping
from typing import TYPE_CHECKING

import stormvogel.bird as _bird
from stormvogel.teaching.belief import Belief

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State


class FrontierBelief(Belief):
    """A belief that was cut off at the exploration boundary.

    In the belief MDP a frontier belief has a single ``"cut"`` action.
    The probability of reaching the target is the dot product
    ``Σ_s c(s) · b(s)`` of the per-state cutoff function *c* with the
    frontier belief; the remainder goes to the fresh absorbing sink.
    Frontier beliefs receive the label ``"frontier"``.
    """

    def __hash__(self) -> int:
        # Must differ from Belief.__hash__ so a normal and a frontier belief
        # with the same distribution are distinct nodes.
        return hash(("frontier", self._key))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrontierBelief):
            return self._key == other._key
        if isinstance(other, Belief):
            # TODO this is a hack and violates symmetry of eq.
            return False  # FrontierBelief is never equal to a plain Belief
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrontierBelief({self.dist!r})"


class _Terminal:
    """Shared absorbing terminal state (``"target"`` or ``"sink"``)."""

    def __init__(self, label: str) -> None:
        self._label = label

    def __repr__(self) -> str:
        return self._label


_TARGET = _Terminal("target")
_SINK = _Terminal("sink")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def belief_mdp(
    pomdp: "Model",
    initial_belief: "Mapping[State, Fraction | int]",
    cutoff: "Mapping[State, Fraction | float] | Fraction | int | float" = Fraction(0),
    max_states: int = 1000,
) -> "Model":
    """Explore the belief MDP of a POMDP up to *max_states* distinct beliefs.

    Starting from *initial_belief*, successor beliefs are computed by the
    standard Bayesian belief update and explored BFS-style via the bird API.
    Once *max_states* distinct :class:`Belief` nodes have been committed,
    any new successor belief becomes a :class:`FrontierBelief` instead.

    **Frontier transitions**: each :class:`FrontierBelief` has a single
    ``"cut"`` action.  The probability of reaching the shared target state is
    the dot product :math:`c \\cdot b_f = \\sum_s c(s)\\, b_f(s)` of the
    cutoff function with the frontier belief; the remainder goes to the shared
    sink.  Both terminal states are absorbing.

    *cutoff* may be:

    - A scalar in ``[0, 1]``: uniform per-state value.  ``0`` (default) gives
      a pessimistic lower bound; ``1`` gives an optimistic upper bound.
    - A ``Mapping[State, Fraction | float]``: per-state cutoff function
      :math:`c \\colon S \\to [0,1]`.  Absent states default to 0.
      Passing the MDP value function (e.g. from
      :func:`~stormvogel.teaching.pomdp_backup.mdp_bound_alpha`) yields the
      MDP-value upper bound.

    **Rewards**: propagated as expected POMDP reward ``Σ_s b[s] · r(s)``.
    Frontier, target, and sink states carry reward 0 in every reward model.

    **Warnings**: emitted when belief-support states disagree on their
    available actions or on the presence of a label.

    :param pomdp: A POMDP with deterministic (non-stochastic) state
        observations.
    :param initial_belief: Mapping from POMDP states to non-negative
        probabilities summing to 1.
    :param cutoff: Per-state cutoff function :math:`c\\colon S\\to[0,1]`, or
        a scalar applied uniformly.  Defaults to ``0`` (pessimistic).
    :param max_states: Maximum number of distinct :class:`Belief` nodes to
        expand fully (including the initial belief).
    :returns: The explored belief MDP as a stormvogel MDP model.
    :raises ValueError: If *pomdp* is not a POMDP, any state has a stochastic
        observation, *initial_belief* does not sum to 1, or a scalar *cutoff*
        is outside [0, 1].
    """
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.model import ModelType

    if pomdp.model_type != ModelType.POMDP:
        raise ValueError(f"belief_mdp requires a POMDP; got {pomdp.model_type}.")

    # Normalise cutoff to a per-state dict or a scalar Fraction.
    if isinstance(cutoff, Mapping):
        cutoff_map: dict["State", Fraction] = {
            s: Fraction(v) for s, v in cutoff.items()
        }
        _scalar_cutoff: Fraction | None = None
    else:
        _scalar_cutoff = Fraction(cutoff)
        cutoff_map = {}
        if not (Fraction(0) <= _scalar_cutoff <= Fraction(1)):
            raise ValueError(f"cutoff must be in [0, 1]; got {cutoff!r}.")

    # --- Validate initial belief ----------------------------------------------

    initial_b: dict["State", Fraction] = {
        s: Fraction(p) for s, p in initial_belief.items() if p != 0
    }
    total = sum(initial_b.values())
    if abs(total - 1) > Fraction(1, 10**9):
        raise ValueError(f"initial_belief must sum to 1; got {total}.")

    # --- Precompute POMDP structure ------------------------------------------

    # Validate: all state observations must be deterministic.
    for state in pomdp.states:
        if isinstance(pomdp.state_observations.get(state), Distribution):
            raise ValueError(
                f"State {state!r} has a stochastic observation; "
                "belief_mdp requires deterministic state observations."
            )

    # obs_of[state]: the Observation object for each POMDP state (may be None).
    obs_of: dict["State", object] = {
        s: pomdp.state_observations.get(s) for s in pomdp.states
    }

    # trans[state][action_label] = [(Fraction prob, target_state), ...]
    trans: dict["State", dict[str, list[tuple[Fraction, "State"]]]] = {}
    for state, choices in pomdp.transitions.items():
        trans[state] = {}
        for action, branch in choices:
            al = action.label if action.label is not None else ""
            trans[state][al] = [(Fraction(val), tgt) for val, tgt in branch]

    reward_names: list[str] = [rm.name for rm in pomdp.rewards]
    rewards_of: dict["State", dict[str, Fraction | int | float]] = {
        s: {rm.name: rm.rewards.get(s, 0) for rm in pomdp.rewards} for s in pomdp.states
    }
    # POMDP state labels to propagate (bird adds "init" automatically)
    propagate_labels: set[str] = set(pomdp.state_labels.keys()) - {"init"}

    # --- Exploration budget --------------------------------------------------

    # Beliefs already committed to full expansion.
    seen: set[Belief] = {Belief(initial_b)}

    # --- Belief update -------------------------------------------------------

    def _update(belief: Belief, action_label: str) -> list[tuple[Fraction, Belief]]:
        """Return [(Pr(obs|b,a), updated Belief)] for each reachable observation."""
        # Unnormalised weight of each reachable successor state.
        unnorm: dict["State", Fraction] = {}
        for s, b_s in belief.dist.items():
            for prob, tgt in trans.get(s, {}).get(action_label, []):
                unnorm[tgt] = unnorm.get(tgt, Fraction(0)) + b_s * prob

        # Group successor states by their observation.
        obs_groups: dict[object, dict["State", Fraction]] = {}
        for tgt, weight in unnorm.items():
            grp = obs_groups.setdefault(obs_of[tgt], {})
            grp[tgt] = grp.get(tgt, Fraction(0)) + weight

        # Normalise each group into a Belief.
        result: list[tuple[Fraction, Belief]] = []
        for grp in obs_groups.values():
            obs_prob = sum(grp.values(), Fraction(0))
            if obs_prob > 0:
                result.append((obs_prob, Belief.normalize(grp)))
        return result

    # --- Bird callbacks ------------------------------------------------------

    def available_actions(b: object) -> list[str]:
        if b is _TARGET or b is _SINK:
            return ["absorb"]
        if isinstance(b, FrontierBelief):
            return ["cut"]
        assert isinstance(b, Belief)
        support = list(b.dist)
        action_sets = [set(trans.get(s, {}).keys()) for s in support]
        common = action_sets[0].intersection(*action_sets[1:]) if action_sets else set()
        for s, aset in zip(support, action_sets):
            extra = aset - common
            if extra:
                warnings.warn(
                    f"Support state {s!r} has extra actions {sorted(extra)} "
                    "not shared by all belief-support states; using intersection.",
                    UserWarning,
                    stacklevel=2,
                )
        return sorted(common)

    def _cut_prob(b: FrontierBelief) -> Fraction:
        """Probability of reaching target from frontier belief b under cutoff c."""
        if _scalar_cutoff is not None:
            return _scalar_cutoff
        return sum(
            (cutoff_map.get(s, Fraction(0)) * p for s, p in b.dist.items()),
            Fraction(0),
        )

    def delta(b: object, action: str) -> list:
        if b is _TARGET:
            return [(1, _TARGET)]
        if b is _SINK:
            return [(1, _SINK)]
        if isinstance(b, FrontierBelief):
            cp = _cut_prob(b)
            if cp == 0:
                return [(1, _SINK)]
            if cp == 1:
                return [(1, _TARGET)]
            return [(cp, _TARGET), (1 - cp, _SINK)]
        assert isinstance(b, Belief)
        result = []
        for obs_prob, successor in _update(b, action):
            if successor in seen:
                result.append((obs_prob, successor))
            elif len(seen) < max_states:
                seen.add(successor)
                result.append((obs_prob, successor))
            else:
                result.append((obs_prob, FrontierBelief(successor.dist)))
        return result

    def labels(b: object) -> list[str]:
        if b is _TARGET:
            return ["target"]
        if b is _SINK:
            return ["sink"]
        if isinstance(b, FrontierBelief):
            return ["frontier"]
        assert isinstance(b, Belief)
        support = {s for s, p in b.dist.items() if p > 0}
        result = []
        for lbl in propagate_labels:
            lbl_states = pomdp.state_labels.get(lbl, set())
            n_with = sum(1 for s in support if s in lbl_states)
            if n_with == len(support):
                result.append(lbl)
            elif n_with > 0:
                warnings.warn(
                    f"Label '{lbl}' is present in {n_with}/{len(support)} "
                    "belief-support states; not propagating to belief state.",
                    UserWarning,
                    stacklevel=2,
                )
        return result

    def rewards_fn(b: object) -> dict:
        if b is _TARGET or b is _SINK or isinstance(b, FrontierBelief):
            return {name: 0 for name in reward_names}
        assert isinstance(b, Belief)
        return {
            name: sum(p * rewards_of[s][name] for s, p in b.dist.items())
            for name in reward_names
        }

    def friendly_names(b: object) -> str:
        if b is _TARGET or b is _SINK:
            return repr(b)
        assert isinstance(b, (Belief, FrontierBelief))
        idx = {s: i for i, s in enumerate(pomdp.states)}
        parts = [
            f"s{idx[s]}:{p if max(abs(p.numerator), p.denominator) < 10000 else f'{float(p):.3f}'}"
            for s, p in sorted(b.dist.items(), key=lambda kv: idx[kv[0]])
        ]
        prefix = "frontier" if isinstance(b, FrontierBelief) else "b"
        return f"{prefix}{{{', '.join(parts)}}}"

    # --- Assemble and run bird -----------------------------------------------

    kwargs: dict = dict(
        delta=delta,
        init=Belief(initial_b),
        available_actions=available_actions,
        labels=labels,
        friendly_names=friendly_names,
        modeltype=ModelType.MDP,
        max_size=10_000_000,  # we control exploration via `seen`
    )
    if reward_names:
        kwargs["rewards"] = rewards_fn

    return _bird.build_bird(**kwargs)
