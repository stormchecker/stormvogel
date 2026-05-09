"""Lovejoy grid MDP: finite upper-bounding MDP for POMDPs.

Constructs the Lovejoy grid MDP for POMDPs where every observation class has
at most two states.  For each such class the belief simplex is a line segment
``[0, 1]``, and a uniform grid of step ``1/k`` covers it exactly.  Off-grid
successor beliefs are replaced by a convex (linear) interpolation over the two
adjacent grid points, preserving the over-approximation guarantee.

The result is a stormvogel MDP whose maximal reachability probability is an
upper bound on the true POMDP value at the initial belief.
"""

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


def lovejoy_grid_mdp(
    pomdp: "Model",
    initial_belief: "Mapping[State, Fraction | int]",
    k: int,
) -> "Model":
    """Build the Lovejoy grid MDP for a POMDP with at most 2 states per observation.

    Every observation class must contain at most two POMDP states.  For each
    such class the belief simplex is 1-dimensional; a uniform grid of ``k + 1``
    points ``{0, 1/k, 2/k, …, 1}`` covers it.  Successor beliefs that fall
    between two grid points are replaced by their linear convex interpolation,
    which gives an over-approximation because ``V*`` is convex.

    The returned MDP has exactly the reachable grid beliefs as states.
    Its maximal reachability probability (under label *target*) is an upper
    bound on ``V*(initial_belief)``.

    :param pomdp: A POMDP where every observation class has at most 2 states.
    :param initial_belief: Distribution over POMDP states summing to 1.  Its
        support must lie within a single observation class and every probability
        must be an exact multiple of ``1/k``.
    :param k: Grid resolution.  Grid points for each 2-state observation class
        are at ``{0, 1/k, 2/k, …, 1}``.
    :returns: A stormvogel MDP whose optimal reachability value upper-bounds
        ``V*(initial_belief)``.
    :raises ValueError: If *pomdp* is not a POMDP, any observation class has
        more than 2 states, *initial_belief* does not sum to 1, its support
        spans more than one observation class, or any probability is not an
        exact multiple of ``1/k``.
    """
    from stormvogel.model.model import ModelType

    if pomdp.model_type != ModelType.POMDP:
        raise ValueError(f"lovejoy_grid_mdp requires a POMDP; got {pomdp.model_type}.")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")

    # --- Observation groups --------------------------------------------------

    obs_of: dict["State", object] = {
        s: pomdp.state_observations.get(s) for s in pomdp.states
    }
    obs_groups: dict[object, list["State"]] = {}
    for s in pomdp.states:
        obs_groups.setdefault(obs_of[s], []).append(s)

    for obs_key, states in obs_groups.items():
        if len(states) > 2:
            raise ValueError(
                f"Observation {obs_key!r} has {len(states)} states; "
                "lovejoy_grid_mdp only supports at most 2 states per observation."
            )
        states.sort(key=lambda s: s.state_id)  # canonical s_L / s_R ordering

    # --- Precompute POMDP transition and reward tables -----------------------

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
    propagate_labels: set[str] = set(pomdp.state_labels.keys()) - {"init"}

    # --- Validate initial belief ---------------------------------------------

    initial_b: dict["State", Fraction] = {
        s: Fraction(p) for s, p in initial_belief.items() if p != 0
    }
    total = sum(initial_b.values())
    if abs(total - 1) > Fraction(1, 10**9):
        raise ValueError(f"initial_belief must sum to 1; got {total}.")

    init_obs_keys = {obs_of[s] for s in initial_b}
    if len(init_obs_keys) != 1:
        raise ValueError(
            "initial_belief support must lie within a single observation class; "
            f"got observations {init_obs_keys!r}."
        )

    for s, p in initial_b.items():
        if (p * k).denominator != 1:
            raise ValueError(
                f"initial_belief[{s.friendly_name!r}] = {p} is not a multiple "
                f"of 1/{k}. Choose probabilities that are multiples of 1/{k}, "
                f"or use a different k."
            )

    init_node = Belief(initial_b)

    # --- Grid helpers --------------------------------------------------------

    def _make_grid_belief(s_L: "State", s_R: "State", j: int) -> Belief:
        dist: dict["State", Fraction] = {}
        if j > 0:
            dist[s_L] = Fraction(j, k)
        if j < k:
            dist[s_R] = Fraction(k - j, k)
        return Belief(dist)

    def _interpolate(
        s_L: "State", s_R: "State", p_L: Fraction
    ) -> list[tuple[Fraction, Belief]]:
        """Linear interpolation of {s_L: p_L, s_R: 1-p_L} onto the grid."""
        pk = p_L * k  # exact Fraction
        j = int(pk)  # floor (non-negative, so int() == floor)
        alpha = pk - j  # fractional overshoot; 0 means exactly on grid
        if alpha == 0:
            return [(Fraction(1), _make_grid_belief(s_L, s_R, j))]
        return [
            (1 - alpha, _make_grid_belief(s_L, s_R, j)),
            (alpha, _make_grid_belief(s_L, s_R, j + 1)),
        ]

    # --- Bird callbacks ------------------------------------------------------

    def available_actions(b: Belief) -> list[str]:
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

    def delta(b: Belief, action: str) -> list:
        unnorm: dict["State", Fraction] = {}
        for s, b_s in b.dist.items():
            for prob, tgt in trans.get(s, {}).get(action, []):
                unnorm[tgt] = unnorm.get(tgt, Fraction(0)) + b_s * prob

        by_obs: dict[object, dict["State", Fraction]] = {}
        for tgt, weight in unnorm.items():
            by_obs.setdefault(obs_of[tgt], {})[tgt] = weight

        result: list[tuple[Fraction, Belief]] = []
        for obs_key, group in by_obs.items():
            obs_prob = sum(group.values(), Fraction(0))
            if obs_prob == 0:
                continue
            group_states = obs_groups[obs_key]  # sorted by state_id
            if len(group_states) == 1:
                result.append((obs_prob, Belief({group_states[0]: Fraction(1)})))
            else:
                s_L, s_R = group_states
                p_L = group.get(s_L, Fraction(0)) / obs_prob
                for w, succ_b in _interpolate(s_L, s_R, p_L):
                    result.append((obs_prob * w, succ_b))

        return result

    def labels(b: Belief) -> list[str]:
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

    def rewards_fn(b: Belief) -> dict:
        return {
            name: sum(p * rewards_of[s][name] for s, p in b.dist.items())
            for name in reward_names
        }

    def friendly_names(b: Belief) -> str:
        if len(b.dist) == 1:
            s = next(iter(b.dist))
            return s.friendly_name or str(s.state_id)
        s_L, _ = sorted(b.dist, key=lambda s: s.state_id)
        obs_name = getattr(obs_of[s_L], "alias", repr(obs_of[s_L]))
        p_L = b.dist.get(s_L, Fraction(0))
        return f"b_{obs_name}[{p_L}]"

    # --- Assemble and run bird -----------------------------------------------

    kwargs: dict = dict(
        delta=delta,
        init=init_node,
        available_actions=available_actions,
        labels=labels,
        friendly_names=friendly_names,
        modeltype=ModelType.MDP,
        max_size=10_000_000,
    )
    if reward_names:
        kwargs["rewards"] = rewards_fn

    return _bird.build_bird(**kwargs)
