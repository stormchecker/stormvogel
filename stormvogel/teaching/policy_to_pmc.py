"""Transformation: resolve POMDP/MDP nondeterminism with a parametric policy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State


def _sanitize(name: str) -> str:
    """Return a valid sympy/Python identifier derived from *name*."""
    s = re.sub(r"[^a-zA-Z0-9]", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _fresh(desired: str, taken: set[str]) -> str:
    """Return *desired* if not in *taken*, else append a numeric suffix."""
    if desired not in taken:
        taken.add(desired)
        return desired
    i = 0
    while f"{desired}_{i}" in taken:
        i += 1
    name = f"{desired}_{i}"
    taken.add(name)
    return name


def policy_to_pmc(model: "Model") -> "Model":
    """Resolve POMDP/MDP nondeterminism with a parametric observation policy.

    For each observation *o* (POMDP) or state *s* (MDP) with *k ≥ 2* available
    actions *a₁ … aₖ*, fresh parameters ``y_{o,a}`` (or ``y_{s,a}``) are
    introduced, representing the probability of choosing action *a*.  The
    resulting pMC transition from state *s* to *s'* is:

    .. math::

        \\sum_{a} y_{\\text{key},a} \\cdot P(s, a, s')

    where *key* is the observation (POMDP) or state (MDP).  States with only
    one available action are copied without introducing a parameter.

    Existing model parameters (e.g. transition probabilities in a pMDP) are
    declared on the new model and preserved in the combined expressions.
    New policy parameter names are chosen to be disjoint from all pre-existing
    names in the model.

    .. note::
        The constraint :math:`\\sum_a y_{\\text{key},a} = 1` is not enforced
        in the model structure and must hold for any concrete policy.

    :param model: An MDP, pMDP, POMDP, or MA with deterministic state
        observations.
    :returns: A new parametric DTMC.
    :raises ValueError: If the model type is not supported, or if a POMDP
        state has a stochastic (distribution-valued) observation.
    """
    import sympy as sp

    from stormvogel.model.action import EmptyAction
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.model import ModelType, new_dtmc

    _SUPPORTED = (ModelType.MDP, ModelType.POMDP, ModelType.MA)
    if model.model_type not in _SUPPORTED:
        raise ValueError(
            f"policy_to_pmc requires an MDP, POMDP, or MA; got {model.model_type}."
        )

    is_pomdp = model.model_type == ModelType.POMDP

    # Validate POMDP observations upfront so we always raise, even for
    # single-action states whose _group_key would otherwise never be called.
    if is_pomdp:
        for state in model.states:
            obs = model.state_observations.get(state)
            if isinstance(obs, Distribution):
                raise ValueError(
                    f"State {state!r} has a stochastic observation; "
                    "policy_to_pmc requires deterministic state observations."
                )

    # --- build the new pMC ---------------------------------------------------

    pmc = new_dtmc(create_initial_state=False)

    # Copy existing parameters so combined expressions pass _validate_parametric_choices.
    pmc._parameters.update(model._parameters)

    # Names already taken: existing params + will-be-created policy params.
    taken: set[str] = set(model.parameters)

    # Map old state → new state.
    state_map: dict[State, State] = {}
    for old_state in model.states:
        new_state = pmc.new_state(
            labels=list(old_state.labels),
            valuations=dict(model.state_valuations[old_state]),
            friendly_name=model.friendly_names.get(old_state),
        )
        state_map[old_state] = new_state

    # --- helpers (use pmc and taken from enclosing scope) --------------------

    obs_index: dict[object, int] = {}  # stable index for un-aliased observations

    def _group_key(state: State) -> str:
        if is_pomdp:
            obs = model.state_observations.get(state)
            if obs is None:
                return f"s{model.states.index(state)}"
            if isinstance(obs, Distribution):
                raise ValueError(
                    f"State {state!r} has a stochastic observation; "
                    "policy_to_pmc requires deterministic state observations."
                )
            alias = model.observation_aliases.get(obs)
            if alias:
                return _sanitize(alias)
            if obs not in obs_index:
                obs_index[obs] = len(obs_index)
            return f"obs{obs_index[obs]}"
        else:
            fn = model.friendly_names.get(state)
            if fn:
                return _sanitize(fn)
            return f"s{model.states.index(state)}"

    def _action_key(action) -> str:
        if action == EmptyAction or action.label is None:
            return "eps"
        return _sanitize(action.label)

    param_cache: dict[tuple[str, str], sp.Symbol] = {}

    def _get_param(group_key: str, action) -> sp.Symbol:
        cache_key = (group_key, _action_key(action))
        if cache_key not in param_cache:
            name = _fresh(f"y_{group_key}_{_action_key(action)}", taken)
            sym = pmc.declare_parameter(name, positive=True)
            param_cache[cache_key] = sym  # type: ignore[assignment]
        return param_cache[cache_key]

    # --- build transitions ---------------------------------------------------

    for old_state, choices in model.transitions.items():
        new_state = state_map[old_state]
        action_list = choices.actions

        if len(action_list) <= 1:
            # Single (or no) action: copy the distribution directly.
            for _, branch in choices:
                pmc.set_choices(
                    new_state,
                    [(val, state_map[tgt]) for val, tgt in branch],
                )
            continue

        # Multiple actions: build combined parametric distribution.
        gk = _group_key(old_state)
        combined: dict[State, sp.Expr] = {}
        for action, branch in choices:
            y = _get_param(gk, action)
            for val, tgt in branch:
                new_tgt = state_map[tgt]
                term: sp.Expr = y * sp.sympify(val)
                combined[new_tgt] = combined.get(new_tgt, sp.Integer(0)) + term

        pmc.set_choices(
            new_state,
            [(sp.cancel(expr), tgt) for tgt, expr in combined.items()],
        )

    # --- copy state rewards --------------------------------------------------

    for rm in model.rewards:
        new_rm = pmc.new_reward_model(rm.name)
        for old_s, value in rm.rewards.items():
            new_rm.rewards[state_map[old_s]] = value

    return pmc
