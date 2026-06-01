"""Semantic fingerprinting for stormvogel models.

Computes a stable, content-based SHA-256 hash over the semantic content of a
model.  Two models with identical structure, labels, probabilities, rewards, and
observations will produce the same hash regardless of how they were constructed
(state UUIDs are not included).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State
    from stormvogel.model.action import Action


def _action_sort_key(action: "Action") -> str:
    return action.label if action.label is not None else ""


def _canonical_value(v) -> str:
    """Stable string representation of a probability/reward value."""
    from fractions import Fraction
    from stormvogel.model.value import Interval

    if isinstance(v, Interval):
        return f"Interval({_canonical_value(v.lower)},{_canonical_value(v.upper)})"
    if isinstance(v, Fraction):
        return f"{v.numerator}/{v.denominator}"
    try:
        from stormvogel import parametric

        if parametric.is_parametric(v):
            return str(v)
    except Exception:
        pass
    return repr(v)


def _canonical_obs(model: "Model", state: "State") -> str:
    """Return a stable string for the observation of a state."""
    from stormvogel.model.distribution import Distribution

    obs = model.state_observations.get(state)
    if obs is None:
        return "obs:none"
    if isinstance(obs, Distribution):
        parts = sorted(
            (model.observation_aliases[o], _canonical_value(p)) for p, o in obs
        )
        return "obs:dist:" + ",".join(f"{a}={p}" for a, p in parts)
    return f"obs:{model.observation_aliases[obs]}"


def semantic_hash(model: "Model") -> str:
    """Return a hex SHA-256 digest of the model's semantic content.

    The hash covers model type, state labels, friendly names, state valuations,
    observations, transitions (actions + distributions), reward models, and
    markovian states.  State UUIDs are excluded; states are identified by their
    position in ``model.states``.

    :param model: The model to fingerprint.
    :returns: A 64-character lowercase hex string.
    """
    state_index: dict["State", int] = {s: i for i, s in enumerate(model.states)}
    lines: list[str] = []

    # Model type
    lines.append(f"type:{model.model_type.name}")

    # States
    for idx, state in enumerate(model.states):
        labels = sorted(state.labels)
        fname = state.friendly_name or ""
        valuations = sorted(
            (str(var), _canonical_value(val))
            for var, val in model.state_valuations.get(state, {}).items()
        )
        val_str = ";".join(f"{k}={v}" for k, v in valuations)
        lines.append(
            f"state:{idx}:labels={','.join(labels)}:name={fname}:val={val_str}"
        )

        if model.supports_observations():
            lines.append(f"state:{idx}:{_canonical_obs(model, state)}")

    # Transitions
    for idx, state in enumerate(model.states):
        if state not in model.transitions:
            continue
        choices = model.transitions[state]
        for action in sorted(choices.actions, key=_action_sort_key):
            branch = choices[action]
            alabel = action.label if action.label is not None else ""
            dist = sorted(
                (state_index[target], _canonical_value(prob)) for prob, target in branch
            )
            dist_str = ";".join(f"{t}:{p}" for t, p in dist)
            lines.append(f"trans:{idx}:action={alabel}:dist={dist_str}")

    # Reward models (sorted by name for stability)
    for rm in sorted(model.rewards, key=lambda r: r.name):
        for state, val in sorted(rm.rewards.items(), key=lambda kv: state_index[kv[0]]):
            lines.append(
                f"reward:{rm.name}:state={state_index[state]}:{_canonical_value(val)}"
            )
        for (s, a, s2), val in sorted(
            rm.transition_rewards.items(),
            key=lambda kv: (
                state_index[kv[0][0]],
                _action_sort_key(kv[0][1]),
                state_index[kv[0][2]],
            ),
        ):
            alabel = a.label if a.label is not None else ""
            lines.append(
                f"reward:{rm.name}:trans={state_index[s]},{alabel},{state_index[s2]}:{_canonical_value(val)}"
            )

    # Markovian states (Markov automata)
    if model.markovian_states:
        ms = sorted(state_index[s] for s in model.markovian_states)
        lines.append(f"markovian:{','.join(str(i) for i in ms)}")

    canonical = "\n".join(lines).encode()
    return hashlib.sha256(canonical).hexdigest()
