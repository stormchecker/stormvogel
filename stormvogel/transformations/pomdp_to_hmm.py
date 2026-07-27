"""Transformation: uniformise a POMDP's actions away to obtain an HMM."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State


def pomdp_to_hmm(pomdp: "Model") -> "Model":
    """Return the HMM obtained by resolving a POMDP's actions uniformly at random.

    Every state's choices are merged into a single unlabelled distribution,
    each action contributing with weight ``1 / n`` where ``n`` is the number of
    actions available in that state::

        P(s' | s) = (1/n) · Σ_a P(s' | s, a)

    Observations, state labels, valuations, friendly names and reward models
    are carried over unchanged; rewards attached to *actions* are dropped,
    since the actions are gone.

    This is the standard way to view a POMDP benchmark as a hidden Markov
    model: there is no longer anything to control, so the only question left
    is which observation sequences the model produces.  It is what makes
    :func:`~stormvogel.teaching.belief_search.dijkstra_belief_search` report an
    honest probability rather than a probability conditioned on a choice of
    actions.

    :param pomdp: The POMDP to convert.  Not modified.
    :returns: A new :class:`~stormvogel.model.model.Model` of type
        :attr:`~stormvogel.model.model.ModelType.HMM`.
    :raises ValueError: If *pomdp* is not a POMDP, or if any state lacks an
        observation or has a stochastic one.
    """
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.model import ModelType, new_hmm

    if pomdp.model_type != ModelType.POMDP:
        raise ValueError(f"pomdp_to_hmm requires a POMDP; got {pomdp.model_type}.")

    hmm = new_hmm(create_initial_state=False)

    observations = {
        observation: hmm.new_observation(
            alias, pomdp.observation_valuations.get(observation) or None
        )
        for observation, alias in pomdp.observation_aliases.items()
    }

    states: dict["State", "State"] = {}
    for state in pomdp.states:
        observation = pomdp.state_observations.get(state)
        if isinstance(observation, Distribution):
            raise ValueError(
                f"State {state!r} has a stochastic observation; "
                "pomdp_to_hmm requires deterministic state observations."
            )
        if observation is None:
            raise ValueError(f"State {state!r} has no observation.")
        states[state] = hmm.new_state(
            list(state.labels),
            valuations=dict(pomdp.state_valuations.get(state, {})) or None,
            friendly_name=pomdp.friendly_names.get(state),
            observation=observations[observation],
        )

    for state, choices in pomdp.transitions.items():
        branches = list(choices)
        if not branches:
            continue
        weight = Fraction(1, len(branches))
        merged: dict["State", Fraction] = {}
        for _action, branch in branches:
            for probability, target in branch:
                merged[target] = (
                    merged.get(target, Fraction(0)) + Fraction(probability) * weight
                )
        hmm.set_choices(
            states[state],
            [(probability, states[target]) for target, probability in merged.items()],
        )

    for reward_model in pomdp.rewards:
        new_rewards = hmm.new_reward_model(reward_model.name)
        for state, value in reward_model.rewards.items():
            if state in states:
                new_rewards.rewards[states[state]] = value

    return hmm
