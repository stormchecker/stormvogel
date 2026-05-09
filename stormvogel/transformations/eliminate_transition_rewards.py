"""Transformation: rewrite transition rewards into state rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model


def eliminate_transition_rewards(model: "Model") -> "Model":
    """Rewrite transition rewards into state rewards via auxiliary entry states.

    For each transition (s, a, s') carrying reward r in any reward model,
    inserts an auxiliary state e and reroutes:
        s --a, p--> s'   =>   s --a, p--> e --τ, 1--> s'
    with state_reward(e) = r.

    Returns a new model with only state rewards. If no transition rewards are
    present, returns *model* unchanged.

    :param model: The source model.
    :returns: A new model with only state rewards, or *model* if it had none.
    """
    from stormvogel.model.action import EmptyAction
    from stormvogel.model.choices import Choices
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.model import Model as _Model

    has_any = any(rw.has_transition_rewards() for rw in model.rewards)
    if not has_any:
        return model

    new_model = _Model(model_type=model.model_type, create_initial_state=False)

    state_map: dict = {}
    for s in model.states:
        new_s = new_model.new_state()
        for label in s.labels:
            new_s.add_label(label)
        if s.friendly_name:
            new_s.set_friendly_name(s.friendly_name)
        state_map[s] = new_s

    rewarded_triples: set = set()
    for rw in model.rewards:
        for (s, a, s_next), v in rw.transition_rewards.items():
            if v:
                rewarded_triples.add((s, a, s_next))

    entry_map: dict = {}
    for s, a, s_next in rewarded_triples:
        e = new_model.new_state()
        s_name = s.friendly_name or str(s.state_id)
        t_name = s_next.friendly_name or str(s_next.state_id)
        a_name = a.label if a != EmptyAction else "ε"
        e.set_friendly_name(f"e({s_name},{a_name},{t_name})")
        entry_map[(s, a, s_next)] = e

    uses_actions = model.supports_actions()
    tau = new_model.action("τ") if uses_actions else EmptyAction

    for s, choice in model.transitions.items():
        new_s = state_map[s]
        new_choices: dict = {}
        for a, branch in choice:
            if a != EmptyAction:
                assert a.label is not None
                new_a = new_model.action(a.label)
            else:
                new_a = EmptyAction
            new_branch = [
                (
                    prob,
                    entry_map[(s, a, s_next)]
                    if (s, a, s_next) in entry_map
                    else state_map[s_next],
                )
                for prob, s_next in branch
            ]
            new_choices[new_a] = Distribution(new_branch)
        new_model.add_choices(new_s, Choices(new_choices))

    for (s, a, s_next), e in entry_map.items():
        dest = state_map[s_next]
        new_model.add_choices(
            e,
            Choices({tau: Distribution([(1, dest)])}),
        )

    for rw in model.rewards:
        new_rw = new_model.new_reward_model(rw.name)
        for s, v in rw.rewards.items():
            new_rw.set_state_reward(state_map[s], v)
        for (s, a, s_next), e in entry_map.items():
            v = rw.get_transition_reward(s, a, s_next)
            if v:
                new_rw.set_state_reward(e, v)

    return new_model
