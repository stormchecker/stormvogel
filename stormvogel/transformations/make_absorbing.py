"""Transformation: make a set of states absorbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State


def make_absorbing(
    model: "Model",
    states: "State | set[State] | str",
) -> None:
    """Make states absorbing by replacing their outgoing transitions with a self-loop.

    Each target state's :class:`~stormvogel.model.choices.Choices` is replaced
    by a single ``EmptyAction → {state: 1}`` self-loop via
    :meth:`~stormvogel.model.model.Model.set_choices`.  The model is mutated
    in-place.

    :param model: The model to mutate.
    :param states: A single :class:`~stormvogel.model.state.State`, a
        ``set[State]``, or a label string resolved via
        :meth:`~stormvogel.model.model.Model.get_states_with_label`.
    :raises KeyError: If *states* is a label that does not exist in *model*.
    """
    from stormvogel.model.state import State as _State

    if isinstance(states, str):
        target: set[State] = model.get_states_with_label(states)
    elif isinstance(states, _State):
        target = {states}
    else:
        target = states

    for state in target:
        model.set_choices(state, [(1, state)])
