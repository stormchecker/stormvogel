"""MEC detection and elimination for MDPs.

Uses stormpy for the heavy lifting; exposes a stormvogel-native API.
"""

try:
    import stormpy
except ImportError:
    stormpy = None  # type: ignore[assignment]

import stormvogel.model as model
from stormvogel.stormpy_utils import mapping


def detect_mecs(
    mdp: model.Model,
) -> list[frozenset[model.State]]:
    """Return the maximal end components of *mdp* as frozensets of stormvogel states.

    Each MEC is returned as a :class:`frozenset` of :class:`~stormvogel.model.State`
    objects from the original model.  Trivial MECs (single states with a
    self-loop only) are included.

    :param mdp: A stormvogel MDP.
    :returns: List of MECs, each represented as a frozenset of states.
    :raises RuntimeError: If stormpy is not available.
    """
    if stormpy is None:
        raise RuntimeError("stormpy is required for MEC detection.")

    sp_model = mapping.stormvogel_to_stormpy(mdp)
    sp_id_to_sv: dict[int, model.State] = {i: s for s, i in mdp.stormpy_id.items()}

    return [
        frozenset(sp_id_to_sv[s_id] for s_id, _ in mec)
        for mec in stormpy.get_maximal_end_components(sp_model)
    ]


def eliminate_mecs(
    mdp: model.Model,
    remove_representative_selfloops: bool = False,
    make_representatives_absorbing: bool = False,
) -> tuple[model.Model, dict[model.State, model.State]]:
    """Eliminate all end components from *mdp* and return the resulting model.

    Each MEC is collapsed to a single representative state that inherits the
    union of labels from all states that were merged into it.

    Two mutually exclusive post-processing options control how representative
    states (those produced by merging a non-trivial MEC) are treated:

    - ``remove_representative_selfloops=True``: strip the stormpy-added
      self-loop from any representative that still has exit actions.  Useful
      for *maximum* reachability IVI so the upper bound can descend.
    - ``make_representatives_absorbing=True``: strip all exit actions from
      each representative, leaving only the self-loop.  Useful for *minimum*
      reachability: the representative becomes a trap (value 0), reflecting
      that the scheduler can choose to stay in the MEC forever.

    :param mdp: A stormvogel MDP.
    :param remove_representative_selfloops: Remove self-loop choices from
        representative states that have at least one exit action.
    :param make_representatives_absorbing: Remove all non-self-loop choices
        from representative states, making them absorbing sinks.
    :returns: A pair ``(new_mdp, state_map)`` where *state_map* maps every
        state of *mdp* to its corresponding state in *new_mdp*.
    :raises RuntimeError: If stormpy is not available.
    :raises ValueError: If both post-processing options are requested.
    """
    if remove_representative_selfloops and make_representatives_absorbing:
        raise ValueError(
            "remove_representative_selfloops and make_representatives_absorbing "
            "are mutually exclusive."
        )
    if stormpy is None:
        raise RuntimeError("stormpy is required for MEC elimination.")

    sp_model = mapping.stormvogel_to_stormpy(mdp)

    subsystem = stormpy.BitVector(sp_model.nr_states, True)
    possible_ec_rows = stormpy.BitVector(sp_model.nr_choices, True)
    res = stormpy.eliminate_ECs(
        matrix=sp_model.transition_matrix,
        subsystem=subsystem,
        possible_ecs=possible_ec_rows,
        add_sink_row_states=subsystem,
        add_self_loop_at_sink_states=True,
    )

    # Build state labeling for the new model: each new state receives the
    # union of labels from all old states that map to it.
    sl = stormpy.StateLabeling(res.matrix.nr_columns)
    for s_old in range(sp_model.nr_states):
        s_new = res.old_to_new_state_mapping[s_old]
        for label in sp_model.labeling.get_labels_of_state(s_old):
            sl.add_label(label)
            sl.add_label_to_state(label, s_new)

    components = stormpy.SparseModelComponents(
        transition_matrix=res.matrix,
        state_labeling=sl,
    )
    sv_new = mapping.stormpy_to_stormvogel(stormpy.storage.SparseMdp(components))

    # stormvogel_to_stormpy assigns stormpy ID i to mdp.states[i].
    # stormpy_to_stormvogel adds states in stormpy ID order, so sv_new.states[j]
    # corresponds to new stormpy state j.
    state_map: dict[model.State, model.State] = {
        sv_old: sv_new.states[res.old_to_new_state_mapping[sp_id]]
        for sv_old, sp_id in mdp.stormpy_id.items()
    }

    if remove_representative_selfloops or make_representatives_absorbing:
        # Collect representative states: new states that multiple old states map to.
        new_id_count: dict[int, int] = {}
        for sp_id in range(sp_model.nr_states):
            nid = res.old_to_new_state_mapping[sp_id]
            new_id_count[nid] = new_id_count.get(nid, 0) + 1
        representative_new_ids = {nid for nid, cnt in new_id_count.items() if cnt > 1}

        for new_id in representative_new_ids:
            s = sv_new.states[new_id]
            is_selfloop = lambda branch: all(succ is s for _, succ in branch)  # noqa: E731
            to_shorthand = lambda branch: list(branch)  # noqa: E731
            if remove_representative_selfloops:
                filtered = [(a, b) for a, b in s.choices if not is_selfloop(b)]
                if filtered:
                    s.set_choices({a: to_shorthand(b) for a, b in filtered})
            else:
                # make_representatives_absorbing: keep only the self-loop.
                selfloops = [(a, b) for a, b in s.choices if is_selfloop(b)]
                if selfloops:
                    s.set_choices({a: to_shorthand(b) for a, b in selfloops[:1]})

    return sv_new, state_map
