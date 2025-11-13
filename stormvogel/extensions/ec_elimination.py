from typing import Tuple
from bidict import bidict

try:
    import stormpy
except ImportError:
    stormpy = None

import stormvogel.model
import stormvogel.stormpy_utils.mapping as mapping


def map_state_labels(m, res):
    """Based on the result of EC elimination, create a new state labeling that can be used for a new model that captures the result.
    Args:
        m: stormpy model
        res (EndComponentEliminatorReturnTypeDouble): EC result
    """
    assert stormpy is not None
    old_nr_columns = m.transition_matrix.nr_columns
    new_nr_columns = res.matrix.nr_columns
    sl = stormpy.StateLabeling(new_nr_columns)

    for s_old in range(old_nr_columns):
        s_new = res.old_to_new_state_mapping[s_old]
        for label in m.labeling.get_labels_of_state(s_old):
            sl.add_label(label)
            sl.add_label_to_state(label, s_new)
    return sl


def map_choice_labels(m_old, m_new, res):
    """Based on the result of EC elimination, create a new choice labeling that can be used for a new model that captures the result.
    Args:
        m_old: old stormpy model
        m_new: new strompy model that is based on res
        res (EndComponentEliminatorReturnTypeDouble): EC result
    """
    assert stormpy is not None
    new_nr_rows = m_new.transition_matrix.nr_rows
    cl = stormpy.storage.ChoiceLabeling(new_nr_rows)
    old_nr_columns = m_old.transition_matrix.nr_columns

    for s_old in range(old_nr_columns):
        s_new = res.old_to_new_state_mapping[s_old]
        old_no_choices = m_old.get_nr_available_actions(s_old)
        new_no_choices = m_new.get_nr_available_actions(s_new)
        if old_no_choices == new_no_choices:
            for action_no in range(old_no_choices):
                old_index = m_old.get_choice_index(s_old, action_no)
                labels = m_old.choice_labeling.get_labels_of_choice(old_index)
                new_index = m_new.get_choice_index(s_new, action_no)
                for label in labels:
                    cl.add_label(label)
                    cl.add_label_to_choice(label, new_index)
    return cl


def simple_ec_elimination(m):
    """Perform End Component elimination on a stormpy model while preserving labels.
    Label sets of merged states are unified.
    Action labels are preserved when possible.
    Args:
        m: stormpy model
    """
    assert stormpy is not None
    # Keep all states, and consider ecs to be possible anywhere in the model
    subsystem = stormpy.BitVector(m.nr_states, True)
    possible_ec_rows = stormpy.BitVector(m.nr_choices, True)
    res = stormpy.eliminate_ECs(
        matrix=m.transition_matrix,
        subsystem=subsystem,
        possible_ecs=possible_ec_rows,
        add_sink_row_states=subsystem,
        add_self_loop_at_sink_states=True,
    )
    new_labels = map_state_labels(m, res)
    components = stormpy.SparseModelComponents(
        transition_matrix=res.matrix, state_labeling=new_labels
    )
    m_new = stormpy.storage.SparseMdp(components)
    components.choice_labeling = map_choice_labels(m, m_new, res)
    m_updated = stormpy.storage.SparseMdp(components)
    return m_updated


def stormvogel_get_maximal_end_components(
    sv_model: stormvogel.model.Model,
) -> list[Tuple[set[int], set[stormvogel.model.Action]]]:
    """Get the maximal end components of this model.
    They are returned as a list of tuples where the first element is a set of state ids, and the second a set of actions.
    """
    assert stormpy is not None
    sp_model = mapping.stormvogel_to_stormpy(sv_model)
    f = choice_mapping(sv_model, sp_model)
    decomposition = stormpy.get_maximal_end_components(sp_model)
    res = []
    for mec in decomposition:
        states = set()
        actions = set()
        for s_id, choices in mec:
            states.add(s_id)
            actions = actions | set(map(lambda x: f.inverse[x], choices))
        res.append((frozenset(states), frozenset(actions)))
    return res


def choice_mapping(sv_model, sp_model):
    """Return a bijective mapping between the stormvogel state-action pairs and the stormvogel model.
    WARNING: This function will be depricated later. It might also be faulty, I don't know :))
    """
    res = bidict({})
    choice_id = 0
    for _, s in sv_model:
        for a in s.available_actions():
            res[s.id, a] = choice_id
            choice_id += 1
    return res
