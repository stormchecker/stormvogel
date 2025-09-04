# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # End component elimination
# End Components (ECs) are sets of states in an MDP where:
# * Every state can reach every other state. (Strongly Connected Component)
# * All actions always lead to a state that is also in the MEC (there is no escape)
#
# For analysis, it is often useful to eliminate these MECs to a single state. We will show how to do this using stormpy and stormvogel.

# %%
from stormvogel import *

init = "init"

def available_actions(s: bird.State):
    if s == "init":
        return [["one"], ["two"]]
    elif s == "mec1" or s == "mec2":
        return [["one"], ["two"]]
    return [[]]

def delta(s: bird.State, a: bird.Action):
    if s == "init" and "one" in a:
        return [(0.5, "mec1"), (0.5, "mec2")]
    elif s == "mec1":
        return [(1, "mec2")]
    elif s == "mec2":
        return [(1, "mec1")]
    elif s == "init" and "two" in a:
        return [(1, "mec1")]
    return [(1,s)]

labels = lambda s: s

mdp = bird.build_bird(
    delta=delta,
    init=init,
    available_actions=available_actions,
    labels=labels,
    modeltype=ModelType.MDP,)
vis = show(mdp, layout=Layout("layouts/mec.json"))


# %% [markdown]
# First, let's show the maximal end components of this model.

# %%
def stormvogel_get_maximal_end_components(sv_model):
    sp_model = stormpy_utils.mapping.stormvogel_to_stormpy(sv_model)
    f = extensions.choice_mapping(sv_model, sp_model)
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

decomp = stormvogel_get_maximal_end_components(mdp)
print(decomp)

# %%
vis.highlight_decomposition(decomp)

# %% [markdown]
# Now we perform end component elimination by converting to stormpy and back.

# %%
import stormpy

def map_state_labels(m, res):
    """Based on the result of EC elimination, create a new state labeling that can be used for a new model that captures the result.
    Args:
        m: stormpy model
        res (EndComponentEliminatorReturnTypeDouble): EC result
    """
    old_nr_columns = m.transition_matrix.nr_columns
    new_nr_columns = res.matrix.nr_columns
    sl = stormpy.StateLabeling(new_nr_columns)

    for s_old in range(old_nr_columns):
        s_new = res.old_to_new_state_mapping[s_old]
        for l in m.labeling.get_labels_of_state(s_old):
            sl.add_label(l)
            sl.add_label_to_state(l, s_new)
    return sl

def map_choice_labels(m_old, m_new, res):
    """Based on the result of EC elimination, create a new choice labeling that can be used for a new model that captures the result.
    Args:
        m_old: old stormpy model
        m_new: new strompy model that is based on res
        res (EndComponentEliminatorReturnTypeDouble): EC result
    """
    new_nr_rows = m_new.transition_matrix.nr_rows
    cl = stormpy.storage.ChoiceLabeling(new_nr_rows)
    old_nr_columns = m_old.transition_matrix.nr_columns

    for s_old in range(old_nr_columns):
        s_new = res.old_to_new_state_mapping[s_old]
        old_no_choices = m_old.get_nr_available_actions(s_old)
        new_no_choices = m_new.get_nr_available_actions(s_new)
        if (old_no_choices == new_no_choices):
            for action_no in range(old_no_choices):
                old_index = m_old.get_choice_index(s_old, action_no)
                labels = m_old.choice_labeling.get_labels_of_choice(old_index)
                new_index = m_new.get_choice_index(s_new, action_no)
                for l in labels:
                    cl.add_label(l)
                    cl.add_label_to_choice(l, new_index)
    return cl

def simple_ec_elimination(m):
    """Perform EC elimination on a stormpy model while preserving labels.
    Label sets of merged states are unified.
    Action labels are preserved when possible.
    Args:
        m: stormpy model
    """
    # Keep all states, and consider ecs to be possible anywhere in the model
    subsystem = stormpy.BitVector(m.nr_states, True)
    possible_ec_rows = stormpy.BitVector(m.nr_choices, True)
    res = stormpy.eliminate_ECs(
        matrix = m.transition_matrix,
        subsystem = subsystem,
        possible_ecs = possible_ec_rows,
        add_sink_row_states = subsystem,
        add_self_loop_at_sink_states=True
    )
    new_labels = map_state_labels(m, res)
    components = stormpy.SparseModelComponents(transition_matrix=res.matrix, state_labeling=new_labels)
    m_new = stormpy.storage.SparseMdp(components)
    components.choice_labeling = map_choice_labels(m, m_new, res)
    m_updated = stormpy.storage.SparseMdp(components)
    return m_updated


# %%
sp_mdp = stormpy_utils.mapping.stormvogel_to_stormpy(mdp)
sp_mdp_elim = simple_ec_elimination(sp_mdp)
sv_mdp_elim = stormpy_utils.mapping.stormpy_to_stormvogel(sp_mdp_elim)
vis = show(sv_mdp_elim)

# %% [markdown]
# These functions are also available under stormvogel.extensions.ec_elimination.

# %%
sp_mdp_elim2 = extensions.simple_ec_elimination(sp_mdp)
sv_mdp_elim2 = stormpy_utils.mapping.stormpy_to_stormvogel(sp_mdp_elim2)
sv_mdp_elim == sv_mdp_elim2
