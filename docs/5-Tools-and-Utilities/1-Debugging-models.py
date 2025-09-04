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
# # Debugging models
# Using stormvogel and stormpy, you can do a number of things to debug your models.
#
# * Showing End Components
# * Showing Prob01 sets
# * Showing shortest stochastic paths
# * Adding assertions to your models
#
# We will demonstrate this with this simple toy MDP model.

# %%
from stormvogel import *
mdp = examples.create_debugging_mdp()
vis = show(mdp, layout=Layout("layouts/mec.json"))

# %% [markdown]
# ## Showing Maximal End Components
# (This is already included in another notebook, but also here for completeness)

# %%
decomp = extensions.stormvogel_get_maximal_end_components(mdp)
vis.highlight_decomposition(decomp)

# %% [markdown]
# ## Showing Prob01 sets
# Given an MDP, a set of states $\phi$, and a set of states $\psi$.
#
# * The Prob01 **maximal** set is the set of states where $\phi$ until $\psi$ holds under **all** policies (schedulers).
# * The Prob01 **minimal** set is the set of states where $\phi$ until $\psi$ holds under **some** policy (scheduler).
#
# In a DTMC the concept of maximal or minmal does not make sense, so we simply talk about the Prob01 set.
#
# Let us calculate the prob01 max states and min states for our model.

# %%
from stormvogel.extensions import to_bit_vector as bv

sp_mdp = stormpy_utils.mapping.stormvogel_to_stormpy(mdp)
max_res = stormpy.compute_prob01max_states(sp_mdp, bv({0, 1}, sp_mdp), bv({2}, sp_mdp))
min_res = stormpy.compute_prob01min_states(sp_mdp, bv({0, 1}, sp_mdp), bv({2}, sp_mdp))
print(0, mdp[0].labels)
print(1, mdp[1].labels)
# Note that for a DTMC, we can use `compute_prob01_states`.
max_0 = set(max_res[0])
max_1 = set(max_res[1])
min_0 = set(min_res[0])
min_1 = set(min_res[1])

# %%
vis = show(mdp, layout=Layout("layouts/mec.json"))

# %%
vis.highlight_state_set(max_0, color="pink")

# %%
vis.clear_highlighting()

# %%
vis.highlight_state_set(max_1, color="orange")

# %%
vis.clear_highlighting()

# %%
vis.highlight_state_set(min_0, color="pink")

# %%
vis.clear_highlighting()

# %%
vis.highlight_state_set(min_1, color="pink")

# %%
vis.clear_highlighting()
