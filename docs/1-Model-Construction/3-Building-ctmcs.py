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
# # Building CTMCs
# A Continuous Time Markov Chain (CTMC) is similar to a DTMC. The differences are that time is *continuous*, not discrete, and that transitions have *rates* instead of probabilities. So to summarize, a CTMC has:
# * states (includig an initial state)
# * transitions with rates. The rate is an indication of the speed in which a transition occurs. To be precise, the probability that state $s_i$ goes to state $s_j$ in $t$ time steps is $1- e^{R(s_i,s_j)\cdot t}$, where $R(s_i,s_j)$ is the *rate* between $s_i$ and $s_j$.
# * labels
#
# As an example for a CTMC, we have a model of a star. It will first fuse hydrogen into helium until there is no hydrogen left, then it will fuse helium into carbon, etc. until there is only iron left and the star goes supernova.

# %%
from stormvogel import *
# Create a new model with the name "Nuclear fusion"
ctmc = stormvogel.model.new_ctmc("Nuclear fusion")

# hydrogen fuses into helium
ctmc.get_state_by_id(0).set_choice([(3, ctmc.new_state("helium"))])
# helium fuses into carbon
ctmc.get_state_by_id(1).set_choice([(2, ctmc.new_state("carbon"))])
# carbon fuses into iron
ctmc.get_state_by_id(2).set_choice([(7, ctmc.new_state("iron"))])
# supernova
ctmc.get_state_by_id(3).set_choice([(12, ctmc.new_state("Supernova"))])

# we add the rates which are equal to whats in the transitions since the probabilities are all 1
rates = [3, 2, 7, 12, 0]
for i in range(5):
    ctmc.set_rate(ctmc.get_state_by_id(i), rates[i])

# we add self loops to all states with no outgoing transitions
ctmc.add_self_loops()
vis = show(ctmc, layout=Layout("layouts/star.json"))

# %% [markdown]
# Or using the bird API

# %%
from stormvogel import *

init = bird.State(x="")

def delta(s: bird.State):
    if s == init:
        return [(3, bird.State(x=["helium"]))]
    elif "helium" in s.x:
        return [(2, bird.State(x=["carbon"]))]
    elif "carbon" in s.x:
        return [(7, bird.State(x=["iron"]))]
    elif "iron" in s.x:
        return [(12, bird.State(x=["Supernova"]))]
    else:
        return [(0, s)]

labels = lambda s: s.x

bird_star = bird.build_bird(
    delta=delta,
    init=init,
    labels=labels,
    modeltype=ModelType.CTMC
)

vis2 = show(bird_star, layout=Layout("layouts/star.json"))
