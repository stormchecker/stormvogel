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
# # Simulator
# If we have a transition system, it might be nice to run a simulation. In this case, we have an MDP that models a hungry lion. Depending on the state it is in, it needs to decide whether it wants to 'rawr' or 'hunt' in order to prevent reaching the state 'dead'.

# %%
from stormvogel import *

lion = examples.create_lion_mdp()
show(lion)

# %% [markdown]
# Now, let's run a simulation of the lion! If we do not provide a scheduling function, then the simulator just does a random walk, taking a random choice each time.

# %%
path = simulate_path(lion, steps=5, seed=1234)


# %% [markdown]
# We could also provide a scheduling function to choose the actions ourselves. This is somewhat similar to the `bird` API.


# %%
def scheduler(s: State) -> Action:
    return Action("rawr")


path2 = stormvogel.simulator.simulate_path(
    lion, steps=5, seed=1234, scheduler=scheduler
)
# %% [markdown]
# We can also use the scheduler to create a partial model. This model contains all the states that have been discovered by the the simulation.

# %%
partial_model = stormvogel.simulator.simulate(
    lion, steps=5, scheduler=scheduler, seed=1234
)
show(partial_model)
