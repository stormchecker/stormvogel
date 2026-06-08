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
import stormvogel

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

# %% [markdown]
# ## Gymnasium-Compliant Environment
#
# Stormvogel models can be wrapped as a [Gymnasium](https://gymnasium.farama.org/)
# environment via `ModelEnv`. This lets you use standard reinforcement-learning
# libraries directly on a stormvogel MDP or DTMC without any manual glue code.
#
# `ModelEnv` supports both **MDP** and **DTMC** models:
# * For an MDP the action space is `Discrete(n_actions)`, one index per named action.
# * For a DTMC there is no choice, so the action space is `Discrete(1)` — always pass `0`.
#
# The observation space has two modes, selected by `obs_type`:
# * `"index"` (default): a plain integer — the index of the current state.
# * `"valuations"`: a `Dict` space built from variables that have a declared domain
#   (`IntDomain`, `BoolDomain`, or `CategoricalDomain`), one `Discrete` component
#   per variable.

# %%
from stormvogel.gym_env import ModelEnv, ActionUnavailableError

env = ModelEnv(lion)
print("Observation space:", env.observation_space)
print("Action space:     ", env.action_space)
print("Actions:          ", env._index_to_action)

# %% [markdown]
# The standard Gymnasium loop works as-is.  `reset()` returns the initial
# observation and an info dict; `step(action)` returns the next observation,
# reward, terminated flag, truncated flag, and an info dict.  The info dict
# always contains the raw `stormvogel.model.State` under the key `"state"`.

# %%
obs, info = env.reset(seed=42)
print("Initial state index:", obs, "— state:", info["state"])

hunt_idx = next(i for i, a in enumerate(env._index_to_action) if "hunt" in str(a))
obs, reward, terminated, truncated, info = env.step(hunt_idx)
print("After 'hunt': obs =", obs, "| reward =", reward, "| terminated =", terminated)

# %% [markdown]
# Passing an action that is not available in the current state raises
# `ActionUnavailableError` rather than silently producing incorrect behaviour.
#
# ### Variable-domain observations
#
# When all variables in the model carry a declared domain, `obs_type="valuations"`
# gives a `Dict` observation whose keys are variable names and whose values are
# non-negative integers (domain encoding).  This is more informative than a raw
# state index and compatible with structured RL policies.

# %%
from stormvogel.examples.monty_hall import create_monty_hall_mdp

mh = create_monty_hall_mdp()
mh_env = ModelEnv(mh, obs_type="valuations")
print("Observation space:", mh_env.observation_space)
obs, info = mh_env.reset()
print("Initial obs (all variables at sentinel -1):", obs)
