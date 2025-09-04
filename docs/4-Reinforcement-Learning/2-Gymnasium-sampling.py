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
# # Gymnasium sampling
# For the Frozenlake, Cliffwalking, and Taxi models, we were able to access the internal state of the gym environments to convert it to an accurate stormvogel model. However, this is not the case for arbitrary gym environments, hence we can use sampling to get an approximation of the gym envrironment in stormvogel. In this notebook we give some example usages of gymnasium sampling from `stormvogel.extensions`. Note that sampling is actually quite fast, but the visualization gets slow quickly when the amount of states increases.
#
# * `sample_gym` samples a gym environment and gives the result as 5 defaultdicts and an integer.
# * `sample_to_stormvogel` converts such a sample to a stormvogel model.
# * `sample_gym_to_stormvogel` combines the two for convenience.

# %% [markdown]
# ## FrozenLake
# Since FrozenLake does not have too many states, and it is fully observable, we are actually very likely to get the correct model if we use enough samples. If you lower the sample rate, you will observe that at some point, transitions and states will disappear. You can also enable is_slippery. You will then get an approximation of the FrozenLake with slipping ice.

# %%
from stormvogel import *
import gymnasium as gym
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
model = extensions.sample_gym_to_stormvogel(env, no_samples=200)
print(model.summary())
show(model, layout=Layout("layouts/frozenlake.json"))

# %% [markdown]
# ## Blackjack
# Blackjack is not fully observable, hence states that are not identical in the gymnasium model are merged in the sample model.

# %%
import gymnasium as gym
from stormvogel import *
env = gym.make("Blackjack-v1", render_mode="rgb_array")
model = extensions.sample_gym_to_stormvogel(env, no_samples=50)
print(model.summary())
show(model)

# %% [markdown]
# ## Acrobot
# We can even sample continuous environments and treat them like MDPs. In this particular case, all numbers are rounded to 1 decimal (in `convert_obs`). The more accurate you want to be, the more states are required!

# %%
env = gym.make('Acrobot-v1', render_mode="rgb_array")

def convert_obs(xs):
    return tuple([round(float(x),1) for x in xs])

model = extensions.sample_gym_to_stormvogel(env, no_samples=10, sample_length=5, convert_obs=convert_obs, max_size=10000)
print(model.summary())
show(model)
