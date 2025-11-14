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
# # Building POMDPs
# In Stormvogel, a **Partially Observable Markov Decision Process (POMDP)** consists of
# * states $S$, actions $A$, an initial state $s_0$, a mapping of *enabled actions*, and a successor distribution $P(s,a)$, and a labelling function $L$ as for MDPs,
# * a set of observations $Z$,
# * and a deterministic state-observation function $O\colon S \rightarrow Z$.
#
# The key idea is that the observations encode what information an agent sees.
# An agent will have to make its decisions not based on the current state, but based on the history of observations it has seen.
# Note that usually when we refer to MDPs we actually mean *fully observable* MDPs, which are POMDPs with $Z = S$ and $O(s) = s$.

# %% [markdown]
# We introduce a simple example to illustrate the difference between MDPs and POMDPs.
# The idea is that a coin is flipped while the agent is not looking, and then the agent has to guess if it's heads or tails.
# We first construct an MDP.

# %%
from stormvogel import *

init = ("flip",)


def available_actions(s):
    if "heads" in s or "tails" in s:
        return ["guess_heads", "guess_tails"]
    return [""]


def delta(s, a):
    if s == init:
        return [(0.5, ("heads",)), (0.5, ("tails",))]
    elif a.startswith("guess"):
        if "heads" in s and a == "guess_heads" or "tails" in s and a == "guess_tails":
            return [(1, ("correct", "done"))]
        else:
            return [(1, ("wrong", "done"))]
    else:
        return [(1, s)]


labels = lambda s: list(s)


def rewards(s, a):
    if "correct" in s:
        return {"R": 100}
    return {"R": 0}


coin_mdp = bird.build_bird(
    delta=delta,
    init=init,
    available_actions=available_actions,
    labels=labels,
    modeltype=ModelType.MDP,
    rewards=rewards,
)
vis = show(coin_mdp)

# %% [markdown]
# Since this MDP is fully observable, the agent can actually see what state the world is in. In other words, the agent *knows* whether the coin is head or tails. If we ask stormpy to calculate the policy that maximizes the reward, we see that the agent can always 'guess' correctly because of this information. The chosen actions are highlighted in red. (More on model checking later.)

# %%
result = model_checking(coin_mdp, "Rmax=? [S]")
vis3 = show(coin_mdp, result=result)


# %% [markdown]
# To model the fact that our agent does not know the state correctly, we will need to use a POMDP! (Note that we re-use a lot of code from before)


# %%
def observations(s):
    return 0


coin_pomdp = bird.build_bird(
    delta=delta,
    init=init,
    available_actions=available_actions,
    labels=labels,
    modeltype=ModelType.POMDP,
    rewards=rewards,
    observations=observations,
)

vis3 = show(coin_pomdp)

# %% [markdown]
# Unfortunately, model checking POMDPs turns out to be very hard in general, even undecidable. For this model, the result of model checking would look similar to this. The agent doesn't know if it's currently in the state heads or tails, therefore it just guesses heads and has only a 50 percent chance of winning.

# %%
import stormvogel.result

taken_actions = {}
for id, state in coin_pomdp.states.items():
    taken_actions[id] = state.available_actions()[0]
scheduler2 = stormvogel.result.Scheduler(coin_pomdp, taken_actions)
values = {0: 50, 1: 50, 2: 50, 3: 100.0, 4: 0.0}
result2 = stormvogel.result.Result(coin_pomdp, values, scheduler2)

vis4 = show(coin_pomdp, result=result2)

# %% [markdown]
# We can also create stochastic observations. For example, we could say that the agent sees the correct observation with probability 0.8, and the wrong one with probability 0.2.
# To make the observations more readable we can give them a valuation.


# %%
def observations_stochastic(s):
    if "heads" in s:
        return [(0.8, 0), (0.2, 1)]
    elif "tails" in s:
        return [(0.2, 0), (0.8, 1)]
    else:
        return [(1.0, 2)]


def observation_valuations(o):
    if o == 0:
        return {"heads": True, "tails": False, "done": False}
    elif o == 1:
        return {"heads": False, "tails": True, "done": False}
    else:
        return {"done": True, "heads": False, "tails": False}


coin_pomdp_stochastic = bird.build_bird(
    delta=delta,
    init=init,
    available_actions=available_actions,
    labels=labels,
    modeltype=ModelType.POMDP,
    rewards=rewards,
    observations=observations_stochastic,
    observation_valuations=observation_valuations,
)

vis5 = show(coin_pomdp_stochastic)

# %% [markdown]
# This model cannot be given to stormpy for model checking as is, we first need to determinize the observations.

# %%
coin_pomdp_stochastic.make_observations_deterministic()
vis6 = show(coin_pomdp_stochastic)
