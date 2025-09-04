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
# # Building MDPs
# In Stormvogel, a **Markov Decision Process (MDP)** consists of:
# * states $S$,
# * actions $A$,
# * an initial state $s_0$,
# * a mapping from states to sets of *enabled actions*,
# * a successor distribution $P(s,a)$ for every state $s$ and every enabled action $a$, i.e., sets of transitions between states $s$ and $s'$, each annotated with an action and a probability.
# * state labels $L(s)$.
#
#
# Here we show how to construct a simple example mdp using the bird API and the model builder API.
# The idea is that you can choose to study (you will likely pass the exam but you have less free time) or not to study (you will have more free time but risk failing the exam).

# %% [markdown]
# ## The study dilemma
# This little MDP is supposed to help you decide whether you should stuy or not.

# %% [markdown]
# ### PGC API
# For MDPs, you specify the availaible actions in `available_actions`. An action here is simply a list of labels. You specify the transition of a state-action pair in `delta`.

# %%
from stormvogel import *

def available_actions(s):
    if s == "init": # Either study or not
        return [["study"], ["don't study"]]
    else: # Otherwise, we have no choice (DTMC-like behavior)
        return [[]]

def delta(s, a):
    if "study" in a:
        return ["studied"]
    elif "don't study" in a:
        return [(1, "didn't study")]
    elif s == "studied":
        return [(9/10, "pass test"), (1/10, "fail test")]
    elif s == "didn't study":
        return [(2/5, "pass test"), (3/5, "fail test")]
    else:
        return [(1, "end")]

def labels(s):
    return s

# For rewards, you have to provide a dict. This enables multiple reward models if you use a non-singleton list.
def rewards(s: bird.State, a: bird.Action):
    if s == "pass test":
        return {"R":100}
    if s == "didn't study":
        return {"R":15}
    else:
        return {"R":0}

bird_study = bird.build_bird(
    delta=delta,
    init="init",
    available_actions=available_actions,
    labels=labels,
    modeltype=ModelType.MDP,
    rewards=rewards
)
vis = show(bird_study, layout=Layout("layouts/pinkgreen.json"))

# %% [markdown]
# ### Model API

# %%
from stormvogel import *

mdp = stormvogel.model.new_mdp("Study")

init = mdp.get_initial_state()
study = mdp.action("study")
not_study = mdp.action("don't study")

studied = mdp.new_state("studied")
not_studied = mdp.new_state("didn't study")
pass_test = mdp.new_state("pass test")
fail_test = mdp.new_state("fail test")
end = mdp.new_state("end")

init.set_choice([
    (study, studied),
    (not_study, not_studied)
])

studied.set_choice([
    (9/10, pass_test),
    (1/10, fail_test)
])

not_studied.set_choice([
    (4/10, pass_test),
    (6/10, fail_test)
])

pass_test.set_choice([(1, end)])
fail_test.set_choice([(1, end)])

reward_model = mdp.new_reward_model("R")
reward_model.set_state_action_reward(pass_test, EmptyAction, 100)
reward_model.set_state_action_reward(fail_test, EmptyAction, 0)
reward_model.set_state_action_reward(not_studied, EmptyAction, 15)
reward_model.set_unset_rewards(0)

vis2 = show(mdp, layout=Layout("layouts/pinkgreen.json"))

# %% [markdown]
# ## Grid model
# An MDP model that consists of a 3x3 grid. The direction to walk is chosen by an action.

# %% [markdown]
# ### PGC API

# %%
N = 3

ACTION_SEMANTICS = {
    "l": (-1, 0),
    "r": (1, 0),
    "u": (0, -1),
    "d": (0, 1) }

def available_actions(s):
    res = []
    if s[0] > 0:
        res.append("l")
    if s[0] < N-1:
        res.append("r")
    if s[1] > 0:
        res.append("u")
    if s[1] < N-1:
        res.append("d")
    return res

def pairwise_plus(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def delta(s, a):
    return [(1, pairwise_plus(s, ACTION_SEMANTICS[a]))]

def labels(s):
    return [str(s)]

m1 = bird.build_bird(
    init=(0,0),
    available_actions=available_actions,
    labels=labels,
    delta=delta)
vis3 = show(m1)

# %% [markdown]
# ### Model API

# %%
grid_model = stormvogel.model.new_mdp(create_initial_state=False)

for x in range(N):
    for y in range(N):
        grid_model.new_state(f"({x},{y})")

for x in range(N):
    for y in range(N):
        state_tile_label = str((x,y)).replace(" ", "")
        state = grid_model.get_states_with_label(state_tile_label)[0]
        av = available_actions((x,y))
        for a in av:
            target_tile_label = str(pairwise_plus((x,y), ACTION_SEMANTICS[a])).replace(" ", "")
            target_state = grid_model.get_states_with_label(target_tile_label)[0]
            state.add_choice([(grid_model.action(a), target_state)])
vis4 = show(grid_model)
