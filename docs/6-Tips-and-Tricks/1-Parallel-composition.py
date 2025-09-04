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
# # Parallel composition
# Using the Bird API, you can define your own parallel composition logic. This works similar to PRISM models. We give an example here. We create two MDP models `m1` and `m2`, and then we create the quotient model `q`. They synchronize on the action `r`.

# %% [markdown]
# ## m1
# `m1` is a simple 2x2 grid model where taking `l` `r` `u` and `d` move to the next tile.

# %%
from stormvogel import *
N = 2

ACTION_SEMANTICS = {
    "l": (-1, 0),
    "r": (1, 0),
    "u": (0, -1),
    "d": (0, 1) }

def available_actions_m1(s):
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

def delta_m1(s, a):
    return [(1, pairwise_plus(s, ACTION_SEMANTICS[a]))]

def labels_m1(s):
    return [str(s)]

m1 = bird.build_bird(
    init=(0,0),
    available_actions=available_actions_m1,
    labels=labels_m1,
    delta=delta_m1)
vis_m1 = show(m1)


# %% [markdown]
# ## m2
# `m2` is a model that counts the number of `r` up until two, and has a faulty reset button `c` to reset the counter. It only works in 80% of the cases. It only allows `r` if the count is not already 2.

# %%
def available_actions_m2(s):
    if s <= 1:
        return ["r", "c"]
    if s == 2:
        return ["c"]

def delta_m2(s, a):
    if s <= 1:
        if "r" in a:
            return [(1, s+1)]
        elif s == 0:
            return [(1,0)]
        else:
            return [(.8, 0), (.2, s)]
    else:
        return [(.8, 0), (.2, s)]

def labels_m2(s):
    return [str(s)]

m2 = bird.build_bird(
    init=0,
    available_actions=available_actions_m2,
    labels=labels_m2,
    delta=delta_m2)
vis_m2 = show(m2)

# %% [markdown]
# ## Quotient model
# Now we construct the quotient model.

# %%
SYNC = ["r"]

def prob_product(branch1, branch2):
    return [(p1 * p2, (r1, r2)) for p1, r1 in branch1 for p2, r2 in branch2]

def available_actions(s):
    a1 = available_actions_m1(s[0])
    a2 = available_actions_m2(s[1])
    union = set(a1 + a2)
    intersection = set(a1) & set(a2)
    return [x for x in union if x in intersection or x not in SYNC]

def delta(s, a):
    a1 = available_actions_m1(s[0])
    a2 = available_actions_m2(s[1])

    if a in a1 and a in a2:
        return prob_product(delta_m1(s[0], a), delta_m2(s[1], a))
    elif a in a1:
        return [(p, (s_, s[1])) for p,s_ in delta_m1(s[0],a)]
    elif a in a2:
        return [(p, (s[0], s_)) for p,s_ in delta_m2(s[1],a)]
    else:
        return [(1,s)]

def labels(s):
    return labels_m1(s[0]) + labels_m2(s[1])

q = bird.build_bird(
    init=((0,0),0),
    available_actions=available_actions,
    labels=labels,
    delta=delta)
vis_q = show(q)
