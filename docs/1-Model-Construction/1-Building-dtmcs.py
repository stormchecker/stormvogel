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
# # Building DTMCs
# In Stormvogel, a **Discrete Time Markov Chain (DTMC)** consists of:
# * a set of states $S$,
# * an initial state $s_0$,
# * a successor distribution $P(s)$ for every state $s$, i.e., transitions between states $s$ and $s'$, each annotated with a probability.
# * state labels $L(s)$.
#
# In this notebook, we demonstrate how to construct two simple DTMCs from various sources. We show how to construct a model using the `bird` API and the `model` API. Do note that stormvogel supports seemless conversion to and from stormpy. This means that you can also use any way of buidling models that is supported by stormpy. This includes [PRISM](https://www.prismmodelchecker.org/), [JANI](https://www.stormchecker.org/files/BDHHJT17.pdf) and [stormpy's own APIs](https://moves-rwth.github.io/stormpy/). For these, we refer to their respective documentations.
#
# **Note:** unfortunately, the visualisation of the DTMC is not always correct when it is rendered out of view. To re-center, you can simply double-click inside the window.

# %% [markdown]
# ## The Knuth die
# The idea of the Knuth die is to simulate a 6-sided die using coin flips. It is widely used in model checking education.

# %% [markdown]
# ### Bird API
# The Bird API is probably the most intuitive way to create a model. The user defines a `delta` function which maps a state to a distribution of successor states. Its design was inspired by the PRISM format.

# %%
from stormvogel import *

# Create an initial state. States can be of any type. In this case we use integers.
init = 0

TRANSITIONS = \
{0: [(1/2, 1), (1/2, 2)],
 1: [(1/2, 3), (1/2, 4)],
 2: [(1/2, 5), (1/2, 6)],
 3: [(1/2, 1), (1/2, 7)],
 4: [(1/2, 8), (1/2, 9)],
 5: [(1/2, 10), (1/2, 11)],
 6: [(1/2, 2), (1/2, 12)]}

# This user-defined delta function takes as an argument a single state, and returns a
# list of 2-tuples where the first argument is a probability and the second elment is a state (a distribution).
def delta(s):
    if s <= 6:
        return TRANSITIONS[s]
    return [(1, s)]

# Labels is a function that tells the bird API what the label should be for a state.
def labels(s):
    if s <= 6:
        return [str(s)]
    return ["r", str(s-6)]

bird_die = bird.build_bird(
    delta=delta,
    init=init,
    labels=labels,
    modeltype=ModelType.DTMC
)
vis = show(bird_die, layout=Layout("layouts/die.json"))

# %% [markdown]
# ### model API
# This same model can also be constructed using the model API. This API requires adding each state and transition explicitly. This is a lot closer to how the models are represented in stormvogel. The bird API actually uses the model API internally. We generally recommend just using the bird API, but if you need more control, the model API can also be useful.

# %%
# If we use the model API, we need to create all states and transitions explicitly.
# Note the re-use of TRANSITIONS and labels which we defined previously.

# Create a new model with an initial state with id 0.
die_model = stormvogel.model.new_dtmc(create_initial_state=True)
init = die_model.get_initial_state()

# Create all the states (need 12 more to have 13 in total).
for sid in range(1,13):
    die_model.new_state(labels(sid))

# Create all the transitions
for k,v in TRANSITIONS.items():
    state = die_model.get_state_by_id(k) # Get the state with id k
    if k <= 6:
        state.set_choice(
            [(p,die_model.get_state_by_id(sid)) for p,sid in TRANSITIONS[k]])

die_model.add_self_loops() # Of course, we could also add the self-loops explicitly like in the previous example.
vis2 = show(die_model, layout=Layout("layouts/die.json"))


# %% [markdown]
# ## Simple communication
# This example is based on slides by Dave Parker and Ralf Wimmer. It models a very simple communication protocol. This time we use the string type for states.

# %% [markdown]
# ### PGC API

# %%
def delta(s):
    match s:
        case "zero":
            return [(1, "one")]
        case "one":
            return [(0.01, "one"), (0.01, "two"), (0.98, "three")]
        case "two":
            return [(1, "zero")]
        case "three":
            return [(1, "three")]

def labels(s):
    match s:
        case "one":
            return ["try"]
        case "two":
            return ["fail"]
        case "three":
            return ["success"]
        case _:
            return []

bird_commu = bird.build_bird(
    delta=delta,
    init="zero",
    labels=labels,
    modeltype=ModelType.DTMC
)
vis3 = show(bird_commu, layout=Layout("layouts/commu.json"))

# %% [markdown]
# ### model API

# %%
commu_model = stormvogel.model.new_dtmc(create_initial_state=True)
init = die_model.get_initial_state()

TRANSITIONS =\
{0: [(1, 1)],
 1: [(0.01, 1), (0.01, 2), (0.98, 3)],
 2: [(1, 0)],
 3: [(1, 3)]}

LABELS =\
{0: [],
 1: ["try"],
 2: ["fail"],
 3: ["success"]}

for sid in range(1,4):
    commu_model.new_state(LABELS[sid])

for sid in range(0,4):
    state = commu_model.get_state_by_id(sid)
    state.set_choice(
        [(p,die_model.get_state_by_id(sid_)) for p,sid_ in TRANSITIONS[sid]])

vis4 = show(commu_model, layout=Layout("layouts/commu.json"))
