# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step One: Modeling

# %% [markdown]
# ## The Orchard Game
# The *Orchard game* (also called *Obstgarten*, *Le verger*, *Boomgaard*, *El Frutal*, *Første Frukthage*) is a simple children’s game. In the following, we model the
# First Orchard game variant based on the official rule book [available online](https://cdn.hff.de/image/upload/v1/22/30/28/14/22302814.pdf).
#
# The game is played cooperatively.
# There are four types of **fruit**: **apples** 🍏, **pears** 🍐, **cherries** 🍒 and **plums** 🍇. (We use the grape symbol here as there is no plum symbol).
# For each type of fruit, there is a **tree** with four pieces of this fruit.
# There is a **basket** 🧺 for collecting fruit.
# As antagonist, there is a **raven** 🐦‍⬛ which reaches the orchard after five steps.
#
# The goal of the game is to collect all fruit before the raven reaches the orchard. In each round, a player throws a six-sided dice and performs an action depending on its outcome:
# - If the outcome is a type of fruit 🍏🍐🍒🍇, then the player picks a piece of this type from the tree and places it in the basket. If no fruit is left on the tree, the player cannot pick anything.
# - If the outcome is the fruit basket 🧺, then the player may pick any fruit.
# - If the outcome is the raven 🐦‍⬛, the raven moves one step towards the orchard.
#
# The players win iff they collect all fruit before the raven arrives at the orchard.

# %% [markdown]
# ## The Orchard Model
# Before starting the actual modeling, we make some general modeling decisions.
# First, we decide which parts are relevant for the model. In the Orchard game,
# we need to keep track of the remaining fruit on the trees and the position of the
# raven. As the game is cooperative, it is irrelevant how many players are playing
# and we assume one player. We also do not need to keep track of the basket as
# only the remaining fruit on the trees are relevant for the winning condition.
#
# In our Orchard model, we divide each round into two distinct phases: first
# throwing the dice and then acting on the outcome. This allows to separate the
# probabilistic dice throw from the player’s (potentially nondeterministic) action.
#
# Lastly, to facilitate analysis of different variants of the game, we parameterize
# the model in terms of fruit types, number of fruit and distance of the raven.

# %% [markdown]
# ## Data structures with Stormvogel
# We start by importing the stormvogel library and additional Python libraries.

# %%
# Imports
import stormvogel

from enum import Enum
from copy import deepcopy


# %% [markdown]
# We start the modelling by defining some general data structures for the Orchard game, for the different types of fruit `Fruit`, for the possible outcomes of the dice `DiceOutcome` and for the game state `GameState`.


# %%
# General data structures
class Fruit(Enum):
    APPLE = "🍏"
    PEAR = "🍐"
    CHERRY = "🍒"
    PLUM = "🍇"  # Use grape as there is no Emoji for plum

    def __str__(self):
        return self.value


class DiceOutcome(Enum):
    FRUIT = "🍉"
    BASKET = "🧺"
    RAVEN = "🐦‍⬛"

    def __str__(self):
        return self.value


class GameState(Enum):
    NOT_ENDED = 0
    PLAYERS_WON = 1
    RAVEN_WON = 2


# %% [markdown]
# We introduce a class `Orchard` which represents the current state of the game.
# This class inherits from the Stormvogel `State` class.
# The Orchard object is initialized with a list of configuration parameters such as the considered types of fruit `fruit_types`, the number of fruit per tree `num_fruits`, and the distance of the raven `raven_distance`.
# The game initializes the variable `trees` which keeps track of the remaining number of fruit per tree. It also keeps
# track of the outcome of the `dice` which can be either 🧺, 🐦‍⬛ or a fruit.
# In case of a fruit, the second entry of the dice tuple denotes the specific type of fruit which was thrown.
# Value `None` represents that the dice needs to be thrown first.


# %%
# Main class for the orchard game
class Orchard(stormvogel.bird.State):
    def __init__(self, fruit_types, num_fruits, raven_distance):
        self.trees = {fruit: num_fruits for fruit in fruit_types}
        self.raven = raven_distance
        self.dice = None

    def game_state(self):
        if all(self.is_tree_empty(tree) for tree in self.trees.keys()):
            assert self.raven > 0
            return GameState.PLAYERS_WON
        elif self.raven == 0:
            return GameState.RAVEN_WON
        else:
            return GameState.NOT_ENDED

    def is_tree_empty(self, tree):
        return self.trees[tree] == 0

    def pick_fruit(self, fruit):
        if self.trees[fruit] > 0:
            self.trees[fruit] -= 1

    def next_round(self):
        self.dice = None

    def move_raven(self):
        self.raven -= 1

    def __hash__(self):
        trees = [hash((f, n)) for f, n in self.trees.items()]
        return hash((tuple(trees), self.raven, self.dice))

    def label(self):
        if self.dice is None:
            # Output game state
            return str(self)
        else:
            if self.dice[0] == DiceOutcome.FRUIT:
                return "🎲" + str(self.dice[1])
            else:
                return "🎲" + str(self.dice[0])

    def __str__(self):
        s = ", ".join(["{}{}".format(n, f) for f, n in self.trees.items()])
        s += ", {}←{}".format(self.raven, DiceOutcome.RAVEN)
        return s


# %% [markdown]
# The Orchard class also defines various functions for ease of modeling.
# For instance, `game_state` returns whether the game is ongoing or who has won.
# Function `is_tree_empty` returns whether the tree for a given fruit type is contains no fruits anymore.
# Function `pick_fruit` picks the given fruit from the tree (if available).
# Function `next_round` resets the dice for the next round, `move_raven` moves the raven.
# Modeling these functions separately also yields a modular design which allows to easily modify specific behavior, such as the movement of the raven.

# %% [markdown]
# An important aspect of the Orchard class is the hash function `__hash__`.
# The hash combines the number of left-over fruit per tree (via `self.trees.items()`), the position of the raven (`self.raven`) and the dice outcome (`self.dice`).
# All other information, for example whose turn it is, is deemed irrelevant to distinguish states.
# The hash function is crucial in reducing the size of the state space by only focusing on the relevant parts.
# If not explicitly given, the bird API hashes all members of a class, which could increase the state space.

# %% [markdown]
# The function `label` is used to either output the current game state or the outcome of the dice.
# Lastly, we add a custom string representation `__str__`to nicely represent states.

# %% [markdown]
# ## Defining the Orchard MDP
# An MDP is represented by the tuple $\mathcal{M} = (S, s_0, \textit{Act}, \mathbf{P}, AP, L)$ where
# - $S$ is a finite set of **states**
# - $s_0 \in S$ is the **initial state**
# - $\textit{Act}$ is a finite set of **actions**
# - $\mathbf{P}: S \times Act \nrightarrow  \mathsf{Distr}(S)$ is the partial **transition function** mapping state-action pairs to distributions over successors
# - $AP$ is a finite set of **atomic propositions**
# - $L: S \to 2^{AP}$ is the **labeling function**

# %% [markdown]
# In order to create the MDP model of the Orchard game in Stormvogel, we need to provide all ingredients of the tuple in the following.

# %% [markdown]
# ### Initial state
# The initial state $s_0$ is given by the initialization of our `Orchard` class.
# At first, we use a smaller configuration of the game with only two different types of fruit, two pieces of fruit per tree, and a distance of two for the raven.

# %% editable=true slideshow={"slide_type": ""}
# Initialize the game
init_small = Orchard([Fruit.APPLE, Fruit.CHERRY], num_fruits=2, raven_distance=2)

# %% [markdown]
# Printing the initial state of the simplified game shows that there are two apples and two cherries on the trees, and the raven is two steps away from its goal.

# %%
print(init_small)


# %% [markdown]
# ### Available actions
# The transition function returning successor states requires information which actions are enabled in each state. We define $\textit{Act}(s)$ for any state $s$.
# In the Orchard game, we use the following actions:
# - `gameEnded`: the only action available once the game has ended. This action represents the self-loop for such final states.
# - `nextRound`: the dice must be thrown again.
# - `pickF`: the dice outcome is one of 🍏,🍐,🍒,🍇 and the player picks the corresponding fruit $F$ from the tree.
# - `chooseF`: the dice outcome is 🧺 and a fruit (🍏,🍐,🍒,🍇) is chosen.
# - `moveRaven`: the dice outcome is 🐦‍⬛ and the raven should be moved next.


# %% editable=true slideshow={"slide_type": ""}
# Define available actions
def available_actions(state):
    if state.game_state() != GameState.NOT_ENDED:
        return ["gameEnded"]
    if state.dice is None:
        return ["nextRound"]
    if state.dice[0] == DiceOutcome.FRUIT:
        return ["pick{}".format(state.dice[1].name)]
    if state.dice[0] == DiceOutcome.BASKET:
        available_fruits = []
        # Choice over available fruits
        for fruit in state.trees.keys():
            if not state.is_tree_empty(fruit):
                available_fruits.append(fruit)
        return ["choose{}".format(fruit.name) for fruit in available_fruits]
    if state.dice[0] == DiceOutcome.RAVEN:
        return ["moveRaven"]
    assert False


# %% [markdown]
# For most states, a single action is available.
# The only exception is dice outcome 🧺, for which we list all fruit which are still available for choosing by the player.

# %% [markdown]
# ### Transition function
# Stormvogel defines the transition function $\mathbf{P}$ via function `delta` which takes as arguments a state and an available action, and returns
# the distribution over successor states as a sparse list of nonzero transition probabilities for different target states.


# %% editable=true slideshow={"slide_type": ""}
# The transition function
def delta(state, action):
    if state.game_state() != GameState.NOT_ENDED:
        assert action == "gameEnded"
        # Game has ended -> self loop
        return [(1, state)]

    if state.dice is None:
        assert action == "nextRound"
        # Player throws dice and considers outcomes
        outcomes = []
        # Probability of fair dice throw over
        # each fruit type + 1 basket + 1 raven
        fair_dice_prob = 1 / (len(state.trees.keys()) + 2)

        # 1. Dice shows fruit
        for fruit in state.trees.keys():
            next_state = deepcopy(state)
            next_state.dice = DiceOutcome.FRUIT, fruit
            outcomes.append((fair_dice_prob, next_state))

        # 2. Dice shows basket
        next_state = deepcopy(state)
        next_state.dice = DiceOutcome.BASKET, None
        outcomes.append((fair_dice_prob, next_state))

        # 3. Dice shows raven
        next_state = deepcopy(state)
        next_state.dice = DiceOutcome.RAVEN, None
        outcomes.append((fair_dice_prob, next_state))

        assert sum([o[0] for o in outcomes]) == 1
        return outcomes

    elif state.dice[0] == DiceOutcome.FRUIT:
        assert action.startswith("pick")
        # Player picks specified fruit
        fruit = state.dice[1]
        next_state = deepcopy(state)
        next_state.pick_fruit(fruit)
        next_state.next_round()
        return [(1, next_state)]

    elif state.dice[0] == DiceOutcome.BASKET:
        assert action.startswith("choose")
        # Player chooses fruit specified by action
        fruit = Fruit[action.removeprefix("choose")]
        next_state = deepcopy(state)
        next_state.pick_fruit(fruit)
        next_state.next_round()
        return [(1, next_state)]

    elif state.dice[0] == DiceOutcome.RAVEN:
        assert action == "moveRaven"
        next_state = deepcopy(state)
        next_state.move_raven()
        next_state.next_round()
        return [(1, next_state)]

    assert False


# %% [markdown]
# If the game has ended, we add a self-loop to the same state with probability one.
# Otherwise, we perform each round in two phases:
# first throwing the dice and then acting on the outcome. These are two separate
# transitions in the model.
#
# In the first phase, the dice outcome is `None` and the player needs to throw
# the dice. We need to consider the outcomes for the different fruit types plus two
# additional outcomes 🧺 and 🐦‍⬛. Each outcome has the same uniform probability
# `fair_dice_prob`, which is `1 / (len(state.trees.keys()) + 2)`.
# The rest of the first phase creates the different successor states depending on the dice outcomes.
# To this end, we create the next_state as a copy of the current state, and set the dice outcome
# `next_state.dice`.
# Afterwards, we store the successor state together with the
# corresponding probability in the list outcomes of successor states.
#
# The remainder of the `delta` function handles the second phase, where the player
# acts depending on the dice outcome `state.dice`.
# If the dice outcome is a fruit,
# the player performs `pick_fruit(fruit)` on the next_state and then finishes
# the turn with `next_round`.
# If the dice outcome is 🧺, the player needs to choose
# a fruit. This choice is modeled by the different available actions `chooseF` in this
# state. Based on the chosen `action` given as argument to the `delta` function, we
# can extract the chosen fruit. For example, action name `chooseAPPLE` corresponds
# to 🍏. In the successor state, we pick the chosen fruit and then continue with
# the next round.
#
# Lastly, if dice outcome is 🐦‍⬛, the raven moves.

# %% [markdown]
# ### Labeling
# We use a labeling $L$ to associate atomic propositions $AP$ to states
# which satisfy them. These labels help identify states which fulfill a certain property which is relevant to our analysis.
#
# In Stormvogel, the function `labels()` returns a list of atomic propositions
# per state. As we are mainly interested in the winner of a game, we introduce
# two labels `"PlayersWon"` and `"RavenWon'` which mark each state with the party who won.
# We also label each state with the string representation of the game state (`state.label()`) to help understanding and investigating the resulting model as the internals of a state become visible.
# For scalable analysis, such kind of internal information is typically omitted in the final version of a model.


# %%
# Add labels for game state
def labels(state):
    labels = [state.label()]
    if state.game_state() == GameState.PLAYERS_WON:
        labels.append("PlayersWon")
    elif state.game_state() == GameState.RAVEN_WON:
        labels.append("RavenWon")
    return labels


# %% [markdown]
# ### Rewards
# Lastly, we specify **rewards** (also called costs).
# A model can have multiple reward structures that each associate rewards to states (or to state-action pairs).
# Visiting a state or action accumulates the associated rewards.
# As example, we define a reward that counts the number of rounds.
# Every state in which the dice is thrown (`state.dice is None`) is associated with one additional round of the game. All other states have reward zero.


# %% editable=true slideshow={"slide_type": ""}
# Reward function
def rewards(state):
    if state.game_state() == GameState.NOT_ENDED:
        if state.dice is None:
            return {"rounds": 1}
    return {"rounds": 0}


# %% [markdown]
# ## Building the models
# All ingredients for the model have been specified and we can finally build the MDP model for the Orchard game.
# The function specifies the type of model (MDP here) and the MDP tuple entries we defined before.

# %% editable=true slideshow={"slide_type": ""}
# Build the simple orchard model
orchard_simple = stormvogel.bird.build_bird(
    modeltype=stormvogel.ModelType.MDP,
    init=init_small,
    available_actions=available_actions,
    delta=delta,
    labels=labels,
    rewards=rewards,
)

print(orchard_simple.summary())

# %% [markdown]
# We see that we built an MDP model with 90 states.

# %% [markdown]
# We can also easily build the full model by configuring the initial state `init_game` accordingly and calling `build_bird()` as before.
# Note that we need to increase the `max_size` of the explored state space as the resulting model has around $22,500$ states.

# %%
# Build the full model
init_game = Orchard(
    [Fruit.APPLE, Fruit.CHERRY, Fruit.PEAR, Fruit.PLUM], num_fruits=4, raven_distance=5
)


# For the full model, we only set the relevant labels for the winning conditions
# and do not expose the internal state information
def labels_full(state):
    labels = []
    if state.game_state() == GameState.PLAYERS_WON:
        labels.append("PlayersWon")
    elif state.game_state() == GameState.RAVEN_WON:
        labels.append("RavenWon")
    return labels


orchard = stormvogel.bird.build_bird(
    modeltype=stormvogel.ModelType.MDP,
    init=init_game,
    available_actions=available_actions,
    delta=delta,
    labels=labels_full,
    rewards=rewards,
    max_size=100000,
)

print(orchard.summary())

# %% [markdown]
# Some of the analyses in the following require a representation of the Orchard model in Storm data structures.
# We can simply translate the model via `stormvogel_to_stormpy`.

# %%
# Stormpy model
import stormpy

orchard_storm = stormvogel.mapping.stormvogel_to_stormpy(orchard)
print(orchard_storm)

# %% [markdown]
# ## Defining the MDP in the Prism language
# We showed how to create an MDP model via Stormvogel.
# Storm also supports creating models from other input formats such as the [Prism language](https://www.prismmodelchecker.org/manual/ThePRISMLanguage/Introduction) or the [JANI interchange format](https://jani-spec.org/).
# Here, we also provide a Prism model for our Orchard game. The specification is given in `orchard/orchard_stormvogel.pm` and we print it here.

# %%
# Print the Prism model
# !cat orchard/orchard_stormvogel.pm

# %% [markdown]
# The Prism specification follows the same structure as the Stormvogel model.
# It can be configured via global constants such as `NUM_FRUIT` and `DISTANCE_RAVEN`.
# The current game state is represented by variables such as `raven` or `apple`.
#
# The Prism specification can be built via stormpy.

# %%
# Build the prism model
prism_program = stormpy.parse_prism_program("orchard/orchard_stormvogel.pm")
constants = "NUM_FRUIT=4, DISTANCE_RAVEN=5"
prism_program = stormpy.preprocess_symbolic_input(prism_program, [], constants)[
    0
].as_prism_program()
options = stormpy.BuilderOptions()
options.set_build_state_valuations()
options.set_build_choice_labels()
options.set_build_with_choice_origins()
orchard_prism = stormpy.build_sparse_model_with_options(prism_program, options)
print(
    "Model with {} states and {} transitions".format(
        orchard_prism.nr_states, orchard_prism.nr_transitions
    )
)

# %% [markdown]
# We can see that the model from the Prism specification has the same number of states than the model defined via Stormvogel.

# %% [markdown]
# ## Model inspection
# Before applying model checking, we inspect the model to spot potential modelling issues.
# Stormvogel visualizes the model with `stormvogel.show(orchard_simple)`.
#
# The interactive visualization is based on JavaScript and can be seen directly in this notebook when executing the command below.
# Note that the visualization might take a couple of seconds to load.
#
# Note that a potential warning of the form `Test request failed` can be safely ignored.

# %% editable=true slideshow={"slide_type": ""}
# Visualize
vis = stormvogel.show(orchard_simple)

# %% [markdown]
# One can zoom in/out as usual, and move the viewpoint by dragging the view.
# Elements are automatically arranged but can also be dragged.
#
# States are depicted as ellipses, actions by blue boxes and transitions by arrows.
# Labels are given inside the states and actions, and the rewards are indicated by the € symbol.
#
# The initial state should be visible on the right hand side and is the only state with the additional label `init`.
# Starting from this initial state, the only possible action is `nextRound`, which corresponds to throwing the dice. It has reward one.
# The four outcomes are depicted by the four successor states with symbol 🎲 and are each
# reached with transition probability $\frac{1}{4}$.
# Afterwards, the corresponding action can be performed. For example, from state 🎲🍏, action `pickAPPLE` is possible and
# leads to a state 1🍏, 2🍒, 2←🐦‍⬛ where only one apple is remaining.
