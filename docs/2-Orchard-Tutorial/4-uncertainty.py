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
# # Step Four: MDPs with More Uncertainty
# While MDPs are the standard model to describe processes that are subject to
# both nondeterminism and probabilistic uncertainty, not every process can adequately be represented as an MDP. We briefly show how Storm can go beyond the verification of MDPs by means of **imprecise probabilities** and **partial observations**, while noting that support for such extensions is still actively developed.

# %% [markdown]
# ## Preparation
# We load the general function of the Orchard Stormvogel model. These were defined in *Step One*.

# %%
# Import prism model from the previous step
import stormvogel
import stormpy

from copy import deepcopy
from orchard.orchard_game_stormvogel import (
    Orchard,
    available_actions,
    delta,
    Fruit,
    GameState,
    DiceOutcome,
)
from orchard.orchard_builder import build_simple


# %% [markdown]
# ## Imprecise Probabilities
# Transition probabilities in MDPs are often an abstraction of more complicated
# processes, such as dice rolls in the Orchard game. Instead of precise point estimates for probabilities, Storm supports interval estimates in two flavors: **interval MDPs** and **parametric MDPs**.

# %% [markdown]
# ### Interval MDP
# Interval MDPs (iMDPs) replace the precise probabilities in MDPs with intervals of possible probabilities.
# For the sake of this tutorial, one can consider interval MDPs as a set of MDPs, covering all contained point
# estimates that yield proper successor distributions.

# %% [markdown]
# #### Adapted delta function
# In order to model an interval MDP, we replace the dice roll probabilities from the point estimates $\frac{1}{6}$ to the intervals $[\frac{5}{36}, \frac{7}{36}]$.
# This can be done by changing all occurrences of `fair_dice_prob` in the `delta` function with the interval `Interval(fair_dice_prob-(1/36), fair_dice_prob+(1/36))`. We indicate the change by the comment `NEW` in the following.
# No further change is needed, and the rest of the delta function remains unmodified.


# %%
# The transition function
def delta(state, action):
    if state.game_state() != GameState.NOT_ENDED:
        # Game has ended -> self loop
        return [(1, state)]

    if state.dice is None:
        # Player throws dice and considers outcomes
        outcomes = []
        # Probability of fair dice throw over
        # each fruit type + 1 basket + 1 raven
        fair_dice_prob = 1 / (len(state.trees.keys()) + 2)
        # NEW: adapted probability
        fair_dice_prob = stormvogel.model.Interval(
            fair_dice_prob - (1 / 36), fair_dice_prob + (1 / 36)
        )

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
        return outcomes

    elif state.dice[0] == DiceOutcome.FRUIT:
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
        next_state = deepcopy(state)
        next_state.move_raven()
        next_state.next_round()
        return [(1, next_state)]

    assert False


# %% [markdown]
# We can now build the model as before and obtain the Storm representation.

# %%
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
    max_size=100000,
)

# Convert to stormpy model
orchard_storm = stormvogel.mapping.stormvogel_to_stormpy(orchard)

# %% [markdown]
# Printing the model reveals that it has type IMDP now.

# %%
print(orchard_storm)

# %% [markdown]
# #### Analysis
# Uncertainty in interval MDPs can be resolved in two ways.
#
# First, in the **cooperative** or angelic interpretation, the uncertainty is in favor of the policy. Therefore, if the policy maximizes the probability to win Orchard, the dice probability will also be chosen in order to maximize the probability to win.
#
# Second, in the **robust** or demonic interpretation, the uncertainty is resolved against the policy.
# If the policy maximizes the probability to win, the dice probability will be chosen
# in order to minimize the policy’s probability to win.
#
# We show how to check an interval MDP in the following and start with the cooperative setting, followed by the robust setting.

# %%
# Parse properties
properties = stormpy.parse_properties('Pmax=? [F "PlayersWon"]')
task = stormpy.CheckTask(properties[0].raw_formula)
# Set cooperative resolution mode, alternatively: ROBUST
task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.COOPERATIVE)

# Prepare model checking
env = stormpy.Environment()
# Set VI algorithm to plain VI
env.solver_environment.minmax_solver_environment.method = (
    stormpy.MinMaxMethod.value_iteration
)
# Check model
stormpy_result = stormpy.check_interval_mdp(orchard_storm, task, env)
print(stormpy_result.at(orchard_storm.initial_states[0]))

# %%
# Set robust resolution mode
task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)
# Check model
stormpy_result = stormpy.check_interval_mdp(orchard_storm, task, env)
print(stormpy_result.at(orchard_storm.initial_states[0]))

# %% [markdown]
# In the cooperative setting, the player can achieve a maximal winning probability of $0.7961$, while in
# the robust setting, the player can only achieve a maximal winning probability of
# $0.4315$.
# It turns out that in Orchard, modifying the dice probabilities by $\frac{1}{36} has a much larger influence on the winning probability than the player’s strategy.

# %% [markdown]
# ### Parametric MDP
# The interval MDP for Orchard bounds the dice probabilities within an interval at each state.
# Different values may be optimal at different states, and thus, the dice probabilities may differ per state. If we want to have uncertain but fixed dice probabilities, we use a parametric MDP (pMDP).
#
# We take the simplified two-fruit variant `orchard_simple` and parametrize it as
# follows:
# the parameter `p` describes the probability to roll 🍏 or 🍐, the parameter `q` describes the probability to roll 🧺, and the probability to roll 🐦‍⬛ is described by `1−2p−q`.
# Similar to the iMDP, we can again replace the transition probabilities in the `delta` function with these functions over the parameters.
#

# %%
# Create the correct polynomials (a parser is planned)
one = stormvogel.parametric.Polynomial([])
one.add_term((0,), 1)

params = [f"p{i}" for i in range(2)]

parameters = [stormvogel.parametric.Polynomial([p]) for p in params]
for i in range(2):
    parameters[i].add_term((1,), 1)

parameters.append(stormvogel.parametric.Polynomial(params))
parameters[-1].add_term((0, 1), -1)
parameters[-1].add_term((1, 0), -2)
parameters[-1].add_term((0, 0), 1)


# %%
# Change delta function
def delta_pmc(state, action):
    if state.game_state() != GameState.NOT_ENDED:
        # Game has ended -> self loop
        return [(1, state)]

    if state.dice is None:
        # Player throws dice and considers outcomes
        outcomes = []
        # Total outcomes is number of fruits + 1 for basket + 1 for raven
        total_outcomes = len(state.trees.keys()) + 2
        assert total_outcomes == 4

        i = 0
        # 1. Dice shows fruit
        for fruit in state.trees.keys():
            next_state = deepcopy(state)
            next_state.dice = DiceOutcome.FRUIT, fruit
            # NEW: parametric probabilities
            outcomes.append((parameters[i], next_state))
        i += 1

        # 2. Dice shows basket
        next_state = deepcopy(state)
        next_state.dice = DiceOutcome.BASKET, 0
        # NEW: parametric probabilities
        outcomes.append((parameters[i], next_state))

        # 3. Dice shows raven
        next_state = deepcopy(state)
        next_state.dice = DiceOutcome.RAVEN, None
        # NEW: parametric probabilities
        outcomes.append((parameters[i + 1], next_state))
        assert len(outcomes) == total_outcomes
        return outcomes

    elif state.dice[0] == DiceOutcome.FRUIT:
        # Player picks specified fruit
        fruit = state.dice[1]
        next_state = deepcopy(state)
        next_state.pick_fruit(fruit)
        next_state.next_round()
        return [(one, next_state)]

    elif state.dice[0] == DiceOutcome.BASKET:
        assert action.startswith("choose")
        # Player chooses fruit specified by action
        fruit = Fruit[action.removeprefix("choose")]
        next_state = deepcopy(state)
        next_state.pick_fruit(fruit)
        next_state.next_round()
        return [(one, next_state)]

    elif state.dice[0] == DiceOutcome.RAVEN:
        next_state = deepcopy(state)
        next_state.move_raven()
        next_state.next_round()
        return [(one, next_state)]

    assert False


# %% [markdown]
# We can again build the simple Orchard model which is now parametric.

# %%
init_small = Orchard([Fruit.APPLE, Fruit.CHERRY], num_fruits=2, raven_distance=2)
orchard_pmc = stormvogel.bird.build_bird(
    modeltype=stormvogel.ModelType.MDP,
    init=init_small,
    available_actions=available_actions,
    delta=delta_pmc,
    labels=labels_full,
    max_size=100000,
)

# Convert to stormpy model
orchard_stormpy_pmdp = stormvogel.mapping.stormvogel_to_stormpy(orchard_pmc)

# %% [markdown]
# After building the model, we apply the optimal winning policy from the non-parameterized game on the pMDP and obtain the induced parametric Markov chain (pMC).

# %%
# Get scheduler from MDP
orchard_mdp_stormpy = stormvogel.mapping.stormvogel_to_stormpy(build_simple())
result = stormpy.model_checking(
    orchard_mdp_stormpy, properties[0].raw_formula, extract_scheduler=True
)

# Apply scheduler on pMDP
scheduler = result.scheduler.cast_to_parametric_datatype()
orchard_stormpy_pmc = orchard_stormpy_pmdp.apply_scheduler(scheduler)

# %% [markdown]
# Last, we computing the winning probability on the pMC. This yields a rationnal function over parameters `p` and `q`.

# %%
# Perform model checking
properties2 = stormpy.parse_properties('P=? [F "PlayersWon"]')
stormpy_result = stormpy.model_checking(orchard_stormpy_pmc, properties2[0].raw_formula)
solution_function = stormpy_result.at(0)
print(solution_function)

# %% [markdown]
# This rational function corresponds to the one in Fig. 5 in the tutorial paper. Parameter `r_2` is `p` and parameter `r_1` is `q`.

# %% [markdown]
# ## Partially Observable MDPs
# In MDP verification, we consider maximizing over all policies. In particular, the policy has
# perfect information about the current state, which influences the choice. In many
# (cyberphysical or distributed) systems, the policy should only depend on states
# that are known by the agent.
#
# Similarly, in the Orchard game, the state of the
# trees may not be visible to the players, however, we do assume that we know
# which types of fruit can still be picked. To model this setting faithfully, the policy
# should not depend on the state, but instead on a set of observations: leading to
# a **partially observable MDP (POMDP)**.

# %% [markdown]
# We adapted the Prism specification of the Orchard game and introduced observations.
# The resulting specification can be found in `orchard/orchard_pomdp.pm`.
#
# In the following, we load the POMDP.

# %%
import stormpy
import stormpy.pomdp

prism_program = stormpy.parse_prism_program("orchard/orchard_pomdp.pm")
formula_str = 'Pmax=? [!"RavenWon" U "PlayersWon"]'
properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
prism_program, properties = stormpy.preprocess_symbolic_input(
    prism_program, properties, ""
)
prism_program = prism_program.as_prism_program()
options = stormpy.BuilderOptions([p.raw_formula for p in properties])
options.set_build_state_valuations()
options.set_build_choice_labels()
pomdp = stormpy.build_model(prism_program, properties)
pomdp = stormpy.pomdp.make_canonic(pomdp)
print(pomdp)

# %% [markdown]
# We see that the POMDP now contains information regarding the number of
# different observations ($546$ observations), as well as information regarding the action names for the
# choices in the model (as they matter semantically).
# The semantics of a POMDP can be given in terms of a **belief MDP**, which we can partially explore and verify.

# %%
belexpl_options = stormpy.pomdp.BeliefExplorationModelCheckerOptionsDouble(True, True)
belexpl_options.use_clipping = False
belexpl_options.refine = True

belmc = stormpy.pomdp.BeliefExplorationModelCheckerDouble(pomdp, belexpl_options)
result = belmc.check(properties[0].raw_formula, [])
print(f"Result in: [{result.lower_bound}, {result.upper_bound}]")

# %% [markdown]
# First, note that the result approximates the true optimum from below and above.
# The result here is tight, as the underlying belief MDP is sufficiently simple. The imprecision stems from the overall precision which is set to $10^{-6}$ by default.
#
# Second, note that the result coincides with the fully observable MDP case! That
# is correct as a policy can track how much fruit is available from every type.

# %% [markdown]
# We can consider a modified Orchard game, where another player randomly steals fruit before the game starts.
# This is modeled in `orchard/orchard_pomdp_steal.pm`.

# %%
# Load model with stealing
prism_program = stormpy.parse_prism_program("orchard/orchard_pomdp_steal.pm")
formula_str = 'Pmax=? [!"RavenWon" U "PlayersWon"]'
properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
prism_program, properties = stormpy.preprocess_symbolic_input(
    prism_program, properties, ""
)
prism_program = prism_program.as_prism_program()
options = stormpy.BuilderOptions([p.raw_formula for p in properties])
options.set_build_state_valuations()
options.set_build_choice_labels()
pomdp_steal = stormpy.build_model(prism_program, properties)
pomdp_steal = stormpy.pomdp.make_canonic(pomdp_steal)
print(pomdp_steal)

# %% [markdown]
# Afterwards, we can analyze the fully observable model and the partially observable model.

# %%
# Fully observable
mdp_res = stormpy.model_checking(
    pomdp_steal, properties[0], force_fully_observable=True
)
print(mdp_res.at(pomdp_steal.initial_states[0]))

# %%
belexpl_options = stormpy.pomdp.BeliefExplorationModelCheckerOptionsDouble(True, True)
belexpl_options.use_clipping = False
belexpl_options.refine = True

belmc = stormpy.pomdp.BeliefExplorationModelCheckerDouble(pomdp_steal, belexpl_options)
result = belmc.check(properties[0].raw_formula, [])
print(f"Result in: [{result.lower_bound}, {result.upper_bound}]")

# %% [markdown]
# We see that stealing fruits makes the game easier to win, but now an observation-based policy will perform worse than the state-based policies.
