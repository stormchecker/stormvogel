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
# # Step Three: Working with Policies
# The use of modern probabilistic model checking increasingly asks for diagnostic information, for example the policies that optimize the objectives that are being checked.
# In the previous *Step Two*, we already saw that Stormvogel can visualize the model checking result and also the chosen actions in each state. However, this approach is only feasible for smaller state spaces.
# In this notebook, we consider a more general approach of extracting policies, the analysis of such policies with respect to alternative objectives, and the extraction of policies that can be concisely represented.

# %% [markdown]
# ## Preparation
# As in the previous notebook, we start by loading the model.
# In this notebook, we will only use the Prism specification `orchard_prism` of the full Orchard game.
# While variable `orchard_prism` stores the MDP model, the variable `prism_program` stores the Prism specification of the Orchard game.

# %%
# Import prism model from the previous step
import stormvogel
import stormpy

from orchard.orchard_builder import build_prism

# %%
# We use the prism encoding in this part of the tutorial.
orchard_prism, prism_program = build_prism()
# We are primarily interested in the probability of winning.
formula = stormpy.parse_properties('Pmax=? [F "PlayersWon"]')[0]


# %% [markdown]
# Similarly to before, we use the helper function `model_check`.


# %%
# Helper function
def model_check(model, prop):
    formula = stormpy.parse_properties(prop)[0]
    result = stormpy.model_checking(model, formula, only_initial_states=True)
    return result.at(model.initial_states[0])


# %% [markdown]
# ## Computing Policies
# We are interested in the optimal of winning the Orchard game.
# First, we inform Storm that we want to compute an optimal policy as a by-product of the model checking call, by adding the argument `extract_schedulers=True`.
# Computing policies can induce overhead due to the additional bookkeeping: we also note that this bookkeeping is not implemented for all types of properties.

# %%
result = stormpy.model_checking(orchard_prism, formula, extract_scheduler=True)

# %% [markdown]
# The output of the model checking call is a quantitative `result`, which now also holds a policy (also called scheduler).

# %%
string_scheduler = str(result.scheduler)

# %% [markdown]
# Policies for maximal reachability probabilities as in the `formula` are memoryless and
# deterministic: a mapping from states to actions.
# The Storm representation mildly deviates: it represents for every state $s$ (index) the local index $x$ of the choice. These local indices encode the $x$’th choice for state $s$, using some internal ordering.
# Note that two choices in different states can yield the same action.
#
# Note that the resulting policy representation has more than $22k$ lines; one line for each of the over $22k$ states of the model.
# We therefore only print the first 20 lines as given in `max_lines`.

# %%
import os

max_lines = 20
print(os.linesep.join(string_scheduler.split(os.linesep)[:max_lines]))

# %% [markdown]
# The policy can also be queried to understand for every state which action is selected.
# We use the high-level state information present in `orchard_prism`, and obtain a more informative representation of the policy.
#
# Again, we stop the output after `max_lines`.

# %%
i = 0
for state in orchard_prism.states:
    choice = result.scheduler.get_choice(state)
    action_index = choice.get_deterministic_choice()
    action = state.actions[action_index]
    print(f"In state {state.valuations} choose action {action.labels}")
    i += 1
    if i >= max_lines:
        break

# %% [markdown]
# We are mostly interested in the choices `chooseF` as these are the player's choices. In all other states, only one action can be chosen anyway.
# We therefore only output these types of actions in the following.

# %%
i = 0
for state in orchard_prism.states:
    choice = result.scheduler.get_choice(state)
    action_index = choice.get_deterministic_choice()
    action = state.actions[action_index]
    action_name = next(iter(action.labels))
    if action_name.startswith("choose"):
        print(f"In state {state.valuations} choose action {action_name}")
        i += 1
        if i >= max_lines:
            break

# %% [markdown]
# We see that initially, when all trees are full (`apple=4`, `pear=4`, etc.), the player chooses 🍏 for the dice outcome 🧺.
# If for one apple was already picked (`apple=3`), the optimal strategy chooses 🍐 instead.
#
# Overall, we can see that the optimal policy chooses the type of fruit for which the most fruits are still available in the trees.

# %% [markdown]
# ## Analyzing the Induced Submodel
# After obtaining a policy, a standard question is to analyze the MDP and policy with respect to additional properties.
#
# For example, we may want to calculate the probability to collect all 🍒 before the raven arrives using the current policy and contrast this probability with a policy that optimizes collecting 🍒.
# To support analyzing policies, we can create the induced Markov chain $\mathcal{M}^\sigma$ by applying policy $\sigma$ on MDP $\mathcal{M}$.
#
# In our example, we apply the policy we previously obtained.
# Afterwards, we compute the maximal probability of picking all cherries: once on the induced model (which was optimized for winning) and once on the global model.

# %%
# Get induced model
induced = orchard_prism.apply_scheduler(result.scheduler, True)
# Analysis
all_cherries = 'Pmax=? [F "AllCherriesPicked"]'
print(f"Prob for fixed: {model_check(induced, all_cherries)}")
print(f"Prob for optimal: {model_check(orchard_prism, all_cherries)}")


# %% [markdown]
# We see that the policy optimized for collecting 🍒 yields a higher probability of picking all 🍒 than the submodel induced by the overall winning strategy.

# %% [markdown]
# ## Compact Policies
# The policy in the previous part was represented as an explicit list.
# There are various ways to compress a given policy: Via decision tree learning (available via the [dtControl tool](https://dtcontrol.model.in.tum.de/) one can create heuristically small trees for a policy, whereas
# dtMap (via the [PAYNT tool](https://github.com/randriu/synthesis)) allows to find a minimal decision tree for
# a given policy. Both tools allow interfacing with Storm.
# In the following, we showcase the use of dtMap.

# %% [markdown]
# Before we can create a compact policy, let us note that the policy above is not compactly representatable as a decision tree over linear predicates using the variables in the prism program (including the amount of fruit left). In this tutorial, we manually set the predicates we want to use.
# Here, we use `most_F` to indicate that fruit type `F` has the most fruits still left on the trees.


# %%
def _declare_extra_orchard_variables(manager):
    apple_variable: stormpy.Variable = manager.get_variable("apple")
    plum_variable: stormpy.Variable = manager.get_variable("plum")
    pear_variable: stormpy.Variable = manager.get_variable("pear")
    cherry_variable: stormpy.Variable = manager.get_variable("cherry")
    # The else cases below are only relevant for rerunning this function multiple times.
    if not manager.has_variable("most_apples"):
        most_apples_variable = manager.create_boolean_variable("most_apples")
    else:
        most_apples_variable = manager.get_variable("most_apples")
    if not manager.has_variable("most_pears"):
        most_pears_variable = manager.create_boolean_variable("most_pears")
    else:
        most_pears_variable = manager.get_variable("most_pears")
    if not manager.has_variable("most_plums"):
        most_plums_variable = manager.create_boolean_variable("most_plums")
    else:
        most_plums_variable = manager.get_variable("most_plums")
    if not manager.has_variable("most_cherries"):
        most_cherries_variable = manager.create_boolean_variable("most_cherries")
    else:
        most_cherries_variable = manager.get_variable("most_cherries")

    apple_variable_expression = apple_variable.get_expression()
    plum_variable_expression = plum_variable.get_expression()
    pear_variable_expression = pear_variable.get_expression()
    cherry_variable_expression = cherry_variable.get_expression()
    # Add the extra definitions
    extra_definitions = []
    extra_definitions.append(
        (
            most_apples_variable,
            stormpy.Expression.Conjunction(
                [
                    stormpy.Expression.Geq(
                        apple_variable_expression, plum_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        apple_variable_expression, cherry_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        apple_variable_expression, pear_variable_expression
                    ),
                ]
            ),
        )
    )
    extra_definitions.append(
        (
            most_pears_variable,
            stormpy.Expression.Conjunction(
                [
                    stormpy.Expression.Geq(
                        pear_variable_expression, plum_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        pear_variable_expression, cherry_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        pear_variable_expression, apple_variable_expression
                    ),
                ]
            ),
        )
    )
    extra_definitions.append(
        (
            most_plums_variable,
            stormpy.Expression.Conjunction(
                [
                    stormpy.Expression.Geq(
                        plum_variable_expression, pear_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        plum_variable_expression, cherry_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        plum_variable_expression, apple_variable_expression
                    ),
                ]
            ),
        )
    )
    extra_definitions.append(
        (
            most_cherries_variable,
            stormpy.Expression.Conjunction(
                [
                    stormpy.Expression.Geq(
                        cherry_variable_expression, pear_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        cherry_variable_expression, plum_variable_expression
                    ),
                    stormpy.Expression.Geq(
                        cherry_variable_expression, apple_variable_expression
                    ),
                ]
            ),
        )
    )
    return extra_definitions


# %% [markdown]
# Specifically, we build new state valuations that assign variable assignments to every state. The following code achieves this. While this can be modified, we suggest to not change the settings here for the tutorial.

# %%
# We allow a few options here.
add_additional_definitions = True
"""add_additional_definitions adds expressions saying that one fruit is among the types of fruit that we have most."""
maintain_old_valuations = False
"""maintain_old_valuations preserves the old variables and their assignments"""
copy_fruit_amounts = False
"""copy_fruit_amounts preserves the amount of fruit of every type, useful if we are not maintaining the old valuations."""
assert not maintain_old_valuations or not copy_fruit_amounts

# Create the transformer.
svt = stormpy.StateValuationTransformer(orchard_prism.state_valuations)
# Add extra Boolean definitions:
if add_additional_definitions:
    extra_definitions = _declare_extra_orchard_variables(
        prism_program.expression_manager
    )
    for v, definition in extra_definitions:
        svt.add_boolean_expression(v, definition)
# Add integer definitions. Here, we consider copies of the number of items, only useful we do not maintain the old variables.
if copy_fruit_amounts:
    manager = prism_program.expression_manager
    svt.add_integer_expression(
        manager.create_integer_variable("nr_apples"),
        manager.get_variable("apple").get_expression(),
    )
    svt.add_integer_expression(
        manager.create_integer_variable("nr_plums"),
        manager.get_variable("plum").get_expression(),
    )
    svt.add_integer_expression(
        manager.create_integer_variable("nr_cherries"),
        manager.get_variable("cherry").get_expression(),
    )
    svt.add_integer_expression(
        manager.create_integer_variable("nr_pears"),
        manager.get_variable("pear").get_expression(),
    )
# Create a new MDP.
new_mdp = stormpy.set_state_valuations(
    orchard_prism, svt.build(maintain_old_valuations)
)

# %% [markdown]
# We are now ready to run synthesis of decision tree policies. We use the tool PAYNT.

# %%
import paynt

# %% [markdown]
# We distinguish two versions of decision tree synthesis:
#  - matching/mapping a given policy into a concise format.
#  - finding a decision tree policy that achieves the goal.
#
# We define the synthesis task with these two hyperparameters.

# %%
tree_depth = 3
"""The maximal depth of the policy"""
match_fixed_policy = False
"""Whether we should match a given policy (an optimal one found by storm)"""
pass

# %% [markdown]
# We are now ready to run the code. Note that this may take a while. For `tree_depth` 3 and the `match_fixed_policy` set to `False`, the code should run in under two minutes, assuming that we have added additional definitions, that we did not maintain the old valuations, and that we did not copy the fruit amounts.
# If you want to play with settings, we suggest to use a smaller version of the orchard model by setting the amount of fruit and the distance of the raven to smaller constants.

# %%
# We run model checking once more, and optionally extract a scheduler.
mc_result = stormpy.model_checking(
    new_mdp, formula, extract_scheduler=match_fixed_policy
)
print(f"Overall optimal result: {mc_result.at(new_mdp.initial_states[0])}")
# Paynt considers synthesis via annotated MDPs called coloured MDPs.
# For synthesis of decision trees, we use paynt.dt.
colored_mdp_factory = paynt.dt.DtColoredMdpFactory(new_mdp)
# We now set the task as specified in the hyperparameters above.
task = paynt.dt.DtTask([formula], tree_depth)
if match_fixed_policy:
    task.set_scheduler_to_map(mc_result.scheduler)
# With the synthesis task specified, we run the synthesis routine.
result = paynt.dt.synthesize(colored_mdp_factory, task)
print("success:", result.success)

# %%
# If successful, we display the obtained result.
if result.success:
    tree = result.tree
    print("value:", result.value)
    print("decision tree:\n", result.tree.to_string())

# %% [markdown]
# The decision tree represents the optimal policy (which achieves the winning probability of $0.6313$.
# The decision tree is presented as simple Python code.
# Condition `most_plums<=0` represents that `most_plums` is `False` whereas the else case represents that `most_plums` is `True`.
# Following the first condition, we see that if neither `most_plums`, `most_pears` nor `most_apples` then we should choose 🍒. If however, neither `most_plums` nor `most_pears` but `most_apples`, then we should choose 🍏.
