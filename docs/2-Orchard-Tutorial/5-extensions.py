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
# # Next Steps
# Up until now, we showcased various analysis approaches for different model types in the
# previous steps.
# This notebook presents typical use cases for the Python API: Using symbolic model representations, using preprocessing algorithms, configuring solvers, and implementing own prototypes on top of existing routines.

# %% [markdown]
# ## Preparation
# As usual, we load the Prism Orchard model.

# %%
# Import prism models from the previous step
import stormpy
from orchard.orchard_builder import build_prism

# %%
orchard_prism, prism_program = build_prism()

# %% [markdown]
# ## Model representation
# So far, we have represented the MDP through an explicit (sparse) transition matrix. Instead, we can build the MDP symbolically and represent it through **decision diagrams (DD)**.

# %%
orchard_symbolic = stormpy.build_symbolic_model(prism_program)
print(orchard_symbolic)

# %% [markdown]
# The symbolic MDP has the same size as the sparse MDP, but instead of explicitly
# storing nearly 45,000 transitions, the function is represented through a DD with
# 946 nodes.

# %% [markdown]
# We can perform model checking on this symbolic representation and obtain the same winning probability.

# %%
formula = stormpy.parse_properties('Pmax=? [F "PlayersWon"]')[0]
symbolic_result = stormpy.model_checking(
    orchard_symbolic, formula, only_initial_states=True
)
filter = stormpy.create_filter_initial_states_symbolic(orchard_symbolic)
symbolic_result.filter(filter)
print("Maximal probability: {}".format(symbolic_result))

# %% [markdown]
# ## Bisimulation
# Bisimulation minimization can be applied to minimize the state space without affecting the model behaviour.

# %%
formula = stormpy.parse_properties('Pmax=? [F "PlayersWon"]')
print(
    "Model with {} states and {} transitions".format(
        orchard_prism.nr_states, orchard_prism.nr_transitions
    )
)
orchard_bisim = stormpy.perform_bisimulation(
    orchard_prism, formula, stormpy.BisimulationType.STRONG
)
print(
    "Model with {} states and {} transitions".format(
        orchard_bisim.nr_states, orchard_bisim.nr_transitions
    )
)

# %% [markdown]
# For the orchard model, we can reduce the state space size by $95\%$ from $22,469$ states to $956$ states.
# This reduction also speeds up subsequent analysis on the reduced model.

# %% [markdown]
# ## Model checking algorithms
# The underlying model checking algorithms in Storm are configured through so-called environments.
# For example, we can set the precision requirement from the default value of $10^{−6}$ to $0.1$ and see how that
# affects the resulting probability.

# %%
# Change precision
env = stormpy.Environment()
prec = stormpy.Rational(0.1)
env.solver_environment.minmax_solver_environment.precision = prec
result = stormpy.model_checking(orchard_prism, formula[0], environment=env)
print(result.at(orchard_prism.initial_states[0]))

# %% [markdown]
# We can see that the resulting probability $0.5815$ is within the precision guarantee of $0.1$.

# %% [markdown]
# We can also change the underlying algorithm and compare different approaches such as **value iteration**, **policy iteration** and **optimistic value iteration**.

# %%
# Change algorithm
import time

methods = [
    stormpy.MinMaxMethod.value_iteration,
    stormpy.MinMaxMethod.policy_iteration,
    stormpy.MinMaxMethod.optimistic_value_iteration,
]
for m in methods:
    env = stormpy.Environment()
    env.solver_environment.minmax_solver_environment.method = m
    start = time.time()
    result = stormpy.model_checking(
        orchard_prism, formula[0], environment=env, extract_scheduler=True
    )
    print(f"Method: {m}")
    print(f"Result: {result.at(orchard_prism.initial_states[0])}")
    print(f"Time: {time.time() - start:.2}s")

# %% [markdown]
# We see that all methods provide the same result, but their timings differ.
# For example, policy iteration is typically slower.
#
# Running the above cell multiple times allows to get better insights into the performance of the different algorithms.

# %% [markdown]
# ## Writing an LP-based MDP Model Checker
# One further step is extending stormpy with own algorithms. Extending stormpy allows to profit
# from existing data structure and efficient algorithms when integrating own ideas.
# We exemplify this with a fundamental approach to solve MDPs using a **linear program (LP)** solver, an approach that is popular due to the flexibility to add additional constraints.
#
# The LP encoding below computes reachability probabilities.

# %% [markdown]
# Given a target label $B \in AP$,
# this LP characterizes the variables $(x_s)_{s \in S}$ as follows:
#
# Minimize $\sum_{s \in S} x_s$ such that:
# 1. If $B \in L(s)$, then $x_s =1$,
# 2. If there is no path from $s$ to any state with label $B$, then $x_s=0$,
# 3. Otherwise, $0 \leq x_s \leq 1$, and for all actions $\alpha \in \textit{Act}(s)$:
#     $x_s \geq \sum_{s' \in S} \mathbb{P}(s, \alpha, s') \cdot x_{s'}$

# %% [markdown]
# We encode the LP by using functionality of stormpy.
# We first compute all states that satisfy the requirements in (1) and (2) using graph algorithms provided in stormpy.
# For instance, we compute the states $s$ which have to path to label $B$ using the function `compute_prob01max_states()`.
# We then construct and solve the LP by inspecting the model's transition matrix.

# %%
# These are the target states
players_won = stormpy.parse_properties('"PlayersWon"')[0].raw_formula
psi_states = stormpy.model_checking(orchard_prism, players_won).get_truth_values()

# These are the states that can never reach the target states
phi_states = stormpy.BitVector(orchard_prism.nr_states, True)
prob0max_states = stormpy.compute_prob01max_states(
    orchard_prism, phi_states, psi_states
)[0]

# SCIP is an LP solver
from pyscipopt import Model

m = Model()

# Create a variable for each state
num_states = orchard_prism.nr_states
state_vars = [m.addVar(f"x_{i}", lb=0, ub=1) for i in range(num_states)]

# Encode LP
for state in range(num_states):
    if psi_states.get(state):  # Case 1
        m.addCons(state_vars[state] == 1)
    elif prob0max_states.get(state):  # Case 2
        m.addCons(state_vars[state] == 0)
    else:  # Case 3
        for row in orchard_prism.transition_matrix.get_rows_for_group(state):
            summed_prob = 0
            for transition in orchard_prism.transition_matrix.get_row(row):
                prob = transition.value()
                next_state = transition.column
                summed_prob += prob * state_vars[next_state]
            m.addCons(state_vars[state] >= summed_prob)

# %% [markdown]
# Lastly, we can solve the LP and obtain the same winning probability as before.
# We also compare the LP result with the model checking result.

# %%
# Solve LP
m.setObjective(sum(state_vars), sense="minimize")
m.optimize()
sol = m.getBestSol()
result_lp = sol[state_vars[orchard_prism.initial_states[0]]]

# Compare with default VI
properties = stormpy.parse_properties_without_context('Pmax=? [F "PlayersWon"]')
vi_result = stormpy.model_checking(orchard_prism, properties[0].raw_formula)
result_vi = vi_result.at(orchard_prism.initial_states[0])

print(result_lp, result_vi)
assert abs(result_lp - result_vi) < 1e-6  # => True
