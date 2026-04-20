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
# # Parametric and Interval Models
# In this notebook, we will show how to create and work with parametric and
# interval models.
# ## Parametric Models
#
# Stormvogel represents parametric transition values as **sympy expressions**.
# Any polynomial or rational function over `sympy.Symbol` parameters is a valid
# transition value, which means you can write them using ordinary Python
# arithmetic.

# %%
import sympy as sp

# %% [markdown]
# For example, a polynomial in two variables:

# %%
x, y, z = sp.symbols("x y z")

polynomial1 = x**2 + y**2
print(polynomial1)

polynomial2 = 6 * z**3 + z + 2
print(polynomial2)

rational_function = polynomial1 / polynomial2
print(rational_function)

# %% [markdown]
# To create a parametric model (e.g. pMC or pMDP) we simply use these sympy
# expressions as transition probabilities. As an example, we build a
# parametric Knuth–Yao dice: a coin with success probability `x` is flipped
# repeatedly to simulate a six-sided die.

# %%
from stormvogel import model, bird
from stormvogel.show import show

# A single symbolic parameter for the coin bias.
x = sp.Symbol("x")


# we build the Knuth–Yao dice using the bird model builder
def delta(
    s: bird.State,
) -> list[tuple[float | sp.Expr, bird.State]] | None:
    match s.s:
        case 0:
            return [(x, bird.State(s=1)), (1 - x, bird.State(s=2))]
        case 1:
            return [(x, bird.State(s=3)), (1 - x, bird.State(s=4))]
        case 2:
            return [(x, bird.State(s=5)), (1 - x, bird.State(s=6))]
        case 3:
            return [(x, bird.State(s=1)), (1 - x, bird.State(s=7, d=1))]
        case 4:
            return [
                (x, bird.State(s=7, d=2)),
                (1 - x, bird.State(s=7, d=3)),
            ]
        case 5:
            return [
                (x, bird.State(s=7, d=4)),
                (1 - x, bird.State(s=7, d=5)),
            ]
        case 6:
            return [(x, bird.State(s=2)), (1 - x, bird.State(s=7, d=6))]
        case 7:
            return [(1, s)]


def labels(s: bird.State):
    if s.s == 7:
        return f"rolled{str(s.d)}"


knuth_yao_pmc = bird.build_bird(
    delta=delta,
    init=bird.State(s=0),
    labels=labels,
    modeltype=model.ModelType.DTMC,
)

show(knuth_yao_pmc)

# %% [markdown]
# The symbol `x` was auto-declared on the model as soon as it appeared in a
# transition. You can inspect the parameters any time:

# %%
print("Parameters:", knuth_yao_pmc.parameters)

# %% [markdown]
# We can now evaluate the model by assigning the parameter `x` to any concrete
# value. This induces a regular DTMC with fixed probabilities.

# %%
p = 1 / 2

eval_knuth_yao_pmc = knuth_yao_pmc.get_instantiated_model({"x": p})
show(eval_knuth_yao_pmc)

# %% [markdown]
# ## Interval Models
# We can also set an interval between two values x and y as transition value,
# meaning that we don't know the probability precisely, but we know it is
# between x and y. We represent intervals using `model.Interval`, where we
# have two attributes: `lower` and `upper`. Both of these should be an
# element of type `Number`, i.e., `int`, `float` or `Fraction`.

# %%
interval = model.Interval(1 / 3, 2 / 3)
print(interval)

# %% [markdown]
# Similar to parametric models, creating an interval model is as
# straightforward as just setting some interval objects as transition values.

# %%
from stormvogel import bird
from stormvogel.show import show

# We create our interval values
interval = model.Interval(2 / 7, 6 / 7)
inv_interval = model.Interval(1 / 7, 5 / 7)


# we build the knuth yao dice using the bird model builder
def delta(s: bird.State) -> list[tuple[float | model.Interval, bird.State]] | None:
    match s.s:
        case 0:
            return [(interval, bird.State(s=1)), (inv_interval, bird.State(s=2))]
        case 1:
            return [(interval, bird.State(s=3)), (inv_interval, bird.State(s=4))]
        case 2:
            return [(interval, bird.State(s=5)), (inv_interval, bird.State(s=6))]
        case 3:
            return [(interval, bird.State(s=1)), (inv_interval, bird.State(s=7, d=1))]
        case 4:
            return [
                (interval, bird.State(s=7, d=2)),
                (inv_interval, bird.State(s=7, d=3)),
            ]
        case 5:
            return [
                (interval, bird.State(s=7, d=4)),
                (inv_interval, bird.State(s=7, d=5)),
            ]
        case 6:
            return [(interval, bird.State(s=2)), (inv_interval, bird.State(s=7, d=6))]
        case 7:
            return [(1, s)]


def labels(s: bird.State):
    if s.s == 7:
        return f"rolled{str(s.d)}"


knuth_yao_imc = bird.build_bird(
    delta=delta,
    init=bird.State(s=0),
    labels=labels,
    modeltype=model.ModelType.DTMC,
)

show(knuth_yao_imc)

# %%
