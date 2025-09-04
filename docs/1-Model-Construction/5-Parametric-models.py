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
# # Parametric models
# Instead of setting numeric values as transition probabilities, we may also use parameters, polynomials or even rational functions in multiple variables.

# %%
from stormvogel import parametric

# %% [markdown]
# Polynomials are represented as dictionaries where the keys are the exponents and the values are coefficients. In addition, we must also supply a list of variable names. Rational functions are then represented as a pair of two polynomials (numerator and denominator).

# %%
polynomial1 = parametric.Polynomial(["x","y"])
polynomial1.add_term((2,0),1)
polynomial1.add_term((0,2),1)

print(polynomial1)

polynomial2 = parametric.Polynomial(["z"])
polynomial2.add_term((0,),2)
polynomial2.add_term((1,),1)
polynomial2.add_term((3,),6)

print(polynomial2)

rational_function = parametric.RationalFunction(polynomial1, polynomial2)

print(rational_function)

# %% [markdown]
# To create a parametric model (e.g. pmc or pmdp) we simply have to set such a value as a transition probability. As an example, we provide the knuth yao dice, but with parameters instead of concrete probabilities.

# %%
from stormvogel import model, bird
from stormvogel.show import show

#we first make polynomials 'x' and '1-x'
x = parametric.Polynomial(["x"])
x.add_term((1,),1)

invx = parametric.Polynomial(["x"])
invx.add_term((1,),-1)
invx.add_term((0,),1)

#we build the knuth yao dice using the bird model builder
def delta(s: bird.State):
    match s.s:
        case 0:
            return [(x, bird.State(s=1)), (invx, bird.State(s=2))]
        case 1:
            return [(x, bird.State(s=3)), (invx, bird.State(s=4))]
        case 2:
            return [(x, bird.State(s=5)), (invx, bird.State(s=6))]
        case 3:
            return [(x, bird.State(s=1)), (invx, bird.State(s=7, d=1))]
        case 4:
            return [
                (x, bird.State(s=7, d=2)),
                (invx, bird.State(s=7, d=3)),
            ]
        case 5:
            return [
                (x, bird.State(s=7, d=4)),
                (invx, bird.State(s=7, d=5)),
            ]
        case 6:
            return [(x, bird.State(s=2)), (invx, bird.State(s=7, d=6))]
        case 7:
            return [(1, s)]

def labels(s: bird.State):
    if s.s == 7:
        return f"rolled{str(s.d)}"

knuth_yao = bird.build_bird(
    delta=delta,
    init=bird.State(s=0),
    labels=labels,
    modeltype=model.ModelType.DTMC,
)

vis = show(knuth_yao)

# %% [markdown]
# We can now evaluate the model by assigning the variable x to any concrete value. This induces a regular dtmc with fixed probabilities.

# %%
p = 1/2

eval_knuth_yao = knuth_yao.parameter_valuation({"x":p})
vis = show(eval_knuth_yao)
