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
# # Stormpy Conversion
# Models can be converted back and forth between stormvogel and stormpy with ease using the `mapping` module.<br>
# This is useful, because this allows you to combine both APIs. For example, you could create a model in stormvogel becuase it has an easy API, do some model checking in stormpy, and then convert it back to display the results. (Note that there is also a direct model checking function available that uses stormpy behind the scenes.)

# %%
from stormvogel import *
stormvogel_model = examples.create_car_mdp()
vis = show(stormvogel_model, layout=stormvogel.layout.Layout("layouts/car.json"))

# %% [markdown]
# First, let's convert the stormvogel model to the same model in stormpy.

# %%
import stormvogel.stormpy_utils.mapping as mapping

stormpy_model = mapping.stormvogel_to_stormpy(stormvogel_model)
print(stormpy_model)

# %% [markdown]
# And now we convert it back.

# %%
stormvogel_model2 = mapping.stormpy_to_stormvogel(stormpy_model)
vis = show(stormvogel_model2, layout=Layout("layouts/car.json"))

# %% [markdown]
# The result of the double conversion is equal to the original model.

# %%
stormvogel_model == stormvogel_model2
