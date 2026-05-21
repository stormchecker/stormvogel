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
# # Working with Stormvogel and Stormpy
# In this notebook, we will show how to integrate stormvogel and stormpy. Stormvogel has a built in function for model checking that uses stormpy behind the scenes, but you can also use stormpy directly if you want to. In that case, you can convert your model to a stormpy model, do some model checking, and then convert it back to display the results.
# We first use the simple study model. The idea is that if you do not study, then you save some time, hence you will gain 15 reward. If you pass the test you get 100 reward because you want to graduate eventually. If you study, then the chance of passing the test becomes higher. Now should you study?

# %%
from stormvogel import *

study = examples.create_study_mdp()
show(study)

# %% [markdown]
# Now we let stormpy solve the question whether you should study. Model checking requires a *property string*. This is a string that specifies what result the model checker should aim for. Stormvogel has a graphical interface that makes it easier to create these. Try to create a property that maximizes the reward at the end. The result should be `Rmax=? [F "end"]`.

# %%
# build_property_string(study)

# %% [markdown]
# Now we run our model checking, and then display the result on the model. The action that is chosen to maximize the reward is marked in red. In conclusion, you should study!<br>
# The red action is called the *scheduled action*. The star symbol indicates the *result* of a state. In this case, the result can be seen as the expected reward. The scheduler finds that going to the state with 'didn't study' results in an expected reward of 55, while the state with 'studied' results in an expected value of 90.

# %%
result = model_checking(
    study, 'Rmax=? [F "end"]', True
)  # true lets it return a scheduler as well
show(study, result=result)

# %% [markdown]
# Now, imagine that the exam is not so important after all. Say you only get 20 reward for passing. Is it still worth studying now? It turns out not to be the case! (The turning point is 30)

# %%
study2 = examples.create_study_mdp()

reward_model = study2.get_rewards("R")
pass_test = next(iter(study2.get_states_with_label("pass test")))
reward_model.set_state_reward(pass_test, 20)
result3 = model_checking(study2, 'Rmax=? [F "end"]')
show(study2, result=result3)

# %% [markdown]
# Using the simulator, we can get a path from the scheduler that we found.

# %%
from stormvogel.simulator import simulate_path

path = simulate_path(study2, 5, result3.scheduler)

# %% [markdown]
# Let's do another example with the lion model from before. We want to minimize the chance that it dies. We do this by asking the model checker to minimize the chance of reaching 'dead'. It turns out that our lion is really doomed, it will always die eventually, no matter what it chooses... The result (☆) at the initial state is 1. This means that the probability of the forumula [F "dead"] (eventually, the model reaches a state with "dead"), is 1.

# %%
lion = examples.create_lion_mdp()
result = model_checking(lion, 'Pmin=? [F "dead"]', True)
show(lion, result=result)

# %% [markdown]
# On the other hand, our lion might as well have a good time while it's alive. All a lion really wants is to roar while being full. If it does this, it gets a reward of 100. Let's try to maximize this reward. The scheduler that is found always roars when it's full and hunts otherwise.

# %%
result2 = model_checking(lion, 'Rmax=? [ F "dead" ]', True)
show(lion, result=result2)

# %% [markdown]
# Do you remember our Monty Hall model?

# %%
mdp = examples.create_monty_hall_mdp()
show(mdp)

# %% [markdown]
# Model checking requires a PRISM property string. In this string, it is specified what property of the model should be used for model checking. In this example, we want a property that maximizes our winning chances.<br>

# %%
result = model_checking(mdp, 'Pmax=? [F "target"]')

# %%
show(mdp, result=result)

# %% [markdown]
# Now, the resulting agent chooses to stay when the car is already behind the door, and to switch when it is not. It always wins... This is because in an MDP, the agent always knows where the car is. In order to solve this problem propertly, we would need model checking on POMDPs, unfortunately this is undecidable.
#
# A possible way to still use MDP model checking to solve this example, is by giving states a label 'should stay' or 'should switch', and then calculate the probablity that you reach such a state.
#
# Now we have stormpy calculate the probablity that we reach a state where we should switch. It turns out to be 2/3rd (see inital state, ☆). Confirm that this still works if you choose another favorite door.

# %%
favorite_door = 2  # 0, 1, or 2
new_mdp = examples.create_monty_hall_mdp2()
result = model_checking(
    new_mdp,
    f'Pmax=? [((("init" | "carchosen" | "o_{favorite_door}") U "should_switch"))]',
)
show(new_mdp, result=result)


# %% [markdown]
# Models can be converted back and forth between stormvogel and stormpy with ease using the `mapping` module.<br>
# This is useful, because this allows you to combine both APIs. For example, you could create a model in stormvogel becuase it has an easy API, do some model checking in stormpy, and then convert it back to display the results. (Note that there is also a direct model checking function available that uses stormpy behind the scenes.)

# %%
from stormvogel import *

stormvogel_model = examples.create_car_mdp()
show(stormvogel_model)

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
show(stormvogel_model2)
