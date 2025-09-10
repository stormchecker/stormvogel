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
# # Policy iteration
# In policy iteration, you start with an arbitrary policy.
# Then, the the policy is improved at every iteration by first creating a DTMC for the previous policy, and then applying whichever choice would be best in that DTMC for the updated policy.

# %%
from stormvogel import *
from stormvogel.visualization import JSVisualization
from time import sleep

def arg_max(funcs, args):
    """Takes a list of callables and arguments and return the argument that yields the highest value."""
    executed = [f(x) for f,x in zip(funcs,args)]
    index = executed.index(max(executed))
    return args[index]

def policy_iteration(
        model: Model,
        prop: str,
        visualize: bool = True,
        layout: Layout = stormvogel.layout.DEFAULT(),
        delay:int=2,
        clear:bool=False) -> Result:
    """Performs policy iteration on the given mdp.
    Args:
        model (Model): MDP.
        prop (str): PRISM property string to maximize. Rembember that this is a property on the induced DTMC, not the MDP.
        visualize (bool): Whether the intermediate and final results should be visualized. Defaults to True.
        layout (Layout): Layout to use to show the intermediate results.
        delay (int): Seconds to wait between each iteration.
        clear (bool): Whether to clear the visualization of each previous iteration.
        """
    old = None
    new = random_scheduler(model)

    while not old == new:
        old = new

        dtmc = old.generate_induced_dtmc()
        dtmc_result = model_checking(dtmc, prop=prop)

        if visualize:
            vis = JSVisualization(model, layout=layout, scheduler=old, result=dtmc_result)
            vis.show()
            sleep(delay)
            if clear:
                vis.clear()

        choices = {i:
            arg_max(
                [lambda a: sum([(p * dtmc_result.get_result_of_state(s2.id)) for p, s2 in s1.get_outgoing_choice(a)])
                    for _ in s1.available_actions()],
                s1.available_actions())
        for i,s1 in model.states.items()}
        new = Scheduler(model, choices)
    if visualize:
        print("Value iteration done:")
        show(model, layout=layout, scheduler=new, result=dtmc_result)
    return dtmc_result


# %%
lion = examples.create_lion_mdp()
prop = 'P=?[F "full"]'
res = policy_iteration(lion, prop, layout=Layout("layouts/lion_policy.json"))

# %% [markdown]
# Policy iteration is also available under `stormvogel.extensions.visual_algos`.
