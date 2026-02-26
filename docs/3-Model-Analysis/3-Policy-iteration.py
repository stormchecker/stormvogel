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
    executed = [f(x) for f, x in zip(funcs, args)]
    index = executed.index(max(executed))
    return args[index]


def policy_iteration(
    model: Model,
    prop: str,
    visualize: bool = True,
    layout: Layout = stormvogel.layout.DEFAULT(),
    delay: int = 2,
    clear: bool = False,
) -> Result:
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
            mapped_values = {
                model.states[i]: dtmc_result.values.get(dtmc.states[i])
                for i in range(len(model.states))
            }
            mapped_result = Result(model, mapped_values, old)
            vis = JSVisualization(
                model, layout=layout, scheduler=old, result=mapped_result
            )
            vis.show()
            sleep(delay)
            if clear:
                vis.clear()

        # We need a state mapping from the induced DTMC back to the original model.
        # generate_induced_dtmc creates states that correspond 1-to-1 to the original model.
        # Wait, the indices are exactly the same! Let's just use indices.

        state_to_index = {state: idx for idx, state in enumerate(model.states)}

        choices = {}
        for i, s1 in enumerate(model.states):

            def compute_val(a):
                val = 0
                for p, s2 in s1.get_outgoing_transitions(a):
                    # We get the state index in the original model, and look up in DTMC
                    s2_idx = state_to_index[s2]
                    dtmc_s2 = dtmc.states[s2_idx]
                    val += p * dtmc_result.get_result_of_state(dtmc_s2)
                return val

            # arg_max evaluates the functions over the arguments, so we pass a list of lambdas
            lambdas = [
                eval("lambda a: compute_val(a)", {"compute_val": compute_val})
                for _ in s1.available_actions()
            ]
            best_action = arg_max(lambdas, s1.available_actions())
            choices[s1] = best_action

        new = Scheduler(model, choices)
    if visualize:
        print("Value iteration done:")
        mapped_values = {
            model.states[i]: dtmc_result.values.get(dtmc.states[i])
            for i in range(len(model.states))
        }
        mapped_result = Result(model, mapped_values, new)
        show(model, layout=layout, scheduler=new, result=mapped_result)
    return dtmc_result


# %%
lion = examples.create_lion_mdp()
prop = 'P=?[F "full"]'
res = policy_iteration(lion, prop, layout=Layout("layouts/lion_policy.json"))

# %% [markdown]
# Policy iteration is also available under `stormvogel.extensions.visual_algos`.
