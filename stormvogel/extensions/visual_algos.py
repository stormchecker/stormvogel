"""Simple implementations of two model checking algorithms in stormvogel,
along with a function to display the workings of the algorithms."""

from typing import Any
import stormvogel.model
import matplotlib.pyplot as plt
from time import sleep


def naive_value_iteration(
    model: stormvogel.model.Model, epsilon: float, target_state: stormvogel.model.State
) -> list[list[stormvogel.model.Value]]:
    """Run naive value iteration. The result is a 2D list where result[n][m] is the probability to be in state m at step n.

    Args:
        model (stormvogel.model.Model): Target model.
        steps (int): Amount of steps.
        target_state (stormvogel.model.State): Target state of the model.

    Returns:
        list[list[float]]: The result is a 2D list where result[n][m] is the value of state m at iteration n.
    """
    if epsilon <= 0:
        RuntimeError("The algorithm will not terminate if epsilon is zero.")

    # Create a dynamic matrix (list of lists) to store the result.
    values_matrix = [[0 for state in model.get_states()]]
    values_matrix[0][target_state.id] = 1

    terminate = False
    while not terminate:
        old_values = values_matrix[len(values_matrix) - 1]
        new_values = [None for state in model.get_states()]
        for sid, state in model:
            choices = model.get_choice(state)
            # Now we have to take a decision for an action.
            action_values = {}
            for action, branch in choices:
                branch_value = sum(
                    [prob * old_values[state.id] for (prob, state) in branch]  # type: ignore
                )
                action_values[action] = branch_value
            # We take the action with the highest value.
            highest_value = max(action_values.values())
            new_values[sid] = highest_value
        values_matrix.append(new_values)  # type: ignore
        terminate = (
            sum([abs(x - y) for (x, y) in zip(new_values, old_values)]) < epsilon  # type: ignore
        )
    return values_matrix  # type: ignore


def dtmc_evolution(model: stormvogel.model.Model, steps: int) -> list[list[float]]:
    """Run DTMC evolution. The result is a 2D list where result[n][m] is the probability to be in state m at step n.

    Args:
        model (stormvogel.model.Model): Target model.
        steps (int): Amount of steps.

    Returns:
        list[list[float]]: The result is a 2D list where result[n][m] is the probability to be in state m at step n.
    """
    if steps < 2:
        RuntimeError("Need at least two steps")
    if model.type != stormvogel.model.ModelType.DTMC:
        RuntimeError("Only works for DTMC")

    # Create a matrix and set the value for the starting state to 1 on the first step.
    matrix_steps_states = [[0.0 for s in model.get_states()] for x in range(steps)]
    matrix_steps_states[0][model.get_initial_state().id] = 1

    # Apply the updated values for each step.
    for current_step in range(steps - 1):
        next_step = current_step + 1
        for s_id, s in model:
            branch = model.get_branch(s)
            for transition_prob, target in branch:
                current_prob = matrix_steps_states[current_step][s_id]
                assert isinstance(transition_prob, (int, float))
                matrix_steps_states[next_step][target.id] += current_prob * float(
                    transition_prob
                )

    return matrix_steps_states


def invert_2d_list(li: list[list[Any]]) -> list[list[Any]]:
    res = []
    for i in range(len(li[0])):
        sublist = []
        for j in range(len(li)):
            sublist.append(li[j][i])
        res.append(sublist)
    return res


def display_value_iteration_result(
    res: list[list[float]], hor_size: int, labels: list[str]
):
    """Display a value iteration results using matplotlib.

    Args:
        res (list[list[float]]): 2D list where result[n][m] is the probability to be in state m at step n.
        hor_size (int): the horizontal size of the plot, in inches.
        labels (list[str]): the names of all the states.
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_xticks(range(len(res)))
    ax.set_yticks(range(len(res[0]) + 1))
    ax.set_yticklabels(labels + [""])

    ax.imshow(invert_2d_list(res), cmap="hot", interpolation="nearest", aspect="equal")
    plt.xlabel("steps")
    plt.ylabel("states")
    fig.set_size_inches(hor_size, hor_size)

    plt.show()


def arg_max(funcs, args):
    """Takes a list of callables and arguments and return the argument that yields the highest value."""
    executed = [f(x) for f, x in zip(funcs, args)]
    index = executed.index(max(executed))
    return args[index]


def policy_iteration(
    model: stormvogel.model.Model,
    prop: str,
    visualize: bool = True,
    layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
    delay: int = 2,
    clear: bool = True,
) -> stormvogel.Result:
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
    new = stormvogel.random_scheduler(model)

    while not old == new:
        old = new

        dtmc = old.generate_induced_dtmc()
        dtmc_result = stormvogel.model_checking(dtmc, prop=prop)  # type: ignore

        if visualize:
            vis = stormvogel.visualization.JSVisualization(
                model, layout=layout, scheduler=old, result=dtmc_result
            )
            vis.show()
            sleep(delay)
            if clear:
                vis.clear()

        choices = {
            i: arg_max(
                [
                    lambda a: sum(
                        [
                            (p * dtmc_result.get_result_of_state(s2.id))  # type: ignore
                            for p, s2 in s1.get_outgoing_choice(a)  # type: ignore
                        ]
                    )
                    for _ in s1.available_actions()
                ],
                s1.available_actions(),
            )
            for i, s1 in model
        }
        new = stormvogel.Scheduler(model, choices)
    if visualize:
        print("Value iteration done:")
        stormvogel.show(model, layout=layout, scheduler=new, result=dtmc_result)  # type: ignore
    return dtmc_result  # type: ignore
