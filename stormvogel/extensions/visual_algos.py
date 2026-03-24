"""Simple implementations of two model checking algorithms in stormvogel,
along with a function to display the workings of the algorithms."""

from typing import Any
import stormvogel.model
from time import sleep


def naive_value_iteration(
    model: stormvogel.model.Model, epsilon: float, target_state: stormvogel.model.State
) -> list[list[stormvogel.model.Value]]:
    """Run naive value iteration.

    Return a 2D list where ``result[n][m]`` is the value of state *m* at
    iteration *n*.

    :param model: Target model.
    :param epsilon: Convergence threshold.
    :param target_state: Target state of the model.
    :returns: 2D list of values per iteration per state.
    :raises RuntimeError: If *epsilon* is zero or negative.
    """
    if epsilon <= 0:
        raise RuntimeError("The algorithm will not terminate if epsilon is zero.")

    # Create a dynamic list of dicts to store the result.
    values_matrix: list[dict[stormvogel.model.State, float]] = [
        {state: 0.0 for state in model.states}
    ]
    values_matrix[0][target_state] = 1.0

    terminate = False
    while not terminate:
        old_values = values_matrix[-1]
        new_values: dict[stormvogel.model.State, float] = {
            state: 0.0 for state in model.states
        }
        for state in model.states:
            choices = model.transitions[state]
            # Now we have to take a decision for an action.
            action_values = {}
            for action, branch in choices:
                branch_value = sum(
                    [
                        prob * old_values[target_state_in_branch]
                        for (prob, target_state_in_branch) in branch
                    ]  # type: ignore
                )
                action_values[action] = (
                    float(branch_value)
                    if not isinstance(branch_value, float)
                    else branch_value
                )
            # We take the action with the highest value.
            new_values[state] = max(action_values.values()) if action_values else 0
        values_matrix.append(new_values)  # type: ignore
        terminate = (
            sum([abs(new_values[s] - old_values[s]) for s in model.states]) < epsilon  # type: ignore
        )

    # Convert back to list of lists for compatibility with return type and display
    return [
        [step_values[state] for state in model.states] for step_values in values_matrix
    ]  # type: ignore


def dtmc_evolution(model: stormvogel.model.Model, steps: int) -> list[list[float]]:
    """Run DTMC evolution.

    Return a 2D list where ``result[n][m]`` is the probability to be in
    state *m* at step *n*.

    :param model: Target model.
    :param steps: Number of steps.
    :returns: 2D list of probabilities per step per state.
    :raises RuntimeError: If *steps* < 2 or the model is not a DTMC.
    """
    if steps < 2:
        raise RuntimeError("Need at least two steps")
    if model.model_type != stormvogel.model.ModelType.DTMC:
        raise RuntimeError("Only works for DTMC")

    # Create a list of dicts to store values
    matrix_steps_states = [{s: 0.0 for s in model.states} for _ in range(steps)]
    matrix_steps_states[0][model.initial_state] = 1

    # Apply the updated values for each step.
    for current_step in range(steps - 1):
        next_step = current_step + 1
        for s in model.states:
            branch = model.transitions[s][stormvogel.model.EmptyAction]
            for transition_prob, target in branch:
                current_prob = matrix_steps_states[current_step][s]
                assert isinstance(transition_prob, (int, float))
                matrix_steps_states[next_step][target] += current_prob * float(
                    transition_prob
                )

    # Convert to list of lists for display
    return [
        [step_values[state] for state in model.states]
        for step_values in matrix_steps_states
    ]


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
    """Display a value iteration result using matplotlib.

    :param res: 2D list where ``result[n][m]`` is the probability to be in
        state *m* at step *n*.
    :param hor_size: Horizontal size of the plot in inches.
    :param labels: Names of all the states.
    """
    import matplotlib.pyplot as plt

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
    """Return the argument that yields the highest value across callables.

    :param funcs: List of callables.
    :param args: List of arguments, one per callable.
    :returns: The argument whose callable returned the highest value.
    """
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
    """Perform policy iteration on the given MDP.

    :param model: MDP model.
    :param prop: PRISM property string to maximize. Note that this is a
        property on the induced DTMC, not the MDP.
    :param visualize: Whether to visualize intermediate and final results.
    :param layout: Layout used to show intermediate results.
    :param delay: Seconds to wait between each iteration.
    :param clear: Whether to clear the visualization of each previous iteration.
    :returns: Result of model checking on the final induced DTMC.
    """
    old = None
    new = stormvogel.random_scheduler(model)

    while not old == new:
        old = new

        dtmc = old.generate_induced_dtmc()
        dtmc_result = stormvogel.model_checking(dtmc, prop=prop)  # type: ignore

        if visualize:
            vis = stormvogel.visualization.JSVisualization(
                model, scheduler=old, result=dtmc_result
            )
            vis.show()
            sleep(delay)
            if clear:
                vis.clear()

        choices = {
            s1: arg_max(
                [
                    lambda a: sum(
                        [
                            (
                                p
                                * dtmc_result.get_result_of_state(  # type: ignore
                                    dtmc.states[model.get_state_index(target)]
                                )
                            )  # type: ignore
                            for (p, target) in s1.get_outgoing_transitions(a)  # type: ignore
                        ]
                    )
                    for _ in s1.available_actions()
                ],
                s1.available_actions(),
            )
            for s1 in model
        }
        new = stormvogel.Scheduler(model, choices)
    if visualize:
        print("Value iteration done:")
        stormvogel.show(model, scheduler=new, result=dtmc_result)  # type: ignore
    return dtmc_result  # type: ignore
