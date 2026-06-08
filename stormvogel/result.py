__all__ = [
    "Scheduler",
    "random_scheduler",
    "Result",
    "ParetoResult",
    "plot_pareto_result",
]

import stormvogel.model
import random
from dataclasses import dataclass
from typing import Callable

from deprecated import deprecated  # type: ignore[import]

from stormvogel import parametric


class Scheduler:
    """Specify what action to take in each state.

    All schedulers are nondeterministic and memoryless.

    :param model: Model associated with the scheduler (must support actions).
    :param taken_actions: For each state, the action chosen in that state.
    """

    model: stormvogel.model.Model
    # taken actions are hashed by the state id
    taken_actions: dict[stormvogel.model.State, stormvogel.model.Action]

    def __init__(
        self,
        model: stormvogel.model.Model,
        taken_actions: dict[stormvogel.model.State, stormvogel.model.Action],
    ):
        self.model = model
        self.taken_actions = taken_actions

    def get_action_at_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Action:
        """Return the action in the scheduler for the given state.

        :param state: The state to look up.
        :returns: The action chosen for the given state.
        :raises RuntimeError: If the state is not a part of the model.
        """
        if state in self.model.states:
            return self.taken_actions[state]
        else:
            raise RuntimeError("This state is not a part of the model")

    def generate_induced_dtmc(
        self, drop_unreachable: bool = True
    ) -> stormvogel.model.Model:
        """Resolve the nondeterminacy of the MDP and return the scheduler-induced DTMC.

        Copies the MDP (preserving state UUIDs), changes the model type to DTMC,
        and replaces each state's choices with only the scheduled action's branch.

        :param drop_unreachable: When ``True`` (default), states not reachable from
            the initial state under the scheduler are pruned from the result.
            Set to ``False`` to keep the full state space.
        :returns: The induced DTMC.
        :raises ValueError: If the model is not an MDP or POMDP.
        """
        if self.model.model_type not in (
            stormvogel.model.ModelType.MDP,
            stormvogel.model.ModelType.POMDP,
        ):
            raise ValueError(
                f"generate_induced_dtmc requires an MDP or POMDP, "
                f"got {self.model.model_type}."
            )
        induced = self.model.copy()
        induced.model_type = stormvogel.model.ModelType.DTMC

        for orig_state in self.model:
            new_state = induced.get_state_by_id(orig_state.state_id)
            action = self.get_action_at_state(orig_state)
            transitions = orig_state.get_outgoing_transitions(action)
            assert transitions is not None
            # Replace the full Choices with just the scheduled branch.
            # Targets are already the copied states (same UUIDs).
            remapped = [
                (prob, induced.get_state_by_id(target.state_id))
                for prob, target in transitions
            ]
            induced.set_choices(new_state, remapped)

        if drop_unreachable:
            reachable: set[stormvogel.model.State] = set()
            queue = [induced.initial_state]
            reachable.add(induced.initial_state)
            while queue:
                s = queue.pop()
                for _, branch in induced.transitions[s]:
                    for _, t in branch:
                        if t not in reachable:
                            reachable.add(t)
                            queue.append(t)
            if len(reachable) < len(induced.states):
                induced = induced.get_sub_model(reachable)

        return induced

    def __str__(self) -> str:
        return "taken actions: " + str(self.taken_actions)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Scheduler):
            return False

        return self.taken_actions == other.taken_actions


def random_scheduler(model: stormvogel.model.Model) -> Scheduler:
    """Create a random scheduler for the provided model.

    :param model: The model to create a scheduler for.
    :returns: A new :class:`Scheduler` with randomly chosen actions.
    """
    choices = {}
    for state in model:
        actions = state.available_actions()
        if not actions:
            raise ValueError(
                f"State {state} has no available actions and cannot be scheduled."
            )
        choices[state] = random.choice(actions)
    return Scheduler(model, taken_actions=choices)


class Result:
    """Represent the model checking results for a given model.

    :param model: Stormvogel representation of the model associated with the results.
    :param values: For each state, the model checking result.
    :param scheduler: In case the model is an MDP, optionally store a scheduler.
    """

    model: stormvogel.model.Model
    # values are hashed by the state id:
    values: dict[stormvogel.model.State, stormvogel.model.Value]
    scheduler: Scheduler | None

    def __init__(
        self,
        model: stormvogel.model.Model,
        values: dict[stormvogel.model.State, stormvogel.model.Value],
        scheduler: Scheduler | None = None,
    ):
        self.model = model
        self.values = values

        if isinstance(scheduler, Scheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = None

    def at(self, state: stormvogel.model.State) -> stormvogel.model.Value:
        """Return the model checking result for a given state.

        :param state: The state to look up.
        :returns: The model checking result value for the state.
        :raises RuntimeError: If the state is not a part of the model.
        """
        if isinstance(state, stormvogel.model.State) and state in self.values:
            return self.values[state]
        else:
            raise RuntimeError("This state is not a part of the model")

    def at_init(self) -> stormvogel.model.Value:
        """Return the model checking result for the initial state."""
        return self.at(self.model.initial_state)

    @deprecated(version="0.12.0", reason="use at() instead.")
    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Value:
        return self.at(state)

    def filter(
        self, value_predicate: Callable[[stormvogel.model.Value], bool]
    ) -> list[stormvogel.model.State]:
        """ """
        return [s for s, v in self.values.items() if value_predicate(v)]

    def filter_true(self):
        """
        Obtain the set of states S'  where the result for each s in S' is true.
        """
        return self.filter(lambda v: (v is True))

    def __str__(self) -> str:
        return (
            "values: \n "
            + str(self.values)
            + "\n"
            + "scheduler: \n "
            + str(self.scheduler)
        )

    def maximum_result(self) -> stormvogel.model.Value:
        """Return the maximum result.

        :returns: The maximum value across all states.
        :raises RuntimeError: If the model uses interval or parametric values.
        """
        values = list(self.values.values())
        max_val = values[0]
        for v in values:
            if isinstance(v, stormvogel.model.Interval) or parametric.is_parametric(v):
                raise RuntimeError(
                    "maximum result function does not work for interval/parametric models"
                )
            assert isinstance(v, stormvogel.model.Number)
            if v > max_val:
                max_val = v
        return max_val

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False

        if len(self.values) != len(other.values):
            return False

        if len(self.model.states) != len(other.model.states):
            return False

        for index, s1 in enumerate(self.model.states):
            if s1 not in self.values:
                continue
            v1 = self.values[s1]
            s2 = other.model.states[index]
            if s2 not in other.values:
                return False
            v2 = other.values[s2]
            if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                if v1 != v2:
                    return False
            elif abs(float(v1) - float(v2)) > 1e-6:
                return False

        if (self.scheduler is None) != (other.scheduler is None):
            return False

        if self.scheduler is not None and other.scheduler is not None:
            if self.scheduler.taken_actions != other.scheduler.taken_actions:
                for index, s1 in enumerate(self.model.states):
                    if s1 in self.scheduler.taken_actions:
                        a1 = self.scheduler.taken_actions[s1]
                        if index >= len(other.model.states):
                            return False
                        s2 = other.model.states[index]
                        if (
                            s2 not in other.scheduler.taken_actions
                            or other.scheduler.taken_actions[s2] != a1
                        ):
                            return False
        return True

    def __iter__(self):
        return iter(self.values.items())


@dataclass
class ParetoResult:
    """Result of a multiobjective Pareto model checking query.

    Holds the under- and over-approximation of the Pareto front as vertex lists.
    Each point is a list of floats with one entry per objective. The number of
    objectives is unrestricted, but plotting is limited to 2 objectives.

    :param lower_points: Vertices of the under-approximation (known achievable region).
    :param upper_points: Vertices of the over-approximation.
    :param property_labels: One label per objective, auto-extracted from the formula.
    """

    lower_points: list[list[float]]
    upper_points: list[list[float]]
    property_labels: list[str] | None = None

    def plot(self, ax=None, labels: tuple[str, str] | None = None):
        """Plot the Pareto front. Only 2-objective results are supported.

        :param ax: Target axes; creates a new figure if None.
        :param labels: Override axis labels; defaults to :attr:`property_labels`.
        :returns: The populated axes.
        """
        return plot_pareto_result(self, ax=ax, labels=labels)


def plot_pareto_result(
    result: "ParetoResult",
    ax=None,
    labels: "tuple[str, str] | None" = None,
    bbox_pad: float = 0.2,
):
    """Plot the under- and over-approximation of a 2-objective Pareto model checking result.

    Renders:
    - Green filled polygon: under-approximation (known achievable region)
    - Blue dashed outline: over-approximation (upper bound on achievable region)
    - Black dots: vertices of the under-approximation

    :param result: A :class:`ParetoResult` from multiobjective model checking.
    :param ax: Target axes; creates a new figure if None.
    :param labels: Override axis labels; defaults to :attr:`~ParetoResult.property_labels`.
    :param bbox_pad: Fractional padding around the points for axis limits.
    :returns: The populated axes.
    :raises ValueError: If *result* does not contain exactly 2-dimensional points.
    """
    import numpy as np
    import matplotlib.patches as mpatches
    from stormvogel.teaching.pareto import _prepare_ax, _finalize_ax

    all_points = result.lower_points + result.upper_points
    if not all_points:
        raise ValueError("ParetoResult contains no points.")
    if len(all_points[0]) != 2:
        raise ValueError(
            f"plot_pareto_result only supports 2-objective results; "
            f"got {len(all_points[0])} objectives."
        )

    if labels is not None:
        l1, l2 = labels
    elif result.property_labels is not None and len(result.property_labels) >= 2:
        l1, l2 = result.property_labels[0], result.property_labels[1]
    else:
        l1, l2 = "Objective 1", "Objective 2"

    resolved_ax, x_hi, y_hi = _prepare_ax(all_points, bbox_pad, ax)

    if result.upper_points:
        resolved_ax.add_patch(
            mpatches.Polygon(
                np.array(result.upper_points),
                closed=True,
                facecolor="none",
                edgecolor="steelblue",
                linestyle="--",
                linewidth=1.2,
                label="Over-approximation",
            )
        )

    if result.lower_points:
        resolved_ax.add_patch(
            mpatches.Polygon(
                np.array(result.lower_points),
                closed=True,
                facecolor="green",
                edgecolor="darkgreen",
                linewidth=0.5,
                alpha=0.4,
                label="Under-approximation",
            )
        )
        resolved_ax.scatter(
            [p[0] for p in result.lower_points],
            [p[1] for p in result.lower_points],
            color="black",
            zorder=5,
            s=30,
            label="Achievable points",
        )

    _finalize_ax(resolved_ax, x_hi, y_hi, l1, l2)
    return resolved_ax
