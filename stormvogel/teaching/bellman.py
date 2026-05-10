from collections import OrderedDict
from typing import Any, Iterable

import sympy as sp

import stormvogel
import stormvogel.model as model


class Gets(sp.Function):
    """Sympy function rendering as a LaTeX assignment arrow (``\\gets``)."""

    nargs = 2


def _latex(self, printer):
    left, right = self.args
    return f"{printer._print(left)} \\gets {printer._print(right)}"


Gets._latex = _latex


def equations_prob(
    model: model.Model[float],
    min: bool,
    zero_states: Iterable[model.State],
    one_states: Iterable[model.State],
    operator=False,
):
    if min:
        opt = sp.Min
    else:
        opt = sp.Max
    if operator:
        opsymbol = Gets
        x = {s: sp.Symbol(f"V({s.friendly_name})") for s in model.states}
    else:
        opsymbol = sp.Eq
        x = {s: sp.Symbol(f"x_{s.friendly_name}") for s in model.states}
    # variables x_s

    equations = []
    for s in model.sorted_states:
        action_exprs = []
        if s in one_states:
            equations.append(opsymbol(x[s], 1.0))
        elif s in zero_states:
            equations.append(opsymbol(x[s], 0.0))
        else:
            for _, branch in s.choices:
                transition_sum = sum(prob * x[s_next] for prob, s_next in branch)
                expr = transition_sum
                action_exprs.append(expr)
            rhs = opt(*action_exprs)
            equations.append(opsymbol(x[s], rhs))
    return equations


def maxreachprob(model: model.Model[float], target_label: str, operator=False):
    target_states = model.state_labels[target_label]
    modelchecking_result = stormvogel.model_checking(
        model, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    return equations_prob(
        model,
        False,
        modelchecking_result.filter_true(),
        one_states=target_states,
        operator=operator,
    )


def minreachprob(model: model.Model[float], target_label: str, operator=False):
    target_states = model.state_labels[target_label]
    modelchecking_result = stormvogel.model_checking(
        model, f'Pmin<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    return equations_prob(
        model,
        True,
        modelchecking_result.filter_true(),
        one_states=target_states,
        operator=operator,
    )


def zero_value(model: model.Model) -> dict[model.State, model.Value]:
    return OrderedDict([(s, 0.0) for s in model.sorted_states])


def one_value(model: model.Model) -> dict[model.State, model.Value]:
    return OrderedDict([(s, 1.0) for s in model.sorted_states])


def value_to_latex(values: dict[model.State, model.Value], name="V"):
    return [f"{name}({s.friendly_name}) = {v}" for s, v in values.items()]


def make_operator_minreachprob(mdp, target_label):
    modelchecking_result = stormvogel.model_checking(
        mdp, f'Pmin<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    zerostates = modelchecking_result.filter_true()
    onestates = mdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        action_values = []
        for _, branch in s.choices:
            action_values.append(sum(prob * values[s_next] for prob, s_next in branch))
        return min(action_values)

    return BellmanOperator(mdp, zerostates, onestates, _bellman)


def make_operator_maxreachprob(mdp, target_label):
    modelchecking_result = stormvogel.model_checking(
        mdp, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    zerostates = modelchecking_result.filter_true()
    onestates = mdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        action_values = []
        for _, branch in s.choices:
            action_values.append(sum(prob * values[s_next] for prob, s_next in branch))
        return max(action_values)

    return BellmanOperator(mdp, zerostates, onestates, _bellman)


# ---------------------------------------------------------------------------
# Interval MDP operators
# ---------------------------------------------------------------------------


def _robust_action_value(branch: Any, values: Any, nature_maximizes: bool) -> Any:
    """Greedy corner-point expected value for one action on an interval transition.

    Assigns probabilities within [lower, upper] to maximise or minimise the
    expected value by greedily shifting remaining mass towards the most
    favourable successor states.

    :param branch: List of (Interval, State) pairs for the action.
    :param values: Current value estimate per state.
    :param nature_maximizes: If True, nature pushes mass to high-value states;
        if False, nature pushes mass to low-value states.
    """
    lo = {s: p.lower for p, s in branch}
    hi = {s: p.upper for p, s in branch}
    remaining = 1 - sum(lo.values())
    prob = dict(lo)
    for s in sorted(lo, key=lambda s: values[s], reverse=nature_maximizes):
        fill = min(hi[s] - prob[s], remaining)
        prob[s] += fill
        remaining -= fill
        if remaining <= 0:
            break
    return sum(prob[s] * values[s] for s in prob)


def make_operator_robust_maxreachprob(imdp, target_label):
    """Bellman operator for robust maximum reachability on an interval MDP.

    The agent maximises; nature plays adversarially (minimises expected value).
    Requires stormpy.
    """
    modelchecking_result = stormvogel.model_checking(
        imdp, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    zerostates = modelchecking_result.filter_true()
    onestates = imdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        return max(
            _robust_action_value(branch, values, nature_maximizes=False)
            for _, branch in s.choices
        )

    return BellmanOperator(imdp, zerostates, onestates, _bellman)


def make_operator_robust_minreachprob(imdp, target_label):
    """Bellman operator for robust minimum reachability on an interval MDP.

    The agent minimises; nature plays adversarially (maximises expected value).
    Requires stormpy.
    """
    modelchecking_result = stormvogel.model_checking(
        imdp, f'Pmin<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    zerostates = modelchecking_result.filter_true()
    onestates = imdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        return min(
            _robust_action_value(branch, values, nature_maximizes=True)
            for _, branch in s.choices
        )

    return BellmanOperator(imdp, zerostates, onestates, _bellman)


def make_operator_coop_maxreachprob(imdp, target_label):
    """Bellman operator for cooperative maximum reachability on an interval MDP.

    The agent maximises; nature plays cooperatively (also maximises expected value).
    Requires stormpy.
    """
    modelchecking_result = stormvogel.model_checking(
        imdp, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    zerostates = modelchecking_result.filter_true()
    onestates = imdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        return max(
            _robust_action_value(branch, values, nature_maximizes=True)
            for _, branch in s.choices
        )

    return BellmanOperator(imdp, zerostates, onestates, _bellman)


def make_operator_coop_minreachprob(imdp, target_label):
    """Bellman operator for cooperative minimum reachability on an interval MDP.

    The agent minimises; nature plays cooperatively (also minimises expected value).
    Requires stormpy.
    """
    modelchecking_result = stormvogel.model_checking(
        imdp, f'Pmin<=0 [F "{target_label}"]', scheduler=False
    )
    assert modelchecking_result is not None
    zerostates = modelchecking_result.filter_true()
    onestates = imdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        return min(
            _robust_action_value(branch, values, nature_maximizes=False)
            for _, branch in s.choices
        )

    return BellmanOperator(imdp, zerostates, onestates, _bellman)


# ---------------------------------------------------------------------------
# Reward operators
# ---------------------------------------------------------------------------


def make_operator_max_discounted_reward(mdp, reward_name: str, discount: float):
    """Bellman operator for maximum discounted expected reward.

    V(s) = r(s) + discount * max_a Σ p(s,a,s') * V(s').
    All states are updated; discount < 1 ensures convergence.

    :param reward_name: Name of the reward model on *mdp*.
    :param discount: Discount factor γ ∈ (0, 1).
    """
    reward_model = mdp.get_rewards(reward_name)

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        r = reward_model.get_state_reward(s) or 0
        return r + discount * max(
            sum(prob * values[s_next] for prob, s_next in branch)
            for _, branch in s.choices
        )

    return BellmanOperator(mdp, frozenset(), frozenset(), _bellman)


def make_operator_min_discounted_reward(mdp, reward_name: str, discount: float):
    """Bellman operator for minimum discounted expected reward.

    V(s) = r(s) + discount * min_a Σ p(s,a,s') * V(s').
    All states are updated; discount < 1 ensures convergence.

    :param reward_name: Name of the reward model on *mdp*.
    :param discount: Discount factor γ ∈ (0, 1).
    """
    reward_model = mdp.get_rewards(reward_name)

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        r = reward_model.get_state_reward(s) or 0
        return r + discount * min(
            sum(prob * values[s_next] for prob, s_next in branch)
            for _, branch in s.choices
        )

    return BellmanOperator(mdp, frozenset(), frozenset(), _bellman)


def make_operator_max_reachreward(mdp, reward_name: str, done_label: str):
    """Bellman operator for maximum expected reachability reward.

    V(s) = r(s) + max_a Σ p(s,a,s') * V(s'), with done states fixed at 0.

    :param reward_name: Name of the reward model on *mdp*.
    :param done_label: Label identifying absorbing/terminal states (value fixed at 0).
    """
    reward_model = mdp.get_rewards(reward_name)
    done_states = frozenset(mdp.state_labels[done_label])

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        r = reward_model.get_state_reward(s) or 0
        return r + max(
            sum(prob * values[s_next] for prob, s_next in branch)
            for _, branch in s.choices
        )

    return BellmanOperator(mdp, done_states, frozenset(), _bellman)


def make_operator_min_reachreward(mdp, reward_name: str, done_label: str):
    """Bellman operator for minimum expected reachability reward.

    V(s) = r(s) + min_a Σ p(s,a,s') * V(s'), with done states fixed at 0.

    :param reward_name: Name of the reward model on *mdp*.
    :param done_label: Label identifying absorbing/terminal states (value fixed at 0).
    """
    reward_model = mdp.get_rewards(reward_name)
    done_states = frozenset(mdp.state_labels[done_label])

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        r = reward_model.get_state_reward(s) or 0
        return r + min(
            sum(prob * values[s_next] for prob, s_next in branch)
            for _, branch in s.choices
        )

    return BellmanOperator(mdp, done_states, frozenset(), _bellman)


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------


class BellmanOperator[ValueType: model.Value]:
    def __init__(
        self, mdp: model.Model[ValueType], zero_states, one_states, bellman_update
    ):
        self._model = mdp
        self._zerostates = zero_states
        self._onestates = one_states
        self._bellman_update = bellman_update

    def apply(
        self, values: dict[model.State, model.Value]
    ) -> dict[model.State, model.Value]:
        result = OrderedDict()
        for s in self._model.sorted_states:
            if s in self._zerostates:
                result[s] = 0.0
            elif s in self._onestates:
                result[s] = 1.0
            else:
                result[s] = self._bellman_update(s, values)
        return result


def visualise_iterations(
    iterations, background_gradient: str | None = None, digits: int = 5
):
    import pandas as pd

    def _round(v):
        if isinstance(v, tuple):
            return tuple(round(x, digits) for x in v)
        try:
            return round(v, digits)
        except TypeError:
            return v

    states = list(iterations[0].keys())
    data = {
        i: [_round(iteration[s]) for s in states]
        for i, iteration in enumerate(iterations)
    }

    df = pd.DataFrame(data, index=states)  # type: ignore

    df.index.name = "state"
    df.columns.name = "iteration"

    df.index = [s.friendly_name for s in df.index]
    if background_gradient is not None:
        df.style.background_gradient(cmap=background_gradient)
    return df


class VI:
    """Plain value iteration."""

    def __init__(
        self, operator: BellmanOperator, initial_values: dict[model.State, model.Value]
    ):
        self._operator = operator
        self._values = initial_values

    def step(self) -> dict[model.State, model.Value]:
        self._values = self._operator.apply(self._values)
        return self._values

    @property
    def current_values(self) -> dict[model.State, model.Value]:
        return self._values


class IVI:
    """Interval value iteration: two VI runs from below (0) and above (1) simultaneously.

    Both use the same operator. The gap between lower and upper bounds shrinks
    each iteration; convergence is detected when they agree within a tolerance.
    """

    def __init__(self, lower_vi: VI, upper_vi: VI):
        self._lower_vi = lower_vi
        self._upper_vi = upper_vi

    def step(self) -> dict[model.State, tuple[Any, Any]]:
        self._lower_vi.step()
        self._upper_vi.step()
        return self.current_values

    @property
    def current_values(self) -> dict[model.State, tuple[Any, Any]]:
        """Return per-state (lower, upper) bound pairs."""
        lower = self._lower_vi.current_values
        upper = self._upper_vi.current_values
        return {s: (lower[s], upper[s]) for s in lower}
