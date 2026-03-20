from collections import OrderedDict

import stormvogel
import stormvogel.model as model
from typing import Iterable
import sympy
import sympy as sp


class Gets(sp.Function):
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
            for action, branch in s.choices.choices.items():
                transition_sum = sum(
                    prob * x[s_next] for prob, s_next in branch.branches
                )
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
    return equations_prob(
        model,
        True,
        modelchecking_result.filter_true(),
        one_states=target_states,
        operator=operator,
    )


def zero_value(model: model.Model):
    return OrderedDict([(s, 0) for s in model.sorted_states])


def one_value(model: model.Model):
    return OrderedDict([(s, 1) for s in model.sorted_states])


def value_to_latex(values: dict[model.State, model.Value], name="V"):
    return [f"{name}({s.friendly_name}) = {v}" for s, v in values.items()]


def make_operator_minreachprob(mdp, target_label):
    modelchecking_result = stormvogel.model_checking(
        mdp, f'Pmin<=0 [F "{target_label}"]', scheduler=False
    )
    zerostates = modelchecking_result.filter_true()
    onestates = mdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        action_values = []
        for action, branch in s.choices.choices.items():
            action_values.append(
                sum(prob * values[s_next] for prob, s_next in branch.branches)
            )
        return min(action_values)

    return BellmanOperator(mdp, zerostates, onestates, _bellman)


def make_operator_maxreachprob(mdp, target_label):
    modelchecking_result = stormvogel.model_checking(
        mdp, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    zerostates = modelchecking_result.filter_true()
    onestates = mdp.state_labels[target_label]

    def _bellman(s: model.State, values: dict[model.State, model.Value]):
        action_values = []
        for action, branch in s.choices.choices.items():
            action_values.append(
                sum(prob * values[s_next] for prob, s_next in branch.branches)
            )
        return max(action_values)

    return BellmanOperator(mdp, zerostates, onestates, _bellman)


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


def visualise_iterations(iterations, background_gradient: str | None = None):
    import pandas as pd

    states = list(iterations[0].keys())

    data = {i: [iteration[s] for s in states] for i, iteration in enumerate(iterations)}

    df = pd.DataFrame(data, index=states)

    df.index.name = "state"
    df.columns.name = "iteration"

    # nicer labels if State objects are used
    df.index = [s.friendly_name for s in df.index]
    if background_gradient is not None:
        df.style.background_gradient(cmap=background_gradient)
    return df


class VI:
    def __init__(
        self, operator: BellmanOperator, initial_values: dict[model.State, model.Value]
    ):
        self._operator = operator
        self._values = initial_values

    def step(self):
        self._values = self._operator.apply(self._values)
        return self._values

    @property
    def current_values(self):
        return self._values


class IVI:
    def __init__(self, lowerVI: VI, upperVI: VI):
        self._lowerVI = lowerVI
        self._upperVI = upperVI

    def step(self):
        self._lowerVI.step()
        self._upperVI.step()
