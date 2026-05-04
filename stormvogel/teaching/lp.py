import fractions
from dataclasses import dataclass, field
from typing import Any, Iterable

import sympy as sp

import stormvogel
import stormvogel.model as model
from stormvogel.model.action import EmptyAction


@dataclass
class LP:
    """LP formulation for MDP reachability, as sympy expressions.

    Renders as a formatted LP in Jupyter via _repr_latex_.
    """

    sense: str  # "minimize" or "maximize"
    objective: sp.Expr
    constraints: list[sp.Basic] = field(default_factory=list)

    def _repr_latex_(self) -> str:
        rows = [f"\\mathrm{{{self.sense}}} & {sp.latex(self.objective)} \\\\"]
        first_constraint = True
        for c in self.constraints:
            prefix = "\\mathrm{subject\\ to}" if first_constraint else ""
            rows.append(f"{prefix} & {sp.latex(c)} \\\\")
            first_constraint = False
        body = "\n".join(rows)
        return f"$$\\begin{{array}}{{ll}}\n{body}\n\\end{{array}}$$"

    def _repr_markdown_(self) -> str:
        return self._repr_latex_()


@dataclass
class LPSolution:
    """Exact rational solution to an LP returned by :func:`solve_lp`.

    :param objective: Optimal objective value.
    :param values: Variable assignments, keyed by the sympy Symbol used in the LP.
    """

    objective: fractions.Fraction
    values: dict[sp.Symbol, fractions.Fraction] = field(default_factory=dict)


def _expr_to_z3(expr: sp.Expr, var_map: dict[sp.Symbol, Any]) -> Any:
    import z3

    if isinstance(expr, sp.Symbol):
        return var_map[expr]
    if isinstance(expr, sp.Integer):
        return z3.IntVal(int(expr))
    if isinstance(expr, sp.Rational):
        return z3.Q(int(expr.p), int(expr.q))
    if expr.is_number:
        frac = fractions.Fraction(str(float(expr)))
        return z3.Q(frac.numerator, frac.denominator)
    if isinstance(expr, sp.Add):
        return z3.Sum([_expr_to_z3(a, var_map) for a in expr.args])
    if isinstance(expr, sp.Mul):
        result = _expr_to_z3(expr.args[0], var_map)
        for a in expr.args[1:]:
            result = result * _expr_to_z3(a, var_map)
        return result
    raise TypeError(f"Cannot convert sympy type {type(expr).__name__} to z3")


def _constraint_to_z3(c: sp.Basic, var_map: dict[sp.Symbol, Any]) -> Any:
    lhs = _expr_to_z3(c.args[0], var_map)  # type: ignore[arg-type]
    rhs = _expr_to_z3(c.args[1], var_map)  # type: ignore[arg-type]
    if isinstance(c, sp.Eq):
        return lhs == rhs
    if isinstance(c, sp.Le):
        return lhs <= rhs
    if isinstance(c, sp.Ge):
        return lhs >= rhs
    raise TypeError(f"Unsupported constraint type: {type(c).__name__}")


def solve_lp(lp: LP) -> LPSolution | None:
    """Solve an LP using z3 (exact rational arithmetic).

    :param lp: The LP to solve.
    :returns: Exact rational solution, or ``None`` if infeasible or unbounded.
    :raises ImportError: If z3-solver is not installed.
    """
    try:
        import z3
    except ImportError as exc:
        raise ImportError(
            "z3-solver is required for LP solving: pip install z3-solver"
        ) from exc

    symbols: set[sp.Symbol] = set()
    for c in lp.constraints:
        symbols |= c.free_symbols  # type: ignore[operator]
    symbols |= lp.objective.free_symbols  # type: ignore[operator]

    var_map: dict[sp.Symbol, Any] = {s: z3.Real(s.name) for s in symbols}

    opt = z3.Optimize()
    for c in lp.constraints:
        opt.add(_constraint_to_z3(c, var_map))

    obj_z3 = _expr_to_z3(lp.objective, var_map)
    if lp.sense == "minimize":
        opt.minimize(obj_z3)
    else:
        opt.maximize(obj_z3)

    if opt.check() != z3.sat:
        return None

    m = opt.model()

    def _z3_to_fraction(val: Any) -> fractions.Fraction:
        if z3.is_rational_value(val):
            return fractions.Fraction(
                val.numerator_as_long(), val.denominator_as_long()
            )
        if z3.is_int_value(val):
            return fractions.Fraction(val.as_long())
        return fractions.Fraction(str(val))

    values = {
        s: _z3_to_fraction(m.eval(z3_var, model_completion=True))
        for s, z3_var in var_map.items()
    }
    obj_val = _z3_to_fraction(m.eval(obj_z3, model_completion=True))

    return LPSolution(objective=obj_val, values=values)


def lp_prob(
    mdp: model.Model[float],
    min: bool,
    zero_states: Iterable[model.State],
    one_states: Iterable[model.State],
) -> LP:
    """Build the LP formulation for MDP reachability as sympy expressions.

    :param mdp: The stormvogel MDP.
    :param min: True for min-reachability (maximize), False for max-reachability (minimize).
    :param zero_states: States fixed at value 0.
    :param one_states: States fixed at value 1 (target states).
    :returns: LP with objective and constraints as sympy expressions.
    """
    zero_set = frozenset(zero_states)
    one_set = frozenset(one_states)

    x = {s: sp.Symbol(f"x_{s.friendly_name}") for s in mdp.states}

    fixed_constraints: list[sp.Basic] = []
    transition_constraints: list[sp.Basic] = []
    nonneg_constraints: list[sp.Basic] = []

    for s in mdp.sorted_states:
        if s in one_set:
            fixed_constraints.append(sp.Eq(x[s], sp.Integer(1)))
        elif s in zero_set:
            fixed_constraints.append(sp.Eq(x[s], sp.Integer(0)))
        else:
            for _action, branch in s.choices:
                rhs = sum(prob * x[s_next] for prob, s_next in branch)
                if min:
                    transition_constraints.append(sp.Le(x[s], rhs))
                else:
                    transition_constraints.append(sp.Ge(x[s], rhs))
            nonneg_constraints.append(sp.Ge(x[s], sp.Integer(0)))

    free_vars = [
        x[s] for s in mdp.sorted_states if s not in one_set and s not in zero_set
    ]
    objective = sp.Add(*free_vars) if free_vars else sp.Integer(0)
    sense = "maximize" if min else "minimize"

    return LP(
        sense=sense,
        objective=objective,
        constraints=fixed_constraints + transition_constraints + nonneg_constraints,
    )


def lp_dual_prob(
    mdp: model.Model[float],
    zero_states: Iterable[model.State],
    one_states: Iterable[model.State],
) -> LP:
    """Occupancy-measure (dual) LP for single-objective reachability."""
    one_set = frozenset(one_states)
    zero_set = frozenset(zero_states)
    free_states = [
        s for s in mdp.sorted_states if s not in one_set and s not in zero_set
    ]
    free_set = frozenset(free_states)
    init = mdp.initial_state

    y: dict[tuple, sp.Expr] = {
        (s, action): sp.Symbol(f"y_{s.friendly_name}_{action.label}")
        for s in free_states
        for action, _ in s.choices
    }

    reach_terms: list[sp.Expr] = []
    for s in free_states:
        for action, branch in s.choices:
            reach = sum(prob for prob, t in branch if t in one_set)
            if reach:
                reach_terms.append(sp.Float(reach) * y[s, action])
    objective = sp.Add(*reach_terms) if reach_terms else sp.Integer(0)

    inflow: dict = {s: sp.Integer(0) for s in free_states}
    for s_prime in free_states:
        for action, branch in s_prime.choices:
            for prob, t in branch:
                if t in free_set:
                    inflow[t] = inflow[t] + sp.Float(prob) * y[s_prime, action]

    flow_constraints: list[sp.Basic] = []
    for s in free_states:
        outflow = sp.Add(*[y[s, action] for action, _ in s.choices])
        rhs = sp.Integer(1) if s is init else sp.Integer(0)
        flow_constraints.append(sp.Eq(outflow - inflow[s], rhs))

    nonneg: list[sp.Basic] = [sp.Ge(v, sp.Integer(0)) for v in y.values()]

    return LP(
        sense="maximize", objective=objective, constraints=flow_constraints + nonneg
    )


def lp_dual_maxreachprob(mdp: model.Model[float], target_label: str) -> LP:
    """Dual LP for maximum reachability probability to states labelled target_label."""
    target_states = mdp.state_labels[target_label]
    result = stormvogel.model_checking(
        mdp, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    assert result is not None
    return lp_dual_prob(mdp, result.filter_true(), target_states)


def lp_maxreachprob(mdp: model.Model[float], target_label: str) -> LP:
    """LP for maximum reachability probability to states labelled target_label."""
    target_states = mdp.state_labels[target_label]
    result = stormvogel.model_checking(
        mdp, f'Pmax<=0 [F "{target_label}"]', scheduler=False
    )
    assert result is not None
    return lp_prob(mdp, False, result.filter_true(), one_states=target_states)


def lp_minreachprob(mdp: model.Model[float], target_label: str) -> LP:
    """LP for minimum reachability probability to states labelled target_label."""
    target_states = mdp.state_labels[target_label]
    result = stormvogel.model_checking(
        mdp, f'Pmin<=0 [F "{target_label}"]', scheduler=False
    )
    assert result is not None
    return lp_prob(mdp, True, result.filter_true(), one_states=target_states)


def extract_policy_from_dual_lp(
    mdp: model.Model[float],
    sol: LPSolution,
) -> dict:
    """Extract a deterministic policy from an occupancy-measure (dual) LP solution.

    The dual LP variables ``y_{s,a}`` represent expected visit counts for each
    state-action pair under the optimal policy. For each state the action with
    the largest ``y`` value is chosen. States that are unreachable under the
    optimal policy (all ``y`` values zero or absent in *sol*) receive the first
    available non-empty action as a default; the choice does not affect the
    objective value.

    :param mdp: The MDP for which the dual LP was built.
    :param sol: Solution returned by :func:`solve_lp`.
    :returns: Dict mapping each state with at least one non-empty action to its
        chosen :class:`~stormvogel.model.Action`.
    """
    policy: dict = {}
    for s in mdp.states:
        best_action = None
        best_val = fractions.Fraction(-1)
        for action, _ in s.choices:
            if action is EmptyAction or action.label is None:
                continue
            sym = sp.Symbol(f"y_{s.friendly_name}_{action.label}")
            val = sol.values.get(sym, fractions.Fraction(0))
            if best_action is None or val > best_val:
                best_val = val
                best_action = action
        if best_action is not None:
            policy[s] = best_action
    return policy
