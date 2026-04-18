"""Policy iteration for MDPs, using exact sympy-based DTMC evaluation.

Each step induces a DTMC from the current scheduler, evaluates it exactly via
:mod:`stormvogel.teaching.dtmc_evaluation`, then improves the scheduler
greedily. Results are exact sympy rationals throughout.
"""

from typing import Iterable

import sympy as sp

import stormvogel.model as model
import stormvogel.result as result
from stormvogel.teaching.dtmc_evaluation import (
    solve_expected_reward,
    solve_reachability,
)
from stormvogel.teaching.qualitative_mdp import compute_spos, compute_sposmin


def initial_scheduler(
    mdp: model.Model,
    one_states: Iterable[model.State],
    minimize: bool,
) -> result.Scheduler:
    """Return a suitable starting scheduler for policy iteration.

    Uses a one-step Bellman lookahead from the zero vector (one_states fixed
    to 1, all others 0) to bias the initial scheduler:

    * **minimize** — picks the action with the *lowest* one-step expected
      value, giving a pessimistic start that avoids the target.  This ensures
      the induced DTMC has no spurious fixed points due to cycles.
    * **maximize** — picks the action with the *highest* one-step expected
      value, giving an optimistic start that heads towards the target.

    :param mdp: The MDP.
    :param one_states: States with reachability probability 1 (target states).
    :param minimize: If True, build a pessimistic scheduler; otherwise
        an optimistic one.
    :returns: A :class:`~stormvogel.result.Scheduler` for *mdp*.
    """
    one_ids = {s.state_id for s in one_states}
    v0 = {
        s: sp.Integer(1) if s.state_id in one_ids else sp.Integer(0) for s in mdp.states
    }
    opt = min if minimize else max
    taken: dict[model.State, model.Action] = {}
    for s in mdp.states:
        best_action, _ = opt(
            s.choices,
            key=lambda ab: sum(
                sp.nsimplify(prob) * v0[s_next] for prob, s_next in ab[1]
            ),
        )
        taken[s] = best_action
    return result.Scheduler(mdp, taken)


def policy_improvement(
    mdp: model.Model,
    values: dict[model.State, sp.Expr],
    one_states: Iterable[model.State],
    minimize: bool,
    current_scheduler: result.Scheduler,
) -> result.Scheduler:
    """Return the improved scheduler for *values*.

    For each state, pick the action that minimises (or maximises) the expected
    value under the current value function.  If the current action is already
    optimal (ties included), it is kept — only a *strict* improvement causes a
    switch.  This tie-breaking rule guarantees that PI terminates: every
    iteration that changes the scheduler strictly improves the objective.

    :param mdp: The MDP.
    :param values: Current value for each state (from DTMC evaluation).
    :param one_states: States fixed to value 1; any action is kept for them.
    :param minimize: If True, minimize; otherwise maximize.
    :param current_scheduler: The scheduler used to produce *values*.
    :returns: A new :class:`~stormvogel.result.Scheduler`.
    """
    one = set(one_states)
    opt = min if minimize else max
    taken: dict[model.State, model.Action] = {}

    for s in mdp.states:
        if s in one:
            taken[s] = current_scheduler.get_action_at_state(s)
            continue

        current_action = current_scheduler.get_action_at_state(s)

        def _expected(branch) -> sp.Expr:
            return sum(sp.nsimplify(prob) * values[s_next] for prob, s_next in branch)  # type: ignore[return-value]

        best_action, best_branch = opt(s.choices, key=lambda ab: _expected(ab[1]))
        current_branch = s.get_outgoing_transitions(current_action)

        best_val = _expected(best_branch)
        curr_val = _expected(current_branch)

        if (minimize and best_val < curr_val) or (not minimize and best_val > curr_val):
            taken[s] = best_action
        else:
            taken[s] = current_action

    return result.Scheduler(mdp, taken)


class PI:
    """Policy iteration for reachability probabilities on an MDP.

    Usage::

        sched0 = initial_scheduler(mdp, one_states=[s_target], minimize=False)
        pi = PI(mdp, sched0, one_states=[s_target], minimize=False)
        while not pi.has_converged():
            scheduler, values = pi.step()

    :param mdp: The MDP to optimise.
    :param scheduler: Starting scheduler.  Use :func:`initial_scheduler` to
        obtain a well-initialised one.
    :param one_states: States with reachability probability 1 (target states).
    :param minimize: If True, minimize reachability; otherwise maximize.
    """

    def __init__(
        self,
        mdp: model.Model,
        scheduler: result.Scheduler,
        one_states: Iterable[model.State],
        minimize: bool,
        reward_model: model.RewardModel | None = None,
        discount: sp.Expr = sp.Integer(1),
    ):
        self._mdp = mdp
        self._scheduler = scheduler
        self._one_states = list(one_states)
        self._minimize = minimize
        self._reward_model = reward_model
        self._discount = sp.nsimplify(discount)
        self._values: dict[model.State, sp.Expr] | None = None
        self._converged = False
        if reward_model is None:
            # Reachability: fix Sminzero states to 0 for minimisation.
            if minimize:
                sposmin_states = compute_sposmin(mdp, list(one_states))
                self._mdp_zero_ids: frozenset = frozenset(
                    s.state_id for s in mdp.states if s not in sposmin_states
                )
            else:
                self._mdp_zero_ids = frozenset()
        elif self._discount == sp.Integer(1):
            # Undiscounted expected reward: ill-defined when terminal is
            # unreachable.  Catch the bad cases early with a clear message.
            if minimize:
                bad = frozenset(mdp.states) - compute_spos(mdp, list(one_states))
            else:
                bad = frozenset(mdp.states) - compute_sposmin(mdp, list(one_states))
            if bad:
                names = sorted(s.friendly_name or str(s.state_id) for s in bad)
                raise ValueError(
                    f"Undiscounted expected reward is ill-defined: states {names} "
                    f"{'cannot reach' if minimize else 'can avoid'} the terminal "
                    f"set under {'some' if not minimize else 'every'} policy."
                )
            self._mdp_zero_ids = frozenset()
        else:
            # Discounted expected reward: always well-defined.
            self._mdp_zero_ids = frozenset()

    @property
    def current_scheduler(self) -> result.Scheduler:
        return self._scheduler

    @property
    def current_values(self) -> dict[model.State, sp.Expr] | None:
        """Values from the last evaluation step, or None before the first step."""
        return self._values

    def has_converged(self) -> bool:
        """True after a step where the scheduler did not change."""
        return self._converged

    def step(self) -> tuple[result.Scheduler, dict[model.State, sp.Expr]]:
        """Run one policy-iteration step: evaluate then improve.

        :returns: The new scheduler and the exact values under the old one.
        :raises RuntimeError: If the scheduler cannot induce a DTMC.
        """
        dtmc = self._scheduler.generate_induced_dtmc()
        if dtmc is None:
            raise RuntimeError("Could not induce a DTMC from the current scheduler.")

        if self._reward_model is not None:
            dtmc_rewards = model.RewardModel(
                self._reward_model.name,
                dtmc,
                {
                    dtmc.get_state_by_id(s.state_id): v
                    for s, v in self._reward_model.rewards.items()
                },
            )
            dtmc_values = solve_expected_reward(
                dtmc, dtmc_rewards, self._one_states, self._discount
            )
        else:
            dtmc_zeros = [dtmc.get_state_by_id(uid) for uid in self._mdp_zero_ids]
            dtmc_values = solve_reachability(
                dtmc, self._one_states, zero_states=dtmc_zeros or None
            )

        # Lift DTMC values back to MDP states by matching UUIDs.
        id_to_value = {s.state_id: v for s, v in dtmc_values.items()}
        mdp_values: dict[model.State, sp.Expr] = {
            s: id_to_value[s.state_id] for s in self._mdp.states
        }

        new_scheduler = policy_improvement(
            self._mdp,
            mdp_values,
            self._one_states,
            self._minimize,
            self._scheduler,
        )

        self._converged = new_scheduler.taken_actions == self._scheduler.taken_actions
        self._scheduler = new_scheduler
        self._values = mdp_values
        return self._scheduler, self._values


def visualise_pi_iterations(pi: "PI", max_steps: int = 50):
    """Run *pi* to convergence and return a pandas DataFrame of each step.

    Each column group ``step k`` shows the scheduler evaluated at that step
    and the exact values it produced — so the action and value in the same
    column are always consistent.

    Rows are states; columns are a two-level MultiIndex
    ``(step k, "value")`` / ``(step k, "action")``.

    :param pi: A :class:`PI` instance (not yet converged).
    :param max_steps: Safety limit on the number of steps.
    :returns: A :class:`pandas.DataFrame` indexed by state friendly name.
    """
    import pandas as pd  # optional dependency — only needed for display

    snapshots: list[tuple[result.Scheduler, dict[model.State, sp.Expr]]] = []
    step = 0
    while not pi.has_converged() and step < max_steps:
        sched_k = pi.current_scheduler
        _, values_k = pi.step()
        snapshots.append((sched_k, values_k))
        step += 1

    if not snapshots:
        return pd.DataFrame()

    states = sorted(snapshots[0][1].keys(), key=lambda st: st.friendly_name or "")
    columns = pd.MultiIndex.from_tuples(
        [
            (f"step {i}", col)
            for i, _ in enumerate(snapshots)
            for col in ("value", "action")
        ]
    )
    rows = []
    for s in states:
        row = []
        for sched, values in snapshots:
            row.append(str(values[s]))
            row.append(sched.get_action_at_state(s).label)
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.index = [s.friendly_name for s in states]
    df.index.name = "state"
    return df
