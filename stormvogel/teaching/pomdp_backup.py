"""Teaching module: belief-space Bellman operator and alpha-vector iteration for POMDPs.

Mirrors the structure of :mod:`stormvogel.teaching.bellman`:

- :func:`make_operator_pomdp_maxreachprob` is a factory that returns a
  :class:`BeliefBackupOperator` (analogous to ``make_operator_maxreachprob``).
- :class:`BeliefBackupOperator` has an :meth:`~BeliefBackupOperator.apply`
  method (analogous to ``BellmanOperator.apply``).
- :class:`AlphaVI` drives iteration step by step (analogous to ``VI``).

Only reachability objectives and deterministic state observations are
supported.  All arithmetic uses :class:`~fractions.Fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product as _iproduct
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from stormvogel.model.action import Action
    from stormvogel.model.model import Model
    from stormvogel.model.state import State

from stormvogel.teaching.belief import Belief


# ---------------------------------------------------------------------------
# Alpha vector
# ---------------------------------------------------------------------------


@dataclass
class AlphaVector:
    """Alpha vector with policy annotation.

    Represents a hyperplane over the belief simplex (via :attr:`values`) and
    the policy fragment that witnesses its value: which :attr:`action` to take
    and, for each possible next observation, which :class:`AlphaVector` to
    follow (:attr:`successors`).

    :param values: Mapping from model state to value.  States absent from the
        dict are treated as having value zero.
    :param action: The action recommended at the witness belief.  ``None`` for
        the initial leaf vector (horizon 0).
    :param successors: Mapping from observation alias to the next
        :class:`AlphaVector` to follow after that observation.
    """

    values: dict["State", Fraction]
    action: "Action | None" = None
    successors: dict[str, "AlphaVector"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def dot(alpha: AlphaVector, belief: Belief) -> Fraction:
    """Inner product α · b = Σ_s α(s) · b(s).

    :param alpha: An alpha vector.
    :param belief: A belief distribution over states.
    :returns: The inner product as an exact :class:`~fractions.Fraction`.
    """
    return sum(
        (alpha.values.get(s, Fraction(0)) * p for s, p in belief.items()),
        Fraction(0),
    )


def initial_alpha(model: "Model", target_label: str) -> AlphaVector:
    """Return the horizon-0 leaf alpha vector.

    Assigns value 1 to every state carrying *target_label* and 0 to all
    others.  Has no :attr:`~AlphaVector.action` and no
    :attr:`~AlphaVector.successors`.

    :param model: The POMDP model.
    :param target_label: Label identifying target (goal) states.
    :returns: A leaf :class:`AlphaVector`.
    """
    target_states = set(model.get_states_with_label(target_label))
    return AlphaVector(
        values={
            s: Fraction(1) if s in target_states else Fraction(0) for s in model.states
        }
    )


def value_function(alphas: list[AlphaVector], belief: Belief) -> Fraction:
    """Evaluate the PWLC value function at *belief*.

    :param alphas: The current set of alpha vectors.
    :param belief: A belief point.
    :returns: max_{α ∈ alphas} α · b.
    """
    return max(dot(a, belief) for a in alphas)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _obs_groups(model: "Model") -> dict[str, set["State"]]:
    """Partition model states by observation alias."""
    groups: dict[str, set["State"]] = {}
    for i, state in enumerate(model.states):
        obs = model.state_observations.get(state)
        alias = (
            model.observation_aliases[obs]
            if obs is not None and obs in model.observation_aliases
            else f"__s{i}"
        )
        groups.setdefault(alias, set()).add(state)
    return groups


def _build_trans_table(
    model: "Model",
) -> dict["State", dict[str, list[tuple[Fraction, "State"]]]]:
    """Pre-compute P(s' | s, a) as a nested dict.

    :returns: ``table[s][action_label] = [(prob, s'), ...]``.
        The empty string is used as the key for the :data:`EmptyAction`.
    """
    from stormvogel.model.action import EmptyAction

    table: dict["State", dict[str, list[tuple[Fraction, "State"]]]] = {}
    for state, choices in model.transitions.items():
        row: dict[str, list[tuple[Fraction, "State"]]] = {}
        for action, branch in choices:
            key = (
                action.label
                if action is not EmptyAction and action.label is not None
                else ""
            )
            row.setdefault(key, [])
            for prob, tgt in branch:
                row[key].append((Fraction(prob), tgt))
        table[state] = row
    return table


def _cond_alpha_values(
    alpha: AlphaVector,
    trans: dict["State", dict[str, list[tuple[Fraction, "State"]]]],
    action_label: str,
    group: set["State"],
) -> dict["State", Fraction]:
    """Compute the conditional alpha β^{a,o,α} for one observation group.

    β(s) = Σ_{s' ∈ group} P(s' | s, a) · α(s').

    :param alpha: The alpha vector to condition.
    :param trans: Pre-computed transition table.
    :param action_label: Action label *a*.
    :param group: The observation class G_o.
    :returns: Mapping from each source state *s* to its conditional value.
    """
    result: dict["State", Fraction] = {}
    for s, action_map in trans.items():
        v = sum(
            (
                prob * alpha.values.get(tgt, Fraction(0))
                for prob, tgt in action_map.get(action_label, [])
                if tgt in group
            ),
            Fraction(0),
        )
        result[s] = v
    return result


def _dominated(alpha: AlphaVector, others: list[AlphaVector]) -> bool:
    """Return ``True`` if *alpha* is component-wise dominated by some other vector."""
    states = list(alpha.values)
    for other in others:
        if other is alpha:
            continue
        if all(
            other.values.get(s, Fraction(0)) >= alpha.values.get(s, Fraction(0))
            for s in states
        ) and any(
            other.values.get(s, Fraction(0)) > alpha.values.get(s, Fraction(0))
            for s in states
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Bellman operator
# ---------------------------------------------------------------------------


class BeliefBackupOperator:
    """Belief-space Bellman backup operator for POMDP max-reachability.

    Analogous to :class:`~stormvogel.teaching.bellman.BellmanOperator`:
    :meth:`apply` maps the current set of alpha vectors to a new one by
    performing a point-based backup at each belief in :attr:`beliefs`.

    Construct via :func:`make_operator_pomdp_maxreachprob`.

    :param pomdp: The POMDP model.
    :param beliefs: Fixed belief points at which backups are performed.
    :param backup_fn: Inner per-belief backup callable
        ``(belief, alphas) → AlphaVector``.
    """

    def __init__(
        self,
        pomdp: "Model",
        beliefs: list[Belief],
        backup_fn: Callable[[Belief, list[AlphaVector]], AlphaVector],
    ) -> None:
        self._pomdp = pomdp
        self.beliefs = beliefs
        self._backup_fn = backup_fn

    def apply(self, alphas: list[AlphaVector]) -> list[AlphaVector]:
        """Apply one full backup pass across all belief points.

        For each belief in :attr:`beliefs`, calls the inner backup function
        and collects the resulting alpha vectors.  Dominated vectors are
        pruned before returning.

        :param alphas: The current alpha-vector set (value function approximation).
        :returns: Updated set of alpha vectors.
        """
        new_alphas = [self._backup_fn(b, alphas) for b in self.beliefs]
        pruned = [a for a in new_alphas if not _dominated(a, new_alphas)]
        return pruned if pruned else new_alphas[:1]


def make_operator_pomdp_maxreachprob(
    pomdp: "Model",
    target_label: str,
    beliefs: list[Belief],
) -> BeliefBackupOperator:
    """Return the Bellman backup operator for POMDP max-reachability.

    Pre-computes the observation partition, target indicators, and transition
    table from *pomdp* once.  The inner backup function mirrors the
    ``_bellman(s, values)`` pattern from
    :mod:`~stormvogel.teaching.bellman`: for each action it selects the best
    alpha vector per observation group (by maximising the conditional dot
    product with the current belief), combines the resulting conditional
    alphas into a new :class:`AlphaVector`, and annotates it with the optimal
    action and its successors.

    :param pomdp: A POMDP with deterministic state observations.
    :param target_label: Label identifying target (goal) states.
    :param beliefs: Belief points at which backups are performed.
    :returns: A :class:`BeliefBackupOperator`.
    :raises ValueError: If *pomdp* is not a POMDP.
    """
    from stormvogel.model.model import ModelType

    if pomdp.model_type != ModelType.POMDP:
        raise ValueError(
            f"make_operator_pomdp_maxreachprob requires a POMDP; got {pomdp.model_type}."
        )

    target_states: set["State"] = set(pomdp.get_states_with_label(target_label))
    obs_groups: dict[str, set["State"]] = _obs_groups(pomdp)
    trans = _build_trans_table(pomdp)

    action_labels: list[str] = sorted(
        {
            a.label
            for _s, choices in pomdp.transitions.items()
            for a, _ in choices
            if a.label is not None
        }
    )
    label_to_action: dict[str, "Action"] = {
        a.label: a
        for _s, choices in pomdp.transitions.items()
        for a, _ in choices
        if a.label is not None
    }

    def _bellman(belief: Belief, alphas: list[AlphaVector]) -> AlphaVector:
        """Backup at one belief point.

        For each action *a* and each observation group G_o, selects the
        alpha that maximises β^{a,o,α} · b (the conditional dot product
        with the full belief — valid because b_o is proportional).
        Combines the chosen conditional alphas, fixes target states at 1,
        and returns the best-action :class:`AlphaVector` with
        :attr:`~AlphaVector.action` and :attr:`~AlphaVector.successors` set.

        :param belief: The belief point at which to back up.
        :param alphas: Current alpha-vector set.
        :returns: New :class:`AlphaVector` with policy annotation.
        """
        best: AlphaVector | None = None
        best_val = Fraction(-1)

        for action_label in action_labels:
            # For each obs group, choose the alpha that maximises β^{a,o,α}·b.
            chosen: dict[str, AlphaVector] = {
                obs_alias: max(
                    alphas,
                    key=lambda α, al=action_label, grp=group: sum(
                        (
                            _cond_alpha_values(α, trans, al, grp).get(s, Fraction(0))
                            * p
                            for s, p in belief.items()
                        ),
                        Fraction(0),
                    ),
                )
                for obs_alias, group in obs_groups.items()
            }

            # Pre-compute conditional alpha values for the chosen vectors.
            cond: dict[str, dict["State", Fraction]] = {
                obs_alias: _cond_alpha_values(
                    chosen[obs_alias], trans, action_label, group
                )
                for obs_alias, group in obs_groups.items()
            }

            # Build new alpha: target states fixed at 1, others summed over groups.
            new_values: dict["State", Fraction] = {
                s: (
                    Fraction(1)
                    if s in target_states
                    else sum(
                        (c.get(s, Fraction(0)) for c in cond.values()), Fraction(0)
                    )
                )
                for s in pomdp.states
            }

            candidate = AlphaVector(
                values=new_values,
                action=label_to_action[action_label],
                successors=chosen,
            )
            val = dot(candidate, belief)
            if val > best_val:
                best_val = val
                best = candidate

        assert best is not None, "No named actions found in POMDP."
        return best

    return BeliefBackupOperator(pomdp, beliefs, _bellman)


class ExactBeliefBackupOperator:
    """Exact (non-point-based) belief backup operator for POMDP max-reachability.

    Instead of backing up at a fixed set of belief points, :meth:`apply`
    generates *all* alpha vectors that are optimal somewhere on the belief
    simplex: for each action and each assignment of current alpha vectors to
    observation groups, one candidate alpha vector is produced.  Dominated
    candidates are then pruned.

    This corresponds to the inner backup pass of exact PBVI / full alpha-vector
    iteration.  No belief grid is required.

    Construct via :func:`make_operator_pomdp_maxreachprob_exact`.
    """

    def __init__(
        self,
        pomdp: "Model",
        backup_fn: Callable[[list[AlphaVector]], list[AlphaVector]],
    ) -> None:
        self._pomdp = pomdp
        self._backup_fn = backup_fn

    def apply(self, alphas: list[AlphaVector]) -> list[AlphaVector]:
        """Generate all candidate alpha vectors and prune dominated ones.

        :param alphas: The current alpha-vector set.
        :returns: Updated set of alpha vectors.
        """
        new_alphas = self._backup_fn(alphas)
        pruned = [a for a in new_alphas if not _dominated(a, new_alphas)]
        return pruned if pruned else new_alphas[:1]


def make_operator_pomdp_maxreachprob_exact(
    pomdp: "Model",
    target_label: str,
) -> ExactBeliefBackupOperator:
    """Return the exact Bellman backup operator for POMDP max-reachability.

    Unlike :func:`make_operator_pomdp_maxreachprob`, no belief grid is needed.
    For each action and each assignment of current alpha vectors to observation
    groups, one candidate alpha vector is produced.  Duplicate candidates
    (arising when two alphas yield identical conditional vectors for an
    observation group) are removed before enumeration.

    :param pomdp: A POMDP with deterministic state observations.
    :param target_label: Label identifying target (goal) states.
    :returns: An :class:`ExactBeliefBackupOperator`.
    :raises ValueError: If *pomdp* is not a POMDP.
    """
    from stormvogel.model.model import ModelType

    if pomdp.model_type != ModelType.POMDP:
        raise ValueError(
            f"make_operator_pomdp_maxreachprob_exact requires a POMDP; got {pomdp.model_type}."
        )

    target_states: set["State"] = set(pomdp.get_states_with_label(target_label))
    obs_groups: dict[str, set["State"]] = _obs_groups(pomdp)
    trans = _build_trans_table(pomdp)

    action_labels: list[str] = sorted(
        {
            a.label
            for _s, choices in pomdp.transitions.items()
            for a, _ in choices
            if a.label is not None
        }
    )
    label_to_action: dict[str, "Action"] = {
        a.label: a
        for _s, choices in pomdp.transitions.items()
        for a, _ in choices
        if a.label is not None
    }

    def _all_candidates(alphas: list[AlphaVector]) -> list[AlphaVector]:
        candidates: list[AlphaVector] = []
        for action_label in action_labels:
            # For each obs group, collect distinct conditional alpha vectors.
            group_choices: list[tuple[str, list[tuple[AlphaVector, dict]]]] = []
            for obs_alias, group in obs_groups.items():
                seen_keys: set = set()
                choices: list[tuple[AlphaVector, dict]] = []
                for alpha in alphas:
                    cond = _cond_alpha_values(alpha, trans, action_label, group)
                    key = tuple(sorted((str(s.state_id), v) for s, v in cond.items()))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        choices.append((alpha, cond))
                group_choices.append((obs_alias, choices))

            obs_aliases = [g[0] for g in group_choices]
            choice_lists = [g[1] for g in group_choices]

            for combo in _iproduct(*choice_lists):
                chosen = {obs_aliases[i]: combo[i][0] for i in range(len(obs_aliases))}
                cond_vals = {
                    obs_aliases[i]: combo[i][1] for i in range(len(obs_aliases))
                }

                new_values: dict["State", Fraction] = {
                    s: (
                        Fraction(1)
                        if s in target_states
                        else sum(
                            (cv.get(s, Fraction(0)) for cv in cond_vals.values()),
                            Fraction(0),
                        )
                    )
                    for s in pomdp.states
                }
                candidates.append(
                    AlphaVector(
                        values=new_values,
                        action=label_to_action[action_label],
                        successors=chosen,
                    )
                )

        assert candidates, "No named actions found in POMDP."
        return candidates

    return ExactBeliefBackupOperator(pomdp, _all_candidates)


# ---------------------------------------------------------------------------
# Alpha-vector value iteration (mirrors VI from bellman.py)
# ---------------------------------------------------------------------------


class AlphaVI:
    """Alpha-vector value iteration for POMDP max-reachability.

    Analogous to :class:`~stormvogel.teaching.bellman.VI`: holds an operator
    and drives iteration one step at a time.

    :param operator: A :class:`BeliefBackupOperator` or
        :class:`ExactBeliefBackupOperator`.
    :param initial_alphas: Starting set of alpha vectors (typically a single
        :func:`initial_alpha`).
    """

    def __init__(
        self,
        operator: "Union[BeliefBackupOperator, ExactBeliefBackupOperator]",
        initial_alphas: list[AlphaVector],
    ) -> None:
        self._operator = operator
        self._alphas = list(initial_alphas)

    def step(self) -> list[AlphaVector]:
        """Apply one backup pass and update the stored alpha vectors.

        :returns: The updated alpha-vector set.
        """
        self._alphas = self._operator.apply(self._alphas)
        return self._alphas

    @property
    def current_alphas(self) -> list[AlphaVector]:
        """The current alpha-vector set."""
        return self._alphas


# ---------------------------------------------------------------------------
# MDP upper bound and QMDP (require stormpy via stormvogel.model_checking)
# ---------------------------------------------------------------------------


def _mdp_values(pomdp: "Model", target_label: str) -> dict["State", Fraction]:
    """Return the optimal fully-observable MDP value for every POMDP state.

    Copies *pomdp*, strips observations, runs max-reachability model checking,
    and maps results back to the original state objects via their stable UUIDs.

    :param pomdp: Source POMDP.
    :param target_label: Label identifying target states.
    :returns: Mapping from each state of *pomdp* to its MDP value.
    """
    import stormvogel as sv

    mdp = pomdp.copy().make_fully_observable()
    result = sv.model_checking(mdp, f'Pmax=? [F "{target_label}"]')
    assert result is not None
    id_to_val = {s.state_id: result.at(s) for s in mdp.states}
    return {
        s: Fraction(id_to_val[s.state_id]).limit_denominator(10**9)
        for s in pomdp.states
    }


def mdp_bound_alpha(pomdp: "Model", target_label: str) -> AlphaVector:
    """Return the fully-observable MDP value function as an alpha vector.

    The MDP upper bound lifts the optimal fully-observable value $V_\\text{MDP}$
    to the belief space by expectation:

    .. math::

        V_\\text{MDP}(b) = \\sum_s b(s)\\, V_\\text{MDP}(s)
                         = \\alpha_\\text{MDP} \\cdot b.

    Because any observation-based policy is also an MDP policy, this is an
    upper bound: $V^*(b) \\leq V_\\text{MDP}(b)$.

    Requires stormpy (via :func:`stormvogel.model_checking`).

    :param pomdp: The POMDP model.
    :param target_label: Label identifying target states.
    :returns: An :class:`AlphaVector` whose values equal $V_\\text{MDP}(s)$.
    """
    return AlphaVector(values=_mdp_values(pomdp, target_label))


def qmdp_alphas(pomdp: "Model", target_label: str) -> list[AlphaVector]:
    """Return the QMDP alpha vectors, one per named action.

    QMDP assumes the state becomes fully observable after the *first* action,
    so the agent picks $a$ to maximise the expected MDP value of the successor:

    .. math::

        V_\\text{QMDP}(b)
        = \\max_{a} \\sum_s b(s) \\sum_{s'} P(s'\\mid s,a)\\, V_\\text{MDP}(s').

    Each alpha vector encodes one action's contribution:

    .. math::

        \\alpha_a(s) = \\mathbb{1}[s \\in T]
            + \\mathbb{1}[s \\notin T]
            \\cdot \\sum_{s'} P(s'\\mid s,a)\\, V_\\text{MDP}(s').

    The value function is $V_\\text{QMDP}(b) = \\max_a \\alpha_a \\cdot b$, and
    satisfies $V^*(b) \\leq V_\\text{QMDP}(b) \\leq V_\\text{MDP}(b)$.

    Requires stormpy (via :func:`stormvogel.model_checking`).

    :param pomdp: The POMDP model.
    :param target_label: Label identifying target states.
    :returns: One :class:`AlphaVector` per named action, in sorted label order.
    :raises ValueError: If *pomdp* is not a POMDP.
    """
    from stormvogel.model.model import ModelType

    if pomdp.model_type != ModelType.POMDP:
        raise ValueError(f"qmdp_alphas requires a POMDP; got {pomdp.model_type}.")

    mdp_vals = _mdp_values(pomdp, target_label)
    target_states: set["State"] = set(pomdp.get_states_with_label(target_label))
    trans = _build_trans_table(pomdp)

    action_labels: list[str] = sorted(
        {
            a.label
            for _s, choices in pomdp.transitions.items()
            for a, _ in choices
            if a.label is not None
        }
    )
    label_to_action: dict[str, "Action"] = {
        a.label: a
        for _s, choices in pomdp.transitions.items()
        for a, _ in choices
        if a.label is not None
    }

    alphas: list[AlphaVector] = []
    for action_label in action_labels:
        values: dict["State", Fraction] = {
            s: (
                Fraction(1)
                if s in target_states
                else sum(
                    (
                        prob * mdp_vals.get(tgt, Fraction(0))
                        for prob, tgt in trans.get(s, {}).get(action_label, [])
                    ),
                    Fraction(0),
                )
            )
            for s in pomdp.states
        }
        alphas.append(AlphaVector(values=values, action=label_to_action[action_label]))

    return alphas


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_alpha_vectors(
    alphas: list[AlphaVector],
    s_left: "State",
    s_right: "State",
    left_label: str | None = None,
    right_label: str | None = None,
    n_points: int = 300,
    ax=None,
):
    """Plot alpha vectors over the 1-D belief simplex spanned by two states.

    Each alpha vector is drawn as a line over the belief axis.  The x-axis
    represents the probability of being in *s_right* (0 = certainly *s_left*,
    1 = certainly *s_right*).  Lines are colored by action label.  The upper
    envelope (the PWLC value function) is overlaid as a thick black curve.

    :param alphas: Alpha vectors to plot.
    :param s_left: State at the left extreme (belief = 0).
    :param s_right: State at the right extreme (belief = 1).
    :param left_label: Display name for *s_left* (defaults to friendly name or id).
    :param right_label: Display name for *s_right* (defaults to friendly name or id).
    :param n_points: Number of sample points along the belief axis.
    :param ax: Matplotlib axes to draw on.  A new figure is created if ``None``.
    :returns: The axes object.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _name(s, override):
        if override is not None:
            return override
        fn = getattr(s, "friendly_name", None)
        return fn if fn else str(s.state_id)

    left_label = _name(s_left, left_label)
    right_label = _name(s_right, right_label)

    if ax is None:
        _, ax = plt.subplots()

    xs = np.linspace(0.0, 1.0, n_points)  # x = Pr(s_right)

    # Assign a color to each distinct action label.
    action_labels = sorted(
        {a.action.label if a.action else None for a in alphas},
        key=lambda lbl: ("" if lbl is None else lbl),
    )
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    color_map: dict = {
        lbl: ("gray" if lbl is None else palette[i % len(palette)])
        for i, lbl in enumerate(action_labels)
    }

    seen: set = set()
    ys_all = []
    for alpha in alphas:
        v_left = float(alpha.values.get(s_left, Fraction(0)))
        v_right = float(alpha.values.get(s_right, Fraction(0)))
        ys = v_left * (1.0 - xs) + v_right * xs
        ys_all.append(ys)

        lbl = alpha.action.label if alpha.action else None
        display = lbl if lbl is not None else "initial"
        ax.plot(
            xs,
            ys,
            color=color_map[lbl],
            linestyle="--" if lbl is None else "-",
            alpha=0.55,
            label=display if display not in seen else None,
        )
        seen.add(display)

    # Upper envelope.
    envelope = np.max(ys_all, axis=0)
    ax.plot(xs, envelope, color="black", linewidth=2.0, label="value function")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([left_label or "", right_label or ""])
    ax.set_xlabel("belief")
    ax.set_ylabel("value")
    ax.legend()
    return ax


def plot_alpha_vector_iterations(
    iterations: list[list[AlphaVector]],
    s_left: "State",
    s_right: "State",
    left_label: str | None = None,
    right_label: str | None = None,
    start_k: int = 1,
    n_points: int = 300,
    figsize: "tuple[float, float] | None" = None,
):
    """Plot alpha-vector sets for successive VI steps as side-by-side panels.

    Each panel shows the alpha vectors after one Bellman backup step, with the
    step number *k* as the panel title.  Pass the results of repeated
    :meth:`AlphaVI.step` calls::

        vi = AlphaVI(op, [initial_alpha(model, "target")])
        iters = [vi.step() for _ in range(3)]
        plot_alpha_vector_iterations(iters, s1, s2)

    :param iterations: List of alpha-vector sets, one per step.
    :param s_left: State at the left extreme of the belief axis (belief = 0).
    :param s_right: State at the right extreme (belief = 1).
    :param left_label: Override display name for *s_left*.
    :param right_label: Override display name for *s_right*.
    :param start_k: Step index label for the first panel (default 1).
    :param n_points: Sample resolution passed to :func:`plot_alpha_vectors`.
    :param figsize: Figure size override; defaults to (4·n, 4).
    :returns: ``(fig, axes)`` tuple.
    """
    import matplotlib.pyplot as plt

    n = len(iterations)
    if figsize is None:
        figsize = (4.0 * n, 4.0)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for i, (alphas, ax) in enumerate(zip(iterations, axes)):
        plot_alpha_vectors(
            alphas,
            s_left,
            s_right,
            left_label=left_label,
            right_label=right_label,
            n_points=n_points,
            ax=ax,
        )
        ax.set_title(f"k = {start_k + i}")
        if i > 0:
            ax.set_ylabel("")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    fig.tight_layout()
    return fig, axes


def plot_value_function_comparison(
    alphas_list: "list[tuple[list[AlphaVector], str]]",
    s_left: "State",
    s_right: "State",
    left_label: str | None = None,
    right_label: str | None = None,
    n_points: int = 300,
    ax=None,
):
    """Plot the upper envelopes of multiple alpha-vector sets in one panel.

    Each entry in *alphas_list* is drawn as a single PWLC curve — the upper
    envelope of its alpha vectors.  This is useful for comparing bounds, e.g.
    the exact-VI lower bound against QMDP and the plain MDP upper bound.

    :param alphas_list: List of ``(alphas, label)`` pairs to compare.
    :param s_left: State at the left extreme of the belief axis (belief = 0).
    :param s_right: State at the right extreme (belief = 1).
    :param left_label: Override display name for *s_left*.
    :param right_label: Override display name for *s_right*.
    :param n_points: Sample resolution for each curve.
    :param ax: Matplotlib axes to draw on.  A new figure is created if ``None``.
    :returns: The axes object.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _name(s, override):
        if override is not None:
            return override
        fn = getattr(s, "friendly_name", None)
        return fn if fn else str(s.state_id)

    left_label = _name(s_left, left_label)
    right_label = _name(s_right, right_label)

    if ax is None:
        _, ax = plt.subplots()

    xs = np.linspace(0.0, 1.0, n_points)
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (alphas, label) in enumerate(alphas_list):
        ys_all = np.array(
            [
                [
                    float(alpha.values.get(s_left, Fraction(0))) * (1.0 - x)
                    + float(alpha.values.get(s_right, Fraction(0))) * x
                    for x in xs
                ]
                for alpha in alphas
            ]
        )
        envelope = ys_all.max(axis=0)
        ax.plot(xs, envelope, color=palette[i % len(palette)], linewidth=2, label=label)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([left_label or "", right_label or ""])
    ax.set_xlabel("belief")
    ax.set_ylabel("value")
    ax.legend()
    return ax
