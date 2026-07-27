"""Teaching module: POMDP belief type and belief computations.

Provides the canonical :class:`Belief` type and exact Bayesian belief
tracking for POMDPs with deterministic state observations.  All arithmetic
uses :class:`~fractions.Fraction`.

Typical usage::

    from stormvogel.teaching.belief import Belief, initial_belief, belief_trace

    b0 = initial_belief(pomdp, "z")
    beliefs = belief_trace(pomdp, b0, [("b", "z"), ("a", "z_target")])
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Union
from uuid import UUID

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.observation import Observation
    from stormvogel.model.state import State


#: Decimal places that floating-point belief values are rounded to when forming
#: a belief's identity key.  Belief search depends on recognising a belief it
#: has already seen, and two runs of floating-point arithmetic that are
#: mathematically equal routinely differ in the last bits.  Without rounding
#: those would be distinct nodes and the search would revisit the same belief
#: forever.  :class:`~fractions.Fraction` values are exact and used as-is.
FLOAT_KEY_PRECISION = 12

#: Relative gain a floating-point path probability must show before a search
#: accepts it as a genuinely better path.  Two float computations of the same
#: path routinely differ in the last bits, and a search that treats every such
#: difference as an improvement reopens the same belief forever.  Exact
#: arithmetic needs no tolerance and uses zero.
FLOAT_IMPROVEMENT_TOLERANCE = 1e-12

#: A belief probability: an exact :class:`~fractions.Fraction` by default, or a
#: ``float`` when a belief is tracked in approximate arithmetic.
BeliefValue = Union[Fraction, float]


def _key_value(probability):
    """Return the identity-key form of a belief probability."""
    if isinstance(probability, Fraction):
        return probability
    return round(float(probability), FLOAT_KEY_PRECISION)


class Belief(Mapping["State", "BeliefValue"]):
    """Probability distribution over POMDP states.

    Implements the :class:`~collections.abc.Mapping` interface over
    ``State → Fraction``, so ``b[s]``, ``b.get(s, 0)``, ``b.items()``,
    etc. all work directly.  Zero-probability states are silently dropped.

    Values are normally :class:`~fractions.Fraction`, which makes belief
    identity exact.  ``float`` values are also accepted, in which case two
    beliefs count as the same node when their probabilities agree to
    :data:`FLOAT_KEY_PRECISION` decimal places.

    :param dist: Mapping from POMDP states to their belief probabilities.
    """

    def __init__(self, dist: "Mapping[State, BeliefValue]") -> None:
        self.dist: dict["State", BeliefValue] = {s: p for s, p in dist.items() if p > 0}
        self._key: tuple[tuple[UUID, BeliefValue], ...] = tuple(
            sorted(
                ((s.state_id, _key_value(p)) for s, p in self.dist.items()),
                key=lambda x: x[0],
            )
        )

    # --- Mapping interface ---------------------------------------------------

    def __getitem__(self, key: "State") -> "BeliefValue":
        return self.dist[key]

    def __iter__(self) -> Iterator["State"]:
        return iter(self.dist)

    def __len__(self) -> int:
        return len(self.dist)

    # --- Identity ------------------------------------------------------------

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Belief):
            return self._key == other._key
        return NotImplemented

    def __repr__(self) -> str:
        return f"Belief({self.dist!r})"

    # --- LaTeX / Jupyter display ---------------------------------------------

    @staticmethod
    def _state_name(state: "State") -> str:
        """Best short name for *state*: friendly_name > first label > index."""
        if state.friendly_name is not None:
            return state.friendly_name
        labels = list(state.labels)
        if labels:
            return labels[0]
        return str(state.state_id)

    @staticmethod
    def _fraction_latex(f: "BeliefValue") -> str:
        if not isinstance(f, Fraction):
            return f"{float(f):.4g}"
        if f.denominator == 1:
            return str(f.numerator)
        return rf"\tfrac{{{f.numerator}}}{{{f.denominator}}}"

    def _repr_latex_(self) -> str:
        if not self.dist:
            return r"$\emptyset$"
        entries = ",\\quad ".join(
            rf"{self._state_name(s)} \mapsto {self._fraction_latex(p)}"
            for s, p in self.dist.items()
        )
        return rf"$\textstyle\left\{{\, {entries} \,\right\}}$"

    @classmethod
    def normalize(cls, unnorm: "Mapping[State, BeliefValue]") -> "Belief":
        """Normalize *unnorm* to a probability distribution and return a Belief.

        :param unnorm: Unnormalized weights (non-negative, at least one > 0).
        :raises ValueError: If all weights are zero.
        """
        total = sum(unnorm.values(), Fraction(0))
        if total == 0:
            raise ValueError("Cannot normalize a zero-weight distribution.")
        return cls({s: v / total for s, v in unnorm.items()})


class BeliefTransitions:
    """Precomputed transition/observation index of a POMDP.

    Walking :attr:`~stormvogel.model.model.Model.transitions` once and reusing
    the result makes repeated belief updates cheap, which matters for anything
    that explores the belief space (see
    :mod:`stormvogel.teaching.belief_mdp` and
    :mod:`stormvogel.teaching.belief_search`).

    Works for HMMs as well as POMDPs.  An HMM simply has one unlabelled
    choice per state, so :meth:`actions` returns ``[""]`` everywhere and
    :meth:`successors` degenerates to the forward pass of a hidden Markov
    model: the belief splits over the observations reachable in one step.

    Set *exact* to ``False`` to track beliefs in ``float`` instead of
    :class:`~fractions.Fraction`.  Exact arithmetic keeps belief identity
    exact, but the denominators grow with every step of a trace, so the cost
    per belief rises the deeper the search goes; floats keep it flat at the
    price of rounding (see :data:`FLOAT_KEY_PRECISION`).

    :param pomdp: A POMDP or HMM with deterministic (non-stochastic) state
        observations.
    :param exact: Whether to use exact rational arithmetic.
    :raises ValueError: If *pomdp* does not support observations, or if any
        state has a stochastic observation.
    """

    def __init__(self, pomdp: "Model", exact: bool = True) -> None:
        from stormvogel.model.distribution import Distribution

        if not pomdp.supports_observations():
            raise ValueError(f"Expected a POMDP or HMM; got {pomdp.model_type}.")

        for state in pomdp.states:
            if isinstance(pomdp.state_observations.get(state), Distribution):
                raise ValueError(
                    f"State {state!r} has a stochastic observation; "
                    "belief tracking requires deterministic state observations."
                )

        self.pomdp = pomdp
        #: Whether beliefs are tracked in exact rational arithmetic.
        self.exact = exact
        #: Zero of the arithmetic in use, for seeding sums.
        self.zero = Fraction(0) if exact else 0.0
        convert = Fraction if exact else float
        #: Observation of each POMDP state (``None`` if it has none).
        self.obs_of: dict["State", "Observation | None"] = {
            s: pomdp.state_observations.get(s) for s in pomdp.states
        }
        #: ``trans[state][action_label]`` → ``[(probability, target), ...]``.
        #: The label of the empty action is the empty string.
        self.trans: dict["State", dict[str, list[tuple[BeliefValue, "State"]]]] = {}
        #: Action labels available in each state.
        self.actions_of: dict["State", set[str]] = {}
        for state, choices in pomdp.transitions.items():
            per_action: dict[str, list[tuple[BeliefValue, "State"]]] = {}
            for action, branch in choices:
                label = action.label if action.label is not None else ""
                per_action[label] = [(convert(val), tgt) for val, tgt in branch]
            self.trans[state] = per_action
            self.actions_of[state] = set(per_action)

    def actions(self, belief: Belief) -> list[str]:
        """Return the action labels available in *every* support state of *belief*.

        A belief only has a well-defined choice of action if all states it
        might be in offer that action, so the intersection is taken.

        :param belief: A belief over POMDP states.
        :returns: Sorted list of commonly available action labels.
        """
        action_sets = [self.actions_of.get(s, set()) for s in belief]
        if not action_sets:
            return []
        return sorted(action_sets[0].intersection(*action_sets[1:]))

    def successors(
        self, belief: Belief, action_label: str
    ) -> "list[tuple[BeliefValue, Observation | None, Belief]]":
        """Expand *belief* under *action_label*, one entry per reachable observation.

        Applies the Bayesian belief update
        :math:`b'(s') \\propto \\sum_s P(s' \\mid s, a)\\, b(s)` to the
        successor states grouped by their observation.  The returned
        probabilities are the observation probabilities
        :math:`P(o \\mid b, a)` and sum to 1 whenever *action_label* is
        available in the whole support of *belief*.

        :param belief: The current belief.
        :param action_label: Label of the action taken (empty string for the
            empty action).
        :returns: List of ``(P(o | b, a), observation, updated belief)``
            triples, one per observation reachable with positive probability.
        """
        # Unnormalised weight of every reachable successor state.
        unnorm: dict["State", BeliefValue] = {}
        for s, b_s in belief.items():
            for prob, tgt in self.trans.get(s, {}).get(action_label, []):
                unnorm[tgt] = unnorm.get(tgt, self.zero) + b_s * prob

        # Group successor states by the observation they emit.
        groups: dict["Observation | None", dict["State", BeliefValue]] = {}
        for tgt, weight in unnorm.items():
            group = groups.setdefault(self.obs_of[tgt], {})
            group[tgt] = group.get(tgt, self.zero) + weight

        result: "list[tuple[BeliefValue, Observation | None, Belief]]" = []
        for obs, group in groups.items():
            obs_prob = sum(group.values(), self.zero)
            if obs_prob > 0:
                result.append((obs_prob, obs, Belief.normalize(group)))
        return result

    def observation(self, belief: Belief) -> "Observation | None":
        """Return the observation shared by the whole support of *belief*.

        :param belief: A belief over POMDP states.
        :returns: The common :class:`~stormvogel.model.observation.Observation`,
            or ``None`` if the support is empty or its states disagree.
        """
        observations = {self.obs_of.get(s) for s in belief}
        if len(observations) == 1:
            return observations.pop()
        return None


def initial_belief(pomdp: "Model", obs_alias: str) -> Belief:
    """Derive the initial belief from the POMDP's initial-state distribution.

    The initial state is assumed to have a single EmptyAction transition that
    encodes the prior distribution over states.  The resulting distribution is
    filtered to states whose observation matches *obs_alias* and then
    normalised.

    :param pomdp: The POMDP model.
    :param obs_alias: Observation alias that filters which successor states
        belong to the initial belief support.
    :returns: Normalised belief over states with observation *obs_alias*.
    :raises ValueError: If no states with *obs_alias* are reachable from the
        initial state, or if the initial state has no EmptyAction transition.
    """
    from stormvogel.model.action import EmptyAction

    init = pomdp.initial_state
    obs_states = pomdp.compute_states_per_observation()[
        pomdp.get_observation(obs_alias)
    ]

    unnorm: defaultdict["State", BeliefValue] = defaultdict(Fraction)
    for action, branch in pomdp.transitions[init]:
        if action is not EmptyAction:
            continue
        for prob, tgt in branch:
            if tgt in obs_states:
                unnorm[tgt] += Fraction(prob)

    if not unnorm:
        raise ValueError(
            f"No states with observation '{obs_alias}' are reachable from the "
            f"initial state under the EmptyAction transition."
        )
    return Belief.normalize(unnorm)


def belief_update(
    pomdp: "Model",
    belief: Belief,
    action_label: str,
    obs_alias: str,
) -> Belief:
    """Compute the updated belief after taking an action and receiving an observation.

    Applies the standard Bayesian filter for POMDPs with deterministic
    observations::

        b'(s') ∝  Σ_s  P(s' | s, a) · b(s)   if obs(s') = o
                  0                             otherwise

    :param pomdp: The POMDP model.
    :param belief: Current belief distribution over states.
    :param action_label: Label of the action taken.
    :param obs_alias: Alias of the observation received after the action.
    :returns: Updated, normalised belief.
    :raises ValueError: If the observation is unreachable from the current
        belief under the given action.
    """
    from stormvogel.model.action import EmptyAction

    obs_states = pomdp.compute_states_per_observation()[
        pomdp.get_observation(obs_alias)
    ]

    unnorm: defaultdict["State", BeliefValue] = defaultdict(Fraction)
    for state, choices in pomdp.transitions.items():
        b_s = belief.get(state, Fraction(0))
        if b_s == 0:
            continue
        for action, branch in choices:
            lbl = action.label if action is not EmptyAction else None
            if lbl != action_label:
                continue
            for prob, tgt in branch:
                if tgt in obs_states:
                    unnorm[tgt] += Fraction(prob) * b_s

    if not unnorm:
        raise ValueError(
            f"Belief update failed: observation '{obs_alias}' is unreachable "
            f"from the current belief under action '{action_label}'."
        )
    return Belief.normalize(unnorm)


def belief_table(
    beliefs: "list[Belief]",
    trace: "list[tuple[str, str]]",
) -> None:
    """Render a belief trace as an HTML table in a Jupyter notebook.

    Displays a table with columns Step / Action / Observation / Belief,
    where the belief column uses the LaTeX representation of each
    :class:`Belief`.  Row 0 shows the initial belief (no action/observation).

    :param beliefs: List of beliefs as returned by :func:`belief_trace`
        (length ``len(trace) + 1``).
    :param trace: Sequence of ``(action_label, obs_alias)`` pairs passed to
        :func:`belief_trace`.
    """
    try:
        from IPython.display import HTML, display
    except ImportError as e:
        raise ImportError("belief_table requires IPython (pip install ipython).") from e

    rows = [("", "", beliefs[0])]
    for (action, obs), b in zip(trace, beliefs[1:]):
        rows.append((action, obs, b))

    header = "<tr><th>Step</th><th>Action</th><th>Observation</th><th>Belief</th></tr>"
    body = "".join(
        f"<tr><td>{i}</td><td>{a or '—'}</td><td>{o or '—'}</td><td style='white-space:nowrap'>{b._repr_latex_()}</td></tr>"
        for i, (a, o, b) in enumerate(rows)
    )
    display(HTML(f"<table>{header}{body}</table>"))


def belief_trace(
    pomdp: "Model",
    b0: Belief,
    trace: list[tuple[str, str]],
) -> list[Belief]:
    """Compute the sequence of beliefs induced by an observation trace.

    :param pomdp: The POMDP model.
    :param b0: Initial belief distribution.
    :param trace: Sequence of ``(action_label, obs_alias)`` pairs.
    :returns: List of beliefs of length ``len(trace) + 1``: the initial belief
        followed by one updated belief per step.
    """
    beliefs: list[Belief] = [b0]
    current = b0
    for action_label, obs_alias in trace:
        current = belief_update(pomdp, current, action_label, obs_alias)
        beliefs.append(current)
    return beliefs
