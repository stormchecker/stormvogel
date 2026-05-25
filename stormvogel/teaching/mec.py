"""Teaching-facing MEC detection/elimination re-exports, plus exhaustive EC enumeration.

The MEC detection and elimination implementations live in
:mod:`stormvogel.stormpy_utils.mec` and require stormpy.
:func:`enumerate_ecs` is pure Python and has no stormpy dependency.
"""

from dataclasses import dataclass
import warnings

import stormvogel.model as model
from stormvogel.model.state import State
from stormvogel.model.action import Action
from stormvogel.stormpy_utils.mec import detect_mecs, eliminate_mecs  # NOQA


def _state_name(s: State) -> str:
    return s.friendly_name if s.friendly_name is not None else str(s.state_id)


@dataclass(frozen=True)
class EndComponent:
    """An end component (EC) of an MDP.

    Formally, a non-empty set of state-action pairs X ⊆ S × A such that:

    - *Closure*: for every (s, a) ∈ X and every s' with δ(s, a)(s') > 0,
      s' ∈ S' where S' = {s | (s, a) ∈ X}.
    - *Strong connectivity*: the graph on S' induced by X is strongly connected.

    :param choices: The state-action pairs comprising this EC, as a
        ``frozenset`` of ``(State, Action)`` tuples.
    """

    choices: frozenset  # frozenset[tuple[State, Action]]

    @property
    def states(self) -> frozenset:
        """The set of states S' = {s | (s, a) ∈ choices}."""
        return frozenset(s for s, _ in self.choices)

    @property
    def actions(self) -> frozenset:
        """The set of distinct actions used across all choices."""
        return frozenset(a for _, a in self.choices)

    def satisfies_buchi(self, target_states) -> bool:
        """Return ``True`` if this EC contains at least one Büchi target state.

        :param target_states: An iterable of :class:`~stormvogel.model.state.State`
            objects that are Büchi states.
        """
        return bool(self.states & set(target_states))

    def __repr__(self) -> str:
        state_names = sorted({_state_name(s) for s, _ in self.choices})
        choice_strs = sorted(f"({_state_name(s)}, {a.label})" for s, a in self.choices)
        return (
            f"EndComponent("
            f"states={{{', '.join(state_names)}}}, "
            f"choices={{{', '.join(choice_strs)}}})"
        )


def _reachable(start: State, adj: dict, within: frozenset) -> frozenset:
    visited: set = {start}
    stack = [start]
    while stack:
        s = stack.pop()
        for t in adj.get(s, ()):
            if t in within and t not in visited:
                visited.add(t)
                stack.append(t)
    return frozenset(visited)


def _is_strongly_connected(states: frozenset, adj: dict) -> bool:
    if not states:
        return False
    start = next(iter(states))
    if _reachable(start, adj, states) != states:
        return False
    rev: dict = {s: set() for s in states}
    for s in states:
        for t in adj.get(s, ()):
            if t in states:
                rev[t].add(s)
    return _reachable(start, rev, states) == states


_EC_CHOICE_THRESHOLD = 15


def enumerate_ecs(mdp: model.Model) -> list[EndComponent]:
    """Return all end components of *mdp*.

    Uses an exhaustive bitmask search over all state-action pairs.
    For each candidate subset X the function checks closure and strong
    connectivity.  The number of subsets is 2^n where n is the total number
    of state-action pairs (choices) in the MDP — this is only tractable for
    small teaching examples.

    :param mdp: A stormvogel MDP.
    :returns: All end components as :class:`EndComponent` objects.
    """
    all_choices: list[tuple[State, Action]] = []
    all_succ: list[set] = []
    for s in mdp.states:
        if s not in mdp.transitions:
            continue
        for a, branch in mdp.transitions[s]:
            all_choices.append((s, a))
            all_succ.append(branch.support)

    n = len(all_choices)
    if n > _EC_CHOICE_THRESHOLD:
        warnings.warn(
            f"enumerate_ecs: the MDP has {n} choices (threshold {_EC_CHOICE_THRESHOLD}). "
            "Enumerating up to 2^n subsets — may be slow.",
            stacklevel=2,
        )

    results: list[EndComponent] = []
    for mask in range(1, 1 << n):
        indices = [i for i in range(n) if mask & (1 << i)]
        S_prime = frozenset(all_choices[i][0] for i in indices)

        # Closure: every successor of every chosen (s,a) must lie in S'.
        if any(t not in S_prime for i in indices for t in all_succ[i]):
            continue

        # Forward adjacency on S' (closure guarantees all targets are in S').
        adj: dict = {s: set() for s in S_prime}
        for i in indices:
            adj[all_choices[i][0]].update(all_succ[i])

        if _is_strongly_connected(S_prime, adj):
            results.append(
                EndComponent(choices=frozenset(all_choices[i] for i in indices))
            )

    return results
