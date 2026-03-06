"""Test helpers for semantic equality of model objects.

Production code uses identity-based equality for all model types.
These helpers perform deep structural comparison for tests only.
"""

from __future__ import annotations

from stormvogel.model.model import Model
from stormvogel.model.choices import Choices
from stormvogel.model.observation import Observation
from stormvogel.simulator import Path


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def _norm_choices(choices: Choices, idx: dict[int, int]) -> dict:
    """Normalize a Choices into {action: [(value, target_index), ...]} sorted by target index."""
    return {
        a: [(v, idx[id(s)]) for v, s in sorted(b.branches, key=lambda t: idx[id(t[1])])]
        for a, b in choices
    }


def _obs_eq(o1, o2) -> bool:
    """Compare observations by alias + valuations, ignoring UUID."""
    if o1 is None and o2 is None:
        return True
    if o1 is None or o2 is None:
        return False
    if type(o1) is not type(o2):
        return False
    if isinstance(o1, Observation):
        return o1.alias == o2.alias and o1.valuations == o2.valuations
    return True  # Distribution or unknown — skip deep comparison


def models_equal(a: Model, b: Model) -> tuple[bool, str]:
    """Deep structural comparison of two models.

    Returns (True, "") on success, or (False, reason) on mismatch.
    """
    if a.model_type != b.model_type:
        return False, f"model_type: {a.model_type} != {b.model_type}"
    if len(a.states) != len(b.states):
        return False, f"state count: {len(a.states)} != {len(b.states)}"

    a_idx = {id(s): i for i, s in enumerate(a.states)}
    b_idx = {id(s): i for i, s in enumerate(b.states)}

    for i, (s1, s2) in enumerate(zip(a.states, b.states)):
        if set(s1.labels) != set(s2.labels):
            return False, f"state[{i}] labels: {set(s1.labels)} != {set(s2.labels)}"
        if a.state_valuations.get(s1) != b.state_valuations.get(s2):
            return False, (
                f"state[{i}] valuations: "
                f"{a.state_valuations.get(s1)} != {b.state_valuations.get(s2)}"
            )
        c1 = s1 in a.choices
        c2 = s2 in b.choices
        if c1 != c2:
            return False, f"state[{i}] has_choices mismatch: {c1} vs {c2}"
        if c1:
            nc1 = _norm_choices(a.choices[s1], a_idx)
            nc2 = _norm_choices(b.choices[s2], b_idx)
            if nc1 != nc2:
                return False, f"state[{i}] choices differ:\n  {nc1}\n  {nc2}"

    # sort rewards by name
    a.rewards = sorted(a.rewards, key=lambda rm: rm.name)
    b.rewards = sorted(b.rewards, key=lambda rm: rm.name)

    if len(a.rewards) != len(b.rewards):
        return False, f"reward model count: {len(a.rewards)} != {len(b.rewards)}"
    for ri, (rm1, rm2) in enumerate(zip(a.rewards, b.rewards)):
        if rm1.name != rm2.name:
            return False, f"reward[{ri}] name: {rm1.name!r} != {rm2.name!r}"
        for i, (s1, s2) in enumerate(zip(a.states, b.states)):
            r1 = rm1.get_state_reward(s1)
            r2 = rm2.get_state_reward(s2)
            if r1 != r2:
                return False, (
                    f"reward[{ri}] state[{i}] ({list(s1.labels)}): {r1} != {r2}"
                )

    if (a.state_observations is None) != (b.state_observations is None):
        return False, "state_observations: one is None, the other is not"
    if a.state_observations is not None and b.state_observations is not None:
        # Compare observation partitions: which groups of states share an observation
        def _obs_partition(model):
            groups = {}
            for i, s in enumerate(model.states):
                obs = model.state_observations.get(s)
                if obs is not None:
                    key = id(obs)
                    groups.setdefault(key, set()).add(i)
            return set(frozenset(v) for v in groups.values())

        ap = _obs_partition(a)
        bp = _obs_partition(b)
        if ap != bp:
            return False, f"observation partitions differ:\n  {ap}\n  {bp}"

    if (a.markovian_states is None) != (b.markovian_states is None):
        return False, "markovian_states: one is None, the other is not"
    if a.markovian_states is not None and b.markovian_states is not None:
        am = {a_idx[id(s)] for s in a.markovian_states}
        bm = {b_idx[id(s)] for s in b.markovian_states}
        if am != bm:
            return False, f"markovian_states indices: {am} != {bm}"

    return True, ""


def assert_models_equal(a: Model, b: Model) -> None:
    """Assert two models are structurally identical. Gives a clear diff on failure."""
    ok, reason = models_equal(a, b)
    assert ok, f"Models differ: {reason}"


def assert_choices_equal(a: Choices, b: Choices) -> None:
    """Assert two Choices are structurally identical.

    Both must reference states from the same model (identity comparison).
    """
    assert len(a) == len(b), f"Choices length: {len(a)} != {len(b)}"
    a_map = {act.label: list(br) for act, br in a}
    b_map = {act.label: list(br) for act, br in b}
    assert (
        a_map.keys() == b_map.keys()
    ), f"Action labels differ: {set(a_map.keys())} != {set(b_map.keys())}"
    for label in a_map:
        a_dist = a_map[label]
        b_dist = b_map[label]
        assert len(a_dist) == len(
            b_dist
        ), f"Action {label!r} branch count: {len(a_dist)} != {len(b_dist)}"
        for j, ((av, ast), (bv, bst)) in enumerate(zip(a_dist, b_dist)):
            assert av == bv, f"Action {label!r} entry {j} value: {av} != {bv}"
            assert (
                ast is bst
            ), f"Action {label!r} entry {j} state: {list(ast.labels)} is not {list(bst.labels)}"


# ---------------------------------------------------------------------------
# Path comparison
# ---------------------------------------------------------------------------


def assert_paths_equal(a: Path, b: Path) -> None:
    """Assert two paths are structurally identical.

    Paths must reference the same underlying model object.
    """
    assert a.model is b.model, "Paths reference different model objects"
    assert len(a.path) == len(
        b.path
    ), f"Path lengths differ: {len(a.path)} != {len(b.path)}"
    # States and actions within the same model can be compared by identity / label
    for i, (ea, eb) in enumerate(zip(a.path, b.path)):
        if isinstance(ea, tuple) and isinstance(eb, tuple):
            # (Action, State) pair
            assert (
                ea[0].label == eb[0].label
            ), f"Step {i} action: {ea[0].label!r} != {eb[0].label!r}"
            assert (
                ea[1] is eb[1]
            ), f"Step {i} state: {list(ea[1].labels)} is not {list(eb[1].labels)}"
        else:
            assert ea is eb, f"Step {i}: {ea} is not {eb}"
